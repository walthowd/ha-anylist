"""AnyList realtime push: refresh as soon as AnyList says something changed.

AnyList's apps keep a WebSocket open to ``/data/add-user-listener``. The server
sends short text invalidations ("refresh-shopping-lists", ...) whenever the
account changes anywhere (the phone app, the web app, another integration),
and both sides exchange a ``--heartbeat--`` every few seconds. This listener
does the same: each invalidation marks the matching cached resources stale and,
when the coordinator shows that data, asks for a refresh. A change made in the
AnyList app shows up in Home Assistant within a second or two instead of at
the next poll. Polling stays on as the fallback whenever the socket is down.

Protocol details follow AnyList's web client (and the aioanylist library's
RealtimeClient): close code 4010 means the access token expired, so refresh it
and reconnect at once; any other drop reconnects with exponential backoff.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
import logging
import random
import re
from typing import TYPE_CHECKING
from urllib.parse import quote

import aiohttp

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .client import AnyListClient

_LOGGER = logging.getLogger(__name__)

WS_URL = "wss://www.anylist.com/data/add-user-listener"
HEARTBEAT = "--heartbeat--"
HEARTBEAT_INTERVAL = 5.0
MAX_MISSED_HEARTBEATS = 3
CONNECT_TIMEOUT = 30.0
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 300.0
STABLE_AFTER = 10.0  # a connection that lasted this long resets the backoff
COALESCE = 1.0  # gather a burst of invalidations into one refresh
TOKEN_EXPIRED = 4010

# What each invalidation makes stale: the client's cached resources to mark,
# and whether the coordinator (lists, items, categories, favourites) must
# refresh. Recipes are read on demand by services, so marking them is enough;
# the meal plan is only exposed as an iCal URL, so it needs nothing.
MESSAGES: dict[str, tuple[frozenset[str], bool]] = {
    "refresh-shopping-lists": (frozenset({"lists"}), True),
    "refresh-categorized-items": (frozenset({"lists"}), True),
    "refresh-list-folders": (frozenset({"lists"}), True),
    "refresh-user-categories": (frozenset({"lists"}), True),
    "refresh-list-settings": (frozenset({"settings"}), True),
    "refresh-starter-lists": (frozenset({"favourites"}), True),
    "refresh-ordered-starter-list-ids": (frozenset({"favourites"}), True),
    "refresh-starter-list-settings": (frozenset({"favourites"}), True),
    "refresh-user-recipe-data": (frozenset({"recipes"}), False),
}

_TOKEN_RE = re.compile(r"(access_token=)[^&'\"\s]+")


def safe_error(err: BaseException) -> str:
    """Describe an error without the access token aiohttp puts in its URL."""
    status = getattr(err, "status", None)
    text = _TOKEN_RE.sub(r"\1***", str(err))
    return f"{type(err).__name__}{f' {status}' if status else ''}: {text}"


class AnyListRealtime:
    """Keeps the AnyList listener socket open and triggers refreshes."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: AnyListClient,
        session: aiohttp.ClientSession,
        on_change: Callable[[frozenset[str], bool], Awaitable[None]],
    ) -> None:
        self._hass = hass
        self._client = client
        self._session = session
        self._on_change = on_change
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._retry_delay = INITIAL_RETRY_DELAY
        self._pending: set[str] = set()
        self._pending_coordinator = False
        self._flush: asyncio.Task | None = None
        self.connected = False
        self.messages = 0
        self.connects = 0

    def start(self) -> None:
        """Start the listener in the background."""
        if self._task is None:
            self._stop.clear()
            self._task = self._hass.async_create_background_task(
                self._run(), name="anylist-realtime"
            )

    async def stop(self) -> None:
        """Close the socket, stop reconnecting and drop a pending refresh."""
        self._stop.set()
        tasks = [t for t in (self._task, self._flush) if t is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._task = self._flush = None
        self._pending.clear()
        self._pending_coordinator = False
        self.connected = False

    def diagnostics(self) -> dict:
        """Connection state for diagnostics."""
        return {"connected": self.connected, "connects": self.connects, "messages": self.messages}

    async def _run(self) -> None:
        last_expired = False
        while not self._stop.is_set():
            code = 1006
            started = self._hass.loop.time()
            client_id, token = self._client.realtime_credentials()
            try:
                code = await self._connection(client_id, token)
            except aiohttp.WSServerHandshakeError as err:
                # An expired token is rejected at the handshake with 401/403.
                code = TOKEN_EXPIRED if err.status in (401, 403) else 1006
                _LOGGER.debug("AnyList realtime handshake failed: %s", safe_error(err))
            except (aiohttp.ClientError, TimeoutError, OSError) as err:
                _LOGGER.debug("AnyList realtime connection failed: %s", safe_error(err))
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - never let the listener die
                _LOGGER.warning("AnyList realtime listener error: %s", safe_error(err))
            self.connected = False
            if self._stop.is_set():
                return
            stable = self._hass.loop.time() - started >= STABLE_AFTER
            if stable:
                self._retry_delay = INITIAL_RETRY_DELAY
            if code == TOKEN_EXPIRED:
                # Once straight away; a second quick expiry backs off, so a
                # server that keeps refusing can't drive a refresh loop.
                expired_again = last_expired and not stable
                last_expired = True
                if not expired_again and await self._refresh_token(token):
                    continue
            else:
                last_expired = False
            delay = self._retry_delay * random.uniform(0.8, 1.2)
            _LOGGER.debug(
                "AnyList realtime disconnected (code %s); reconnecting in %.0fs (polling meanwhile)",
                code,
                delay,
            )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                return
            except TimeoutError:
                pass
            self._retry_delay = min(self._retry_delay * 2, MAX_RETRY_DELAY)

    async def _refresh_token(self, used: str) -> bool:
        """Refresh the token the socket was refused with, unless already done."""
        if self._client.realtime_credentials()[1] != used:
            return True  # the client refreshed it meanwhile (its own 401 path)
        try:
            await self._hass.async_add_executor_job(self._client.refresh_access_token)
        except Exception as err:  # noqa: BLE001 - e.g. a 429 cooldown: back off
            _LOGGER.debug("AnyList realtime token refresh failed: %s", safe_error(err))
            return False
        _LOGGER.debug("AnyList realtime: token refreshed, reconnecting")
        return True

    async def _connection(self, client_id: str, token: str) -> int:
        url = (
            f"{WS_URL}?client_id={quote(client_id, safe='')}"
            f"&access_token={quote(token, safe='')}"
        )
        async with asyncio.timeout(CONNECT_TIMEOUT):
            ws = await self._session.ws_connect(url, heartbeat=None)
        async with ws:
            self.connected = True
            self.connects += 1
            if self.connects == 1:
                _LOGGER.info("AnyList realtime connected: changes from the AnyList apps now refresh at once")
            else:
                # Anything that changed while we were away.
                self._schedule({"lists", "favourites"}, True)
            loop = self._hass.loop
            missed = 0
            next_heartbeat = loop.time() + HEARTBEAT_INTERVAL
            while not self._stop.is_set():
                try:
                    msg = await asyncio.wait_for(
                        ws.receive(), timeout=max(0.0, next_heartbeat - loop.time())
                    )
                except TimeoutError:
                    missed += 1
                    if missed >= MAX_MISSED_HEARTBEATS:
                        await ws.close()
                        return 1006
                    await ws.send_str(HEARTBEAT)
                    next_heartbeat = loop.time() + HEARTBEAT_INTERVAL
                    continue
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == HEARTBEAT:
                        missed = 0
                    elif (what := MESSAGES.get(msg.data)) is not None:
                        self.messages += 1
                        _LOGGER.debug("AnyList realtime: %s", msg.data)
                        self._schedule(*what)
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    return int(ws.close_code or 1000)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    return int(ws.close_code or 1006)
                if loop.time() >= next_heartbeat:
                    missed += 1
                    if missed >= MAX_MISSED_HEARTBEATS:
                        await ws.close()
                        return 1006
                    await ws.send_str(HEARTBEAT)
                    next_heartbeat = loop.time() + HEARTBEAT_INTERVAL
            await ws.close()
            return 1000

    def _schedule(self, resources: Iterable[str], coordinator: bool) -> None:
        """Collect a burst of invalidations; one refresh follows shortly."""
        self._pending.update(resources)
        self._pending_coordinator = self._pending_coordinator or coordinator
        if self._flush is None or self._flush.done():
            self._flush = self._hass.async_create_task(self._flush_later(), eager_start=False)

    async def _flush_later(self) -> None:
        await asyncio.sleep(COALESCE)
        resources, coordinator = frozenset(self._pending), self._pending_coordinator
        self._pending.clear()
        self._pending_coordinator = False
        if not resources and not coordinator:
            return
        try:
            await self._on_change(resources, coordinator)
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - a failed refresh must not kill the socket
            _LOGGER.debug("AnyList realtime refresh failed: %s", safe_error(err))
