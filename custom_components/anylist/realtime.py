"""AnyList realtime push: refresh as soon as AnyList says something changed.

AnyList's apps keep a WebSocket open to ``/data/add-user-listener``. The server
sends short text invalidations ("refresh-shopping-lists", ...) whenever the
account changes anywhere (the phone app, the web app, another integration),
and both sides exchange a ``--heartbeat--`` every few seconds. This listener
does the same and asks the coordinator for a refresh on every invalidation
that matters, so a change made in the AnyList app shows up in Home Assistant
within a second or two instead of at the next poll. Polling stays on as the
fallback whenever the socket is down.

Protocol details follow AnyList's web client (and the aioanylist library's
RealtimeClient): close code 4010 means the access token expired, so refresh it
and reconnect at once; any other drop reconnects with exponential backoff.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging
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
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 300.0
STABLE_AFTER = 30.0  # a connection that lasted this long resets the backoff
TOKEN_EXPIRED = 4010

# Invalidations that change what this integration shows: lists and their
# items, categories, and the meal plan calendar.
REFRESH_MESSAGES = frozenset(
    {
        "refresh-shopping-lists",
        "refresh-categorized-items",
        "refresh-list-folders",
        "refresh-list-settings",
        "refresh-user-categories",
        "refresh-user-recipe-data",
        "refresh-meal-plan-calendar",
    }
)


class AnyListRealtime:
    """Keeps the AnyList listener socket open and triggers refreshes."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: AnyListClient,
        session: aiohttp.ClientSession,
        on_change: Callable[[], Awaitable[None]],
    ) -> None:
        self._hass = hass
        self._client = client
        self._session = session
        self._on_change = on_change
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._retry_delay = INITIAL_RETRY_DELAY
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
        """Close the socket and stop reconnecting."""
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.connected = False

    def diagnostics(self) -> dict:
        """Connection state for diagnostics."""
        return {"connected": self.connected, "connects": self.connects, "messages": self.messages}

    def _url(self) -> str:
        token = self._client.realtime_credentials()
        return (
            f"{WS_URL}?client_id={quote(token[0], safe='')}"
            f"&access_token={quote(token[1], safe='')}"
        )

    async def _run(self) -> None:
        while not self._stop.is_set():
            code = 1006
            started = self._hass.loop.time()
            try:
                code = await self._connection()
            except (aiohttp.ClientError, TimeoutError, OSError) as err:
                _LOGGER.debug("AnyList realtime connection failed: %s", err)
            except Exception:  # noqa: BLE001 - never let the listener die silently
                _LOGGER.exception("AnyList realtime listener error")
            self.connected = False
            if self._stop.is_set():
                return
            if self._hass.loop.time() - started >= STABLE_AFTER:
                self._retry_delay = INITIAL_RETRY_DELAY
            if code == TOKEN_EXPIRED:
                try:
                    await self._hass.async_add_executor_job(self._client.refresh_access_token)
                    _LOGGER.debug("AnyList realtime: token refreshed, reconnecting")
                    continue
                except Exception as err:  # noqa: BLE001 - fall back to the normal backoff
                    _LOGGER.debug("AnyList realtime token refresh failed: %s", err)
            _LOGGER.debug(
                "AnyList realtime disconnected (code %s); reconnecting in %.0fs (polling meanwhile)",
                code,
                self._retry_delay,
            )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._retry_delay)
                return
            except TimeoutError:
                pass
            self._retry_delay = min(self._retry_delay * 2, MAX_RETRY_DELAY)

    async def _connection(self) -> int:
        async with self._session.ws_connect(self._url(), heartbeat=None) as ws:
            self.connected = True
            self.connects += 1
            if self.connects == 1:
                _LOGGER.info("AnyList realtime connected: changes from the AnyList apps now refresh at once")
            else:
                # Anything that changed while we were away.
                self._schedule_refresh()
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
                    elif msg.data in REFRESH_MESSAGES:
                        self.messages += 1
                        _LOGGER.debug("AnyList realtime: %s", msg.data)
                        self._schedule_refresh()
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

    def _schedule_refresh(self) -> None:
        # Off the socket's receive loop, so a slow refresh can't miss heartbeats.
        self._hass.async_create_task(self._on_change(), eager_start=False)
