"""Tests for the AnyList realtime push listener.

Every wait is bounded (asyncio.wait_for / short sleeps), so a regression fails
instead of hanging the suite.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from aiohttp import WSMsgType, WSServerHandshakeError
import pytest

from custom_components.anylist import realtime as rt


class FakeClient:
    """Just the client surface the listener uses."""

    def __init__(self, *, refresh_fails: bool = False) -> None:
        self.token = "tok1"
        self.refreshes = 0
        self.refresh_fails = refresh_fails

    def realtime_credentials(self):
        return "client-1", self.token

    def refresh_access_token(self):
        self.refreshes += 1
        if self.refresh_fails:
            raise RuntimeError("cooldown")
        self.token = f"tok{self.refreshes + 1}"


class FakeWS:
    """A scripted AnyList listener socket."""

    def __init__(self, script: dict | None = None) -> None:
        script = script or {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.sent: list[str] = []
        self.close_code = script.get("close_code")
        self.on_heartbeat = list(script.get("on_heartbeat", []))
        for item in script.get("on_open", []):
            self.queue.put_nowait(item)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def receive(self):
        return await self.queue.get()

    async def send_str(self, data: str) -> None:
        self.sent.append(data)
        for item in self.on_heartbeat:
            self.queue.put_nowait(item)
        self.on_heartbeat = []

    async def close(self, code: int = 1000) -> None:
        self.close_code = self.close_code or code
        self.queue.put_nowait(SimpleNamespace(type=WSMsgType.CLOSED, data=None))


def text(data: str):
    return SimpleNamespace(type=WSMsgType.TEXT, data=data)


CLOSE = SimpleNamespace(type=WSMsgType.CLOSE, data=None)


class Session:
    """Hands out sockets from a factory; records the URLs."""

    def __init__(self, factory) -> None:
        self.factory = factory
        self.urls: list[str] = []

    async def ws_connect(self, url, **_kwargs):
        self.urls.append(url)
        result = self.factory(len(self.urls))
        if isinstance(result, BaseException):
            raise result
        if asyncio.iscoroutine(result):
            return await result
        return result


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(rt, "HEARTBEAT_INTERVAL", 0.05)
    monkeypatch.setattr(rt, "COALESCE", 0.02)
    monkeypatch.setattr(rt, "INITIAL_RETRY_DELAY", 0.05)


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[frozenset, bool]] = []
        self.event = asyncio.Event()

    async def __call__(self, resources, refresh) -> None:
        self.calls.append((resources, refresh))
        self.event.set()


async def until(predicate, timeout: float = 3.0) -> None:
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout)


async def test_push_maps_messages_and_coalesces(hass):
    """Pushes mark only what they name; a burst becomes one refresh."""
    burst = [text("refresh-shopping-lists"), text("refresh-starter-lists"), text("refresh-shopping-lists"),
             text("refresh-meal-plan-calendar"), text("refresh-account-info")]
    session = Session(lambda n: FakeWS({"on_open": burst}))
    rec = Recorder()
    listener = rt.AnyListRealtime(hass, FakeClient(), session, rec)
    listener.start()
    await asyncio.wait_for(rec.event.wait(), 3)
    await asyncio.sleep(0.1)
    assert rec.calls == [(frozenset({"lists", "favourites"}), True)]
    assert listener.diagnostics()["messages"] == 3  # meal plan / account ignored
    # Recipes: marked, no coordinator refresh.
    rec.calls.clear()
    listener._schedule(*rt.MESSAGES["refresh-user-recipe-data"])
    await until(lambda: rec.calls)
    assert rec.calls == [(frozenset({"recipes"}), False)]
    await listener.stop()


async def test_heartbeat_and_dead_socket(hass):
    """Heartbeats go out; a socket that never answers is replaced."""
    sockets: list[FakeWS] = []

    def factory(n):
        sockets.append(FakeWS())
        return sockets[-1]

    session = Session(factory)
    listener = rt.AnyListRealtime(hass, FakeClient(), session, Recorder())
    listener.start()
    await until(lambda: len(session.urls) >= 2)
    assert rt.HEARTBEAT in sockets[0].sent
    assert sockets[0].close_code == 1000  # we closed it after missed heartbeats
    await listener.stop()


async def test_token_expiry_refreshes_once_then_backs_off(hass):
    """4010 refreshes the token and reconnects at once; repeated 4010s back off."""
    session = Session(lambda n: FakeWS({"on_open": [CLOSE], "close_code": rt.TOKEN_EXPIRED}))
    client = FakeClient()
    listener = rt.AnyListRealtime(hass, client, session, Recorder())
    listener.start()
    await until(lambda: len(session.urls) >= 3)
    assert "access_token=tok1" in session.urls[0] and "access_token=tok2" in session.urls[1]
    await asyncio.sleep(0.4)
    # Not a hot loop: with backoff from 50 ms, a handful of attempts at most.
    assert len(session.urls) < 10, len(session.urls)
    assert client.refreshes <= 3
    await listener.stop()


async def test_token_refresh_failure_backs_off(hass):
    """A failed refresh (e.g. a 429 cooldown) falls back to the backoff."""
    session = Session(lambda n: FakeWS({"on_open": [CLOSE], "close_code": rt.TOKEN_EXPIRED}))
    client = FakeClient(refresh_fails=True)
    listener = rt.AnyListRealtime(hass, client, session, Recorder())
    listener.start()
    await asyncio.sleep(0.4)
    assert 1 <= len(session.urls) < 8
    assert client.refreshes <= len(session.urls)
    await listener.stop()


async def test_handshake_401_counts_as_expired_and_never_logs_token(hass, caplog):
    """A 401 at the handshake refreshes the token; logs never contain it."""
    caplog.set_level(logging.DEBUG, logger="custom_components.anylist.realtime")

    def factory(n):
        if n == 1:
            return WSServerHandshakeError(
                SimpleNamespace(real_url="wss://x/?client_id=c&access_token=tok1"), (), status=401,
                message="Invalid response status",
            )
        return FakeWS()

    client = FakeClient()
    session = Session(factory)
    listener = rt.AnyListRealtime(hass, client, session, Recorder())
    listener.start()
    await until(lambda: len(session.urls) >= 2)
    await listener.stop()
    assert client.refreshes == 1
    assert "tok1" not in caplog.text and "tok2" not in caplog.text
    assert "handshake failed" in caplog.text


def test_safe_error_redacts_token():
    err = RuntimeError("401, message='x', url='wss://h/p?client_id=c&access_token=SECRET.abc-1'")
    assert "SECRET" not in rt.safe_error(err)
    assert "access_token=***" in rt.safe_error(err)


async def test_refresh_error_does_not_kill_socket(hass):
    """A failing refresh (cooldown, closed client) is logged quietly; pushes keep working."""
    ws = FakeWS({"on_open": [text("refresh-shopping-lists")]})
    calls = 0

    async def failing(resources, refresh):
        nonlocal calls
        calls += 1
        raise RuntimeError("AnyList is cooling down")

    listener = rt.AnyListRealtime(hass, FakeClient(), Session(lambda n: ws), failing)
    listener.start()
    await until(lambda: calls == 1)
    ws.queue.put_nowait(text("refresh-shopping-lists"))
    await until(lambda: calls == 2)
    assert listener.connected
    await listener.stop()


async def test_stop_during_connect_and_drops_pending_refresh(hass):
    """stop() returns promptly mid-handshake, and a scheduled refresh never runs."""
    hang = asyncio.Event()

    async def never():
        await hang.wait()

    rec = Recorder()
    listener = rt.AnyListRealtime(hass, FakeClient(), Session(lambda n: never()), rec)
    listener.start()
    await asyncio.sleep(0.05)
    listener._schedule({"lists"}, True)  # pending, not yet flushed
    await asyncio.wait_for(listener.stop(), 2)
    await asyncio.sleep(0.1)
    assert rec.calls == []
    assert not listener.connected


async def test_backoff_when_unreachable(hass):
    """An unreachable server is retried with growing delays."""
    session = Session(lambda n: OSError("unreachable"))
    listener = rt.AnyListRealtime(hass, FakeClient(), session, Recorder())
    listener.start()
    await asyncio.sleep(0.5)
    await asyncio.wait_for(listener.stop(), 2)
    assert 2 <= len(session.urls) < 8  # 50, 100, 200 ms ... not a tight loop
