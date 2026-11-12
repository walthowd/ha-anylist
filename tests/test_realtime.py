"""Tests for the AnyList realtime push listener."""

from __future__ import annotations

import asyncio

from types import SimpleNamespace

from aiohttp import WSMsgType
import pytest

from custom_components.anylist import realtime as rt


class FakeClient:
    """Just the client surface the listener uses."""

    def __init__(self) -> None:
        self.token = "tok1"
        self.refreshes = 0

    def realtime_credentials(self):
        return "client-1", self.token

    def refresh_access_token(self):
        self.refreshes += 1
        self.token = f"tok{self.refreshes + 1}"


class FakeWS:
    """A scripted AnyList listener socket."""

    def __init__(self, script) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.sent: list[str] = []
        self.close_code = None
        self.script = script
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
        for item in self.script.get("on_heartbeat", []):
            self.queue.put_nowait(item)
        self.script["on_heartbeat"] = []

    async def close(self, code: int = 1000) -> None:
        self.close_code = code
        self.queue.put_nowait(SimpleNamespace(type=WSMsgType.CLOSED, data=None))


def text(data: str):
    return SimpleNamespace(type=WSMsgType.TEXT, data=data)


async def test_push_refresh_heartbeat_and_token_expiry(hass, monkeypatch):
    """A push refreshes; heartbeats go out; 4010 refreshes the token and reconnects at once."""
    monkeypatch.setattr(rt, "HEARTBEAT_INTERVAL", 0.05)
    urls: list[str] = []
    sockets: list[FakeWS] = []
    expired = SimpleNamespace(type=WSMsgType.CLOSE, data=None)

    class Session:
        def ws_connect(self, url, **_kwargs):
            urls.append(url)
            if len(urls) == 1:
                ws = FakeWS({
                    "on_open": [text("refresh-shopping-lists"), text("refresh-account-info")],
                    # After our first heartbeat: the server answers, then expires the token.
                    "on_heartbeat": [text(rt.HEARTBEAT), expired],
                })
                ws.close_code = rt.TOKEN_EXPIRED
            else:
                ws = FakeWS({})
            sockets.append(ws)
            return ws

    calls = 0
    refreshed = asyncio.Event()

    async def on_change() -> None:
        nonlocal calls
        calls += 1
        refreshed.set()

    client = FakeClient()
    listener_obj = rt.AnyListRealtime(hass, client, Session(), on_change)
    listener_obj.start()
    await asyncio.wait_for(refreshed.wait(), 5)
    for _ in range(100):
        if len(urls) >= 2:
            break
        await asyncio.sleep(0.02)
    await hass.async_block_till_done()
    assert "client_id=client-1" in urls[0] and "access_token=tok1" in urls[0]
    assert client.refreshes == 1
    assert "access_token=tok2" in urls[1]  # reconnected at once with the new token
    assert rt.HEARTBEAT in sockets[0].sent
    assert listener_obj.diagnostics()["messages"] == 1  # refresh-account-info ignored
    assert calls >= 2  # the push, and the catch-up after reconnecting
    await listener_obj.stop()
    assert not listener_obj.connected


async def test_backoff_when_unreachable(hass, monkeypatch):
    """An unreachable server is retried with backoff and stop() ends it cleanly."""
    attempts = 0

    class Session:
        def ws_connect(self, *_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise OSError("unreachable")

    monkeypatch.setattr(rt, "INITIAL_RETRY_DELAY", 0.01)

    async def on_change() -> None:
        raise AssertionError("no refresh without a connection")

    listener_obj = rt.AnyListRealtime(hass, FakeClient(), Session(), on_change)
    listener_obj.start()
    await asyncio.sleep(0.2)
    await listener_obj.stop()
    assert 2 <= attempts < 20  # retried, but backing off
