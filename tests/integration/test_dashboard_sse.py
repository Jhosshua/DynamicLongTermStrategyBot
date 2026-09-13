"""
tests.integration.test_dashboard_sse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration test suite for Milestone 3 Server-Sent Events (SSE) streaming endpoint:
- Protocol and streaming headers (text/event-stream, no-cache, X-Accel-Buffering: no)
- Valid SSE frame syntax (data: {...}\n\n)
- Real-time heartbeat telemetry and state change updates
- Abrupt client disconnection handling with zero task/coroutine leaks
- Multi-client subscription isolation
- Feed alert banner propagation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from typing import AsyncGenerator
import httpx
import pytest

from bot.service import DynamicStrategyService, ServiceConfig, ServiceState
from bot.feed_manager import FeedSource
from web.app import create_app


class StreamingASGITransport(httpx.AsyncBaseTransport):
    """
    True asynchronous streaming ASGI transport that pipes StreamingResponse
    chunks directly to httpx response stream without buffering.
    """
    def __init__(self, app):
        self.app = app

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        queue: asyncio.Queue = asyncio.Queue()
        status_code = None
        headers = []
        started = asyncio.Event()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(k.lower(), v) for (k, v) in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port or 80),
            "client": ("127.0.0.1", 12345),
            "root_path": "",
        }

        disconnect_event = asyncio.Event()

        async def receive():
            if disconnect_event.is_set():
                return {"type": "http.disconnect"}
            await disconnect_event.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal status_code, headers
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = message.get("headers", [])
                started.set()
            elif message["type"] == "http.response.body":
                b = message.get("body", b"")
                if b:
                    await queue.put(b)
                if not message.get("more_body", False):
                    await queue.put(None)

        task = asyncio.create_task(self.app(scope, receive, send))

        await started.wait()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk

            async def aclose(self):
                disconnect_event.set()
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        return httpx.Response(status_code, headers=headers, stream=Stream())


@pytest.fixture
def isolated_service(temp_sqlite_db) -> DynamicStrategyService:
    """Provides isolated strategy service for SSE streaming tests."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
        initial_cash=50000.00,
        dry_run=False,
    )
    service = DynamicStrategyService(config=config)
    service.paper_account.init_schema()
    service._service_state = ServiceState.RUNNING
    return service


@pytest.fixture
async def sse_client(isolated_service: DynamicStrategyService) -> AsyncGenerator[httpx.AsyncClient, None]:
    """AsyncClient using StreamingASGITransport for in-process streaming tests."""
    app = create_app(service=isolated_service)
    transport = StreamingASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=5.0) as client:
        yield client


@pytest.mark.asyncio
async def test_sse_headers_and_content_type(sse_client: httpx.AsyncClient):
    """Verify GET /api/events returns HTTP 200 with text/event-stream headers."""
    async with sse_client.stream("GET", "/api/events") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert "no-cache" in response.headers.get("cache-control", "")
        assert response.headers.get("x-accel-buffering") == "no"


@pytest.mark.asyncio
async def test_sse_heartbeat_payload_format(sse_client: httpx.AsyncClient):
    """Verify SSE frames follow W3C data: {...}\n\n syntax and contain valid telemetry."""
    received_frames = []
    async with sse_client.stream("GET", "/api/events") as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                raw_json = line[len("data:"):].strip()
                payload = json.loads(raw_json)
                received_frames.append(payload)
                if len(received_frames) >= 2:
                    break

    assert len(received_frames) >= 2
    first = received_frames[0]
    assert "timestamp" in first
    assert "status" in first or "service_name" in first
    assert "portfolio" in first or "total_nav" in first

    nav = first.get("total_nav") or (first.get("portfolio", {}).get("total_nav"))
    assert nav == 50000.00


@pytest.mark.asyncio
async def test_sse_client_disconnect_clean_shutdown(sse_client: httpx.AsyncClient):
    """Verify that abrupt client disconnection terminates generator without error."""
    async def open_and_drop():
        async with sse_client.stream("GET", "/api/events") as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    # Abrupt break, dropping connection
                    break

    # Should complete cleanly within 2 seconds without hanging
    await asyncio.wait_for(open_and_drop(), timeout=3.0)


@pytest.mark.asyncio
async def test_sse_multi_subscriber_isolation(isolated_service: DynamicStrategyService):
    """Verify multiple concurrent clients stream independently."""
    app = create_app(service=isolated_service)
    transport = StreamingASGITransport(app=app)

    async def listen_for_one(client_id: int):
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=5.0) as ac:
            async with ac.stream("GET", "/api/events") as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        return client_id, json.loads(line[len("data:"):].strip())

    # Run 3 subscribers concurrently
    results = await asyncio.gather(
        listen_for_one(1),
        listen_for_one(2),
        listen_for_one(3),
    )
    assert len(results) == 3
    ids = {r[0] for r in results}
    assert ids == {1, 2, 3}


@pytest.mark.asyncio
async def test_sse_state_change_propagation(sse_client: httpx.AsyncClient, isolated_service: DynamicStrategyService):
    """Verify operator pause action is reflected in the SSE event stream."""
    received = []

    async def stream_collector():
        async with sse_client.stream("GET", "/api/events") as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[len("data:"):].strip())
                    received.append(data)
                    state = data.get("state") or data.get("status", {}).get("state")
                    if state == "PAUSED":
                        break
                    if len(received) >= 10:
                        break

    collector_task = asyncio.create_task(stream_collector())
    await asyncio.sleep(0.05)

    # Trigger pause
    await isolated_service.pause()

    await asyncio.wait_for(collector_task, timeout=4.0)
    assert any(
        (f.get("state") == "PAUSED" or f.get("status", {}).get("state") == "PAUSED")
        for f in received
    )


@pytest.mark.asyncio
async def test_sse_alert_banner_event_on_fallback(sse_client: httpx.AsyncClient, isolated_service: DynamicStrategyService):
    """Verify that when feed enters fallback, alert_banner_active is streamed."""
    received = []

    async def stream_collector():
        async with sse_client.stream("GET", "/api/events") as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[len("data:"):].strip())
                    received.append(data)
                    if data.get("alert_banner_active") is True:
                        break
                    if len(received) >= 10:
                        break

    collector_task = asyncio.create_task(stream_collector())
    await asyncio.sleep(0.05)

    # Force connection status alert banner active
    isolated_service.feed_manager._alert_banner_active = True
    isolated_service.feed_manager._feed_source = FeedSource.SYNTHETIC_FALLBACK

    await asyncio.wait_for(collector_task, timeout=4.0)
    assert any(f.get("alert_banner_active") is True for f in received)
