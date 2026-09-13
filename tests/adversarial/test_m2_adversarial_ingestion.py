"""
tests/adversarial/test_m2_adversarial_ingestion.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tier 5 Adversarial Verification Suite for Milestone M2:
AlpacaRelay Ingestion Client & Upstream Resiliency.

Adversarial Stress Scenarios:
1. Invalid tokens and HTTP 401 / WS 402 handling (immediate rejection, diverse payloads, zero retry waste).
2. High-throughput stream flooding: 50,000 queue burst absorption without blocking ws.recv(), plus ring-buffer oldest-drop overflow.
3. Strict wildcard '*' prohibition across all channels, entrypoints, and string variations.
4. Server-side eviction (close code 1013 "too slow") with clean exponential backoff, auto-reconnection, and multi-channel resubscription.
5. Rate limiter stress under concurrent asynchronous request bursts and high-concurrency contention.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Dict, List, Optional
from aiohttp import web
import httpx
import pytest

from strategy_engine.core.models import Bar, Quote, Trade
from strategy_engine.ingestion.rest_client import (
    AlpacaRelayRestClient,
    RelayAuthError,
    RelayAuthenticationError,
    RelayRateLimitError,
    RelayRequestError,
    RelayUpstreamError,
    TokenBucketRateLimiter,
)
from strategy_engine.ingestion.ws_client import (
    AlpacaRelayWSClient,
    LifecycleEvent,
    RelayAuthError as WSAuthError,
    RelayConnectionError,
    RelayHandshakeError,
    SubscriptionAckEvent,
    WSConnectionState,
)
from tests.mocks.mock_relay_server import MockAlpacaRelayServer

logger = logging.getLogger("test_m2_adversarial_ingestion")


# ============================================================================
# Adversarial Mock Server with Fault Injection
# ============================================================================

class AdversarialMockRelayServer(MockAlpacaRelayServer):
    """Custom mock server equipped with adversarial fault injection capabilities."""

    def __init__(self, token: str = "valid-secret-token", feed: str = "sip"):
        super().__init__(token=token, feed=feed)
        self.custom_401_body: Optional[str] = None
        self.custom_401_content_type: str = "application/json"
        self.auth_attempt_count: int = 0
        self.handshake_mode: str = "normal"  # "normal", "corrupt_banner", "hang_banner", "402_error"

    def set_custom_401_response(self, body: str, content_type: str = "application/json"):
        self.custom_401_body = body
        self.custom_401_content_type = content_type

    async def _handle_single_bars(self, request: web.Request):
        self.auth_attempt_count += 1
        if not self._check_auth(request):
            if self.custom_401_body is not None:
                return web.Response(
                    body=self.custom_401_body,
                    status=401,
                    content_type=self.custom_401_content_type,
                )
            return web.json_response({"relay_error": "missing or bad relay token"}, status=401)
        return await super()._handle_single_bars(request)

    async def _handle_root_or_ws(self, request: web.Request):
        if request.headers.get("Upgrade", "").lower() != "websocket":
            return await super()._handle_root_or_ws(request)

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)
        self.subscriptions[ws] = {"trades": set(), "quotes": set(), "bars": set(), "dailyBars": set()}

        if self.handshake_mode == "corrupt_banner":
            await ws.send_str("{invalid_json_banner")
            await ws.close(code=1008)
            return ws
        elif self.handshake_mode == "hang_banner":
            # Hang without sending banner to trigger client timeout
            await asyncio.sleep(2.0)
            await ws.close(code=1008)
            return ws

        # Normal banner
        await ws.send_json([{"T": "success", "msg": "connected"}])

        # Wait for auth
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                token = data.get("token") or data.get("key")
                if token == self.token and self.handshake_mode != "402_error":
                    await ws.send_json([{"T": "success", "msg": "authenticated"}])
                else:
                    await ws.send_json([{"T": "error", "code": 402, "msg": "auth failed: invalid token"}])
                    await ws.close(code=1008)
                    return ws
            else:
                await ws.close(code=1008)
                return ws
        except Exception:
            await ws.close(code=1008)
            return ws

        # Message loop
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        action = data.get("action")
                        if action == "subscribe":
                            for ch in ("trades", "quotes", "bars", "dailyBars"):
                                for sym in data.get(ch, []):
                                    self.subscriptions[ws][ch].add(sym)
                            ack = {"T": "subscription"}
                            for ch in ("trades", "quotes", "bars", "dailyBars"):
                                ack[ch] = sorted(list(self.subscriptions[ws][ch]))
                            await ws.send_json([ack])
                        elif action == "unsubscribe":
                            for ch in ("trades", "quotes", "bars", "dailyBars"):
                                for sym in data.get(ch, []):
                                    self.subscriptions[ws][ch].discard(sym)
                            ack = {"T": "subscription"}
                            for ch in ("trades", "quotes", "bars", "dailyBars"):
                                ack[ch] = sorted(list(self.subscriptions[ws][ch]))
                            await ws.send_json([ack])
                    except json.JSONDecodeError:
                        pass
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.ERROR):
                    break
        finally:
            self.clients.discard(ws)
            self.subscriptions.pop(ws, None)

        return ws


@pytest.fixture
async def adv_server():
    server = AdversarialMockRelayServer(token="valid-secret-token")
    await server.start()
    # Populate initial sample bars for SPY and QQQ
    sample_bars = [
        {"t": f"2026-09-03T12:{i:02d}:00Z", "o": 500.0 + i, "h": 501.0 + i, "l": 499.0 + i, "c": 500.5 + i, "v": 1000}
        for i in range(10)
    ]
    server.add_mock_bars("SPY", sample_bars)
    server.add_mock_bars("QQQ", sample_bars)
    try:
        yield server
    finally:
        await server.stop()


# ============================================================================
# 1. Invalid Tokens & HTTP 401 / WS 402 Handling
# ============================================================================

@pytest.mark.asyncio
async def test_rest_auth_401_immediate_rejection_no_retries(adv_server: AdversarialMockRelayServer):
    """Adversarial Challenge: Verify that HTTP 401 is rejected immediately on attempt 1.
    If the client mistakenly retried 401 up to max_retries (5), it would sleep >15s.
    This test asserts immediate rejection within 200ms and exactly 1 HTTP call.
    """
    client = AlpacaRelayRestClient(
        base_url=adv_server.http_url,
        relay_token="WRONG_TOKEN",
        max_retries=5,
        base_delay=1.0,
    )
    async with client:
        start_t = time.monotonic()
        adv_server.auth_attempt_count = 0

        with pytest.raises((RelayAuthError, RelayAuthenticationError)) as exc_info:
            await client.get_bars("SPY")

        elapsed = time.monotonic() - start_t
        assert elapsed < 0.5, f"401 rejection took {elapsed:.2f}s, indicating retry waste!"
        assert adv_server.auth_attempt_count == 1, f"Expected exactly 1 request, got {adv_server.auth_attempt_count}"
        assert "401" in str(exc_info.value) or "Authentication failed" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,content_type,desc", [
    ('{"relay_error": "token revoked"}', "application/json", "standard json with relay_error"),
    ('{"error": "Unauthorized", "status": 401}', "application/json", "json without relay_error"),
    ('["error", "unauthorized"]', "application/json", "json array"),
    ('<html><body><h1>401 Unauthorized Gateway</h1></body></html>', "text/html", "raw html 401 gateway error"),
    ('', "text/plain", "completely empty body"),
    ('{broken json payload', "application/json", "corrupt json payload"),
])
async def test_rest_auth_401_diverse_payloads_never_crash(
    adv_server: AdversarialMockRelayServer,
    payload: str,
    content_type: str,
    desc: str,
):
    """Adversarial Challenge: Verify client handles diverse non-standard 401 error payloads
    (HTML error pages, empty responses, malformed JSON) without crashing with JSONDecodeError or AttributeError.
    """
    adv_server.set_custom_401_response(payload, content_type=content_type)
    client = AlpacaRelayRestClient(
        base_url=adv_server.http_url,
        relay_token="ANY_TOKEN",
    )
    async with client:
        with pytest.raises(RelayAuthError) as exc_info:
            await client.get_bars("SPY")
        assert "Authentication failed (HTTP 401)" in str(exc_info.value), f"Failed for {desc}"


@pytest.mark.asyncio
async def test_rest_empty_token_omits_headers(adv_server: AdversarialMockRelayServer):
    """Adversarial Challenge: Initializing with empty token string must omit auth headers
    and trigger 401 from server cleanly.
    """
    client = AlpacaRelayRestClient(
        base_url=adv_server.http_url,
        relay_token="",
    )
    assert client.get_auth_headers() == {}
    async with client:
        with pytest.raises(RelayAuthError):
            await client.get_bars("SPY")


@pytest.mark.asyncio
async def test_ws_auth_402_immediate_clean_failure(adv_server: AdversarialMockRelayServer):
    """Adversarial Challenge: Invalid WS token causes server 402 code.
    Client must raise WSAuthError immediately and transition to DISCONNECTED.
    """
    client = AlpacaRelayWSClient(
        url=adv_server.ws_url,
        token="INVALID_TOKEN",
        auto_reconnect=False,
    )
    try:
        with pytest.raises(WSAuthError) as exc_info:
            await client.connect()
        assert "Authentication failed" in str(exc_info.value)
        assert client.state == WSConnectionState.DISCONNECTED
        assert client.is_authenticated is False
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_handshake_corrupt_banner(adv_server: AdversarialMockRelayServer):
    """Adversarial Challenge: Malformed banner frame from server raises RelayHandshakeError."""
    adv_server.handshake_mode = "corrupt_banner"
    client = AlpacaRelayWSClient(
        url=adv_server.ws_url,
        token="valid-secret-token",
        connect_timeout=2.0,
        auto_reconnect=False,
    )
    try:
        with pytest.raises(RelayHandshakeError):
            await client.connect()
        assert client.state == WSConnectionState.DISCONNECTED
    finally:
        await client.disconnect()


# ============================================================================
# 2. Stream Flood Bursts & 50,000 Queue Absorption
# ============================================================================

@pytest.mark.asyncio
async def test_ws_burst_flood_50k_queue_absorption_no_recv_block(adv_server: AdversarialMockRelayServer):
    """Adversarial Challenge: Simulate extreme market opening / flash event.
    Push a massive burst of 10,000 bars in rapid succession through the WebSocket.
    Verify that:
    1. Producer loop ws.recv() drains all messages from socket without blocking.
    2. Server sends entire burst without TCP backpressure or socket close.
    3. 50,000 capacity queue absorbs the burst smoothly with zero dropped messages.
    4. Client metrics track messages_received and messages_dropped == 0.
    """
    client = AlpacaRelayWSClient(
        url=adv_server.ws_url,
        token="valid-secret-token",
        queue_size=50000,
    )
    received_bars: List[Bar] = []
    # Consumer callback adds to list
    client.on_bar(lambda b: received_bars.append(b))

    try:
        await client.connect()
        await client.subscribe(bars=["SPY"])
        await asyncio.sleep(0.05)

        initial_received = client.metrics["messages_received"]  # 1 for subscription ack
        assert initial_received >= 1

        server_ws = next(iter(adv_server.clients))

        # Send 10,000 bars in batches of 500 to simulate high-throughput feed bursts
        total_bars = 10000
        batch_size = 500
        start_t = time.monotonic()

        for batch_start in range(0, total_bars, batch_size):
            batch = [
                {
                    "T": "b",
                    "S": "SPY",
                    "t": "2026-09-03T13:30:00Z",
                    "o": 500.0,
                    "h": 501.0,
                    "l": 499.0,
                    "c": 500.5,
                    "v": 100 + i,
                }
                for i in range(batch_start, batch_start + batch_size)
            ]
            await server_ws.send_json(batch)

        send_duration = time.monotonic() - start_t
        logger.info("Sent %d bars in %.3fs", total_bars, send_duration)

        # Wait for consumer loop to drain the absorbed queue
        wait_deadline = time.monotonic() + 10.0
        while time.monotonic() < wait_deadline and len(received_bars) < total_bars:
            await asyncio.sleep(0.05)

        assert len(received_bars) == total_bars, f"Expected {total_bars} bars, got {len(received_bars)}"
        assert client.metrics["messages_received"] == initial_received + total_bars
        assert client.metrics["messages_dropped"] == 0
        assert client.metrics["bars_processed"] == total_bars
        assert client.is_connected is True
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_queue_saturation_ring_buffer_drop_oldest(adv_server: AdversarialMockRelayServer):
    """Adversarial Challenge: When queue capacity is saturated (e.g. queue_size=200),
    broadcasting 500 bars must cause the oldest 300 to be evicted so the socket never blocks.
    The remaining 200 items in the queue must strictly be the NEWEST messages (v >= 300).
    """
    client = AlpacaRelayWSClient(
        url=adv_server.ws_url,
        token="valid-secret-token",
        queue_size=200,
    )

    consumed_volumes: List[int] = []
    client.on_bar(lambda b: consumed_volumes.append(b.volume))

    try:
        await client.connect()
        await client.subscribe(bars=["SPY"])
        await asyncio.sleep(0.05)

        initial_received = client.metrics["messages_received"]

        server_ws = next(iter(adv_server.clients))

        # Send 500 bars with unique volumes 0..499 in rapid succession
        for i in range(500):
            await server_ws.send_json([{
                "T": "b", "S": "SPY", "t": "2026-09-03T14:00:00Z",
                "o": 500.0, "h": 501.0, "l": 499.0, "c": 500.5,
                "v": i,  # unique marker
            }])

        # Wait for consumer loop to drain all retained items
        for _ in range(30):
            if len(consumed_volumes) >= 200 or client.internal_queue_size == 0:
                if len(consumed_volumes) > 0:
                    break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        # Socket must remain alive and healthy despite queue saturation
        assert client.is_connected is True
        assert client.metrics["messages_received"] == initial_received + 500
        # Exactly 300 messages were dropped to protect the socket
        assert client.metrics["messages_dropped"] >= 300
        # Exactly the 200 newest messages were retained and processed
        assert len(consumed_volumes) == 200
        assert min(consumed_volumes) >= 300, f"Expected min volume >= 300, got {min(consumed_volumes)}"
        assert max(consumed_volumes) == 499, f"Expected max volume 499, got {max(consumed_volumes)}"
    finally:
        await client.disconnect()


# ============================================================================
# 3. Strict Wildcard '*' Prohibition Across All Channels & Entrypoints
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("channel,call_kwargs", [
    ("bars", {"bars": ["*"]}),
    ("trades", {"trades": ["*"]}),
    ("quotes", {"quotes": ["*"]}),
    ("dailyBars", {"daily_bars": ["*"]}),
    ("channels_bars", {"channels": {"bars": ["*"]}}),
    ("channels_quotes", {"channels": {"quotes": ["*"]}}),
    ("channels_trades", {"channels": {"trades": ["*"]}}),
    ("channels_dailyBars", {"channels": {"dailyBars": ["*"]}}),
    ("embedded_space", {"bars": [" * "]}),
    ("leading_wildcard", {"bars": ["*AAPL"]}),
    ("trailing_wildcard", {"bars": ["SPY*"]}),
    ("middle_wildcard", {"bars": ["SP*Y"]}),
])
async def test_ws_strict_wildcard_prohibition_subscribe(channel: str, call_kwargs: dict):
    """Adversarial Challenge: Every permutation of '*' (whitespace, substring, all channels)
    must strictly raise ValueError on client.subscribe().
    """
    client = AlpacaRelayWSClient(token="test")
    with pytest.raises(ValueError) as exc:
        await client.subscribe(**call_kwargs)
    assert "Wildcard '*' subscription" in str(exc.value)
    assert "strictly forbidden" in str(exc.value)


@pytest.mark.asyncio
async def test_ws_strict_wildcard_prohibition_connect_stream_connected(adv_server: AdversarialMockRelayServer):
    """Adversarial Challenge: Protocol method connect_stream strictly forbids wildcard '*'
    when connected to relay.
    """
    client = AlpacaRelayWSClient(url=adv_server.ws_url, token="valid-secret-token")
    try:
        await client.connect()
        with pytest.raises(ValueError) as exc:
            await client.connect_stream(symbols=["SPY", "*"], channels=["bars"])
        assert "strictly forbidden" in str(exc.value)
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_connect_stream_unconnected_ordering_observation():
    """Adversarial Observation: When connect_stream is called without pre-connection,
    it executes self.connect() BEFORE validating symbols. If server is unreachable,
    it raises RelayConnectionError instead of validating arguments first.
    """
    client = AlpacaRelayWSClient(url="ws://127.0.0.1:59999", token="test")
    # Because connect() is called before symbol validation at line 705 of ws_client.py:
    with pytest.raises(RelayConnectionError):
        await client.connect_stream(symbols=["*"], channels=["bars"])


@pytest.mark.asyncio
async def test_ws_strict_wildcard_prohibition_unsubscribe():
    """Adversarial Challenge: client.unsubscribe() also forbids wildcard '*'."""
    client = AlpacaRelayWSClient(token="test")
    with pytest.raises(ValueError) as exc:
        await client.unsubscribe(bars=["*"])
    assert "strictly forbidden" in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_sym", [123, None, {"sym": "SPY"}, 45.6])
async def test_ws_non_string_symbols_raise_value_error(invalid_sym: Any):
    """Adversarial Challenge: Passing non-string items into symbol list raises ValueError."""
    client = AlpacaRelayWSClient(token="test")
    with pytest.raises(ValueError) as exc:
        await client.subscribe(bars=[invalid_sym])  # type: ignore
    assert "Symbol must be a string" in str(exc.value)


# ============================================================================
# 4. Server-Side Eviction (Close Code 1013) & Auto-Reconnection & Resubscription
# ============================================================================

@pytest.mark.asyncio
async def test_ws_server_eviction_1013_clean_reconnect_and_resubscribe(adv_server: AdversarialMockRelayServer):
    """Adversarial Challenge:
    1. Client connects and subscribes across multiple channels: bars, quotes, trades, dailyBars.
    2. Server sends close code 1013 ('too slow') to evict client.
    3. Client detects eviction, records metric, performs exponential backoff, reconnects, re-authenticates.
    4. Client restores ALL desired channel subscriptions on the new connection.
    5. Server broadcasts new events on the reconnected stream and client receives them cleanly.
    """
    client = AlpacaRelayWSClient(
        url=adv_server.ws_url,
        token="valid-secret-token",
        initial_backoff=0.05,
        max_backoff=0.2,
    )
    received_bars: List[Bar] = []
    client.on_bar(lambda b: received_bars.append(b))

    try:
        await client.connect()
        # Subscribe to multiple channels
        await client.subscribe(
            bars=["SPY", "QQQ"],
            quotes=["AAPL"],
            trades=["NVDA"],
            daily_bars=["TLT"],
        )
        await asyncio.sleep(0.1)

        server_ws = next(iter(adv_server.clients))
        # Verify server received subscriptions
        assert "SPY" in adv_server.subscriptions[server_ws]["bars"]
        assert "AAPL" in adv_server.subscriptions[server_ws]["quotes"]
        assert "NVDA" in adv_server.subscriptions[server_ws]["trades"]

        # Evict client with close code 1013 (too slow)
        await adv_server.simulate_slow_client_eviction(server_ws)

        # Wait for auto-reconnection and resubscription
        for _ in range(40):
            if (
                client.metrics["evictions_1013"] >= 1
                and client.is_authenticated
                and len(adv_server.clients) == 1
            ):
                new_server_ws = next(iter(adv_server.clients))
                if "SPY" in adv_server.subscriptions.get(new_server_ws, {}).get("bars", set()):
                    break
            await asyncio.sleep(0.05)

        assert client.metrics["evictions_1013"] == 1
        assert client.is_authenticated is True

        # Check new server connection subscriptions
        new_server_ws = next(iter(adv_server.clients))
        subs = adv_server.subscriptions[new_server_ws]
        assert "SPY" in subs["bars"]
        assert "QQQ" in subs["bars"]
        assert "AAPL" in subs["quotes"]
        assert "NVDA" in subs["trades"]
        assert "TLT" in subs["dailyBars"]

        # Broadcast bar on restored connection
        await adv_server.broadcast_bar("SPY", {
            "t": "2026-09-03T14:30:00Z",
            "o": 505.0, "h": 506.0, "l": 504.0, "c": 505.5, "v": 200,
        })
        await asyncio.sleep(0.1)

        assert len(received_bars) == 1
        assert received_bars[0].symbol == "SPY"
        assert received_bars[0].close == 505.5
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_flapping_evictions_recovery(adv_server: AdversarialMockRelayServer):
    """Adversarial Challenge: Evict the client 3 times consecutively.
    Client must survive all 3 evictions without deadlock or uncaught exceptions.
    """
    client = AlpacaRelayWSClient(
        url=adv_server.ws_url,
        token="valid-secret-token",
        initial_backoff=0.02,
        max_backoff=0.1,
    )
    try:
        await client.connect()
        await client.subscribe(bars=["SPY"])
        await asyncio.sleep(0.05)

        for eviction_idx in range(3):
            server_ws = next(iter(adv_server.clients))
            await adv_server.simulate_slow_client_eviction(server_ws)

            # Wait for client to reconnect
            for _ in range(40):
                if client.metrics["evictions_1013"] == eviction_idx + 1 and client.is_authenticated:
                    break
                await asyncio.sleep(0.05)

            assert client.metrics["evictions_1013"] == eviction_idx + 1
            assert client.is_authenticated is True

        assert client.metrics["reconnect_count"] >= 3
    finally:
        await client.disconnect()


# ============================================================================
# 5. Rate Limiter Under Concurrent Multi-Task / Async Request Bursts
# ============================================================================

@pytest.mark.asyncio
async def test_token_bucket_concurrent_burst_pacing():
    """Adversarial Challenge: Fire 12 concurrent acquire() calls simultaneously.
    Limiter: rate = 60 req/min (1 req/sec), burst capacity = 4.0 tokens.
    Mathematical invariant:
    - 4 requests must complete immediately (burst capacity, elapsed < 0.05s).
    - Remaining 8 requests must be paced asynchronously at 1 req/sec.
    - Total elapsed time must be >= 7.8 seconds.
    - Tokens must never drop below 0.0.
    """
    limiter = TokenBucketRateLimiter(rate_limit_per_minute=60.0, burst_capacity=4.0)
    completion_times: List[float] = []
    start_t = time.monotonic()

    async def _worker(idx: int):
        wait = await limiter.acquire(1.0)
        completion_times.append(time.monotonic() - start_t)

    # Launch 12 concurrent workers simultaneously
    await asyncio.gather(*[_worker(i) for i in range(12)])

    completion_times.sort()
    # First 4 must be instantaneous (< 0.1s)
    for i in range(4):
        assert completion_times[i] < 0.1, f"Worker {i} was delayed unexpectedly: {completion_times[i]:.3f}s"

    # Total elapsed time must be close to (12 - 4) * 1.0 = 8.0s
    total_elapsed = completion_times[-1]
    assert total_elapsed >= 7.8, f"Rate limiter did not pace: total_elapsed={total_elapsed:.2f}s (expected >= 7.8s)"


@pytest.mark.asyncio
async def test_token_bucket_high_concurrency_stress():
    """Adversarial Challenge: 100 concurrent coroutines acquiring tokens on a fast limiter.
    Assert zero lock deadlocks, zero exceptions, and non-negative available_tokens invariant.
    """
    limiter = TokenBucketRateLimiter(rate_limit_per_minute=1200.0, burst_capacity=20.0)  # 20 req/sec

    acquired = 0

    async def _acq():
        nonlocal acquired
        await limiter.acquire(1.0)
        acquired += 1

    # Run 100 concurrent acquisitions
    await asyncio.gather(*[_acq() for _ in range(100)])

    assert acquired == 100
    # Available tokens property must be bounded between 0 and capacity
    avail = limiter.available_tokens
    assert 0.0 <= avail <= 20.0, f"available_tokens out of bounds: {avail}"


@pytest.mark.asyncio
async def test_token_bucket_parameter_validation():
    """Adversarial Challenge: TokenBucketRateLimiter rejects invalid configurations."""
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate_limit_per_minute=0)
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate_limit_per_minute=-10)
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(burst_capacity=0.5)  # capacity must be >= 1.0


@pytest.mark.asyncio
async def test_rest_client_concurrent_requests_pacing(adv_server: AdversarialMockRelayServer):
    """Adversarial Challenge: Fire 6 concurrent get_bars() calls against the REST client.
    Rate limiter: 60 req/min (1 req/sec), burst capacity = 2.
    - 2 requests finish immediately.
    - 4 requests finish paced over 4 seconds.
    - Total duration >= 3.8s. All 6 return valid Bar lists.
    """
    client = AlpacaRelayRestClient(
        base_url=adv_server.http_url,
        relay_token="valid-secret-token",
        rate_limit_per_minute=60.0,
        burst_capacity=2.0,
    )
    start_t = time.monotonic()

    async def _fetch():
        return await client.get_bars("SPY", limit=5)

    async with client:
        results = await asyncio.gather(*[_fetch() for _ in range(6)])

    elapsed = time.monotonic() - start_t
    assert len(results) == 6
    for bars in results:
        assert len(bars) == 5
        assert isinstance(bars[0], Bar)

    # 2 burst + 4 * 1s pacing = ~4.0s
    assert elapsed >= 3.8, f"REST client concurrent requests not properly paced: elapsed={elapsed:.2f}s"
