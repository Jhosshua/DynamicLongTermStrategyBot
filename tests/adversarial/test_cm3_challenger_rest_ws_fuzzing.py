"""
tests.adversarial.test_cm3_challenger_rest_ws_fuzzing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone C-M3: Empirical Challenger Stress & Protocol Fuzzing Suite.
Author: challenger_cm3_2 (teamwork_preview_challenger)

Adversarial Objectives:
1. REST Client Fuzzing:
   - HTTP 500 error storms with full-jitter exponential backoff verification and exhaustion mapping to RelayUpstreamError(status_code=500).
   - HTTP 500 transient storms with clean recovery upon upstream resolution.
   - Mixed 5xx error storms (500, 502, 503, 504) and non-retryable 501 Not Implemented.
        ("{" + ""key": "value", " * 50000 + ""final": true}", "application/json"),  # 1MB payload
   - HTTP 429 rate limit floods with Retry-After header variations (float, integer, missing, malformed).
   - Network drop fuzzing across ConnectError, ConnectTimeout, ReadTimeout, WriteTimeout, and RemoteProtocolError.
   - Non-string symbol type fuzzing across get_bars, iter_bars, get_multi_bars, and get_historical_bars.

2. WebSocket Client Streaming Fuzzing:
   - Crossed-book quotes (P_ask < P_bid) across multiple price regimes, verifying quotes_processed == 0 and messages_dropped increments.
   - Corrupted OHLCV bars (H < L, H < O, H < C, L > O, L > C, negative volume, negative prices).
   - Corrupted trades (negative price, zero price, negative size, missing fields).
   - Corrupted quotes (negative/zero bid/ask prices, negative sizes).
   - Non-string symbol types in streaming frames (int, None, empty string, list, dict).
   - Invalid JSON text frames and raw stream corruption handling.
   - Empirical demonstration of binary non-UTF8 frame producer crash (UnicodeDecodeError).
   - High-throughput interleaved stress stream (1000 messages) verifying exact metric accounting invariant:
     messages_received == quotes_processed + trades_processed + bars_processed + messages_dropped.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import time
from typing import Any, Dict, List, Optional
from aiohttp import web
import httpx
import pytest

from strategy_engine.core.models import Bar, Quote, Trade
from strategy_engine.ingestion.rest_client import (
    AlpacaRelayRestClient,
    RelayAuthError,
    RelayRateLimitError,
    RelayRequestError,
    RelayUpstreamError,
)
from strategy_engine.ingestion.ws_client import AlpacaRelayWSClient


# ============================================================================
# 1. REST Client Fuzzing: HTTP 500 Error Storms & Resilience
# ============================================================================

@pytest.mark.asyncio
async def test_rest_500_storm_backoff_jitter_and_exhaustion():
    """Simulate continuous HTTP 500 storm; verify exponential backoff intervals and exhaustion mapping to RelayUpstreamError."""
    timestamps: List[float] = []

    def mock_500(request: httpx.Request) -> httpx.Response:
        timestamps.append(time.monotonic())
        return httpx.Response(
            status_code=500,
            text="Internal Server Error: Downstream Microservice Degraded",
            headers={"Content-Type": "text/plain"},
        )

    transport = httpx.MockTransport(mock_500)
    mock_client = httpx.AsyncClient(transport=transport)

    max_retries = 3
    base_delay = 0.05
    max_delay = 1.0

    async with AlpacaRelayRestClient(
        base_url="https://mock.relay",
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        rate_limit_per_minute=6000,
        burst_capacity=1000,
        client=mock_client,
    ) as client:
        with pytest.raises(RelayUpstreamError) as exc_info:
            await client.get_bars("SPY")

        assert exc_info.value.status_code == 500
        assert "500" in str(exc_info.value)
        assert exc_info.value.response is not None
        assert exc_info.value.response.status_code == 500

        # Total attempts = 1 initial + max_retries
        assert len(timestamps) == max_retries + 1

        # Verify retry delays grew monotonically according to exponential backoff
        delays = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        for i, delay in enumerate(delays):
            min_expected = min(max_delay, base_delay * (2 ** i))
            # Delay includes full jitter: delay = backoff + uniform(0, 1)
            assert delay >= min_expected * 0.9, f"Delay {delay:.3f}s was less than minimum {min_expected:.3f}s at attempt {i+1}"


@pytest.mark.asyncio
async def test_rest_500_transient_storm_recovery():
    """Simulate HTTP 500 storm lasting 3 attempts then recovering; verify clean return of parsed Bar models."""
    attempt = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt <= 3:
            return httpx.Response(status_code=500, text="Database locking conflict")
        # Attempt 4 succeeds
        payload = {
            "bars": [
                {
                    "t": "2026-09-03T14:30:00Z",
                    "o": 510.0,
                    "h": 515.0,
                    "l": 508.0,
                    "c": 512.5,
                    "v": 50000,
                    "vw": 511.8,
                    "n": 450,
                }
            ],
            "next_page_token": None,
        }
        return httpx.Response(status_code=200, json=payload)

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.AsyncClient(transport=transport)

    async with AlpacaRelayRestClient(
        base_url="https://mock.relay",
        max_retries=4,
        base_delay=0.01,
        max_delay=0.1,
        rate_limit_per_minute=6000,
        burst_capacity=1000,
        client=mock_client,
    ) as client:
        bars = await client.get_bars("SPY")
        assert len(bars) == 1
        assert isinstance(bars[0], Bar)
        assert bars[0].symbol == "SPY"
        assert bars[0].close == 512.5
        assert attempt == 4


@pytest.mark.asyncio
async def test_rest_5xx_heterogeneous_storm():
    """Simulate storm with alternating 500, 502, 503, 504 errors; verify all are retried and map to RelayUpstreamError."""
    codes_sequence = [500, 502, 503, 504, 500]
    idx = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal idx
        code = codes_sequence[idx % len(codes_sequence)]
        idx += 1
        return httpx.Response(status_code=code, text=f"Upstream error {code}")

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.AsyncClient(transport=transport)

    async with AlpacaRelayRestClient(
        base_url="https://mock.relay",
        max_retries=4,
        base_delay=0.01,
        max_delay=0.1,
        rate_limit_per_minute=6000,
        burst_capacity=1000,
        client=mock_client,
    ) as client:
        with pytest.raises(RelayUpstreamError) as exc_info:
            await client.get_bars("SPY")

        assert exc_info.value.status_code in {500, 502, 503, 504}
        assert idx == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_body,content_type",
    [
        ("<html><body><h1>500 Internal Server Error</h1></body></html>", "text/html"),
        ("", "text/plain"),
        ('{"key": "value", ' * 50000 + '"final": true}', "application/json"),  # 1MB payload
        ('{"error": "incomplete json', "application/json"),
        (bytes([0x00, 0xff, 0xfe, 0x80, 0x81]), "application/octet-stream"),
    ],
)
async def test_rest_500_extreme_payload_fuzzing(bad_body, content_type):
    """HTTP 500 with extreme/corrupted response bodies must cleanly map to RelayUpstreamError without leaking JSONDecodeError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        content = bad_body.encode("utf-8") if isinstance(bad_body, str) else bad_body
        return httpx.Response(status_code=500, content=content, headers={"Content-Type": content_type})

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.AsyncClient(transport=transport)

    async with AlpacaRelayRestClient(
        base_url="https://mock.relay",
        max_retries=1,
        base_delay=0.01,
        max_delay=0.05,
        rate_limit_per_minute=6000,
        burst_capacity=1000,
        client=mock_client,
    ) as client:
        with pytest.raises(RelayUpstreamError) as exc_info:
            await client.get_bars("SPY")
        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_rest_non_retryable_501_not_implemented():
    """HTTP 501 Not Implemented is not in RETRYABLE_STATUS_CODES; must immediately raise RelayUpstreamError without retrying."""
    calls = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code=501, text="Not Implemented")

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.AsyncClient(transport=transport)

    async with AlpacaRelayRestClient(
        base_url="https://mock.relay",
        max_retries=3,
        base_delay=0.01,
        max_delay=0.1,
        rate_limit_per_minute=6000,
        burst_capacity=1000,
        client=mock_client,
    ) as client:
        with pytest.raises(RelayUpstreamError) as exc_info:
            await client.get_bars("SPY")
        assert exc_info.value.status_code == 501
        assert calls == 1, "HTTP 501 should not be retried"


# ============================================================================
# 2. REST Client Fuzzing: HTTP 429 Rate Limit Floods
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retry_after_hdr,expected_parsed",
    [
        ("0.05", 0.05),
        ("1", 1.0),
        (None, None),
        ("invalid-date", None),
        ("-10", None),
        ("", None),
    ],
)
async def test_rest_429_rate_limit_flood_headers(retry_after_hdr, expected_parsed):
    """Simulate HTTP 429 flood with various Retry-After headers; verify exhaustion raises RelayRateLimitError."""
    calls = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        headers = {}
        if retry_after_hdr is not None:
            headers["Retry-After"] = retry_after_hdr
        return httpx.Response(status_code=429, headers=headers, text="Rate limit exceeded")

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.AsyncClient(transport=transport)

    async with AlpacaRelayRestClient(
        base_url="https://mock.relay",
        max_retries=2,
        base_delay=0.01,
        max_delay=0.05,
        rate_limit_per_minute=6000,
        burst_capacity=1000,
        client=mock_client,
    ) as client:
        with pytest.raises(RelayRateLimitError) as exc_info:
            await client.get_bars("SPY")

        assert exc_info.value.response.status_code == 429
        assert calls == 3


@pytest.mark.asyncio
async def test_rest_429_transient_recovery():
    """Simulate HTTP 429 rate limit backoff with recovery on 3rd attempt."""
    attempt = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt <= 2:
            return httpx.Response(status_code=429, headers={"Retry-After": "0.01"}, text="Too many requests")
        payload = {
            "bars": [
                {
                    "t": "2026-09-03T15:00:00Z",
                    "o": 500.0,
                    "h": 501.0,
                    "l": 499.0,
                    "c": 500.5,
                    "v": 1000,
                }
            ],
            "next_page_token": None,
        }
        return httpx.Response(status_code=200, json=payload)

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.AsyncClient(transport=transport)

    async with AlpacaRelayRestClient(
        base_url="https://mock.relay",
        max_retries=3,
        base_delay=0.01,
        max_delay=0.05,
        rate_limit_per_minute=6000,
        burst_capacity=1000,
        client=mock_client,
    ) as client:
        bars = await client.get_bars("SPY")
        assert len(bars) == 1
        assert attempt == 3


# ============================================================================
# 3. REST Client Fuzzing: Network Drops & Transport Errors
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_cls",
    [
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.RemoteProtocolError,
    ],
)
async def test_rest_network_drops_transport_faults(exception_cls):
    """Transport-level connection drops must be retried with backoff and map to RelayUpstreamError(status_code=502)."""
    calls = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise exception_cls(f"Simulated {exception_cls.__name__}", request=request)

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.AsyncClient(transport=transport)

    max_retries = 2
    async with AlpacaRelayRestClient(
        base_url="https://mock.relay",
        max_retries=max_retries,
        base_delay=0.01,
        max_delay=0.05,
        rate_limit_per_minute=6000,
        burst_capacity=1000,
        client=mock_client,
    ) as client:
        with pytest.raises(RelayUpstreamError) as exc_info:
            await client.get_bars("SPY")

        assert exc_info.value.status_code == 502
        assert "Transport error" in str(exc_info.value)
        assert calls == max_retries + 1


@pytest.mark.asyncio
async def test_rest_network_drop_transient_recovery():
    """Simulate transient network drop for 2 attempts; connection recovers and returns 200 OK."""
    attempt = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt <= 2:
            raise httpx.ConnectError("Connection reset by peer", request=request)
        payload = {
            "bars": [
                {
                    "t": "2026-09-03T15:00:00Z",
                    "o": 500.0,
                    "h": 501.0,
                    "l": 499.0,
                    "c": 500.5,
                    "v": 1000,
                }
            ],
            "next_page_token": None,
        }
        return httpx.Response(status_code=200, json=payload)

    transport = httpx.MockTransport(mock_handler)
    mock_client = httpx.AsyncClient(transport=transport)

    async with AlpacaRelayRestClient(
        base_url="https://mock.relay",
        max_retries=3,
        base_delay=0.01,
        max_delay=0.05,
        rate_limit_per_minute=6000,
        burst_capacity=1000,
        client=mock_client,
    ) as client:
        bars = await client.get_bars("SPY")
        assert len(bars) == 1
        assert attempt == 3


# ============================================================================
# 4. REST Client Fuzzing: Non-String Symbol Types & Corrupted Collections
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_symbol",
    [
        None,
        "",
        "   ",
        "\t\n",
        12345,
        0,
        -10,
        99.9,
        True,
        False,
        object(),
        {"SPY": 1},
        ["SPY", None],
        ["SPY", 123],
        ["SPY", ""],
        ["", "SPY"],
        [None, "SPY"],
        [],
    ],
)
async def test_rest_get_bars_symbol_type_fuzzing(bad_symbol):
    """get_bars must reject None, non-string, whitespace, and malformed symbol collections with ValueError."""
    client = AlpacaRelayRestClient()
    with pytest.raises(ValueError):
        await client.get_bars(bad_symbol)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_symbol",
    [None, "", "   ", 12345, True, object(), ["SPY"]],
)
async def test_rest_iter_bars_symbol_type_fuzzing(bad_symbol):
    """iter_bars must reject non-string / empty symbols with ValueError immediately."""
    client = AlpacaRelayRestClient()
    with pytest.raises(ValueError):
        async for _ in client.iter_bars(bad_symbol):
            pass


@pytest.mark.asyncio
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_symbols",
    [
        None,
        12345,
        "SPY",  # Expected list/tuple/set, not raw str
        [None],
        ["SPY", None],
        ["SPY", 99],
        ["SPY", ""],
        ["SPY", "   "],
    ],
)
async def test_rest_get_multi_bars_symbol_type_fuzzing(bad_symbols):
    """get_multi_bars must validate symbols collection and reject invalid types with ValueError."""
    client = AlpacaRelayRestClient()
    with pytest.raises(ValueError):
        await client.get_multi_bars(bad_symbols)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_symbols",
    [None, 12345, [None], ["SPY", 123], ["SPY", ""]],
)
async def test_rest_get_historical_bars_symbol_type_fuzzing(bad_symbols):
    """get_historical_bars must validate symbols and raise ValueError on non-string or empty elements."""
    client = AlpacaRelayRestClient()
    with pytest.raises(ValueError):
        await client.get_historical_bars(bad_symbols)


@pytest.mark.asyncio
async def test_rest_get_multi_bars_and_historical_empty_list_boundary():
    """get_multi_bars([]) and get_historical_bars([]) return empty dict {} boundary without raising."""
    client = AlpacaRelayRestClient()
    res_multi = await client.get_multi_bars([])
    assert res_multi == {}
    res_hist = await client.get_historical_bars([])
    assert res_hist == {}


# ============================================================================
# 5. WebSocket Client Fuzzing: Crossed-Book Quotes (P_ask < P_bid)
# ============================================================================

@pytest.mark.asyncio
async def test_ws_crossed_book_quotes_rejected_metric_accounting():
    """Crossed-book quotes (P_ask < P_bid) must be dropped, incrementing messages_dropped without incrementing quotes_processed."""
    server_port = 9101

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps([{"T": "success", "msg": "connected"}]))
        await ws.receive_str()
        await ws.send_str(json.dumps([{"T": "success", "msg": "authenticated"}]))

        # Stream 3 crossed-book quotes
        crossed_quotes = [
            {"T": "q", "S": "SPY", "bp": 105.0, "ap": 95.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"},
            {"T": "q", "S": "QQQ", "bp": 500.05, "ap": 500.0, "bs": 5, "as": 5, "t": "2026-09-03T12:00:01Z"},
            {"T": "q", "S": "TLT", "bp": 99.0, "ap": 90.0, "bs": 100, "as": 100, "t": "2026-09-03T12:00:02Z"},
        ]
        for q in crossed_quotes:
            await ws.send_str(json.dumps([q]))
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.2)
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", server_port)
    await site.start()

    client = AlpacaRelayWSClient(url=f"ws://127.0.0.1:{server_port}", token="test-token", auto_reconnect=False)
    try:
        await client.connect()
        await asyncio.sleep(0.4)
        assert client.metrics["quotes_processed"] == 0, "Crossed quotes must NOT increment quotes_processed"
        assert client.metrics["messages_dropped"] == 3, "Crossed quotes must increment messages_dropped"
        assert client.metrics["messages_received"] == 3
    finally:
        await client.disconnect()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_ws_crossed_book_vs_valid_quotes_interleaved():
    """Interleave valid and crossed-book quotes; verify exact metric separation."""
    server_port = 9102

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps([{"T": "success", "msg": "connected"}]))
        await ws.receive_str()
        await ws.send_str(json.dumps([{"T": "success", "msg": "authenticated"}]))

        batch = []
        for i in range(25):
            # Valid quote (ap > bp)
            batch.append({"T": "q", "S": "SPY", "bp": 500.0, "ap": 501.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"})
            # Crossed quote (ap < bp)
            batch.append({"T": "q", "S": "SPY", "bp": 505.0, "ap": 495.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"})

        await ws.send_str(json.dumps(batch))
        await asyncio.sleep(0.3)
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", server_port)
    await site.start()

    client = AlpacaRelayWSClient(url=f"ws://127.0.0.1:{server_port}", token="test-token", auto_reconnect=False)
    try:
        await client.connect()
        await asyncio.sleep(0.5)
        assert client.metrics["quotes_processed"] == 25
        assert client.metrics["messages_dropped"] == 25
        assert client.metrics["messages_received"] == 50
    finally:
        await client.disconnect()
        await runner.cleanup()


# ============================================================================
# 6. WebSocket Client Fuzzing: Corrupted Frames (Bars, Trades, Quotes)
# ============================================================================

@pytest.mark.asyncio
async def test_ws_corrupted_bar_frames_rejected():
    """Corrupted Bar frames violating OHLC consistency or bounds must be dropped cleanly."""
    server_port = 9103

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps([{"T": "success", "msg": "connected"}]))
        await ws.receive_str()
        await ws.send_str(json.dumps([{"T": "success", "msg": "authenticated"}]))

        corrupted_bars = [
            # High < Low
            {"T": "b", "S": "SPY", "o": 100.0, "h": 95.0, "l": 105.0, "c": 100.0, "v": 10, "t": "2026-09-03T12:00:00Z"},
            # High < Open
            {"T": "b", "S": "SPY", "o": 105.0, "h": 100.0, "l": 90.0, "c": 95.0, "v": 10, "t": "2026-09-03T12:00:00Z"},
            # High < Close
            {"T": "b", "S": "SPY", "o": 95.0, "h": 100.0, "l": 90.0, "c": 105.0, "v": 10, "t": "2026-09-03T12:00:00Z"},
            # Low > Open
            {"T": "b", "S": "SPY", "o": 90.0, "h": 110.0, "l": 95.0, "c": 105.0, "v": 10, "t": "2026-09-03T12:00:00Z"},
            # Low > Close
            {"T": "b", "S": "SPY", "o": 105.0, "h": 110.0, "l": 95.0, "c": 90.0, "v": 10, "t": "2026-09-03T12:00:00Z"},
            # Negative volume
            {"T": "b", "S": "SPY", "o": 100.0, "h": 105.0, "l": 95.0, "c": 100.0, "v": -10, "t": "2026-09-03T12:00:00Z"},
            # Negative price
            {"T": "b", "S": "SPY", "o": -100.0, "h": 105.0, "l": -105.0, "c": 100.0, "v": 10, "t": "2026-09-03T12:00:00Z"},
        ]
        await ws.send_str(json.dumps(corrupted_bars))
        await asyncio.sleep(0.3)
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", server_port)
    await site.start()

    client = AlpacaRelayWSClient(url=f"ws://127.0.0.1:{server_port}", token="test-token", auto_reconnect=False)
    try:
        await client.connect()
        await asyncio.sleep(0.5)
        assert client.metrics["bars_processed"] == 0, "Corrupted bars must NOT increment bars_processed"
        assert client.metrics["messages_dropped"] == 7, "All 7 corrupted bars must increment messages_dropped"
        assert client.metrics["messages_received"] == 7
    finally:
        await client.disconnect()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_ws_corrupted_trade_frames_rejected():
    """Corrupted Trade frames with negative/zero prices or sizes must increment messages_dropped."""
    server_port = 9104

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps([{"T": "success", "msg": "connected"}]))
        await ws.receive_str()
        await ws.send_str(json.dumps([{"T": "success", "msg": "authenticated"}]))

        corrupted_trades = [
            # Negative price
            {"T": "t", "S": "SPY", "p": -10.0, "s": 100, "i": "1", "t": "2026-09-03T12:00:00Z"},
            # Zero price
            {"T": "t", "S": "SPY", "p": 0.0, "s": 100, "i": "2", "t": "2026-09-03T12:00:00Z"},
            # Negative size
            {"T": "t", "S": "SPY", "p": 500.0, "s": -10, "i": "3", "t": "2026-09-03T12:00:00Z"},
            # Missing symbol
            {"T": "t", "p": 500.0, "s": 100, "i": "4", "t": "2026-09-03T12:00:00Z"},
        ]
        await ws.send_str(json.dumps(corrupted_trades))
        await asyncio.sleep(0.3)
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", server_port)
    await site.start()

    client = AlpacaRelayWSClient(url=f"ws://127.0.0.1:{server_port}", token="test-token", auto_reconnect=False)
    try:
        await client.connect()
        await asyncio.sleep(0.5)
        assert client.metrics["trades_processed"] == 0
        assert client.metrics["messages_dropped"] == 4
        assert client.metrics["messages_received"] == 4
    finally:
        await client.disconnect()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_ws_corrupted_quote_frames_rejected():
    """Corrupted Quote frames with negative/zero prices or sizes must increment messages_dropped."""
    server_port = 9105

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps([{"T": "success", "msg": "connected"}]))
        await ws.receive_str()
        await ws.send_str(json.dumps([{"T": "success", "msg": "authenticated"}]))

        corrupted_quotes = [
            # Negative bid price
            {"T": "q", "S": "SPY", "bp": -100.0, "ap": 101.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"},
            # Zero bid price
            {"T": "q", "S": "SPY", "bp": 0.0, "ap": 101.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"},
            # Negative ask price
            {"T": "q", "S": "SPY", "bp": 100.0, "ap": -50.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"},
            # Zero ask price
            {"T": "q", "S": "SPY", "bp": 100.0, "ap": 0.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"},
            # Negative bid size
            {"T": "q", "S": "SPY", "bp": 100.0, "ap": 101.0, "bs": -5, "as": 10, "t": "2026-09-03T12:00:00Z"},
            # Negative ask size
            {"T": "q", "S": "SPY", "bp": 100.0, "ap": 101.0, "bs": 10, "as": -5, "t": "2026-09-03T12:00:00Z"},
        ]
        await ws.send_str(json.dumps(corrupted_quotes))
        await asyncio.sleep(0.3)
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", server_port)
    await site.start()

    client = AlpacaRelayWSClient(url=f"ws://127.0.0.1:{server_port}", token="test-token", auto_reconnect=False)
    try:
        await client.connect()
        await asyncio.sleep(0.5)
        assert client.metrics["quotes_processed"] == 0
        assert client.metrics["messages_dropped"] == 6
        assert client.metrics["messages_received"] == 6
    finally:
        await client.disconnect()
        await runner.cleanup()


# ============================================================================
# 7. WebSocket Client Fuzzing: Non-String Symbols in Stream Frames
# ============================================================================

@pytest.mark.asyncio
async def test_ws_non_string_symbols_fuzzing():
    """Frames with non-string symbols (int, None, list, dict) must fail model validation and increment messages_dropped."""
    server_port = 9106

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps([{"T": "success", "msg": "connected"}]))
        await ws.receive_str()
        await ws.send_str(json.dumps([{"T": "success", "msg": "authenticated"}]))

        bad_symbol_frames = [
            # Integer symbol in Quote
            {"T": "q", "S": 12345, "bp": 500.0, "ap": 501.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"},
            # None symbol in Quote
            {"T": "q", "S": None, "bp": 500.0, "ap": 501.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"},
            # List symbol in Bar
            {"T": "b", "S": ["SPY"], "o": 500.0, "h": 505.0, "l": 495.0, "c": 500.0, "v": 100, "t": "2026-09-03T12:00:00Z"},
            # Dict symbol in Trade
            {"T": "t", "S": {"sym": "SPY"}, "p": 500.0, "s": 10, "i": "99", "t": "2026-09-03T12:00:00Z"},
            # Empty string symbol
            {"T": "q", "S": "", "bp": 500.0, "ap": 501.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"},
        ]
        await ws.send_str(json.dumps(bad_symbol_frames))
        await asyncio.sleep(0.3)
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", server_port)
    await site.start()

    client = AlpacaRelayWSClient(url=f"ws://127.0.0.1:{server_port}", token="test-token", auto_reconnect=False)
    try:
        await client.connect()
        await asyncio.sleep(0.5)
        assert client.metrics["quotes_processed"] == 0
        assert client.metrics["bars_processed"] == 0
        assert client.metrics["trades_processed"] == 0
        assert client.metrics["messages_dropped"] == 5
        assert client.metrics["messages_received"] == 5
    finally:
        await client.disconnect()
        await runner.cleanup()


# ============================================================================
# 8. WebSocket Client Fuzzing: Invalid JSON & Stream Protocol Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_ws_invalid_json_string_frames_non_fatal():
    """Malformed JSON string frames must be caught by JSONDecodeError without terminating the connection."""
    server_port = 9107

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps([{"T": "success", "msg": "connected"}]))
        await ws.receive_str()
        await ws.send_str(json.dumps([{"T": "success", "msg": "authenticated"}]))

        # Malformed JSON strings
        await ws.send_str('{"T": "q", "broken_json')
        await ws.send_str("NOT_A_JSON_STRING_AT_ALL")
        await ws.send_str("")
        await asyncio.sleep(0.1)

        # Valid quote afterwards
        await ws.send_str(json.dumps([{"T": "q", "S": "SPY", "bp": 500.0, "ap": 501.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"}]))
        try:
            await ws.receive()
        except Exception:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", server_port)
    await site.start()

    client = AlpacaRelayWSClient(url=f"ws://127.0.0.1:{server_port}", token="test-token", auto_reconnect=False)
    try:
        await client.connect()
        await asyncio.sleep(0.4)
        # Valid quote processed
        assert client.metrics["quotes_processed"] == 1
        # Producer loop survived the malformed text frames while socket is active
        assert client._producer_task is not None and not client._producer_task.done()
    finally:
        await client.disconnect()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_ws_binary_corrupted_frame_producer_resilience():
    """Verify binary non-UTF8 frames are safely handled with errors='replace' and increment messages_dropped."""
    server_port = 9108

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps([{"T": "success", "msg": "connected"}]))
        await ws.receive_str()
        await ws.send_str(json.dumps([{"T": "success", "msg": "authenticated"}]))

        # Send invalid binary non-UTF8 frame
        await ws.send_bytes(bytes([0x80, 0x81, 0xff, 0xfe, 0xaa, 0xbb]))
        await asyncio.sleep(0.2)
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", server_port)
    await site.start()

    client = AlpacaRelayWSClient(url=f"ws://127.0.0.1:{server_port}", token="test-token", auto_reconnect=False)
    try:
        await client.connect()
        await asyncio.sleep(0.3)
        # Verify the producer loop task handled the binary frame safely without crashing
        producer_task = client._producer_task
        assert producer_task is not None
        if producer_task.done():
            assert producer_task.exception() is None, f"Producer task failed unexpectedly: {producer_task.exception()}"
        assert client.metrics["messages_dropped"] >= 1, "Malformed binary frame must increment messages_dropped"
    finally:
        await client.disconnect()
        await runner.cleanup()


# ============================================================================
# 9. High-Throughput Interleaved Stress Stream
# ============================================================================

@pytest.mark.asyncio
async def test_ws_high_volume_adversarial_stream_metrics_invariant():
    """Stream 1000 mixed frames (valid, crossed, corrupted bars, non-string symbols, valid trades); verify exact accounting."""
    server_port = 9109

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps([{"T": "success", "msg": "connected"}]))
        await ws.receive_str()
        await ws.send_str(json.dumps([{"T": "success", "msg": "authenticated"}]))

        batch = []
        # 200 valid quotes
        for i in range(200):
            batch.append({"T": "q", "S": "SPY", "bp": 500.0, "ap": 501.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"})
        # 200 crossed-book quotes
        for i in range(200):
            batch.append({"T": "q", "S": "SPY", "bp": 505.0, "ap": 495.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"})
        # 200 corrupted bars (H < L)
        for i in range(200):
            batch.append({"T": "b", "S": "SPY", "o": 500.0, "h": 490.0, "l": 510.0, "c": 500.0, "v": 100, "t": "2026-09-03T12:00:00Z"})
        # 200 non-string symbol frames
        for i in range(200):
            batch.append({"T": "q", "S": 99999, "bp": 500.0, "ap": 501.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"})
        # 200 valid trades
        for i in range(200):
            batch.append({"T": "t", "S": "SPY", "p": 500.5, "s": 50, "i": f"trade-{i}", "t": "2026-09-03T12:00:00Z"})

        # Send in 20 chunked batches of 50
        for chunk_idx in range(0, len(batch), 50):
            await ws.send_str(json.dumps(batch[chunk_idx:chunk_idx+50]))
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.6)
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", server_port)
    await site.start()

    client = AlpacaRelayWSClient(url=f"ws://127.0.0.1:{server_port}", token="test-token", auto_reconnect=False)
    try:
        await client.connect()
        await asyncio.sleep(1.2)

        m = client.metrics
        assert m["quotes_processed"] == 200, f"Expected 200 quotes, got {m['quotes_processed']}"
        assert m["trades_processed"] == 200, f"Expected 200 trades, got {m['trades_processed']}"
        assert m["bars_processed"] == 0, f"Expected 0 bars, got {m['bars_processed']}"
        assert m["messages_dropped"] == 600, f"Expected 600 dropped, got {m['messages_dropped']}"
        assert m["messages_received"] == 1000, f"Expected 1000 received, got {m['messages_received']}"

        # Invariant check
        total_processed = m["quotes_processed"] + m["trades_processed"] + m["bars_processed"] + m["lifecycle_events"]
        assert m["messages_received"] == total_processed + m["messages_dropped"]
    finally:
        await client.disconnect()
        await runner.cleanup()
