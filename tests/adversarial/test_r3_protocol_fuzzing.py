"""
tests.adversarial.test_r3_protocol_fuzzing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone C-M3: API Boundary & Protocol Fuzzing Test Suite.
Verifies:
1. CLI Subcommand Fuzzing:
   - Negative, zero, NaN, and Inf equity in `dry-run` and `rebalance`
   - Negative, zero, NaN, and Inf capital in `backtest`
   - Invalid table and format arguments in `export-metrics`
   - Global traceback suppression on all malformed/corrupted inputs
2. REST Proxy Client Fuzzing:
   - HTTP 500 Internal Server Error storms with exponential backoff and exhaustion
   - HTTP 500 transient recovery
   - HTTP 429 rate limit floods with Retry-After backoff
   - Network transport drops mapped to RelayUpstreamError
   - Symbol parameter fuzzing (None, non-string, empty string, corrupted lists)
3. WebSocket Streaming Client Fuzzing:
   - Crossed-book quotes (P_ask < P_bid) dropped without incrementing quotes_processed
   - Corrupted OHLCV bars and trades dropped without incrementing processed counters
   - Proper accounting of messages_dropped vs processed metrics
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import math
from typing import Any, Dict, List
import httpx
import pytest
from typer.testing import CliRunner

from strategy_engine.cli.main import app
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
# 1. CLI Boundary & Subcommand Fuzzing
# ============================================================================

@pytest.mark.parametrize("bad_equity", [-1000.0, -0.001, 0.0, float("nan"), float("inf"), float("-inf")])
def test_cli_dry_run_fuzzing_equity(bad_equity):
    """dry-run must reject negative, zero, NaN, and Inf equity with code 1 and red error."""
    runner = CliRunner()
    res = runner.invoke(app, ["dry-run", "--equity", str(bad_equity)])
    assert res.exit_code == 1, f"dry-run with equity={bad_equity} must exit with code 1"
    assert "traceback" not in res.output.lower(), f"Raw traceback found: {res.output}"
    assert "Portfolio equity must be a positive finite number" in res.output


@pytest.mark.parametrize("bad_equity", [-50000.0, -1.0, 0.0, float("nan"), float("inf"), float("-inf")])
def test_cli_rebalance_fuzzing_equity(bad_equity):
    """rebalance must reject negative, zero, NaN, and Inf equity with code 1 and red error."""
    runner = CliRunner()
    res = runner.invoke(app, ["rebalance", "--equity", str(bad_equity)])
    assert res.exit_code == 1, f"rebalance with equity={bad_equity} must exit with code 1"
    assert "traceback" not in res.output.lower(), f"Raw traceback found: {res.output}"
    assert "Portfolio equity must be a positive finite number" in res.output


@pytest.mark.parametrize("bad_capital", [-100000.0, -0.5, 0.0, float("nan"), float("inf"), float("-inf")])
def test_cli_backtest_fuzzing_capital(bad_capital):
    """backtest must reject negative, zero, NaN, and Inf capital with code 1 and red error."""
    runner = CliRunner()
    res = runner.invoke(app, ["backtest", "--capital", str(bad_capital)])
    assert res.exit_code == 1, f"backtest with capital={bad_capital} must exit with code 1"
    assert "traceback" not in res.output.lower(), f"Raw traceback found: {res.output}"
    assert "capital must be a positive finite number" in res.output


@pytest.mark.parametrize("bad_table", ["invalid_table", "users", "passwords", "portfolio_states", "fake"])
def test_cli_export_metrics_fuzzing_invalid_table(bad_table):
    """export-metrics must reject tables outside allowed set with code 1 and red error."""
    runner = CliRunner()
    res = runner.invoke(app, ["export-metrics", "--table", bad_table])
    assert res.exit_code == 1, f"export-metrics with table={bad_table} must exit with code 1"
    assert "traceback" not in res.output.lower(), f"Raw traceback found: {res.output}"
    assert "Invalid table" in res.output


@pytest.mark.parametrize("bad_format", ["xml", "yaml", "parquet", "protobuf", "html", "tsv"])
def test_cli_export_metrics_fuzzing_invalid_format(bad_format):
    """export-metrics must reject formats outside ('json', 'csv') with code 1 and red error."""
    runner = CliRunner()
    res = runner.invoke(app, ["export-metrics", "--format", bad_format])
    assert res.exit_code == 1, f"export-metrics with format={bad_format} must exit with code 1"
    assert "traceback" not in res.output.lower(), f"Raw traceback found: {res.output}"
    assert "Invalid format" in res.output


@pytest.mark.parametrize("bad_interval", ["nan", "inf", "+inf", "-inf", "0", "-10s"])
def test_cli_daemon_fuzzing_interval(bad_interval):
    """daemon must reject NaN, Inf, +Inf, -Inf, zero, and negative interval with code 1 and red error."""
    runner = CliRunner()
    res = runner.invoke(app, ["daemon", "--interval", bad_interval, "--once"])
    assert res.exit_code == 1, f"daemon with interval={bad_interval} must exit with code 1"
    assert "traceback" not in res.output.lower(), f"Raw traceback found: {res.output}"
    assert "Invalid interval" in res.output
    assert "Interval must be a positive finite number" in res.output


def test_cli_traceback_suppression_on_malformed_inputs():
    """All CLI subcommands must cleanly exit without raw Python tracebacks on corrupt inputs."""
    runner = CliRunner()
    fuzz_commands = [
        ["dry-run", "--scenario", "nonexistent_scenario_xyz"],
        ["dry-run", "--as-of", "not-a-valid-date"],
        ["dry-run", "--current-weights", "{malformed_json}"],
        ["dry-run", "--current-weights", "[1, 2, 3]"],
        ["rebalance", "--scenario", "nonexistent_scenario_xyz"],
        ["rebalance", "--current-weights", "not-json-at-all"],
        ["backtest", "--scenario", "invalid_scenario_123"],
        ["daemon", "--interval", "0"],
        ["daemon", "--interval", "-10s"],
        ["daemon", "--interval", "nan", "--once"],
        ["daemon", "--interval", "inf", "--once"],
        ["daemon", "--interval", "+inf", "--once"],
        ["daemon", "--interval", "not-an-interval"],
        ["export-metrics", "--start-date", "invalid-start-date"],
        ["export-metrics", "--end-date", "invalid-end-date"],
        ["export-metrics", "--output", "/nonexistent_parent_dir_xyz/out.json"],
    ]

    for cmd in fuzz_commands:
        res = runner.invoke(app, cmd)
        assert res.exit_code != 0, f"Command {cmd} unexpectedly succeeded: {res.output}"
        assert "traceback" not in res.output.lower(), f"Traceback leaked in command {cmd}: {res.output}"


# ============================================================================
# 2. REST Client Boundary & Protocol Fuzzing
# ============================================================================

@pytest.mark.asyncio
async def test_rest_client_500_storm_retry_and_exhaustion():
    """HTTP 500 error storms must trigger exponential backoff and raise RelayUpstreamError on exhaustion."""
    request_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(status_code=500, text="Internal Server Error: Database Connection Pool Exhausted")

    transport = httpx.MockTransport(mock_handler)
    mock_http_client = httpx.AsyncClient(transport=transport)

    async with AlpacaRelayRestClient(
        base_url="https://test-relay.mock",
        relay_token="test-token",
        max_retries=3,
        base_delay=0.005,
        max_delay=0.02,
        client=mock_http_client,
    ) as client:
        with pytest.raises(RelayUpstreamError) as exc_info:
            await client.get_bars("SPY")

        assert exc_info.value.status_code == 500
        assert "500" in str(exc_info.value)
        # Initial attempt + 3 retries = 4 total requests
        assert request_count == 4


@pytest.mark.asyncio
async def test_rest_client_500_transient_recovery():
    """Transient HTTP 500 blips must be retried and recover cleanly when upstream recovers."""
    attempt = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt <= 2:
            return httpx.Response(status_code=500, text="Temporary upstream glitch")
        # 3rd attempt succeeds
        payload = {
            "bars": [
                {
                    "t": "2026-09-03T12:00:00Z",
                    "o": 500.0,
                    "h": 505.0,
                    "l": 498.0,
                    "c": 502.5,
                    "v": 10000,
                    "vw": 501.5,
                    "n": 100,
                }
            ],
            "next_page_token": None,
        }
        return httpx.Response(status_code=200, json=payload)

    transport = httpx.MockTransport(mock_handler)
    mock_http_client = httpx.AsyncClient(transport=transport)

    async with AlpacaRelayRestClient(
        base_url="https://test-relay.mock",
        relay_token="test-token",
        max_retries=4,
        base_delay=0.005,
        max_delay=0.02,
        client=mock_http_client,
    ) as client:
        bars = await client.get_bars("SPY")
        assert len(bars) == 1
        assert bars[0].symbol == "SPY"
        assert bars[0].close == 502.5
        assert attempt == 3


@pytest.mark.asyncio
async def test_rest_client_429_rate_limit_flood():
    """HTTP 429 rate limit floods must trigger backoff and raise RelayRateLimitError on exhaustion."""
    request_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            status_code=429,
            headers={"Retry-After": "0.01"},
            text="Too Many Requests: Fleet quota depleted",
        )

    transport = httpx.MockTransport(mock_handler)
    mock_http_client = httpx.AsyncClient(transport=transport)

    async with AlpacaRelayRestClient(
        base_url="https://test-relay.mock",
        relay_token="test-token",
        max_retries=3,
        base_delay=0.005,
        max_delay=0.02,
        client=mock_http_client,
    ) as client:
        with pytest.raises(RelayRateLimitError) as exc_info:
            await client.get_bars("SPY")

        assert exc_info.value.response.status_code == 429
        assert "429" in str(exc_info.value)
        assert request_count == 4


@pytest.mark.asyncio
async def test_rest_client_network_transport_drop():
    """Simulated abrupt network drop / connection failure must raise RelayUpstreamError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection reset by peer during network drop", request=request)

    transport = httpx.MockTransport(mock_handler)
    mock_http_client = httpx.AsyncClient(transport=transport)

    async with AlpacaRelayRestClient(
        base_url="https://test-relay.mock",
        relay_token="test-token",
        max_retries=2,
        base_delay=0.005,
        max_delay=0.02,
        client=mock_http_client,
    ) as client:
        with pytest.raises(RelayUpstreamError) as exc_info:
            await client.get_bars("SPY")

        assert exc_info.value.status_code == 502
        assert "Transport error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_rest_client_symbol_validation_fuzzing():
    """REST client must reject None, non-string, empty strings, and malformed symbol collections with ValueError."""
    client = AlpacaRelayRestClient()
    bad_inputs = [
        None,
        "",
        "   ",
        12345,
        99.9,
        object(),
        ["SPY", None],
        ["SPY", ""],
        ["SPY", 456],
        [123, 456],
    ]

    for bad in bad_inputs:
        # Test get_bars rejects invalid symbols
        with pytest.raises(ValueError):
            await client.get_bars(bad)

        # Test get_multi_bars rejects non-string elements / invalid collections
        if not isinstance(bad, str):
            with pytest.raises(ValueError):
                await client.get_multi_bars(bad)

            with pytest.raises(ValueError):
                await client.get_historical_bars(bad)

    # Empty list in get_bars must also raise ValueError
    with pytest.raises(ValueError):
        await client.get_bars([])


# ============================================================================
# 3. WebSocket Streaming Client Protocol Fuzzing
# ============================================================================

@pytest.mark.asyncio
async def test_ws_client_crossed_book_quote_dropped_metric_accounting():
    """Crossed-book quotes (P_ask < P_bid) must be dropped, incrementing messages_dropped without incrementing quotes_processed."""
    client = AlpacaRelayWSClient(url="ws://mock", token="test-token")
    client._running = True

    # Start consumer loop task
    consumer_task = asyncio.create_task(client._consumer_loop())

    try:
        # Invariant: ask_price must be >= bid_price. Here ask 95 < bid 105 (crossed book)
        crossed_quote = {
            "T": "q",
            "S": "SPY",
            "bp": 105.0,
            "ap": 95.0,
            "bs": 100,
            "as": 50,
            "t": "2026-09-03T12:00:00Z",
        }

        await client._queue.put(crossed_quote)
        # Allow consumer loop to process the frame
        await asyncio.sleep(0.05)

        assert client.metrics["quotes_processed"] == 0, "Crossed quote must NOT increment quotes_processed"
        assert client.metrics["messages_dropped"] == 1, "Crossed quote must increment messages_dropped"

    finally:
        client._running = False
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_ws_client_corrupted_frames_dropped_metric_accounting():
    """Corrupted frames (e.g. high < low, negative trade price, missing fields) must increment messages_dropped."""
    client = AlpacaRelayWSClient(url="ws://mock", token="test-token")
    client._running = True

    consumer_task = asyncio.create_task(client._consumer_loop())

    try:
        corrupted_frames = [
            # Corrupted Bar: high (490.0) < low (510.0)
            {
                "T": "b",
                "S": "SPY",
                "o": 500.0,
                "h": 490.0,
                "l": 510.0,
                "c": 500.0,
                "v": 100,
                "t": "2026-09-03T12:00:00Z",
            },
            # Corrupted Trade: negative price and negative size
            {
                "T": "t",
                "S": "SPY",
                "p": -50.0,
                "s": -10,
                "i": "999",
                "t": "2026-09-03T12:00:00Z",
            },
            # Malformed payload missing required fields
            {
                "T": "q",
                "S": "SPY",
            },
        ]

        for frame in corrupted_frames:
            await client._queue.put(frame)

        await asyncio.sleep(0.05)

        assert client.metrics["bars_processed"] == 0
        assert client.metrics["trades_processed"] == 0
        assert client.metrics["quotes_processed"] == 0
        assert client.metrics["messages_dropped"] == 3

    finally:
        client._running = False
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_ws_client_valid_frames_increment_processed_counters():
    """Valid frames must increment their respective processed counters and leave messages_dropped at 0."""
    client = AlpacaRelayWSClient(url="ws://mock", token="test-token")
    client._running = True

    received_events = []
    client.on_any(lambda e: received_events.append(e))

    consumer_task = asyncio.create_task(client._consumer_loop())

    try:
        valid_frames = [
            # Valid Quote
            {
                "T": "q",
                "S": "SPY",
                "bp": 500.10,
                "ap": 500.20,
                "bs": 10,
                "as": 20,
                "t": "2026-09-03T12:00:00Z",
            },
            # Valid Bar
            {
                "T": "b",
                "S": "SPY",
                "o": 500.0,
                "h": 505.0,
                "l": 498.0,
                "c": 502.5,
                "v": 1000,
                "vw": 501.0,
                "n": 50,
                "t": "2026-09-03T12:00:00Z",
            },
            # Valid Trade
            {
                "T": "t",
                "S": "SPY",
                "p": 500.15,
                "s": 100,
                "i": "12345",
                "t": "2026-09-03T12:00:00Z",
            },
        ]

        for frame in valid_frames:
            await client._queue.put(frame)

        await asyncio.sleep(0.05)

        assert client.metrics["quotes_processed"] == 1
        assert client.metrics["bars_processed"] == 1
        assert client.metrics["trades_processed"] == 1
        assert client.metrics["messages_dropped"] == 0
        assert len(received_events) == 3

    finally:
        client._running = False
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_ws_client_producer_loop_binary_and_malformed_json_dropped_telemetry():
    """Producer loop must decode binary frames safely and increment messages_dropped on JSON/Unicode decode errors."""
    client = AlpacaRelayWSClient(url="ws://mock", token="test-token")
    client._running = True

    class FakeWebSocket:
        def __init__(self, frames):
            self.frames = list(frames)

        async def recv(self):
            if self.frames:
                return self.frames.pop(0)
            # Sleep until cancelled to simulate open connection
            await asyncio.sleep(10.0)

    # 1. Invalid binary non-UTF8 bytes
    # 2. Malformed JSON text
    # 3. Valid JSON text
    fake_frames = [
        bytes([0x80, 0x81, 0xff, 0xfe, 0xaa, 0xbb]),
        '{"T": "q", "unclosed_json',
        '[{"T": "q", "S": "SPY", "bp": 500.0, "ap": 501.0, "bs": 10, "as": 10, "t": "2026-09-03T12:00:00Z"}]',
    ]

    client._ws = FakeWebSocket(fake_frames)
    producer_task = asyncio.create_task(client._producer_loop())

    try:
        await asyncio.sleep(0.1)
        assert client.metrics["messages_dropped"] == 2, f"Expected 2 dropped, got {client.metrics['messages_dropped']}"
        assert client.metrics["messages_received"] == 1, f"Expected 1 received, got {client.metrics['messages_received']}"
        assert client._queue.qsize() == 1
        assert not producer_task.done(), "Producer task should remain alive"
    finally:
        client._running = False
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass

