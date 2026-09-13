"""
tests.integration.test_rest_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration tests for AlpacaRelayRestClient against the in-process mock relay.
Validates:
- Dual auth header injection (X-Relay-Token and APCA-API-KEY-ID)
- 401 rejection without retries
- Health check inspection
- Single-symbol and multi-symbol historical bars with domain model parsing
- Auto-pagination across next_page_token
- Latest bars, quotes, and trades endpoints
- Token bucket rate limiter pacing
- Exponential backoff with jitter on HTTP 429 and 502 with Retry-After handling
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
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


def _generate_sample_bars(symbol: str, count: int) -> list[dict]:
    """Generate valid mock Alpaca bar dictionaries."""
    bars = []
    base_time = 1725364800  # 2024-09-03 12:00:00 UTC
    for i in range(count):
        t_str = datetime.fromtimestamp(base_time + i * 60, tz=timezone.utc).isoformat()
        bars.append({
            "t": t_str,
            "o": 500.0 + i * 0.1,
            "h": 505.0 + i * 0.1,
            "l": 498.0 + i * 0.1,
            "c": 502.0 + i * 0.1,
            "v": 1000 + i * 10,
            "n": 50,
            "vw": 501.5 + i * 0.1,
        })
    return bars


@pytest.mark.asyncio
async def test_rest_client_auth_relay_header(mock_relay):
    mock_relay.add_mock_bars("SPY", _generate_sample_bars("SPY", 5))
    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
        auth_header_mode="relay",
    ) as client:
        bars = await client.get_bars("SPY")
        assert len(bars) == 5
        assert isinstance(bars[0], Bar)
        assert bars[0].symbol == "SPY"


@pytest.mark.asyncio
async def test_rest_client_auth_apca_header(mock_relay):
    mock_relay.add_mock_bars("SPY", _generate_sample_bars("SPY", 3))
    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
        auth_header_mode="apca",
    ) as client:
        bars = await client.get_bars("SPY")
        assert len(bars) == 3


@pytest.mark.asyncio
async def test_rest_client_auth_both_headers(mock_relay):
    mock_relay.add_mock_bars("QQQ", _generate_sample_bars("QQQ", 2))
    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
        auth_header_mode="both",
    ) as client:
        bars = await client.get_bars("QQQ")
        assert len(bars) == 2


@pytest.mark.asyncio
async def test_rest_client_invalid_token_raises_auth_error(mock_relay):
    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="WRONG-TOKEN",
    ) as client:
        with pytest.raises(RelayAuthError) as exc_info:
            await client.get_bars("SPY")
        assert "401" in str(exc_info.value)
        assert issubclass(RelayAuthError, RelayAuthenticationError)


@pytest.mark.asyncio
async def test_rest_client_health_check(mock_relay):
    async with AlpacaRelayRestClient(base_url=mock_relay.http_url) as client:
        health = await client.get_health()
        assert health["upstream"] == "connected"
        assert await client.is_upstream_connected() is True

        mock_relay.set_upstream_status(False)
        health_down = await client.get_health()
        assert health_down["upstream"] == "down"
        assert await client.is_upstream_connected() is False


@pytest.mark.asyncio
async def test_rest_client_multi_bars_and_protocol(mock_relay):
    mock_relay.add_mock_bars("SPY", _generate_sample_bars("SPY", 4))
    mock_relay.add_mock_bars("QQQ", _generate_sample_bars("QQQ", 6))

    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
    ) as client:
        # Multi bars
        res = await client.get_multi_bars(["SPY", "QQQ"])
        assert len(res["SPY"]) == 4
        assert len(res["QQQ"]) == 6
        assert isinstance(res["SPY"][0], Bar)

        # Protocol method single
        proto_single = await client.get_historical_bars(["SPY"], timeframe="1Day")
        assert len(proto_single["SPY"]) == 4

        # Protocol method multi
        proto_multi = await client.get_historical_bars(["SPY", "QQQ"], timeframe="1Day")
        assert len(proto_multi["SPY"]) == 4
        assert len(proto_multi["QQQ"]) == 6


@pytest.mark.asyncio
async def test_rest_client_latest_endpoints(mock_relay):
    mock_relay.add_mock_bars("SPY", _generate_sample_bars("SPY", 2))
    mock_relay.add_mock_quotes("SPY", [{
        "t": "2026-09-03T12:00:00Z",
        "bp": 500.5,
        "bs": 100,
        "ap": 500.6,
        "as": 200,
        "bx": "V",
        "ax": "V",
    }])
    mock_relay.add_mock_trades("SPY", [{
        "t": "2026-09-03T12:00:00Z",
        "p": 500.55,
        "s": 50,
        "i": "trade-999",
        "x": "V",
    }])

    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
    ) as client:
        latest_bar = await client.get_latest_bar("SPY")
        assert isinstance(latest_bar, Bar)
        assert latest_bar.symbol == "SPY"

        latest_quote = await client.get_latest_quote("SPY")
        assert isinstance(latest_quote, Quote)
        assert latest_quote.bid_price == 500.5
        assert latest_quote.ask_price == 500.6

        latest_trade = await client.get_latest_trade("SPY")
        assert isinstance(latest_trade, Trade)
        assert latest_trade.price == 500.55
        assert latest_trade.size == 50


@pytest.mark.asyncio
async def test_rest_client_auto_pagination(mock_relay):
    # Add 35 bars
    mock_relay.add_mock_bars("SPY", _generate_sample_bars("SPY", 35))

    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
    ) as client:
        # Request with limit=10, should fetch 4 pages (10, 10, 10, 5)
        bars = await client.get_bars("SPY", limit=10)
        assert len(bars) == 35

        # With max_pages=2
        bars_capped = await client.get_bars("SPY", limit=10, max_pages=2)
        assert len(bars_capped) == 20


@pytest.mark.asyncio
async def test_rest_client_rate_limiter_pacing(mock_relay):
    mock_relay.add_mock_bars("SPY", _generate_sample_bars("SPY", 1))

    # Configure 60 req/min (1 req/sec) with burst capacity 1.0
    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
        rate_limit_per_minute=60.0,
        burst_capacity=1.0,
    ) as client:
        t0 = time.monotonic()
        # Make 3 requests: 1st immediate (tokens=1), 2nd waits 1s, 3rd waits 1s
        await client.get_bars("SPY")
        await client.get_bars("SPY")
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.8  # Paced by rate limiter


@pytest.mark.asyncio
async def test_rest_client_429_retry_and_recovery(mock_relay):
    mock_relay.add_mock_bars("SPY", _generate_sample_bars("SPY", 2))
    # Inject 429 for 2 requests with Retry-After 0.05
    mock_relay.set_simulate_status(429, count=2, retry_after="0.05")

    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
        base_delay=0.05,
        max_delay=0.5,
    ) as client:
        bars = await client.get_bars("SPY")
        assert len(bars) == 2


@pytest.mark.asyncio
async def test_rest_client_429_exhaustion_raises(mock_relay):
    mock_relay.set_simulate_status(429, count=10)

    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
        max_retries=2,
        base_delay=0.01,
        max_delay=0.05,
    ) as client:
        with pytest.raises(RelayRateLimitError):
            await client.get_bars("SPY")


@pytest.mark.asyncio
async def test_rest_client_502_upstream_retry_and_exhaustion(mock_relay):
    mock_relay.add_mock_bars("SPY", _generate_sample_bars("SPY", 1))

    # Recovery after 1 failure
    mock_relay.set_simulate_status(502, count=1)
    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
        base_delay=0.02,
    ) as client:
        bars = await client.get_bars("SPY")
        assert len(bars) == 1

    # Exhaustion
    mock_relay.set_simulate_status(502, count=10)
    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
        max_retries=2,
        base_delay=0.01,
        max_delay=0.05,
    ) as client:
        with pytest.raises(RelayUpstreamError):
            await client.get_bars("SPY")


@pytest.mark.asyncio
async def test_token_bucket_limiter_unit():
    limiter = TokenBucketRateLimiter(rate_limit_per_minute=60.0, burst_capacity=2.0)
    assert limiter.available_tokens <= 2.0
    assert limiter.try_acquire(1.0) is True
    assert limiter.try_acquire(1.0) is True
    # Now empty
    assert limiter.try_acquire(1.0) is False

    # Invalid params
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate_limit_per_minute=0)
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(burst_capacity=0.5)


@pytest.mark.asyncio
async def test_rest_client_empty_symbols_latest(mock_relay):
    async with AlpacaRelayRestClient(base_url=mock_relay.http_url) as client:
        assert await client.get_latest_bars([]) == {}
        assert await client.get_latest_quotes([]) == {}
        assert await client.get_latest_trades([]) == {}


@pytest.mark.asyncio
async def test_rest_client_iter_bars_params(mock_relay):
    mock_relay.add_mock_bars("SPY", _generate_sample_bars("SPY", 5))
    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
    ) as client:
        t_start = datetime(2024, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
        t_end = datetime(2024, 9, 3, 13, 0, 0, tzinfo=timezone.utc)
        bars = []
        async for b in client.iter_bars("SPY", start=t_start, end=t_end, asof="2026-09-03"):
            bars.append(b)
        assert len(bars) == 5


@pytest.mark.asyncio
async def test_rest_client_400_client_error_raises(mock_relay):
    async with AlpacaRelayRestClient(
        base_url=mock_relay.http_url,
        relay_token="secret-relay-token-123",
    ) as client:
        # Mock relay rejects POST/PUT on /data with 400
        with pytest.raises(RelayRequestError) as exc_info:
            await client._request("POST", "v2/stocks/bars")
        assert exc_info.value.status_code == 400

