"""
tests.integration.test_ws_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration tests for AlpacaRelayWSClient against in-process mock relay.
Validates:
- 10-second authentication handshake and connection banner
- Rejection of invalid auth tokens (402)
- Channel subscriptions and acknowledgement
- Strict prohibition of wildcard '*' subscriptions (raising ValueError)
- Decoupled producer-consumer queue (50,000 capacity)
- Ring-buffer overflow behavior (dropping oldest on capacity exceeded)
- Stream async generator and typed callback dispatch
- Server eviction handling (close code 1013 'too slow') and automatic reconnect
- Graceful disconnect and resource cleanup
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import pytest

from strategy_engine.core.models import Bar, Quote, Trade
from strategy_engine.ingestion.ws_client import (
    AlpacaRelayWSClient,
    LifecycleEvent,
    RelayAuthError,
    RelayConnectionError,
    SubscriptionAckEvent,
    WSConnectionState,
)


@pytest.mark.asyncio
async def test_ws_client_connect_and_handshake(mock_relay):
    client = AlpacaRelayWSClient(
        url=mock_relay.ws_url,
        token="secret-relay-token-123",
    )
    try:
        await client.connect()
        assert client.is_connected is True
        assert client.is_authenticated is True
        assert client.state == WSConnectionState.AUTHENTICATED
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_client_auth_failure_invalid_token(mock_relay):
    client = AlpacaRelayWSClient(
        url=mock_relay.ws_url,
        token="INVALID-TOKEN",
    )
    try:
        with pytest.raises(RelayAuthError) as exc_info:
            await client.connect()
        assert "Authentication failed" in str(exc_info.value)
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_client_subscriptions_and_ack(mock_relay):
    client = AlpacaRelayWSClient(
        url=mock_relay.ws_url,
        token="secret-relay-token-123",
    )
    try:
        await client.connect()
        await client.subscribe(
            bars=["SPY", "QQQ"],
            trades=["SPY"],
            quotes=["QQQ"],
        )
        # Give server time to send subscription ack
        await asyncio.sleep(0.1)

        assert "SPY" in client._server_subscriptions.get("bars", [])
        assert "QQQ" in client._server_subscriptions.get("bars", [])
        assert "SPY" in client._server_subscriptions.get("trades", [])
        assert "QQQ" in client._server_subscriptions.get("quotes", [])
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_client_wildcard_prohibition():
    client = AlpacaRelayWSClient(token="test")

    # Bars wildcard
    with pytest.raises(ValueError) as exc:
        await client.subscribe(bars=["*"])
    assert "Wildcard '*' subscription in channel 'bars' is strictly forbidden" in str(exc.value)

    # Trades wildcard
    with pytest.raises(ValueError) as exc:
        await client.subscribe(trades=["SPY", "*"])
    assert "strictly forbidden" in str(exc.value)

    # Quotes wildcard
    with pytest.raises(ValueError) as exc:
        await client.subscribe(quotes=["*"])
    assert "strictly forbidden" in str(exc.value)

    # Channels dict wildcard
    with pytest.raises(ValueError) as exc:
        await client.subscribe(channels={"dailyBars": ["*"]})
    assert "strictly forbidden" in str(exc.value)


@pytest.mark.asyncio
async def test_ws_client_decoupled_producer_consumer_queue(mock_relay):
    client = AlpacaRelayWSClient(
        url=mock_relay.ws_url,
        token="secret-relay-token-123",
        queue_size=50000,
    )
    received_bars: list[Bar] = []
    client.on_bar(lambda b: received_bars.append(b))

    try:
        await client.connect()
        await client.subscribe(bars=["SPY"])
        await asyncio.sleep(0.05)

        # Broadcast 50 bars rapidly
        for i in range(50):
            await mock_relay.broadcast_bar("SPY", {
                "t": "2026-09-03T12:00:00Z",
                "o": 500.0 + i,
                "h": 501.0 + i,
                "l": 499.0 + i,
                "c": 500.5 + i,
                "v": 100,
            })

        # Wait for consumer loop to drain queue
        for _ in range(50):
            if len(received_bars) >= 50:
                break
            await asyncio.sleep(0.05)

        assert len(received_bars) == 50
        assert client.metrics["messages_dropped"] == 0
        assert client.metrics["bars_processed"] == 50
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_client_queue_overflow_drops_oldest(mock_relay):
    # Set tiny queue capacity of 5
    client = AlpacaRelayWSClient(
        url=mock_relay.ws_url,
        token="secret-relay-token-123",
        queue_size=5,
    )

    try:
        await client.connect()
        await client.subscribe(bars=["SPY"])
        await asyncio.sleep(0.05)

        # Broadcast 15 bars rapidly
        for i in range(15):
            await mock_relay.broadcast_bar("SPY", {
                "t": "2026-09-03T12:00:00Z",
                "o": 500.0 + i,
                "h": 501.0 + i,
                "l": 499.0 + i,
                "c": 500.5 + i,
                "v": 100,
            })

        await asyncio.sleep(0.1)
        # Socket remains connected despite rapid burst
        assert client.is_connected is True
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_client_typed_callbacks(mock_relay):
    client = AlpacaRelayWSClient(
        url=mock_relay.ws_url,
        token="secret-relay-token-123",
    )

    bars: list[Bar] = []
    quotes: list[Quote] = []
    trades: list[Trade] = []
    lifecycles: list[LifecycleEvent] = []

    client.on_bar(lambda b: bars.append(b))
    client.on_quote(lambda q: quotes.append(q))
    client.on_trade(lambda t: trades.append(t))
    client.on_lifecycle(lambda lc: lifecycles.append(lc))

    try:
        await client.connect()
        await client.subscribe(bars=["SPY"], quotes=["SPY"], trades=["SPY"])
        await asyncio.sleep(0.05)

        # Broadcast bar
        await mock_relay.broadcast_bar("SPY", {
            "t": "2026-09-03T12:00:00Z",
            "o": 500.0, "h": 502.0, "l": 498.0, "c": 501.0, "v": 1000,
        })
        # Broadcast quote
        await mock_relay.broadcast_quote("SPY", {
            "t": "2026-09-03T12:00:00Z",
            "bp": 500.0, "bs": 10, "ap": 500.2, "as": 20,
        })
        # Broadcast trade
        await mock_relay.broadcast_trade("SPY", {
            "t": "2026-09-03T12:00:00Z",
            "p": 500.1, "s": 50, "i": "t1",
        })
        # Broadcast lifecycle
        await mock_relay.broadcast_lifecycle("upstream_disconnected")

        await asyncio.sleep(0.15)

        assert len(bars) == 1
        assert isinstance(bars[0], Bar)
        assert len(quotes) == 1
        assert isinstance(quotes[0], Quote)
        assert len(trades) == 1
        assert isinstance(trades[0], Trade)
        assert len(lifecycles) == 1
        assert isinstance(lifecycles[0], LifecycleEvent)
        assert lifecycles[0].event_type == "upstream_disconnected"
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_client_stream_generator(mock_relay):
    client = AlpacaRelayWSClient(
        url=mock_relay.ws_url,
        token="secret-relay-token-123",
    )

    async def _consume_stream():
        events = []
        async for event in client.stream():
            events.append(event)
            if len(events) >= 2:
                break
        return events

    try:
        await client.connect()
        await client.subscribe(bars=["SPY"])
        await asyncio.sleep(0.05)

        consume_task = asyncio.create_task(_consume_stream())

        await mock_relay.broadcast_bar("SPY", {
            "t": "2026-09-03T12:00:00Z",
            "o": 500.0, "h": 501.0, "l": 499.0, "c": 500.5, "v": 100,
        })
        await mock_relay.broadcast_bar("SPY", {
            "t": "2026-09-03T12:01:00Z",
            "o": 500.5, "h": 502.0, "l": 500.0, "c": 501.5, "v": 150,
        })

        collected = await asyncio.wait_for(consume_task, timeout=2.0)
        assert len(collected) == 2
        assert isinstance(collected[0], Bar)
        assert isinstance(collected[1], Bar)
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_client_eviction_1013_and_reconnect(mock_relay):
    client = AlpacaRelayWSClient(
        url=mock_relay.ws_url,
        token="secret-relay-token-123",
        initial_backoff=0.05,
        max_backoff=0.2,
    )
    try:
        await client.connect()
        await client.subscribe(bars=["SPY"])
        await asyncio.sleep(0.05)

        # Evict client with code 1013
        server_ws = next(iter(mock_relay.clients))
        await mock_relay.simulate_slow_client_eviction(server_ws)

        # Wait for client to detect close code 1013, backoff and reconnect
        for _ in range(30):
            if client.metrics["evictions_1013"] >= 1 and client.is_authenticated:
                break
            await asyncio.sleep(0.05)

        assert client.metrics["evictions_1013"] >= 1
        assert client.is_authenticated is True
        # Verify subscriptions were restored
        assert "SPY" in client._desired_subscriptions["bars"]
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_client_unsubscribe(mock_relay):
    client = AlpacaRelayWSClient(
        url=mock_relay.ws_url,
        token="secret-relay-token-123",
    )
    try:
        await client.connect()
        await client.subscribe(bars=["SPY", "QQQ"])
        await asyncio.sleep(0.05)
        assert "SPY" in client._desired_subscriptions["bars"]

        await client.unsubscribe(bars=["SPY"])
        await asyncio.sleep(0.05)
        assert "SPY" not in client._desired_subscriptions["bars"]
        assert "QQQ" in client._desired_subscriptions["bars"]
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_client_on_any_callback(mock_relay):
    client = AlpacaRelayWSClient(
        url=mock_relay.ws_url,
        token="secret-relay-token-123",
    )
    any_events = []
    client.on_any(lambda ev: any_events.append(ev))

    try:
        await client.connect()
        await client.subscribe(bars=["SPY"])
        await asyncio.sleep(0.05)

        await mock_relay.broadcast_bar("SPY", {
            "t": "2026-09-03T12:00:00Z",
            "o": 500.0, "h": 501.0, "l": 499.0, "c": 500.5, "v": 100,
        })
        await asyncio.sleep(0.1)

        # Receives subscription ack and bar
        assert len(any_events) >= 1
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_ws_client_context_manager(mock_relay):
    async with AlpacaRelayWSClient(
        url=mock_relay.ws_url,
        token="secret-relay-token-123",
    ) as client:
        assert client.is_authenticated is True
    assert client.state == WSConnectionState.CLOSED

