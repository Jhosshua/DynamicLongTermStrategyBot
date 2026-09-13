"""
tests.integration.test_lifecycle_resilience
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration tests for IngestionStateMachine and AlpacaRelayClient resiliency.
Validates:
- Initial GET /health check handling (connected vs down)
- Zero-exception safe transition to STALE_DATA_HOLD on upstream_disconnected
- Strict allocation freeze (is_safe_to_rebalance is False, regime override)
- Automated REST gap backfill on upstream_connected when gap > 60s
- Watchdog timeout detecting dead/silent streams (> 300s)
- Flapping connection resilience across multiple rapid cycles
- Full end-to-end lifecycle coordination via unified AlpacaRelayClient
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from strategy_engine.core.models import Bar, MarketRegime
from strategy_engine.ingestion.client import AlpacaRelayClient
from strategy_engine.ingestion.state_machine import (
    IngestionStateMachine,
    LifecycleState,
    StateTransitionRecord,
)


def _make_bar(symbol: str, timestamp: datetime, price: float = 500.0) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=timestamp,
        open=price,
        high=price + 1.0,
        low=price - 1.0,
        close=price,
        volume=1000,
    )


@pytest.mark.asyncio
async def test_lifecycle_initial_health_check_connected():
    sm = IngestionStateMachine()
    assert sm.current_state == LifecycleState.INITIALIZING

    state = await sm.check_initial_health({"upstream": "connected"})
    assert state == LifecycleState.MONITORING_STREAM
    assert sm.current_state == LifecycleState.MONITORING_STREAM
    assert sm.is_upstream_connected is True
    assert sm.is_safe_to_rebalance() is True
    assert sm.get_market_regime_override() is None


@pytest.mark.asyncio
async def test_lifecycle_initial_health_check_down():
    sm = IngestionStateMachine()
    state = await sm.check_initial_health({"upstream": "down"})
    assert state == LifecycleState.STALE_DATA_HOLD
    assert sm.current_state == LifecycleState.STALE_DATA_HOLD
    assert sm.is_upstream_connected is False
    assert sm.is_safe_to_rebalance() is False
    assert sm.get_market_regime_override() == MarketRegime.STALE_DATA_HOLD
    assert sm.disconnect_timestamp is not None


@pytest.mark.asyncio
async def test_lifecycle_upstream_disconnected_to_stale_hold():
    sm = IngestionStateMachine()
    await sm.check_initial_health({"upstream": "connected"})
    assert sm.is_safe_to_rebalance() is True

    # Upstream disconnected event
    await sm.handle_upstream_disconnected("relayed upstream_disconnected frame")

    assert sm.current_state == LifecycleState.STALE_DATA_HOLD
    assert sm.is_upstream_connected is False
    assert sm.is_safe_to_rebalance() is False
    assert sm.get_market_regime_override() == MarketRegime.STALE_DATA_HOLD
    assert sm.disconnect_timestamp is not None


@pytest.mark.asyncio
async def test_lifecycle_zero_exceptions_on_callback_failure():
    sm = IngestionStateMachine()

    # Register buggy callback that raises exception
    def _bad_cb(prev, next_st, reason):
        raise RuntimeError("Callback crash simulation")

    sm.register_state_change_callback(_bad_cb)

    # Must complete with zero unhandled exceptions
    await sm.handle_upstream_disconnected("test disconnect")
    assert sm.current_state == LifecycleState.STALE_DATA_HOLD

    await sm.handle_upstream_connected("test connect")
    assert sm.current_state == LifecycleState.MONITORING_STREAM


@pytest.mark.asyncio
async def test_lifecycle_gap_backfill_on_reconnect():
    backfill_called = False
    gap_start = None
    gap_end = None

    async def _mock_backfill(start: datetime, end: datetime) -> int:
        nonlocal backfill_called, gap_start, gap_end
        backfill_called = True
        gap_start = start
        gap_end = end
        return 15

    sm = IngestionStateMachine(gap_backfill_callback=_mock_backfill)
    await sm.check_initial_health({"upstream": "connected"})

    # Record bar 120 seconds in past
    t_old = datetime.now(timezone.utc) - timedelta(seconds=120)
    sm.record_bar(_make_bar("SPY", t_old))

    # Disconnect
    await sm.handle_upstream_disconnected()
    assert sm.current_state == LifecycleState.STALE_DATA_HOLD

    # Reconnect
    await sm.handle_upstream_connected()

    assert backfill_called is True
    assert gap_start == t_old
    assert gap_end is not None
    assert sm.current_state == LifecycleState.MONITORING_STREAM
    assert sm.is_safe_to_rebalance() is True


@pytest.mark.asyncio
async def test_lifecycle_short_gap_skips_backfill():
    backfill_called = False

    async def _mock_backfill(start: datetime, end: datetime) -> int:
        nonlocal backfill_called
        backfill_called = True
        return 0

    sm = IngestionStateMachine(gap_backfill_callback=_mock_backfill)
    await sm.check_initial_health({"upstream": "connected"})

    # Record bar 20s in past (<= 60s)
    t_recent = datetime.now(timezone.utc) - timedelta(seconds=20)
    sm.record_bar(_make_bar("SPY", t_recent))

    await sm.handle_upstream_disconnected()
    await sm.handle_upstream_connected()

    # Backfill skipped for short gap
    assert backfill_called is False
    assert sm.current_state == LifecycleState.MONITORING_STREAM


@pytest.mark.asyncio
async def test_lifecycle_watchdog_timeout():
    sm = IngestionStateMachine(stale_timeout_seconds=300.0)
    await sm.check_initial_health({"upstream": "connected"})

    # Message received now -> watchdog returns False
    now = datetime.now(timezone.utc)
    sm.record_message(now)
    stale = await sm.check_watchdog(now + timedelta(seconds=100))
    assert stale is False
    assert sm.current_state == LifecycleState.MONITORING_STREAM

    # Check at now + 350s -> watchdog triggers STALE_DATA_HOLD
    stale_triggered = await sm.check_watchdog(now + timedelta(seconds=350))
    assert stale_triggered is True
    assert sm.current_state == LifecycleState.STALE_DATA_HOLD
    assert sm.is_safe_to_rebalance() is False


@pytest.mark.asyncio
async def test_lifecycle_flapping_connection_cycles():
    sm = IngestionStateMachine()
    await sm.check_initial_health({"upstream": "connected"})

    for i in range(5):
        await sm.handle_upstream_disconnected(f"cycle {i} disconnect")
        assert sm.current_state == LifecycleState.STALE_DATA_HOLD
        await sm.handle_upstream_connected(f"cycle {i} reconnect")
        assert sm.current_state == LifecycleState.MONITORING_STREAM

    # 1 initial + 10 flapping transitions = 11 total
    assert len(sm.transition_history) == 11
    diagnostics = sm.get_status_diagnostics()
    assert diagnostics["total_transitions"] == 11
    assert diagnostics["state"] == "MONITORING_STREAM"


@pytest.mark.asyncio
async def test_lifecycle_unified_client_end_to_end(mock_relay):
    # Setup mock bars for backfill
    mock_relay.add_mock_bars("SPY", [{
        "t": "2026-09-03T12:00:00Z",
        "o": 500.0, "h": 501.0, "l": 499.0, "c": 500.5, "v": 100,
    }])

    client = AlpacaRelayClient(
        base_url=mock_relay.http_url,
        ws_url=mock_relay.ws_url,
        relay_token="secret-relay-token-123",
    )

    try:
        # 1. Connect stream
        await client.connect_stream(symbols=["SPY"], channels=["bars"])
        assert client.get_connection_status() == "MONITORING_STREAM"
        assert client.is_safe_to_rebalance() is True

        # 2. Simulate upstream disconnect broadcast
        await mock_relay.broadcast_lifecycle("upstream_disconnected")
        await asyncio.sleep(0.1)

        assert client.get_connection_status() == "STALE_DATA_HOLD"
        assert client.is_safe_to_rebalance() is False
        assert client.get_market_regime_override() == MarketRegime.STALE_DATA_HOLD

        # 3. Simulate upstream reconnect broadcast
        await mock_relay.broadcast_lifecycle("upstream_connected")
        await asyncio.sleep(0.1)

        assert client.get_connection_status() == "MONITORING_STREAM"
        assert client.is_safe_to_rebalance() is True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_lifecycle_unified_client_methods(mock_relay):
    mock_relay.add_mock_bars("SPY", [{
        "t": "2026-09-03T12:00:00Z",
        "o": 500.0, "h": 501.0, "l": 499.0, "c": 500.5, "v": 100,
    }])
    client = AlpacaRelayClient(
        base_url=mock_relay.http_url,
        ws_url=mock_relay.ws_url,
        relay_token="secret-relay-token-123",
    )
    try:
        # Protocol historical bars
        bars = await client.get_historical_bars(["SPY"], timeframe="1Day", start=None, end=None)
        assert len(bars["SPY"]) == 1

        # Unknown lifecycle event
        await client.handle_lifecycle_event("unknown_event_xyz")

        # Bar callback wiring
        b = _make_bar("SPY", datetime.now(timezone.utc))
        client._on_ws_bar(b)
        assert client.state_machine.last_bar_timestamp == b.timestamp
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_lifecycle_state_machine_edge_cases():
    sm = IngestionStateMachine()
    # In INITIALIZING, watchdog returns False
    assert await sm.check_watchdog() is False

    # In MONITORING_STREAM without any messages, returns False
    await sm.check_initial_health({"upstream": "connected"})
    assert await sm.check_watchdog() is False

    # record_message without timestamp uses now
    sm.record_message()
    assert sm.last_msg_timestamp is not None

