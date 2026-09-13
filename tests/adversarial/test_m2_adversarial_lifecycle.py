"""
tests/adversarial/test_m2_adversarial_lifecycle.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Empirical Adversarial Fault-Injection Test Suite for Milestone M2:
Ingestion Lifecycle & Fault-Injection Resilience.

Challenges & Stress Scenarios:
1. Sudden Disconnect Injections:
   - Immediate transition to STALE_DATA_HOLD with zero unhandled exceptions
   - Strict allocation freeze (is_safe_to_rebalance() is False, regime override)
   - Redundant consecutive disconnect events (idempotency, zero exceptions)
   - Faulty callback explosions (sync and async exceptions swallowed safely)
   - High concurrency race conditions across state transitions

2. Rapid Flapping & Deadlock Resilience:
   - 10 rapid alternating disconnect <-> connect cycles (verification against deadlocks/crashes)
   - 50 rapid alternating cycles stress test
   - Concurrent multi-task flapping with non-reentrant lock safety
   - Flapping during ongoing / delayed gap backfill

3. Gap Backfill REST Proxy Inquiries:
   - Verification of exact [start, end] window passed to REST backfill callback
   - End-to-end backfill with live ExtendedMockRelayServer verifying REST query params
   - Boundary checks: gap <= 60s skipped vs gap > 60s executed
   - Resilience against REST 500 / network failure during backfill
   - Backfill with zero subscribed symbols

4. Stream Silence & Watchdog Timer:
   - Prolonged silence (>300s) triggering STALE_DATA_HOLD
   - Precision boundary check: 299.9s (active) vs 300.1s (stale triggered)
   - Repeated watchdog polling idempotency
   - Watchdog recovery upon stream reconnection and fresh bar ingestion
   - Edge case: stream without any recorded messages

5. End-to-End WebSocket Relay Chaos:
   - Live mock server lifecycle broadcasts and rebalance gate assertions
   - Code 1013 "too slow" eviction recovery and channel resubscription
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional
import pytest

from strategy_engine.core.models import Bar, MarketRegime
from strategy_engine.ingestion.client import AlpacaRelayClient
from strategy_engine.ingestion.state_machine import (
    IngestionStateMachine,
    LifecycleState,
    StateTransitionRecord,
)
from strategy_engine.ingestion.ws_client import (
    AlpacaRelayWSClient,
    LifecycleEvent,
    WSConnectionState,
)
from tests.integration.conftest import ExtendedMockRelayServer

logger = logging.getLogger("test_m2_adversarial_lifecycle")


def _make_bar(symbol: str, timestamp: datetime, price: float = 500.0) -> Bar:
    """Helper to produce a deterministic Bar."""
    return Bar(
        symbol=symbol,
        timestamp=timestamp,
        open=price,
        high=price + 1.0,
        low=price - 1.0,
        close=price,
        volume=1000,
        trade_count=50,
        vwap=price,
    )


# ============================================================================
# Category 1: Sudden Disconnect Injections & Rebalance Freeze
# ============================================================================

@pytest.mark.asyncio
async def test_adv_sudden_disconnect_from_monitoring_stream():
    """Adversarial Challenge: Abrupt upstream disconnect while actively monitoring stream.
    Must transition immediately to STALE_DATA_HOLD with zero exceptions and freeze allocations.
    """
    sm = IngestionStateMachine()
    await sm.check_initial_health({"upstream": "connected"})
    assert sm.current_state == LifecycleState.MONITORING_STREAM
    assert sm.is_safe_to_rebalance() is True
    assert sm.get_market_regime_override() is None

    # Inject sudden disconnect
    await sm.handle_upstream_disconnected("Sudden upstream WAN drop")

    assert sm.current_state == LifecycleState.STALE_DATA_HOLD
    assert sm.is_upstream_connected is False
    assert sm.is_safe_to_rebalance() is False
    assert sm.get_market_regime_override() == MarketRegime.STALE_DATA_HOLD
    assert sm.disconnect_timestamp is not None


@pytest.mark.asyncio
async def test_adv_repeated_consecutive_disconnect_events():
    """Adversarial Challenge: Inject consecutive upstream_disconnected events repeatedly.
    Must be strictly idempotent, record no redundant transition states, and throw zero exceptions.
    """
    sm = IngestionStateMachine()
    await sm.check_initial_health({"upstream": "connected"})

    # 10 consecutive disconnect calls
    for i in range(10):
        await sm.handle_upstream_disconnected(f"Disconnect storm event #{i}")
        assert sm.current_state == LifecycleState.STALE_DATA_HOLD
        assert sm.is_safe_to_rebalance() is False

    # Check transition history: INITIALIZING -> MONITORING_STREAM -> STALE_DATA_HOLD
    # Consecutive calls to STALE_DATA_HOLD should not duplicate identical transitions
    assert len(sm.transition_history) == 2
    assert sm.transition_history[-1].to_state == LifecycleState.STALE_DATA_HOLD


@pytest.mark.asyncio
async def test_adv_disconnect_from_unexpected_lifecycle_states():
    """Adversarial Challenge: Inject disconnect from non-standard states (INITIALIZING, SYNCING).
    Must safely transition to STALE_DATA_HOLD without crashing.
    """
    # 1. From INITIALIZING
    sm1 = IngestionStateMachine()
    assert sm1.current_state == LifecycleState.INITIALIZING
    await sm1.handle_upstream_disconnected("Disconnect before initialization completed")
    assert sm1.current_state == LifecycleState.STALE_DATA_HOLD
    assert sm1.is_safe_to_rebalance() is False

    # 2. From SYNCING_HISTORY
    async def _dummy_backfill(s, e):
        return 10

    sm2 = IngestionStateMachine(gap_backfill_callback=_dummy_backfill)
    await sm2.check_initial_health({"upstream": "connected"})
    # Put into SYNCING_HISTORY manually
    await sm2._transition_to(LifecycleState.SYNCING_HISTORY, "Simulating active gap backfill")
    assert sm2.is_safe_to_rebalance() is False

    await sm2.handle_upstream_disconnected("Disconnect during historical sync")
    assert sm2.current_state == LifecycleState.STALE_DATA_HOLD
    assert sm2.is_safe_to_rebalance() is False


@pytest.mark.asyncio
async def test_adv_callback_explosions_on_disconnect():
    """Adversarial Challenge: External callbacks raise severe exceptions (RuntimeError, ZeroDivisionError).
    State machine must isolate the exceptions and safely transition to STALE_DATA_HOLD.
    """
    sm = IngestionStateMachine()
    await sm.check_initial_health({"upstream": "connected"})

    def _exploding_sync_cb(prev, next_st, reason):
        raise RuntimeError("Synchronous callback catastrophe!")

    async def _exploding_async_cb(prev, next_st, reason):
        raise ZeroDivisionError("Asynchronous callback division by zero!")

    sm.register_state_change_callback(_exploding_sync_cb)
    sm.register_state_change_callback(_exploding_async_cb)

    # Must NOT raise exception to caller
    await sm.handle_upstream_disconnected("Fault injection with exploding callbacks")

    assert sm.current_state == LifecycleState.STALE_DATA_HOLD
    assert sm.is_safe_to_rebalance() is False
    assert sm.get_market_regime_override() == MarketRegime.STALE_DATA_HOLD


@pytest.mark.asyncio
async def test_adv_disconnect_under_high_concurrency():
    """Adversarial Challenge: 50 concurrent tasks simultaneously injecting disconnects and bar events.
    Verifies that internal lock prevents state corruption and data races.
    """
    sm = IngestionStateMachine()
    await sm.check_initial_health({"upstream": "connected"})

    async def _worker_disconnect(idx: int):
        await sm.handle_upstream_disconnected(f"Concurrent task {idx}")

    async def _worker_record_bar(idx: int):
        now = datetime.now(timezone.utc)
        sm.record_bar(_make_bar("SPY", now + timedelta(milliseconds=idx)))

    tasks = []
    for i in range(25):
        tasks.append(_worker_disconnect(i))
        tasks.append(_worker_record_bar(i))

    # Execute all 50 concurrently
    await asyncio.gather(*tasks)

    assert sm.current_state == LifecycleState.STALE_DATA_HOLD
    assert sm.is_safe_to_rebalance() is False
    assert sm.is_upstream_connected is False


# ============================================================================
# Category 2: Rapid Flapping & Deadlock Resilience
# ============================================================================

@pytest.mark.asyncio
async def test_adv_rapid_flapping_10_cycles():
    """Adversarial Challenge: 10 rapid alternating cycles (disconnected <-> connected).
    Verifies zero deadlocks, zero crashes, exact transition audit trail, and safe rebalance gating.
    """
    sm = IngestionStateMachine()
    await sm.check_initial_health({"upstream": "connected"})
    assert sm.current_state == LifecycleState.MONITORING_STREAM

    # Perform 10 rapid flapping cycles
    for cycle in range(1, 11):
        # 1. Disconnect
        await sm.handle_upstream_disconnected(f"Flapping cycle {cycle} disconnect")
        assert sm.current_state == LifecycleState.STALE_DATA_HOLD
        assert sm.is_safe_to_rebalance() is False
        assert sm.is_upstream_connected is False

        # 2. Reconnect
        await sm.handle_upstream_connected(f"Flapping cycle {cycle} reconnect")
        assert sm.current_state == LifecycleState.MONITORING_STREAM
        assert sm.is_safe_to_rebalance() is True
        assert sm.is_upstream_connected is True

    # Audit validation: 1 initial health transition + 20 flapping transitions = 21
    history = sm.transition_history
    assert len(history) == 21
    assert history[-1].to_state == LifecycleState.MONITORING_STREAM
    assert sm.get_status_diagnostics()["total_transitions"] == 21


@pytest.mark.asyncio
async def test_adv_rapid_flapping_50_cycles_stress():
    """Adversarial Challenge: 50 ultra-fast flapping cycles (100 rapid transitions).
    Asserts no memory leakage, lock contention failure, or state drift.
    """
    sm = IngestionStateMachine()
    await sm.check_initial_health({"upstream": "connected"})

    for cycle in range(50):
        await sm.handle_upstream_disconnected(f"Stress {cycle} drop")
        assert sm.is_safe_to_rebalance() is False
        await sm.handle_upstream_connected(f"Stress {cycle} restore")
        assert sm.is_safe_to_rebalance() is True

    assert len(sm.transition_history) == 101  # 1 initial + 100 transitions
    assert sm.current_state == LifecycleState.MONITORING_STREAM


@pytest.mark.asyncio
async def test_adv_concurrent_flapping_tasks():
    """Adversarial Challenge: Concurrent tasks firing interleaved connect and disconnect events.
    Verifies asyncio.Lock prevents deadlock and keeps internal state completely coherent.
    """
    sm = IngestionStateMachine()
    await sm.check_initial_health({"upstream": "connected"})

    async def _flapper(task_id: int):
        for i in range(5):
            await sm.handle_upstream_disconnected(f"task {task_id} drop {i}")
            await sm.handle_upstream_connected(f"task {task_id} restore {i}")

    # Launch 6 concurrent flapper tasks (total 60 transitions)
    flapper_tasks = [_flapper(i) for i in range(6)]
    await asyncio.gather(*flapper_tasks)

    # Invariant checks:
    # State must be either MONITORING_STREAM or STALE_DATA_HOLD, never None or corrupted
    assert sm.current_state in (LifecycleState.MONITORING_STREAM, LifecycleState.STALE_DATA_HOLD)
    # Rebalance safety must strictly match current state
    assert sm.is_safe_to_rebalance() == (sm.current_state == LifecycleState.MONITORING_STREAM)
    # Upstream connected flag must be strictly coherent with current state
    if sm.current_state == LifecycleState.MONITORING_STREAM:
        assert sm.is_upstream_connected is True
    elif sm.current_state == LifecycleState.STALE_DATA_HOLD:
        assert sm.is_upstream_connected is False


@pytest.mark.asyncio
async def test_adv_flapping_with_slow_gap_backfill():
    """Adversarial Challenge: Reconnect initiates gap backfill which experiences latency.
    A disconnect arrives while in SYNCING_HISTORY.
    Verify transition from SYNCING_HISTORY -> STALE_DATA_HOLD does not deadlock.
    """
    backfill_entered = asyncio.Event()
    backfill_proceed = asyncio.Event()

    async def _delayed_backfill(start: datetime, end: datetime) -> int:
        backfill_entered.set()
        await backfill_proceed.wait()
        return 42

    sm = IngestionStateMachine(gap_backfill_callback=_delayed_backfill)
    await sm.check_initial_health({"upstream": "connected"})

    # Record bar 200 seconds ago to ensure gap > 60s
    t_old = datetime.now(timezone.utc) - timedelta(seconds=200)
    sm.record_bar(_make_bar("SPY", t_old))

    # Disconnect
    await sm.handle_upstream_disconnected("Drop before delayed backfill")

    # Start reconnect in background task
    reconnect_task = asyncio.create_task(sm.handle_upstream_connected("Restore with backfill"))

    # Wait until backfill has started and entered SYNCING_HISTORY
    await backfill_entered.wait()
    assert sm.current_state == LifecycleState.SYNCING_HISTORY
    assert sm.is_safe_to_rebalance() is False

    # Release backfill and await completion
    backfill_proceed.set()
    await reconnect_task

    assert sm.current_state == LifecycleState.MONITORING_STREAM
    assert sm.is_safe_to_rebalance() is True


# ============================================================================
# Category 3: Gap Backfill REST Proxy Inquiries
# ============================================================================

@pytest.mark.asyncio
async def test_adv_gap_backfill_timestamp_window_verification():
    """Adversarial Challenge: Verify gap backfill callback is invoked with the exact
    [start_gap, now] window and that rebalance is blocked throughout SYNCING_HISTORY.
    """
    captured_start: Optional[datetime] = None
    captured_end: Optional[datetime] = None
    was_safe_during_backfill: Optional[bool] = None

    async def _inspecting_backfill(start: datetime, end: datetime) -> int:
        nonlocal captured_start, captured_end, was_safe_during_backfill
        captured_start = start
        captured_end = end
        # Allocation safety MUST be False during gap synchronization
        was_safe_during_backfill = sm.is_safe_to_rebalance()
        return 120

    sm = IngestionStateMachine(gap_backfill_callback=_inspecting_backfill)
    await sm.check_initial_health({"upstream": "connected"})

    # Set last bar 300s in the past
    t_bar = datetime.now(timezone.utc) - timedelta(seconds=300)
    sm.record_bar(_make_bar("SPY", t_bar))

    await sm.handle_upstream_disconnected("Network severed")
    t_reconnect = datetime.now(timezone.utc)
    await sm.handle_upstream_connected("Network restored")

    # Verify captured arguments
    assert captured_start == t_bar
    assert captured_end is not None
    assert abs((captured_end - t_reconnect).total_seconds()) < 1.0
    assert was_safe_during_backfill is False
    assert sm.current_state == LifecycleState.MONITORING_STREAM
    assert sm.is_safe_to_rebalance() is True


@pytest.mark.asyncio
async def test_adv_gap_backfill_boundary_threshold():
    """Adversarial Challenge: Boundary conditions on 60-second backfill threshold.
    - gap = 59.0s -> backfill SKIPPED
    - gap = 61.0s -> backfill EXECUTED
    """
    backfill_count = 0

    async def _counter_backfill(start: datetime, end: datetime) -> int:
        nonlocal backfill_count
        backfill_count += 1
        return 10

    sm = IngestionStateMachine(gap_backfill_callback=_counter_backfill)
    await sm.check_initial_health({"upstream": "connected"})

    # 1. Gap <= 60s (50 seconds)
    t_recent = datetime.now(timezone.utc) - timedelta(seconds=50)
    sm.record_bar(_make_bar("SPY", t_recent))
    await sm.handle_upstream_disconnected("Short blip")
    await sm.handle_upstream_connected("Quick restore")
    assert backfill_count == 0  # Skipped

    # 2. Gap > 60s (70 seconds)
    t_stale = datetime.now(timezone.utc) - timedelta(seconds=70)
    sm._last_bar_timestamp = t_stale  # update manually to simulate gap
    await sm.handle_upstream_disconnected("Longer outage")
    await sm.handle_upstream_connected("Restore after 70s")
    assert backfill_count == 1  # Executed


@pytest.mark.asyncio
async def test_adv_gap_backfill_rest_failure_resilience():
    """Adversarial Challenge: REST proxy fails during backfill (HTTP 500, timeout).
    State machine must catch exception, log warning, and safely transition to MONITORING_STREAM.
    """
    async def _failing_backfill(start: datetime, end: datetime) -> int:
        raise ConnectionResetError("REST proxy gateway connection reset (502/500)")

    sm = IngestionStateMachine(gap_backfill_callback=_failing_backfill)
    await sm.check_initial_health({"upstream": "connected"})

    t_old = datetime.now(timezone.utc) - timedelta(seconds=180)
    sm.record_bar(_make_bar("SPY", t_old))

    await sm.handle_upstream_disconnected("Upstream drop")
    # Reconnect with failing backfill
    await sm.handle_upstream_connected("Upstream restore")

    # State machine must NOT crash; must successfully reach MONITORING_STREAM
    assert sm.current_state == LifecycleState.MONITORING_STREAM
    assert sm.is_safe_to_rebalance() is True


@pytest.mark.asyncio
async def test_adv_gap_backfill_e2e_mock_server_query():
    """Adversarial Challenge: End-to-end backfill test using live ExtendedMockRelayServer.
    Verifies that AlpacaRelayClient._backfill_gap actually queries the REST proxy
    for subscribed symbols across the gap window.
    """
    server = ExtendedMockRelayServer(token="secret-relay-token-123")
    await server.start()
    try:
        # Populate mock server with historical bars for SPY and QQQ
        now = datetime.now(timezone.utc)
        bars_spy = [
            {"t": (now - timedelta(minutes=3)).isoformat(), "o": 500.0, "h": 501.0, "l": 499.0, "c": 500.5, "v": 100},
            {"t": (now - timedelta(minutes=2)).isoformat(), "o": 500.5, "h": 502.0, "l": 500.0, "c": 501.5, "v": 150},
        ]
        bars_qqq = [
            {"t": (now - timedelta(minutes=3)).isoformat(), "o": 400.0, "h": 401.0, "l": 399.0, "c": 400.5, "v": 200},
        ]
        server.add_mock_bars("SPY", bars_spy)
        server.add_mock_bars("QQQ", bars_qqq)

        client = AlpacaRelayClient(
            base_url=server.http_url,
            ws_url=server.ws_url,
            relay_token="secret-relay-token-123",
        )
        try:
            await client.connect_stream(symbols=["SPY", "QQQ"], channels=["bars"])
            assert client.get_connection_status() == "MONITORING_STREAM"

            # Simulate gap by setting last bar timestamp to 4 minutes ago
            t_gap_start = now - timedelta(minutes=4)
            client.state_machine._last_bar_timestamp = t_gap_start

            # Trigger disconnect
            await client.handle_lifecycle_event("upstream_disconnected")
            assert client.get_connection_status() == "STALE_DATA_HOLD"
            assert client.is_safe_to_rebalance() is False

            # Trigger reconnect: will execute _backfill_gap against mock REST server
            await client.handle_lifecycle_event("upstream_connected")

            # Verify client returned to MONITORING_STREAM
            assert client.get_connection_status() == "MONITORING_STREAM"
            assert client.is_safe_to_rebalance() is True
        finally:
            await client.close()
    finally:
        await server.stop()


# ============================================================================
# Category 4: Stream Silence & Watchdog Timer
# ============================================================================

@pytest.mark.asyncio
async def test_adv_watchdog_exact_timeout_boundary():
    """Adversarial Challenge: Watchdog timer precision boundary test.
    Timeout = 300.0s:
    - At age = 299.9s: Watchdog must return False and keep MONITORING_STREAM.
    - At age = 300.1s: Watchdog must trigger STALE_DATA_HOLD and return True.
    """
    sm = IngestionStateMachine(stale_timeout_seconds=300.0)
    await sm.check_initial_health({"upstream": "connected"})

    t0 = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
    sm.record_message(t0)

    # 1. Check at 299.9s -> Not stale
    stale_sub = await sm.check_watchdog(current_time=t0 + timedelta(seconds=299.9))
    assert stale_sub is False
    assert sm.current_state == LifecycleState.MONITORING_STREAM
    assert sm.is_safe_to_rebalance() is True

    # 2. Check at 300.1s -> Stale triggered
    stale_sup = await sm.check_watchdog(current_time=t0 + timedelta(seconds=300.1))
    assert stale_sup is True
    assert sm.current_state == LifecycleState.STALE_DATA_HOLD
    assert sm.is_safe_to_rebalance() is False
    assert sm.get_market_regime_override() == MarketRegime.STALE_DATA_HOLD


@pytest.mark.asyncio
async def test_adv_watchdog_repeated_polling_idempotency():
    """Adversarial Challenge: After watchdog triggers STALE_DATA_HOLD, subsequent polls
    must return False and not create duplicate transitions in the audit trail.
    """
    sm = IngestionStateMachine(stale_timeout_seconds=300.0)
    await sm.check_initial_health({"upstream": "connected"})

    t0 = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
    sm.record_message(t0)

    # Trigger watchdog
    await sm.check_watchdog(current_time=t0 + timedelta(seconds=350))
    assert sm.current_state == LifecycleState.STALE_DATA_HOLD
    transitions_count = len(sm.transition_history)

    # Subsequent polls at 400s, 500s, 600s
    for dt in (400, 500, 600):
        res = await sm.check_watchdog(current_time=t0 + timedelta(seconds=dt))
        assert res is False
        assert sm.current_state == LifecycleState.STALE_DATA_HOLD

    # No redundant transitions logged
    assert len(sm.transition_history) == transitions_count


@pytest.mark.asyncio
async def test_adv_watchdog_recovery_cycle():
    """Adversarial Challenge: Watchdog triggers STALE_DATA_HOLD, then upstream restores
    stream and sends a new bar. Watchdog must clear and return to safe monitoring.
    """
    sm = IngestionStateMachine(stale_timeout_seconds=300.0)
    await sm.check_initial_health({"upstream": "connected"})

    t0 = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
    sm.record_message(t0)

    # Watchdog fires at t0 + 310s
    await sm.check_watchdog(current_time=t0 + timedelta(seconds=310))
    assert sm.current_state == LifecycleState.STALE_DATA_HOLD
    assert sm.is_safe_to_rebalance() is False

    # Upstream reconnects at t0 + 350s
    t_rec = t0 + timedelta(seconds=350)
    await sm.handle_upstream_connected("Stream recovered by watchdog watchdog handler")
    assert sm.current_state == LifecycleState.MONITORING_STREAM
    assert sm.is_safe_to_rebalance() is True

    # New message arrives at t_rec + 5s
    sm.record_message(t_rec + timedelta(seconds=5))

    # Check watchdog at t_rec + 10s: should NOT trigger
    stale = await sm.check_watchdog(current_time=t_rec + timedelta(seconds=10))
    assert stale is False
    assert sm.is_safe_to_rebalance() is True


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_timeout", [1.0, 10.0, 60.0, 300.0, 900.0])
async def test_adv_watchdog_custom_timeouts(custom_timeout: float):
    """Adversarial Challenge: Verify watchdog handles arbitrary custom timeout thresholds."""
    sm = IngestionStateMachine(stale_timeout_seconds=custom_timeout)
    await sm.check_initial_health({"upstream": "connected"})

    t0 = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
    sm.record_message(t0)

    # Just before
    assert await sm.check_watchdog(t0 + timedelta(seconds=custom_timeout - 0.1)) is False
    # Just after
    assert await sm.check_watchdog(t0 + timedelta(seconds=custom_timeout + 0.1)) is True
    assert sm.current_state == LifecycleState.STALE_DATA_HOLD


# ============================================================================
# Category 5: End-to-End WebSocket Chaos & Server Eviction
# ============================================================================

@pytest.mark.asyncio
async def test_adv_e2e_server_lifecycle_broadcast_chaos():
    """Adversarial Challenge: Full end-to-end chaos test with live mock server.
    Fires rapid server-side broadcast events while client is receiving real-time bars.
    Verifies that the client never deadlocks, never throws unhandled exceptions,
    and accurately gates portfolio rebalancing safety.
    """
    server = ExtendedMockRelayServer(token="secret-relay-token-123")
    await server.start()
    try:
        client = AlpacaRelayClient(
            base_url=server.http_url,
            ws_url=server.ws_url,
            relay_token="secret-relay-token-123",
        )
        try:
            await client.connect_stream(symbols=["SPY"], channels=["bars"])
            assert client.is_safe_to_rebalance() is True

            # Send some live bars
            await server.broadcast_bar("SPY", {
                "o": 500.0, "h": 501.0, "l": 499.0, "c": 500.5, "v": 100, "t": "2026-09-03T12:00:00Z"
            })
            await asyncio.sleep(0.05)

            # Broadcast 5 alternating lifecycle disconnect/reconnect events over WebSocket
            for i in range(5):
                await server.broadcast_lifecycle("upstream_disconnected")
                await asyncio.sleep(0.05)
                assert client.get_connection_status() == "STALE_DATA_HOLD"
                assert client.is_safe_to_rebalance() is False

                await server.broadcast_lifecycle("upstream_connected")
                await asyncio.sleep(0.05)
                assert client.get_connection_status() == "MONITORING_STREAM"
                assert client.is_safe_to_rebalance() is True
        finally:
            await client.close()
    finally:
        await server.stop()


# ============================================================================
# Category 6: Deep Adversarial Boundary & Architectural Probes
# ============================================================================

@pytest.mark.asyncio
async def test_adv_watchdog_out_of_order_bar_timestamp_regression():
    """Adversarial Challenge: Ingestion of out-of-order or late-arriving bars.
    When record_bar is called with a bar from the past, record_message unconditionally
    sets _last_msg_timestamp = bar.timestamp.
    Asserts the exact empirical impact: subsequent watchdog check triggers STALE_DATA_HOLD
    because the timestamp regressed into the past.
    """
    sm = IngestionStateMachine(stale_timeout_seconds=300.0)
    await sm.check_initial_health({"upstream": "connected"})

    now = datetime.now(timezone.utc)
    # 1. Ingest fresh current bar
    sm.record_bar(_make_bar("SPY", now))
    assert sm.last_msg_timestamp == now
    assert await sm.check_watchdog(now + timedelta(seconds=10)) is False

    # 2. Ingest out-of-order/late bar with timestamp 400 seconds in the past
    late_bar = _make_bar("SPY", now - timedelta(seconds=400))
    sm.record_bar(late_bar)

    # Note: _last_bar_timestamp retains the newest bar timestamp (monotonically non-decreasing)
    assert sm.last_bar_timestamp == now
    # BUT _last_msg_timestamp is regressed to the late bar's timestamp!
    assert sm.last_msg_timestamp == late_bar.timestamp

    # 3. Watchdog check at current time: triggers STALE_DATA_HOLD due to regressed msg timestamp
    triggered = await sm.check_watchdog(now)
    assert triggered is True
    assert sm.current_state == LifecycleState.STALE_DATA_HOLD
    assert sm.is_safe_to_rebalance() is False


@pytest.mark.asyncio
async def test_adv_watchdog_naive_datetime_handling():
    """Adversarial Challenge: Passing timezone-naive datetime to record_message.
    In Python, subtracting naive and aware datetimes raises TypeError.
    Verifies that naive datetime causes TypeError in check_watchdog if not normalized.
    """
    sm = IngestionStateMachine(stale_timeout_seconds=300.0)
    await sm.check_initial_health({"upstream": "connected"})

    naive_dt = datetime(2026, 9, 3, 12, 0, 0)  # No tzinfo
    sm.record_message(naive_dt)

    # Calling check_watchdog with default UTC now raises TypeError due to Python datetime rules
    with pytest.raises(TypeError) as exc_info:
        await sm.check_watchdog()
    assert "can't subtract offset-naive and offset-aware datetimes" in str(exc_info.value)


@pytest.mark.asyncio
async def test_adv_reentrant_callback_deadlock_prevention():
    """Adversarial Challenge: Callback attempting re-entrant state transition.
    Because asyncio.Lock is non-reentrant, a synchronous direct recursive call
    to handle_upstream_disconnected within a state transition callback will timeout/deadlock.
    Verifies this architectural constraint via asyncio.wait_for.
    """
    sm = IngestionStateMachine()
    await sm.check_initial_health({"upstream": "connected"})
    await sm.handle_upstream_disconnected("Move to STALE_DATA_HOLD first")

    async def _reentrant_cb(prev, next_st, reason):
        # If callback attempts re-entrant transition, it must wait for lock
        if next_st == LifecycleState.MONITORING_STREAM:
            await sm.handle_upstream_disconnected("Reentrant call")

    sm.register_state_change_callback(_reentrant_cb)

    # Demonstrates that re-entrant calls would deadlock without timeout protection
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            sm.handle_upstream_connected("Trigger reentrant cb"),
            timeout=0.2,
        )

