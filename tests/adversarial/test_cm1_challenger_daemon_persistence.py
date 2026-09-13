"""
tests.adversarial.test_cm1_challenger_daemon_persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone C-M1 Empirical Challenger Stress Test Suite:
Challenger 2 Verification for Daemon Scheduler Retrying & Persistence Resilience:
1. Daemon scheduler retrying after persistence failures:
   - Full DecisionDaemon integration with transient DB locks/failures retrying until success.
   - Market close evaluation window boundary (stopping retries after 16:00 ET and resuming next day).
   - Multi-cadence decoupling and independent retry resilience (DAILY_CLOSE vs WEEKLY_REBALANCE).
   - Multi-handler failure isolation.
2. Multi-entity SQLite atomic persistence rollback in _handle_daily_close:
   - Atomic rollback when failure occurs at signals.save.
   - Atomic rollback when failure occurs at allocations.save.
   - Atomic rollback when failure occurs at record_transition_if_changed.
   - Historical state preservation: rollback of current day preserves historical records intact.
   - Read isolation under concurrency during failing/rolling back transactions.
3. BaseException / asyncio.CancelledError transaction rollback in Database.transaction():
   - Real asyncio.Task cancellation in file-backed database.
   - In-memory (:memory:) CancelledError rollback preventing dirty data leakage into subsequent commits.
   - Full BaseException suite (CancelledError, KeyboardInterrupt, SystemExit, CustomBaseException).
   - Concurrent tasks with mid-transaction cancellations without mutex deadlocks.
   - Multi-table atomic coordinated rollback upon task cancellation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import pytest

from strategy_engine.core.models import (
    Bar,
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderType,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.daemon.daemon import DaemonConfig, DecisionDaemon
from strategy_engine.daemon.scheduler import (
    CadenceType,
    MarketCalendar,
    MarketScheduler,
)
from strategy_engine.storage.audit_logger import JSONLAuditLogger
from strategy_engine.storage.database import Database
from strategy_engine.storage.repositories import (
    AllocationRepository,
    MarketBarRepository,
    PortfolioStateRepository,
    RebalanceOrderRepository,
    RegimeEventRepository,
    SignalSnapshotRepository,
    StorageService,
)

ET = ZoneInfo("America/New_York")


def _create_dummy_bars(base_t: datetime, count: int = 250) -> Dict[str, List[Bar]]:
    """Helper to generate dummy historical bars for SPY and QQQ."""
    bars_map = {}
    for sym, p_start in [("SPY", 450.0), ("QQQ", 380.0)]:
        bars = [
            Bar(
                symbol=sym,
                timestamp=base_t - timedelta(days=count - i),
                open=p_start + i * 0.1,
                high=p_start + i * 0.1 + 1.0,
                low=p_start + i * 0.1 - 1.0,
                close=p_start + i * 0.1 + 0.5,
                volume=1000000,
            )
            for i in range(count)
        ]
        bars_map[sym] = bars
    return bars_map


# ============================================================================
# 1. Daemon Scheduler Retrying after Persistence Failures
# ============================================================================

@pytest.mark.asyncio
async def test_daemon_scheduler_retrying_after_mock_db_failure(tmp_path):
    """Empirical challenge: Full DecisionDaemon integration where DB fails on first 2

    ticks during DAILY_CLOSE, and succeeds on the 3rd tick.
    Verify:
    - Tick 1: DB fails -> transaction rolled back -> scheduler tick returns empty -> last_executed is None -> 0 audit entries.
    - Tick 2: DB fails again -> transaction rolled back -> scheduler tick returns empty -> last_executed is None -> 0 audit entries.
    - Tick 3: DB recovers -> transaction commits -> scheduler tick returns [DAILY_CLOSE] -> last_executed is today -> 1 audit entry.
    - Tick 4: Same day subsequent tick is idempotent (does not re-execute).
    """
    db_path = tmp_path / "daemon_retry.db"
    log_dir = tmp_path / "daemon_retry_logs"
    config = DaemonConfig(db_path=str(db_path), log_dir=str(log_dir), dry_run=True)
    storage = StorageService(db_path)
    audit_logger = JSONLAuditLogger(log_dir)

    daemon = DecisionDaemon(config=config, storage=storage, audit_logger=audit_logger)
    await daemon.client.state_machine.check_initial_health({"upstream": "connected"})

    # Populate daily bars
    base_t = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    daemon._cached_daily_bars = _create_dummy_bars(base_t)

    # Wednesday Sep 2, 2026
    today = date(2026, 9, 2)
    t_eval_1 = datetime(2026, 9, 2, 15, 50, tzinfo=ET)
    t_eval_2 = datetime(2026, 9, 2, 15, 52, tzinfo=ET)
    t_eval_3 = datetime(2026, 9, 2, 15, 54, tzinfo=ET)
    t_eval_4 = datetime(2026, 9, 2, 15, 56, tzinfo=ET)

    fail_count = 0
    orig_save_signals = storage.signals.save

    def flaky_signals_save(*args, **kwargs):
        nonlocal fail_count
        if fail_count < 2:
            fail_count += 1
            raise sqlite3.OperationalError(f"Simulated database locked error (attempt {fail_count})")
        return orig_save_signals(*args, **kwargs)

    storage.signals.save = flaky_signals_save

    # --- Tick 1 (15:50 ET) ---
    trig1 = await daemon.scheduler.tick(t_eval_1)
    assert CadenceType.DAILY_CLOSE not in trig1, "DAILY_CLOSE must not trigger on failure"
    assert daemon.scheduler._last_executed[CadenceType.DAILY_CLOSE] is None
    assert fail_count == 1

    # Verify zero data in database
    with storage.db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 0
        c.execute("SELECT COUNT(*) FROM allocations;")
        assert c.fetchone()[0] == 0
        c.execute("SELECT COUNT(*) FROM regime_events;")
        assert c.fetchone()[0] == 0

    # Verify zero data in audit logger
    records_1 = audit_logger.read_date(str(today))
    assert len(records_1) == 0, "No audit log should be written on DB failure"

    # --- Tick 2 (15:52 ET) ---
    trig2 = await daemon.scheduler.tick(t_eval_2)
    assert CadenceType.DAILY_CLOSE not in trig2
    assert daemon.scheduler._last_executed[CadenceType.DAILY_CLOSE] is None
    assert fail_count == 2

    # Still zero records
    with storage.db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 0

    records_2 = audit_logger.read_date(str(today))
    assert len(records_2) == 0

    # --- Tick 3 (15:54 ET) ---
    # DB failure has cleared now (fail_count == 2, so flaky_signals_save calls orig_save_signals)
    trig3 = await daemon.scheduler.tick(t_eval_3)
    assert CadenceType.DAILY_CLOSE in trig3, "DAILY_CLOSE must trigger on successful retry"
    assert daemon.scheduler._last_executed[CadenceType.DAILY_CLOSE] == today

    # Verify exactly 1 set of records committed to DB
    with storage.db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 1
        c.execute("SELECT COUNT(*) FROM allocations;")
        assert c.fetchone()[0] == 1
        c.execute("SELECT COUNT(*) FROM regime_events;")
        assert c.fetchone()[0] == 1

    # Verify exactly 1 audit log record written
    records_3 = audit_logger.read_date(str(today))
    assert len(records_3) == 1
    assert records_3[0]["trigger"] == "DAILY_CLOSE"

    # --- Tick 4 (15:56 ET) ---
    # Idempotency check: should not re-run
    trig4 = await daemon.scheduler.tick(t_eval_4)
    assert len(trig4) == 0
    assert daemon.scheduler._last_executed[CadenceType.DAILY_CLOSE] == today

    with storage.db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 1

    records_4 = audit_logger.read_date(str(today))
    assert len(records_4) == 1

    await daemon.shutdown("Test complete")


@pytest.mark.asyncio
async def test_scheduler_stops_retrying_after_market_close_and_resumes_next_day(tmp_path):
    """Verify that if persistence failures persist until market close (16:00 ET),

    the scheduler ceases retries for that day, and correctly triggers on the next day.
    """
    cal = MarketCalendar()
    sched = MarketScheduler(calendar=cal, daily_close_offset_minutes=10)

    # Day 1: Wednesday Sep 2, 2026
    day1_eval_1 = datetime(2026, 9, 2, 15, 50, tzinfo=ET)
    day1_eval_2 = datetime(2026, 9, 2, 15, 58, tzinfo=ET)
    day1_close = datetime(2026, 9, 2, 16, 0, 0, tzinfo=ET)  # Market closed
    day1_after_close = datetime(2026, 9, 2, 16, 1, 0, tzinfo=ET)

    # Day 2: Thursday Sep 3, 2026
    day2_eval = datetime(2026, 9, 3, 15, 50, tzinfo=ET)

    attempt_count = 0
    db_operational = False

    def simulated_db_handler(dt: datetime):
        nonlocal attempt_count
        attempt_count += 1
        if not db_operational:
            raise sqlite3.OperationalError("Database down")

    sched.on_cadence(CadenceType.DAILY_CLOSE, simulated_db_handler)

    # Day 1 ticks during evaluation window (retries occur)
    t1 = await sched.tick(day1_eval_1)
    assert len(t1) == 0
    assert sched._last_executed[CadenceType.DAILY_CLOSE] is None

    t2 = await sched.tick(day1_eval_2)
    assert len(t2) == 0
    assert sched._last_executed[CadenceType.DAILY_CLOSE] is None
    assert attempt_count == 2

    # Market close: 16:00 ET
    t3 = await sched.tick(day1_close)
    assert len(t3) == 0
    assert attempt_count == 2, "No handler dispatch should occur at or after market close"

    # After market close: 16:01 ET
    t4 = await sched.tick(day1_after_close)
    assert len(t4) == 0
    assert attempt_count == 2

    # Day 2: DB restored
    db_operational = True
    t5 = await sched.tick(day2_eval)
    assert CadenceType.DAILY_CLOSE in t5
    assert sched._last_executed[CadenceType.DAILY_CLOSE] == date(2026, 9, 3)
    assert attempt_count == 3


@pytest.mark.asyncio
async def test_multi_cadence_independent_retry_isolation():
    """Verify that if DAILY_CLOSE succeeds but WEEKLY_REBALANCE fails on Friday 15:50 ET:

    - DAILY_CLOSE is marked executed.
    - WEEKLY_REBALANCE is not marked executed.
    - On the subsequent tick (15:51 ET), DAILY_CLOSE is NOT re-executed,
      and WEEKLY_REBALANCE independently retries and succeeds.
    """
    cal = MarketCalendar()
    sched = MarketScheduler(calendar=cal, daily_close_offset_minutes=10)

    # Friday Sep 4, 2026 at 15:50 ET
    friday_1550 = datetime(2026, 9, 4, 15, 50, tzinfo=ET)
    friday_1551 = datetime(2026, 9, 4, 15, 51, tzinfo=ET)
    today = friday_1550.date()

    daily_calls = 0
    weekly_calls = 0
    weekly_should_fail = True

    def daily_handler(dt: datetime):
        nonlocal daily_calls
        daily_calls += 1

    def weekly_handler(dt: datetime):
        nonlocal weekly_calls
        weekly_calls += 1
        if weekly_should_fail:
            raise sqlite3.OperationalError("Weekly order persistence failed")

    sched.on_cadence(CadenceType.DAILY_CLOSE, daily_handler)
    sched.on_cadence(CadenceType.WEEKLY_REBALANCE, weekly_handler)

    # Tick 1: DAILY_CLOSE succeeds, WEEKLY_REBALANCE fails
    trig1 = await sched.tick(friday_1550)
    assert CadenceType.DAILY_CLOSE in trig1
    assert CadenceType.WEEKLY_REBALANCE not in trig1
    assert sched._last_executed[CadenceType.DAILY_CLOSE] == today
    assert sched._last_executed[CadenceType.WEEKLY_REBALANCE] is None
    assert daily_calls == 1
    assert weekly_calls == 1

    # Clear failure for weekly handler
    weekly_should_fail = False

    # Tick 2: DAILY_CLOSE skipped (already executed today), WEEKLY_REBALANCE retried
    trig2 = await sched.tick(friday_1551)
    assert CadenceType.DAILY_CLOSE not in trig2
    assert CadenceType.WEEKLY_REBALANCE in trig2
    assert sched._last_executed[CadenceType.WEEKLY_REBALANCE] == today
    assert daily_calls == 1, "DAILY_CLOSE must not be re-executed on weekly retry"
    assert weekly_calls == 2


@pytest.mark.asyncio
async def test_scheduler_multiple_handlers_failure_isolation():
    """Verify that when multiple handlers are registered for a cadence,

    an exception in the first handler does not prevent subsequent handlers from executing,
    and the scheduler registers the overall tick as failed.
    """
    cal = MarketCalendar()
    sched = MarketScheduler(calendar=cal, daily_close_offset_minutes=10)

    eval_t = datetime(2026, 9, 2, 15, 50, tzinfo=ET)
    h1_called = 0
    h2_called = 0

    def failing_handler(dt: datetime):
        nonlocal h1_called
        h1_called += 1
        raise RuntimeError("Handler 1 explosion")

    def successful_handler(dt: datetime):
        nonlocal h2_called
        h2_called += 1

    sched.on_cadence(CadenceType.DAILY_CLOSE, failing_handler)
    sched.on_cadence(CadenceType.DAILY_CLOSE, successful_handler)

    trig = await sched.tick(eval_t)
    assert CadenceType.DAILY_CLOSE not in trig
    assert sched._last_executed[CadenceType.DAILY_CLOSE] is None
    assert h1_called == 1
    assert h2_called == 1, "Handler 2 must still be executed despite Handler 1 failure"


# ============================================================================
# 2. Multi-Entity SQLite Atomic Persistence Rollback in _handle_daily_close
# ============================================================================

@pytest.mark.asyncio
async def test_atomic_rollback_on_signals_save_failure(tmp_path):
    """Verify atomic rollback when failure occurs at signals.save:

    zero records in signal_snapshots, allocations, and regime_events.
    """
    db_path = tmp_path / "rollback_sig.db"
    log_dir = tmp_path / "rollback_sig_logs"
    storage = StorageService(db_path)
    audit_logger = JSONLAuditLogger(log_dir)
    daemon = DecisionDaemon(storage=storage, audit_logger=audit_logger)
    await daemon.client.state_machine.check_initial_health({"upstream": "connected"})

    base_t = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    daemon._cached_daily_bars = _create_dummy_bars(base_t)
    eval_dt = datetime(2026, 9, 2, 15, 50, tzinfo=ET)

    def fail_sig(*args, **kwargs):
        raise sqlite3.OperationalError("Signal table locked")

    storage.signals.save = fail_sig

    with pytest.raises(sqlite3.OperationalError, match="Signal table locked"):
        await daemon._handle_daily_close(eval_dt)

    with storage.db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 0
        c.execute("SELECT COUNT(*) FROM allocations;")
        assert c.fetchone()[0] == 0
        c.execute("SELECT COUNT(*) FROM regime_events;")
        assert c.fetchone()[0] == 0

    assert len(audit_logger.read_date("2026-09-02")) == 0
    await daemon.shutdown("Test complete")


@pytest.mark.asyncio
async def test_atomic_rollback_on_allocations_save_failure(tmp_path):
    """Verify atomic rollback when failure occurs at allocations.save:

    signals.save had already executed an INSERT, but all tables must roll back to 0.
    """
    db_path = tmp_path / "rollback_alloc.db"
    log_dir = tmp_path / "rollback_alloc_logs"
    storage = StorageService(db_path)
    audit_logger = JSONLAuditLogger(log_dir)
    daemon = DecisionDaemon(storage=storage, audit_logger=audit_logger)
    await daemon.client.state_machine.check_initial_health({"upstream": "connected"})

    base_t = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    daemon._cached_daily_bars = _create_dummy_bars(base_t)
    eval_dt = datetime(2026, 9, 2, 15, 50, tzinfo=ET)

    def fail_alloc(*args, **kwargs):
        raise sqlite3.IntegrityError("Simulated allocation constraint error")

    storage.allocations.save = fail_alloc

    with pytest.raises(sqlite3.IntegrityError, match="Simulated allocation constraint error"):
        await daemon._handle_daily_close(eval_dt)

    with storage.db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 0, "Signal snapshot was not rolled back!"
        c.execute("SELECT COUNT(*) FROM allocations;")
        assert c.fetchone()[0] == 0
        c.execute("SELECT COUNT(*) FROM regime_events;")
        assert c.fetchone()[0] == 0

    assert len(audit_logger.read_date("2026-09-02")) == 0
    await daemon.shutdown("Test complete")


@pytest.mark.asyncio
async def test_atomic_rollback_on_regime_transition_save_failure(tmp_path):
    """Verify atomic rollback when failure occurs at record_transition_if_changed:

    both signals.save AND allocations.save had already executed INSERTs,
    but the entire multi-entity transaction must be completely rolled back.
    """
    db_path = tmp_path / "rollback_regime.db"
    log_dir = tmp_path / "rollback_regime_logs"
    storage = StorageService(db_path)
    audit_logger = JSONLAuditLogger(log_dir)
    daemon = DecisionDaemon(storage=storage, audit_logger=audit_logger)
    await daemon.client.state_machine.check_initial_health({"upstream": "connected"})

    base_t = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    daemon._cached_daily_bars = _create_dummy_bars(base_t)
    eval_dt = datetime(2026, 9, 2, 15, 50, tzinfo=ET)

    def fail_regime(*args, **kwargs):
        raise sqlite3.OperationalError("Regime events disk write failed")

    storage.regimes.record_transition_if_changed = fail_regime

    with pytest.raises(sqlite3.OperationalError, match="Regime events disk write failed"):
        await daemon._handle_daily_close(eval_dt)

    with storage.db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 0, "Signal snapshot should be rolled back"
        c.execute("SELECT COUNT(*) FROM allocations;")
        assert c.fetchone()[0] == 0, "Allocation should be rolled back"
        c.execute("SELECT COUNT(*) FROM regime_events;")
        assert c.fetchone()[0] == 0, "Regime event should be rolled back"

    assert len(audit_logger.read_date("2026-09-02")) == 0
    await daemon.shutdown("Test complete")


@pytest.mark.asyncio
async def test_atomic_rollback_preserves_historical_records(tmp_path):
    """Verify that when Day 2 fails mid-transaction, Day 1's historical records

    remain strictly intact and uncorrupted, and Day 2's subsequent retry succeeds cleanly.
    """
    db_path = tmp_path / "preserve_history.db"
    log_dir = tmp_path / "preserve_history_logs"
    storage = StorageService(db_path)
    audit_logger = JSONLAuditLogger(log_dir)
    daemon = DecisionDaemon(storage=storage, audit_logger=audit_logger)
    await daemon.client.state_machine.check_initial_health({"upstream": "connected"})

    base_t = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    daemon._cached_daily_bars = _create_dummy_bars(base_t)

    # --- Day 1 (Sep 1): Sells cleanly ---
    day1_dt = datetime(2026, 9, 1, 15, 50, tzinfo=ET)
    await daemon._handle_daily_close(day1_dt)

    with storage.db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 1
        c.execute("SELECT COUNT(*) FROM allocations;")
        assert c.fetchone()[0] == 1
        c.execute("SELECT COUNT(*) FROM regime_events;")
        assert c.fetchone()[0] == 1

    # --- Day 2 (Sep 2): Mid-transaction failure at allocations.save ---
    orig_alloc_save = storage.allocations.save

    def fail_day2(*args, **kwargs):
        raise sqlite3.OperationalError("Simulated Day 2 disk error")

    storage.allocations.save = fail_day2
    day2_dt = datetime(2026, 9, 2, 15, 50, tzinfo=ET)

    with pytest.raises(sqlite3.OperationalError):
        await daemon._handle_daily_close(day2_dt)

    # Verify Day 1 data is PRESERVED, Day 2 data is 100% rolled back
    with storage.db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 1, "Historical signals corrupted!"
        c.execute("SELECT COUNT(*) FROM allocations;")
        assert c.fetchone()[0] == 1, "Historical allocations corrupted!"
        c.execute("SELECT COUNT(*) FROM regime_events;")
        assert c.fetchone()[0] == 1, "Historical regimes corrupted!"

    # --- Day 2 Retry: Success ---
    storage.allocations.save = orig_alloc_save
    await daemon._handle_daily_close(day2_dt)

    with storage.db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 2
        c.execute("SELECT COUNT(*) FROM allocations;")
        assert c.fetchone()[0] == 2

    await daemon.shutdown("Test complete")


def test_concurrent_read_isolation_during_rollback(tmp_path):
    """Verify that concurrent readers never observe dirty/uncommitted records

    while an aborted transaction is executing and rolling back in SQLite WAL.
    """
    db_file = tmp_path / "read_isolation.db"
    db = Database(db_file)

    stop_readers = False
    dirty_reads: List[str] = []

    def reader_worker(wid: int):
        while not stop_readers:
            with db.transaction() as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM signal_snapshots WHERE rationale LIKE 'DIRTY_%';")
                count = c.fetchone()[0]
                if count > 0:
                    dirty_reads.append(f"Reader {wid} saw {count} dirty uncommitted rows!")
            time.sleep(0.001)

    # Start 5 concurrent readers
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        reader_futs = [executor.submit(reader_worker, i) for i in range(5)]

        # Execute 20 transactions that insert dirty rows and immediately abort/rollback
        for tx in range(20):
            try:
                with db.transaction() as conn:
                    c = conn.cursor()
                    c.execute(
                        """
                        INSERT INTO signal_snapshots (
                            timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                            drawdown_gate, atr_stop_triggered, rationale, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"2026-09-01T15:{tx:02d}:00+00:00",
                            "BEAR_CRISIS",
                            450.0,
                            380.0,
                            0.25,
                            0.5,
                            0.8,
                            1,
                            f"DIRTY_ROW_{tx}",
                            "{}",
                        ),
                    )
                    # Simulate processing delay then abort
                    time.sleep(0.005)
                    raise RuntimeError("Aborting transaction")
            except RuntimeError:
                pass

        stop_readers = True
        concurrent.futures.wait(reader_futs)

    assert len(dirty_reads) == 0, f"Dirty reads detected: {dirty_reads[:5]}"

    # Verify 0 rows in database
    with db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 0

    db.close()


# ============================================================================
# 3. BaseException / asyncio.CancelledError Rollback in Database.transaction()
# ============================================================================

@pytest.mark.asyncio
async def test_real_asyncio_task_cancellation_triggers_rollback(tmp_path):
    """Empirical challenge: Spawn a real asyncio.Task running Database.transaction(),

    cancel the task midway at an await point, and verify the transaction is cleanly rolled back.
    """
    db_file = tmp_path / "async_cancel.db"
    db = Database(db_file)

    task_started = asyncio.Event()

    async def async_transaction_worker():
        with db.transaction() as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO signal_snapshots (
                    timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                    drawdown_gate, atr_stop_triggered, rationale, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "2026-09-01T15:50:00+00:00",
                    "BULL_NORMAL",
                    500.0,
                    400.0,
                    0.12,
                    1.0,
                    1.0,
                    0,
                    "TO_BE_CANCELLED",
                    "{}",
                ),
            )
            # Signal that row has been inserted
            task_started.set()
            # Await indefinitely until cancelled
            await asyncio.sleep(60.0)

    task = asyncio.create_task(async_transaction_worker())
    await task_started.wait()

    # Cancel the running task
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Assert row was rolled back and is NOT committed
    with db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 0, "Cancelled async task row was not rolled back!"

    db.close()


def test_in_memory_cancelled_error_prevents_dirty_commit_leakage():
    """Empirical challenge for in-memory Database(':memory:'):

    If asyncio.CancelledError occurs, verify:
    1. The uncommitted dirty row is rolled back.
    2. A SUBSEQUENT transaction on the same shared in-memory connection does NOT
       accidentally commit the dirty row from the cancelled transaction.
    """
    db = Database(":memory:")

    # Transaction 1: Cancelled via asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        with db.transaction() as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO signal_snapshots (
                    timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                    drawdown_gate, atr_stop_triggered, rationale, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "2026-09-01T15:50:00+00:00",
                    "BULL_NORMAL",
                    500.0,
                    400.0,
                    0.12,
                    1.0,
                    1.0,
                    0,
                    "DIRTY_GHOST_ROW",
                    "{}",
                ),
            )
            raise asyncio.CancelledError("Simulated async task cancellation")

    # Transaction 2: Normal valid transaction commits
    with db.transaction() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO signal_snapshots (
                timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                drawdown_gate, atr_stop_triggered, rationale, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-09-01T15:51:00+00:00",
                "BULL_NORMAL",
                501.0,
                401.0,
                0.12,
                1.0,
                1.0,
                0,
                "CLEAN_VALID_ROW",
                "{}",
            ),
        )

    # Verification: EXACTLY 1 row, and it must be CLEAN_VALID_ROW
    with db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 1, "Dirty ghost row leaked into subsequent transaction commit!"

        c.execute("SELECT rationale FROM signal_snapshots;")
        row = c.fetchone()[0]
        assert row == "CLEAN_VALID_ROW"

    db.close()


@pytest.mark.parametrize("exc_cls", [
    asyncio.CancelledError,
    KeyboardInterrupt,
    SystemExit,
])
def test_all_base_exceptions_trigger_rollback_and_mutex_release(exc_cls):
    """Verify that all standard BaseException subclasses (CancelledError,

    KeyboardInterrupt, SystemExit):
    1. Roll back the transaction.
    2. Properly release the in-memory RLock so subsequent transactions proceed without deadlock.
    """
    db = Database(":memory:")

    # Abort with BaseException subclass
    with pytest.raises(exc_cls):
        with db.transaction() as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO signal_snapshots (
                    timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                    drawdown_gate, atr_stop_triggered, rationale, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "2026-09-01T15:50:00+00:00",
                    "BULL_NORMAL",
                    500.0,
                    400.0,
                    0.12,
                    1.0,
                    1.0,
                    0,
                    f"ABORTED_BY_{exc_cls.__name__}",
                    "{}",
                ),
            )
            raise exc_cls()

    # Verify table is empty
    with db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 0

    # Verify lock is completely free and supports new transactions
    with db.transaction() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO signal_snapshots (
                timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                drawdown_gate, atr_stop_triggered, rationale, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-09-01T15:52:00+00:00", "BULL_NORMAL", 500.0, 400.0, 0.12, 1.0, 1.0, 0, "POST_ABORT_OK", "{}"),
        )

    with db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 1

    db.close()


@pytest.mark.asyncio
async def test_concurrent_tasks_cancellation_no_mutex_deadlock():
    """Subject in-memory Database(':memory:') to 10 concurrent asyncio tasks where

    5 tasks are cancelled while inside their transaction, and 5 tasks complete successfully.
    Verify:
    - Zero deadlocks
    - Exactly 5 committed rows
    - Zero cancelled rows committed
    """
    db = Database(":memory:")
    completed_ids: List[int] = []

    async def worker_task(worker_id: int, should_cancel: bool):
        # Stagger slightly
        await asyncio.sleep(worker_id * 0.005)
        with db.transaction() as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO signal_snapshots (
                    timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                    drawdown_gate, atr_stop_triggered, rationale, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"2026-09-01T15:{worker_id:02d}:00+00:00",
                    "BULL_NORMAL",
                    500.0,
                    400.0,
                    0.12,
                    1.0,
                    1.0,
                    0,
                    f"WORKER_{worker_id}",
                    "{}",
                ),
            )
            if should_cancel:
                raise asyncio.CancelledError(f"Task {worker_id} cancelled")
            completed_ids.append(worker_id)

    tasks = []
    for i in range(10):
        should_cancel = (i % 2 == 1)  # Odd workers cancel (1, 3, 5, 7, 9)
        t = asyncio.create_task(worker_task(i, should_cancel))
        tasks.append(t)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 5 cancelled, 5 completed
    cancelled_count = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
    assert cancelled_count == 5
    assert len(completed_ids) == 5

    # Verify DB contents
    with db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 5

        c.execute("SELECT rationale FROM signal_snapshots ORDER BY rationale;")
        rows = [r[0] for r in c.fetchall()]
        expected_rows = [f"WORKER_{i}" for i in [0, 2, 4, 6, 8]]
        assert rows == expected_rows

    db.close()


@pytest.mark.asyncio
async def test_cancelled_error_multi_table_coordinated_rollback(tmp_path):
    """Verify that when asyncio.CancelledError interrupts a multi-table transaction,

    all modified tables (signal_snapshots, allocations, rebalance_orders, market_bars)
    are atomically rolled back together.
    """
    db_file = tmp_path / "multi_table_cancel.db"
    db = Database(db_file)

    async def coordinated_tx():
        with db.transaction() as conn:
            c = conn.cursor()
            # 1. Insert signal
            c.execute(
                """
                INSERT INTO signal_snapshots (
                    timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                    drawdown_gate, atr_stop_triggered, rationale, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("2026-09-01T15:50:00+00:00", "BULL_NORMAL", 500.0, 400.0, 0.12, 1.0, 1.0, 0, "MULTI_TX", "{}"),
            )
            # 2. Insert allocation
            c.execute(
                """
                INSERT INTO allocations (timestamp, regime, weights_json, rationale)
                VALUES (?, ?, ?, ?)
                """,
                ("2026-09-01T15:50:00+00:00", "BULL_NORMAL", '{"SPY": 1.0}', "MULTI_TX"),
            )
            # 3. Insert order
            c.execute(
                """
                INSERT INTO rebalance_orders (id, timestamp, symbol, side, shares, price, notional, target_weight, current_weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("ORD_MULTI_1", "2026-09-01T15:50:00+00:00", "SPY", "BUY", 10.0, 500.0, 5000.0, 1.0, 0.0),
            )
            # 4. Insert bar
            c.execute(
                """
                INSERT INTO market_bars (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("SPY", "1Day", "2026-09-01T15:50:00+00:00", 500.0, 502.0, 498.0, 501.0, 1000000),
            )
            # Cancel before commit
            raise asyncio.CancelledError("Cancellation mid-write storm")

    with pytest.raises(asyncio.CancelledError):
        await coordinated_tx()

    # Verify all 4 tables are empty
    with db.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 0, "signal_snapshots was not rolled back"

        c.execute("SELECT COUNT(*) FROM allocations;")
        assert c.fetchone()[0] == 0, "allocations was not rolled back"

        c.execute("SELECT COUNT(*) FROM rebalance_orders;")
        assert c.fetchone()[0] == 0, "rebalance_orders was not rolled back"

        c.execute("SELECT COUNT(*) FROM market_bars;")
        assert c.fetchone()[0] == 0, "market_bars was not rolled back"

    db.close()
