"""
tests.adversarial.test_r1_concurrency_faults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone C-M1 Adversarial Concurrency & Persistence Fault Injection Test Suite:
1. 20+ concurrent workers write storm against SQLite WAL with 0 lock errors and 0 corruption.
2. JSONL audit logger handles raw non-UTF8 binary corruption without crashing.
3. JSONL audit logger recovers clean records after un-flushed partial line append.
4. In-memory SQLite (:memory:) concurrency with 20 writer threads and zero errors.
5. Daemon scheduler retries DAILY_CLOSE evaluation on persistence failures.
6. Atomic rollback across signals, allocations, and regime tables in DecisionDaemon.
7. BaseException / asyncio.CancelledError rollback safety in Database.transaction().
"""

import asyncio
import concurrent.futures
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import List
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


# ============================================================================
# 1. 20+ Concurrent Workers Write Storm against SQLite WAL
# ============================================================================

def test_sqlite_wal_20_concurrent_workers_stress(tmp_path):
    """Subject SQLite WAL to 20 concurrent writer threads + 5 reader threads (25 workers total).

    Assert 0 database lock errors, 0 data loss, and 100% database integrity.
    """
    db_file = tmp_path / "wal_stress_20w.db"
    db = Database(db_file)

    # Verify WAL mode is configured
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"

    sig_repo = SignalSnapshotRepository(db)
    alloc_repo = AllocationRepository(db)
    order_repo = RebalanceOrderRepository(db)

    num_signal_writers = 8
    num_alloc_writers = 7
    num_order_writers = 5
    num_readers = 5
    iterations_per_writer = 50

    base_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)

    def write_signals(worker_id: int):
        for i in range(iterations_per_writer):
            ts = base_time + timedelta(seconds=worker_id * 1000 + i)
            sig = SignalSnapshot(
                timestamp=ts,
                spy_price=500.0 + i * 0.1,
                spy_sma50=495.0,
                spy_sma200=480.0,
                realized_vol_20d=0.12,
                vol_scale_factor=1.0,
                drawdown_pct=-0.02,
                circuit_breaker_active=False,
                regime=MarketRegime.BULL_NORMAL,
                indicators={"worker": float(worker_id), "iter": float(i)},
            )
            sig_repo.save(sig, rationale=f"Signal_w{worker_id}_i{i}")

    def write_allocations(worker_id: int):
        for i in range(iterations_per_writer):
            ts = base_time + timedelta(seconds=worker_id * 1000 + i)
            alloc = TargetAllocation(
                timestamp=ts,
                regime=MarketRegime.BULL_AGGRESSIVE,
                weights={"SPY": 0.6, "QQQ": 0.4},
                cash_weight=0.0,
                rationale=f"Alloc_w{worker_id}_i{i}",
            )
            alloc_repo.save(alloc)

    def write_orders(worker_id: int):
        for i in range(iterations_per_writer):
            ts = base_time + timedelta(seconds=worker_id * 1000 + i)
            order = OrderIntent(
                symbol="SPY",
                side=OrderSide.BUY,
                action="BUY",
                target_shares=10.0,
                delta_shares=10.0,
                estimated_price=500.0,
                target_weight=0.6,
                current_weight=0.5,
                delta_weight=0.1,
                timestamp=ts,
                rationale=f"Order_w{worker_id}_i{i}",
            )
            order_repo.save_batch([order], status="TEST_SUBMITTED")

    def read_records(reader_id: int):
        for _ in range(iterations_per_writer):
            sig = sig_repo.get_latest()
            alloc = alloc_repo.get_latest()
            orders = order_repo.get_by_status("TEST_SUBMITTED")
            assert isinstance(orders, list)

    workers = []
    # 8 signal writers
    for w in range(num_signal_writers):
        workers.append(("signal", w, write_signals))
    # 7 allocation writers
    for w in range(num_alloc_writers):
        workers.append(("alloc", w, write_allocations))
    # 5 order writers
    for w in range(num_order_writers):
        workers.append(("order", w, write_orders))
    # 5 readers
    for r in range(num_readers):
        workers.append(("reader", r, read_records))

    assert len(workers) == 25, f"Expected 25 concurrent threads, got {len(workers)}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        futures = {executor.submit(fn, wid): (tag, wid) for tag, wid, fn in workers}
        for fut in concurrent.futures.as_completed(futures):
            tag, wid = futures[fut]
            try:
                fut.result()
            except Exception as e:
                pytest.fail(f"Worker {tag}_{wid} failed with unexpected exception: {type(e).__name__}: {e}")

    # Database integrity check
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        status = cursor.fetchone()[0]
        assert status == "ok", f"Integrity check failed: {status}"

        # Verify exact write counts
        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
        actual_signals = cursor.fetchone()[0]
        expected_signals = num_signal_writers * iterations_per_writer
        assert actual_signals == expected_signals, f"Signal count mismatch: expected {expected_signals}, got {actual_signals}"

        cursor.execute("SELECT COUNT(*) FROM allocations;")
        actual_allocs = cursor.fetchone()[0]
        expected_allocs = num_alloc_writers * iterations_per_writer
        assert actual_allocs == expected_allocs, f"Allocation count mismatch: expected {expected_allocs}, got {actual_allocs}"

        cursor.execute("SELECT COUNT(*) FROM rebalance_orders;")
        actual_orders = cursor.fetchone()[0]
        expected_orders = num_order_writers * iterations_per_writer
        assert actual_orders == expected_orders, f"Order count mismatch: expected {expected_orders}, got {actual_orders}"

    db.close()


# ============================================================================
# 2. JSONL Audit Logger Non-UTF8 Binary Corruption Resilience
# ============================================================================

def test_jsonl_non_utf8_binary_corruption_tolerance(tmp_path):
    """Verify JSONLAuditLogger handles raw non-UTF8 binary corruption without crashing.

    Tests read_date, read_decisions, read_latest, and find_decision on corrupted log files.
    """
    log_dir = tmp_path / "corrupt_audit_logs"
    logger = JSONLAuditLogger(log_dir)
    target_date = "2026-09-02"
    log_file = log_dir / f"decisions_{target_date}.jsonl"

    t1 = datetime(2026, 9, 2, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 2, 14, 0, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 9, 2, 15, 50, 0, tzinfo=timezone.utc)

    # 1. Log clean record 1
    did1 = logger.log_rebalance_decision(
        trigger="MORNING_EVAL",
        regime=MarketRegime.BULL_NORMAL,
        timestamp=t1,
        rationale="Clean Decision 1",
    )

    # 2. Inject corrupted non-UTF8 bytes directly into the file
    with open(log_file, "ab") as f:
        # 0x80, 0xFF, 0xFE are invalid UTF-8 start/continuation bytes
        f.write(b"\x80\x81\xff\xfe\xca\xfe corrupted binary stream\n")
        f.write(b"\xc3\x28 bad 2-byte utf8 sequence\n")
        f.write(b"NOT_JSON_AT_ALL\n")

    # 3. Log clean record 2
    did2 = logger.log_rebalance_decision(
        trigger="MIDDAY_EVAL",
        regime=MarketRegime.BULL_AGGRESSIVE,
        timestamp=t2,
        rationale="Clean Decision 2",
    )

    # 4. Inject more corrupted non-UTF8 binary lines
    with open(log_file, "ab") as f:
        f.write(b"\xf0\x28\x8c\x28 bad 4-byte sequence\n")

    # 5. Log clean record 3
    did3 = logger.log_rebalance_decision(
        trigger="DAILY_CLOSE",
        regime=MarketRegime.CORRECTION_FRAGILE,
        timestamp=t3,
        rationale="Clean Decision 3",
    )

    # Verify read_date does not raise UnicodeDecodeError and returns all 3 clean records
    records_date = logger.read_date(target_date)
    assert len(records_date) == 3, f"Expected 3 recovered records, got {len(records_date)}"
    assert [r["decision_id"] for r in records_date] == [did1, did2, did3]
    assert records_date[0]["regime"] == "BULL_NORMAL"
    assert records_date[1]["regime"] == "BULL_AGGRESSIVE"
    assert records_date[2]["regime"] == "CORRECTION_FRAGILE"

    # Verify read_decisions alias works identically
    records_alias = logger.read_decisions(target_date)
    assert records_alias == records_date

    # Verify read_latest returns newest first without crashing on corrupt bytes
    latest = logger.read_latest(limit=10)
    assert len(latest) == 3
    assert [r["decision_id"] for r in latest] == [did3, did2, did1]

    # Verify find_decision locates each record and gracefully returns None for missing
    assert logger.find_decision(did1) is not None
    assert logger.find_decision(did1)["decision_id"] == did1
    assert logger.find_decision(did2)["decision_id"] == did2
    assert logger.find_decision(did3)["decision_id"] == did3
    assert logger.find_decision("nonexistent_id") is None


# ============================================================================
# 3. JSONL Un-flushed Partial Line Append Recovery
# ============================================================================

def test_jsonl_unflushed_partial_line_newline_recovery(tmp_path):
    """Verify that when a prior write was truncated mid-stream without a newline,

    subsequent calls to log_rebalance_decision write a newline first so the new
    record does not fuse with the corrupt fragment.
    """
    log_dir = tmp_path / "partial_logs"
    logger = JSONLAuditLogger(log_dir)
    target_date = "2026-09-03"
    log_file = log_dir / f"decisions_{target_date}.jsonl"

    # Simulate an unflushed truncated partial write without trailing newline
    partial_bytes = b'{"version": "1.0", "decision_id": "truncated_001", "timestamp": "2026-09-03T14:00:00Z", "rationale": "half wr'
    with open(log_file, "wb") as f:
        f.write(partial_bytes)

    # Immediately append a new valid decision via logger
    t_valid = datetime(2026, 9, 3, 15, 50, 0, tzinfo=timezone.utc)
    did_valid = logger.log_rebalance_decision(
        trigger="DAILY_CLOSE",
        regime=MarketRegime.BULL_NORMAL,
        timestamp=t_valid,
        rationale="Clean decision after power cut",
    )

    # Inspect file content directly: valid record must not be fused to truncated line
    with open(log_file, "rb") as f:
        raw_lines = f.readlines()

    assert len(raw_lines) == 2, f"Expected 2 separate lines, got {len(raw_lines)}: {raw_lines}"
    assert raw_lines[0].startswith(b'{"version": "1.0", "decision_id": "truncated_001"')
    assert raw_lines[0].endswith(b"\n")
    assert raw_lines[1].startswith(b'{"allocations":') or raw_lines[1].startswith(b'{"')
    assert raw_lines[1].endswith(b"\n")

    # Verify reader recovers the clean valid record while discarding the partial line
    recovered = logger.read_date(target_date)
    assert len(recovered) == 1, f"Expected 1 recovered record, got {len(recovered)}"
    assert recovered[0]["decision_id"] == did_valid
    assert recovered[0]["rationale"] == "Clean decision after power cut"

    # Verify find_decision finds the valid decision
    found = logger.find_decision(did_valid)
    assert found is not None
    assert found["decision_id"] == did_valid


# ============================================================================
# 4. In-Memory SQLite (:memory:) 20 Concurrent Writer Threads
# ============================================================================

def test_in_memory_database_20_concurrent_writers_zero_errors():
    """Verify Database(':memory:') protected by threading.RLock handles 20 concurrent

    writer threads without 'cannot commit - no transaction is active' or race conditions.
    """
    db = Database(":memory:")

    num_threads = 20
    writes_per_thread = 50

    def writer_task(worker_id: int):
        for i in range(writes_per_thread):
            ts = f"2026-09-01T10:{worker_id:02d}:{i:02d}+00:00"
            with db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO signal_snapshots (
                        timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                        drawdown_gate, atr_stop_triggered, rationale, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ts, "BULL_NORMAL", 500.0, 400.0, 0.12, 1.0, 1.0, 0, f"w{worker_id}_i{i}", "{}"),
                )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(writer_task, wid) for wid in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                pytest.fail(f"Concurrent in-memory writer thread failed: {type(e).__name__}: {e}")

    # Verify row count exactly equals total transactions
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
        total_rows = cursor.fetchone()[0]
        assert total_rows == num_threads * writes_per_thread, (
            f"Expected {num_threads * writes_per_thread} rows, got {total_rows}"
        )

    db.close()


# ============================================================================
# 5. Daemon Scheduler Retrying on Persistence Failure
# ============================================================================

@pytest.mark.asyncio
async def test_scheduler_retry_on_persistence_failure():
    """Verify MarketScheduler does not mark CadenceType.DAILY_CLOSE as executed for

    today when a handler raises an exception, allowing retry on subsequent ticks.
    """
    cal = MarketCalendar()
    sched = MarketScheduler(calendar=cal, daily_close_offset_minutes=10)

    # Wednesday Sep 2, 2026 at 15:52 ET (within daily close eval window)
    t_eval = datetime(2026, 9, 2, 15, 52, tzinfo=ET)
    today = t_eval.date()

    call_count = 0

    def flaky_handler(dt: datetime):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            # Simulate transient persistence lock failure
            raise sqlite3.OperationalError("Simulated locked database during persistence")
        # 3rd attempt succeeds

    sched.on_cadence(CadenceType.DAILY_CLOSE, flaky_handler)

    # Tick 1: Handler fails on attempt 1
    trig1 = await sched.tick(t_eval)
    assert CadenceType.DAILY_CLOSE not in trig1
    assert sched._last_executed[CadenceType.DAILY_CLOSE] is None, (
        "DAILY_CLOSE must NOT be marked executed after handler failure"
    )
    assert call_count == 1

    # Tick 2: Handler fails on attempt 2 (retry on next tick before close)
    t_eval_2 = datetime(2026, 9, 2, 15, 53, tzinfo=ET)
    trig2 = await sched.tick(t_eval_2)
    assert CadenceType.DAILY_CLOSE not in trig2
    assert sched._last_executed[CadenceType.DAILY_CLOSE] is None
    assert call_count == 2

    # Tick 3: Handler succeeds on attempt 3
    t_eval_3 = datetime(2026, 9, 2, 15, 54, tzinfo=ET)
    trig3 = await sched.tick(t_eval_3)
    assert CadenceType.DAILY_CLOSE in trig3
    assert sched._last_executed[CadenceType.DAILY_CLOSE] == today
    assert call_count == 3

    # Tick 4: Subsequent tick on same day is idempotent and does not re-run
    t_eval_4 = datetime(2026, 9, 2, 15, 55, tzinfo=ET)
    trig4 = await sched.tick(t_eval_4)
    assert len(trig4) == 0
    assert call_count == 3


# ============================================================================
# 6. Atomic Persistence Rollback in DecisionDaemon
# ============================================================================

@pytest.mark.asyncio
async def test_daemon_handle_daily_close_atomic_persistence_rollback(tmp_path):
    """Verify that if an error occurs while saving to SQLite inside _handle_daily_close,

    the entire transaction is rolled back so no partial records remain committed.
    """
    db_path = tmp_path / "atomic_daemon.db"
    log_dir = tmp_path / "logs"
    config = DaemonConfig(db_path=str(db_path), log_dir=str(log_dir), dry_run=True)
    storage = StorageService(db_path)
    audit_logger = JSONLAuditLogger(log_dir)

    daemon = DecisionDaemon(config=config, storage=storage, audit_logger=audit_logger)
    await daemon.client.state_machine.check_initial_health({"upstream": "connected"})

    # Populate dummy market bars
    base_t = datetime(2026, 9, 1, 16, 0, tzinfo=timezone.utc)
    for sym, start_p in [("SPY", 450.0), ("QQQ", 380.0)]:
        bars = [
            Bar(
                symbol=sym,
                timestamp=base_t - timedelta(days=250 - i),
                open=start_p + i * 0.1,
                high=start_p + i * 0.1 + 1.0,
                low=start_p + i * 0.1 - 1.0,
                close=start_p + i * 0.1 + 0.5,
                volume=1000000,
            )
            for i in range(250)
        ]
        daemon._cached_daily_bars[sym] = bars

    eval_dt = datetime(2026, 9, 2, 15, 50, tzinfo=ET)

    # Monkeypatch storage.allocations.save to raise an exception mid-transaction
    orig_save = storage.allocations.save

    def failing_alloc_save(*args, **kwargs):
        raise sqlite3.OperationalError("Simulated allocation table write failure")

    storage.allocations.save = failing_alloc_save

    with pytest.raises(sqlite3.OperationalError, match="Simulated allocation table write failure"):
        await daemon._handle_daily_close(eval_dt)

    # Verify atomic rollback: zero signals, zero allocations, and zero regime events
    with storage.db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert cursor.fetchone()[0] == 0, "Signal snapshots should have been rolled back"
        cursor.execute("SELECT COUNT(*) FROM allocations;")
        assert cursor.fetchone()[0] == 0, "Allocations should have been rolled back"
        cursor.execute("SELECT COUNT(*) FROM regime_events;")
        assert cursor.fetchone()[0] == 0, "Regime events should have been rolled back"

    # Restore original method and verify clean execution
    storage.allocations.save = orig_save
    await daemon._handle_daily_close(eval_dt)

    with storage.db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT COUNT(*) FROM allocations;")
        assert cursor.fetchone()[0] == 1

    await daemon.shutdown("Test complete")


# ============================================================================
# 7. BaseException / asyncio.CancelledError Rollback Safety in Database
# ============================================================================

def test_database_transaction_cancelleterror_triggers_rollback(tmp_path):
    """Verify that asyncio.CancelledError (which inherits from BaseException)

    triggers conn.rollback() in Database.transaction().
    """
    db_file = tmp_path / "cancel_test.db"
    db = Database(db_file)

    # Attempt a transaction that gets cancelled via asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO signal_snapshots (
                    timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                    drawdown_gate, atr_stop_triggered, rationale, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("2026-09-01T15:50:00+00:00", "BULL_NORMAL", 500.0, 400.0, 0.12, 1.0, 1.0, 0, "Cancelled", "{}"),
            )
            # Simulate asyncio.CancelledError
            raise asyncio.CancelledError("Simulated task cancellation")

    # Verify row was cleanly rolled back
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert cursor.fetchone()[0] == 0, "Cancelled transaction row should be rolled back"

    db.close()
