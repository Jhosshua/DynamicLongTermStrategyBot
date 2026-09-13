"""
tests.adversarial.test_cm1_challenger_stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone C-M1 Empirical Challenger Stress Test Suite:
1. SQLite WAL 30-worker write/read storm (20 writers + 10 readers) with zero lock errors and 100% integrity.
2. In-memory SQLite (:memory:) 30 concurrent threads with interleaved transaction rollbacks and zero state leakage.
3. JSONL audit logger extreme corruption injection (un-flushed partial lines, raw non-UTF8 binary, non-dict JSON, malformed tokens).
4. JSONL concurrent writer storm (25 threads, 1,000 records) interleaved with raw binary corruption injection.
5. Multi-table coordinated transaction atomicity under simulated persistence failure.
"""

import asyncio
import concurrent.futures
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List
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


# ============================================================================
# 1. SQLite WAL 30-Worker High-Contention Stress Storm
# ============================================================================

def test_wal_30_concurrent_threads_write_read_storm(tmp_path):
    """Subject SQLite WAL to 30 concurrent threads:

    - 5 signal writers (50 writes each = 250)
    - 5 allocation writers (50 writes each = 250)
    - 5 order writers (50 writes each = 250)
    - 5 bar writers (50 writes each = 250)
    - 10 concurrent readers querying latest and range queries
    Total writes: 1,000 writes across 4 tables with 10 continuous readers.
    Assert: 0 database lock errors, PRAGMA integrity_check == ok, exact row counts.
    """
    db_file = tmp_path / "wal_stress_30w.db"
    db = Database(db_file)

    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal", f"Database not in WAL mode: {mode}"

    sig_repo = SignalSnapshotRepository(db)
    alloc_repo = AllocationRepository(db)
    order_repo = RebalanceOrderRepository(db)
    bar_repo = MarketBarRepository(db)

    iterations_per_writer = 50
    base_time = datetime(2026, 9, 1, 9, 30, 0, tzinfo=timezone.utc)

    lock_errors: List[str] = []
    other_errors: List[str] = []

    def write_signals(worker_id: int):
        for i in range(iterations_per_writer):
            ts = base_time + timedelta(seconds=worker_id * 1000 + i)
            sig = SignalSnapshot(
                timestamp=ts,
                spy_price=510.0 + i * 0.05,
                spy_sma50=505.0,
                spy_sma200=490.0,
                realized_vol_20d=0.14,
                vol_scale_factor=0.95,
                drawdown_pct=-0.015,
                circuit_breaker_active=False,
                regime=MarketRegime.BULL_NORMAL,
                indicators={"worker": float(worker_id), "iter": float(i)},
            )
            try:
                sig_repo.save(sig, rationale=f"WAL_Sig_w{worker_id}_i{i}")
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    lock_errors.append(f"Signal w{worker_id}: {e}")
                else:
                    other_errors.append(f"Signal w{worker_id}: {e}")
            except Exception as e:
                other_errors.append(f"Signal w{worker_id}: {e}")

    def write_allocations(worker_id: int):
        for i in range(iterations_per_writer):
            ts = base_time + timedelta(seconds=worker_id * 1000 + i)
            alloc = TargetAllocation(
                timestamp=ts,
                regime=MarketRegime.BULL_AGGRESSIVE,
                weights={"SPY": 0.5, "QQQ": 0.5},
                cash_weight=0.0,
                rationale=f"WAL_Alloc_w{worker_id}_i{i}",
            )
            try:
                alloc_repo.save(alloc)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    lock_errors.append(f"Alloc w{worker_id}: {e}")
                else:
                    other_errors.append(f"Alloc w{worker_id}: {e}")
            except Exception as e:
                other_errors.append(f"Alloc w{worker_id}: {e}")

    def write_orders(worker_id: int):
        for i in range(iterations_per_writer):
            ts = base_time + timedelta(seconds=worker_id * 1000 + i)
            order = OrderIntent(
                symbol="QQQ",
                side=OrderSide.BUY,
                action="BUY",
                target_shares=20.0,
                delta_shares=5.0,
                estimated_price=450.0,
                target_weight=0.5,
                current_weight=0.4,
                delta_weight=0.1,
                timestamp=ts,
                rationale=f"WAL_Order_w{worker_id}_i{i}",
            )
            try:
                order_repo.save_batch([order], status="FILLED")
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    lock_errors.append(f"Order w{worker_id}: {e}")
                else:
                    other_errors.append(f"Order w{worker_id}: {e}")
            except Exception as e:
                other_errors.append(f"Order w{worker_id}: {e}")

    def write_bars(worker_id: int):
        for i in range(iterations_per_writer):
            ts = base_time + timedelta(minutes=worker_id * 100 + i)
            bar = Bar(
                symbol=f"SYM_{worker_id}",
                timestamp=ts,
                open=100.0 + i,
                high=105.0 + i,
                low=99.0 + i,
                close=102.0 + i,
                volume=10000 + i,
            )
            try:
                bar_repo.save_bars([bar], timeframe="1Day")
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    lock_errors.append(f"Bar w{worker_id}: {e}")
                else:
                    other_errors.append(f"Bar w{worker_id}: {e}")
            except Exception as e:
                other_errors.append(f"Bar w{worker_id}: {e}")

    def read_queries(reader_id: int):
        for _ in range(iterations_per_writer):
            try:
                sig = sig_repo.get_latest()
                alloc = alloc_repo.get_latest()
                orders = order_repo.get_by_status("FILLED")
                assert isinstance(orders, list)
                time.sleep(0.001)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    lock_errors.append(f"Reader r{reader_id}: {e}")
                else:
                    other_errors.append(f"Reader r{reader_id}: {e}")
            except Exception as e:
                other_errors.append(f"Reader r{reader_id}: {e}")

    threads = []
    for w in range(5):
        threads.append(("sig_w", w, write_signals))
    for w in range(5):
        threads.append(("alloc_w", w, write_allocations))
    for w in range(5):
        threads.append(("order_w", w, write_orders))
    for w in range(5):
        threads.append(("bar_w", w, write_bars))
    for r in range(10):
        threads.append(("reader", r, read_queries))

    assert len(threads) == 30, f"Expected 30 threads, got {len(threads)}"

    start_t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futs = [executor.submit(fn, wid) for tag, wid, fn in threads]
        concurrent.futures.wait(futs)
    elapsed = time.perf_counter() - start_t

    # Assert 0 lock errors
    assert len(lock_errors) == 0, f"Encountered {len(lock_errors)} lock errors: {lock_errors[:5]}"
    assert len(other_errors) == 0, f"Encountered {len(other_errors)} unexpected errors: {other_errors[:5]}"

    # Verify integrity and row counts
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        res = cursor.fetchone()[0]
        assert res == "ok", f"Integrity check failed: {res}"

        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert cursor.fetchone()[0] == 250, "Expected 250 signals"

        cursor.execute("SELECT COUNT(*) FROM allocations;")
        assert cursor.fetchone()[0] == 250, "Expected 250 allocations"

        cursor.execute("SELECT COUNT(*) FROM rebalance_orders;")
        assert cursor.fetchone()[0] == 250, "Expected 250 orders"

        cursor.execute("SELECT COUNT(*) FROM market_bars;")
        assert cursor.fetchone()[0] == 250, "Expected 250 bars"

    total_ops = 1000 + (10 * iterations_per_writer)
    throughput = total_ops / elapsed
    print(f"\n[WAL 30-Worker Stress] Completed {total_ops} operations in {elapsed:.2f}s ({throughput:.1f} ops/sec)")
    db.close()


# ============================================================================
# 2. In-Memory SQLite (:memory:) 30 Concurrent Workers with Rollback Safety
# ============================================================================

def test_in_memory_sqlite_30_concurrent_threads_with_rollbacks():
    """Verify Database(':memory:') handles 30 concurrent threads where:

    - 20 threads execute successful transactions (60 writes each = 1,200 committed rows)
    - 5 threads execute failing transactions that trigger BaseException rollback (60 aborts each = 300 aborts)
    - 5 threads perform continuous read queries
    Assert:
    - 0 mutex deadlocks or 'cannot commit - no transaction is active' errors
    - Exactly 1,200 rows committed in database
    - 0 aborted rows present in database
    - PRAGMA integrity_check returns 'ok'
    """
    db = Database(":memory:")
    iterations = 60
    lock_errors: List[str] = []
    other_errors: List[str] = []

    def successful_writer(worker_id: int):
        for i in range(iterations):
            ts = f"2026-09-01T11:{worker_id:02d}:{i:02d}+00:00"
            try:
                with db.transaction() as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(
                            """
                            INSERT INTO signal_snapshots (
                                timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                                drawdown_gate, atr_stop_triggered, rationale, raw_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (ts, "BULL_NORMAL", 520.0, 410.0, 0.11, 1.0, 1.0, 0, f"SUCCESS_w{worker_id}_i{i}", "{}"),
                        )
                    finally:
                        cursor.close()
            except Exception as e:
                other_errors.append(f"Writer w{worker_id}: {e}")

    def failing_writer(worker_id: int):
        for i in range(iterations):
            ts = f"2026-09-01T12:{worker_id:02d}:{i:02d}+00:00"
            try:
                with db.transaction() as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(
                            """
                            INSERT INTO signal_snapshots (
                                timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
                                drawdown_gate, atr_stop_triggered, rationale, raw_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (ts, "BEAR_CRISIS", 400.0, 300.0, 0.40, 0.2, 0.5, 1, f"FAIL_w{worker_id}_i{i}", "{}"),
                        )
                        raise RuntimeError(f"Simulated abort for worker {worker_id}")
                    finally:
                        cursor.close()
            except RuntimeError:
                pass  # Expected rollback
            except Exception as e:
                other_errors.append(f"Failing writer w{worker_id}: {e}")

    def reader_task(reader_id: int):
        for _ in range(iterations):
            try:
                with db.transaction() as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
                        row = cursor.fetchone()
                        _ = row[0]
                    finally:
                        cursor.close()
                time.sleep(0.0005)
            except Exception as e:
                other_errors.append(f"Reader r{reader_id}: {e}")

    tasks = []
    # 20 successful writers
    for w in range(20):
        tasks.append(("good_w", w, successful_writer))
    # 5 failing writers
    for w in range(5):
        tasks.append(("bad_w", w, failing_writer))
    # 5 readers
    for r in range(5):
        tasks.append(("reader", r, reader_task))

    assert len(tasks) == 30, f"Expected 30 threads, got {len(tasks)}"

    start_t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futs = [executor.submit(fn, wid) for tag, wid, fn in tasks]
        concurrent.futures.wait(futs)
    elapsed = time.perf_counter() - start_t

    assert len(other_errors) == 0, f"Encountered unexpected errors in in-memory stress: {other_errors[:5]}"

    # Verify committed count and zero aborted records
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        assert cursor.fetchone()[0] == "ok"

        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
        total_rows = cursor.fetchone()[0]
        assert total_rows == 20 * iterations, (
            f"Expected {20 * iterations} rows, got {total_rows}"
        )

        cursor.execute("SELECT COUNT(*) FROM signal_snapshots WHERE rationale LIKE 'FAIL_%';")
        failed_rows = cursor.fetchone()[0]
        assert failed_rows == 0, f"Found {failed_rows} unrolled-back failed records!"

    print(f"\n[In-Memory 30-Worker Stress] Completed 1,500 operations in {elapsed:.2f}s")
    db.close()


# ============================================================================
# 3. JSONL Extreme Corruption Injection & Fault Recovery
# ============================================================================

def test_jsonl_extreme_corruption_and_partial_write_injection(tmp_path):
    """Adversarially inject multiple corruption variants into JSONL:

    - Raw invalid non-UTF8 bytes (0x80, 0xFF, 0xFE, bad multibyte sequences)
    - Truncated partial JSON line lacking trailing newline
    - Truncated JSON line with trailing newline (unclosed brace, missing value)
    - Valid JSON lines containing non-dict types (int, string, list, boolean, null)
    - Corrupted syntax tokens (unquoted keys, trailing commas)
    - Blank lines, whitespace-only lines, NUL byte sequences (\x00\x00)
    Verify:
    - Subsequent write cleanly appends without fusing to partial line
    - read_date skips corrupted lines and returns exactly all valid records
    - read_decisions alias returns identical records
    - read_latest recovers valid records sorted descending by timestamp
    - find_decision finds existing records and returns None for corrupted/missing
    - Zero unhandled exceptions
    """
    log_dir = tmp_path / "extreme_corrupt_logs"
    logger = JSONLAuditLogger(log_dir)
    target_date = "2026-09-04"
    log_file = log_dir / f"decisions_{target_date}.jsonl"

    t1 = datetime(2026, 9, 4, 9, 30, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 9, 4, 15, 50, 0, tzinfo=timezone.utc)

    # 1. Log first valid record
    did1 = logger.log_rebalance_decision(
        trigger="OPEN_EVAL",
        regime=MarketRegime.BULL_NORMAL,
        timestamp=t1,
        rationale="Clean Valid Decision 1",
    )

    # 2. Inject adversarial corruptions:
    with open(log_file, "ab") as f:
        # (a) Raw invalid UTF-8 bytes
        f.write(b"\x80\x81\xff\xfe\xca\xfe\xba\xbe raw binary noise\n")
        # (b) Illegal overlong UTF-8 sequence
        f.write(b"\xc0\xaf illegal overlong slash\n")
        # (c) NUL bytes
        f.write(b"\x00\x00\x00\x00\n")
        # (d) Valid JSON but non-dict values
        f.write(b"[1, 2, 3, \"not a dict\"]\n")
        f.write(b"\"just a standalone string\"\n")
        f.write(b"123456789\n")
        f.write(b"null\n")
        f.write(b"true\n")
        # (e) Corrupted JSON syntax
        f.write(b"{key_without_quotes: true}\n")
        f.write(b"{\"trailing_comma\": 1,}\n")
        f.write(b"{\"unclosed_brace\": \"broken\"\n")
        # (f) Whitespace / blank lines
        f.write(b"   \n")
        f.write(b"\t\t\n")
        f.write(b"\n")
        # (g) Truncated partial line WITHOUT trailing newline (power failure simulation)
        f.write(b'{"version": "1.0", "decision_id": "truncated_power_loss", "timestamp": "2026-09-04T11:00:00Z", "rationale": "half wr')

    # 3. Append second valid record via logger
    # Logger MUST detect missing trailing newline, inject newline, and preserve did2 cleanly
    did2 = logger.log_rebalance_decision(
        trigger="MIDDAY_EVAL",
        regime=MarketRegime.BULL_AGGRESSIVE,
        timestamp=t2,
        rationale="Clean Valid Decision 2",
    )

    # 4. Inject another unclosed raw binary truncated fragment WITHOUT newline
    with open(log_file, "ab") as f:
        f.write(b"\xfe\xed\xfa\xce{\"partially_written_binary\": \xff\xff")

    # 5. Append third valid record via logger
    did3 = logger.log_rebalance_decision(
        trigger="DAILY_CLOSE",
        regime=MarketRegime.CORRECTION_FRAGILE,
        timestamp=t3,
        rationale="Clean Valid Decision 3",
    )

    # Verify read_date returns all 3 valid decisions cleanly
    records = logger.read_date(target_date)
    assert len(records) == 3, f"Expected exactly 3 valid recovered records, got {len(records)}: {records}"
    assert [r["decision_id"] for r in records] == [did1, did2, did3]
    assert records[0]["rationale"] == "Clean Valid Decision 1"
    assert records[1]["rationale"] == "Clean Valid Decision 2"
    assert records[2]["rationale"] == "Clean Valid Decision 3"

    # Verify read_decisions alias
    assert logger.read_decisions(target_date) == records

    # Verify read_latest returns descending by timestamp
    latest = logger.read_latest(limit=10)
    assert len(latest) == 3
    assert [r["decision_id"] for r in latest] == [did3, did2, did1]

    # Verify find_decision finds all 3
    assert logger.find_decision(did1)["decision_id"] == did1
    assert logger.find_decision(did2)["decision_id"] == did2
    assert logger.find_decision(did3)["decision_id"] == did3
    assert logger.find_decision("nonexistent_uuid") is None
    assert logger.find_decision("truncated_power_loss") is None


# ============================================================================
# 4. JSONL 25-Worker Concurrent Write Storm with Raw Binary Corruptors
# ============================================================================

def test_jsonl_concurrent_writer_storm_with_adversarial_injection(tmp_path):
    """Stress JSONLAuditLogger with 25 concurrent writer threads + 5 concurrent raw corruptor threads:

    - 25 writer threads write 40 valid decisions each = 1,000 valid records
    - 5 corruptor threads append raw binary gibberish concurrently to the same log file
    Assert:
    - 0 unhandled exceptions or logger crashes
    - All 1,000 valid decisions are safely appended and recoverable by read_date
    """
    log_dir = tmp_path / "concurrent_corrupt_logs"
    logger = JSONLAuditLogger(log_dir)
    target_date = "2026-09-04"
    log_file = log_dir / f"decisions_{target_date}.jsonl"

    num_writers = 25
    writes_per_thread = 40
    total_valid = num_writers * writes_per_thread

    written_ids: List[str] = []
    id_lock = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    base_t = datetime(2026, 9, 4, 8, 0, 0, tzinfo=timezone.utc)

    def writer_task(worker_id: int):
        local_ids = []
        for i in range(writes_per_thread):
            t = base_t + timedelta(seconds=worker_id * 100 + i)
            did = logger.log_rebalance_decision(
                trigger="STRESS_WRITER",
                regime=MarketRegime.BULL_NORMAL,
                timestamp=t,
                rationale=f"w{worker_id}_i{i}",
            )
            local_ids.append(did)
        with logger._lock:
            written_ids.extend(local_ids)

    def corruptor_task(corruptor_id: int):
        for _ in range(20):
            corrupt_bytes = f"\x80\xff\xfe CORRUPT_{corruptor_id} \n".encode("latin-1")
            with logger._lock:
                with open(log_file, "ab") as f:
                    f.write(corrupt_bytes)
            time.sleep(0.002)

    threads = []
    for w in range(num_writers):
        threads.append(("writer", w, writer_task))
    for c in range(5):
        threads.append(("corruptor", c, corruptor_task))

    start_t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futs = [executor.submit(fn, wid) for tag, wid, fn in threads]
        concurrent.futures.wait(futs)
    elapsed = time.perf_counter() - start_t

    assert len(written_ids) == total_valid, f"Expected {total_valid} logged IDs, got {len(written_ids)}"

    # Recover all valid decisions
    recovered = logger.read_date(target_date)
    assert len(recovered) == total_valid, (
        f"Expected to recover all {total_valid} valid records, but recovered {len(recovered)}"
    )

    recovered_ids = {r["decision_id"] for r in recovered}
    assert recovered_ids == set(written_ids), "Recovered IDs do not match written IDs!"

    print(f"\n[JSONL Concurrent Stress] Successfully wrote and recovered {total_valid} records in {elapsed:.2f}s")


# ============================================================================
# 5. Coordinated Multi-Table Transaction Atomicity under Simulated Persistence Faults
# ============================================================================

def test_coordinated_transaction_atomicity_under_faults(tmp_path):
    """Verify that StorageService.save_daily_close executes in a single transaction

    and guarantees all-or-nothing atomicity across signal_snapshots, allocations,
    and regime_events when a simulated error occurs midway.
    """
    db_file = tmp_path / "atomicity.db"
    storage = StorageService(db_file)

    t = datetime(2026, 9, 4, 15, 50, 0, tzinfo=timezone.utc)
    sig = SignalSnapshot(
        timestamp=t,
        spy_price=500.0,
        spy_sma50=495.0,
        spy_sma200=480.0,
        realized_vol_20d=0.12,
        vol_scale_factor=1.0,
        drawdown_pct=-0.01,
        circuit_breaker_active=False,
        regime=MarketRegime.BULL_NORMAL,
        indicators={"atr": 5.0},
    )
    alloc = TargetAllocation(
        timestamp=t,
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 1.0},
        cash_weight=0.0,
        rationale="Normal bull",
    )

    # 1. Normal save works atomically
    storage.save_daily_close(sig, alloc, t, "Normal bull")

    with storage.db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT COUNT(*) FROM allocations;")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT COUNT(*) FROM regime_events;")
        assert cursor.fetchone()[0] == 1

    # 2. Simulate failure during allocation save
    orig_alloc_save = storage.allocations.save

    def failing_save(*args, **kwargs):
        raise sqlite3.OperationalError("Simulated disk write fault")

    storage.allocations.save = failing_save

    t2 = t + timedelta(days=1)
    sig2 = SignalSnapshot(
        timestamp=t2,
        spy_price=505.0,
        spy_sma50=496.0,
        spy_sma200=481.0,
        realized_vol_20d=0.12,
        vol_scale_factor=1.0,
        drawdown_pct=-0.01,
        circuit_breaker_active=False,
        regime=MarketRegime.BULL_AGGRESSIVE,
        indicators={"atr": 5.0},
    )
    alloc2 = TargetAllocation(
        timestamp=t2,
        regime=MarketRegime.BULL_AGGRESSIVE,
        weights={"SPY": 0.5, "QQQ": 0.5},
        cash_weight=0.0,
        rationale="Aggressive bull",
    )

    with pytest.raises(sqlite3.OperationalError, match="Simulated disk write fault"):
        storage.save_daily_close(sig2, alloc2, t2, "Aggressive bull")

    # Verify complete rollback: table counts must strictly remain at 1
    with storage.db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert cursor.fetchone()[0] == 1, "Signal snapshot was not rolled back!"
        cursor.execute("SELECT COUNT(*) FROM allocations;")
        assert cursor.fetchone()[0] == 1, "Allocation was not rolled back!"
        cursor.execute("SELECT COUNT(*) FROM regime_events;")
        assert cursor.fetchone()[0] == 1, "Regime event was not rolled back!"

    # Restore and verify subsequent write succeeds
    storage.allocations.save = orig_alloc_save
    storage.save_daily_close(sig2, alloc2, t2, "Aggressive bull")

    with storage.db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert cursor.fetchone()[0] == 2
        cursor.execute("SELECT COUNT(*) FROM allocations;")
        assert cursor.fetchone()[0] == 2
        cursor.execute("SELECT COUNT(*) FROM regime_events;")
        assert cursor.fetchone()[0] == 2

    storage.db.close()
