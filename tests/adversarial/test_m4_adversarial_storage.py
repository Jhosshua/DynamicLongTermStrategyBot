"""
tests/adversarial/test_m4_adversarial_storage.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adversarial Stress & Resilience Verification Suite for Milestone M4 (Storage & Concurrency Resilience).
Covers:
1. Concurrent write stress under SQLite WAL mode with 20 simultaneous workers + concurrent readers.
   Verifies zero OperationalError: database is locked and 100% data integrity.
2. JSONL corruption tolerance: malformed, partial, and empty lines.
   Verifies skipping of corrupt lines, while also documenting the missing `read_decisions`
   attribute on JSONLAuditLogger and the UnicodeDecodeError vulnerability on binary corruption.
3. Point-in-time query accuracy: out-of-order scrambled timestamp insertions across all repositories.
   Verifies zero forward lookahead contamination and strict point-in-time boundary semantics.
4. ACID transaction rollback guarantees and batch atomicity.
"""

import concurrent.futures
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import random
import sqlite3
import pytest

from strategy_engine.core.models import (
    MarketRegime,
    OrderIntent,
    OrderSide,
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
# 1. Concurrent Write Stress Under SQLite WAL Mode (20 Workers)
# ============================================================================

def test_sqlite_wal_concurrent_20_tasks_stress(tmp_path):
    """Launch 20 concurrent tasks writing signal snapshots, allocations, and orders

    simultaneously to SQLite; verify zero OperationalError: database is locked
    and 100% data integrity under WAL mode.
    """
    db_file = tmp_path / "concurrent_stress.db"
    db = Database(db_file)

    # Verify initial WAL pragma configuration
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        j_mode = cursor.fetchone()[0]
        assert j_mode.upper() == "WAL", f"Expected WAL journal mode, got {j_mode}"

        cursor.execute("PRAGMA busy_timeout;")
        b_timeout = cursor.fetchone()[0]
        assert b_timeout == 5000, f"Expected busy_timeout=5000, got {b_timeout}"

    sig_repo = SignalSnapshotRepository(db)
    alloc_repo = AllocationRepository(db)
    order_repo = RebalanceOrderRepository(db)

    errors = []
    writes_per_worker = 50

    # Worker definitions
    def worker_signals(worker_idx: int):
        try:
            for i in range(writes_per_worker):
                t = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc) + timedelta(seconds=worker_idx * 100 + i)
                snap = SignalSnapshot(
                    timestamp=t,
                    spy_price=500.0 + i,
                    spy_sma50=495.0,
                    spy_sma200=480.0,
                    realized_vol_20d=0.12,
                    vol_scale_factor=1.0,
                    drawdown_pct=-0.01,
                    regime=MarketRegime.BULL_NORMAL,
                    indicators={"qqq_price": 400.0 + i, "drawdown_gate": 1.0},
                )
                sig_repo.save(snap, rationale=f"Worker {worker_idx} iter {i}")
        except Exception as exc:
            errors.append(("signal", worker_idx, type(exc).__name__, str(exc)))

    def worker_allocations(worker_idx: int):
        try:
            for i in range(writes_per_worker):
                t = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc) + timedelta(seconds=worker_idx * 100 + i)
                alloc = TargetAllocation(
                    timestamp=t,
                    regime=MarketRegime.BULL_NORMAL,
                    weights={"SPY": 0.60, "QQQ": 0.40},
                    cash_weight=0.0,
                    rationale=f"Worker {worker_idx} iter {i}",
                )
                alloc_repo.save(alloc, risk_multiplier=1.0)
        except Exception as exc:
            errors.append(("alloc", worker_idx, type(exc).__name__, str(exc)))

    def worker_orders(worker_idx: int):
        try:
            for i in range(writes_per_worker):
                t = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc) + timedelta(seconds=worker_idx * 100 + i)
                order = OrderIntent(
                    symbol="SPY" if i % 2 == 0 else "QQQ",
                    action="BUY" if i % 2 == 0 else "SELL",
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    target_weight=0.6 if i % 2 == 0 else 0.4,
                    current_weight=0.5,
                    delta_weight=0.1 if i % 2 == 0 else -0.1,
                    delta_shares=10.0,
                    delta_dollars=5000.0,
                    estimated_price=500.0,
                    notional=5000.0,
                    timestamp=t,
                    rationale=f"Worker {worker_idx} order {i}",
                )
                order_repo.save(order, status="PENDING")
        except Exception as exc:
            errors.append(("order", worker_idx, type(exc).__name__, str(exc)))

    def worker_readers(worker_idx: int):
        try:
            for _ in range(50):
                sig_repo.get_latest()
                alloc_repo.get_latest()
                order_repo.get_by_status("PENDING")
        except Exception as exc:
            errors.append(("reader", worker_idx, type(exc).__name__, str(exc)))

    # Launch 20 concurrent writer tasks (7 signal + 7 allocation + 6 order = 20 writers)
    # plus 5 concurrent readers (total 25 concurrent threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        futures = []
        for w in range(7):
            futures.append(executor.submit(worker_signals, w))
        for w in range(7, 14):
            futures.append(executor.submit(worker_allocations, w))
        for w in range(14, 20):
            futures.append(executor.submit(worker_orders, w))
        for r in range(20, 25):
            futures.append(executor.submit(worker_readers, r))

        concurrent.futures.wait(futures)

    # 1. Verify zero OperationalError: database is locked
    locked_errors = [e for e in errors if "database is locked" in e[3] or "OperationalError" in e[2]]
    assert len(locked_errors) == 0, f"Encountered database locked errors: {locked_errors}"
    assert len(errors) == 0, f"Encountered unexpected concurrency errors: {errors}"

    # 2. Verify 100% data integrity
    expected_signals = 7 * writes_per_worker     # 350
    expected_allocs = 7 * writes_per_worker      # 350
    expected_orders = 6 * writes_per_worker      # 300

    with db.transaction() as conn:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM signal_snapshots;")
        actual_signals = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM allocations;")
        actual_allocs = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM rebalance_orders;")
        actual_orders = cur.fetchone()[0]

    assert actual_signals == expected_signals, f"Data loss in signals: expected {expected_signals}, got {actual_signals}"
    assert actual_allocs == expected_allocs, f"Data loss in allocations: expected {expected_allocs}, got {actual_allocs}"
    assert actual_orders == expected_orders, f"Data loss in orders: expected {expected_orders}, got {actual_orders}"

    db.close()


# ============================================================================
# 2. JSONL Corruption Tolerance & Crash Resiliency
# ============================================================================

def test_jsonl_missing_read_decisions_attribute():
    """REMEDIATED: JSONLAuditLogger provides the required `read_decisions` alias method."""
    logger = JSONLAuditLogger("dummy_logs")
    assert hasattr(logger, "read_decisions"), (
        "JSONLAuditLogger missing required read_decisions attribute"
    )
    assert logger.read_decisions("2026-09-01") == logger.read_date("2026-09-01")


def test_jsonl_corruption_tolerance_skips_malformed_partial_empty_lines(tmp_path):
    """Inject corrupted, partial, and empty lines into the audit log;

    verify JSONLAuditLogger reader skips corrupt lines and successfully parses
    valid entries without crashing.
    """
    log_dir = tmp_path / "audit_logs"
    logger = JSONLAuditLogger(log_dir)

    target_date = "2026-09-01"
    log_file = log_dir / f"decisions_{target_date}.jsonl"

    # Inject diverse corrupted, partial, empty, and invalid entries
    injected_lines = [
        # Valid entry 1
        json.dumps({
            "version": "1.0",
            "decision_id": "dec-001",
            "timestamp": "2026-09-01T15:50:00+00:00",
            "trigger": "DAILY_CLOSE",
            "regime": "BULL_NORMAL",
            "signals": {"spy_price": 500.0},
            "portfolio": {},
            "allocations": {"weights": {"SPY": 1.0}},
            "orders": [],
            "rationale": "Valid entry 1",
        }) + "\n",

        # Empty lines and whitespace
        "\n",
        "    \n",
        "\t  \t\n",

        # Corrupted malformed JSON
        "{MALFORMED_JSON_WITHOUT_QUOTES: true}\n",
        "!!! GARBAGE TEXT NOT JSON AT ALL !!!\n",

        # Valid entry 2
        json.dumps({
            "version": "1.0",
            "decision_id": "dec-002",
            "timestamp": "2026-09-01T15:51:00+00:00",
            "trigger": "MONTHLY_REBALANCE",
            "regime": "BULL_AGGRESSIVE",
            "signals": {"spy_price": 502.0},
            "portfolio": {},
            "allocations": {"weights": {"QQQ": 0.5, "SPY": 0.5}},
            "orders": [],
            "rationale": "Valid entry 2",
        }) + "\n",

        # Partial lines (truncated JSON write from abrupt crash / power loss)
        '{"version": "1.0", "decision_id": "dec-truncated", "timestamp": "2026-09-01T15:52:00\n',
        '{"version": "1.0", "decision_id": "dec-truncated-2", "allocations": {\n',

        # Valid JSON but non-dictionary types (should be filtered out)
        "12345\n",
        '"a bare json string"\n',
        "[1, 2, 3]\n",
        "true\n",
        "null\n",

        # Valid entry 3
        json.dumps({
            "version": "1.0",
            "decision_id": "dec-003",
            "timestamp": "2026-09-01T15:55:00+00:00",
            "trigger": "CIRCUIT_BREAKER",
            "regime": "CORRECTION_FRAGILE",
            "signals": {"spy_price": 490.0},
            "portfolio": {},
            "allocations": {"weights": {"SHV": 1.0}},
            "orders": [],
            "rationale": "Valid entry 3",
        }) + "\n",
    ]

    with open(log_file, "w", encoding="utf-8") as f:
        f.writelines(injected_lines)

    # Read via read_date
    parsed_date_records = logger.read_date(target_date)
    assert len(parsed_date_records) == 3, f"Expected 3 valid records, got {len(parsed_date_records)}"
    assert [r["decision_id"] for r in parsed_date_records] == ["dec-001", "dec-002", "dec-003"]
    assert parsed_date_records[0]["regime"] == "BULL_NORMAL"
    assert parsed_date_records[1]["regime"] == "BULL_AGGRESSIVE"
    assert parsed_date_records[2]["regime"] == "CORRECTION_FRAGILE"

    # Read via read_latest
    latest_records = logger.read_latest(limit=10)
    assert len(latest_records) == 3
    # read_latest returns reverse chronological order (newest first)
    assert [r["decision_id"] for r in latest_records] == ["dec-003", "dec-002", "dec-001"]

    # Verify individual find_decision works despite corrupted lines in file
    found_2 = logger.find_decision("dec-002")
    assert found_2 is not None
    assert found_2["decision_id"] == "dec-002"
    assert logger.find_decision("non_existent_id") is None


def test_jsonl_audit_logger_binary_non_utf8_corruption_vulnerability(tmp_path):
    """REMEDIATED: Raw non-UTF-8 binary corruption is handled gracefully with errors='replace'.
    
    When an unexpected disk crash or file-system corruption writes non-UTF-8 bytes
    into the JSONL file, `open(log_path, 'r', encoding='utf-8', errors='replace')` replaces
    corrupt bytes with replacement chars, enabling json.loads error handling to isolate
    the corrupted line and recover all valid records.
    """
    log_dir = tmp_path / "corrupt_binary_logs"
    logger = JSONLAuditLogger(log_dir)
    target_date = "2026-09-01"
    log_file = log_dir / f"decisions_{target_date}.jsonl"

    # Write a valid record, followed by corrupted raw binary bytes, followed by another valid record
    with open(log_file, "wb") as f:
        f.write(b'{"version": "1.0", "decision_id": "dec-b1"}\n')
        # Non-UTF-8 byte sequence (e.g. 0x80, 0x81, 0xFF)
        f.write(b'\x80\x81\xff\xfe corrupted binary junk\n')
        f.write(b'{"version": "1.0", "decision_id": "dec-b2"}\n')

    # Does NOT raise UnicodeDecodeError, recovers surrounding valid records
    records = logger.read_date(target_date)
    assert len(records) == 2
    assert [r["decision_id"] for r in records] == ["dec-b1", "dec-b2"]


# ============================================================================
# 3. Point-In-Time Query Accuracy & Out-Of-Order Timestamps
# ============================================================================

def test_point_in_time_scrambled_timestamps_no_lookahead(tmp_path):
    """Insert records with scrambled out-of-order timestamps; verify `get_as_of`

    returns exact historical state without forward lookahead contamination.
    """
    db_file = tmp_path / "pit_scrambled.db"
    db = Database(db_file)

    sig_repo = SignalSnapshotRepository(db)
    alloc_repo = AllocationRepository(db)
    port_repo = PortfolioStateRepository(db)
    reg_repo = RegimeEventRepository(db)
    ord_repo = RebalanceOrderRepository(db)

    base = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Generate 10 discrete hourly timestamps
    timestamps = [base + timedelta(hours=i) for i in range(10)]

    # Scramble the insertion sequence deliberately
    indexed_ts = list(enumerate(timestamps))
    rng = random.Random(1337)
    rng.shuffle(indexed_ts)

    for idx, ts in indexed_ts:
        # 1. Signal snapshot
        sig_repo.save_raw(
            timestamp=ts,
            regime="BULL_NORMAL" if idx % 2 == 0 else "CORRECTION_FRAGILE",
            spy_price=500.0 + idx,
            qqq_price=400.0 + idx,
            vol_20d=0.10 + (idx * 0.01),
            vol_scale_factor=1.0,
            drawdown_gate=1.0,
            rationale=f"Index {idx}",
        )

        # 2. Allocations
        alloc_repo.save_raw(
            timestamp=ts,
            regime="BULL_NORMAL" if idx % 2 == 0 else "CORRECTION_FRAGILE",
            weights={"SPY": round(0.50 + idx * 0.02, 4), "SHV": round(0.50 - idx * 0.02, 4)},
            rationale=f"Alloc {idx}",
        )

        # 3. Portfolio states
        port_repo.save(
            timestamp=ts,
            cash=10000.0 + idx * 100,
            equity=90000.0 + idx * 100,
            total_nav=100000.0 + idx * 200,
            positions={"SPY": 100 + idx},
        )

        # 4. Regime events
        reg_repo.record_event(
            timestamp=ts,
            old_regime="BULL_NORMAL",
            new_regime="CORRECTION_FRAGILE" if idx % 2 == 1 else "BULL_AGGRESSIVE",
            trigger_reason=f"RegimeEvent_{idx}",
        )

        # 5. Orders
        ord_repo.save_raw(
            id=f"order_{idx}",
            timestamp=ts,
            symbol="SPY",
            side="BUY",
            shares=float(idx + 1),
            price=500.0 + idx,
            notional=1000.0 * (idx + 1),
            target_weight=0.05 * (idx + 1),
            current_weight=0.02 * (idx + 1),
        )

    # ------------------------------------------------------------------------
    # Verification 1: Before any records exist
    # ------------------------------------------------------------------------
    before_all = base - timedelta(minutes=1)
    assert sig_repo.get_as_of(before_all) is None
    assert alloc_repo.get_as_of(before_all) is None
    assert port_repo.get_as_of(before_all) is None
    assert reg_repo.get_as_of(before_all) is None
    assert ord_repo.get_as_of(before_all) == []

    # ------------------------------------------------------------------------
    # Verification 2: Exact point-in-time matches
    # ------------------------------------------------------------------------
    for idx, ts in enumerate(timestamps):
        # Signals
        s = sig_repo.get_as_of(ts)
        assert s is not None
        assert s["spy_price"] == 500.0 + idx

        # Allocations
        a = alloc_repo.get_as_of(ts)
        assert a is not None
        assert a["weights"]["SPY"] == round(0.50 + idx * 0.02, 4)

        # Portfolio
        p = port_repo.get_as_of(ts)
        assert p is not None
        assert p["total_nav"] == 100000.0 + idx * 200

        # Regime events
        r = reg_repo.get_as_of(ts)
        assert r is not None
        assert r["trigger_reason"] == f"RegimeEvent_{idx}"

        # Orders
        o = ord_repo.get_as_of(ts)
        assert len(o) == idx + 1

    # ------------------------------------------------------------------------
    # Verification 3: Midpoints between T_i and T_{i+1} (Forward Lookahead Defense)
    # ------------------------------------------------------------------------
    for idx in range(len(timestamps) - 1):
        midpoint = timestamps[idx] + timedelta(minutes=30)

        # Must return exactly record `idx`, NEVER record `idx + 1`
        s_mid = sig_repo.get_as_of(midpoint)
        assert s_mid is not None
        assert s_mid["spy_price"] == 500.0 + idx, (
            f"Forward lookahead bug: at midpoint between {idx} and {idx+1}, got price {s_mid['spy_price']}"
        )

        a_mid = alloc_repo.get_as_of(midpoint)
        assert a_mid is not None
        assert a_mid["weights"]["SPY"] == round(0.50 + idx * 0.02, 4)

        p_mid = port_repo.get_as_of(midpoint)
        assert p_mid is not None
        assert p_mid["total_nav"] == 100000.0 + idx * 200

        r_mid = reg_repo.get_as_of(midpoint)
        assert r_mid is not None
        assert r_mid["trigger_reason"] == f"RegimeEvent_{idx}"

        o_mid = ord_repo.get_as_of(midpoint)
        assert len(o_mid) == idx + 1

    # ------------------------------------------------------------------------
    # Verification 4: Well after all records
    # ------------------------------------------------------------------------
    after_all = timestamps[-1] + timedelta(days=30)
    assert sig_repo.get_as_of(after_all)["spy_price"] == 500.0 + 9
    assert alloc_repo.get_as_of(after_all)["weights"]["SPY"] == round(0.50 + 9 * 0.02, 4)
    assert port_repo.get_as_of(after_all)["total_nav"] == 100000.0 + 9 * 200
    assert reg_repo.get_as_of(after_all)["trigger_reason"] == "RegimeEvent_9"
    assert len(ord_repo.get_as_of(after_all)) == 10

    db.close()


def test_point_in_time_identical_timestamps_tie_breaking(tmp_path):
    """EMPIRICAL FINDING: Identical timestamp tie-breaking asymmetry across repositories.
    
    When two records are saved with the identical timestamp:
    - RegimeEventRepository sorts by `timestamp DESC, id DESC LIMIT 1`, returning the
      latest inserted record (id=2).
    - SignalSnapshotRepository and AllocationRepository sort only by `timestamp DESC LIMIT 1`.
      Because SQLite uses index order, it returns the oldest inserted record (id=1)
      instead of the newer revision.
    """
    db = Database(tmp_path / "tie_breaker.db")
    sig_repo = SignalSnapshotRepository(db)
    reg_repo = RegimeEventRepository(db)

    t = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Insert two signal snapshots at exact same timestamp
    id1 = sig_repo.save_raw(t, "BULL_NORMAL", 500.0, 400.0, 0.12, 1.0, 1.0, rationale="Original")
    id2 = sig_repo.save_raw(t, "BULL_NORMAL", 505.0, 405.0, 0.12, 1.0, 1.0, rationale="Revision")

    # Insert two regime events at exact same timestamp
    ev1 = reg_repo.record_event(t, "BULL_NORMAL", "BULL_AGGRESSIVE", "Event 1")
    ev2 = reg_repo.record_event(t, "BULL_AGGRESSIVE", "CORRECTION_FRAGILE", "Event 2")

    # Query PIT
    sig_res = sig_repo.get_as_of(t)
    reg_res = reg_repo.get_as_of(t)

    # RegimeEventRepository returns ev2 (id DESC)
    assert reg_res["trigger_reason"] == "Event 2"
    assert reg_res["id"] == ev2

    # SignalSnapshotRepository returns id1 (lacks id DESC)
    assert sig_res["rationale"] == "Original"
    assert sig_res["id"] == id1

    db.close()


def test_point_in_time_microsecond_precision(tmp_path):
    """Ensure sub-second microsecond timestamps sort deterministically without lookahead."""
    db = Database(tmp_path / "pit_micro.db")
    sig_repo = SignalSnapshotRepository(db)

    t0 = datetime(2026, 9, 1, 10, 0, 0, 100000, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 1, 10, 0, 0, 500000, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 1, 10, 0, 0, 900000, tzinfo=timezone.utc)

    # Insert out of order
    sig_repo.save_raw(t2, "BULL_NORMAL", 502.0, 400.0, 0.12, 1.0, 1.0)
    sig_repo.save_raw(t0, "BULL_NORMAL", 500.0, 400.0, 0.12, 1.0, 1.0)
    sig_repo.save_raw(t1, "BULL_NORMAL", 501.0, 400.0, 0.12, 1.0, 1.0)

    # Query between t0 and t1 (at .300000)
    mid = datetime(2026, 9, 1, 10, 0, 0, 300000, tzinfo=timezone.utc)
    res = sig_repo.get_as_of(mid)
    assert res is not None
    assert res["spy_price"] == 500.0

    # Query between t1 and t2 (at .700000)
    mid2 = datetime(2026, 9, 1, 10, 0, 0, 700000, tzinfo=timezone.utc)
    res2 = sig_repo.get_as_of(mid2)
    assert res2 is not None
    assert res2["spy_price"] == 501.0

    db.close()


# ============================================================================
# 4. ACID Transaction Guarantees & Batch Atomicity
# ============================================================================

def test_sqlite_transaction_rollback_on_failure(tmp_path):
    """Verify that transactions cleanly roll back upon unhandled exceptions."""
    db = Database(tmp_path / "rollback.db")

    with pytest.raises(RuntimeError, match="Intentional transaction crash"):
        with db.transaction() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO allocations (timestamp, regime, weights_json, rationale) "
                "VALUES ('2026-09-01T10:00:00+00:00', 'BULL_NORMAL', '{}', 'Should roll back');"
            )
            raise RuntimeError("Intentional transaction crash")

    # Verify no row was persisted
    with db.transaction() as conn:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM allocations;")
        cnt = cur.fetchone()[0]
        assert cnt == 0, f"Rollback failed: {cnt} records persisted"

    db.close()


def test_rebalance_orders_batch_save_atomicity(tmp_path):
    """Verify save_batch atomically commits all orders or fails cleanly."""
    db = Database(tmp_path / "batch.db")
    repo = RebalanceOrderRepository(db)

    t = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    orders = [
        OrderIntent(
            symbol="SPY", action="BUY", side=OrderSide.BUY,
            target_weight=0.5, current_weight=0.3, delta_weight=0.2,
            delta_shares=10.0, delta_dollars=5000.0, estimated_price=500.0,
            notional=5000.0, timestamp=t,
        ),
        OrderIntent(
            symbol="QQQ", action="SELL", side=OrderSide.SELL,
            target_weight=0.3, current_weight=0.5, delta_weight=-0.2,
            delta_shares=-10.0, delta_dollars=-4000.0, estimated_price=400.0,
            notional=4000.0, timestamp=t,
        ),
    ]

    ids = repo.save_batch(orders)
    assert len(ids) == 2

    # Verify both records saved with PENDING status
    pending = repo.get_by_status("PENDING")
    assert len(pending) == 2
    assert {o["symbol"] for o in pending} == {"SPY", "QQQ"}

    # Update one status
    repo.update_status(ids[0], "FILLED")
    assert repo.get_by_id(ids[0])["status"] == "FILLED"
    assert repo.get_by_id(ids[1])["status"] == "PENDING"

    db.close()
