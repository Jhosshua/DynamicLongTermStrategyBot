"""
tests.adversarial.test_cm4_empirical_challenge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone C-M4 Independent Empirical Challenger Verification Suite.
Adversarially tests and challenges:
1. Criterion 1: 25+ concurrent threads executing SQLite WAL write storm with 0 lock errors and clean PRAGMA integrity_check.
2. Criterion 2: Non-UTF8 binary corrupted bytes and un-flushed partial lines injected into JSONL audit logger.
3. Criterion 3: Fuzz CLI subcommands (dry-run, daemon, rebalance, backtest, export-metrics, status) with boundary inputs.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, List
import pytest
from typer.testing import CliRunner

from strategy_engine.cli.main import app
from strategy_engine.core.models import (
    Bar,
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
    RebalanceOrderRepository,
    SignalSnapshotRepository,
    StorageService,
)


# ============================================================================
# Criterion 1: 25+ Concurrent Worker Threads SQLite WAL Write Storm
# ============================================================================

def test_criterion1_wal_concurrent_write_storm(tmp_path: Path):
    """Adversarially challenge SQLite WAL persistence under extreme concurrency:

    - 24 writer threads (6 signals, 6 allocations, 6 orders, 6 market bars)
    - 6 reader threads continuously reading latest/ranges
    - Total threads: 30 threads concurrent
    - Total writes: 24 * 50 = 1,200 write transactions
    - Verifies: 0 database lock errors, 0 operational errors, PRAGMA integrity_check == 'ok',
      PRAGMA foreign_key_check is empty, and exact row counts match across all tables.
    """
    db_file = tmp_path / "cm4_wal_challenge.db"
    db = Database(db_file)

    # Verify initial WAL mode pragma
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal", f"Database not configured in WAL mode: {mode}"

    sig_repo = SignalSnapshotRepository(db)
    alloc_repo = AllocationRepository(db)
    order_repo = RebalanceOrderRepository(db)
    bar_repo = MarketBarRepository(db)

    iterations_per_writer = 50
    base_time = datetime(2026, 9, 4, 10, 0, 0, tzinfo=timezone.utc)

    lock_errors: List[str] = []
    other_errors: List[str] = []

    def write_signals(wid: int):
        for i in range(iterations_per_writer):
            ts = base_time + timedelta(seconds=wid * 1000 + i)
            sig = SignalSnapshot(
                timestamp=ts,
                spy_price=500.0 + (wid * 0.1) + i,
                spy_sma50=495.0,
                spy_sma200=480.0,
                realized_vol_20d=0.12,
                vol_scale_factor=1.0,
                drawdown_pct=-0.01,
                circuit_breaker_active=False,
                regime=MarketRegime.BULL_NORMAL,
                indicators={"worker": float(wid), "i": float(i)},
            )
            try:
                sig_repo.save(sig, rationale=f"sig_w{wid}_i{i}")
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    lock_errors.append(f"Signal w{wid}: {e}")
                else:
                    other_errors.append(f"Signal w{wid}: {e}")
            except Exception as e:
                other_errors.append(f"Signal w{wid}: {e}")

    def write_allocations(wid: int):
        for i in range(iterations_per_writer):
            ts = base_time + timedelta(seconds=wid * 1000 + i)
            alloc = TargetAllocation(
                timestamp=ts,
                regime=MarketRegime.BULL_AGGRESSIVE,
                weights={"SPY": 0.6, "QQQ": 0.4},
                cash_weight=0.0,
                rationale=f"alloc_w{wid}_i{i}",
            )
            try:
                alloc_repo.save(alloc)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    lock_errors.append(f"Alloc w{wid}: {e}")
                else:
                    other_errors.append(f"Alloc w{wid}: {e}")
            except Exception as e:
                other_errors.append(f"Alloc w{wid}: {e}")

    def write_orders(wid: int):
        for i in range(iterations_per_writer):
            ts = base_time + timedelta(seconds=wid * 1000 + i)
            order = OrderIntent(
                symbol="SPY",
                side=OrderSide.BUY,
                action="BUY",
                target_shares=10.0,
                delta_shares=2.0,
                estimated_price=500.0,
                target_weight=0.6,
                current_weight=0.5,
                delta_weight=0.1,
                timestamp=ts,
                rationale=f"order_w{wid}_i{i}",
            )
            try:
                order_repo.save_batch([order], status="FILLED")
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    lock_errors.append(f"Order w{wid}: {e}")
                else:
                    other_errors.append(f"Order w{wid}: {e}")
            except Exception as e:
                other_errors.append(f"Order w{wid}: {e}")

    def write_bars(wid: int):
        for i in range(iterations_per_writer):
            ts = base_time + timedelta(minutes=wid * 100 + i)
            bar = Bar(
                symbol=f"TICKER_{wid}",
                timestamp=ts,
                open=150.0 + i,
                high=152.0 + i,
                low=149.0 + i,
                close=151.0 + i,
                volume=50000 + i,
            )
            try:
                bar_repo.save_bars([bar], timeframe="1Day")
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    lock_errors.append(f"Bar w{wid}: {e}")
                else:
                    other_errors.append(f"Bar w{wid}: {e}")
            except Exception as e:
                other_errors.append(f"Bar w{wid}: {e}")

    def read_queries(rid: int):
        for _ in range(iterations_per_writer):
            try:
                _ = sig_repo.get_latest()
                _ = alloc_repo.get_latest()
                _ = order_repo.get_by_status("FILLED")
                time.sleep(0.0005)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    lock_errors.append(f"Reader r{rid}: {e}")
                else:
                    other_errors.append(f"Reader r{rid}: {e}")
            except Exception as e:
                other_errors.append(f"Reader r{rid}: {e}")

    threads = []
    for w in range(6):
        threads.append(("sig", w, write_signals))
    for w in range(6):
        threads.append(("alloc", w, write_allocations))
    for w in range(6):
        threads.append(("order", w, write_orders))
    for w in range(6):
        threads.append(("bar", w, write_bars))
    for r in range(6):
        threads.append(("read", r, read_queries))

    assert len(threads) == 30, f"Expected 30 threads, got {len(threads)}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futs = [executor.submit(fn, wid) for _, wid, fn in threads]
        concurrent.futures.wait(futs)

    # Verification: 0 lock errors, 0 operational errors
    assert len(lock_errors) == 0, f"Encountered lock errors: {lock_errors}"
    assert len(other_errors) == 0, f"Encountered unexpected errors: {other_errors}"

    # Database integrity checks
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        check = cursor.fetchone()[0]
        assert check == "ok", f"Integrity check failed: {check}"

        cursor.execute("PRAGMA foreign_key_check;")
        fk_check = cursor.fetchall()
        assert len(fk_check) == 0, f"Foreign key check failed: {fk_check}"

        cursor.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert cursor.fetchone()[0] == 6 * iterations_per_writer

        cursor.execute("SELECT COUNT(*) FROM allocations;")
        assert cursor.fetchone()[0] == 6 * iterations_per_writer

        cursor.execute("SELECT COUNT(*) FROM rebalance_orders;")
        assert cursor.fetchone()[0] == 6 * iterations_per_writer

        cursor.execute("SELECT COUNT(*) FROM market_bars;")
        assert cursor.fetchone()[0] == 6 * iterations_per_writer

    db.close()


# ============================================================================
# Criterion 2: JSONL Logger Non-UTF8 & Un-flushed Partial Line Resilience
# ============================================================================

def test_criterion2_jsonl_non_utf8_and_partial_line_recovery(tmp_path: Path):
    """Adversarially challenge JSONLAuditLogger against:

    - Raw corrupted non-UTF8 bytes: 0xFF, 0xFE, 0x80, 0x81, 0xC0 0xAF (overlong)
    - Incomplete multibyte UTF-8 sequences
    - Un-flushed partial lines lacking a newline (crash/kill simulation)
    - Mixed valid records before, in-between, and after corrupted segments
    Verify:
    - Zero unhandled UnicodeDecodeError
    - Zero JSON parser crashes
    - All valid records are recovered bit-for-bit intact
    - Subsequent appends do not fuse onto the unflushed partial line
    """
    log_dir = tmp_path / "jsonl_corrupt_harness"
    logger = JSONLAuditLogger(log_dir)
    target_date = "2026-09-04"
    log_file = log_dir / f"decisions_{target_date}.jsonl"

    t1 = datetime(2026, 9, 4, 9, 30, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 9, 4, 15, 50, 0, tzinfo=timezone.utc)

    # Step 1: Write valid decision 1
    did1 = logger.log_rebalance_decision(
        trigger="OPEN_CHECK",
        regime=MarketRegime.BULL_NORMAL,
        timestamp=t1,
        rationale="VALID_RECORD_1",
    )

    # Step 2: Inject non-UTF8 binary byte corruptions and unclosed partial line
    with open(log_file, "ab") as f:
        # Invalid UTF-8 bytes and overlong encodings
        f.write(b"\xff\xfe\xca\xfe\xba\xbe\x80\x81 raw invalid non-utf8 binary\n")
        f.write(b"\xc2 invalid incomplete 2-byte utf8 sequence\n")
        f.write(b"\xe2\x82 invalid incomplete 3-byte utf8 sequence\n")
        f.write(b"\xf0\x90\x80 invalid incomplete 4-byte utf8 sequence\n")
        f.write(b"\x00\x00\x00 null bytes\n")
        # Syntactically broken JSON
        f.write(b"{broken_json_no_quotes: 123}\n")
        f.write(b'{"key": "value", trailing_comma: true,}\n')
        # Non-dict JSON objects
        f.write(b"[1, 2, 3]\n")
        f.write(b'"just a string"\n')
        f.write(b"42\n")
        # Empty whitespace lines
        f.write(b"   \n\t\n")
        # UN-FLUSHED PARTIAL LINE WITHOUT NEWLINE
        f.write(b'{"version": "1.0", "decision_id": "abrupt_crash", "rationale": "partial unclosed wr')

    # Step 3: Write valid decision 2 via logger
    # Must detect missing newline on last byte, prepend newline, and write clean JSON line
    did2 = logger.log_rebalance_decision(
        trigger="REBALANCE_MIDDAY",
        regime=MarketRegime.CORRECTION_FRAGILE,
        timestamp=t2,
        rationale="VALID_RECORD_2",
    )

    # Step 4: Inject another unclosed raw binary partial line without newline
    with open(log_file, "ab") as f:
        f.write(b"\x80\x81\x82{\"another_partial_binary\": \xff\xfe")

    # Step 5: Write valid decision 3 via logger
    did3 = logger.log_rebalance_decision(
        trigger="CLOSE_EVAL",
        regime=MarketRegime.BEAR_CRISIS,
        timestamp=t3,
        rationale="VALID_RECORD_3",
    )

    # Step 6: Verify reading log records
    records = logger.read_date(target_date)
    assert len(records) == 3, f"Expected 3 valid records, got {len(records)}"
    assert [r["decision_id"] for r in records] == [did1, did2, did3]
    assert records[0]["rationale"] == "VALID_RECORD_1"
    assert records[1]["rationale"] == "VALID_RECORD_2"
    assert records[2]["rationale"] == "VALID_RECORD_3"

    # Verify read_decisions alias
    alias_records = logger.read_decisions(target_date)
    assert alias_records == records

    # Verify read_latest ordering
    latest = logger.read_latest(limit=10)
    assert len(latest) == 3
    assert [r["decision_id"] for r in latest] == [did3, did2, did1]

    # Verify find_decision
    assert logger.find_decision(did1)["rationale"] == "VALID_RECORD_1"
    assert logger.find_decision(did2)["rationale"] == "VALID_RECORD_2"
    assert logger.find_decision(did3)["rationale"] == "VALID_RECORD_3"
    assert logger.find_decision("nonexistent_id") is None
    assert logger.find_decision("abrupt_crash") is None


# ============================================================================
# Criterion 3: Fuzz CLI Subcommands Across Boundary Inputs
# ============================================================================

@pytest.mark.parametrize(
    "subcmd, bad_args, expected_code",
    [
        # dry-run boundary fuzzing
        ("dry-run", ["--equity", "-1000"], 1),
        ("dry-run", ["--equity", "-0.01"], 1),
        ("dry-run", ["--equity", "0"], 1),
        ("dry-run", ["--equity", "nan"], 1),
        ("dry-run", ["--equity", "inf"], 1),
        ("dry-run", ["--equity", "-inf"], 1),
        ("dry-run", ["--scenario", "invalid_scenario_abc"], 1),
        ("dry-run", ["--as-of", "not_a_valid_timestamp"], 1),
        ("dry-run", ["--current-weights", "{malformed_json}"], 1),
        ("dry-run", ["--current-weights", "[1, 2, 3]"], 1),
        ("dry-run", ["--scenario", "none", "--db-path", "/nonexistent_path_abc/db.sqlite"], 1),
        # rebalance boundary fuzzing
        ("rebalance", ["--equity", "-500"], 1),
        ("rebalance", ["--equity", "0"], 1),
        ("rebalance", ["--equity", "nan"], 1),
        ("rebalance", ["--equity", "inf"], 1),
        ("rebalance", ["--equity", "-inf"], 1),
        ("rebalance", ["--scenario", "bad_scen"], 1),
        ("rebalance", ["--current-weights", "broken_json"], 1),
        ("rebalance", ["--current-weights", "12345"], 1),
        # backtest boundary fuzzing
        ("backtest", ["--capital", "-1000"], 1),
        ("backtest", ["--capital", "0"], 1),
        ("backtest", ["--capital", "nan"], 1),
        ("backtest", ["--capital", "inf"], 1),
        ("backtest", ["--capital", "-inf"], 1),
        ("backtest", ["--scenario", "invalid_backtest_scenario"], 1),
        ("backtest", ["--export-csv", "/nonexistent_dir_abc/out.csv"], 1),
        # daemon boundary fuzzing
        ("daemon", ["--interval", "0", "--once"], 1),
        ("daemon", ["--interval", "-5s", "--once"], 1),
        ("daemon", ["--interval", "nan", "--once"], 1),
        ("daemon", ["--interval", "inf", "--once"], 1),
        ("daemon", ["--interval", "+inf", "--once"], 1),
        ("daemon", ["--interval", "-inf", "--once"], 1),
        ("daemon", ["--interval", "not_a_number", "--once"], 1),
        # export-metrics boundary fuzzing
        ("export-metrics", ["--table", "invalid_table_name"], 1),
        ("export-metrics", ["--table", "drop_table;"], 1),
        ("export-metrics", ["--format", "xml"], 1),
        ("export-metrics", ["--format", "parquet"], 1),
        ("export-metrics", ["--db-path", "/nonexistent_path_xyz/db.sqlite"], 1),
        ("export-metrics", ["--start-date", "bad_start_date"], 1),
        ("export-metrics", ["--end-date", "bad_end_date"], 1),
        ("export-metrics", ["--output", "/nonexistent_dir_xyz/out.json"], 1),
        # status boundary fuzzing
        ("status", ["--db-path", "/nonexistent_path_xyz/db.sqlite", "--json"], 0),
        ("status", ["--relay-url", "http://127.0.0.1:9", "--json"], 0),
    ],
)
def test_criterion3_cli_subcommand_fuzzing(subcmd: str, bad_args: List[str], expected_code: int):
    """Fuzz all CLI subcommands with boundary inputs.

    Verify:
    1. Exit code is expected (1 for validation errors, 0 for status with unreachable targets)
    2. Output has NO raw unhandled Python tracebacks ('Traceback (most recent call last)')
    3. Descriptive error message is present when exit code != 0
    """
    runner = CliRunner()
    result = runner.invoke(app, [subcmd] + bad_args)
    assert result.exit_code == expected_code, (
        f"CLI '{subcmd}' with {bad_args} returned exit code {result.exit_code} (expected {expected_code}). Output:\n{result.output}"
    )
    assert "traceback (most recent call last)" not in result.output.lower(), (
        f"Unhandled Python traceback detected in '{subcmd}' with {bad_args}:\n{result.output}"
    )
    if expected_code != 0:
        assert "error" in result.output.lower() or "invalid" in result.output.lower(), (
            f"Expected error message missing in output for '{subcmd}' with {bad_args}:\n{result.output}"
        )


def test_criterion3_cli_syntax_and_missing_arguments():
    """Verify that invalid subcommands and missing required parameters exit with code 2 and zero tracebacks."""
    runner = CliRunner()

    # Completely invalid subcommand
    res = runner.invoke(app, ["nonexistent-command-xyz"])
    assert res.exit_code == 2
    assert "traceback" not in res.output.lower()
    assert "no such command" in res.output.lower()

    # Unknown option
    res = runner.invoke(app, ["dry-run", "--completely-bogus-option"])
    assert res.exit_code == 2
    assert "traceback" not in res.output.lower()
    assert "no such option" in res.output.lower()


# ============================================================================
# Additional Stress: 40-Worker WAL Storm & Concurrent Corruptor Interleaving
# ============================================================================

def test_criterion1_wal_extreme_40_worker_storm(tmp_path: Path):
    """Empirically challenge WAL under 40 concurrent workers executing 4,000 transactions."""
    db_file = tmp_path / "extreme_wal_40w.db"
    db = Database(db_file)

    sig_repo = SignalSnapshotRepository(db)
    alloc_repo = AllocationRepository(db)
    order_repo = RebalanceOrderRepository(db)
    bar_repo = MarketBarRepository(db)

    num_workers = 40
    iters = 100
    base_ts = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)

    lock_errs: List[str] = []
    other_errs: List[str] = []

    def worker_fn(kind: str, wid: int):
        for i in range(iters):
            ts = base_ts + timedelta(seconds=wid * 1000 + i)
            try:
                if kind == "sig":
                    sig = SignalSnapshot(
                        timestamp=ts,
                        spy_price=500.0,
                        spy_sma50=490.0,
                        spy_sma200=480.0,
                        realized_vol_20d=0.15,
                        vol_scale_factor=1.0,
                        drawdown_pct=-0.02,
                        circuit_breaker_active=False,
                        regime=MarketRegime.BULL_NORMAL,
                        indicators={"i": float(i)},
                    )
                    sig_repo.save(sig)
                elif kind == "alloc":
                    alloc = TargetAllocation(
                        timestamp=ts,
                        regime=MarketRegime.BULL_NORMAL,
                        weights={"SPY": 1.0},
                        cash_weight=0.0,
                        rationale="storm",
                    )
                    alloc_repo.save(alloc)
                elif kind == "order":
                    order = OrderIntent(
                        symbol="SPY",
                        side=OrderSide.BUY,
                        action="BUY",
                        target_shares=10.0,
                        delta_shares=1.0,
                        estimated_price=500.0,
                        target_weight=1.0,
                        current_weight=0.9,
                        delta_weight=0.1,
                        timestamp=ts,
                    )
                    order_repo.save_batch([order])
                elif kind == "bar":
                    bar = Bar(
                        symbol=f"SYM_{wid}",
                        timestamp=ts,
                        open=100.0,
                        high=105.0,
                        low=95.0,
                        close=102.0,
                        volume=1000,
                    )
                    bar_repo.save_bars([bar], timeframe="1Min")
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    lock_errs.append(f"{kind}_{wid}_{i}: {e}")
                else:
                    other_errs.append(f"{kind}_{wid}_{i}: {e}")
            except Exception as e:
                other_errs.append(f"{kind}_{wid}_{i}: {e}")

    tasks = []
    for w in range(10):
        tasks.append(("sig", w))
    for w in range(10):
        tasks.append(("alloc", w))
    for w in range(10):
        tasks.append(("order", w))
    for w in range(10):
        tasks.append(("bar", w))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(worker_fn, kind, wid) for kind, wid in tasks]
        concurrent.futures.wait(futs)

    assert len(lock_errs) == 0, f"Lock errors: {lock_errs}"
    assert len(other_errs) == 0, f"Other errors: {other_errs}"

    with db.transaction() as conn:
        c = conn.cursor()
        c.execute("PRAGMA integrity_check;")
        assert c.fetchone()[0] == "ok"
        c.execute("PRAGMA quick_check;")
        assert c.fetchone()[0] == "ok"
        c.execute("SELECT COUNT(*) FROM signal_snapshots;")
        assert c.fetchone()[0] == 1000
        c.execute("SELECT COUNT(*) FROM allocations;")
        assert c.fetchone()[0] == 1000
        c.execute("SELECT COUNT(*) FROM rebalance_orders;")
        assert c.fetchone()[0] == 1000
        c.execute("SELECT COUNT(*) FROM market_bars;")
        assert c.fetchone()[0] == 1000

    db.close()


def test_criterion2_jsonl_extreme_concurrent_corruption(tmp_path: Path):
    """Stress JSONL logger with 20 concurrent writers, 10 readers, and 5 binary corruptors."""
    log_dir = tmp_path / "extreme_jsonl"
    logger = JSONLAuditLogger(log_dir)
    target_date = "2026-09-04"
    log_file = log_dir / f"decisions_{target_date}.jsonl"

    t0 = datetime(2026, 9, 4, 10, 0, 0, tzinfo=timezone.utc)
    did0 = logger.log_rebalance_decision("PRE_TRIGGER", MarketRegime.BULL_NORMAL, timestamp=t0, rationale="PRE_SEED")

    # Inject un-flushed partial line
    with open(log_file, "ab") as f:
        f.write(b'\x80\x81\xff\xfe non utf8\n')
        f.write(b'{"partial": 123')  # NO newline!

    t1 = datetime(2026, 9, 4, 11, 0, 0, tzinfo=timezone.utc)
    did1 = logger.log_rebalance_decision("MID_TRIGGER", MarketRegime.BULL_AGGRESSIVE, timestamp=t1, rationale="AFTER_PARTIAL")

    # Inject another unclosed binary partial line
    with open(log_file, "ab") as f:
        f.write(b'\xca\xfe\xba\xbe{"unclosed": "bad')  # NO newline!

    t2 = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    did2 = logger.log_rebalance_decision("POST_TRIGGER", MarketRegime.BEAR_CRISIS, timestamp=t2, rationale="AFTER_BINARY_PARTIAL")

    num_writers = 20
    num_readers = 10
    num_corruptors = 5
    writes_per_thread = 50

    writer_dids: List[str] = []
    writer_lock = threading.Lock()
    errors: List[str] = []

    def writer_task(wid: int):
        for i in range(writes_per_thread):
            ts = t0 + timedelta(seconds=wid * 1000 + i)
            try:
                did = logger.log_rebalance_decision(
                    f"W_{wid}_{i}", MarketRegime.BULL_NORMAL, timestamp=ts, rationale=f"R_{wid}_{i}"
                )
                with writer_lock:
                    writer_dids.append(did)
            except Exception as e:
                errors.append(f"Writer error {wid}: {e}")

    def reader_task(rid: int):
        for _ in range(30):
            try:
                recs = logger.read_date(target_date)
                assert isinstance(recs, list)
                latest = logger.read_latest(limit=5)
                assert isinstance(latest, list)
                time.sleep(0.001)
            except Exception as e:
                errors.append(f"Reader error {rid}: {e}")

    def corruptor_task(cid: int):
        for _ in range(10):
            try:
                with open(log_file, "ab") as f:
                    f.write(b"\x80\x81 corrupt bytes\n")
                time.sleep(0.002)
            except Exception as e:
                errors.append(f"Corruptor error {cid}: {e}")

    threads = []
    for w in range(num_writers):
        threads.append(writer_task)
    for r in range(num_readers):
        threads.append(reader_task)
    for c in range(num_corruptors):
        threads.append(corruptor_task)

    with concurrent.futures.ThreadPoolExecutor(max_workers=35) as ex:
        futs = [ex.submit(fn, i) for i, fn in enumerate(threads)]
        concurrent.futures.wait(futs)

    assert len(errors) == 0, f"Errors encountered: {errors}"

    final_records = logger.read_date(target_date)
    recovered_dids = {r["decision_id"] for r in final_records}
    expected_dids = {did0, did1, did2} | set(writer_dids)
    assert expected_dids.issubset(recovered_dids)


def test_criterion3_cli_direct_subprocess_fuzzing():
    """Verify actual subprocess execution of CLI across all subcommands with boundary inputs."""
    import subprocess

    cmds = [
        (["dry-run", "--equity", "-1"], 1),
        (["dry-run", "--equity", "nan"], 1),
        (["dry-run", "--equity", "inf"], 1),
        (["daemon", "--interval", "0", "--once"], 1),
        (["daemon", "--interval", "nan", "--once"], 1),
        (["daemon", "--interval", "inf", "--once"], 1),
        (["daemon", "--interval", "+inf", "--once"], 1),
        (["daemon", "--interval", "-10s", "--once"], 1),
        (["rebalance", "--equity", "-50"], 1),
        (["rebalance", "--equity", "nan"], 1),
        (["backtest", "--capital", "-100"], 1),
        (["backtest", "--capital", "nan"], 1),
        (["export-metrics", "--table", "nonexistent_tbl"], 1),
        (["export-metrics", "--format", "yaml"], 1),
        (["status", "--relay-url", "http://127.0.0.1:1"], 0),
        (["nonexistent_subcmd"], 2),
    ]

    for args, expected_code in cmds:
        proc = subprocess.run(
            [sys.executable, "-m", "strategy_engine.cli"] + args,
            capture_output=True,
            text=True,
        )
        out = proc.stdout + proc.stderr
        assert proc.returncode == expected_code, (
            f"Cmd {args} returned {proc.returncode} instead of {expected_code}. Output:\n{out}"
        )
        assert "traceback (most recent call last)" not in out.lower(), (
            f"Traceback in {args}:\n{out}"
        )

