"""
tests/adversarial/test_m4_adversarial_smoke_and_reset.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adversarial Challenger Stress Suite for Milestone 4 (Features 15, 17, R4):
- E2E Smoke Test 7-Phase Execution and Structured Reporting
- 5x Consecutive Reset Idempotence Stress
- Injected Deep Corruption Recovery (Negative Cash, Orphaned Positions, Corrupted Ledgers)
- 17 Invariant Checks Verification Under Adversarial Conditions
- Blank/Damaged Schema Recovery
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List

import httpx
import pytest

from bot.discord_alerts import DiscordNotifier
from bot.service import DynamicStrategyService, ServiceConfig
from scripts.e2e_smoke_test import SmokeTestRunner
from scripts.reset_pristine_for_monday import (
    InvariantCheckResult,
    PristineResetManager,
    ResetReport,
    format_tabular_report,
)
from web.app import create_app


# ============================================================================
# Section 1: E2E Smoke Test 7-Phase In-Process Verification
# ============================================================================

@pytest.mark.asyncio
async def test_smoke_runner_in_process_all_7_phases(tmp_path: Path):
    """Verify all 7 phases execute cleanly in-process with structured results."""
    db_file = str(tmp_path / "smoke_adversarial.db")
    config = ServiceConfig(db_path=db_file, initial_cash=50000.00)
    notifier = DiscordNotifier(suppress_in_test=True)
    service = DynamicStrategyService(config=config, discord_notifier=notifier)
    service.paper_account.init_schema()

    await service.client.state_machine.handle_upstream_connected("Adversarial smoke boot")
    await service.feed_manager._transition_to_live("Adversarial smoke boot")

    app = create_app(service=service)
    app.state.service = service
    app.state._service_injected = True

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000", timeout=10.0) as client:
        runner = SmokeTestRunner(
            client=client,
            service=service,
            db_path=db_file,
            timeout=10.0,
            target_display_url="http://127.0.0.1:8000 (Adversarial In-Process)",
        )

        success = await runner.run_all(skip_reset=False)
        assert success is True, "Smoke test runner reported failure"

        # Check summary metrics
        assert runner.summary.total_phases == 7
        assert runner.summary.passed_phases == 7
        assert runner.summary.failed_phases == 0
        assert runner.summary.success_rate_pct == 100.0
        assert runner.summary.verdict == "READY_FOR_MONDAY_OPEN"
        assert runner.summary.total_duration_s > 0.0

        # Check each phase specifically
        phase_names = [r.name for r in runner.results]
        assert "Service Boot & Health Check" in phase_names
        assert "Synthetic Market Regime Transitions" in phase_names
        assert "Out-of-Cadence Rebalance & Execution" in phase_names
        assert "Network Chaos & Alert Banner Toggle" in phase_names
        assert "Operator Pause & Resume Controls" in phase_names
        assert "Institutional Discord v2 Cards" in phase_names
        assert "Guaranteed Clean Reset for Monday" in phase_names

        for r in runner.results:
            assert r.status == "PASS", f"Phase {r.phase_num} ({r.name}) did not pass: {r.error}"
            assert r.latency_ms >= 0.0
            assert r.timestamp != ""

        # Check formatting
        terminal_report = runner.format_terminal_report()
        assert "RESULT: 7 OF 7 PHASES PASSED (100% SUCCESS)" in terminal_report
        assert "VERDICT: SYSTEM VERIFIED & IN PRISTINE READINESS FOR MONDAY MARKET OPEN" in terminal_report

        json_dict = runner.to_json_dict()
        assert json_dict["summary"]["verdict"] == "READY_FOR_MONDAY_OPEN"
        assert len(json_dict["phases"]) == 7

    await service.shutdown()


@pytest.mark.asyncio
async def test_smoke_runner_cli_subprocess(tmp_path: Path):
    """Execute the smoke test CLI script directly via subprocess in-process mode."""
    db_file = tmp_path / "smoke_cli_subp.db"
    cmd = [
        sys.executable,
        "scripts/e2e_smoke_test.py",
        "--in-process",
        "--db-path",
        str(db_file),
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Smoke test CLI failed with code {res.returncode}: {res.stderr}"

    # Parse JSON output
    # Note: stderr or stdout might contain alert logs, find json block
    stdout_text = res.stdout.strip()
    json_start = stdout_text.find("{")
    assert json_start != -1, f"Could not find JSON output: {stdout_text}"
    data = json.loads(stdout_text[json_start:])
    assert data["summary"]["passed_phases"] == 7
    assert data["summary"]["verdict"] == "READY_FOR_MONDAY_OPEN"


# ============================================================================
# Section 2: Reset Idempotence Stress (5x Consecutive Execution)
# ============================================================================

def test_reset_idempotence_5x_consecutive(tmp_path: Path):
    """Run reset_pristine_for_monday.py 5 times in a row and verify state remains pristine."""
    db_file = tmp_path / "idempotent_5x.db"
    ws_root = tmp_path

    # Initially populate with active trading state
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("""
        CREATE TABLE paper_account_state (
            id INTEGER PRIMARY KEY,
            initial_balance REAL, cash REAL, realized_pnl REAL,
            cumulative_fees REAL, cumulative_slippage REAL,
            equity REAL, total_nav REAL, unrealized_pnl REAL,
            created_at TEXT, updated_at TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE paper_positions (
            symbol TEXT PRIMARY KEY, shares REAL, avg_entry_price REAL,
            cost_basis REAL, current_price REAL, market_value REAL,
            unrealized_pnl REAL, weight REAL, created_at TEXT, updated_at TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE paper_trades (
            trade_id TEXT PRIMARY KEY, order_id TEXT, timestamp TEXT,
            symbol TEXT, side TEXT, shares REAL, price REAL,
            notional REAL, fee REAL, slippage REAL, realized_pnl REAL,
            positions_json TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE rebalance_orders (
            id TEXT PRIMARY KEY, timestamp TEXT, symbol TEXT, side TEXT,
            shares REAL, price REAL, notional REAL, target_weight REAL,
            current_weight REAL, status TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE paper_equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
            cash REAL, equity REAL, total_nav REAL, realized_pnl REAL,
            unrealized_pnl REAL, cumulative_fees REAL, positions_json TEXT
        );
    """)

    conn.execute("INSERT INTO paper_account_state VALUES (1, 50000, 20000, 500, 10, 5, 30000, 50000, 100, 't', 't');")
    conn.execute("INSERT INTO paper_positions VALUES ('SPY', 50, 400, 20000, 410, 20500, 500, 0.41, 't', 't');")
    conn.execute("INSERT INTO paper_trades VALUES ('tr1', 'o1', 't', 'SPY', 'BUY', 50, 400, 20000, 5, 2, 0, '{}');")
    conn.execute("INSERT INTO rebalance_orders VALUES ('o1', 't', 'SPY', 'BUY', 50, 400, 20000, 0.4, 0.0, 'FILLED');")
    conn.commit()
    conn.close()

    mgr = PristineResetManager(db_path=str(db_file), workspace_root=str(ws_root))

    for run_idx in range(1, 6):
        # Purge & compact
        mgr.execute_purge()
        busy_code, wal_bytes = mgr.execute_compaction()
        assert busy_code == 0, f"Run {run_idx}: WAL busy code != 0 ({busy_code})"
        assert wal_bytes == 0, f"Run {run_idx}: WAL bytes != 0 ({wal_bytes})"

        # Verify invariants
        invariants = mgr.verify_invariants()
        assert len(invariants) == 17, f"Run {run_idx}: expected 17 invariants, got {len(invariants)}"
        failed = [inv for inv in invariants if not inv.passed]
        assert not failed, f"Run {run_idx}: Invariants failed: {[(f.name, f.expected, f.actual) for f in failed]}"

        # Directly inspect SQLite database state
        chk_conn = sqlite3.connect(str(db_file))
        cursor = chk_conn.cursor()
        cursor.execute("SELECT cash, equity, total_nav, realized_pnl, unrealized_pnl FROM paper_account_state WHERE id = 1;")
        row = cursor.fetchone()
        assert row is not None, f"Run {run_idx}: missing account row id=1"
        assert math.isclose(row[0], 50000.00, abs_tol=1e-5), f"Run {run_idx}: Cash not 50k: {row[0]}"
        assert math.isclose(row[1], 0.00, abs_tol=1e-5), f"Run {run_idx}: Equity not 0: {row[1]}"
        assert math.isclose(row[2], 50000.00, abs_tol=1e-5), f"Run {run_idx}: NAV not 50k: {row[2]}"
        assert math.isclose(row[3], 0.00, abs_tol=1e-5), f"Run {run_idx}: Realized PnL not 0: {row[3]}"
        assert math.isclose(row[4], 0.00, abs_tol=1e-5), f"Run {run_idx}: Unrealized PnL not 0: {row[4]}"

        cursor.execute("SELECT COUNT(*) FROM paper_positions;")
        assert cursor.fetchone()[0] == 0, f"Run {run_idx}: positions not empty"

        cursor.execute("SELECT COUNT(*) FROM paper_trades;")
        assert cursor.fetchone()[0] == 0, f"Run {run_idx}: trades not empty"

        cursor.execute("SELECT COUNT(*) FROM rebalance_orders;")
        assert cursor.fetchone()[0] == 0, f"Run {run_idx}: orders not empty"

        cursor.execute("SELECT COUNT(*) FROM paper_equity_snapshots;")
        assert cursor.fetchone()[0] == 1, f"Run {run_idx}: snapshots count != 1"

        chk_conn.close()


def test_reset_cli_subprocess_5x_idempotence(tmp_path: Path):
    """Run reset script 5 times via subprocess CLI and assert exit code 0 every time."""
    db_file = tmp_path / "cli_idempotent_5x.db"
    for i in range(1, 6):
        cmd = [sys.executable, "scripts/reset_pristine_for_monday.py", "--db-path", str(db_file)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI reset run {i} failed: {res.stderr}"
        assert "VERDICT: 17/17 INVARIANTS PASSED" in res.stdout


# ============================================================================
# Section 3: Injected Corruption Recovery (Adversarial Stress Attacks)
# ============================================================================

def test_injected_corruption_severe_negative_cash_and_nav(tmp_path: Path):
    """Attack Scenario: Injected severe negative cash (-$1,000,000) and negative NAV."""
    db_file = tmp_path / "corrupt_negative_cash.db"
    mgr = PristineResetManager(db_path=str(db_file), workspace_root=str(tmp_path))

    # Initialize baseline
    mgr.execute_purge()

    # Inject toxic corruption: negative cash, negative NAV, huge realized loss
    conn = mgr._get_connection()
    conn.execute("""
        UPDATE paper_account_state SET
            cash = -1000000.00,
            equity = 250000.00,
            total_nav = -750000.00,
            realized_pnl = -800000.00,
            unrealized_pnl = -50000.00
        WHERE id = 1;
    """)
    conn.commit()
    conn.close()

    # Verify that invariants fail before reset
    pre_audit = mgr.verify_invariants()
    failed_names = {inv.name for inv in pre_audit if not inv.passed}
    assert "Account Cash Balance" in failed_names
    assert "Portfolio Total NAV" in failed_names
    assert "Open Positions Equity" in failed_names
    assert "Realized Cumulative P&L" in failed_names
    assert "Unrealized Holdings P&L" in failed_names

    # Execute purge & compaction
    mgr.execute_purge()
    busy_code, wal_bytes = mgr.execute_compaction()
    assert busy_code == 0
    assert wal_bytes == 0

    # Post-purge: must be 100% clean and pass all 17 invariants
    post_audit = mgr.verify_invariants()
    assert len(post_audit) == 17
    for inv in post_audit:
        assert inv.passed, f"Failed post-purge invariant: {inv.name} ({inv.actual} vs {inv.expected})"

    # Double check values
    chk_conn = mgr._get_connection()
    row = chk_conn.execute("SELECT cash, equity, total_nav FROM paper_account_state WHERE id = 1;").fetchone()
    assert row["cash"] == 50000.00
    assert row["equity"] == 0.00
    assert row["total_nav"] == 50000.00
    chk_conn.close()


def test_injected_corruption_orphaned_positions_and_dirty_ledgers(tmp_path: Path):
    """Attack Scenario: Injected orphaned positions, negative shares, ghost trades, corrupt JSON."""
    db_file = tmp_path / "corrupt_orphans.db"
    mgr = PristineResetManager(db_path=str(db_file), workspace_root=str(tmp_path))
    mgr.execute_purge()

    conn = mgr._get_connection()
    # Injected corrupt position with negative shares
    conn.execute("""
        INSERT INTO paper_positions VALUES (
            'GHOST_SYM', -999.0, 150.0, -149850.0, 10.0, -9990.0, 139860.0, -0.2,
            '2026-09-01T00:00:00Z', '2026-09-01T00:00:00Z'
        );
    """)
    # Injected corrupt position with invalid symbol and astronomical shares
    conn.execute("""
        INSERT INTO paper_positions VALUES (
            'CORRUPT_XYZ!@#', 1000000.0, 0.01, 10000.0, 500.0, 500000000.0, 499990000.0, 1.5,
            '2026-09-01T00:00:00Z', '2026-09-01T00:00:00Z'
        );
    """)
    # Injected orphaned trades
    conn.execute("""
        INSERT INTO paper_trades VALUES (
            'orphan_tr_1', 'ghost_order_999', '2026-09-01T00:00:00Z', 'GHOST_SYM',
            'INVALID_ACTION', -999.0, -10.0, 9990.0, -100.0, -50.0, -5000.0, '{INVALID_JSON'
        );
    """)
    # Injected orders
    conn.execute("""
        INSERT INTO rebalance_orders VALUES (
            'corrupt_ord_1', '2026-09-01T00:00:00Z', 'GHOST_SYM', 'BUY', -100.0, -5.0, 500.0, 0.5, 0.0, 'CORRUPTED'
        );
    """)
    # Injected simulated events
    conn.execute("""
        INSERT INTO simulated_events (timestamp, event_type, payload)
        VALUES ('2026-09-01T00:00:00Z', 'MALFORMED_EVENT', '<<<CORRUPT_BLOB>>>');
    """)
    conn.commit()
    conn.close()

    # Pre-audit must fail
    pre_audit = mgr.verify_invariants()
    failed_names = {inv.name for inv in pre_audit if not inv.passed}
    assert "Active Holdings Count" in failed_names
    assert "Paper Trades Count" in failed_names
    assert "Rebalance Orders Count" in failed_names
    assert "Simulated Events Count" in failed_names

    # Purge & Compact
    mgr.execute_purge()
    busy, wal_bytes = mgr.execute_compaction()
    assert busy == 0
    assert wal_bytes == 0

    # Post-audit must pass 17/17
    post_audit = mgr.verify_invariants()
    assert len(post_audit) == 17
    assert all(inv.passed for inv in post_audit)


def test_injected_corruption_with_filesystem_debris(tmp_path: Path):
    """Attack Scenario: Injected filesystem debris (dummy_logs, temp dbs, test logfiles)."""
    db_file = tmp_path / "corrupt_debris.db"
    dummy_logs = tmp_path / "dummy_logs" / "nested" / "sub"
    dummy_logs.mkdir(parents=True, exist_ok=True)
    (dummy_logs / "corrupt_trace.log").write_text("corrupted trace log")
    (dummy_logs / "mock_feed.dump").write_bytes(b"\x00\xff\xfe")

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "smoke_test_run_adversarial.log").write_text("smoke test run")
    (logs_dir / "mock_trade_audit.log").write_text("mock trade")
    (logs_dir / "decisions_live.jsonl").write_text('{"event": "keep"}\n')

    temp_db_1 = tmp_path / "paper_test_adversarial_1.db"
    temp_db_1.write_text("temp sqlite data")
    temp_db_2 = tmp_path / "paper_test_adversarial_2.db-wal"
    temp_db_2.write_text("temp wal data")

    mgr = PristineResetManager(db_path=str(db_file), workspace_root=str(tmp_path))
    mgr.execute_purge()

    # Pre-audit detects debris
    pre_audit = mgr.verify_invariants()
    debris_inv = next(i for i in pre_audit if i.name == "Smoke Test Artifacts Purge")
    assert debris_inv.passed is False

    # Execute compaction & file cleanup
    mgr.execute_compaction()
    cleaned = mgr.purge_file_artifacts(clean_all_logs=False)

    assert "dummy_logs/" in cleaned
    assert not (tmp_path / "dummy_logs").exists()
    assert not temp_db_1.exists()
    assert not temp_db_2.exists()
    assert not (logs_dir / "smoke_test_run_adversarial.log").exists()
    assert not (logs_dir / "mock_trade_audit.log").exists()
    # Decisions file must be preserved
    assert (logs_dir / "decisions_live.jsonl").exists()

    # Post-audit passes
    post_audit = mgr.verify_invariants()
    assert all(i.passed for i in post_audit)


def test_schema_recovery_from_blank_file(tmp_path: Path):
    """Attack Scenario: Empty file or newly created DB with zero schema."""
    db_file = tmp_path / "empty_file.db"
    db_file.touch()

    mgr = PristineResetManager(db_path=str(db_file), workspace_root=str(tmp_path))
    # Execute reset on completely empty DB
    mgr.execute_purge()
    mgr.execute_compaction()

    # Invariants should all pass after schema compatibility initializes tables
    invariants = mgr.verify_invariants()
    assert len(invariants) == 17
    assert all(inv.passed for inv in invariants)


def test_missing_unrealized_pnl_column_migration(tmp_path: Path):
    """Attack Scenario: Legacy database schema missing unrealized_pnl column."""
    db_file = tmp_path / "legacy_schema.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
        CREATE TABLE paper_account_state (
            id INTEGER PRIMARY KEY,
            initial_balance REAL NOT NULL,
            cash REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0.0,
            cumulative_fees REAL NOT NULL DEFAULT 0.0,
            cumulative_slippage REAL NOT NULL DEFAULT 0.0,
            equity REAL NOT NULL DEFAULT 0.0,
            total_nav REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.execute("INSERT INTO paper_account_state VALUES (1, 50000, 50000, 0, 0, 0, 0, 50000, 't', 't');")
    conn.commit()
    conn.close()

    mgr = PristineResetManager(db_path=str(db_file), workspace_root=str(tmp_path))
    mgr.execute_purge()
    mgr.execute_compaction()

    # Verify column was added and invariant verified
    conn = mgr._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(paper_account_state);")
    cols = {row["name"] for row in cursor.fetchall()}
    assert "unrealized_pnl" in cols
    conn.close()

    invariants = mgr.verify_invariants()
    assert all(inv.passed for inv in invariants)


# ============================================================================
# Section 4: Concurrency Stress Under Reset
# ============================================================================

def test_concurrent_reads_during_reset(tmp_path: Path):
    """Concurrent readers querying portfolio state while reset is executing."""
    db_file = tmp_path / "concurrent_reset.db"
    mgr = PristineResetManager(db_path=str(db_file), workspace_root=str(tmp_path))
    mgr.execute_purge()

    stop_event = threading.Event()
    read_errors = []

    def reader_loop():
        while not stop_event.is_set():
            try:
                c = sqlite3.connect(str(db_file), timeout=5.0)
                cur = c.cursor()
                cur.execute("SELECT cash, total_nav FROM paper_account_state WHERE id = 1;")
                r = cur.fetchone()
                if r:
                    assert r[0] > 0
                c.close()
            except Exception as e:
                read_errors.append(str(e))
            time.sleep(0.005)

    reader_thread = threading.Thread(target=reader_loop, daemon=True)
    reader_thread.start()

    try:
        # Perform 3 reset purges while reader is actively reading
        for _ in range(3):
            mgr.execute_purge()
            mgr.execute_compaction()
            time.sleep(0.02)
    finally:
        stop_event.set()
        reader_thread.join(timeout=2.0)

    assert not read_errors, f"Concurrent reads encountered errors: {read_errors[:5]}"

    # Verify final state is pristine
    invariants = mgr.verify_invariants()
    assert all(inv.passed for inv in invariants)
