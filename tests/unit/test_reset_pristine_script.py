"""
tests.unit.test_reset_pristine_script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for scripts/reset_pristine_for_monday.py:
- PristineResetManager purge and compaction
- 17-point invariant verification suite
- Schema aliasing and compatibility views/triggers
- CLI arguments (--verify-only, --json, --db-path, --clean-all-logs)
- Corruption detection on dirty cash, positions, or P&L
- Artifact file purges
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest

from scripts.reset_pristine_for_monday import (
    InvariantCheckResult,
    PristineResetManager,
    ResetReport,
    format_tabular_report,
    main,
)


@pytest.fixture
def dirty_db_path(tmp_path: Path) -> Path:
    """Creates a temporary SQLite WAL database populated with dirty state."""
    db_file = tmp_path / "dirty_test.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode = WAL;")
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
            unrealized_pnl REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE paper_positions (
            symbol TEXT PRIMARY KEY,
            shares REAL NOT NULL,
            avg_entry_price REAL NOT NULL,
            cost_basis REAL NOT NULL,
            current_price REAL NOT NULL,
            market_value REAL NOT NULL,
            unrealized_pnl REAL NOT NULL,
            weight REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE paper_trades (
            trade_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            shares REAL NOT NULL,
            price REAL NOT NULL,
            notional REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0.0,
            slippage REAL NOT NULL DEFAULT 0.0,
            realized_pnl REAL NOT NULL DEFAULT 0.0,
            positions_json TEXT NOT NULL DEFAULT '{}'
        );
    """)
    conn.execute("""
        CREATE TABLE rebalance_orders (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            shares REAL NOT NULL,
            price REAL NOT NULL,
            notional REAL NOT NULL,
            target_weight REAL NOT NULL,
            current_weight REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING'
        );
    """)

    # Populate with dirty smoke-test data
    conn.execute("""
        INSERT INTO paper_account_state VALUES (
            1, 50000.0, 32000.0, 1500.0, 12.5, 3.2, 18000.0, 50000.0, 200.0,
            '2026-09-10T12:00:00Z', '2026-09-10T12:00:00Z'
        );
    """)
    conn.execute("""
        INSERT INTO paper_positions VALUES (
            'SPY', 35.0, 500.0, 17500.0, 510.0, 17850.0, 350.0, 0.357,
            '2026-09-10T12:00:00Z', '2026-09-10T12:00:00Z'
        );
    """)
    conn.execute("""
        INSERT INTO paper_trades VALUES (
            'tr_01', 'ord_01', '2026-09-10T12:00:00Z', 'SPY', 'BUY', 35.0, 500.0, 17500.0, 1.0, 0.5, 0.0, '{}'
        );
    """)
    conn.execute("""
        INSERT INTO rebalance_orders VALUES (
            'ord_01', '2026-09-10T12:00:00Z', 'SPY', 'BUY', 35.0, 500.0, 17500.0, 0.35, 0.0, 'FILLED'
        );
    """)
    conn.commit()
    conn.close()
    return db_file


def test_reset_manager_on_dirty_database(dirty_db_path: Path, tmp_path: Path):
    """PristineResetManager purges dirty database, vacua WAL, and satisfies all 17 invariants."""
    mgr = PristineResetManager(db_path=str(dirty_db_path), workspace_root=str(tmp_path))

    # Prior to purge: verify-only shows failures
    pre_invariants = mgr.verify_invariants()
    failed_names = [inv.name for inv in pre_invariants if not inv.passed]
    assert "Account Cash Balance" in failed_names
    assert "Open Positions Equity" in failed_names
    assert "Active Holdings Count" in failed_names

    # Execute purge & compaction
    mgr.execute_purge()
    busy_code, wal_bytes = mgr.execute_compaction()
    assert busy_code == 0
    assert wal_bytes == 0

    # Post-purge verification
    post_invariants = mgr.verify_invariants()
    assert len(post_invariants) == 17
    assert all(inv.passed for inv in post_invariants)


def test_verify_only_fails_on_dirty_db(tmp_path: Path):
    """Verify-only correctly detects non-$50k cash or corrupted account state."""
    db_file = tmp_path / "corrupt.db"
    conn = sqlite3.connect(str(db_file))
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
        INSERT INTO paper_account_state VALUES (
            1, 50000.0, 12500.0, 0.0, 0.0, 0.0, 0.0, 12500.0, 0.0, '2026-09-01', '2026-09-01'
        );
    """)
    conn.commit()
    conn.close()

    mgr = PristineResetManager(db_path=str(db_file), workspace_root=str(tmp_path))
    invariants = mgr.verify_invariants()
    cash_inv = next(i for i in invariants if i.name == "Account Cash Balance")
    nav_inv = next(i for i in invariants if i.name == "Portfolio Total NAV")
    assert cash_inv.passed is False
    assert nav_inv.passed is False


def test_schema_aliasing_and_compatibility(tmp_path: Path):
    """Compatibility views/triggers prevent operational errors on prompt entities."""
    db_file = tmp_path / "aliasing.db"
    mgr = PristineResetManager(db_path=str(db_file), workspace_root=str(tmp_path))

    conn = mgr._get_connection()
    mgr.ensure_schema_compatibility(conn)
    cursor = conn.cursor()

    # Entities should exist and be queryable
    cursor.execute("SELECT COUNT(*) FROM paper_trade_history;")
    assert cursor.fetchone()[0] == 0

    cursor.execute("SELECT COUNT(*) FROM orders;")
    assert cursor.fetchone()[0] == 0

    cursor.execute("SELECT COUNT(*) FROM fills;")
    assert cursor.fetchone()[0] == 0

    cursor.execute("SELECT COUNT(*) FROM executions;")
    assert cursor.fetchone()[0] == 0

    cursor.execute("SELECT COUNT(*) FROM simulated_events;")
    assert cursor.fetchone()[0] == 0

    conn.close()


def test_file_artifacts_purge(tmp_path: Path):
    """Artifact purge removes dummy_logs/, test databases, and smoke log files."""
    dummy_dir = tmp_path / "dummy_logs"
    dummy_dir.mkdir(parents=True, exist_ok=True)
    (dummy_dir / "fake_trace.log").write_text("dummy")

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "smoke_test_run_123.log").write_text("smoke test log")
    (logs_dir / "decisions_keep.jsonl").write_text("{}\n")

    temp_db = tmp_path / "paper_test_ephemeral.db"
    temp_db.write_text("temp sqlite")

    mgr = PristineResetManager(db_path=str(tmp_path / "test.db"), workspace_root=str(tmp_path))
    cleaned = mgr.purge_file_artifacts(clean_all_logs=False)

    assert not dummy_dir.exists()
    assert not temp_db.exists()
    assert not (logs_dir / "smoke_test_run_123.log").exists()
    assert (logs_dir / "decisions_keep.jsonl").exists()


def test_corrupted_positions_and_pnl_detection(tmp_path: Path):
    """Verify invariants fail if equity > 0 or realized_pnl != 0."""
    db_file = tmp_path / "corrupt_pnl.db"
    mgr = PristineResetManager(db_path=str(db_file), workspace_root=str(tmp_path))
    mgr.execute_purge()

    # Deliberately contaminate state
    conn = mgr._get_connection()
    conn.execute("UPDATE paper_account_state SET equity = 2500.0, realized_pnl = 450.0 WHERE id = 1;")
    conn.execute("""
        INSERT INTO paper_positions VALUES (
            'AAPL', 10.0, 150.0, 1500.0, 160.0, 1600.0, 100.0, 0.03, '2026-09-01', '2026-09-01'
        );
    """)
    conn.commit()
    conn.close()

    invariants = mgr.verify_invariants()
    failed = {i.name: i for i in invariants if not i.passed}
    assert "Open Positions Equity" in failed
    assert "Realized Cumulative P&L" in failed
    assert "Active Holdings Count" in failed


def test_cli_execution_via_subprocess(tmp_path: Path):
    """Subprocess CLI execution satisfies exit codes and tabular/JSON reporting."""
    db_file = tmp_path / "cli_test.db"

    # 1. Full Reset
    cmd = [sys.executable, "scripts/reset_pristine_for_monday.py", "--db-path", str(db_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "PRISTINE $50,000.00 STATE READY FOR MONDAY OPEN" in res.stdout

    # 2. Verify-Only
    cmd_verify = [sys.executable, "scripts/reset_pristine_for_monday.py", "--db-path", str(db_file), "--verify-only"]
    res_verify = subprocess.run(cmd_verify, capture_output=True, text=True)
    assert res_verify.returncode == 0
    assert "17/17 INVARIANTS PASSED" in res_verify.stdout

    # 3. JSON Output
    cmd_json = [sys.executable, "scripts/reset_pristine_for_monday.py", "--db-path", str(db_file), "--verify-only", "--json"]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True)
    assert res_json.returncode == 0
    data = json.loads(res_json.stdout)
    assert data["success"] is True
    assert len(data["invariants"]) == 17

    # 4. Verify-only on non-existent database exits with status 2
    missing_file = tmp_path / "missing.db"
    cmd_missing = [sys.executable, "scripts/reset_pristine_for_monday.py", "--db-path", str(missing_file), "--verify-only"]
    res_missing = subprocess.run(cmd_missing, capture_output=True, text=True)
    assert res_missing.returncode == 2
