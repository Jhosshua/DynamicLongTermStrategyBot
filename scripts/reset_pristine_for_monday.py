#!/usr/bin/env python3
"""
scripts/reset_pristine_for_monday.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pristine State Reset & Invariant Verification Script for Monday Market Open.
Complies with ORIGINAL_REQUEST.md (R4) and PROJECT.md (Feature 17).

Purges all synthetic smoke-test data, resets paper portfolio to exactly $50,000.00 cash,
truncates SQLite WAL journals, vacua storage, cleans test artifacts, and verifies
all mathematical, persistence, and schema invariants.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class InvariantCheckResult:
    """Individual invariant evaluation result."""
    name: str
    target: str
    expected: str
    actual: str
    passed: bool
    details: str = ""


@dataclass
class ResetReport:
    """Complete summary report of reset operations and invariant verifications."""
    success: bool
    timestamp: str
    mode: str
    db_path: str
    invariants: List[InvariantCheckResult] = field(default_factory=list)
    cleaned_files: List[str] = field(default_factory=list)
    vacuum_performed: bool = False
    wal_checkpoint_code: int = 0
    wal_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "db_path": self.db_path,
            "cleaned_files": self.cleaned_files,
            "vacuum_performed": self.vacuum_performed,
            "wal_checkpoint_code": self.wal_checkpoint_code,
            "wal_bytes": self.wal_bytes,
            "invariants": [asdict(inv) for inv in self.invariants],
        }


class PristineResetManager:
    """Orchestrates atomic database purging, compaction, and invariant verification."""

    def __init__(self, db_path: str = "strategy_engine.db", workspace_root: Optional[str] = None):
        self.db_path = str(Path(db_path).resolve())
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def ensure_schema_compatibility(self, conn: sqlite3.Connection) -> None:
        """Create compatibility tables, views, and columns to guarantee zero schema mismatch errors."""
        cursor = conn.cursor()

        # 0. Ensure base tables exist if database is completely empty
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_account_state';")
        if cursor.fetchone() is None:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_account_state (
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

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_positions';")
        if cursor.fetchone() is None:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_positions (
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

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades';")
        if cursor.fetchone() is None:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
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

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_equity_snapshots';")
        if cursor.fetchone() is None:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    total_nav REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    cumulative_fees REAL NOT NULL,
                    positions_json TEXT NOT NULL
                );
            """)

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rebalance_orders';")
        if cursor.fetchone() is None:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rebalance_orders (
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

        # 1. Ensure unrealized_pnl column exists in paper_account_state
        cursor.execute("PRAGMA table_info(paper_account_state);")
        columns = [row["name"] for row in cursor.fetchall()]
        if columns and "unrealized_pnl" not in columns:
            cursor.execute("ALTER TABLE paper_account_state ADD COLUMN unrealized_pnl REAL NOT NULL DEFAULT 0.00;")

        # 2. Ensure paper_trade_history exists (as table or view)
        cursor.execute("SELECT type, name FROM sqlite_master WHERE name = 'paper_trade_history';")
        res = cursor.fetchone()
        if res is None:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades';")
            if cursor.fetchone():
                cursor.execute("""
                    CREATE VIEW IF NOT EXISTS paper_trade_history AS 
                    SELECT trade_id AS id, order_id, timestamp, symbol, side, shares, price, notional, fee, realized_pnl 
                    FROM paper_trades;
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS trigger_del_pth
                    INSTEAD OF DELETE ON paper_trade_history
                    BEGIN
                        DELETE FROM paper_trades;
                    END;
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trade_history (
                        id TEXT PRIMARY KEY, order_id TEXT, timestamp TEXT, symbol TEXT,
                        side TEXT, shares REAL, price REAL, notional REAL, fee REAL, realized_pnl REAL
                    );
                """)

        # 3. Ensure orders view or table exists
        cursor.execute("SELECT type, name FROM sqlite_master WHERE name = 'orders';")
        if cursor.fetchone() is None:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rebalance_orders';")
            if cursor.fetchone():
                cursor.execute("""
                    CREATE VIEW IF NOT EXISTS orders AS 
                    SELECT id, timestamp, symbol, side, shares, price, notional, target_weight, current_weight, status 
                    FROM rebalance_orders;
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS trigger_del_orders
                    INSTEAD OF DELETE ON orders
                    BEGIN
                        DELETE FROM rebalance_orders;
                    END;
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id TEXT PRIMARY KEY, timestamp TEXT, symbol TEXT, side TEXT,
                        shares REAL, price REAL, notional REAL, status TEXT
                    );
                """)

        # 4. Ensure fills view or table exists
        cursor.execute("SELECT type, name FROM sqlite_master WHERE name = 'fills';")
        if cursor.fetchone() is None:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades';")
            if cursor.fetchone():
                cursor.execute("""
                    CREATE VIEW IF NOT EXISTS fills AS 
                    SELECT trade_id AS id, order_id, timestamp, symbol, side, shares, price, notional, fee, realized_pnl 
                    FROM paper_trades;
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS trigger_del_fills
                    INSTEAD OF DELETE ON fills
                    BEGIN
                        DELETE FROM paper_trades;
                    END;
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fills (
                        id TEXT PRIMARY KEY, order_id TEXT, timestamp TEXT, symbol TEXT,
                        side TEXT, shares REAL, price REAL, fee REAL
                    );
                """)

        # 5. Ensure executions view or table exists
        cursor.execute("SELECT type, name FROM sqlite_master WHERE name = 'executions';")
        if cursor.fetchone() is None:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades';")
            if cursor.fetchone():
                cursor.execute("""
                    CREATE VIEW IF NOT EXISTS executions AS 
                    SELECT trade_id AS id, order_id, timestamp, symbol, side, shares, price, notional, fee, realized_pnl 
                    FROM paper_trades;
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS trigger_del_executions
                    INSTEAD OF DELETE ON executions
                    BEGIN
                        DELETE FROM paper_trades;
                    END;
                """)
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS executions (
                        id TEXT PRIMARY KEY, order_id TEXT, timestamp TEXT, symbol TEXT,
                        side TEXT, shares REAL, price REAL, fee REAL
                    );
                """)

        # 6. Ensure simulated_events table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulated_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL
            );
        """)

    def execute_purge(self) -> None:
        """Executes transactional purge of all trading data and resets account to $50,000.00 cash."""
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            self.ensure_schema_compatibility(conn)
            cursor = conn.cursor()

            # Identify existing tables and views
            cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view');")
            entities = {row["name"] for row in cursor.fetchall()}

            # Purge trading tables
            for tbl in [
                "paper_positions", "paper_trades", "paper_trade_history", "rebalance_orders",
                "orders", "fills", "simulated_events", "portfolio_states", "allocations",
                "signal_snapshots", "regime_events", "executions", "positions", "order_intents"
            ]:
                if tbl in entities:
                    try:
                        cursor.execute(f"DELETE FROM {tbl};")
                    except sqlite3.OperationalError:
                        pass

            # Purge equity snapshots
            if "paper_equity_snapshots" in entities:
                cursor.execute("DELETE FROM paper_equity_snapshots;")

            # Remove synthetic smoke-test market bars
            if "market_bars" in entities:
                cursor.execute("DELETE FROM market_bars WHERE symbol LIKE 'MOCK_%' OR symbol LIKE 'TEST_%';")

            # Reset or insert paper_account_state singleton (id = 1)
            cursor.execute("""
                INSERT INTO paper_account_state (
                    id, initial_balance, cash, realized_pnl, cumulative_fees,
                    cumulative_slippage, equity, total_nav, unrealized_pnl, created_at, updated_at
                ) VALUES (
                    1, 50000.00, 50000.00, 0.00, 0.00,
                    0.00, 0.00, 50000.00, 0.00, ?, ?
                )
                ON CONFLICT(id) DO UPDATE SET
                    initial_balance = 50000.00,
                    cash = 50000.00,
                    equity = 0.00,
                    total_nav = 50000.00,
                    realized_pnl = 0.00,
                    unrealized_pnl = 0.00,
                    cumulative_fees = 0.00,
                    cumulative_slippage = 0.00,
                    updated_at = ?;
            """, (now_iso, now_iso, now_iso))

            # Reset account_balance if exists (contracts.py support)
            if "account_balance" in entities:
                cursor.execute("""
                    INSERT OR REPLACE INTO account_balance (id, cash, realized_pnl, updated_at)
                    VALUES (1, 50000.00, 0.00, ?);
                """, (now_iso,))

            # Insert baseline pristine snapshot
            cursor.execute("""
                INSERT INTO paper_equity_snapshots (
                    timestamp, cash, equity, total_nav, realized_pnl,
                    unrealized_pnl, cumulative_fees, positions_json
                ) VALUES (?, 50000.00, 0.00, 50000.00, 0.00, 0.00, 0.00, '{}');
            """, (now_iso,))

            conn.commit()
        finally:
            conn.close()

    def execute_compaction(self) -> Tuple[int, int]:
        """Executes autocommit VACUUM and PRAGMA wal_checkpoint(TRUNCATE)."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.isolation_level = None  # Autocommit required for VACUUM
            conn.execute("VACUUM;")
            cursor = conn.cursor()
            cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            res = cursor.fetchone()
            busy_code = res[0] if res else 0

            wal_path = Path(self.db_path + "-wal")
            wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0
            return busy_code, wal_bytes
        finally:
            conn.close()

    def purge_file_artifacts(self, clean_all_logs: bool = False) -> List[str]:
        """Removes temporary mock files, fake logs, and smoke-test outputs."""
        cleaned = []

        # 1. Clean dummy_logs/
        dummy_logs_dir = self.workspace_root / "dummy_logs"
        if dummy_logs_dir.exists():
            for p in dummy_logs_dir.glob("**/*"):
                if p.is_file():
                    p.unlink()
                    try:
                        cleaned.append(str(p.relative_to(self.workspace_root)))
                    except ValueError:
                        cleaned.append(p.name)
            shutil.rmtree(dummy_logs_dir, ignore_errors=True)
            cleaned.append("dummy_logs/")

        # 2. Clean smoke logs in logs/
        logs_dir = self.workspace_root / "logs"
        if logs_dir.exists():
            for p in logs_dir.iterdir():
                if p.is_file():
                    if clean_all_logs and p.suffix == ".jsonl":
                        p.unlink()
                        try:
                            cleaned.append(str(p.relative_to(self.workspace_root)))
                        except ValueError:
                            cleaned.append(p.name)
                    elif any(token in p.name.lower() for token in ["smoke", "mock", "test"]):
                        p.unlink()
                        try:
                            cleaned.append(str(p.relative_to(self.workspace_root)))
                        except ValueError:
                            cleaned.append(p.name)

        # 3. Clean temporary sqlite files in workspace root
        for p in self.workspace_root.glob("paper_test_*.db*"):
            p.unlink(missing_ok=True)
            cleaned.append(str(p.name))

        return cleaned

    def verify_invariants(self) -> List[InvariantCheckResult]:
        """Runs complete 17-point suite of mathematical, persistence, and filesystem invariants."""
        results: List[InvariantCheckResult] = []

        if not Path(self.db_path).exists():
            results.append(InvariantCheckResult(
                name="Database File Existence",
                target=self.db_path,
                expected="File exists",
                actual="File missing",
                passed=False,
            ))
            return results

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Check if paper_account_state exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_account_state';")
            if cursor.fetchone() is None:
                results.append(InvariantCheckResult(
                    name="Account State Table",
                    target="paper_account_state",
                    expected="Present",
                    actual="Missing",
                    passed=False,
                ))
            else:
                # Inspect columns
                cursor.execute("PRAGMA table_info(paper_account_state);")
                cols = {row["name"] for row in cursor.fetchall()}
                unreal_expr = "unrealized_pnl" if "unrealized_pnl" in cols else "0.0 AS unrealized_pnl"

                cursor.execute(f"SELECT cash, total_nav, equity, realized_pnl, {unreal_expr} FROM paper_account_state WHERE id = 1;")
                row = cursor.fetchone()
                if row:
                    cash = float(row["cash"])
                    nav = float(row["total_nav"])
                    eq = float(row["equity"])
                    pnl = float(row["realized_pnl"])
                    unreal = float(row["unrealized_pnl"])

                    # INV-01
                    results.append(InvariantCheckResult(
                        name="Account Cash Balance",
                        target="paper_account_state.cash",
                        expected="$50,000.00",
                        actual=f"${cash:,.2f}",
                        passed=math.isclose(cash, 50000.00, abs_tol=1e-5),
                    ))
                    # INV-02
                    results.append(InvariantCheckResult(
                        name="Portfolio Total NAV",
                        target="paper_account_state.total_nav",
                        expected="$50,000.00",
                        actual=f"${nav:,.2f}",
                        passed=math.isclose(nav, 50000.00, abs_tol=1e-5),
                    ))
                    # INV-03
                    results.append(InvariantCheckResult(
                        name="Open Positions Equity",
                        target="paper_account_state.equity",
                        expected="$0.00",
                        actual=f"${eq:,.2f}",
                        passed=math.isclose(eq, 0.00, abs_tol=1e-5),
                    ))
                    # INV-04
                    results.append(InvariantCheckResult(
                        name="Realized Cumulative P&L",
                        target="paper_account_state.realized_pnl",
                        expected="$0.00",
                        actual=f"${pnl:,.2f}",
                        passed=math.isclose(pnl, 0.00, abs_tol=1e-5),
                    ))
                    # INV-05
                    results.append(InvariantCheckResult(
                        name="Unrealized Holdings P&L",
                        target="paper_account_state.unrealized_pnl",
                        expected="$0.00",
                        actual=f"${unreal:,.2f}",
                        passed=math.isclose(unreal, 0.00, abs_tol=1e-5),
                    ))
                else:
                    for inv_name, target in [
                        ("Account Cash Balance", "paper_account_state.cash"),
                        ("Portfolio Total NAV", "paper_account_state.total_nav"),
                        ("Open Positions Equity", "paper_account_state.equity"),
                        ("Realized Cumulative P&L", "paper_account_state.realized_pnl"),
                        ("Unrealized Holdings P&L", "paper_account_state.unrealized_pnl"),
                    ]:
                        results.append(InvariantCheckResult(
                            name=inv_name,
                            target=target,
                            expected="Present record",
                            actual="No row id=1",
                            passed=False,
                        ))

            # Helper for row count checks (INV-06 to INV-13)
            def check_zero_count(entity_name: str, display_name: str):
                cursor.execute("SELECT name FROM sqlite_master WHERE name = ?;", (entity_name,))
                if cursor.fetchone() is not None:
                    cursor.execute(f"SELECT COUNT(*) FROM {entity_name};")
                    cnt = cursor.fetchone()[0]
                    results.append(InvariantCheckResult(
                        name=display_name,
                        target=entity_name,
                        expected="0 rows",
                        actual=f"{cnt} rows",
                        passed=(cnt == 0),
                    ))
                else:
                    results.append(InvariantCheckResult(
                        name=display_name,
                        target=entity_name,
                        expected="0 rows",
                        actual="0 rows (absent)",
                        passed=True,
                    ))

            # INV-06
            check_zero_count("paper_positions", "Active Holdings Count")
            # INV-07
            check_zero_count("paper_trade_history", "Trade History Count")
            # INV-08
            check_zero_count("paper_trades", "Paper Trades Count")
            # INV-09
            check_zero_count("rebalance_orders", "Rebalance Orders Count")
            # INV-10
            check_zero_count("simulated_events", "Simulated Events Count")
            # INV-11
            check_zero_count("orders", "Orders View Count")
            # INV-12
            check_zero_count("fills", "Fills View Count")
            # INV-13
            check_zero_count("executions", "Executions Count")

            # INV-14: WAL Checkpoint State
            cursor.execute("PRAGMA wal_checkpoint(PASSIVE);")
            cp = cursor.fetchone()
            # cp is (busy, log, checkpointed)
            uncheckpointed_frames = (cp[1] - cp[2]) if cp and len(cp) >= 3 else 0
            results.append(InvariantCheckResult(
                name="WAL Journal Checkpoint",
                target="PRAGMA wal_checkpoint",
                expected="0 dirty frames",
                actual=f"{uncheckpointed_frames} dirty frames",
                passed=(uncheckpointed_frames <= 0),
            ))

            # INV-15: WAL Journal File Size
            wal_file = Path(self.db_path + "-wal")
            wal_size = wal_file.stat().st_size if wal_file.exists() else 0
            results.append(InvariantCheckResult(
                name="WAL Journal File Size",
                target=f"{wal_file.name}",
                expected="0 bytes",
                actual=f"{wal_size} bytes",
                passed=(wal_size == 0),
            ))

            # INV-16: Database B-Tree Integrity
            cursor.execute("PRAGMA integrity_check;")
            integ = cursor.fetchone()[0]
            results.append(InvariantCheckResult(
                name="Database B-Tree Integrity",
                target="PRAGMA integrity_check",
                expected="ok",
                actual=str(integ),
                passed=(integ == "ok"),
            ))

            # INV-17: Smoke Test Artifacts Purge
            dummy_logs = self.workspace_root / "dummy_logs"
            dummy_count = len(list(dummy_logs.glob("*"))) if dummy_logs.exists() else 0
            temp_dbs = list(self.workspace_root.glob("paper_test_*.db*"))
            artifact_count = dummy_count + len(temp_dbs)
            results.append(InvariantCheckResult(
                name="Smoke Test Artifacts Purge",
                target="dummy_logs / temp files",
                expected="0 files",
                actual=f"{artifact_count} files",
                passed=(artifact_count == 0),
            ))

        finally:
            conn.close()

        return results


def format_tabular_report(report: ResetReport) -> str:
    """Formats verification results into a high-visibility terminal ASCII table."""
    lines = []
    lines.append("=" * 102)
    lines.append(f"{'PRISTINE MONDAY STATE VERIFICATION REPORT':^102}")
    lines.append("=" * 102)
    lines.append(f"{'Invariant Check':<28} {'Target Entity':<32} {'Expected':<16} {'Actual':<16} {'Status'}")
    lines.append("-" * 102)
    for inv in report.invariants:
        status = "[PASS]" if inv.passed else "[FAIL]"
        lines.append(f"{inv.name:<28} {inv.target:<32} {inv.expected:<16} {inv.actual:<16} {status}")
    lines.append("-" * 102)
    passed_count = sum(1 for inv in report.invariants if inv.passed)
    total_count = len(report.invariants)
    if report.success:
        verdict = f"VERDICT: {passed_count}/{total_count} INVARIANTS PASSED — SYSTEM IN PRISTINE $50,000.00 STATE READY FOR MONDAY OPEN"
    else:
        verdict = f"VERDICT: {total_count - passed_count}/{total_count} INVARIANTS FAILED — SYSTEM NOT IN PRISTINE STATE"
    lines.append(f"{verdict:^102}")
    lines.append("=" * 102)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pristine State Reset & Invariant Verification for Monday Market Open.")
    parser.add_argument("--db-path", default="strategy_engine.db", help="Path to SQLite database.")
    parser.add_argument("--verify-only", action="store_true", help="Audit invariants without modifying database or files.")
    parser.add_argument("--clean-all-logs", action="store_true", help="Purge all decisions_*.jsonl files in logs/ directory.")
    parser.add_argument("--quiet", action="store_true", help="Suppress output except final verdict.")
    parser.add_argument("--json", action="store_true", help="Emit report in JSON format.")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if args.verify_only and not db_path.exists():
        print(f"Error: Target database file '{args.db_path}' does not exist for --verify-only audit.", file=sys.stderr)
        return 2

    manager = PristineResetManager(db_path=args.db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    mode = "VERIFY_ONLY" if args.verify_only else "PURGE_AND_VERIFY"

    cleaned_files = []
    vacuum_performed = False
    busy_code = 0
    wal_bytes = 0

    if not args.verify_only:
        manager.execute_purge()
        busy_code, wal_bytes = manager.execute_compaction()
        vacuum_performed = True
        cleaned_files = manager.purge_file_artifacts(clean_all_logs=args.clean_all_logs)

    invariants = manager.verify_invariants()
    all_passed = bool(invariants) and all(inv.passed for inv in invariants)

    report = ResetReport(
        success=all_passed,
        timestamp=now_iso,
        mode=mode,
        db_path=manager.db_path,
        invariants=invariants,
        cleaned_files=cleaned_files,
        vacuum_performed=vacuum_performed,
        wal_checkpoint_code=busy_code,
        wal_bytes=wal_bytes,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif not args.quiet:
        print(format_tabular_report(report))
        if cleaned_files:
            print(f"Purged {len(cleaned_files)} smoke/temporary artifacts: {', '.join(cleaned_files[:5])}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
