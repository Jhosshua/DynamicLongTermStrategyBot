"""
tests.adversarial.test_m4_adversarial_daemon_cli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adversarial empirical verification suite for Milestone M4:
1. US Equity Market Calendar Edge Cases:
   - DST boundaries (spring forward / fall back) across 2024-2027
   - Leap year boundaries (2024, 2028 Feb 29 trading days and month-ends)
   - All 10 NYSE observed holidays & weekend shifting rules (NYSE Rule 7.2)
   - Scheduled early closes (13:00 ET) vs holiday collisions
   - 1-second boundary checks around 09:30, 13:00, 15:45, 15:50, 16:00 ET
2. STALE_DATA_HOLD Daemon Resilience:
   - Upstream disconnected lifecycle event during rebalance cadence
   - Strict 0-order suppression
   - Dual persistence of HOLD audit event (JSONL + SQLite)
   - Watchdog timeout transition to STALE_DATA_HOLD
3. CLI Fuzzing & Robustness:
   - Fuzz testing of dry-run, rebalance, backtest, export-metrics, daemon
   - Assertion of graceful exit without unhandled exception tracebacks (ValueError,
     AttributeError, OperationalError, FileNotFoundError, OSError)
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner

from strategy_engine.allocator.rebalancer import PortfolioRebalancer
from strategy_engine.cli.main import app
from strategy_engine.core.models import (
    Bar,
    MarketRegime,
    OrderIntent,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.daemon.daemon import DaemonConfig, DecisionDaemon
from strategy_engine.daemon.scheduler import (
    CadenceType,
    MarketCalendar,
    MarketScheduler,
    western_easter,
)
from strategy_engine.storage.audit_logger import JSONLAuditLogger
from strategy_engine.storage.database import Database
from strategy_engine.storage.repositories import StorageService

ET = ZoneInfo("America/New_York")
runner = CliRunner()


# ============================================================================
# PART 1: US EQUITY MARKET CALENDAR & SCHEDULER ADVERSARIAL TESTS
# ============================================================================

class TestMarketCalendarEdgeCases:
    """Stress-test MarketCalendar against edge cases, DST, leap years, and holidays."""

    def test_dst_spring_forward_boundaries(self):
        """Verify market open/close and UTC-to-ET conversion across DST spring forward (March)."""
        cal = MarketCalendar()

        # 2026: DST spring forward on Sunday March 8 (clocks jump 2:00 -> 3:00)
        # Friday March 6, 2026 is EST (UTC-5)
        fri_open_utc = datetime(2026, 3, 6, 14, 30, tzinfo=timezone.utc)
        fri_open_et = cal.to_et(fri_open_utc)
        assert fri_open_et == datetime(2026, 3, 6, 9, 30, tzinfo=ET)
        assert cal.is_market_open(fri_open_utc) is True
        assert cal.is_market_open(fri_open_utc - timedelta(seconds=1)) is False

        fri_close_utc = datetime(2026, 3, 6, 21, 0, tzinfo=timezone.utc)
        assert cal.to_et(fri_close_utc) == datetime(2026, 3, 6, 16, 0, tzinfo=ET)
        assert cal.is_market_open(fri_close_utc) is False
        assert cal.is_market_open(fri_close_utc - timedelta(seconds=1)) is True

        # Sunday March 8, 2026 is weekend (market closed despite clock jump)
        sun_utc = datetime(2026, 3, 8, 14, 30, tzinfo=timezone.utc)
        assert cal.is_trading_day(sun_utc) is False
        assert cal.is_market_open(sun_utc) is False

        # Monday March 9, 2026 is EDT (UTC-4) - Notice 09:30 ET is now 13:30 UTC!
        mon_open_utc = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)
        mon_open_et = cal.to_et(mon_open_utc)
        assert mon_open_et == datetime(2026, 3, 9, 9, 30, tzinfo=ET)
        assert cal.is_market_open(mon_open_utc) is True
        assert cal.is_market_open(mon_open_utc - timedelta(seconds=1)) is False

        mon_close_utc = datetime(2026, 3, 9, 20, 0, tzinfo=timezone.utc)
        assert cal.to_et(mon_close_utc) == datetime(2026, 3, 9, 16, 0, tzinfo=ET)
        assert cal.is_market_open(mon_close_utc) is False
        assert cal.is_market_open(mon_close_utc - timedelta(seconds=1)) is True

    def test_dst_fall_back_boundaries(self):
        """Verify market open/close and UTC-to-ET conversion across DST fall back (November)."""
        cal = MarketCalendar()

        # 2026: DST fall back on Sunday Nov 1 (clocks repeat 1:00 -> 1:00)
        # Friday Oct 30, 2026 is EDT (UTC-4)
        fri_open_utc = datetime(2026, 10, 30, 13, 30, tzinfo=timezone.utc)
        assert cal.to_et(fri_open_utc) == datetime(2026, 10, 30, 9, 30, tzinfo=ET)
        assert cal.is_market_open(fri_open_utc) is True
        assert cal.is_market_open(fri_open_utc - timedelta(seconds=1)) is False

        # Monday Nov 2, 2026 is EST (UTC-5) - Notice 09:30 ET is now 14:30 UTC!
        mon_open_utc = datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc)
        assert cal.to_et(mon_open_utc) == datetime(2026, 11, 2, 9, 30, tzinfo=ET)
        assert cal.is_market_open(mon_open_utc) is True
        assert cal.is_market_open(mon_open_utc - timedelta(seconds=1)) is False

    def test_leap_year_boundaries(self):
        """Verify leap year Feb 29 handling and last-trading-day detection."""
        cal = MarketCalendar()

        # 2024 is a leap year; Feb 29, 2024 was a Thursday
        d_2024_02_29 = date(2024, 2, 29)
        assert cal.is_trading_day(d_2024_02_29) is True
        assert cal.is_last_trading_day_of_month(d_2024_02_29) is True
        assert cal.is_last_trading_day_of_month(date(2024, 2, 28)) is False
        assert cal.is_first_trading_day_of_month(date(2024, 3, 1)) is True

        # 2028 is a leap year; Feb 29, 2028 is a Tuesday
        d_2028_02_29 = date(2028, 2, 29)
        assert cal.is_trading_day(d_2028_02_29) is True
        assert cal.is_last_trading_day_of_month(d_2028_02_29) is True
        assert cal.is_last_trading_day_of_month(date(2028, 2, 28)) is False

        # 2025 is a non-leap year; Feb 28, 2025 is Friday (last trading day)
        d_2025_02_28 = date(2025, 2, 28)
        assert cal.is_trading_day(d_2025_02_28) is True
        assert cal.is_last_trading_day_of_month(d_2025_02_28) is True

    def test_all_10_nyse_holidays(self):
        """Verify all 10 NYSE observed holidays across multiple calendar years."""
        cal = MarketCalendar()

        # For years where Jan 1 does not fall on Saturday (2024-2027), exactly 10 holidays are observed
        for year in [2024, 2025, 2026, 2027]:
            holidays = cal.get_holidays(year)
            assert len(holidays) == 10, f"Expected 10 holidays in {year}, got {len(holidays)}: {holidays}"
            for h in holidays:
                assert h.weekday() < 5, f"Observed holiday {h} falls on a weekend!"
                assert cal.is_trading_day(h) is False
                assert cal.get_market_hours(h) is None

        # Verify Saturday observation rule:
        # In 2026, July 4 is Saturday -> observed Friday July 3
        assert date(2026, 7, 3) in cal.get_holidays(2026)
        # In 2027, Dec 25 is Saturday -> observed Friday Dec 24
        assert date(2027, 12, 24) in cal.get_holidays(2027)
        # In 2027, July 4 is Sunday -> observed Monday July 5
        assert date(2027, 7, 5) in cal.get_holidays(2027)

    def test_scheduled_early_closes_and_collision_avoidance(self):
        """Verify early close days (13:00 ET) and ensure no collision with full holidays."""
        cal = MarketCalendar()

        for year in [2024, 2025, 2026, 2027, 2028]:
            early_closes = cal.get_early_closes(year)
            holidays = cal.get_holidays(year)
            for ec in early_closes:
                # Early close days must be open trading days, not full holidays or weekends
                assert ec not in holidays, f"Early close {ec} in {year} collides with full holiday!"
                assert ec.weekday() < 5, f"Early close {ec} falls on weekend!"
                hours = cal.get_market_hours(ec)
                assert hours is not None
                assert hours[0] == datetime(ec.year, ec.month, ec.day, 9, 30, tzinfo=ET)
                assert hours[1] == datetime(ec.year, ec.month, ec.day, 13, 0, tzinfo=ET)

        # In 2026, July 4 is Saturday -> July 3 is full holiday, NOT early close
        assert date(2026, 7, 3) not in cal.get_early_closes(2026)
        # In 2026, Black Friday (Nov 27) and Christmas Eve (Dec 24) are early closes
        assert date(2026, 11, 27) in cal.get_early_closes(2026)
        assert date(2026, 12, 24) in cal.get_early_closes(2026)

    @pytest.mark.asyncio
    async def test_one_second_boundary_checks(self):
        """Stress-test 1-second tick boundaries at open, early close, daily close, and regular close."""
        cal = MarketCalendar()
        sched = MarketScheduler(calendar=cal, daily_close_offset_minutes=10)

        # 1. Open boundary on regular day (Wednesday Sep 2, 2026)
        t_pre_open = datetime(2026, 9, 2, 9, 29, 59, tzinfo=ET)
        t_open = datetime(2026, 9, 2, 9, 30, 0, tzinfo=ET)
        t_post_open = datetime(2026, 9, 2, 9, 30, 1, tzinfo=ET)
        assert cal.is_market_open(t_pre_open) is False
        assert cal.is_market_open(t_open) is True
        assert cal.is_market_open(t_post_open) is True

        # 2. Regular close boundary (16:00 ET)
        t_pre_close = datetime(2026, 9, 2, 15, 59, 59, tzinfo=ET)
        t_close = datetime(2026, 9, 2, 16, 0, 0, tzinfo=ET)
        t_post_close = datetime(2026, 9, 2, 16, 0, 1, tzinfo=ET)
        assert cal.is_market_open(t_pre_close) is True
        assert cal.is_market_open(t_close) is False
        assert cal.is_market_open(t_post_close) is False

        # 3. Early close boundary on Black Friday (Nov 27, 2026, closes at 13:00 ET)
        t_ec_pre = datetime(2026, 11, 27, 12, 59, 59, tzinfo=ET)
        t_ec_exact = datetime(2026, 11, 27, 13, 0, 0, tzinfo=ET)
        assert cal.is_market_open(t_ec_pre) is True
        assert cal.is_market_open(t_ec_exact) is False

        # 4. Scheduler 1-second cadence trigger on regular day (Eval is 15:50:00 ET)
        sched_reg = MarketScheduler(calendar=cal, daily_close_offset_minutes=10)
        trig_pre = await sched_reg.tick(datetime(2026, 9, 2, 15, 49, 59, tzinfo=ET))
        assert trig_pre == []
        trig_exact = await sched_reg.tick(datetime(2026, 9, 2, 15, 50, 0, tzinfo=ET))
        assert trig_exact == [CadenceType.DAILY_CLOSE]
        # Idempotency at 15:50:01
        trig_post = await sched_reg.tick(datetime(2026, 9, 2, 15, 50, 1, tzinfo=ET))
        assert trig_post == []

        # 5. Early close evaluation trigger (12:50:00 ET on Black Friday)
        sched_ec = MarketScheduler(calendar=cal, daily_close_offset_minutes=10)
        trig_ec_pre = await sched_ec.tick(datetime(2026, 11, 27, 12, 49, 59, tzinfo=ET))
        assert trig_ec_pre == []
        trig_ec_exact = await sched_ec.tick(datetime(2026, 11, 27, 12, 50, 0, tzinfo=ET))
        assert CadenceType.DAILY_CLOSE in trig_ec_exact
        assert CadenceType.WEEKLY_REBALANCE in trig_ec_exact


# ============================================================================
# PART 2: STALE_DATA_HOLD DAEMON RESILIENCE
# ============================================================================

class TestStaleDataHoldDaemonResilience:
    """Stress-test DecisionDaemon under upstream disconnect and stale holding state."""

    @pytest.mark.asyncio
    async def test_upstream_disconnected_suppresses_orders_and_logs_hold(self, tmp_path):
        """Simulate upstream_disconnected event; assert 0 orders and HOLD audit event."""
        db_path = tmp_path / "stale_hold_audit.db"
        log_dir = tmp_path / "logs"

        config = DaemonConfig(
            db_path=str(db_path),
            log_dir=str(log_dir),
            dry_run=True,
        )
        storage = StorageService(db_path)
        audit_logger = JSONLAuditLogger(log_dir)

        daemon = DecisionDaemon(
            config=config,
            storage=storage,
            audit_logger=audit_logger,
        )

        # 1. Establish initial connected state
        await daemon.client.state_machine.check_initial_health({"upstream": "connected"})
        assert daemon.client.is_safe_to_rebalance() is True

        # 2. Simulate upstream_disconnected lifecycle event
        await daemon.client.handle_lifecycle_event("upstream_disconnected")
        assert daemon.client.is_safe_to_rebalance() is False
        assert daemon.client.get_connection_status() == "STALE_DATA_HOLD"

        # 3. Daemon evaluates weekly rebalance cadence
        eval_dt = datetime(2026, 9, 4, 15, 50, tzinfo=ET)
        orders = await daemon._handle_weekly_rebalance(eval_dt)

        # Assertion 1: Strict zero orders emitted
        assert orders == [], f"Expected zero orders emitted during STALE_DATA_HOLD, got {orders}"

        # Assertion 2: JSONL audit logger recorded HOLD audit event
        records = audit_logger.read_date(eval_dt.date())
        assert len(records) > 0, "No JSONL audit records created!"
        latest_record = records[-1]
        assert latest_record["regime"] == MarketRegime.STALE_DATA_HOLD.value
        assert "suppressed" in latest_record["rationale"].lower()
        assert "hold" in latest_record["rationale"].lower()
        assert latest_record["orders"] == []

        # Assertion 3: SQLite allocations table recorded suppressed decision audit
        alloc_records = storage.allocations.get_range()
        assert len(alloc_records) > 0
        latest_db_alloc = alloc_records[-1]
        assert latest_db_alloc["regime"] == MarketRegime.STALE_DATA_HOLD.value
        assert "SUPPRESSED_STALE_DATA_HOLD" in latest_db_alloc["rationale"]

        await daemon.shutdown("Resilience verification complete")

    @pytest.mark.asyncio
    async def test_watchdog_timeout_triggers_stale_hold_and_suppression(self, tmp_path):
        """Simulate silent data feed exceeding stale timeout (> 300s); verify auto-transition."""
        db_path = tmp_path / "watchdog.db"
        log_dir = tmp_path / "logs"

        config = DaemonConfig(
            db_path=str(db_path),
            log_dir=str(log_dir),
            stale_timeout_seconds=300.0,
        )
        storage = StorageService(db_path)
        audit_logger = JSONLAuditLogger(log_dir)

        daemon = DecisionDaemon(config=config, storage=storage, audit_logger=audit_logger)

        # Stream initialized and connected
        await daemon.client.state_machine.check_initial_health({"upstream": "connected"})
        t0 = datetime(2026, 9, 4, 15, 0, 0, tzinfo=timezone.utc)
        daemon.client.state_machine.record_message(t0)
        assert daemon.client.is_safe_to_rebalance() is True

        # Watchdog check 301 seconds later
        t_stale = t0 + timedelta(seconds=301)
        triggered = await daemon.client.state_machine.check_watchdog(t_stale)
        assert triggered is True
        assert daemon.client.is_safe_to_rebalance() is False
        assert daemon.client.get_connection_status() == "STALE_DATA_HOLD"

        # Rebalance attempted -> zero orders emitted
        eval_dt = datetime(2026, 9, 4, 15, 50, tzinfo=ET)
        orders = await daemon._handle_weekly_rebalance(eval_dt)
        assert orders == []

        await daemon.shutdown("Watchdog test complete")


# ============================================================================
# PART 3: CLI FUZZING & CRASH MINING
# ============================================================================

class TestCLIFuzzingAndRobustness:
    """Fuzz all CLI commands with invalid inputs, empty databases, and non-existent files.
    
    Robustness criterion: All invalid invocations must exit with a non-zero exit code
    and a helpful error message WITHOUT unhandled exception tracebacks (e.g. ValueError,
    AttributeError, sqlite3.OperationalError, FileNotFoundError, OSError).
    """

    def test_cli_invalid_option_flags_exit_gracefully(self):
        """Assert that unknown CLI flags are rejected with code 2 and usage help."""
        commands = ["dry-run", "daemon", "rebalance", "backtest", "status", "export-metrics"]
        for cmd in commands:
            res = runner.invoke(app, [cmd, "--non-existent-option"])
            assert res.exit_code != 0
            assert "no such option" in res.output.lower() or "error" in res.output.lower()
            assert "traceback" not in res.output.lower()

    def test_cli_dry_run_invalid_scenario_crashes(self):
        """dry-run --scenario invalid must exit cleanly without unhandled ValueError."""
        res = runner.invoke(app, ["dry-run", "--scenario", "nonexistent_stress"])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in dry-run invalid scenario:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_dry_run_invalid_as_of_crashes(self):
        """dry-run --as-of invalid-date must exit cleanly without unhandled ValueError."""
        res = runner.invoke(app, ["dry-run", "--as-of", "invalid-iso-date"])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in dry-run invalid as-of:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_dry_run_nondict_weights_crashes(self):
        """dry-run -w 'string' must exit cleanly without unhandled AttributeError."""
        res = runner.invoke(app, ["dry-run", "-w", '"hello"'])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in dry-run nondict weights:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_dry_run_empty_db_crashes(self):
        """dry-run --scenario none --db-path '' must exit cleanly without OperationalError."""
        res = runner.invoke(app, ["dry-run", "--scenario", "none", "--db-path", ""])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in dry-run empty db:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_rebalance_invalid_scenario_crashes(self):
        """rebalance --scenario invalid must exit cleanly without unhandled ValueError."""
        res = runner.invoke(app, ["rebalance", "--scenario", "nonexistent_stress"])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in rebalance invalid scenario:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_rebalance_nondict_weights_crashes(self):
        """rebalance -w 'string' must exit cleanly without unhandled AttributeError."""
        res = runner.invoke(app, ["rebalance", "-w", '"not_a_dict"'])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in rebalance nondict weights:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_rebalance_nonfloat_weights_crashes(self):
        """rebalance -w '{\"SPY\": \"bad\"}' must exit cleanly without unhandled ValueError."""
        res = runner.invoke(app, ["rebalance", "-w", '{"SPY": "bad_float"}'])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in rebalance nonfloat weights:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_backtest_invalid_scenario_crashes(self):
        """backtest --scenario invalid must exit cleanly without unhandled ValueError."""
        res = runner.invoke(app, ["backtest", "--scenario", "nonexistent_stress"])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in backtest invalid scenario:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_backtest_invalid_export_csv_crashes(self):
        """backtest --export-csv /nonexistent/file.csv must exit cleanly without FileNotFoundError."""
        res = runner.invoke(app, ["backtest", "--export-csv", "/nonexistent_parent_dir/out.csv"])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in backtest invalid export-csv:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_export_metrics_empty_db_crashes(self):
        """export-metrics --db-path '' must exit cleanly without OperationalError."""
        res = runner.invoke(app, ["export-metrics", "--db-path", ""])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in export-metrics empty db:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_export_metrics_nonexistent_db_crashes(self):
        """export-metrics --db-path nonexistent must exit cleanly without OperationalError."""
        res = runner.invoke(app, ["export-metrics", "--db-path", "/tmp/definitely_nonexistent_db_123.sqlite"])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in export-metrics nonexistent db:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_export_metrics_invalid_date_crashes(self):
        """export-metrics --start-date invalid must exit cleanly without ValueError."""
        res = runner.invoke(app, ["export-metrics", "--start-date", "invalid-date"])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in export-metrics invalid date:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_export_metrics_invalid_output_path_crashes(self):
        """export-metrics --output /nonexistent/out.json must exit cleanly without OperationalError/OSError."""
        res = runner.invoke(app, ["export-metrics", "--output", "/nonexistent_parent_dir/out.json"])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in export-metrics invalid output:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"

    def test_cli_daemon_invalid_interval_crashes(self):
        """daemon --interval invalid must exit cleanly without unhandled ValueError."""
        res = runner.invoke(app, ["daemon", "--interval", "not_a_valid_number"])
        assert res.exit_code != 0
        assert "traceback" not in res.output.lower(), f"Unhandled traceback in daemon invalid interval:\n{res.output}"
        assert res.exception is None or isinstance(res.exception, SystemExit), f"Unhandled exception: {type(res.exception).__name__}"
