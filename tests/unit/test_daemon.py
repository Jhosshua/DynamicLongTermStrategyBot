"""
tests.unit.test_daemon
~~~~~~~~~~~~~~~~~~~~~~

Unit test suite for MarketCalendar, MarketScheduler, and DecisionDaemon.
"""

import asyncio
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import pytest

from strategy_engine.allocator.rebalancer import PortfolioRebalancer
from strategy_engine.core.models import (
    Bar,
    MarketRegime,
    OrderIntent,
    OrderSide,
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
from strategy_engine.ingestion.client import AlpacaRelayClient
from strategy_engine.signals.regime_detector import SignalEngine
from strategy_engine.storage.audit_logger import JSONLAuditLogger
from strategy_engine.storage.database import Database
from strategy_engine.storage.repositories import StorageService

ET = ZoneInfo("America/New_York")


def test_western_easter_computus():
    # 2024: Easter is March 31 -> Good Friday is March 29
    assert western_easter(2024) == date(2024, 3, 31)
    # 2025: Easter is April 20 -> Good Friday is April 18
    assert western_easter(2025) == date(2025, 4, 20)
    # 2026: Easter is April 5 -> Good Friday is April 3
    assert western_easter(2026) == date(2026, 4, 5)


def test_market_calendar_holidays_2026():
    cal = MarketCalendar()
    holidays = cal.get_holidays(2026)

    # 10 NYSE holidays in 2026
    assert date(2026, 1, 1) in holidays   # New Year's
    assert date(2026, 1, 19) in holidays  # MLK Day (3rd Mon Jan)
    assert date(2026, 2, 16) in holidays  # Washington's Bday (3rd Mon Feb)
    assert date(2026, 4, 3) in holidays   # Good Friday
    assert date(2026, 5, 25) in holidays  # Memorial Day (Last Mon May)
    assert date(2026, 6, 19) in holidays  # Juneteenth
    assert date(2026, 7, 3) in holidays   # July 4 observed (July 4 is Sat -> Fri Jul 3)
    assert date(2026, 9, 7) in holidays   # Labor Day (1st Mon Sep)
    assert date(2026, 11, 26) in holidays # Thanksgiving (4th Thu Nov)
    assert date(2026, 12, 25) in holidays # Christmas Day (Fri)


def test_market_calendar_early_closes():
    cal = MarketCalendar()
    # 2026: Black Friday is Nov 27
    early_closes_2026 = cal.get_early_closes(2026)
    assert date(2026, 11, 27) in early_closes_2026
    # Christmas Eve 2026 (Dec 24 is Thursday)
    assert date(2026, 12, 24) in early_closes_2026


def test_market_calendar_trading_day_and_hours():
    cal = MarketCalendar()
    # Regular trading day: Wednesday Sep 2, 2026
    reg_day = date(2026, 9, 2)
    assert cal.is_trading_day(reg_day) is True
    hours = cal.get_market_hours(reg_day)
    assert hours is not None
    open_dt, close_dt = hours
    assert open_dt == datetime(2026, 9, 2, 9, 30, tzinfo=ET)
    assert close_dt == datetime(2026, 9, 2, 16, 0, tzinfo=ET)

    # Weekend: Saturday Sep 5, 2026
    sat = date(2026, 9, 5)
    assert cal.is_trading_day(sat) is False
    assert cal.get_market_hours(sat) is None

    # Early close: Black Friday Nov 27, 2026
    bf = date(2026, 11, 27)
    assert cal.is_trading_day(bf) is True
    bf_hours = cal.get_market_hours(bf)
    assert bf_hours is not None
    assert bf_hours[1] == datetime(2026, 11, 27, 13, 0, tzinfo=ET)


def test_market_calendar_week_and_month_boundaries():
    cal = MarketCalendar()

    # Sep 1, 2026 is Tuesday -> 1st trading day of Sep 2026
    assert cal.is_first_trading_day_of_month(date(2026, 9, 1)) is True
    assert cal.is_first_trading_day_of_month(date(2026, 9, 2)) is False

    # Sep 30, 2026 is Wednesday -> Last trading day of Sep 2026
    assert cal.is_last_trading_day_of_month(date(2026, 9, 30)) is True
    assert cal.is_last_trading_day_of_month(date(2026, 9, 29)) is False

    # Normal week: Friday Sep 4, 2026 is last trading day of week
    assert cal.is_last_trading_day_of_week(date(2026, 9, 4)) is True
    assert cal.is_last_trading_day_of_week(date(2026, 9, 3)) is False

    # Good Friday week: April 3, 2026 is Good Friday (holiday)
    # Thursday April 2, 2026 is last trading day of that week!
    assert cal.is_last_trading_day_of_week(date(2026, 4, 2)) is True
    assert cal.is_last_trading_day_of_week(date(2026, 4, 3)) is False


@pytest.mark.asyncio
async def test_scheduler_cadences_and_idempotency():
    cal = MarketCalendar()
    sched = MarketScheduler(calendar=cal, daily_close_offset_minutes=10)

    events: List[CadenceType] = []

    def record_event(cadence: CadenceType):
        def handler(dt: datetime):
            events.append(cadence)
        return handler

    sched.on_cadence(CadenceType.DAILY_CLOSE, record_event(CadenceType.DAILY_CLOSE))
    sched.on_cadence(CadenceType.WEEKLY_REBALANCE, record_event(CadenceType.WEEKLY_REBALANCE))
    sched.on_cadence(CadenceType.MONTHLY_MOMENTUM, record_event(CadenceType.MONTHLY_MOMENTUM))

    # Test 1: Tuesday Sep 1, 2026 at 15:40 ET (Before evaluation time -> no triggers)
    t_early = datetime(2026, 9, 1, 15, 40, tzinfo=ET)
    trig_early = await sched.tick(t_early)
    assert len(trig_early) == 0

    # Test 2: Tuesday Sep 1, 2026 at 15:46 ET (1st trading day of month -> Monthly Momentum triggers!)
    t_month = datetime(2026, 9, 1, 15, 46, tzinfo=ET)
    trig_month = await sched.tick(t_month)
    assert CadenceType.MONTHLY_MOMENTUM in trig_month
    assert CadenceType.DAILY_CLOSE not in trig_month

    # Test 3: Tuesday Sep 1, 2026 at 15:52 ET (Daily evaluation time -> Daily Close triggers!)
    t_close = datetime(2026, 9, 1, 15, 52, tzinfo=ET)
    trig_close = await sched.tick(t_close)
    assert CadenceType.DAILY_CLOSE in trig_close

    # Test 4: Idempotency - tick again on same day at 15:55 ET -> NO duplicate triggers!
    t_idem = datetime(2026, 9, 1, 15, 55, tzinfo=ET)
    trig_idem = await sched.tick(t_idem)
    assert len(trig_idem) == 0

    # Test 5: Friday Sep 4, 2026 at 15:50 ET -> Weekly Rebalance AND Daily Close trigger!
    t_fri = datetime(2026, 9, 4, 15, 50, tzinfo=ET)
    trig_fri = await sched.tick(t_fri)
    assert CadenceType.DAILY_CLOSE in trig_fri
    assert CadenceType.WEEKLY_REBALANCE in trig_fri

    # Test 6: Weekend Saturday Sep 5, 2026 -> NO triggers
    t_sat = datetime(2026, 9, 5, 15, 50, tzinfo=ET)
    trig_sat = await sched.tick(t_sat)
    assert len(trig_sat) == 0


@pytest.mark.asyncio
async def test_scheduler_early_close_cadence():
    cal = MarketCalendar()
    sched = MarketScheduler(calendar=cal, daily_close_offset_minutes=10)

    # Black Friday Nov 27, 2026 closes at 13:00 ET -> eval at 12:50 ET!
    t_early_eval = datetime(2026, 11, 27, 12, 51, tzinfo=ET)
    trig = await sched.tick(t_early_eval)
    assert CadenceType.DAILY_CLOSE in trig
    assert CadenceType.WEEKLY_REBALANCE in trig


@pytest.mark.asyncio
async def test_decision_daemon_daily_close_and_rebalance(tmp_path):
    db_path = tmp_path / "daemon_test.db"
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
    # Simulate active upstream connection
    await daemon.client.state_machine.check_initial_health({"upstream": "connected"})

    # Populate dummy daily bars for SPY and QQQ
    base_t = datetime(2026, 9, 1, 16, 0, tzinfo=timezone.utc)
    bars_spy = []
    bars_qqq = []
    for i in range(250):
        t = base_t - timedelta(days=250 - i)
        bars_spy.append(
            Bar(symbol="SPY", timestamp=t, open=450.0 + i * 0.2, high=452.0 + i * 0.2, low=449.0 + i * 0.2, close=451.0 + i * 0.2, volume=1000000)
        )
        bars_qqq.append(
            Bar(symbol="QQQ", timestamp=t, open=380.0 + i * 0.3, high=382.0 + i * 0.3, low=379.0 + i * 0.3, close=381.0 + i * 0.3, volume=800000)
        )
    daemon._cached_daily_bars["SPY"] = bars_spy
    daemon._cached_daily_bars["QQQ"] = bars_qqq

    # Execute daily close
    eval_dt = datetime(2026, 9, 2, 15, 50, tzinfo=ET)
    await daemon._handle_daily_close(eval_dt)

    assert daemon._latest_signal is not None
    assert daemon._latest_signal.regime in (MarketRegime.BULL_AGGRESSIVE, MarketRegime.BULL_NORMAL)
    assert daemon._latest_allocation is not None
    assert abs(sum(daemon._latest_allocation.weights.values()) - 1.0) < 1e-4

    # Execute weekly rebalance (simulated current portfolio 100% SHV)
    daemon._current_portfolio_weights = {"SHV": 1.0}
    orders = await daemon._handle_weekly_rebalance(eval_dt)
    assert len(orders) > 0

    # Verify SELLs sequenced before BUYs
    first_actions = [o.action for o in orders]
    assert first_actions[0] == "SELL"  # SHV sold first to liberate cash
    assert "BUY" in first_actions

    # Check persistence
    latest_saved_alloc = storage.allocations.get_latest()
    assert latest_saved_alloc is not None
    orders_saved = storage.orders.get_by_status("DRY_RUN")
    assert len(orders_saved) == len(orders)

    # Check audit log file
    audit_files = list(log_dir.glob("decisions_*.jsonl"))
    assert len(audit_files) > 0

    await daemon.shutdown("Test complete")


@pytest.mark.asyncio
async def test_decision_daemon_stale_data_hold_suppression(tmp_path):
    db_path = tmp_path / "stale_hold.db"
    log_dir = tmp_path / "logs"

    config = DaemonConfig(
        db_path=str(db_path),
        log_dir=str(log_dir),
    )
    storage = StorageService(db_path)
    audit_logger = JSONLAuditLogger(log_dir)

    daemon = DecisionDaemon(
        config=config,
        storage=storage,
        audit_logger=audit_logger,
    )

    # Force client state machine into STALE_DATA_HOLD
    await daemon.client.state_machine.handle_upstream_disconnected("Feed disconnected")
    assert daemon.client.is_safe_to_rebalance() is False

    eval_dt = datetime(2026, 9, 4, 15, 50, tzinfo=ET)
    orders = await daemon._handle_weekly_rebalance(eval_dt)

    # Mandatory invariant: zero orders produced when in STALE_DATA_HOLD
    assert orders == []

    # Check that suppression audit record was stored
    audit_records = audit_logger.read_date(eval_dt.date())
    assert len(audit_records) > 0
    assert audit_records[-1]["regime"] == MarketRegime.STALE_DATA_HOLD.value
    assert "suppressed" in audit_records[-1]["rationale"].lower()

    await daemon.shutdown("Test complete")


@pytest.mark.asyncio
async def test_decision_daemon_streaming_circuit_breaker(tmp_path):
    db_path = tmp_path / "cb.db"
    log_dir = tmp_path / "logs"

    config = DaemonConfig(db_path=str(db_path), log_dir=str(log_dir))
    daemon = DecisionDaemon(config=config)

    # Setup latest signal with Keltner lower band at 500.0
    daemon._latest_signal = SignalSnapshot(
        timestamp=datetime.now(timezone.utc),
        spy_price=510.0,
        spy_sma50=505.0,
        spy_sma200=490.0,
        realized_vol_20d=0.12,
        vol_scale_factor=1.0,
        drawdown_pct=-0.01,
        regime=MarketRegime.BULL_NORMAL,
        indicators={"keltner_lower_band": 500.0},
    )
    daemon._current_portfolio_weights = {"QQQ": 0.5, "SPY": 0.5}

    # Simulate incoming SPY bar breaching lower band
    bad_bar = Bar(
        symbol="SPY",
        timestamp=datetime.now(timezone.utc),
        open=502.0,
        high=502.0,
        low=495.0,
        close=496.0,  # Below 500.0!
        volume=50000,
    )

    daemon._on_bar_received(bad_bar)
    assert daemon._circuit_breaker_triggered_today is True

    # Give event loop a cycle to run async de-risk task
    await asyncio.sleep(0.05)

    # Verify portfolio was immediately rotated to 100% SHV
    assert daemon._current_portfolio_weights == {"SHV": 1.0}

    await daemon.shutdown("Test complete")
