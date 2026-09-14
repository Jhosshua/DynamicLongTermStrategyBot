"""A skipped 15:50 cadence must be retried, and completions must survive restarts.

Before: handlers skipped by returning early, the scheduler counted that as done,
the done-date lived only in memory, and the service registered every handler
twice (so each cadence ran twice).
"""

from datetime import date, datetime

import pytest
from zoneinfo import ZoneInfo

from bot.service import DynamicStrategyService, ServiceConfig
from strategy_engine.daemon.scheduler import CadenceType, MarketCalendar, MarketScheduler

ET = ZoneInfo("America/New_York")
WED = datetime(2026, 9, 2, 15, 50, tzinfo=ET)  # regular trading day


def _at(minute: int, second: int = 0) -> datetime:
    return WED.replace(minute=minute, second=second)


@pytest.mark.asyncio
async def test_handler_returning_false_is_retried_no_faster_than_interval():
    sched = MarketScheduler(calendar=MarketCalendar(), retry_interval_seconds=60)
    calls = []

    def handler(dt):
        calls.append(dt)
        return len(calls) >= 3  # skip twice, then complete

    sched.on_cadence(CadenceType.DAILY_CLOSE, handler)

    assert await sched.tick(_at(50)) == []
    assert await sched.tick(_at(50, 30)) == []  # inside retry interval: not called
    assert len(calls) == 1
    assert await sched.tick(_at(51)) == []  # retry 2 skips
    assert await sched.tick(_at(52)) == [CadenceType.DAILY_CLOSE]
    assert len(calls) == 3
    assert await sched.tick(_at(53)) == []  # done for today
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_completion_hook_and_restore_prevent_rerun():
    saved = {}
    sched = MarketScheduler(calendar=MarketCalendar(), on_executed=lambda c, d: saved.update({c: d}))
    sched.on_cadence(CadenceType.DAILY_CLOSE, lambda dt: None)
    await sched.tick(_at(51))
    assert saved == {CadenceType.DAILY_CLOSE: WED.date()}

    restarted = MarketScheduler(calendar=MarketCalendar())
    calls = []
    restarted.on_cadence(CadenceType.DAILY_CLOSE, lambda dt: calls.append(dt))
    restarted.restore_last_executed(saved)
    assert await restarted.tick(_at(55)) == []
    assert calls == []


def test_service_registers_each_cadence_once(tmp_path):
    svc = DynamicStrategyService(config=ServiceConfig(db_path=str(tmp_path / "s.db")))
    for cadence in (CadenceType.DAILY_CLOSE, CadenceType.WEEKLY_REBALANCE, CadenceType.MONTHLY_MOMENTUM):
        assert len(svc.scheduler._handlers[cadence]) == 1


@pytest.mark.asyncio
async def test_service_skip_on_dead_feed_is_not_marked_done(tmp_path):
    svc = DynamicStrategyService(config=ServiceConfig(db_path=str(tmp_path / "s.db")))
    # Fresh service: feed is synthetic fallback, so DAILY_CLOSE must skip.
    assert await svc.scheduler.tick(_at(51)) == []
    assert svc.scheduler._last_executed[CadenceType.DAILY_CLOSE] is None
    assert svc.scheduler._next_retry_at[CadenceType.DAILY_CLOSE] is not None


def test_service_persists_and_reloads_completed_cadence(tmp_path):
    db = str(tmp_path / "s.db")
    svc = DynamicStrategyService(config=ServiceConfig(db_path=db))
    svc._save_cadence_done(CadenceType.WEEKLY_REBALANCE, date(2026, 9, 4))
    svc.storage.db.close()

    again = DynamicStrategyService(config=ServiceConfig(db_path=db))
    assert again.scheduler._last_executed[CadenceType.WEEKLY_REBALANCE] == date(2026, 9, 4)
