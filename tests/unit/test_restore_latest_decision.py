"""A restart must reload the last saved signal and allocation.

Before this, every deploy showed regime UNKNOWN until 15:50 and the intraday
breaker lost its Keltner band for the rest of the day.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot.service import DynamicStrategyService, ServiceConfig
from tests.live_feed_helper import simulate_live_feed


@pytest.mark.asyncio
async def test_restart_restores_signal_and_allocation(tmp_path: Path):
    db = str(tmp_path / "restore.db")
    svc = DynamicStrategyService(config=ServiceConfig(db_path=db))
    svc.paper_account.init_schema()
    await simulate_live_feed(svc)
    now = datetime.now(timezone.utc)
    await svc._handle_daily_close(now)
    assert svc._latest_signal is not None and svc._latest_allocation is not None
    saved_regime = svc._latest_signal.regime
    saved_weights = svc._latest_allocation.weights
    saved_indicators = dict(svc._latest_signal.indicators)
    svc.storage.db.close()

    again = DynamicStrategyService(config=ServiceConfig(db_path=db))
    assert again._latest_signal is None
    again._restore_latest_decision()
    assert again._latest_signal is not None
    assert again._latest_signal.regime == saved_regime
    assert again._latest_signal.indicators == saved_indicators
    assert again._latest_allocation is not None
    assert again._latest_allocation.weights == saved_weights
    assert again._last_eval_time is not None


@pytest.mark.asyncio
async def test_stale_decision_is_not_restored(tmp_path: Path):
    db = str(tmp_path / "stale.db")
    svc = DynamicStrategyService(config=ServiceConfig(db_path=db))
    svc.paper_account.init_schema()
    await simulate_live_feed(svc)
    await svc._handle_daily_close(datetime.now(timezone.utc) - timedelta(days=10))
    svc.storage.db.close()

    again = DynamicStrategyService(config=ServiceConfig(db_path=db))
    again._restore_latest_decision()
    assert again._latest_signal is None
    assert again._latest_allocation is None


def test_empty_db_restores_nothing(tmp_path: Path):
    svc = DynamicStrategyService(config=ServiceConfig(db_path=str(tmp_path / "empty.db")))
    svc.paper_account.init_schema()
    svc._restore_latest_decision()
    assert svc._latest_signal is None and svc._latest_allocation is None
