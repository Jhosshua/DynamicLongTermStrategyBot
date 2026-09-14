"""Stream-silence watchdog must only fire while the market is open.

No bars print overnight or on weekends. The watchdog used to treat that as a
dead feed and flapped live/fallback every 2 minutes (with Discord alerts).
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from bot.feed_manager import FeedManager, FeedManagerConfig, FeedSource


class _FakeWS:
    is_connected = True


async def _run_watchdog_once(tmp_path, market_open: bool) -> FeedManager:
    fm = FeedManager(config=FeedManagerConfig(db_path=str(tmp_path / "wd.db"), stale_timeout_seconds=120.0))
    fm.client.ws_client = _FakeWS()  # socket looks healthy
    fm._market_calendar.is_market_open = lambda _dt: market_open
    fm._feed_source = FeedSource.ALPACA_RELAY
    fm._is_connected = True
    fm._last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=600)
    fm._ensure_synthetic_ticker = lambda: None  # keep the test quiet
    fm._running = True
    task = asyncio.create_task(fm._watchdog_loop())
    await asyncio.sleep(0.8)
    fm._running = False
    task.cancel()
    return fm


@pytest.mark.asyncio
async def test_silence_while_market_closed_stays_live(tmp_path):
    fm = await _run_watchdog_once(tmp_path, market_open=False)
    assert fm.feed_source == FeedSource.ALPACA_RELAY


@pytest.mark.asyncio
async def test_silence_while_market_open_falls_back(tmp_path):
    fm = await _run_watchdog_once(tmp_path, market_open=True)
    assert fm.feed_source == FeedSource.SYNTHETIC_FALLBACK
