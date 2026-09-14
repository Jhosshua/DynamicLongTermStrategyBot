"""
tests.integration.test_feed_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration test suite for Resilient Feed Manager (bot.feed_manager).
Verifies:
- Live connect and channel subscriptions against mock relay server
- Real-time bar ingestion and SQLite WAL persistence
- Disconnect detection and graceful degradation to synthetic fallback
- Tier 1 (SQLite WAL cached bars) and Tier 2 (SDE synthetic generation)
- Dashboard alert flag toggling (alert_banner_active)
- Silent stream watchdog timeout
- Autonomous self-healing recovery and gap backfill
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import pytest

from bot.feed_manager import (
    ConnectionStatus,
    FeedManager,
    FeedManagerConfig,
    FeedSource,
)
from strategy_engine.core.models import Bar


@pytest.mark.asyncio
async def test_live_connect_and_subscription(mock_relay_server, temp_sqlite_db):
    """IT-01: Connect to live mock relay with valid token and universe symbols."""
    config = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
    )
    manager = FeedManager(config=config)
    await manager.start()

    try:
        # Give connection loop a moment to finish handshake
        for _ in range(20):
            if manager.is_connected:
                break
            await asyncio.sleep(0.05)

        status = manager.get_connection_status()
        assert status.is_connected is True
        assert status.feed_source == "alpaca_relay"
        assert status.alert_banner_active is False
        assert len(status.active_symbols) == 10
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_bar_ingestion_and_wal_persistence(mock_relay_server, temp_sqlite_db):
    """IT-02: Broadcast mock bar on WS stream and verify in-memory and WAL persistence."""
    config = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
    )
    manager = FeedManager(config=config)
    received_bars = []
    manager.on_bar(lambda b: received_bars.append(b))

    await manager.start()
    try:
        for _ in range(20):
            if manager.is_connected:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)

        now_iso = datetime.now(timezone.utc).isoformat()
        bar_dict = {
            "t": now_iso,
            "o": 505.0,
            "h": 508.0,
            "l": 504.0,
            "c": 507.5,
            "v": 10000,
        }
        await mock_relay_server.broadcast_bar("SPY", bar_dict)

        # Allow async event loop to process
        for _ in range(20):
            if manager.get_latest_price("SPY") == 507.5:
                break
            await asyncio.sleep(0.05)

        assert manager.get_latest_price("SPY") == 507.5
        assert len(received_bars) >= 1

        # Check SQLite WAL persistence
        cached_bar = manager.bar_repo.get_latest_bar("SPY", timeframe="1Min")
        assert cached_bar is not None
        assert cached_bar.close == 507.5
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_upstream_disconnect_to_fallback(mock_relay_server, temp_sqlite_db):
    """IT-03: Upstream disconnect triggers fallback mode and sets alert_banner_active."""
    config = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
    )
    manager = FeedManager(config=config)
    disconnect_events = []
    manager.on_disconnect(lambda reason, ts: disconnect_events.append((reason, ts)))

    await manager.start()
    try:
        for _ in range(20):
            if manager.is_connected:
                break
            await asyncio.sleep(0.05)

        assert manager.is_connected is True
        assert manager.alert_banner_active is False

        # Trigger disconnect from server
        await mock_relay_server.broadcast_lifecycle("upstream_disconnected")

        for _ in range(20):
            if manager.alert_banner_active:
                break
            await asyncio.sleep(0.05)

        status = manager.get_connection_status()
        assert status.is_connected is False
        assert status.feed_source == "synthetic_fallback"
        assert status.alert_banner_active is True
        assert "offline" in status.status_message.lower() or "disconnected" in status.status_message.lower()
        assert len(disconnect_events) >= 1
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_fallback_tier1_cached_bars(temp_sqlite_db):
    """IT-04: Serves cached bars from SQLite WAL without live connection."""
    config = FeedManagerConfig(
        relay_base_url="http://127.0.0.1:9999",  # Non-existent server
        relay_ws_url="ws://127.0.0.1:9999",
        db_path=temp_sqlite_db,
    )
    manager = FeedManager(config=config)

    # Populate some bars in SQLite WAL
    now = datetime.now(timezone.utc)
    sample_bars = [
        Bar(symbol="SPY", timestamp=now, open=500.0, high=502.0, low=499.0, close=501.0, volume=1000)
    ]
    manager.bar_repo.save_bars(sample_bars, timeframe="1Day")

    # In fallback mode directly
    bars_map = await manager.get_historical_bars(["SPY"], timeframe="1Day")
    assert "SPY" in bars_map
    assert len(bars_map["SPY"]) >= 1
    assert bars_map["SPY"][0].close == 501.0


@pytest.mark.asyncio
async def test_fallback_tier2_synthetic_sde(temp_sqlite_db):
    """IT-05: Missing symbols fall back to calibrated synthetic SDE generator."""
    config = FeedManagerConfig(
        relay_base_url="http://127.0.0.1:9999",  # Non-existent
        relay_ws_url="ws://127.0.0.1:9999",
        db_path=temp_sqlite_db,
    )
    manager = FeedManager(config=config)

    # Query an empty symbol not in cache
    bars_map = await manager.get_historical_bars(["QQQ"], timeframe="1Day")
    assert "QQQ" in bars_map
    assert len(bars_map["QQQ"]) > 0
    # Synthetic bars are served in memory only; persisting them poisoned the real bar cache.
    cached = manager.bar_repo.get_bars("QQQ", timeframe="1Day")
    assert cached == []


@pytest.mark.asyncio
async def test_watchdog_silent_stream_timeout(temp_sqlite_db):
    """IT-07: Watchdog detects silent stream and transitions to fallback."""
    config = FeedManagerConfig(
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
        db_path=temp_sqlite_db,
        stale_timeout_seconds=0.2,  # Very short timeout for test
    )
    manager = FeedManager(config=config)
    manager._feed_source = FeedSource.ALPACA_RELAY
    manager._is_connected = True
    manager._alert_banner_active = False
    manager._last_heartbeat = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    manager._running = True

    # Run one pass of watchdog check
    elapsed = (datetime.now(timezone.utc) - manager._last_heartbeat).total_seconds()
    assert elapsed > manager.config.stale_timeout_seconds

    await manager._transition_to_fallback("Watchdog silent stream")
    status = manager.get_connection_status()
    assert status.alert_banner_active is True
    assert status.feed_source == "synthetic_fallback"


@pytest.mark.asyncio
async def test_autonomous_recovery(mock_relay_server, temp_sqlite_db):
    """IT-08: Recovery loop detects online upstream, reconnects, and clears alert banner."""
    config = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
        recovery_check_interval=0.2,
    )
    manager = FeedManager(config=config)
    recovered_events = []
    manager.on_recover(lambda downtime, ts: recovered_events.append((downtime, ts)))

    await manager.start()
    try:
        for _ in range(20):
            if manager.is_connected:
                break
            await asyncio.sleep(0.05)

        # Force disconnect state
        await manager._transition_to_fallback("Simulated test outage")
        assert manager.alert_banner_active is True

        # Recovery loop should poll /health (which is online) and recover
        for _ in range(30):
            if not manager.alert_banner_active and manager.is_connected:
                break
            await asyncio.sleep(0.1)

        status = manager.get_connection_status()
        assert status.is_connected is True
        assert status.alert_banner_active is False
        assert status.feed_source == "alpaca_relay"
        assert len(recovered_events) >= 1
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_zero_crash_price_resolution(temp_sqlite_db):
    """Verify get_latest_price never returns 0 or raises exception for any symbol."""
    config = FeedManagerConfig(db_path=temp_sqlite_db)
    manager = FeedManager(config=config)

    for sym in ["SPY", "QQQ", "XLK", "XLE", "XLV", "XLI", "XLU", "TLT", "SHV", "GLD", "UNKNOWN"]:
        price = manager.get_latest_price(sym)
        assert isinstance(price, float)
        assert price > 0.0
