"""
tests.adversarial.test_m1_feed_manager_chaos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adversarial Chaos and Network Fault Injection Suite for DataFeedManager (M1):
1. Upstream disconnect injection:
   - Server broadcasts `upstream_disconnected`.
   - Verifies `alert_banner_active` turns True immediately, feed_source transitions to synthetic_fallback, and is_connected turns False.
2. Slow client eviction injection (Close Code 1013):
   - Server terminates client socket with close code 1013 ("too slow").
   - Stress-tests whether DataFeedManager immediately detects eviction, transitions to fallback, and raises alert_banner_active.
3. Sudden socket drop injection:
   - Socket is abruptly terminated without upstream notice.
   - Evaluates immediate vs delayed watchdog detection and banner behavior.
4. Two-tier fallback simulation under disconnect:
   - Tier 1: SQLite WAL cache retrieval.
   - Tier 2: SDE synthetic generator on cache misses.
   - Zero-crash guarantees on get_latest_price() and get_latest_prices().
   - Synthetic ticker loop maintains dynamic price movement and emits bars.
5. Reconnection and recovery injection:
   - Upstream recovery event or auto-recovery loop restores connection.
   - Verifies alert_banner_active resets to False, is_connected returns to True, feed_source returns to alpaca_relay.
6. Synthetic ticker continuous emission during fallback:
   - Verifies background ticker generates moving prices and fires on_bar callbacks.
7. REST failure fallback triggering:
   - Verifies REST errors during historical bar query safely trigger fallback and return Tier 1/2 data.
8. Connection flapping / oscillating stress:
   - Rapid connect/disconnect cycling does not corrupt state or crash the manager.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List
import pytest

from bot.feed_manager import (
    ConnectionStatus,
    DataFeedManager,
    FeedManager,
    FeedManagerConfig,
    FeedSource,
)
from strategy_engine.core.models import Bar


@pytest.mark.asyncio
async def test_upstream_disconnected_immediate_banner(mock_relay_server, temp_sqlite_db):
    """Chaos-01: Verify upstream_disconnected event immediately sets alert_banner_active=True."""
    config = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
    )
    manager = DataFeedManager(config=config)
    disconnect_notifications: List[tuple] = []
    manager.on_disconnect(lambda reason, ts: disconnect_notifications.append((reason, ts)))

    await manager.start()
    try:
        # Wait for live connection
        for _ in range(30):
            if manager.is_connected:
                break
            await asyncio.sleep(0.05)

        assert manager.is_connected is True
        assert manager.alert_banner_active is False
        assert manager.feed_source == FeedSource.ALPACA_RELAY

        # Inject upstream_disconnected event from relay server
        await mock_relay_server.broadcast_lifecycle("upstream_disconnected")

        # Must turn True immediately (within 100ms)
        for _ in range(10):
            if manager.alert_banner_active:
                break
            await asyncio.sleep(0.02)

        status = manager.get_connection_status()
        assert manager.alert_banner_active is True, "alert_banner_active failed to turn True on upstream_disconnected"
        assert status.alert_banner_active is True
        assert manager.is_connected is False
        assert status.is_connected is False
        assert manager.feed_source == FeedSource.SYNTHETIC_FALLBACK
        assert status.feed_source == "synthetic_fallback"
        assert len(disconnect_notifications) >= 1
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_close_code_1013_slow_client_eviction_chaos(mock_relay_server, temp_sqlite_db):
    """Chaos-02: Server evicts client with close code 1013 (too slow).
    Adversarial check: Does DataFeedManager detect this eviction and raise alert_banner_active immediately?
    """
    config = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
    )
    manager = DataFeedManager(config=config)

    await manager.start()
    try:
        for _ in range(30):
            if manager.is_connected and len(mock_relay_server.clients) > 0:
                break
            await asyncio.sleep(0.05)

        assert manager.is_connected is True
        assert manager.alert_banner_active is False
        assert len(mock_relay_server.clients) >= 1

        # Locate the connected client websocket on the mock server
        server_ws = next(iter(mock_relay_server.clients))

        # Evict with code 1013
        await mock_relay_server.simulate_slow_client_eviction(server_ws)

        # Allow brief time for socket disconnect processing
        await asyncio.sleep(0.2)

        # Observe DataFeedManager state
        status = manager.get_connection_status()
        print(f"\n[Post-1013 Eviction] is_connected={status.is_connected}, alert_banner_active={status.alert_banner_active}, feed_source={status.feed_source}")

        # Invariant check: While socket is evicted, alert banner should be active
        assert status.alert_banner_active is True, (
            "VULNERABILITY DETECTED: DataFeedManager did not set alert_banner_active=True after close code 1013 eviction!"
        )
        assert status.is_connected is False, (
            "VULNERABILITY DETECTED: DataFeedManager still claims is_connected=True after close code 1013 eviction!"
        )
        assert status.feed_source == "synthetic_fallback"
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_sudden_socket_drop_chaos(mock_relay_server, temp_sqlite_db):
    """Chaos-03: Sudden abrupt socket closure from server side without upstream notice.
    Adversarial check: Does DataFeedManager immediately transition to fallback or hang in stale live state?
    """
    config = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
    )
    manager = DataFeedManager(config=config)

    await manager.start()
    try:
        for _ in range(30):
            if manager.is_connected and len(mock_relay_server.clients) > 0:
                break
            await asyncio.sleep(0.05)

        assert manager.is_connected is True
        assert manager.alert_banner_active is False

        # Abruptly close the socket from server side with 1000/1006
        server_ws = next(iter(mock_relay_server.clients))
        await server_ws.close(code=1000)

        # Allow socket drop to propagate
        await asyncio.sleep(0.2)

        status = manager.get_connection_status()
        print(f"\n[Post-Socket Drop] is_connected={status.is_connected}, alert_banner_active={status.alert_banner_active}, feed_source={status.feed_source}")

        assert status.alert_banner_active is True, (
            "VULNERABILITY DETECTED: DataFeedManager did not set alert_banner_active=True upon abrupt socket drop!"
        )
        assert status.is_connected is False, (
            "VULNERABILITY DETECTED: DataFeedManager still claims is_connected=True upon abrupt socket drop!"
        )
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_two_tier_fallback_during_disconnect(temp_sqlite_db):
    """Chaos-04: Verify two-tier fallback during active disconnect:
    - Tier 1: SQLite WAL cached bars returned seamlessly
    - Tier 2: SDE synthetic generator invoked for missing symbols
    - Zero crashes on price resolution
    """
    config = FeedManagerConfig(
        relay_base_url="http://127.0.0.1:9999",  # Offline endpoint
        relay_ws_url="ws://127.0.0.1:9999",
        db_path=temp_sqlite_db,
    )
    manager = DataFeedManager(config=config)

    # Populate Tier 1 data in SQLite WAL for SPY only
    now = datetime.now(timezone.utc)
    cached_spy_bars = [
        Bar(symbol="SPY", timestamp=now - timedelta(days=i), open=500.0, high=505.0, low=495.0, close=502.0, volume=100000)
        for i in range(5)
    ]
    manager.bar_repo.save_bars(cached_spy_bars, timeframe="1Day")

    # Start manager in disconnected state
    await manager.start()
    try:
        assert manager.is_connected is False
        assert manager.alert_banner_active is True
        assert manager.feed_source == FeedSource.SYNTHETIC_FALLBACK

        # Query SPY (Tier 1 cached) and QQQ (Tier 2 synthetic miss)
        bars_map = await manager.get_historical_bars(symbols=["SPY", "QQQ"], timeframe="1Day")

        assert "SPY" in bars_map
        assert len(bars_map["SPY"]) >= 5
        assert bars_map["SPY"][0].close == 502.0

        assert "QQQ" in bars_map
        assert len(bars_map["QQQ"]) > 0
        assert all(b.close > 0.0 for b in bars_map["QQQ"])
        assert all(b.symbol == "QQQ" for b in bars_map["QQQ"])

        # Check latest prices for universe and unknown symbols
        for sym in ["SPY", "QQQ", "XLK", "XLE", "XLV", "XLI", "XLU", "TLT", "SHV", "GLD", "UNKNOWN_CORP"]:
            price = manager.get_latest_price(sym)
            assert isinstance(price, float)
            assert price > 0.0, f"Expected positive price for {sym}, got {price}"

        # In fallback mode, is_safe_to_rebalance must return True to permit paper rebalance on synthetic feed
        assert manager.is_safe_to_rebalance() is True
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_reconnect_and_recovery_injection(mock_relay_server, temp_sqlite_db):
    """Chaos-05: Recovery injection tests auto-reconnection and alert banner reset."""
    config = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
        recovery_check_interval=0.1,
    )
    manager = DataFeedManager(config=config)
    recovered_events: List[tuple] = []
    manager.on_recover(lambda downtime, ts: recovered_events.append((downtime, ts)))

    await manager.start()
    try:
        # Wait for live connect
        for _ in range(30):
            if manager.is_connected:
                break
            await asyncio.sleep(0.05)

        assert manager.is_connected is True
        assert manager.alert_banner_active is False

        # 1. Trigger disconnect
        await mock_relay_server.broadcast_lifecycle("upstream_disconnected")
        for _ in range(20):
            if manager.alert_banner_active:
                break
            await asyncio.sleep(0.05)

        assert manager.alert_banner_active is True
        assert manager.is_connected is False

        # 2. Inject recovery: mock server sets upstream_connected
        await mock_relay_server.broadcast_lifecycle("upstream_connected")

        # 3. Wait for manager to recover
        for _ in range(40):
            if not manager.alert_banner_active and manager.is_connected:
                break
            await asyncio.sleep(0.05)

        status = manager.get_connection_status()
        assert manager.is_connected is True, "Failed to recover live connection"
        assert manager.alert_banner_active is False, "alert_banner_active did not reset to False upon recovery"
        assert status.alert_banner_active is False
        assert manager.feed_source == FeedSource.ALPACA_RELAY
        assert len(recovered_events) >= 1
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_synthetic_ticker_continuous_emission(temp_sqlite_db):
    """Chaos-06: Verify synthetic ticker actively emits dynamic bars to on_bar callbacks during fallback."""
    config = FeedManagerConfig(
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
        db_path=temp_sqlite_db,
        synthetic_tick_interval=0.05,  # fast ticks for testing
    )
    manager = DataFeedManager(config=config)
    received_bars: List[Bar] = []
    manager.on_bar(lambda b: received_bars.append(b))

    await manager.start()
    try:
        assert manager.alert_banner_active is True
        # Allow ticker loop to emit several bars
        for _ in range(20):
            if len(received_bars) >= 10:
                break
            await asyncio.sleep(0.05)

        assert len(received_bars) >= 10, f"Expected >= 10 synthetic bars, received {len(received_bars)}"
        assert all(b.close > 0.0 for b in received_bars)
        assert all(b.volume > 0 for b in received_bars)
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_synthetic_ticker_after_disconnect_transition(mock_relay_server, temp_sqlite_db):
    """Chaos-06b: Verify synthetic ticker runs when transitioning from live to fallback via upstream_disconnected."""
    config = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
        synthetic_tick_interval=0.05,
    )
    manager = DataFeedManager(config=config)
    received_bars: List[Bar] = []
    manager.on_bar(lambda b: received_bars.append(b))

    await manager.start()
    try:
        for _ in range(20):
            if manager.is_connected:
                break
            await asyncio.sleep(0.05)

        assert manager.is_connected is True
        received_bars.clear()

        # Trigger disconnect transition
        await mock_relay_server.broadcast_lifecycle("upstream_disconnected")
        for _ in range(10):
            if manager.alert_banner_active:
                break
            await asyncio.sleep(0.05)

        assert manager.alert_banner_active is True

        # Now wait for synthetic ticker to emit bars
        for _ in range(20):
            if len(received_bars) >= 10:
                break
            await asyncio.sleep(0.05)

        assert len(received_bars) >= 10, f"Expected >= 10 synthetic bars post-disconnect, received {len(received_bars)}"
    finally:
        await manager.stop()



@pytest.mark.asyncio
async def test_rest_failure_triggers_fallback_and_tier1_bars(temp_sqlite_db):
    """Chaos-07: REST failure during get_historical_bars triggers fallback transition and returns cached data."""
    config = FeedManagerConfig(
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
        db_path=temp_sqlite_db,
    )
    manager = DataFeedManager(config=config)

    # Seed WAL with SPY bars
    now = datetime.now(timezone.utc)
    cached_bars = [
        Bar(symbol="SPY", timestamp=now - timedelta(days=i), open=500.0, high=505.0, low=495.0, close=501.5, volume=50000)
        for i in range(3)
    ]
    manager.bar_repo.save_bars(cached_bars, timeframe="1Day")

    # Manually simulate in live mode with offline REST endpoint
    manager._feed_source = FeedSource.ALPACA_RELAY
    manager._is_connected = True
    manager._alert_banner_active = False

    # Calling get_historical_bars will fail REST call, transition to fallback, and return cached bars
    bars_map = await manager.get_historical_bars(["SPY"], timeframe="1Day")

    assert manager.alert_banner_active is True
    assert manager.feed_source == FeedSource.SYNTHETIC_FALLBACK
    assert "SPY" in bars_map
    assert len(bars_map["SPY"]) >= 3
    assert bars_map["SPY"][0].close == 501.5


@pytest.mark.asyncio
async def test_rapid_disconnect_reconnect_flapping_stress(mock_relay_server, temp_sqlite_db):
    """Chaos-08: Rapid disconnect/reconnect lifecycle flapping stress test."""
    config = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
        recovery_check_interval=0.1,
    )
    manager = DataFeedManager(config=config)
    await manager.start()
    try:
        for _ in range(20):
            if manager.is_connected:
                break
            await asyncio.sleep(0.05)

        # Flap lifecycle 5 times in rapid succession
        for _ in range(5):
            await mock_relay_server.broadcast_lifecycle("upstream_disconnected")
            await asyncio.sleep(0.02)
            await mock_relay_server.broadcast_lifecycle("upstream_connected")
            await asyncio.sleep(0.02)

        # Settle
        await asyncio.sleep(0.3)
        status = manager.get_connection_status()
        assert isinstance(status, ConnectionStatus)
        assert status.feed_source in ("alpaca_relay", "synthetic_fallback")
        assert isinstance(status.alert_banner_active, bool)
    finally:
        await manager.stop()
