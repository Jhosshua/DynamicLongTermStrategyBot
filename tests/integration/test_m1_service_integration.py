"""
tests.integration.test_m1_service_integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

End-to-end integration test suite for Milestone 1:
- DynamicStrategyService orchestrating DecisionDaemon, PortfolioRebalancer,
  PaperAccountManager, and FeedManager
- $50,000.00 pristine paper cash ledger execution
- Cadences: DAILY_CLOSE, WEEKLY_REBALANCE, MONTHLY_MOMENTUM
- Operator controls: pause, resume, manual_rebalance, get_health, reset_to_pristine
- Disconnect resilience and synthetic fallback simulation
- Intraday emergency circuit breaker de-risk
- Persistence across service restarts
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from bot.feed_manager import FeedManager, FeedManagerConfig, FeedSource
from bot.paper_account import PaperAccountConfig, PaperAccountManager
from bot.service import (
    DynamicStrategyService,
    ManualRebalanceResult,
    ServiceConfig,
    ServiceState,
    ServiceStatus,
)
from strategy_engine.core.models import Bar, MarketRegime, OrderSide
from strategy_engine.simulator.stress_scenarios import generate_2017_low_vol_bull


@pytest.fixture
def mock_bars_2017():
    """Generates baseline daily bars for all 10 universe symbols."""
    dataset = generate_2017_low_vol_bull(seed=42)
    return dataset.bars


@pytest.mark.asyncio
async def test_service_boot_and_50k_account_bootstrap(mock_relay_server, temp_sqlite_db):
    """IT-01: Service boot initializes $50,000.00 pristine ledger and enters RUNNING state."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
    )
    service = DynamicStrategyService(config=config)
    await service._warmup_historical_bars()

    status = service.get_service_status()
    assert status.service_name == "DynamicLongTermStrategyBot"
    assert status.cash == 50000.00
    assert status.equity == 0.00
    assert status.total_nav == 50000.00
    assert status.active_positions_count == 0
    await service.shutdown()


@pytest.mark.asyncio
async def test_historical_warmup_and_price_caching(temp_sqlite_db):
    """IT-02: Warmup caches bars in memory and SQLite WAL."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",  # non-existent, falls back to SDE
        relay_ws_url="ws://127.0.0.1:9999",
    )
    service = DynamicStrategyService(config=config)
    await service._warmup_historical_bars()

    # Verify cached daily bars
    assert len(service._cached_daily_bars) >= 10
    assert "SPY" in service._cached_daily_bars
    assert len(service._cached_daily_bars["SPY"]) > 0

    # Verify SQLite WAL storage
    cached_bars = service.storage.bars.get_bars("SPY", timeframe="1Day")
    assert len(cached_bars) > 0
    await service.shutdown()


@pytest.mark.asyncio
async def test_daily_close_cadence_evaluation(temp_sqlite_db):
    """IT-03: Daily close evaluates signals, target allocations, and snapshots paper ledger."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
    )
    service = DynamicStrategyService(config=config)
    await service._warmup_historical_bars()

    dt = datetime(2026, 9, 11, 15, 50, 0, tzinfo=timezone.utc)
    await service._handle_daily_close(dt)

    assert service._latest_signal is not None
    assert service._latest_allocation is not None
    total_weights = sum(service._latest_allocation.weights.values())
    assert total_weights == pytest.approx(1.0, abs=1e-4)

    # Verify SQLite storage
    saved_allocation = service.storage.allocations.get_latest()
    assert saved_allocation is not None
    await service.shutdown()


@pytest.mark.asyncio
async def test_weekly_rebalance_execution_on_paper_ledger(temp_sqlite_db):
    """IT-04: Weekly rebalance generates SELLs before BUYs and executes against $50k paper ledger."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
    )
    service = DynamicStrategyService(config=config)
    await service._warmup_historical_bars()

    dt = datetime(2026, 9, 11, 15, 50, 0, tzinfo=timezone.utc)
    await service._handle_daily_close(dt)

    orders = await service._handle_weekly_rebalance(dt)
    assert len(orders) > 0

    # Check that SELL orders come before BUY orders
    seen_buy = False
    for o in orders:
        if o.action == "BUY":
            seen_buy = True
        elif o.action == "SELL":
            assert not seen_buy, "All SELLs must precede BUYs"

    # Check portfolio state on paper account
    portfolio = service.paper_account.get_portfolio_state()
    assert portfolio.total_nav == pytest.approx(50000.00, abs=10.0)  # Minor difference from fractional rounding/slippage
    assert len(portfolio.positions) > 0
    assert portfolio.cash >= 0.00

    # Trade history exists
    trades = service.paper_account.get_trade_history()
    assert len(trades) > 0
    await service.shutdown()


@pytest.mark.asyncio
async def test_drift_band_filtering(temp_sqlite_db):
    """IT-05: Rebalancer suppresses orders when drift is within +/-2.5% tolerance."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
    )
    service = DynamicStrategyService(config=config)
    await service._warmup_historical_bars()

    dt = datetime(2026, 9, 11, 15, 50, 0, tzinfo=timezone.utc)
    await service._handle_daily_close(dt)
    # First rebalance opens initial target positions
    await service._handle_weekly_rebalance(dt)

    # Immediately re-running weekly rebalance with identical prices should generate 0 orders
    second_orders = await service._handle_weekly_rebalance(dt)
    assert len(second_orders) == 0
    await service.shutdown()


@pytest.mark.asyncio
async def test_operator_pause_and_resume(temp_sqlite_db):
    """IT-06 & IT-07: Operator pause freezes rebalancing, resume unfreezes."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
    )
    service = DynamicStrategyService(config=config)
    await service._warmup_historical_bars()

    # Pause service
    status_p = await service.pause()
    assert status_p.is_paused is True
    assert status_p.state == ServiceState.PAUSED

    # Attempt weekly rebalance while paused
    dt = datetime(2026, 9, 11, 15, 50, 0, tzinfo=timezone.utc)
    orders = await service._handle_weekly_rebalance(dt)
    assert len(orders) == 0

    # Resume service
    status_r = await service.resume()
    assert status_r.is_paused is False
    assert status_r.state == ServiceState.RUNNING

    # Weekly rebalance now executes
    orders_after = await service._handle_weekly_rebalance(dt)
    assert len(orders_after) > 0
    await service.shutdown()


@pytest.mark.asyncio
async def test_manual_rebalance_execution(temp_sqlite_db):
    """IT-08: Operator-initiated manual rebalance evaluates and executes."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
    )
    service = DynamicStrategyService(config=config)
    await service._warmup_historical_bars()

    result = await service.manual_rebalance()
    assert result.success is True
    assert result.status in ("EXECUTED", "SKIPPED_WITHIN_BAND")
    assert result.executed_trades_count >= 0
    await service.shutdown()


@pytest.mark.asyncio
async def test_manual_rebalance_rejection_when_paused(temp_sqlite_db):
    """IT-09: Manual rebalance rejected when paused unless force=True."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
    )
    service = DynamicStrategyService(config=config)
    await service._warmup_historical_bars()
    await service.pause()

    res_rejected = await service.manual_rebalance(force=False)
    assert res_rejected.success is False
    assert res_rejected.status == "REJECTED_PAUSED"

    res_override = await service.manual_rebalance(force=True)
    assert res_override.success is True
    await service.shutdown()


@pytest.mark.asyncio
async def test_feed_disconnect_and_fallback_simulation_handshake(mock_relay_server, temp_sqlite_db):
    """IT-10 & IT-11: Feed disconnect toggles alert banner; service ticks on fallback without crashing."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
    )
    service = DynamicStrategyService(config=config)
    await service.feed_manager.start()
    await asyncio.sleep(0.1)

    try:
        assert service.feed_manager.is_connected is True
        assert service.feed_manager.alert_banner_active is False

        # Disconnect upstream
        await mock_relay_server.broadcast_lifecycle("upstream_disconnected")
        for _ in range(20):
            if service.feed_manager.alert_banner_active:
                break
            await asyncio.sleep(0.05)

        assert service.feed_manager.alert_banner_active is True
        assert service.feed_manager.feed_source == FeedSource.SYNTHETIC_FALLBACK

        # Service evaluation runs smoothly on fallback data without exceptions
        status = service.get_service_status()
        assert status.alert_banner_active is True
        assert status.feed_source == "synthetic_fallback"

        # Reconnect upstream
        await mock_relay_server.broadcast_lifecycle("upstream_connected")
        for _ in range(30):
            if not service.feed_manager.alert_banner_active and service.feed_manager.is_connected:
                break
            await asyncio.sleep(0.1)

        assert service.feed_manager.alert_banner_active is False
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_intraday_emergency_circuit_breaker(temp_sqlite_db):
    """IT-12: Intraday flash drop breaches circuit breaker and liquidates into 100% SHV."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
    )
    service = DynamicStrategyService(config=config)
    await service._warmup_historical_bars()

    # Open positions via rebalance
    dt = datetime(2026, 9, 11, 15, 50, 0, tzinfo=timezone.utc)
    await service._handle_daily_close(dt)
    await service._handle_weekly_rebalance(dt)

    assert len(service.paper_account.get_portfolio_state().positions) > 0

    # Stream a catastrophic flash crash bar on SPY (5% drop intraday)
    crash_bar = Bar(
        symbol="SPY",
        timestamp=datetime.now(timezone.utc),
        open=500.0,
        high=500.0,
        low=475.0,
        close=475.0,  # -5.0% flash drop
        volume=5000000,
    )
    service._on_bar_received(crash_bar)

    # Allow async task to complete emergency de-risk
    for _ in range(20):
        if service._circuit_breaker_triggered_today:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.1)

    assert service._circuit_breaker_triggered_today is True
    post_portfolio = service.paper_account.get_portfolio_state()
    # Holdings should be rotated to SHV or cash
    for pos in post_portfolio.positions:
        if pos.symbol != "SHV":
            assert pos.qty == 0.0 or pos.weight < 0.01

    await service.shutdown()


@pytest.mark.asyncio
async def test_health_endpoint_contract(temp_sqlite_db):
    """IT-13: /health conforms strictly to PROJECT.md interface contract."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
    )
    service = DynamicStrategyService(config=config)
    await service._warmup_historical_bars()

    health = service.get_health()
    assert health["status"] == "ok"
    assert health["service"] == "DynamicLongTermStrategyBot"
    assert "relay" in health
    assert "portfolio" in health
    assert health["portfolio"]["nav"] > 0
    assert "regime" in health
    assert "uptime_s" in health
    await service.shutdown()


@pytest.mark.asyncio
async def test_reset_to_pristine_for_monday(temp_sqlite_db):
    """IT-14: reset_to_pristine() clears all test trades and restores clean $50,000.00 cash."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
    )
    service = DynamicStrategyService(config=config)
    await service._warmup_historical_bars()

    # Execute trades
    dt = datetime(2026, 9, 11, 15, 50, 0, tzinfo=timezone.utc)
    await service._handle_daily_close(dt)
    await service._handle_weekly_rebalance(dt)

    assert len(service.paper_account.get_portfolio_state().positions) > 0
    assert len(service.paper_account.get_trade_history()) > 0

    # Execute pristine reset for Monday
    pristine_state = service.reset_to_pristine()
    assert pristine_state.cash == 50000.00
    assert pristine_state.equity == 0.00
    assert pristine_state.total_nav == 50000.00
    assert len(pristine_state.positions) == 0
    assert len(service.paper_account.get_trade_history()) == 0
    await service.shutdown()


@pytest.mark.asyncio
async def test_service_restart_persistence(temp_sqlite_db):
    """IT-15: Ledger state and positions reload accurately on service restart."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
    )
    s1 = DynamicStrategyService(config=config)
    await s1._warmup_historical_bars()
    dt = datetime(2026, 9, 11, 15, 50, 0, tzinfo=timezone.utc)
    await s1._handle_daily_close(dt)
    await s1._handle_weekly_rebalance(dt)

    p1 = s1.paper_account.get_portfolio_state()
    nav1 = p1.total_nav
    cash1 = p1.cash
    await s1.shutdown()

    # Reboot new service on same SQLite database
    s2 = DynamicStrategyService(config=config)
    p2 = s2.paper_account.get_portfolio_state()

    assert p2.total_nav == pytest.approx(nav1, abs=1e-2)
    assert p2.cash == pytest.approx(cash1, abs=1e-2)
    assert len(p2.positions) == len(p1.positions)
    await s2.shutdown()
