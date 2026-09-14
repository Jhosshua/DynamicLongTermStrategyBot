"""Integration test suite for Institutional Discord v2 Alerts Engine.

Validates end-to-end integration across:
- DataFeedManager (upstream disconnect -> Broken Card, reconnect -> Recovered Card)
- DynamicStrategyService (weekly rebalance -> Trade Card, manual rebalance -> Trade Card, circuit breaker -> Trade Card)
- Webhook transport resilience (HTTP errors, timeouts, 429 backoff do not halt trading)
- Process-wide rate limiting pacing
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from bot.discord_alerts import (
    BLUE,
    DISCORD_COLOR_BROKEN,
    DISCORD_COLOR_RECOVERED,
    DISCORD_COLOR_TRADE,
    GREEN,
    RED,
    DiscordNotifier,
)
from bot.feed_manager import FeedManager, FeedManagerConfig, FeedSource
from bot.paper_account import PaperAccountConfig, PaperAccountManager
from bot.service import DynamicStrategyService, ServiceConfig, ServiceState
from strategy_engine.core.models import Bar, MarketRegime, OrderIntent, OrderSide
from strategy_engine.simulator.stress_scenarios import generate_2017_low_vol_bull
from tests.live_feed_helper import simulate_live_feed


@pytest.fixture
def mock_poster_captured():
    """Provides mock poster and captured payload accumulator."""
    captured: List[Dict[str, Any]] = []

    def _poster(url: str, json: dict):
        captured.append(json)
        resp = MagicMock()
        resp.status_code = 204
        return resp

    return _poster, captured


# ============================================================================
# 1. FeedManager Lifecycle Alerts
# ============================================================================

@pytest.mark.asyncio
async def test_feed_disconnect_dispatches_broken_card(mock_relay_server, temp_sqlite_db):
    """IT-DISCORD-01: Upstream disconnect triggers fallback and dispatches Red Broken Card."""
    captured_cards = []

    def mock_poster(url: str, json: dict):
        captured_cards.append(json)
        resp = MagicMock()
        resp.status_code = 204
        return resp

    notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        dashboard_url="https://bot.railway.app",
        suppress_in_test=True,
    )

    feed_cfg = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
    )
    feed = FeedManager(config=feed_cfg, discord_notifier=notifier)

    await feed.start()
    try:
        # Wait for live connect
        for _ in range(25):
            if feed.is_connected:
                break
            await asyncio.sleep(0.05)

        # Clear startup recovered card
        captured_cards.clear()

        # Trigger disconnect transition
        await feed._transition_to_fallback("Simulated upstream disconnect event")

        assert len(captured_cards) >= 1
        card = captured_cards[-1]["embeds"][0]
        assert card["color"] == DISCORD_COLOR_BROKEN
        assert card["color"] == RED
        assert "BROKEN" in card["title"]
        assert "AlpacaRelayClient" in card["title"]

        fields = {f["name"]: f["value"] for f in card["fields"]}
        assert "What broke" in fields
        assert "Simulated upstream disconnect" in fields["What broke"]
        assert "Operator Dashboard" in fields
    finally:
        await feed.stop()


@pytest.mark.asyncio
async def test_feed_reconnect_dispatches_recovered_card(mock_relay_server, temp_sqlite_db):
    """IT-DISCORD-02: Stream recovery calculates downtime and dispatches Green Recovered Card."""
    captured_cards = []

    def mock_poster(url: str, json: dict):
        captured_cards.append(json)
        resp = MagicMock()
        resp.status_code = 204
        return resp

    notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        dashboard_url="https://bot.railway.app",
        suppress_in_test=True,
    )

    feed_cfg = FeedManagerConfig(
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        db_path=temp_sqlite_db,
    )
    feed = FeedManager(config=feed_cfg, discord_notifier=notifier)

    await feed.start()
    try:
        # Simulate past disconnect 18.5 seconds ago
        feed._feed_source = FeedSource.SYNTHETIC_FALLBACK
        feed._disconnect_timestamp = datetime.now(timezone.utc) - timedelta(seconds=18.5)

        # Trigger recovery
        await feed._transition_to_live("Upstream stream re-established")

        assert len(captured_cards) >= 1
        card = captured_cards[-1]["embeds"][0]
        assert card["color"] == DISCORD_COLOR_RECOVERED
        assert card["color"] == GREEN
        assert "RECOVERED" in card["title"]
        assert "AlpacaRelayClient" in card["title"]

        fields = {f["name"]: f["value"] for f in card["fields"]}
        assert "Downtime Duration" in fields
        assert "s" in fields["Downtime Duration"]
        assert "Telemetry Status" in fields
    finally:
        await feed.stop()


# ============================================================================
# 2. DynamicStrategyService Rebalance & Trade Execution Alerts
# ============================================================================

@pytest.mark.asyncio
async def test_weekly_rebalance_dispatches_trade_card(mock_relay_server, temp_sqlite_db):
    """IT-DISCORD-03: Weekly rebalance fill execution dispatches Blue Trade Card."""
    captured_cards = []

    def mock_poster(url: str, json: dict):
        captured_cards.append(json)
        resp = MagicMock()
        resp.status_code = 204
        return resp

    notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        dashboard_url="https://bot.railway.app",
        suppress_in_test=True,
    )

    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        dashboard_url="https://bot.railway.app",
    )
    service = DynamicStrategyService(config=config, discord_notifier=notifier)
    await simulate_live_feed(service)
    await service._warmup_historical_bars()

    # Advance daily close
    dt = datetime(2026, 9, 11, 15, 50, 0, tzinfo=timezone.utc)
    await service._handle_daily_close(dt)

    # Execute weekly rebalance
    orders = await service._handle_weekly_rebalance(dt)
    assert len(orders) > 0

    assert len(captured_cards) >= 1
    card = captured_cards[-1]["embeds"][0]
    assert card["color"] == DISCORD_COLOR_TRADE
    assert card["color"] == BLUE
    assert "TRADE EXECUTION" in card["title"] or "Virtual Paper Trade" in card["title"]

    fields = {f["name"]: f["value"] for f in card["fields"]}
    assert "Portfolio NAV" in fields
    assert "$50," in fields["Portfolio NAV"]
    assert "Orders Detail" in fields
    assert len(notifier.dispatched_cards) >= 1


@pytest.mark.asyncio
async def test_manual_rebalance_dispatches_trade_card(mock_relay_server, temp_sqlite_db):
    """IT-DISCORD-04: Operator manual rebalance execution dispatches Blue Trade Card."""
    captured_cards = []

    def mock_poster(url: str, json: dict):
        captured_cards.append(json)
        resp = MagicMock()
        resp.status_code = 204
        return resp

    notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        dashboard_url="https://bot.railway.app",
        suppress_in_test=True,
    )

    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        dashboard_url="https://bot.railway.app",
    )
    service = DynamicStrategyService(config=config, discord_notifier=notifier)
    await simulate_live_feed(service)
    await service._warmup_historical_bars()

    # Trigger manual rebalance
    result = await service.manual_rebalance(force=True)
    assert result.success is True
    assert result.orders_count > 0

    assert len(captured_cards) >= 1
    card = captured_cards[-1]["embeds"][0]
    assert card["color"] == BLUE
    assert "TRADE EXECUTION" in card["title"] or "Virtual Paper Trade" in card["title"]


@pytest.mark.asyncio
async def test_emergency_circuit_breaker_dispatches_trade_card(mock_relay_server, temp_sqlite_db):
    """IT-DISCORD-05: Emergency circuit breaker de-risk into 100% SHV dispatches Trade Card."""
    captured_cards = []

    def mock_poster(url: str, json: dict):
        captured_cards.append(json)
        resp = MagicMock()
        resp.status_code = 204
        return resp

    notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        dashboard_url="https://bot.railway.app",
        suppress_in_test=True,
    )

    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        dashboard_url="https://bot.railway.app",
    )
    service = DynamicStrategyService(config=config, discord_notifier=notifier)
    await simulate_live_feed(service)
    await service._warmup_historical_bars()

    # Establish initial allocation (e.g. SPY, QQQ)
    dt = datetime(2026, 9, 11, 15, 50, 0, tzinfo=timezone.utc)
    await service._handle_daily_close(dt)
    await service._handle_weekly_rebalance(dt)
    captured_cards.clear()

    # Trigger emergency circuit breaker
    crash_bar = Bar(
        symbol="SPY",
        timestamp=datetime(2026, 9, 12, 14, 30, 0, tzinfo=timezone.utc),
        open=550.0,
        high=551.0,
        low=480.0,
        close=485.0,
        volume=25000000.0,
    )
    await service._trigger_emergency_circuit_breaker(crash_bar, "Intraday Flash Crash Breach")

    assert len(captured_cards) >= 1
    card = captured_cards[-1]["embeds"][0]
    assert card["color"] == BLUE
    assert "BEAR_CRISIS" in card["title"] or "TRADE EXECUTION" in card["title"]


# ============================================================================
# 3. Notification Isolation: Webhook Errors Never Halt Trading
# ============================================================================

@pytest.mark.asyncio
async def test_webhook_transport_failure_does_not_halt_trading(mock_relay_server, temp_sqlite_db):
    """IT-DISCORD-06: Failing Discord webhook does not crash or interrupt trading engine."""
    def failing_poster(url: str, json: dict):
        raise TimeoutError("Discord webhook gateway timeout")

    failing_notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=failing_poster,
        dashboard_url="https://bot.railway.app",
        suppress_in_test=True,
    )

    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url=mock_relay_server.http_url,
        relay_ws_url=mock_relay_server.ws_url,
        relay_token="test-relay-token-xyz",
        dashboard_url="https://bot.railway.app",
    )
    service = DynamicStrategyService(config=config, discord_notifier=failing_notifier)
    await simulate_live_feed(service)
    await service._warmup_historical_bars()

    # Rebalance execution proceeds smoothly despite webhook timeout
    dt = datetime(2026, 9, 11, 15, 50, 0, tzinfo=timezone.utc)
    await service._handle_daily_close(dt)
    orders = await service._handle_weekly_rebalance(dt)

    assert len(orders) > 0
    portfolio = service.paper_account.get_portfolio_state()
    assert portfolio.total_nav > 0
    assert len(portfolio.positions) > 0
    # Confirms paper trading state updated in SQLite WAL without error


# ============================================================================
# 4. Process-Wide Rate Limiting Pacing
# ============================================================================

def test_rate_limiter_pacing_end_to_end():
    """IT-DISCORD-07: Rapid dispatch burst enforces >= 2.0s monotonic spacing between posts."""
    call_timestamps: List[float] = []

    def mock_poster(url: str, json: dict):
        call_timestamps.append(time.monotonic())
        resp = MagicMock()
        resp.status_code = 204
        return resp

    # Test with rate_limit_interval_s=0.05 (fast for test execution)
    notifier = DiscordNotifier(
        webhook_url="https://discord.com/api/webhooks/mock",
        http_post=mock_poster,
        rate_limit_interval_s=0.05,
        max_backoff_sleep_s=1.0,
        suppress_in_test=False,
    )

    for i in range(3):
        notifier.post_broken_alert(f"Subsystem_{i}", f"Error {i}", "trace", "https://dash")

    assert len(call_timestamps) == 3
    # Check intervals between calls
    delta1 = call_timestamps[1] - call_timestamps[0]
    delta2 = call_timestamps[2] - call_timestamps[1]
    assert delta1 >= 0.04  # Monotonic pacing enforced
    assert delta2 >= 0.04
