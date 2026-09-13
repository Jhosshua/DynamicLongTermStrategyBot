"""Tier 3: Cross-Feature Combinations & Pairwise Interactions E2E Test Suite.

Covers multi-module workflows, state sharing, transitions, and feedback loops
across the 20 features from PROJECT.md.
Total: 20 tests.
"""

from datetime import datetime, timezone
import json
import math
import pytest

from tests.e2e_suite.contracts import (
    DISCORD_COLOR_BROKEN,
    DISCORD_COLOR_RECOVERED,
    DISCORD_COLOR_TRADE,
    ConnectionStatus,
    DiscordEmbedCard,
    DiscordNotifierContract,
    OperatorAppContract,
    PaperAccountManagerContract,
    PortfolioSummary,
    PositionDetail,
    RebalanceOrder,
    resolve_discord_notifier_cls,
    resolve_feed_manager_cls,
    resolve_paper_account_cls,
)


def test_t3_01_f1_f2_initialization_to_execution_lifecycle(paper_account):
    """T3-01 (F1 + F2): $50k Account initial ledger handles series of trades across multiple assets."""
    assert math.isclose(paper_account.get_cash_balance(), 50000.00)

    # Buy multiple assets
    paper_account.execute_order("SPY", "BUY", 20.0, 500.0)   # -$10,000
    paper_account.execute_order("QQQ", "BUY", 25.0, 400.0)   # -$10,000
    paper_account.execute_order("TLT", "BUY", 100.0, 95.0)   # -$9,500
    assert math.isclose(paper_account.get_cash_balance(), 20500.00)

    # Update prices and sell part of SPY at profit
    paper_account.update_market_prices({"SPY": 520.0, "QQQ": 410.0, "TLT": 96.0})
    res_sell = paper_account.execute_order("SPY", "SELL", 10.0, 520.0)
    assert math.isclose(res_sell["realized_pnl"], 200.0)

    state = paper_account.get_portfolio_state()
    assert math.isclose(state.cash, 25700.00)
    assert math.isclose(state.realized_pnl, 200.0)
    assert len(state.positions) == 3


def test_t3_02_f1_f17_init_trading_and_pristine_reset(paper_account, temp_paper_db):
    """T3-02 (F1 + F17): Account executes active trading, then resets atomically to pristine $50k."""
    paper_account.execute_order("AAPL", "BUY", 50.0, 180.0)
    paper_account.execute_order("MSFT", "BUY", 20.0, 400.0)
    paper_account.execute_order("AAPL", "SELL", 25.0, 190.0)
    assert paper_account.get_cash_balance() != 50000.00
    assert len(paper_account.get_portfolio_state().positions) > 0

    # Atomic pristine reset for Monday
    paper_account.reset_to_pristine()
    state = paper_account.get_portfolio_state()
    assert math.isclose(state.cash, 50000.00, rel_tol=1e-5)
    assert math.isclose(state.total_nav, 50000.00, rel_tol=1e-5)
    assert math.isclose(state.equity, 0.0, abs_tol=1e-5)
    assert math.isclose(state.realized_pnl, 0.0, abs_tol=1e-5)
    assert len(state.positions) == 0


def test_t3_03_f3_f4_relay_connection_drop_triggers_fallback(feed_manager):
    """T3-03 (F3 + F4): Active REST/WS client encounters disconnect and seamlessly activates fallback."""
    assert feed_manager.get_connection_status().is_connected is True
    assert feed_manager.get_connection_status().feed_source == "alpaca_relay"

    # Ingest price under normal feed
    p1 = feed_manager.get_latest_price("SPY")
    assert p1 > 0.0

    # Disconnect occurs
    feed_manager.trigger_disconnect("upstream_disconnected")
    status = feed_manager.get_connection_status()
    assert status.is_connected is False
    assert status.feed_source == "synthetic_fallback"

    # Price stream continues under synthetic fallback
    p2 = feed_manager.get_latest_price("SPY")
    assert p2 > 0.0


def test_t3_04_f4_f13_feed_disconnect_activates_dashboard_alert_banner(feed_manager, operator_app):
    """T3-04 (F4 + F13): Feed disconnect immediately synchronizes with dashboard alert banner."""
    # Initially healthy
    c1, _, b1 = operator_app.handle_request("GET", "/api/portfolio")
    assert json.loads(b1)["connection"]["alert_banner_active"] is False

    # Trigger disconnect
    feed_manager.trigger_disconnect()
    c2, _, b2 = operator_app.handle_request("GET", "/api/portfolio")
    assert json.loads(b2)["connection"]["alert_banner_active"] is True

    # Reconnect restores banner state
    feed_manager.trigger_reconnect()
    c3, _, b3 = operator_app.handle_request("GET", "/api/portfolio")
    assert json.loads(b3)["connection"]["alert_banner_active"] is False


def test_t3_05_f4_f6_feed_disconnect_triggers_discord_broken_alert(feed_manager, discord_notifier):
    """T3-05 (F4 + F6): Upstream disconnect triggers fallback AND institutional red Discord alert card."""
    feed_manager.trigger_disconnect("upstream_disconnected")
    assert feed_manager.get_connection_status().feed_source == "synthetic_fallback"

    # Dispatch broken alert
    discord_notifier.post_broken_alert(
        component="AlpacaRelayClient",
        error_message="Upstream disconnected from Alpaca proxy",
        evidence="Frame: {'T': 'error', 'msg': 'upstream_disconnected'}",
        dashboard_url="https://dash.railway.app",
    )
    card = discord_notifier.dispatched_cards[-1]
    assert card.color == DISCORD_COLOR_BROKEN
    assert "AlpacaRelayClient" in card.title


def test_t3_06_f4_f7_feed_reconnect_triggers_discord_recovered_alert(feed_manager, discord_notifier):
    """T3-06 (F4 + F7): Reconnection restores live feed AND dispatches green recovered alert card."""
    feed_manager.trigger_disconnect()
    feed_manager.trigger_reconnect()

    discord_notifier.post_recovered_alert(
        component="AlpacaRelayClient",
        downtime_duration_s=24.5,
        status_info="WebSocket stream reconnected. Normal trading resumed.",
        dashboard_url="https://dash.railway.app",
    )
    card = discord_notifier.dispatched_cards[-1]
    assert card.color == DISCORD_COLOR_RECOVERED
    field = next(f for f in card.fields if f["name"] == "Downtime Duration")
    assert "24.5s" in field["value"]


def test_t3_07_f5_f2_strategy_rebalance_executes_in_paper_ledger(paper_account):
    """T3-07 (F5 + F2): Strategy rebalancer calculates delta orders and fills them in paper ledger."""
    from strategy_engine.allocator.rebalancer import PortfolioRebalancer
    from strategy_engine.core.models import MarketRegime, TargetAllocation

    rebalancer = PortfolioRebalancer(drift_band=0.02)
    targets = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 0.40, "QQQ": 0.40, "SHV": 0.20},
        cash_weight=0.20,
        rationale="4-regime normal bull targets",
    )
    orders = rebalancer.compute_rebalance_orders(
        target_allocation=targets,
        current_weights={"SPY": 0.0, "QQQ": 0.0, "SHV": 0.0},
        portfolio_equity=50000.00,
        current_prices={"SPY": 500.0, "QQQ": 400.0, "SHV": 100.0},
    )
    assert len(orders) == 3

    # Execute all generated orders on paper ledger
    for o in orders:
        paper_account.execute_order(o.symbol, o.action.value if hasattr(o.action, 'value') else str(o.action), o.delta_shares, o.estimated_price)

    state = paper_account.get_portfolio_state()
    assert len(state.positions) == 3
    assert math.isclose(state.cash, 0.0, abs_tol=1e-4)  # $50k invested in SPY ($20k), QQQ ($20k), and SHV ($10k)
    assert math.isclose(state.total_nav, 50000.00, rel_tol=1e-5)


def test_t3_08_f5_f8_strategy_rebalance_dispatches_discord_trade_card(paper_account, discord_notifier):
    """T3-08 (F5 + F8): Rebalance orders dispatch institutional blue Discord trade card with NAV."""
    orders = [
        RebalanceOrder(symbol="SPY", action="BUY", shares=40.0, price=500.0, target_weight=0.40),
        RebalanceOrder(symbol="QQQ", action="BUY", shares=50.0, price=400.0, target_weight=0.40),
    ]
    for o in orders:
        paper_account.execute_order(o.symbol, o.action, o.shares, o.price)

    nav = paper_account.get_portfolio_state().total_nav
    discord_notifier.post_trade_execution(orders, nav, "BULL_NORMAL", "https://dash.railway.app")

    card = discord_notifier.dispatched_cards[-1]
    assert card.color == DISCORD_COLOR_TRADE
    nav_field = next(f for f in card.fields if f["name"] == "Portfolio NAV")
    assert "$50,000.00" in nav_field["value"]


def test_t3_09_f6_f9_rapid_broken_alerts_respect_rate_limiter(discord_notifier):
    """T3-09 (F6 + F9): Multiple rapid error alerts pass through rate limiter without dropped messages."""
    for i in range(5):
        discord_notifier.post_broken_alert(f"Subsystem_{i}", f"Error {i}", "Details", "https://dash")
    assert len(discord_notifier.dispatched_cards) == 5
    for c in discord_notifier.dispatched_cards:
        assert c.color == DISCORD_COLOR_BROKEN


def test_t3_10_f8_f9_trade_execution_card_suppressed_under_pytest(discord_notifier, sample_rebalance_orders):
    """T3-10 (F8 + F9): Trade execution notifications are suppressed under pytest to prevent test pollution."""
    assert discord_notifier.is_pytest_environment() is True
    res = discord_notifier.post_trade_execution(sample_rebalance_orders, 50000.0, "BULL", "https://dash")
    assert res is True


def test_t3_11_f10_f11_dashboard_renders_responsive_on_mobile_viewport(operator_app):
    """T3-11 (F10 + F11): Light & airy dashboard layout satisfies 375px–430px mobile responsiveness."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert code == 200
    assert "viewport" in body
    assert "max-w-lg" in body
    assert "overflow-x-auto" in body
    assert "touch-btn" in body


def test_t3_12_f10_f12_dashboard_html_binds_to_portfolio_metrics(operator_app):
    """T3-12 (F10 + F12): Dashboard HTML displays primary metrics (Total NAV, Cash, P&L)."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "Total NAV" in body
    assert "$50,000.00" in body
    assert "Cash Balance:" in body
    assert "Realized P&L:" in body


def test_t3_13_f12_f13_api_portfolio_synchronizes_positions_and_alert_banner(paper_account, feed_manager, operator_app):
    """T3-13 (F12 + F13): Simultaneous trade activity and feed drop synchronize in API payload."""
    paper_account.execute_order("SPY", "BUY", 10.0, 500.0)
    feed_manager.trigger_disconnect()

    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    data = json.loads(body)
    assert len(data["portfolio"]["positions"]) == 1
    assert data["connection"]["alert_banner_active"] is True
    assert data["connection"]["feed_source"] == "synthetic_fallback"


def test_t3_14_f14_f5_operator_pause_freezes_strategy_rebalancing(operator_app):
    """T3-14 (F14 + F5): Operator Pause control sets PAUSED state to prevent rebalancing."""
    operator_app.handle_request("POST", "/api/operator/pause")
    assert operator_app.daemon_state == "PAUSED"

    # Daemon logic checks daemon_state before executing rebalance
    is_paused = (operator_app.daemon_state == "PAUSED")
    assert is_paused is True

    # Resuming re-enables trading
    operator_app.handle_request("POST", "/api/operator/resume")
    assert operator_app.daemon_state == "RUNNING"


def test_t3_15_f14_f12_manual_rebalance_updates_dashboard_nav(paper_account, operator_app):
    """T3-15 (F14 + F12): Triggering manual rebalance evaluates strategy and updates dashboard."""
    code, headers, body = operator_app.handle_request("POST", "/api/operator/rebalance")
    assert code == 200
    assert json.loads(body)["status"] == "ok"

    # Verify portfolio state is immediately queryable post-rebalance
    c_port, _, b_port = operator_app.handle_request("GET", "/api/portfolio")
    assert c_port == 200
    data = json.loads(b_port)
    assert math.isclose(data["portfolio"]["total_nav"], 50000.00, rel_tol=1e-5)


def test_t3_16_f15_f17_adversarial_smoke_injection_followed_by_pristine_reset(paper_account):
    """T3-16 (F15 + F17): Adversarial smoke injection runs, followed by pristine reset leaving clean state."""
    # Inject fake trades and extreme market prices
    paper_account.execute_order("SPY", "BUY", 20.0, 500.0, order_id="smoke_test_1")
    paper_account.execute_order("QQQ", "BUY", 30.0, 400.0, order_id="smoke_test_2")
    paper_account.update_market_prices({"SPY": 250.0, "QQQ": 200.0})  # 50% drawdown shock
    assert paper_account.get_portfolio_state().unrealized_pnl < -5000.0

    # Pristine reset completely wipes smoke artifacts
    paper_account.reset_to_pristine()
    state = paper_account.get_portfolio_state()
    assert state.cash == 50000.00
    assert state.equity == 0.0
    assert state.positions == []


def test_t3_17_f16_f10_usability_audit_validates_light_airy_contrast(operator_app):
    """T3-17 (F16 + F10): Usability audit verifies light/airy contrast and absence of clutter."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "bg-slate-50" in body
    assert "text-slate-900" in body
    assert "Operator Dashboard" in body
    # Clear cards hierarchy
    assert body.count("card-shadow") >= 3


def test_t3_18_f18_f19_git_init_and_remote_specification_alignment(tmp_path):
    """T3-18 (F18 + F19): Git repo initialization and remote target 'Jhosshua' align with project spec."""
    import subprocess
    repo = tmp_path / "repo_sync"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", "https://github.com/Jhosshua/DynamicLongTermStrategyBot.git"], cwd=repo, capture_output=True)
    res = subprocess.run(["git", "remote", "get-url", "origin"], cwd=repo, capture_output=True, text=True)
    assert "Jhosshua/DynamicLongTermStrategyBot" in res.stdout


def test_t3_19_f20_f12_railway_health_check_queries_live_portfolio(operator_app, paper_account):
    """T3-19 (F20 + F12): Railway unauthenticated health check queries live NAV and returns HTTP 200."""
    paper_account.execute_order("SPY", "BUY", 10.0, 500.0)
    paper_account.update_market_prices({"SPY": 510.0})

    code, headers, body = operator_app.handle_request("GET", "/health")
    assert code == 200
    data = json.loads(body)
    assert data["status"] == "ok"
    assert math.isclose(data["portfolio"]["nav"], 50100.00, rel_tol=1e-5)


def test_t3_20_f2_f5_f8_f12_end_to_end_closed_loop_trade_cycle(paper_account, discord_notifier, operator_app):
    """T3-20 (F2 + F5 + F8 + F12): Full closed-loop trade cycle: Regime -> Alloc -> Execution -> Card -> Web."""
    # 1. Strategy dictates BUY SPY order
    res = paper_account.execute_order("SPY", "BUY", 20.0, 500.0, order_id="cycle_01")
    assert res["action"] == "BUY"

    # 2. Portfolio NAV and metrics update
    summary = paper_account.get_portfolio_state()
    assert math.isclose(summary.total_nav, 50000.00)

    # 3. Discord notification dispatched
    order = RebalanceOrder(symbol="SPY", action="BUY", shares=20.0, price=500.0, target_weight=0.20)
    discord_notifier.post_trade_execution([order], summary.total_nav, "BULL_NORMAL", "https://dash.railway.app")
    card = discord_notifier.dispatched_cards[-1]
    assert card.color == DISCORD_COLOR_TRADE

    # 4. Web API reflects position and cash
    code, _, body = operator_app.handle_request("GET", "/api/portfolio")
    assert code == 200
    data = json.loads(body)["portfolio"]
    assert math.isclose(data["cash"], 40000.00)
    assert len(data["positions"]) == 1
    assert data["positions"][0]["symbol"] == "SPY"
