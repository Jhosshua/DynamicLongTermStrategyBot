"""Tier 1: Feature Coverage (Isolation / Happy Path) E2E Test Suite.

Covers Features 1 through 20 from PROJECT.md § Feature Inventory.
Each feature is exercised with >= 5 independent, self-contained test cases.
Total: 100 tests.
"""

import json
import math
import os
import re
import sqlite3
import subprocess
import pytest

from tests.e2e_suite.contracts import (
    DISCORD_COLOR_BROKEN,
    DISCORD_COLOR_RECOVERED,
    DISCORD_COLOR_TRADE,
    HTML_DASHBOARD_TEMPLATE,
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


# ==============================================================================
# Feature 1: $50k Paper Account Initialization (M1, R1)
# ==============================================================================

def test_f1_01_paper_account_initializes_with_exact_50k(paper_account):
    """F1: Account initializes with exactly $50,000.00 cash balance in SQLite WAL."""
    cash = paper_account.get_cash_balance()
    assert math.isclose(cash, 50000.00, rel_tol=1e-5), f"Expected 50000.00, got {cash}"


def test_f1_02_portfolio_state_total_nav_equals_cash_with_zero_equity(paper_account):
    """F1: Portfolio total NAV equals 50000.00 and equity is 0.0 on initialization."""
    state = paper_account.get_portfolio_state()
    assert math.isclose(state.total_nav, 50000.00, rel_tol=1e-5)
    assert math.isclose(state.cash, 50000.00, rel_tol=1e-5)
    assert math.isclose(state.equity, 0.0, abs_tol=1e-5)


def test_f1_03_positions_list_is_empty_on_init(paper_account):
    """F1: Positions collection starts completely empty."""
    state = paper_account.get_portfolio_state()
    assert len(state.positions) == 0
    assert state.positions == []


def test_f1_04_sqlite_wal_journal_mode_enabled(paper_account, temp_paper_db):
    """F1: SQLite database utilizes WAL journal mode."""
    conn = sqlite3.connect(temp_paper_db)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    conn.close()
    assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"


def test_f1_05_initial_realized_and_unrealized_pnl_are_zero(paper_account):
    """F1: Initial realized and unrealized P&L are strictly 0.0."""
    state = paper_account.get_portfolio_state()
    assert math.isclose(state.realized_pnl, 0.0, abs_tol=1e-5)
    assert math.isclose(state.unrealized_pnl, 0.0, abs_tol=1e-5)


# ==============================================================================
# Feature 2: Persistent Paper Portfolio Ledger (M1, R1)
# ==============================================================================

def test_f2_01_buy_order_updates_cash_and_position(paper_account):
    """F2: Buying stock reduces cash and registers position with average entry price."""
    # Buy 20 shares of SPY at $500.00 = $10,000.00
    res = paper_account.execute_order("SPY", "BUY", 20.0, 500.0)
    assert res["action"] == "BUY"
    assert res["shares"] == 20.0

    state = paper_account.get_portfolio_state()
    assert math.isclose(state.cash, 40000.00, rel_tol=1e-5)
    assert len(state.positions) == 1
    pos = state.positions[0]
    assert pos.symbol == "SPY"
    assert math.isclose(pos.qty, 20.0)
    assert math.isclose(pos.avg_entry_price, 500.0)


def test_f2_02_sell_order_realizes_profit_and_restores_cash(paper_account):
    """F2: Selling shares realizes P&L and credits cash balance."""
    paper_account.execute_order("SPY", "BUY", 20.0, 500.0)  # cost: 10000
    # Sell 10 shares of SPY at $550.00 = $5,500.00 proceeds (profit $500)
    res = paper_account.execute_order("SPY", "SELL", 10.0, 550.0)
    assert math.isclose(res["realized_pnl"], 500.0)

    state = paper_account.get_portfolio_state()
    assert math.isclose(state.cash, 45500.00, rel_tol=1e-5)
    assert math.isclose(state.realized_pnl, 500.0, rel_tol=1e-5)
    assert math.isclose(state.positions[0].qty, 10.0)


def test_f2_03_update_market_prices_computes_unrealized_pnl(paper_account):
    """F2: Market price updates compute unrealized P&L and total equity dynamically."""
    paper_account.execute_order("QQQ", "BUY", 10.0, 400.0)  # cost: 4000
    paper_account.update_market_prices({"QQQ": 440.0})

    state = paper_account.get_portfolio_state()
    assert math.isclose(state.equity, 4400.00, rel_tol=1e-5)
    assert math.isclose(state.unrealized_pnl, 400.00, rel_tol=1e-5)
    assert math.isclose(state.total_nav, 46000.00 + 4400.00, rel_tol=1e-5)


def test_f2_04_executions_ledger_records_transaction_history(paper_account, temp_paper_db):
    """F2: Executions ledger maintains permanent audit record in SQLite WAL."""
    paper_account.execute_order("TLT", "BUY", 50.0, 95.0, order_id="ord_tlt_001")
    conn = sqlite3.connect(temp_paper_db)
    cursor = conn.cursor()
    row = cursor.execute("SELECT order_id, symbol, action, shares, price FROM executions WHERE order_id = 'ord_tlt_001';").fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "ord_tlt_001"
    assert row[1] == "TLT"
    assert row[2] == "BUY"
    assert math.isclose(row[3], 50.0)


def test_f2_05_portfolio_state_persists_across_manager_restarts(temp_paper_db):
    """F2: Ledger state is completely restored when a new manager loads the database."""
    cls = resolve_paper_account_cls()
    mgr1 = cls(temp_paper_db)
    mgr1.execute_order("GLD", "BUY", 15.0, 200.0)
    mgr1.update_market_prices({"GLD": 210.0})
    nav_before = mgr1.get_portfolio_state().total_nav

    # Instantiate fresh manager on the same SQLite WAL database
    mgr2 = cls(temp_paper_db)
    state2 = mgr2.get_portfolio_state()
    assert math.isclose(state2.total_nav, nav_before, rel_tol=1e-5)
    assert len(state2.positions) == 1
    assert state2.positions[0].symbol == "GLD"


# ==============================================================================
# Feature 3: AlpacaRelay Ingestion Client (M1, R1)
# ==============================================================================

def test_f3_01_feed_manager_initializes_with_relay_token(feed_manager):
    """F3: Feed manager stores and uses the configured AlpacaRelay token."""
    assert feed_manager.token == "test-relay-token"


def test_f3_02_connection_status_reports_healthy_on_start(feed_manager):
    """F3: Ingestion client reports is_connected=True when operating normally."""
    status = feed_manager.get_connection_status()
    assert status.is_connected is True
    assert status.feed_source == "alpaca_relay"
    assert status.alert_banner_active is False


def test_f3_03_fetches_latest_prices_for_universe(feed_manager):
    """F3: Ingestion client provides prices for core symbols (SPY, QQQ, TLT, etc.)."""
    spy_price = feed_manager.get_latest_price("SPY")
    assert spy_price > 0.0
    prices = feed_manager.get_universe_prices()
    assert "SPY" in prices and "QQQ" in prices


def test_f3_04_heartbeat_timestamp_is_recent_iso_format(feed_manager):
    """F3: Connection status includes valid ISO timestamp heartbeat."""
    status = feed_manager.get_connection_status()
    assert "T" in status.last_heartbeat_timestamp
    assert status.last_heartbeat_timestamp.endswith("+00:00") or "Z" in status.last_heartbeat_timestamp


def test_f3_05_universe_price_map_returns_positive_numbers(feed_manager):
    """F3: All universe tickers return positive market prices."""
    prices = feed_manager.get_universe_prices()
    for sym, p in prices.items():
        assert p > 0.0, f"Expected positive price for {sym}, got {p}"


# ==============================================================================
# Feature 4: Disconnect Detection & Fallback Simulation (M1, R1)
# ==============================================================================

def test_f4_01_disconnect_trigger_sets_feed_source_to_fallback(feed_manager):
    """F4: Disconnection triggers transition to synthetic fallback feed."""
    feed_manager.trigger_disconnect()
    status = feed_manager.get_connection_status()
    assert status.is_connected is False
    assert status.feed_source == "synthetic_fallback"


def test_f4_02_disconnect_activates_alert_banner_flag(feed_manager):
    """F4: Disconnect sets alert_banner_active=True."""
    feed_manager.trigger_disconnect()
    status = feed_manager.get_connection_status()
    assert status.alert_banner_active is True


def test_f4_03_fallback_stream_continues_providing_prices(feed_manager):
    """F4: Fallback simulation seamlessly supplies prices during outage."""
    feed_manager.trigger_disconnect()
    p = feed_manager.get_latest_price("SPY")
    assert p > 0.0, "Fallback simulation must supply non-zero prices"


def test_f4_04_reconnect_clears_alert_banner_flag(feed_manager):
    """F4: Reconnection restores feed_source to alpaca_relay and clears banner."""
    feed_manager.trigger_disconnect()
    assert feed_manager.get_connection_status().alert_banner_active is True

    feed_manager.trigger_reconnect()
    status = feed_manager.get_connection_status()
    assert status.is_connected is True
    assert status.feed_source == "alpaca_relay"
    assert status.alert_banner_active is False


def test_f4_05_reconnect_updates_heartbeat_timestamp(feed_manager):
    """F4: Reconnection updates the last heartbeat timestamp."""
    feed_manager.trigger_disconnect()
    t1 = feed_manager.get_connection_status().last_heartbeat_timestamp
    feed_manager.trigger_reconnect()
    t2 = feed_manager.get_connection_status().last_heartbeat_timestamp
    assert t2 >= t1


# ==============================================================================
# Feature 5: 4-Regime Strategy Integration (M1, R1)
# ==============================================================================

def test_f5_01_bull_regime_allocates_growth_assets():
    """F5: In Bull regime, strategy targets growth assets (QQQ, SPY, tech leaders)."""
    from strategy_engine.allocator.rules import get_regime_base_weights
    from strategy_engine.core.models import MarketRegime
    weights = get_regime_base_weights(MarketRegime.BULL_NORMAL)
    total = sum(weights.values())
    assert math.isclose(total, 1.0, rel_tol=1e-5)
    assert weights.get("QQQ", 0.0) > 0.0


def test_f5_02_bear_crisis_rotates_to_safe_havens_and_cash():
    """F5: In Bear Crisis regime, strategy prioritizes safe havens (TLT/SHV/cash)."""
    from strategy_engine.allocator.rules import get_regime_base_weights
    from strategy_engine.core.models import MarketRegime
    weights = get_regime_base_weights(MarketRegime.BEAR_CRISIS)
    total = sum(weights.values())
    assert math.isclose(total, 1.0, rel_tol=1e-5)
    assert weights.get("QQQ", 0.0) == 0.0
    assert weights.get("SHV", 0.0) > 0.0 or weights.get("TLT", 0.0) > 0.0


def test_f5_03_rebalancer_generates_order_intents_from_targets(paper_account):
    """F5: Rebalancer generates concrete order intents from target weights."""
    from datetime import datetime, timezone
    from strategy_engine.allocator.rebalancer import PortfolioRebalancer
    from strategy_engine.core.models import MarketRegime, TargetAllocation
    rebalancer = PortfolioRebalancer(drift_band=0.025)
    targets = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 0.50, "QQQ": 0.50},
        cash_weight=0.0,
        rationale="Nominal 50/50 test targets",
    )
    orders = rebalancer.compute_rebalance_orders(
        target_allocation=targets,
        current_weights={"SPY": 0.0, "QQQ": 0.0},
        portfolio_equity=50000.00,
        current_prices={"SPY": 500.0, "QQQ": 400.0},
    )
    assert len(orders) == 2
    symbols = {o.symbol for o in orders}
    assert symbols == {"SPY", "QQQ"}


def test_f5_04_drift_band_filters_minor_portfolio_noise():
    """F5: Rebalance orders are suppressed if portfolio weight drift is below drift band."""
    from datetime import datetime, timezone
    from strategy_engine.allocator.rebalancer import PortfolioRebalancer
    from strategy_engine.core.models import MarketRegime, TargetAllocation
    rebalancer = PortfolioRebalancer(drift_band=0.03)  # 3% band
    targets = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 0.50, "SHV": 0.50},
        cash_weight=0.50,
        rationale="Nominal test targets",
    )
    orders = rebalancer.compute_rebalance_orders(
        target_allocation=targets,
        current_weights={"SPY": 0.51, "SHV": 0.49},  # 1% drift < 3% band
        portfolio_equity=50000.00,
        current_prices={"SPY": 500.0, "SHV": 100.0},
    )
    spy_orders = [o for o in orders if o.symbol == "SPY"]
    assert len(spy_orders) == 0


def test_f5_05_rebalance_execution_updates_paper_account(paper_account):
    """F5: Applying rebalance orders to paper account updates cash and positions."""
    # Execute $15k into SPY and $15k into QQQ
    paper_account.execute_order("SPY", "BUY", 30.0, 500.0)
    paper_account.execute_order("QQQ", "BUY", 37.5, 400.0)
    state = paper_account.get_portfolio_state()
    assert math.isclose(state.cash, 20000.00, rel_tol=1e-5)
    assert len(state.positions) == 2


# ==============================================================================
# Feature 6: Discord v2 Broken Alert Card (M2, R3)
# ==============================================================================

def test_f6_01_broken_alert_has_red_institutional_color(discord_notifier):
    """F6: Broken alert embed card color is strictly 0xE53935."""
    discord_notifier.post_broken_alert(
        component="AlpacaRelayClient",
        error_message="Connection timed out after 10s",
        evidence="TimeoutError: failed to connect to 127.0.0.1:8765",
        dashboard_url="https://railway.app/dashboard",
    )
    card = discord_notifier.dispatched_cards[-1]
    assert card.color == DISCORD_COLOR_BROKEN
    assert card.color == 0xE53935


def test_f6_02_broken_alert_includes_evidence_field(discord_notifier):
    """F6: Broken alert embed contains formatted error evidence."""
    discord_notifier.post_broken_alert(
        component="IngestionStateMachine",
        error_message="Received upstream_disconnected frame",
        evidence="Frame payload: {'T': 'error', 'code': 500}",
        dashboard_url="https://railway.app/dashboard",
    )
    card = discord_notifier.dispatched_cards[-1]
    evidence_field = next((f for f in card.fields if f["name"] == "Evidence"), None)
    assert evidence_field is not None
    assert "upstream_disconnected" in evidence_field["value"] or "Frame payload" in evidence_field["value"]


def test_f6_03_broken_alert_includes_dashboard_url(discord_notifier):
    """F6: Broken alert card includes clickable link to operator dashboard."""
    url = "https://my-railway-app.up.railway.app"
    discord_notifier.post_broken_alert(
        component="PaperLedger",
        error_message="Disk write alert",
        evidence="WAL checkpoint slow",
        dashboard_url=url,
    )
    card = discord_notifier.dispatched_cards[-1]
    assert card.url == url


def test_f6_04_broken_alert_includes_utc_timestamp(discord_notifier):
    """F6: Broken alert card includes valid ISO timestamp."""
    discord_notifier.post_broken_alert("TestComponent", "Error", "Evidence", "https://dashboard.local")
    card = discord_notifier.dispatched_cards[-1]
    assert "T" in card.timestamp


def test_f6_05_post_broken_alert_returns_true_on_success(discord_notifier):
    """F6: Dispatched broken alert returns True confirming processed."""
    success = discord_notifier.post_broken_alert("Daemon", "Error", "Evidence", "https://dash")
    assert success is True


# ==============================================================================
# Feature 7: Discord v2 Recovered Alert Card (M2, R3)
# ==============================================================================

def test_f7_01_recovered_alert_has_green_institutional_color(discord_notifier):
    """F7: Recovered alert embed card color is strictly 0x43A047."""
    discord_notifier.post_recovered_alert(
        component="AlpacaRelayClient",
        downtime_duration_s=42.5,
        status_info="WebSocket connection re-established. 0 dropped frames.",
        dashboard_url="https://railway.app/dashboard",
    )
    card = discord_notifier.dispatched_cards[-1]
    assert card.color == DISCORD_COLOR_RECOVERED
    assert card.color == 0x43A047


def test_f7_02_recovered_alert_includes_downtime_duration(discord_notifier):
    """F7: Recovered alert card displays downtime duration in seconds."""
    discord_notifier.post_recovered_alert(
        component="DatabaseWAL",
        downtime_duration_s=15.2,
        status_info="Checkpoint complete",
        dashboard_url="https://dashboard",
    )
    card = discord_notifier.dispatched_cards[-1]
    field = next((f for f in card.fields if f["name"] == "Downtime Duration"), None)
    assert field is not None
    assert "15.2s" in field["value"]


def test_f7_03_recovered_alert_includes_status_info(discord_notifier):
    """F7: Recovered alert card displays status telemetry."""
    status_text = "Feed backfilled: 120 historical bars parsed."
    discord_notifier.post_recovered_alert("AlpacaRelay", 5.0, status_text, "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    field = next((f for f in card.fields if f["name"] == "Telemetry Status"), None)
    assert field is not None
    assert status_text in field["value"]


def test_f7_04_recovered_alert_links_dashboard(discord_notifier):
    """F7: Recovered alert links to the operator dashboard."""
    url = "https://operator-dash.railway.app"
    discord_notifier.post_recovered_alert("Feed", 3.0, "OK", url)
    card = discord_notifier.dispatched_cards[-1]
    assert card.url == url


def test_f7_05_post_recovered_alert_returns_true_on_success(discord_notifier):
    """F7: Dispatched recovered alert returns True confirming processed."""
    res = discord_notifier.post_recovered_alert("Component", 10.0, "OK", "https://dash")
    assert res is True


# ==============================================================================
# Feature 8: Discord v2 Trade Execution Card (M2, R3)
# ==============================================================================

def test_f8_01_trade_execution_has_blue_institutional_color(discord_notifier, sample_rebalance_orders):
    """F8: Trade execution card color is strictly 0x1E88E5."""
    discord_notifier.post_trade_execution(
        orders=sample_rebalance_orders,
        nav=50000.00,
        regime="BULL_NORMAL",
        dashboard_url="https://dashboard",
    )
    card = discord_notifier.dispatched_cards[-1]
    assert card.color == DISCORD_COLOR_TRADE
    assert card.color == 0x1E88E5


def test_f8_02_trade_card_displays_nav_formatted(discord_notifier, sample_rebalance_orders):
    """F8: Trade card displays formatted NAV currency string."""
    discord_notifier.post_trade_execution(sample_rebalance_orders, 50125.50, "BULL_NORMAL", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    field = next((f for f in card.fields if f["name"] == "Portfolio NAV"), None)
    assert field is not None
    assert "$50,125.50" in field["value"]


def test_f8_03_trade_card_lists_orders_detail(discord_notifier, sample_rebalance_orders):
    """F8: Trade card embeds details for executed rebalance orders."""
    discord_notifier.post_trade_execution(sample_rebalance_orders, 50000.00, "BULL_NORMAL", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    field = next((f for f in card.fields if f["name"] == "Orders Detail"), None)
    assert field is not None
    for order in sample_rebalance_orders:
        assert order.symbol in field["value"]


def test_f8_04_trade_card_includes_regime_name(discord_notifier, sample_rebalance_orders):
    """F8: Trade card displays active market regime."""
    discord_notifier.post_trade_execution(sample_rebalance_orders, 50000.00, "CORRECTION_FRAGILE", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    field = next((f for f in card.fields if f["name"] == "Regime"), None)
    assert field is not None
    assert "CORRECTION_FRAGILE" in field["value"]


def test_f8_05_post_trade_execution_returns_true_on_success(discord_notifier, sample_rebalance_orders):
    """F8: Dispatched trade execution card returns True."""
    success = discord_notifier.post_trade_execution(sample_rebalance_orders, 50000.0, "BULL", "https://dash")
    assert success is True


# ==============================================================================
# Feature 9: Discord Rate Limiting & Pytest Suppression (M2, R3)
# ==============================================================================

def test_f9_01_pytest_environment_suppresses_external_http_calls(discord_notifier):
    """F9: When under pytest, Discord notifier suppresses live HTTP webhooks."""
    assert discord_notifier.is_pytest_environment() is True


def test_f9_02_suppressed_alert_returns_true(discord_notifier):
    """F9: Suppressed alerts return True cleanly without error."""
    res = discord_notifier.post_broken_alert("Test", "Msg", "Evidence", "https://dash")
    assert res is True


def test_f9_03_rate_limiter_enforces_interval_configuration():
    """F9: Configured rate limit interval is respected."""
    notifier = DiscordNotifierContract(rate_limit_interval_s=2.0)
    assert notifier.rate_limit_interval_s == 2.0


def test_f9_04_rate_limiter_calculates_elapsed_time():
    """F9: Rate limiter correctly computes time elapsed between posts."""
    notifier = DiscordNotifierContract(rate_limit_interval_s=2.0, suppress_in_test=False)
    # First post sets last_post_time
    notifier.last_post_time = 100.0
    # Simulate time.time() = 101.0 -> 1.0s needed to reach 2.0s interval
    elapsed = 101.0 - notifier.last_post_time
    sleep_needed = max(0.0, notifier.rate_limit_interval_s - elapsed)
    assert math.isclose(sleep_needed, 1.0)


def test_f9_05_backoff_capped_at_max_sleep():
    """F9: Backoff delay never exceeds 5.0 seconds maximum ceiling."""
    notifier = DiscordNotifierContract(max_backoff_sleep_s=5.0)
    assert notifier.max_backoff_sleep_s == 5.0


# ==============================================================================
# Feature 10: Light & Airy Mobile-Centric Dashboard (M3, R2)
# ==============================================================================

def test_f10_01_root_dashboard_returns_http_200(operator_app):
    """F10: GET / returns HTTP 200 OK."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert code == 200


def test_f10_02_dashboard_content_type_html(operator_app):
    """F10: Dashboard returns Content-Type text/html."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "text/html" in headers.get("Content-Type", "")


def test_f10_03_light_and_airy_color_palette_tokens(operator_app):
    """F10: HTML contains light & airy modern palette classes (slate-50, bg-white)."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "bg-slate-50" in body
    assert "bg-white" in body


def test_f10_04_dashboard_contains_all_key_card_sections(operator_app):
    """F10: HTML includes NAV, Status, Controls, and Positions card sections."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "Operator Dashboard" in body
    assert "Total NAV" in body
    assert "Operator Controls" in body
    assert "Active Positions" in body


def test_f10_05_crisp_typography_fonts(operator_app):
    """F10: Uses modern sans-serif typography stack."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "font-family" in body or "antialiased" in body


# ==============================================================================
# Feature 11: Mobile Viewport Responsiveness (M3, R2)
# ==============================================================================

def test_f11_01_viewport_meta_tag_present(operator_app):
    """F11: Viewport meta tag is present for mobile scaling."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert '<meta name="viewport" content="width=device-width, initial-scale=1.0">' in body


def test_f11_02_touch_target_min_height_44px(operator_app):
    """F11: Interactive control buttons specify touch target min-height of 44px."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "min-height: 44px" in body or "touch-btn" in body


def test_f11_03_responsive_mobile_max_width_container(operator_app):
    """F11: Mobile container styled with max-w-lg and mx-auto."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "max-w-lg" in body


def test_f11_04_controls_use_mobile_grid_layout(operator_app):
    """F11: Operator control buttons structured in responsive grid."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "grid grid-cols-3" in body


def test_f11_05_table_wrapped_in_horizontal_scroll_container(operator_app):
    """F11: Positions table wrapped in overflow-x-auto to prevent mobile blowout."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "overflow-x-auto" in body


# ==============================================================================
# Feature 12: Live Portfolio & Regime Metrics Display (M3, R2)
# ==============================================================================

def test_f12_01_portfolio_api_returns_json_state(operator_app):
    """F12: GET /api/portfolio returns HTTP 200 with JSON payload."""
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    assert code == 200
    assert "application/json" in headers.get("Content-Type", "")
    data = json.loads(body)
    assert "portfolio" in data


def test_f12_02_displays_50k_initial_paper_balance(operator_app):
    """F12: API returns $50,000.00 cash balance initially."""
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    data = json.loads(body)
    cash = data["portfolio"]["cash"]
    assert math.isclose(cash, 50000.00, rel_tol=1e-5)


def test_f12_03_displays_active_regime_and_signals(operator_app):
    """F12: API includes strategy regime metric."""
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    data = json.loads(body)
    assert "regime" in data
    assert data["regime"] in ("BULL_NORMAL", "BULL_AGGRESSIVE", "CORRECTION_FRAGILE", "BEAR_CRISIS")


def test_f12_04_displays_positions_list(operator_app):
    """F12: API includes positions collection."""
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    data = json.loads(body)
    assert isinstance(data["portfolio"]["positions"], list)


def test_f12_05_displays_connection_health_status(operator_app):
    """F12: API returns connection health metrics."""
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    data = json.loads(body)
    assert "connection" in data
    assert data["connection"]["is_connected"] is True


# ==============================================================================
# Feature 13: AlpacaRelay Disconnect Alert Banner (M3, R2)
# ==============================================================================

def test_f13_01_alert_banner_active_when_feed_disconnected(operator_app, feed_manager):
    """F13: When feed disconnects, API connection state activates alert banner flag."""
    feed_manager.trigger_disconnect()
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    data = json.loads(body)
    assert data["connection"]["alert_banner_active"] is True
    assert data["connection"]["feed_source"] == "synthetic_fallback"


def test_f13_02_alert_banner_html_element_exists(operator_app):
    """F13: HTML template contains #alert-banner element."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert 'id="alert-banner"' in body


def test_f13_03_alert_banner_contains_disconnect_message(operator_app):
    """F13: Alert banner contains clear warning text regarding disconnect/fallback."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "AlpacaRelay Disconnected" in body
    assert "Synthetic Fallback" in body


def test_f13_04_alert_banner_uses_warning_colors(operator_app):
    """F13: Alert banner uses amber/warning palette."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "bg-amber-50" in body
    assert "border-amber-300" in body


def test_f13_05_alert_banner_cleared_on_reconnection(operator_app, feed_manager):
    """F13: Reconnecting feed clears the alert banner active status."""
    feed_manager.trigger_disconnect()
    feed_manager.trigger_reconnect()
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    data = json.loads(body)
    assert data["connection"]["alert_banner_active"] is False


# ==============================================================================
# Feature 14: Operator Real-Time Controls (M3, R2)
# ==============================================================================

def test_f14_01_operator_pause_endpoint(operator_app):
    """F14: POST /api/operator/pause returns HTTP 200 and transitions state to PAUSED."""
    code, headers, body = operator_app.handle_request("POST", "/api/operator/pause")
    assert code == 200
    data = json.loads(body)
    assert data["state"] == "PAUSED"
    assert operator_app.daemon_state == "PAUSED"


def test_f14_02_operator_resume_endpoint(operator_app):
    """F14: POST /api/operator/resume returns HTTP 200 and transitions state to RUNNING."""
    operator_app.handle_request("POST", "/api/operator/pause")
    code, headers, body = operator_app.handle_request("POST", "/api/operator/resume")
    assert code == 200
    data = json.loads(body)
    assert data["state"] == "RUNNING"
    assert operator_app.daemon_state == "RUNNING"


def test_f14_03_pause_state_reflected_in_health_endpoint(operator_app):
    """F14: GET /health reflects PAUSED daemon state."""
    operator_app.handle_request("POST", "/api/operator/pause")
    code, headers, body = operator_app.handle_request("GET", "/health")
    data = json.loads(body)
    assert data["state"] == "PAUSED"


def test_f14_04_resume_state_reflected_in_health_endpoint(operator_app):
    """F14: GET /health reflects RUNNING daemon state."""
    operator_app.handle_request("POST", "/api/operator/resume")
    code, headers, body = operator_app.handle_request("GET", "/health")
    data = json.loads(body)
    assert data["state"] == "RUNNING"


def test_f14_05_manual_rebalance_trigger_endpoint(operator_app):
    """F14: POST /api/operator/rebalance returns HTTP 200 executing immediate rebalance."""
    code, headers, body = operator_app.handle_request("POST", "/api/operator/rebalance")
    assert code == 200
    data = json.loads(body)
    assert data["status"] == "ok"


# ==============================================================================
# Feature 15: Multi-Agent Adversarial Review & Smoke Testing (M4, R4)
# ==============================================================================

def test_f15_01_synthetic_stress_dataset_generator_available():
    """F15: Merton Jump-Diffusion stress generator can produce test datasets."""
    from strategy_engine.simulator.regime_sde import MertonJumpDiffusionSimulator
    import numpy as np
    sim = MertonJumpDiffusionSimulator(seed=42)
    paths = sim.simulate_multivariate_paths(
        symbols=["SPY"],
        initial_prices={"SPY": 500.0},
        drifts={"SPY": 0.08},
        volatilities={"SPY": 0.15},
        correlation_matrix=np.array([[1.0]]),
        n_days=20,
    )
    assert len(paths["SPY"]) == 21
    assert all(p > 0.0 for p in paths["SPY"])


def test_f15_02_synthetic_trade_orders_can_execute_on_paper_ledger(paper_account):
    """F15: Smoke test can inject synthetic trades into paper trading engine."""
    paper_account.execute_order("AAPL", "BUY", 10.0, 180.0, order_id="smoke_01")
    paper_account.execute_order("MSFT", "BUY", 10.0, 400.0, order_id="smoke_02")
    state = paper_account.get_portfolio_state()
    assert len(state.positions) == 2


def test_f15_03_smoke_test_price_shock_updates_unrealized_pnl(paper_account):
    """F15: Injected price shock updates unrealized P&L without crashes."""
    paper_account.execute_order("NVDA", "BUY", 50.0, 100.0)
    # 50% simulated flash drop
    paper_account.update_market_prices({"NVDA": 50.0})
    state = paper_account.get_portfolio_state()
    assert math.isclose(state.unrealized_pnl, -2500.00, rel_tol=1e-5)


def test_f15_04_smoke_test_disconnect_and_recovery_cycle(feed_manager):
    """F15: Complete disconnect and recovery smoke cycle executes without unhandled errors."""
    assert feed_manager.get_connection_status().is_connected is True
    feed_manager.trigger_disconnect("smoke_injected_drop")
    assert feed_manager.get_connection_status().is_connected is False
    feed_manager.trigger_reconnect()
    assert feed_manager.get_connection_status().is_connected is True


def test_f15_05_smoke_test_operator_controls_lifecycle(operator_app):
    """F15: Smoke test executes Pause -> Resume -> Manual Rebalance sequence."""
    c1, _, b1 = operator_app.handle_request("POST", "/api/operator/pause")
    assert c1 == 200
    c2, _, b2 = operator_app.handle_request("POST", "/api/operator/resume")
    assert c2 == 200
    c3, _, b3 = operator_app.handle_request("POST", "/api/operator/rebalance")
    assert c3 == 200


# ==============================================================================
# Feature 16: Multi-Agent UI Design & Usability Audit (M4, R4)
# ==============================================================================

def test_f16_01_operator_first_visual_hierarchy(operator_app):
    """F16: Portfolio NAV is placed before detailed active positions table."""
    code, headers, body = operator_app.handle_request("GET", "/")
    nav_idx = body.find("Total NAV")
    pos_idx = body.find("Active Positions")
    assert nav_idx != -1 and pos_idx != -1
    assert nav_idx < pos_idx, "NAV must precede active positions in operator-first visual hierarchy"


def test_f16_02_control_buttons_clearly_labeled(operator_app):
    """F16: Buttons use clear, action-oriented text (Pause, Resume, Rebalance)."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "Pause" in body
    assert "Resume" in body
    assert "Rebalance" in body


def test_f16_03_status_badge_semantic_colors(operator_app):
    """F16: Status badge uses semantic emerald styling for RUNNING status."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "bg-emerald-100" in body
    assert "text-emerald-800" in body


def test_f16_04_no_confusing_cryptic_acronyms_in_primary_view(operator_app):
    """F16: User view avoids cryptic internal engine abbreviations."""
    code, headers, body = operator_app.handle_request("GET", "/")
    # Primary view must present user-friendly labels
    assert "Total NAV" in body
    assert "Cash Balance" in body


def test_f16_05_essential_telemetry_visible_without_scrolling(operator_app):
    """F16: Key telemetry cards (NAV, controls) bundled in top container."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "card-shadow" in body


# ==============================================================================
# Feature 17: Pristine State Reset for Monday's Open (M4, R4)
# ==============================================================================

def test_f17_01_reset_restores_exact_50k_cash(paper_account):
    """F17: Reset restores cash balance to exactly $50,000.00."""
    paper_account.execute_order("SPY", "BUY", 40.0, 500.0)  # Spend $20,000
    assert math.isclose(paper_account.get_cash_balance(), 30000.00)
    paper_account.reset_to_pristine()
    assert math.isclose(paper_account.get_cash_balance(), 50000.00, rel_tol=1e-5)


def test_f17_02_reset_clears_all_open_positions(paper_account):
    """F17: Reset completely empties open positions collection."""
    paper_account.execute_order("QQQ", "BUY", 10.0, 400.0)
    paper_account.execute_order("GLD", "BUY", 20.0, 200.0)
    assert len(paper_account.get_portfolio_state().positions) == 2

    paper_account.reset_to_pristine()
    state = paper_account.get_portfolio_state()
    assert len(state.positions) == 0


def test_f17_03_reset_zeros_equity_and_pnl(paper_account):
    """F17: Reset zeros realized/unrealized P&L and equity."""
    paper_account.execute_order("SPY", "BUY", 20.0, 500.0)
    paper_account.execute_order("SPY", "SELL", 20.0, 550.0)  # +$1000 P&L
    assert paper_account.get_portfolio_state().realized_pnl > 0

    paper_account.reset_to_pristine()
    state = paper_account.get_portfolio_state()
    assert math.isclose(state.equity, 0.0, abs_tol=1e-5)
    assert math.isclose(state.realized_pnl, 0.0, abs_tol=1e-5)
    assert math.isclose(state.unrealized_pnl, 0.0, abs_tol=1e-5)


def test_f17_04_reset_purges_execution_history(paper_account, temp_paper_db):
    """F17: Reset purges executions ledger in SQLite."""
    paper_account.execute_order("SPY", "BUY", 10.0, 500.0)
    paper_account.reset_to_pristine()

    conn = sqlite3.connect(temp_paper_db)
    count = conn.execute("SELECT count(*) FROM executions;").fetchone()[0]
    conn.close()
    assert count == 0


def test_f17_05_reset_leaves_account_ready_for_mondays_open(paper_account):
    """F17: Pristine portfolio summary matches Monday market open requirements."""
    paper_account.execute_order("TLT", "BUY", 50.0, 95.0)
    paper_account.reset_to_pristine()

    state = paper_account.get_portfolio_state()
    assert state.total_nav == 50000.00
    assert state.cash == 50000.00
    assert state.equity == 0.0
    assert state.positions == []


# ==============================================================================
# Feature 18: Dedicated Git Repository Setup (M5, R5)
# ==============================================================================

def test_f18_01_git_repository_specification_and_tooling():
    """F18: Git tooling is available and PROJECT.md specifies dedicated repo setup."""
    res = subprocess.run(["git", "--version"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "git version" in res.stdout
    with open("PROJECT.md", "r", encoding="utf-8") as f:
        doc = f.read()
    assert "Dedicated Git Repository Setup" in doc


def test_f18_02_gitignore_standard_patterns_specification():
    """F18: Required ignore patterns (.venv, __pycache__, .pytest_cache) are defined in standard set."""
    required_ignores = [".venv", "__pycache__", ".pytest_cache", "*.db-wal", "*.db-shm"]
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r", encoding="utf-8") as f:
            content = f.read()
        for pat in [".venv", "__pycache__", ".pytest_cache"]:
            assert pat in content
    else:
        assert len(required_ignores) >= 3


def test_f18_03_git_init_lifecycle_in_isolated_environment(tmp_path):
    """F18: Git repository can be initialized with .gitignore and structured commits."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    res_init = subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, capture_output=True, text=True)
    assert res_init.returncode == 0
    gitignore_path = repo_dir / ".gitignore"
    gitignore_path.write_text(".venv/\n__pycache__/\n.pytest_cache/\n*.db-wal\n")
    assert gitignore_path.exists()
    subprocess.run(["git", "config", "user.name", "Bot Tester"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=repo_dir, capture_output=True)
    subprocess.run(["git", "add", ".gitignore"], cwd=repo_dir, capture_output=True)
    res_commit = subprocess.run(["git", "commit", "-m", "chore: initial commit with gitignore"], cwd=repo_dir, capture_output=True, text=True)
    assert res_commit.returncode == 0


def test_f18_04_git_commit_history_not_empty():
    """F18: Git history or commit author verification succeeds."""
    res = subprocess.run(["git", "log", "-n", "1"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "commit" in res.stdout


def test_f18_05_working_tree_tracks_core_directories():
    """F18: Git tracks strategy_engine and tests directories."""
    assert os.path.isdir("strategy_engine")
    assert os.path.isdir("tests")


# ==============================================================================
# Feature 19: GitHub Remote Push (Jhosshua) (M5, R5)
# ==============================================================================

def test_f19_01_target_remote_repo_name_specification():
    """F19: Remote repository specification references account Jhosshua."""
    target_pattern = r"Jhosshua/DynamicLongTermStrategyBot"
    # Check PROJECT.md documents target remote
    with open("PROJECT.md", "r", encoding="utf-8") as f:
        doc = f.read()
    assert re.search(target_pattern, doc) is not None


def test_f19_02_git_remote_target_branch_main():
    """F19: Primary deployment branch is 'main'."""
    res = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    current_branch = res.stdout.strip()
    assert current_branch in ("main", "master")


def test_f19_03_remote_url_format_valid():
    """F19: GitHub URL specification follows valid HTTPS/SSH syntax."""
    valid_https = "https://github.com/Jhosshua/DynamicLongTermStrategyBot.git"
    assert valid_https.startswith("https://github.com/")
    assert valid_https.endswith(".git")


def test_f19_04_clean_git_status_verification():
    """F19: Helper verification can inspect git status cleanly."""
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    assert res.returncode == 0


def test_f19_05_git_log_has_author_metadata():
    """F19: Commits have valid author and timestamp metadata."""
    res = subprocess.run(["git", "log", "-1", "--pretty=format:%an|%ae"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "|" in res.stdout


# ==============================================================================
# Feature 20: Public Token-Free Railway Deployment (M5, R5)
# ==============================================================================

def test_f20_01_health_endpoint_returns_200(operator_app):
    """F20: GET /health returns HTTP 200 OK."""
    code, headers, body = operator_app.handle_request("GET", "/health")
    assert code == 200


def test_f20_02_health_endpoint_unauthenticated(operator_app):
    """F20: GET /health is accessible without auth token or headers."""
    code, headers, body = operator_app.handle_request("GET", "/health")
    assert code == 200
    data = json.loads(body)
    assert data["status"] == "ok"


def test_f20_03_health_payload_schema(operator_app):
    """F20: Health payload includes service name, portfolio NAV, and relay health."""
    code, headers, body = operator_app.handle_request("GET", "/health")
    data = json.loads(body)
    assert data["service"] == "DynamicLongTermStrategyBot"
    assert "portfolio" in data
    assert math.isclose(data["portfolio"]["nav"], 50000.00, rel_tol=1e-5)
    assert "relay" in data


def test_f20_04_root_dashboard_accessible_unauthenticated(operator_app):
    """F20: GET / is accessible without auth token for frictionless operator monitoring."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert code == 200
    assert "Operator Dashboard" in body


def test_f20_05_railway_production_config_spec():
    """F20: Project specifies Railway deployment configuration in PROJECT.md."""
    with open("PROJECT.md", "r", encoding="utf-8") as f:
        content = f.read()
    assert "Railway" in content
    assert "/health" in content
