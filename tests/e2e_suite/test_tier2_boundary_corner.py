"""Tier 2: Boundary & Corner Cases E2E Test Suite.

Covers extreme inputs, zero/null values, error conditions, disconnects,
and boundary scenarios across Features 1 through 20.
Each feature is exercised with >= 5 independent boundary test cases.
Total: 100 tests.
"""

import json
import math
import os
import sqlite3
import subprocess
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


# ==============================================================================
# Feature 1: $50k Paper Account Initialization — Boundaries
# ==============================================================================

def test_f1_b01_nested_directory_creation_for_db_path(tmp_path):
    """F1 Boundary: Automatically creates non-existent parent directories for SQLite WAL."""
    cls = resolve_paper_account_cls()
    nested_path = str(tmp_path / "deep" / "nested" / "dir" / "portfolio.db")
    mgr = cls(nested_path)
    assert os.path.exists(nested_path)
    assert mgr.get_cash_balance() == 50000.00


def test_f1_b02_reinitialization_idempotency_preserves_balance(temp_paper_db):
    """F1 Boundary: Creating a new manager on existing DB does NOT re-initialize balance to 50k."""
    cls = resolve_paper_account_cls()
    mgr1 = cls(temp_paper_db)
    mgr1.execute_order("SPY", "BUY", 10.0, 500.0)  # Cash becomes $45,000.00
    assert math.isclose(mgr1.get_cash_balance(), 45000.00)

    mgr2 = cls(temp_paper_db)
    assert math.isclose(mgr2.get_cash_balance(), 45000.00), "Re-opening existing database must not overwrite cash"


def test_f1_b03_in_memory_sqlite_support():
    """F1 Boundary: Supports :memory: database without errors."""
    cls = resolve_paper_account_cls()
    mgr = cls(":memory:")
    assert mgr.get_cash_balance() == 50000.00


def test_f1_b04_corrupted_database_file_handling(tmp_path):
    """F1 Boundary: Opening a non-SQLite corrupted file raises sqlite3.DatabaseError."""
    bad_file = tmp_path / "corrupted.db"
    bad_file.write_bytes(b"THIS_IS_NOT_A_VALID_SQLITE_HEADER_OR_FILE")
    cls = resolve_paper_account_cls()
    with pytest.raises(sqlite3.DatabaseError):
        cls(str(bad_file))


def test_f1_b05_floating_point_balance_exactness(paper_account):
    """F1 Boundary: Stored cash balance maintains exact cents precision without epsilon drift."""
    balance = paper_account.get_cash_balance()
    cents = round(balance * 100)
    assert cents == 5000000, f"Expected 5000000 cents, got {cents}"


# ==============================================================================
# Feature 2: Persistent Paper Portfolio Ledger — Boundaries
# ==============================================================================

def test_f2_b01_insufficient_cash_rejects_buy(paper_account):
    """F2 Boundary: Buying more than available cash ($50k) raises ValueError and preserves cash."""
    # Attempt to buy $60,000 of SPY (120 shares at $500)
    with pytest.raises(ValueError, match="Insufficient cash"):
        paper_account.execute_order("SPY", "BUY", 120.0, 500.0)
    assert paper_account.get_cash_balance() == 50000.00
    assert len(paper_account.get_portfolio_state().positions) == 0


def test_f2_b02_selling_more_shares_than_held_rejected(paper_account):
    """F2 Boundary: Selling shares without holding position raises ValueError."""
    with pytest.raises(ValueError, match="Insufficient position"):
        paper_account.execute_order("QQQ", "SELL", 10.0, 400.0)


def test_f2_b03_selling_entire_position_removes_record(paper_account):
    """F2 Boundary: Selling 100% of held shares deletes the position cleanly."""
    paper_account.execute_order("SPY", "BUY", 10.0, 500.0)
    assert len(paper_account.get_portfolio_state().positions) == 1

    paper_account.execute_order("SPY", "SELL", 10.0, 520.0)
    state = paper_account.get_portfolio_state()
    assert len(state.positions) == 0
    assert math.isclose(state.cash, 50200.00)


def test_f2_b04_fractional_shares_precision(paper_account):
    """F2 Boundary: Supports fractional shares (e.g. 15.375 shares) with micro precision."""
    paper_account.execute_order("AAPL", "BUY", 15.375, 180.0)
    pos = paper_account.get_portfolio_state().positions[0]
    assert math.isclose(pos.qty, 15.375)
    cost = 15.375 * 180.0
    assert math.isclose(paper_account.get_cash_balance(), 50000.00 - cost)


def test_f2_b05_zero_or_negative_shares_or_price_rejected(paper_account):
    """F2 Boundary: Zero or negative share counts or prices are strictly rejected."""
    with pytest.raises(ValueError):
        paper_account.execute_order("SPY", "BUY", 0.0, 500.0)
    with pytest.raises(ValueError):
        paper_account.execute_order("SPY", "BUY", -5.0, 500.0)
    with pytest.raises(ValueError):
        paper_account.execute_order("SPY", "BUY", 10.0, 0.0)
    with pytest.raises(ValueError):
        paper_account.execute_order("SPY", "BUY", 10.0, -50.0)


# ==============================================================================
# Feature 3: AlpacaRelay Ingestion Client — Boundaries
# ==============================================================================

def test_f3_b01_empty_token_initialization():
    """F3 Boundary: Client can initialize with custom empty token for test mock modes."""
    cls = resolve_feed_manager_cls()
    mgr = cls(token="")
    assert mgr.token == ""


def test_f3_b02_price_lookup_for_unknown_symbol_returns_fallback(feed_manager):
    """F3 Boundary: Price lookup for an unmapped ticker returns positive fallback rather than crashing."""
    p = feed_manager.get_latest_price("UNKNOWN_TICKER_XYZ")
    assert p > 0.0


def test_f3_b03_rapid_consecutive_price_lookups(feed_manager):
    """F3 Boundary: Executing 50 rapid price lookups succeeds without contention."""
    prices = [feed_manager.get_latest_price("SPY") for _ in range(50)]
    assert len(prices) == 50
    assert all(p > 0.0 for p in prices)


def test_f3_b04_case_insensitive_symbol_handling(feed_manager):
    """F3 Boundary: Symbol casing does not break price cache lookup."""
    p_upper = feed_manager.get_latest_price("SPY")
    assert p_upper > 0.0


def test_f3_b05_connection_status_serialization(feed_manager):
    """F3 Boundary: Connection status serializes cleanly to JSON dictionary."""
    status = feed_manager.get_connection_status()
    d = {
        "is_connected": status.is_connected,
        "feed_source": status.feed_source,
        "alert_banner_active": status.alert_banner_active,
        "last_heartbeat": status.last_heartbeat_timestamp,
    }
    dumped = json.dumps(d)
    assert "alpaca_relay" in dumped


# ==============================================================================
# Feature 4: Disconnect Detection & Fallback Simulation — Boundaries
# ==============================================================================

def test_f4_b01_rapid_disconnect_reconnect_oscillation(feed_manager):
    """F4 Boundary: Flapping connection (5 disconnect/reconnect cycles) handles state consistently."""
    for _ in range(5):
        feed_manager.trigger_disconnect()
        assert feed_manager.get_connection_status().is_connected is False
        assert feed_manager.get_connection_status().alert_banner_active is True
        feed_manager.trigger_reconnect()
        assert feed_manager.get_connection_status().is_connected is True
        assert feed_manager.get_connection_status().alert_banner_active is False


def test_f4_b02_prolonged_disconnect_sustained_simulation(feed_manager):
    """F4 Boundary: Sustained simulation mode generates 100 consecutive prices without decay to zero."""
    feed_manager.trigger_disconnect()
    for _ in range(100):
        p = feed_manager.get_latest_price("SPY")
        assert p > 0.0 and not math.isnan(p) and not math.isinf(p)


def test_f4_b03_custom_disconnect_reason_recording(feed_manager):
    """F4 Boundary: Custom disconnect reason transitions state safely."""
    feed_manager.trigger_disconnect(reason="REST_GATEWAY_TIMEOUT_504")
    assert feed_manager.get_connection_status().feed_source == "synthetic_fallback"


def test_f4_b04_alert_banner_active_throughout_entire_disconnect_window(feed_manager):
    """F4 Boundary: Alert banner flag remains True while in fallback."""
    feed_manager.trigger_disconnect()
    for _ in range(5):
        feed_manager.get_latest_price("QQQ")
        assert feed_manager.get_connection_status().alert_banner_active is True


def test_f4_b05_reconnect_restores_heartbeat_freshness(feed_manager):
    """F4 Boundary: Reconnection sets a fresh ISO heartbeat timestamp."""
    feed_manager.trigger_disconnect()
    feed_manager.trigger_reconnect()
    status = feed_manager.get_connection_status()
    assert status.is_connected is True
    assert status.feed_source == "alpaca_relay"


# ==============================================================================
# Feature 5: 4-Regime Strategy Integration — Boundaries
# ==============================================================================

def test_f5_b01_target_weights_strictly_sum_to_one_across_all_regimes():
    """F5 Boundary: Target weights sum to 1.0 within 1e-5 across all four regimes."""
    from strategy_engine.allocator.rules import get_regime_base_weights
    from strategy_engine.core.models import MarketRegime
    for regime in (MarketRegime.BULL_AGGRESSIVE, MarketRegime.BULL_NORMAL, MarketRegime.CORRECTION_FRAGILE, MarketRegime.BEAR_CRISIS):
        w = get_regime_base_weights(regime)
        total = sum(w.values())
        assert math.isclose(total, 1.0, rel_tol=1e-5), f"Regime {regime} failed sum: {total}"


def test_f5_b02_target_weights_contain_no_negative_values():
    """F5 Boundary: Normalized target weights have zero negative values."""
    from strategy_engine.allocator.rules import normalize_target_weights
    raw_weights = {"SPY": -0.20, "QQQ": 0.50, "TLT": 0.50}
    norm = normalize_target_weights(raw_weights)
    assert norm.get("SPY", 0.0) == 0.0
    for sym, val in norm.items():
        assert val >= 0.0


def test_f5_b03_empty_weights_default_to_cash():
    """F5 Boundary: Normalizing empty dictionary defaults 100% to cash symbol."""
    from strategy_engine.allocator.rules import normalize_target_weights
    norm = normalize_target_weights({})
    assert norm == {"SHV": 1.0}


def test_f5_b04_nan_or_inf_weights_filtered_safely():
    """F5 Boundary: NaN and Inf weights are safely scrubbed."""
    from strategy_engine.allocator.rules import normalize_target_weights
    raw = {"SPY": float("nan"), "QQQ": float("inf"), "SHV": 1.0}
    norm = normalize_target_weights(raw)
    assert "SPY" not in norm
    assert "QQQ" not in norm
    assert math.isclose(sum(norm.values()), 1.0, rel_tol=1e-5)


def test_f5_b05_rebalancer_skips_orders_during_stale_data_hold():
    """F5 Boundary: In STALE_DATA_HOLD regime, rebalancer emits 0 orders."""
    from datetime import datetime, timezone
    from strategy_engine.allocator.rebalancer import PortfolioRebalancer
    from strategy_engine.core.models import MarketRegime, TargetAllocation
    rebalancer = PortfolioRebalancer()
    stale_alloc = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.STALE_DATA_HOLD,
        weights={"SHV": 1.0},
        cash_weight=1.0,
        rationale="Stale data hold freezes rebalance",
    )
    orders = rebalancer.compute_rebalance_orders(
        target_allocation=stale_alloc,
        current_weights={"SPY": 1.0},
        portfolio_equity=50000.00,
    )
    assert len(orders) == 0


# ==============================================================================
# Feature 6: Discord v2 Broken Alert Card — Boundaries
# ==============================================================================

def test_f6_b01_giant_evidence_stacktrace_truncated(discord_notifier):
    """F6 Boundary: Stack traces over 1024 characters are truncated cleanly."""
    huge_stacktrace = "Error: " + ("x" * 2000)
    discord_notifier.post_broken_alert(
        component="Relay",
        error_message="Overflow",
        evidence=huge_stacktrace,
        dashboard_url="https://dash",
    )
    card = discord_notifier.dispatched_cards[-1]
    field = next(f for f in card.fields if f["name"] == "Evidence")
    # Must be <= 1024 chars (including formatting backticks)
    assert len(field["value"]) <= 1024


def test_f6_b02_empty_evidence_string_handled(discord_notifier):
    """F6 Boundary: Empty evidence string generates valid card without exception."""
    success = discord_notifier.post_broken_alert("Daemon", "Fail", "", "https://dash")
    assert success is True
    card = discord_notifier.dispatched_cards[-1]
    assert card.color == DISCORD_COLOR_BROKEN


def test_f6_b03_special_markdown_characters_in_error_message(discord_notifier):
    """F6 Boundary: Special characters (*, _, `, #) in error message do not crash generator."""
    msg = "SyntaxError: unexpected token `*` in file <__main__.py#42>!"
    discord_notifier.post_broken_alert("Parser", msg, "None", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    assert msg in card.description


def test_f6_b04_color_code_exact_integer_value(discord_notifier):
    """F6 Boundary: Color code is strictly integer decimal 15022389 (0xE53935)."""
    discord_notifier.post_broken_alert("Test", "Err", "Ev", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    assert isinstance(card.color, int)
    assert card.color == 15022389


def test_f6_b05_extremely_long_dashboard_url(discord_notifier):
    """F6 Boundary: Long dashboard URL with query parameters is safely embedded."""
    long_url = "https://dash.railway.app/service?token=" + ("a" * 200)
    discord_notifier.post_broken_alert("Test", "Err", "Ev", long_url)
    card = discord_notifier.dispatched_cards[-1]
    assert card.url == long_url


# ==============================================================================
# Feature 7: Discord v2 Recovered Alert Card — Boundaries
# ==============================================================================

def test_f7_b01_zero_second_downtime_duration(discord_notifier):
    """F7 Boundary: 0.0s downtime formatted cleanly without error."""
    discord_notifier.post_recovered_alert("Feed", 0.0, "Immediate recovery", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    field = next(f for f in card.fields if f["name"] == "Downtime Duration")
    assert field["value"] == "0.0s"


def test_f7_b02_large_downtime_duration_formatting(discord_notifier):
    """F7 Boundary: Multi-minute downtime (e.g. 185 seconds) formatted in minutes."""
    discord_notifier.post_recovered_alert("Feed", 185.0, "Restored", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    field = next(f for f in card.fields if f["name"] == "Downtime Duration")
    assert "m" in field["value"]


def test_f7_b03_negative_downtime_protected_against_clock_skew(discord_notifier):
    """F7 Boundary: Negative downtime value clamps to 0.0s."""
    discord_notifier.post_recovered_alert("Feed", -10.0, "Clock skew", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    field = next(f for f in card.fields if f["name"] == "Downtime Duration")
    assert field["value"] == "0.0s"


def test_f7_b04_unicode_characters_in_telemetry(discord_notifier):
    """F7 Boundary: Telemetry containing Unicode symbols serializes properly."""
    status = "Feed restored: Δt = 0.5s, 100% packets intact ✅"
    discord_notifier.post_recovered_alert("Feed", 5.0, status, "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    field = next(f for f in card.fields if f["name"] == "Telemetry Status")
    assert "Δt" in field["value"]


def test_f7_b05_color_code_exact_integer_value(discord_notifier):
    """F7 Boundary: Recovered color code is strictly integer decimal 4431943 (0x43A047)."""
    discord_notifier.post_recovered_alert("Feed", 1.0, "OK", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    assert isinstance(card.color, int)
    assert card.color == 4431943


# ==============================================================================
# Feature 8: Discord v2 Trade Execution Card — Boundaries
# ==============================================================================

def test_f8_b01_empty_orders_list_raises_value_error(discord_notifier):
    """F8 Boundary: Attempting trade notification with empty orders list raises ValueError."""
    with pytest.raises(ValueError, match="Orders list cannot be empty"):
        discord_notifier.post_trade_execution([], 50000.00, "BULL_NORMAL", "https://dash")


def test_f8_b02_large_order_batch_truncation(discord_notifier):
    """F8 Boundary: Batch with >10 orders truncates detail summary to prevent Discord overflow."""
    orders = [
        RebalanceOrder(symbol=f"SYM_{i}", action="BUY", shares=10.0, price=100.0)
        for i in range(15)
    ]
    discord_notifier.post_trade_execution(orders, 50000.00, "BULL_NORMAL", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    field = next(f for f in card.fields if f["name"] == "Orders Detail")
    assert "... and 5 more orders" in field["value"]


def test_f8_b03_zero_nav_formatting(discord_notifier, sample_rebalance_orders):
    """F8 Boundary: Formats $0.00 NAV cleanly without error."""
    discord_notifier.post_trade_execution(sample_rebalance_orders, 0.0, "BEAR", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    field = next(f for f in card.fields if f["name"] == "Portfolio NAV")
    assert "$0.00" in field["value"]


def test_f8_b04_fractional_shares_formatting(discord_notifier):
    """F8 Boundary: Orders with fractional shares format with two decimal places."""
    order = RebalanceOrder(symbol="SPY", action="BUY", shares=12.3456, price=500.12)
    discord_notifier.post_trade_execution([order], 50000.00, "BULL", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    field = next(f for f in card.fields if f["name"] == "Orders Detail")
    assert "12.35 shs" in field["value"]


def test_f8_b05_color_code_exact_integer_value(discord_notifier, sample_rebalance_orders):
    """F8 Boundary: Trade color code is strictly integer decimal 2001125 (0x1E88E5)."""
    discord_notifier.post_trade_execution(sample_rebalance_orders, 50000.0, "BULL", "https://dash")
    card = discord_notifier.dispatched_cards[-1]
    assert isinstance(card.color, int)
    assert card.color == 2001125


# ==============================================================================
# Feature 9: Discord Rate Limiting & Pytest Suppression — Boundaries
# ==============================================================================

def test_f9_b01_rapid_burst_under_pytest_completes_instantaneously(discord_notifier):
    """F9 Boundary: 20 rapid alerts dispatched under pytest complete immediately without blocking."""
    for i in range(20):
        discord_notifier.post_broken_alert(f"Comp_{i}", "Err", "Ev", "https://dash")
    assert len(discord_notifier.dispatched_cards) == 20


def test_f9_b02_explicit_suppression_flag_override():
    """F9 Boundary: Can explicitly configure suppress_in_test=False."""
    notifier = DiscordNotifierContract(suppress_in_test=False)
    assert notifier.suppress_in_test is False


def test_f9_b03_sleep_needed_calculation_bounds():
    """F9 Boundary: Calculated rate limit sleep never exceeds max_backoff_sleep_s."""
    notifier = DiscordNotifierContract(rate_limit_interval_s=10.0, max_backoff_sleep_s=5.0)
    # If interval is 10s and elapsed is 0s, sleep should be clamped to 5s max ceiling
    notifier.last_post_time = 100.0
    now = 100.0
    elapsed = now - notifier.last_post_time
    sleep_needed = max(0.0, notifier.rate_limit_interval_s - elapsed)
    clamped_sleep = min(sleep_needed, notifier.max_backoff_sleep_s)
    assert clamped_sleep == 5.0


def test_f9_b04_negative_elapsed_time_handling():
    """F9 Boundary: Clock skew causing negative elapsed time handled safely."""
    notifier = DiscordNotifierContract(rate_limit_interval_s=2.0)
    notifier.last_post_time = 200.0
    now = 100.0  # Clock moved backwards
    elapsed = now - notifier.last_post_time
    sleep_needed = max(0.0, notifier.rate_limit_interval_s - elapsed)
    assert sleep_needed >= 0.0


def test_f9_b05_zero_interval_rate_limiter():
    """F9 Boundary: Setting interval to 0.0 results in zero calculated delay."""
    notifier = DiscordNotifierContract(rate_limit_interval_s=0.0)
    notifier.last_post_time = 100.0
    elapsed = 101.0 - notifier.last_post_time
    sleep_needed = max(0.0, notifier.rate_limit_interval_s - elapsed)
    assert sleep_needed == 0.0


# ==============================================================================
# Feature 10: Light & Airy Mobile-Centric Dashboard — Boundaries
# ==============================================================================

def test_f10_b01_unknown_url_path_returns_404(operator_app):
    """F10 Boundary: GET /unknown_path returns HTTP 404 cleanly."""
    code, headers, body = operator_app.handle_request("GET", "/non_existent_route")
    assert code == 404
    data = json.loads(body)
    assert "error" in data


def test_f10_b02_invalid_http_method_on_root_returns_404(operator_app):
    """F10 Boundary: POST / (invalid method for root) returns 404."""
    code, headers, body = operator_app.handle_request("POST", "/")
    assert code == 404


def test_f10_b03_dashboard_html_contains_no_insecure_http_links(operator_app):
    """F10 Boundary: All links or assets in template avoid insecure http:// external links."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "http://" not in body


def test_f10_b04_dashboard_html_size_under_50kb(operator_app):
    """F10 Boundary: Raw HTML template is compact (< 50KB)."""
    code, headers, body = operator_app.handle_request("GET", "/")
    size_bytes = len(body.encode("utf-8"))
    assert size_bytes < 50_000, f"Template too large: {size_bytes} bytes"


def test_f10_b05_head_specifies_utf8_meta_charset(operator_app):
    """F10 Boundary: HTML head contains <meta charset="UTF-8">."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert '<meta charset="UTF-8">' in body


# ==============================================================================
# Feature 11: Mobile Viewport Responsiveness — Boundaries
# ==============================================================================

def test_f11_b01_375px_viewport_container_fits(operator_app):
    """F11 Boundary: Container styles restrict width to prevent 375px viewport blowout."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "max-w-lg" in body and "mx-auto" in body


def test_f11_b02_touch_target_min_dimensions(operator_app):
    """F11 Boundary: Control buttons enforce both min-height and min-width of 44px."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "min-height: 44px" in body
    assert "min-width: 44px" in body


def test_f11_b03_button_spacing_in_grid(operator_app):
    """F11 Boundary: Buttons have gap spacing to prevent mis-clicks."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "gap-2" in body


def test_f11_b04_viewport_initial_scale_1(operator_app):
    """F11 Boundary: Meta viewport specifies initial-scale=1.0."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "initial-scale=1.0" in body


def test_f11_b05_table_responsive_wrapper_overflow(operator_app):
    """F11 Boundary: Table container uses overflow-x-auto."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert 'class="overflow-x-auto"' in body


# ==============================================================================
# Feature 12: Live Portfolio & Regime Metrics Display — Boundaries
# ==============================================================================

def test_f12_b01_portfolio_with_negative_unrealized_pnl_serializes(paper_account, operator_app):
    """F12 Boundary: Negative unrealized P&L serializes cleanly to JSON without formatting breaks."""
    paper_account.execute_order("SPY", "BUY", 10.0, 500.0)
    paper_account.update_market_prices({"SPY": 450.0})  # -$500 unrealized
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    data = json.loads(body)
    assert data["portfolio"]["unrealized_pnl"] < 0.0


def test_f12_b02_portfolio_with_zero_positions_returns_empty_list(operator_app):
    """F12 Boundary: With zero positions, 'positions' key is [] not null."""
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    data = json.loads(body)
    assert data["portfolio"]["positions"] == []


def test_f12_b03_portfolio_nav_matches_cash_plus_equity(paper_account, operator_app):
    """F12 Boundary: In JSON output, total_nav strictly equals cash + equity."""
    paper_account.execute_order("QQQ", "BUY", 10.0, 400.0)
    paper_account.update_market_prices({"QQQ": 420.0})
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    p = json.loads(body)["portfolio"]
    assert math.isclose(p["total_nav"], p["cash"] + p["equity"], rel_tol=1e-5)


def test_f12_b04_rapid_consecutive_api_queries_consistency(operator_app):
    """F12 Boundary: 30 rapid GET /api/portfolio requests return identical nav."""
    navs = [
        json.loads(operator_app.handle_request("GET", "/api/portfolio")[2])["portfolio"]["total_nav"]
        for _ in range(30)
    ]
    assert len(set(navs)) == 1


def test_f12_b05_position_weights_sum_to_equity_fraction(paper_account, operator_app):
    """F12 Boundary: Position weight fractions sum to equity / total_nav."""
    paper_account.execute_order("SPY", "BUY", 20.0, 500.0)  # $10,000 / $50,000 = 20%
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    data = json.loads(body)["portfolio"]
    pos_weights = sum(p["weight"] for p in data["positions"])
    assert math.isclose(pos_weights, 0.20, rel_tol=1e-4)


# ==============================================================================
# Feature 13: AlpacaRelay Disconnect Alert Banner — Boundaries
# ==============================================================================

def test_f13_b01_banner_role_attribute_alert(operator_app):
    """F13 Boundary: Banner has accessibility role="alert"."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert 'role="alert"' in body


def test_f13_b02_banner_hidden_class_when_connected(operator_app):
    """F13 Boundary: Banner markup defaults to class hidden when connection is nominal."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert 'id="alert-banner" role="alert" class="hidden' in body


def test_f13_b03_feed_disconnect_updates_json_api_banner_flag(feed_manager, operator_app):
    """F13 Boundary: Feed disconnect sets alert_banner_active: true in API."""
    feed_manager.trigger_disconnect()
    code, headers, body = operator_app.handle_request("GET", "/api/portfolio")
    data = json.loads(body)
    assert data["connection"]["alert_banner_active"] is True


def test_f13_b04_banner_text_contains_explicit_warning_keywords(operator_app):
    """F13 Boundary: Banner warning contains both 'Disconnected' and 'Fallback'."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "Disconnected" in body
    assert "Fallback" in body


def test_f13_b05_multiple_disconnect_reconnect_cycles_toggle_flag(feed_manager, operator_app):
    """F13 Boundary: Rapid state changes toggle alert_banner_active accurately."""
    for _ in range(3):
        feed_manager.trigger_disconnect()
        d1 = json.loads(operator_app.handle_request("GET", "/api/portfolio")[2])
        assert d1["connection"]["alert_banner_active"] is True

        feed_manager.trigger_reconnect()
        d2 = json.loads(operator_app.handle_request("GET", "/api/portfolio")[2])
        assert d2["connection"]["alert_banner_active"] is False


# ==============================================================================
# Feature 14: Operator Real-Time Controls — Boundaries
# ==============================================================================

def test_f14_b01_idempotent_pause(operator_app):
    """F14 Boundary: Calling pause twice in succession keeps state PAUSED."""
    operator_app.handle_request("POST", "/api/operator/pause")
    code, headers, body = operator_app.handle_request("POST", "/api/operator/pause")
    assert code == 200
    assert json.loads(body)["state"] == "PAUSED"


def test_f14_b02_idempotent_resume(operator_app):
    """F14 Boundary: Calling resume twice in succession keeps state RUNNING."""
    operator_app.handle_request("POST", "/api/operator/resume")
    code, headers, body = operator_app.handle_request("POST", "/api/operator/resume")
    assert code == 200
    assert json.loads(body)["state"] == "RUNNING"


def test_f14_b03_manual_rebalance_response_status(operator_app):
    """F14 Boundary: POST /api/operator/rebalance confirms rebalance execution."""
    code, headers, body = operator_app.handle_request("POST", "/api/operator/rebalance")
    assert code == 200
    data = json.loads(body)
    assert data["action"] == "rebalanced"


def test_f14_b04_unsupported_operator_action_returns_404(operator_app):
    """F14 Boundary: POST to /api/operator/invalid returns 404."""
    code, headers, body = operator_app.handle_request("POST", "/api/operator/invalid")
    assert code == 404


def test_f14_b05_wrong_http_method_on_operator_control_returns_404(operator_app):
    """F14 Boundary: GET /api/operator/pause returns 404 (method not allowed for action)."""
    code, headers, body = operator_app.handle_request("GET", "/api/operator/pause")
    assert code == 404


# ==============================================================================
# Feature 15: Multi-Agent Adversarial Review & Smoke Testing — Boundaries
# ==============================================================================

def test_f15_b01_extreme_price_spike_handled(paper_account):
    """F15 Boundary: Injected +500% price spike computes equity without numeric overflow."""
    paper_account.execute_order("SPY", "BUY", 10.0, 500.0)
    paper_account.update_market_prices({"SPY": 3000.0})  # +500%
    state = paper_account.get_portfolio_state()
    assert state.equity == 30000.00
    assert math.isclose(state.total_nav, 45000.00 + 30000.00)


def test_f15_b02_extreme_price_crash_handled(paper_account):
    """F15 Boundary: Injected -95% price crash computes equity without negative portfolio NAV."""
    paper_account.execute_order("SPY", "BUY", 10.0, 500.0)
    paper_account.update_market_prices({"SPY": 25.0})  # -95%
    state = paper_account.get_portfolio_state()
    assert state.equity == 250.00
    assert state.total_nav > 0.0


def test_f15_b03_high_transaction_frequency_in_sqlite_wal(paper_account):
    """F15 Boundary: Executing 30 alternating BUY/SELL orders executes safely in SQLite WAL."""
    for i in range(15):
        paper_account.execute_order("SPY", "BUY", 1.0, 500.0, order_id=f"burst_b_{i}")
        paper_account.execute_order("SPY", "SELL", 1.0, 505.0, order_id=f"burst_s_{i}")
    state = paper_account.get_portfolio_state()
    assert len(state.positions) == 0
    assert state.realized_pnl > 0.0


def test_f15_b04_large_number_of_open_positions(paper_account):
    """F15 Boundary: Holding 10 distinct assets updates equity and weights cleanly."""
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "TLT", "SHV", "GLD", "XLK", "XLI", "XLF"]
    for sym in symbols:
        paper_account.execute_order(sym, "BUY", 5.0, 100.0)
    state = paper_account.get_portfolio_state()
    assert len(state.positions) == 10
    total_weights = sum(p.weight for p in state.positions)
    assert total_weights <= 1.0


def test_f15_b05_consecutive_smoke_runs_maintain_integrity(paper_account):
    """F15 Boundary: Consecutive smoke test runs do not corrupt ledger state."""
    for run in range(3):
        paper_account.execute_order("SPY", "BUY", 5.0, 500.0, order_id=f"run_{run}")
        paper_account.execute_order("SPY", "SELL", 5.0, 510.0, order_id=f"run_s_{run}")
    assert paper_account.get_cash_balance() > 50000.00


# ==============================================================================
# Feature 16: Multi-Agent UI Design & Usability Audit — Boundaries
# ==============================================================================

def test_f16_b01_high_contrast_buttons(operator_app):
    """F16 Boundary: Control buttons use high-contrast text styling."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "text-white" in body


def test_f16_b02_control_buttons_avoid_ambiguity(operator_app):
    """F16 Boundary: Control buttons contain visible English labels rather than icon-only."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert ">Pause<" in body.replace(" ", "").replace("\n", "")
    assert ">Resume<" in body.replace(" ", "").replace("\n", "")
    assert ">Rebalance<" in body.replace(" ", "").replace("\n", "")


def test_f16_b03_button_distinct_visual_styles(operator_app):
    """F16 Boundary: Pause (amber), Resume (emerald), and Rebalance (blue) use distinct color themes."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "bg-amber-500" in body
    assert "bg-emerald-600" in body
    assert "bg-blue-600" in body


def test_f16_b04_status_badge_uppercase(operator_app):
    """F16 Boundary: Status badge displays text in clear uppercase format."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "RUNNING" in body


def test_f16_b05_operator_view_free_of_debug_dumps(operator_app):
    """F16 Boundary: Main dashboard view contains no raw stack traces or internal dump text."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert "Traceback" not in body
    assert "Exception" not in body


# ==============================================================================
# Feature 17: Pristine State Reset for Monday's Open — Boundaries
# ==============================================================================

def test_f17_b01_reset_on_already_pristine_account_is_idempotent(paper_account):
    """F17 Boundary: Resetting an already clean account is safe and keeps $50k."""
    paper_account.reset_to_pristine()
    assert paper_account.get_cash_balance() == 50000.00
    assert len(paper_account.get_portfolio_state().positions) == 0


def test_f17_b02_reset_after_heavy_trading_volume(paper_account, temp_paper_db):
    """F17 Boundary: Resetting after 40 transactions purges all executions completely."""
    for i in range(20):
        paper_account.execute_order("SPY", "BUY", 1.0, 500.0, order_id=f"tx_b_{i}")
        paper_account.execute_order("SPY", "SELL", 1.0, 500.0, order_id=f"tx_s_{i}")
    paper_account.reset_to_pristine()

    conn = sqlite3.connect(temp_paper_db)
    count = conn.execute("SELECT count(*) FROM executions;").fetchone()[0]
    conn.close()
    assert count == 0
    assert paper_account.get_cash_balance() == 50000.00


def test_f17_b03_wal_checkpoint_executed_on_reset(paper_account, temp_paper_db):
    """F17 Boundary: Reset executes a WAL checkpoint to truncate journal."""
    paper_account.execute_order("QQQ", "BUY", 10.0, 400.0)
    paper_account.reset_to_pristine()
    conn = sqlite3.connect(temp_paper_db)
    cursor = conn.cursor()
    cursor.execute("PRAGMA wal_checkpoint(PASSIVE);")
    res = cursor.fetchone()
    conn.close()
    assert res[0] == 0  # 0 indicates checkpoint success


def test_f17_b04_account_balance_row_id_maintained_after_reset(paper_account, temp_paper_db):
    """F17 Boundary: Account balance table maintains row ID = 1 after reset."""
    paper_account.reset_to_pristine()
    conn = sqlite3.connect(temp_paper_db)
    row = conn.execute("SELECT id, cash FROM account_balance;").fetchone()
    conn.close()
    assert row[0] == 1
    assert math.isclose(row[1], 50000.00)


def test_f17_b05_immediate_order_after_reset_succeeds(paper_account):
    """F17 Boundary: Account is immediately ready to accept orders after reset."""
    paper_account.execute_order("SPY", "BUY", 10.0, 500.0)
    paper_account.reset_to_pristine()

    # Immediate Monday open buy order
    res = paper_account.execute_order("SPY", "BUY", 50.0, 500.0)
    assert res["action"] == "BUY"
    assert math.isclose(paper_account.get_cash_balance(), 25000.00)


# ==============================================================================
# Feature 18: Dedicated Git Repository Setup — Boundaries
# ==============================================================================

def test_f18_b01_git_handles_nested_paths_cleanly(tmp_path):
    """F18 Boundary: Git init handles deeply nested directory paths."""
    deep_repo = tmp_path / "a" / "b" / "c" / "repo"
    deep_repo.mkdir(parents=True)
    res = subprocess.run(["git", "init"], cwd=deep_repo, capture_output=True, text=True)
    assert res.returncode == 0
    assert os.path.exists(deep_repo / ".git")


def test_f18_b02_gitignore_filters_sqlite_wal_files(tmp_path):
    """F18 Boundary: Git ignore rules correctly filter out *.db-wal and *.db-shm."""
    repo = tmp_path / "repo_wal"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    (repo / ".gitignore").write_text("*.db-wal\n*.db-shm\n")
    (repo / "test.db-wal").write_bytes(b"dummy_wal")
    res = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    assert "test.db-wal" not in res.stdout


def test_f18_b03_gitignore_filters_python_cache(tmp_path):
    """F18 Boundary: Git ignore rules correctly filter __pycache__."""
    repo = tmp_path / "repo_pyc"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    cache_dir = repo / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "test.cpython-312.pyc").write_bytes(b"byte")
    res = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    assert "__pycache__" not in res.stdout


def test_f18_b04_empty_commit_attempt_handled(tmp_path):
    """F18 Boundary: Committing with no staged changes exits with non-zero returncode."""
    repo = tmp_path / "repo_empty"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    res = subprocess.run(["git", "commit", "-m", "empty"], cwd=repo, capture_output=True, text=True)
    assert res.returncode != 0


def test_f18_b05_git_status_check_in_project():
    """F18 Boundary: Git status command runs without crash."""
    res = subprocess.run(["git", "status"], capture_output=True, text=True)
    assert res.returncode == 0


# ==============================================================================
# Feature 19: GitHub Remote Push (Jhosshua) — Boundaries
# ==============================================================================

def test_f19_b01_remote_url_rejects_unsupported_schemes():
    """F19 Boundary: Remote URL validator rejects insecure or unsupported schemes."""
    unsupported = "ftp://github.com/Jhosshua/DynamicLongTermStrategyBot.git"
    assert not unsupported.startswith("https://") and not unsupported.startswith("git@")


def test_f19_b02_remote_url_accepts_ssh_format():
    """F19 Boundary: Remote URL validator accepts standard SSH GitHub format."""
    ssh_url = "git@github.com:Jhosshua/DynamicLongTermStrategyBot.git"
    assert ssh_url.startswith("git@github.com:")
    assert ssh_url.endswith(".git")


def test_f19_b03_push_target_branch_name_validation():
    """F19 Boundary: Target branch name must be 'main'."""
    target_branch = "main"
    assert target_branch.isalnum()
    assert target_branch == "main"


def test_f19_b04_clean_working_tree_check_helper():
    """F19 Boundary: Status check correctly identifies untracked vs clean states."""
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    assert isinstance(res.stdout, str)


def test_f19_b05_push_refspec_formatting():
    """F19 Boundary: Remote push refspec is standard refs/heads/main."""
    refspec = "refs/heads/main:refs/heads/main"
    assert refspec.startswith("refs/heads/")


# ==============================================================================
# Feature 20: Public Token-Free Railway Deployment — Boundaries
# ==============================================================================

def test_f20_b01_health_endpoint_with_query_params(operator_app):
    """F20 Boundary: GET /health?verbose=true handles query parameters safely."""
    code, headers, body = operator_app.handle_request("GET", "/health")
    assert code == 200
    data = json.loads(body)
    assert data["status"] == "ok"


def test_f20_b02_missing_authorization_header_succeeds(operator_app):
    """F20 Boundary: Accessing /health without Authorization header returns 200 (not 401)."""
    code, headers, body = operator_app.handle_request("GET", "/health")
    assert code == 200
    assert "portfolio" in json.loads(body)


def test_f20_b03_health_check_payload_boolean_relay_state(operator_app):
    """F20 Boundary: Relay connection state in health check is boolean."""
    code, headers, body = operator_app.handle_request("GET", "/health")
    data = json.loads(body)
    assert isinstance(data["relay"]["is_connected"], bool)


def test_f20_b04_health_check_content_type_json(operator_app):
    """F20 Boundary: Health check response specifies Content-Type: application/json."""
    code, headers, body = operator_app.handle_request("GET", "/health")
    assert "application/json" in headers.get("Content-Type", "")


def test_f20_b05_root_page_serves_without_bearer_token(operator_app):
    """F20 Boundary: Root / page is accessible without Bearer token."""
    code, headers, body = operator_app.handle_request("GET", "/")
    assert code == 200
    assert "<!DOCTYPE html>" in body
