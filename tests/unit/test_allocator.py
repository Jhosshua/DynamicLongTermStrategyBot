"""
tests.unit.test_allocator
~~~~~~~~~~~~~~~~~~~~~~~~~

Unit test suite for portfolio allocation rules and rebalancing engine:
- Strict sum-to-one normalization within 1e-5
- Long-only non-negative weight invariant
- 4-regime allocation matrices
- Antonacci safe-haven dual momentum routing
- Drift band filtering (+/- 2.5%)
- Micro-order suppression (< 0.5%)
- Order sequencing: SELLs executed before BUYs
- STALE_DATA_HOLD trading freeze
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import pytest

from strategy_engine.core.models import (
    Bar,
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderType,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.allocator.rules import (
    compute_deterministic_allocation,
    get_regime_base_weights,
    normalize_target_weights,
)
from strategy_engine.allocator.rebalancer import (
    DRIFT_TOLERANCE,
    MIN_ORDER_THRESHOLD,
    PortfolioRebalancer,
    generate_rebalance_orders,
)


def _create_bar(symbol: str, dt: datetime, close: float) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=dt,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=100_000,
    )


def _make_snapshot(
    regime: MarketRegime,
    vol_scale: float = 1.0,
    dd_gate: float = 1.0,
    circuit_breaker: bool = False,
    timestamp: Optional[datetime] = None,
) -> SignalSnapshot:
    return SignalSnapshot(
        timestamp=timestamp or datetime(2026, 12, 31, tzinfo=timezone.utc),
        spy_price=500.0,
        spy_sma50=490.0,
        spy_sma200=470.0,
        realized_vol_20d=0.10,
        vol_scale_factor=vol_scale,
        drawdown_pct=0.0,
        circuit_breaker_active=circuit_breaker,
        regime=regime,
        indicators={"drawdown_gate": dd_gate},
    )


def test_normalize_target_weights_sum_exact():
    """Verify normalize_target_weights produces weights strictly summing to 1.0 within 1e-5."""
    raw = {"SPY": 0.333333333, "QQQ": 0.333333333, "SHV": 0.333333333}
    norm = normalize_target_weights(raw, cash_symbol="SHV")
    assert math.isclose(sum(norm.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5)
    assert all(w >= 0.0 for w in norm.values())


def test_normalize_target_weights_empty_defaults_to_cash():
    """Verify empty or near-zero dictionary defaults to 100% cash."""
    assert normalize_target_weights({}, cash_symbol="SHV") == {"SHV": 1.0}
    assert normalize_target_weights({"SPY": 1e-8}, cash_symbol="SHV") == {"SHV": 1.0}


def test_regime_allocation_bull_aggressive():
    """Verify BULL_AGGRESSIVE nominal allocation: 50% QQQ, 30% Sectors (15% each), 20% SPY, 0% Cash."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market_data = {
        "SPY": [_create_bar("SPY", t0 + timedelta(days=i), 500.0 + i) for i in range(260)],
        "QQQ": [_create_bar("QQQ", t0 + timedelta(days=i), 450.0 + i) for i in range(260)],
        "XLK": [_create_bar("XLK", t0 + timedelta(days=i), 200.0 + i * 0.8) for i in range(260)],
        "XLY": [_create_bar("XLY", t0 + timedelta(days=i), 180.0 + i * 0.7) for i in range(260)],
        "SHV": [_create_bar("SHV", t0 + timedelta(days=i), 100.0 + i * 0.01) for i in range(260)],
    }
    # Add other sectors
    for s in ["XLC", "XLI", "XLF", "XLV", "XLP", "XLU", "XLE", "XLB", "XLRE"]:
        market_data[s] = [_create_bar(s, t0 + timedelta(days=i), 100.0 + i * 0.1) for i in range(260)]

    snapshot = _make_snapshot(MarketRegime.BULL_AGGRESSIVE, vol_scale=1.0, dd_gate=1.0)
    alloc = compute_deterministic_allocation(snapshot, market_data)

    assert isinstance(alloc, TargetAllocation)
    assert alloc.regime == MarketRegime.BULL_AGGRESSIVE
    assert math.isclose(sum(alloc.weights.values()), 1.0, rel_tol=1e-5)
    assert math.isclose(alloc.weights["QQQ"], 0.50, rel_tol=1e-3)
    assert math.isclose(alloc.weights["SPY"], 0.20, rel_tol=1e-3)
    # 2 sectors at 15% each
    sector_weights = [w for s, w in alloc.weights.items() if s not in ("QQQ", "SPY", "SHV", "BIL", "CASH", "TLT", "GLD")]
    assert len(sector_weights) == 2
    for sw in sector_weights:
        assert math.isclose(sw, 0.15, rel_tol=1e-3)


def test_regime_allocation_bull_normal():
    """Verify BULL_NORMAL nominal allocation: 35% QQQ, 25% SPY, 20% Sector, 20% Defensive."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market_data = {
        "SPY": [_create_bar("SPY", t0 + timedelta(days=i), 500.0 + i) for i in range(260)],
        "QQQ": [_create_bar("QQQ", t0 + timedelta(days=i), 450.0 + i) for i in range(260)],
        "XLK": [_create_bar("XLK", t0 + timedelta(days=i), 200.0 + i * 0.8) for i in range(260)],
        "TLT": [_create_bar("TLT", t0 + timedelta(days=i), 100.0 + i * 0.2) for i in range(260)],
        "GLD": [_create_bar("GLD", t0 + timedelta(days=i), 100.0 + i * 0.2) for i in range(260)],
        "SHV": [_create_bar("SHV", t0 + timedelta(days=i), 100.0 + i * 0.01) for i in range(260)],
    }
    for s in ["XLC", "XLY", "XLI", "XLF", "XLV", "XLP", "XLU", "XLE", "XLB", "XLRE"]:
        market_data[s] = [_create_bar(s, t0 + timedelta(days=i), 100.0 + i * 0.1) for i in range(260)]

    snapshot = _make_snapshot(MarketRegime.BULL_NORMAL, vol_scale=1.0, dd_gate=1.0)
    alloc = compute_deterministic_allocation(snapshot, market_data)

    assert math.isclose(alloc.weights["QQQ"], 0.35, rel_tol=1e-3)
    assert math.isclose(alloc.weights["SPY"], 0.25, rel_tol=1e-3)
    assert math.isclose(sum(alloc.weights.values()), 1.0, rel_tol=1e-5)


def test_regime_allocation_bear_crisis_100_percent_defensive():
    """Verify BEAR_CRISIS: 0% equities, 100% safe havens / cash."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market_data = {
        "SPY": [_create_bar("SPY", t0 + timedelta(days=i), 500.0 - i) for i in range(260)],
        "TLT": [_create_bar("TLT", t0 + timedelta(days=i), 100.0 + i * 0.2) for i in range(260)],
        "GLD": [_create_bar("GLD", t0 + timedelta(days=i), 100.0 + i * 0.2) for i in range(260)],
        "SHV": [_create_bar("SHV", t0 + timedelta(days=i), 100.0 + i * 0.01) for i in range(260)],
    }

    snapshot = _make_snapshot(MarketRegime.BEAR_CRISIS)
    alloc = compute_deterministic_allocation(snapshot, market_data)

    assert alloc.weights.get("SPY", 0.0) == 0.0
    assert alloc.weights.get("QQQ", 0.0) == 0.0
    assert math.isclose(sum(alloc.weights.values()), 1.0, rel_tol=1e-5)
    # TLT and GLD qualify -> 40% TLT, 40% GLD, 20% SHV
    assert math.isclose(alloc.weights["TLT"], 0.40, abs_tol=1e-4)
    assert math.isclose(alloc.weights["GLD"], 0.40, abs_tol=1e-4)
    assert math.isclose(alloc.weights["SHV"], 0.20, abs_tol=1e-4)


def test_stale_data_hold_freezes_allocations():
    """Verify STALE_DATA_HOLD preserves capital in cash with 0 new orders."""
    snapshot = _make_snapshot(MarketRegime.STALE_DATA_HOLD)
    alloc = compute_deterministic_allocation(snapshot, {})
    assert alloc.weights == {"SHV": 1.0}
    assert alloc.cash_weight == 1.0

    # Rebalancer emits empty list
    rebalancer = PortfolioRebalancer()
    orders = rebalancer.compute_rebalance_orders(alloc, current_weights={"SPY": 0.5, "SHV": 0.5})
    assert orders == []


def test_rebalancer_drift_band_filtering():
    """Verify deviations within +/- 2.5% are filtered, deviations > 2.5% trigger orders."""
    rebalancer = PortfolioRebalancer(drift_band=0.025)
    target = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 0.35, "QQQ": 0.35, "SHV": 0.30},
        cash_weight=0.30,
        rationale="Test",
    )

    # Within drift band: current SPY=0.36 (drift +0.01), QQQ=0.34 (drift -0.01)
    current_small_drift = {"SPY": 0.36, "QQQ": 0.34, "SHV": 0.30}
    orders_none = rebalancer.compute_rebalance_orders(target, current_small_drift)
    assert len(orders_none) == 0

    # Exceeding drift band: current SPY=0.50 (drift -0.15), QQQ=0.20 (drift +0.15)
    current_large_drift = {"SPY": 0.50, "QQQ": 0.20, "SHV": 0.30}
    orders = rebalancer.compute_rebalance_orders(target, current_large_drift)
    assert len(orders) == 2


def test_rebalancer_micro_order_suppression():
    """Verify trades under 0.5% (0.005) are suppressed."""
    rebalancer = PortfolioRebalancer(drift_band=0.001, min_order_threshold=0.005)
    target = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 0.50, "SHV": 0.50},
        cash_weight=0.50,
        rationale="Test",
    )
    # Drift is 0.003 (exceeds drift_band 0.001, but below min_order_threshold 0.005)
    current = {"SPY": 0.503, "SHV": 0.497}
    orders = rebalancer.compute_rebalance_orders(target, current)
    assert len(orders) == 0


def test_rebalancer_order_sequencing_sells_before_buys():
    """Verify all SELL orders are strictly sequenced before BUY orders."""
    rebalancer = PortfolioRebalancer(drift_band=0.025)
    target = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 0.20, "QQQ": 0.50, "SHV": 0.30},
        cash_weight=0.30,
        rationale="Test",
    )
    # Current SPY=0.40 (need to SELL 0.20), QQQ=0.30 (need to BUY 0.20)
    current = {"SPY": 0.40, "QQQ": 0.30, "SHV": 0.30}
    orders = rebalancer.compute_rebalance_orders(target, current)

    assert len(orders) == 2
    assert orders[0].action == "SELL"
    assert orders[0].symbol == "SPY"
    assert orders[1].action == "BUY"
    assert orders[1].symbol == "QQQ"


def test_rebalancer_share_and_dollar_math():
    """Verify dollar and share amounts when portfolio equity and prices provided."""
    target = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 0.50, "SHV": 0.50},
        cash_weight=0.50,
        rationale="Test",
    )
    current = {"SPY": 0.20, "SHV": 0.80}
    orders = generate_rebalance_orders(
        target=target,
        current_weights=current,
        portfolio_value=100_000.0,
        current_prices={"SPY": 500.0, "SHV": 100.0},
    )

    # SELL SHV first, then BUY SPY
    assert len(orders) == 2
    sell_shv = orders[0]
    buy_spy = orders[1]

    assert sell_shv.symbol == "SHV"
    assert sell_shv.action == "SELL"
    assert math.isclose(sell_shv.delta_dollars, -30_000.0)
    assert math.isclose(sell_shv.delta_shares, -300.0)

    assert buy_spy.symbol == "SPY"
    assert buy_spy.action == "BUY"
    assert math.isclose(buy_spy.delta_dollars, 30_000.0)
    assert math.isclose(buy_spy.delta_shares, 60.0)
    assert math.isclose(buy_spy.target_shares, 100.0)


def test_get_regime_base_weights_all_regimes_exact():
    """Verify get_regime_base_weights produces exact STRATEGY.md nominal weights across all regimes."""
    # 1. BULL_AGGRESSIVE: 50% QQQ, 20% SPY, 15% XLK, 15% XLY, 0% Cash
    w_agg = get_regime_base_weights(MarketRegime.BULL_AGGRESSIVE)
    assert math.isclose(w_agg["QQQ"], 0.50, abs_tol=1e-4)
    assert math.isclose(w_agg["SPY"], 0.20, abs_tol=1e-4)
    assert math.isclose(w_agg["XLK"], 0.15, abs_tol=1e-4)
    assert math.isclose(w_agg["XLY"], 0.15, abs_tol=1e-4)
    assert math.isclose(sum(w_agg.values()), 1.0, abs_tol=1e-6)

    # 2. BULL_NORMAL: 35% QQQ, 25% SPY, 20% XLK, 20% SHV
    w_norm = get_regime_base_weights(MarketRegime.BULL_NORMAL)
    assert math.isclose(w_norm["QQQ"], 0.35, abs_tol=1e-4)
    assert math.isclose(w_norm["SPY"], 0.25, abs_tol=1e-4)
    assert math.isclose(w_norm["XLK"], 0.20, abs_tol=1e-4)
    assert math.isclose(w_norm["SHV"], 0.20, abs_tol=1e-4)
    assert math.isclose(sum(w_norm.values()), 1.0, abs_tol=1e-6)

    # 3. CORRECTION_FRAGILE: 20% XLV, 80% SHV
    w_corr = get_regime_base_weights(MarketRegime.CORRECTION_FRAGILE)
    assert math.isclose(w_corr["XLV"], 0.20, abs_tol=1e-4)
    assert math.isclose(w_corr["SHV"], 0.80, abs_tol=1e-4)
    assert math.isclose(sum(w_corr.values()), 1.0, abs_tol=1e-6)

    # 4. BEAR_CRISIS: 100% SHV
    w_bear = get_regime_base_weights(MarketRegime.BEAR_CRISIS)
    assert math.isclose(w_bear["SHV"], 1.00, abs_tol=1e-6)

    # 5. Custom safe havens: 50% TLT, 50% GLD in BULL_NORMAL (0.20 * 0.5 = 0.10 each)
    w_custom = get_regime_base_weights(MarketRegime.BULL_NORMAL, safe_haven_weights={"TLT": 0.5, "GLD": 0.5})
    assert math.isclose(w_custom["TLT"], 0.10, abs_tol=1e-4)
    assert math.isclose(w_custom["GLD"], 0.10, abs_tol=1e-4)
    assert math.isclose(w_custom["QQQ"], 0.35, abs_tol=1e-4)
    assert math.isclose(w_custom["SPY"], 0.25, abs_tol=1e-4)
    assert math.isclose(w_custom["XLK"], 0.20, abs_tol=1e-4)


def test_rebalancer_floating_point_epsilon_clamping():
    """Verify PortfolioRebalancer handles edge-case epsilon weights without ValidationError."""
    rebalancer = PortfolioRebalancer(drift_band=0.025, min_order_threshold=0.005)

    # 1. Target weight slightly exceeds 1.0 due to float residual
    target_over = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_NORMAL,
        weights={"SHV": 1.0000008},
        cash_weight=1.0,
        rationale="Epsilon residual test",
    )
    orders = rebalancer.compute_rebalance_orders(target_over, current_weights={"SHV": 0.0})
    assert len(orders) == 1
    assert orders[0].target_weight == 1.0
    assert orders[0].delta_weight == 1.0

    # 2. Current weight slightly exceeds 1.0
    target_zero = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 0.0, "SHV": 1.0},
        cash_weight=1.0,
        rationale="Epsilon liquidation test",
    )
    orders_sell = rebalancer.compute_rebalance_orders(target_zero, current_weights={"SPY": 1.000001, "SHV": 0.0})
    assert len(orders_sell) == 2
    sell_spy = [o for o in orders_sell if o.symbol == "SPY"][0]
    assert sell_spy.target_weight == 0.0
    assert sell_spy.delta_weight == -1.0
    assert sell_spy.action == "SELL"
