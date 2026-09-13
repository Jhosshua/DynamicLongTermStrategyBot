"""
tests/adversarial/test_m3_adversarial_allocator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tier 5 Adversarial Verification Suite for Milestone M3 (Portfolio Allocator & Rebalancer):
1. Weight sum-to-one validation: 5,000 random permutations of regimes, volatility scaling
   factors (0.05 to 1.50), and drawdown gates (0.0 to 1.0). In 100% of cases,
   TargetAllocation.weights sum to 1.0 +/- 1e-5 and all weights >= 0.0.
2. 2022 Inflation Grind stress test: simultaneous equity and treasury crash (P_TLT < SMA_200).
   Verifying TLT allocation is strictly 0.0% and 100% of defensive capital is routed
   to SHV/BIL cash and GLD.
3. Drift band filtering (+/- 2.5%), micro-order suppression (<0.5%), and strict
   SELL-before-BUY order sequencing.
4. Edge-case and boundary robustness: STALE_DATA_HOLD freeze, extreme inputs, and invariant preservation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import random
from typing import Dict, List
import numpy as np
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
from strategy_engine.signals.momentum import (
    evaluate_safe_haven_dual_momentum,
    evaluate_safe_haven_qualification,
)
from strategy_engine.simulator.stress_scenarios import generate_2022_inflation_grind


# ============================================================================
# Helpers to generate synthetic market data fixtures
# ============================================================================

def _make_bar(symbol: str, dt: datetime, close: float) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=dt,
        open=close,
        high=close * 1.005,
        low=close * 0.995,
        close=close,
        volume=500_000,
    )


def _build_synthetic_universe(
    t0: datetime,
    n_days: int = 260,
    equity_trend: float = 1.0,
    tlt_trend: float = 1.0,
    gld_trend: float = 1.0,
) -> Dict[str, List[Bar]]:
    """Generate deterministic synthetic market data for all universe tickers."""
    symbols = [
        "SPY", "QQQ", "XLK", "XLY", "XLC", "XLI", "XLF",
        "XLV", "XLP", "XLU", "XLE", "XLB", "XLRE",
        "TLT", "GLD", "SHV", "BIL",
    ]
    data: Dict[str, List[Bar]] = {}
    for sym in symbols:
        bars: List[Bar] = []
        base_price = 100.0 if sym not in ("SPY", "QQQ") else 400.0
        for i in range(n_days):
            dt = t0 + timedelta(days=i)
            if sym in ("SPY", "QQQ") or sym.startswith("XL"):
                p = base_price * (1.0 + (equity_trend - 1.0) * (i / n_days) + 0.001 * (i % 5))
            elif sym == "TLT":
                p = base_price * (1.0 + (tlt_trend - 1.0) * (i / n_days) + 0.001 * (i % 3))
            elif sym == "GLD":
                p = base_price * (1.0 + (gld_trend - 1.0) * (i / n_days) + 0.001 * (i % 4))
            else:  # SHV, BIL
                p = 100.0 + 0.005 * i
            bars.append(_make_bar(sym, dt, max(1.0, p)))
        data[sym] = bars
    return data


# ============================================================================
# 1. Empirical Weight Sum-to-One Validation (5,000 Random Permutations)
# ============================================================================

def test_sum_to_one_5000_random_permutations():
    """Adversarially generate 5,000 permutations of regimes, volatility scaling factors

    in [0.05, 1.50], and drawdown gates in [0.0, 1.0].
    Assert in 100% of cases:
    - TargetAllocation.weights strictly sum to 1.0 +/- 1e-5
    - All weights >= 0.0 (strictly long-only non-negative)
    - cash_weight >= 0.0
    - Dataclass invariants hold
    """
    rng = random.Random(1337)
    np_rng = np.random.default_rng(1337)

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

    # Pre-generate 5 distinct market data regimes
    datasets = {
        "bull": _build_synthetic_universe(t0, n_days=260, equity_trend=1.30, tlt_trend=1.05, gld_trend=1.10),
        "bear": _build_synthetic_universe(t0, n_days=260, equity_trend=0.65, tlt_trend=1.15, gld_trend=1.05),
        "inflation_grind": _build_synthetic_universe(t0, n_days=260, equity_trend=0.75, tlt_trend=0.70, gld_trend=1.05),
        "stagflation": _build_synthetic_universe(t0, n_days=260, equity_trend=0.70, tlt_trend=0.65, gld_trend=0.80),
        "sparse": {"SPY": [_make_bar("SPY", t0 + timedelta(days=i), 500.0) for i in range(260)]},
        "empty": {},
    }
    dataset_keys = list(datasets.keys())

    regimes = [
        MarketRegime.BULL_AGGRESSIVE,
        MarketRegime.BULL_NORMAL,
        MarketRegime.CORRECTION_FRAGILE,
        MarketRegime.BEAR_CRISIS,
        MarketRegime.STALE_DATA_HOLD,
    ]

    total_permutations = 5000
    failures = []

    for i in range(total_permutations):
        regime = rng.choice(regimes)
        vol_scale = float(np_rng.uniform(0.05, 1.50))
        dd_gate = float(np_rng.uniform(0.0, 1.0))
        circuit_breaker = rng.choice([True, False])
        data_choice = rng.choice(dataset_keys)
        market_data = datasets[data_choice]

        snapshot = SignalSnapshot.model_construct(
            timestamp=t0 + timedelta(days=260),
            spy_price=500.0,
            spy_sma50=490.0,
            spy_sma200=470.0,
            realized_vol_20d=0.12,
            vol_scale_factor=vol_scale,
            drawdown_pct=-float(np_rng.uniform(0.0, 0.30)),
            circuit_breaker_active=circuit_breaker,
            regime=regime,
            indicators={"drawdown_gate": dd_gate},
        )

        try:
            alloc = compute_deterministic_allocation(snapshot, market_data)
        except Exception as exc:
            failures.append(f"Permutation {i} crashed with {type(exc).__name__}: {exc}")
            continue

        weight_sum = sum(alloc.weights.values())
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            failures.append(f"Permutation {i}: sum {weight_sum:.8f} != 1.0 (regime={regime.value}, vol={vol_scale:.3f}, dd={dd_gate:.3f})")

        negative_weights = {k: v for k, v in alloc.weights.items() if v < 0.0}
        if negative_weights:
            failures.append(f"Permutation {i}: negative weights found: {negative_weights}")

        if alloc.cash_weight < -1e-6:
            failures.append(f"Permutation {i}: negative cash weight {alloc.cash_weight}")

    assert len(failures) == 0, f"{len(failures)} / {total_permutations} permutations failed invariants:\n" + "\n".join(failures[:10])


# ============================================================================
# 2. 2022 Inflation Grind Stress Test: Simultaneous Equity & Treasury Crash
# ============================================================================

def test_2022_inflation_grind_stress_scenario():
    """Verify 2022 Inflation Grind conditions (P_TLT < SMA_200):

    1. Long-duration treasury (TLT) allocation is strictly 0.0%.
    2. 100% of defensive capital is routed to SHV/BIL cash and GLD.
    3. Antonacci absolute momentum filters out collapsing bonds.
    """
    t0 = datetime(2022, 1, 3, tzinfo=timezone.utc)

    # 1. Test using calibrated historical simulator for 2022 Inflation Grind
    sim_bars = generate_2022_inflation_grind(seed=42)

    # Verify TLT indeed crashed below SMA200 in the simulator output
    tlt_closes = [b.close for b in sim_bars["TLT"]]
    tlt_sma200 = sum(tlt_closes[-200:]) / 200.0
    tlt_p = tlt_closes[-1]
    assert tlt_p < tlt_sma200, f"Simulated TLT price {tlt_p} should be below SMA200 {tlt_sma200}"

    # Verify qualification failure
    tlt_qualifies = evaluate_safe_haven_qualification("TLT", sim_bars["TLT"])
    assert not tlt_qualifies, "TLT must NOT qualify when P_TLT < SMA200"

    # Evaluate across regimes under 2022 Inflation Grind data
    for regime in [MarketRegime.BEAR_CRISIS, MarketRegime.CORRECTION_FRAGILE, MarketRegime.BULL_NORMAL]:
        snapshot = SignalSnapshot(
            timestamp=sim_bars["TLT"][-1].timestamp,
            spy_price=sim_bars["SPY"][-1].close,
            spy_sma50=sim_bars["SPY"][-50].close,
            spy_sma200=sum(b.close for b in sim_bars["SPY"][-200:]) / 200.0,
            realized_vol_20d=0.22,
            vol_scale_factor=0.55,
            drawdown_pct=-0.22,
            circuit_breaker_active=False,
            regime=regime,
            indicators={"drawdown_gate": 0.50},
        )

        alloc = compute_deterministic_allocation(snapshot, sim_bars)

        # 1. TLT must be strictly 0.0%
        assert alloc.weights.get("TLT", 0.0) == 0.0, f"TLT allocated {alloc.weights.get('TLT')} in regime {regime}"

        # 2. Defensive capital must consist strictly of SHV/BIL/CASH and GLD
        defensive_assets = {"SHV", "BIL", "CASH", "USD", "GLD"}
        non_equity_weights = {sym: w for sym, w in alloc.weights.items() if sym not in ("SPY", "QQQ") and not sym.startswith("XL")}
        for sym, w in non_equity_weights.items():
            assert sym in defensive_assets, f"Unexpected defensive asset {sym} with weight {w}"

        # 3. Sum to 1.0 within 1e-5
        assert math.isclose(sum(alloc.weights.values()), 1.0, rel_tol=1e-5)


def test_2022_inflation_grind_both_tlt_and_gld_crash():
    """Simulate extreme stagflation crash where BOTH TLT and GLD crash below SMA200.

    Verify that 100% of defensive capital is routed to SHV ultra-short cash.
    """
    t0 = datetime(2022, 1, 1, tzinfo=timezone.utc)
    n = 260
    # Both TLT and GLD crash -40% below their 200 SMA
    data = {
        "SPY": [_make_bar("SPY", t0 + timedelta(days=i), 500.0 * (1.0 - 0.3 * (i / n))) for i in range(n)],
        "QQQ": [_make_bar("QQQ", t0 + timedelta(days=i), 400.0 * (1.0 - 0.35 * (i / n))) for i in range(n)],
        "TLT": [_make_bar("TLT", t0 + timedelta(days=i), 150.0 * (1.0 - 0.4 * (i / n))) for i in range(n)],
        "GLD": [_make_bar("GLD", t0 + timedelta(days=i), 180.0 * (1.0 - 0.4 * (i / n))) for i in range(n)],
        "SHV": [_make_bar("SHV", t0 + timedelta(days=i), 100.0 + 0.01 * i) for i in range(n)],
    }

    # BEAR_CRISIS: 100% defensive
    snapshot = SignalSnapshot(
        timestamp=t0 + timedelta(days=n),
        spy_price=350.0,
        spy_sma50=380.0,
        spy_sma200=430.0,
        realized_vol_20d=0.30,
        vol_scale_factor=0.40,
        drawdown_pct=-0.30,
        circuit_breaker_active=True,
        regime=MarketRegime.BEAR_CRISIS,
        indicators={"drawdown_gate": 0.0},
    )

    alloc = compute_deterministic_allocation(snapshot, data)

    # In BEAR_CRISIS with failing TLT and GLD, allocation must be 100% SHV cash
    assert alloc.weights.get("TLT", 0.0) == 0.0
    assert alloc.weights.get("GLD", 0.0) == 0.0
    assert math.isclose(alloc.weights.get("SHV", 0.0), 1.0, rel_tol=1e-5)
    assert math.isclose(alloc.cash_weight, 1.0, rel_tol=1e-5)


# ============================================================================
# 3. Drift Band Filtering (+/- 2.5%), Micro-Orders (<0.5%), and Sequencing
# ============================================================================

def test_rebalancer_drift_band_and_micro_order_filtering():
    """Adversarially verify drift band and micro-order thresholds:

    - Deviations within +/- 2.5% emit NO orders
    - Deviations between 0.0% and 0.499% (micro-orders) emit NO orders even if band exceeded
    - Deviations strictly > 2.5% and >= 0.5% emit orders
    """
    rebalancer = PortfolioRebalancer(drift_band=0.025, min_order_threshold=0.005)
    target = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 0.40, "QQQ": 0.30, "SHV": 0.30},
        cash_weight=0.30,
        rationale="Drift test",
    )

    # Case A: Boundary test: deviations exactly +0.0249 and -0.0249 (within band)
    current_inside = {"SPY": 0.4249, "QQQ": 0.2751, "SHV": 0.30}
    orders_inside = rebalancer.compute_rebalance_orders(target, current_inside)
    assert len(orders_inside) == 0, f"Expected 0 orders for within-band drift, got {len(orders_inside)}"

    # Case B: Deviation exactly +/- 0.0250 (boundary of band: |delta| <= 0.025)
    current_boundary = {"SPY": 0.4250, "QQQ": 0.2750, "SHV": 0.30}
    orders_boundary = rebalancer.compute_rebalance_orders(target, current_boundary)
    assert len(orders_boundary) == 0, "Exact +/- 2.5% boundary must not emit orders"

    # Case C: Tighter band where deviation is 0.004 (exceeds band 0.002, but < 0.005 micro threshold)
    tight_rebalancer = PortfolioRebalancer(drift_band=0.002, min_order_threshold=0.005)
    current_micro = {"SPY": 0.404, "QQQ": 0.296, "SHV": 0.30}
    orders_micro = tight_rebalancer.compute_rebalance_orders(target, current_micro)
    assert len(orders_micro) == 0, "Micro-orders (< 0.5%) must be suppressed"

    # Case D: Exceeds band (+/- 0.05) and >= 0.5% -> Emits orders
    current_exceeded = {"SPY": 0.45, "QQQ": 0.25, "SHV": 0.30}
    orders_exceeded = rebalancer.compute_rebalance_orders(target, current_exceeded)
    assert len(orders_exceeded) == 2
    assert {o.symbol for o in orders_exceeded} == {"SPY", "QQQ"}


def test_order_sequencing_sells_strictly_before_buys():
    """Adversarially generate 1,000 multi-asset portfolios with random current weights.

    Assert that in 100% of generated order lists:
    - ALL SELL orders appear strictly before ANY BUY orders
    - delta_weight < 0 corresponds to action == SELL
    - delta_weight > 0 corresponds to action == BUY
    - notional is strictly positive
    """
    rng = random.Random(42)
    rebalancer = PortfolioRebalancer(drift_band=0.025, min_order_threshold=0.005)
    symbols = ["SPY", "QQQ", "XLK", "XLF", "XLV", "XLE", "TLT", "GLD", "SHV"]

    for trial in range(1000):
        # Generate random target weights summing to 1.0
        raw_target = {s: rng.uniform(0.01, 1.0) for s in rng.sample(symbols, k=rng.randint(3, 7))}
        target_norm = normalize_target_weights(raw_target)

        target = TargetAllocation(
            timestamp=datetime.now(timezone.utc),
            regime=MarketRegime.BULL_NORMAL,
            weights=target_norm,
            cash_weight=target_norm.get("SHV", 0.0),
            rationale="Sequencing test",
        )

        # Generate random current weights
        raw_current = {s: rng.uniform(0.01, 1.0) for s in rng.sample(symbols, k=rng.randint(3, 7))}
        current_norm = normalize_target_weights(raw_current)

        prices = {s: rng.uniform(50.0, 500.0) for s in symbols}
        portfolio_equity = 100_000.0

        orders = rebalancer.compute_rebalance_orders(
            target_allocation=target,
            current_weights=current_norm,
            portfolio_equity=portfolio_equity,
            current_prices=prices,
        )

        if not orders:
            continue

        actions = [o.action for o in orders]
        seen_buy = False
        for action in actions:
            if action == "BUY":
                seen_buy = True
            elif action == "SELL":
                assert not seen_buy, (
                    f"Trial {trial}: SELL order appeared after a BUY order! Actions: {actions}"
                )

        for o in orders:
            if o.action == "SELL":
                assert o.side == OrderSide.SELL
                assert o.delta_weight < 0.0
                assert o.delta_dollars is not None and o.delta_dollars < 0.0
                assert o.delta_shares is not None and o.delta_shares < 0.0
            else:
                assert o.side == OrderSide.BUY
                assert o.delta_weight > 0.0
                assert o.delta_dollars is not None and o.delta_dollars > 0.0
                assert o.delta_shares is not None and o.delta_shares > 0.0

            assert o.notional is not None and o.notional > 0.0
            assert o.estimated_price is not None and o.estimated_price > 0.0


def test_stale_data_hold_freeze_invariants():
    """Verify that under STALE_DATA_HOLD:

    1. compute_deterministic_allocation returns 100% cash
    2. compute_rebalance_orders emits exactly 0 orders regardless of massive drift
    """
    snapshot = SignalSnapshot(
        timestamp=datetime.now(timezone.utc),
        spy_price=500.0,
        spy_sma50=490.0,
        spy_sma200=470.0,
        realized_vol_20d=0.12,
        vol_scale_factor=1.0,
        drawdown_pct=0.0,
        circuit_breaker_active=False,
        regime=MarketRegime.STALE_DATA_HOLD,
        indicators={},
    )

    alloc = compute_deterministic_allocation(snapshot, {})
    assert alloc.weights == {"SHV": 1.0}
    assert alloc.cash_weight == 1.0

    rebalancer = PortfolioRebalancer()
    # Portfolio currently holds 100% equities, target is 100% cash
    orders = rebalancer.compute_rebalance_orders(alloc, current_weights={"SPY": 0.50, "QQQ": 0.50})
    assert orders == [], "STALE_DATA_HOLD must freeze trading and emit 0 orders"


# ============================================================================
# 4. Pathological & Extreme Edge Cases
# ============================================================================

def test_pathological_weight_normalization():
    """Test normalize_target_weights against pathological input combinations."""
    # 1. Empty dict -> defaults to 100% cash
    assert normalize_target_weights({}) == {"SHV": 1.0}

    # 2. All zero or sub-threshold weights -> defaults to 100% cash
    assert normalize_target_weights({"SPY": 0.0, "QQQ": 1e-8}) == {"SHV": 1.0}

    # 3. Negative weights clamped to 0
    w_neg = {"SPY": -0.5, "QQQ": 0.5, "SHV": 0.5}
    norm_neg = normalize_target_weights(w_neg)
    assert norm_neg.get("SPY", 0.0) == 0.0
    assert math.isclose(norm_neg["QQQ"], 0.5, rel_tol=1e-5)
    assert math.isclose(norm_neg["SHV"], 0.5, rel_tol=1e-5)
    assert math.isclose(sum(norm_neg.values()), 1.0, rel_tol=1e-5)

    # 4. Custom cash symbol
    assert normalize_target_weights({}, cash_symbol="BIL") == {"BIL": 1.0}

    # 5. Massive scale weights
    w_huge = {"SPY": 1e12, "QQQ": 1e12}
    norm_huge = normalize_target_weights(w_huge)
    assert math.isclose(norm_huge["SPY"], 0.5, rel_tol=1e-5)
    assert math.isclose(norm_huge["QQQ"], 0.5, rel_tol=1e-5)
    assert math.isclose(sum(norm_huge.values()), 1.0, rel_tol=1e-5)


# ============================================================================
# 5. Vulnerability & Boundary Mismatch Verification Tests
# ============================================================================

def test_vulnerability_order_intent_epsilon_tolerance_mismatch():
    """Verify PortfolioRebalancer safely clamps floating-point epsilon noise (1.000001)."""
    target = TargetAllocation(
        timestamp=datetime.now(timezone.utc),
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 1.000001},
        cash_weight=0.0,
        rationale="Floating point precision boundary test",
    )
    rebalancer = PortfolioRebalancer()
    orders = rebalancer.compute_rebalance_orders(target, {"SPY": 0.0})
    assert len(orders) == 1
    assert orders[0].symbol == "SPY"
    assert orders[0].target_weight == 1.0
    assert orders[0].delta_weight == 1.0


def test_vulnerability_get_regime_base_weights_safe_haven_double_discount():
    """Verify get_regime_base_weights yields nominal weights without safe-haven double discount."""
    weights = get_regime_base_weights(MarketRegime.BULL_NORMAL)
    assert math.isclose(weights["QQQ"], 0.35, abs_tol=1e-4)
    assert math.isclose(weights["SPY"], 0.25, abs_tol=1e-4)
    assert math.isclose(weights["XLK"], 0.20, abs_tol=1e-4)
    assert math.isclose(weights["SHV"], 0.20, abs_tol=1e-4)

    weights_corr = get_regime_base_weights(MarketRegime.CORRECTION_FRAGILE)
    assert math.isclose(weights_corr["XLV"], 0.20, abs_tol=1e-4)
    assert math.isclose(weights_corr["SHV"], 0.80, abs_tol=1e-4)

