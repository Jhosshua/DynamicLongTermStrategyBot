"""
tests/adversarial/verify_m3_iter2_adversarial.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Challenger verification suite for Milestone M3 Iteration 2:
1. Exact get_regime_base_weights for BULL_NORMAL.
2. Floating-point clamp and negative delta resilience in PortfolioRebalancer.
3. 5,000 randomized Monte Carlo iterations testing sum-to-one invariant.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import random
import sys
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


def test_bull_normal_base_weights_exact():
    weights = get_regime_base_weights(MarketRegime.BULL_NORMAL)
    expected = {"QQQ": 0.35, "SPY": 0.25, "XLK": 0.20, "SHV": 0.20}
    
    assert set(weights.keys()) == set(expected.keys()), f"Keys mismatch: {weights.keys()} != {expected.keys()}"
    for k, v in expected.items():
        val = weights[k]
        assert math.isclose(val, v, abs_tol=1e-6), f"Mismatch for {k}: {val} != {v}"
        assert round(val, 2) == v, f"Mismatch on round for {k}: {val} != {v}"
    
    total = sum(weights.values())
    assert math.isclose(total, 1.0, abs_tol=1e-6), f"Total != 1.0: {total}"


def test_rebalancer_floating_point_clamps_and_negative_deltas():
    rebalancer = PortfolioRebalancer(drift_band=0.025, min_order_threshold=0.005)
    now = datetime.now(timezone.utc)
    
    perturbations = [
        ("1.0 + 1e-7", 1.0 + 1e-7),
        ("1.0 + 1e-5", 1.0 + 1e-5),
        ("1.0 + 1e-4", 1.0 + 1e-4),
        ("1.0 + 1e-2", 1.0 + 1e-2),
        ("0.0 - 1e-7", -1e-7),
        ("0.0 - 1e-5", -1e-5),
    ]

    # Sub-test A: Target weight > 1.0 with current_weight = 0.0 (BUY orders)
    for desc, tw in perturbations:
        if tw > 0.0:
            target = TargetAllocation.model_construct(
                timestamp=now,
                regime=MarketRegime.BULL_NORMAL,
                weights={"SPY": tw},
                cash_weight=0.0,
                rationale=f"Testing {desc}",
            )
            orders = rebalancer.compute_rebalance_orders(target, {"SPY": 0.0})
            assert len(orders) == 1, f"Expected 1 order for {desc}, got {len(orders)}"
            o = orders[0]
            assert o.target_weight <= 1.0, f"Order target_weight {o.target_weight} exceeds 1.0 for {desc}"
            assert o.target_weight >= 0.0, f"Order target_weight {o.target_weight} < 0.0 for {desc}"
            assert o.delta_weight <= 1.0, f"Order delta_weight {o.delta_weight} exceeds 1.0 for {desc}"
            assert o.side == OrderSide.BUY
            assert o.action == "BUY"

    # Sub-test B: Target weight with negative deltas (SELL orders)
    negative_delta_cases = [
        ("Target 0.2, Current 0.8 (delta -0.6)", {"SPY": 0.2, "SHV": 0.8}, {"SPY": 0.8, "SHV": 0.2}),
        ("Target 0.0, Current 1.0 + 1e-7 (delta -1.0)", {"SHV": 1.0}, {"SPY": 1.0 + 1e-7, "SHV": 0.0}),
        ("Target 0.0, Current 1.0 + 1e-5 (delta -1.0)", {"SHV": 1.0}, {"SPY": 1.0 + 1e-5, "SHV": 0.0}),
        ("Target 1.0 + 1e-7 clamped, Current 0.0, other asset Current 1.0 -> 0.0", 
         {"SPY": 1.0 + 1e-7}, {"QQQ": 1.0, "SPY": 0.0}),
        ("Target 1.0 + 1e-5 clamped, Current 0.0, other asset Current 1.0 -> 0.0", 
         {"SPY": 1.0 + 1e-5}, {"QQQ": 1.0, "SPY": 0.0}),
        ("Negative weight perturbation clamped", {"SPY": -1e-5, "SHV": 1.0}, {"SPY": 0.5, "SHV": 0.5}),
    ]

    for name, tw_dict, cw_dict in negative_delta_cases:
        target = TargetAllocation.model_construct(
            timestamp=now,
            regime=MarketRegime.BULL_NORMAL,
            weights=tw_dict,
            cash_weight=tw_dict.get("SHV", 0.0),
            rationale=f"Negative delta test: {name}",
        )
        orders = rebalancer.compute_rebalance_orders(
            target, 
            cw_dict,
            portfolio_equity=100_000.0,
            current_prices={"SPY": 500.0, "QQQ": 450.0, "SHV": 100.0}
        )
        assert len(orders) > 0, f"Expected orders for {name}, got 0"
        
        seen_buy = False
        for o in orders:
            assert o.target_weight is not None and 0.0 <= o.target_weight <= 1.0, (
                f"Order {o.symbol} target_weight {o.target_weight} out of bounds [0, 1] in {name}"
            )
            assert o.current_weight is not None and 0.0 <= o.current_weight <= 1.0, (
                f"Order {o.symbol} current_weight {o.current_weight} out of bounds [0, 1] in {name}"
            )
            assert o.delta_weight is not None and -1.0 <= o.delta_weight <= 1.0, (
                f"Order {o.symbol} delta_weight {o.delta_weight} out of bounds [-1, 1] in {name}"
            )
            if o.action == "BUY":
                seen_buy = True
            elif o.action == "SELL":
                assert not seen_buy, f"SELL appeared after BUY in {name}"
                assert o.delta_weight < 0.0
                assert o.delta_dollars is not None and o.delta_dollars < 0.0
            assert o.notional is not None and o.notional > 0.0


def test_5000_random_allocations_sum_to_one():
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)

    datasets = {
        "bull": _build_synthetic_universe(t0, n_days=260, equity_trend=1.35, tlt_trend=1.05, gld_trend=1.10),
        "bear": _build_synthetic_universe(t0, n_days=260, equity_trend=0.60, tlt_trend=1.20, gld_trend=1.05),
        "inflation_grind": _build_synthetic_universe(t0, n_days=260, equity_trend=0.75, tlt_trend=0.70, gld_trend=1.10),
        "stagflation": _build_synthetic_universe(t0, n_days=260, equity_trend=0.70, tlt_trend=0.65, gld_trend=0.80),
        "flat": _build_synthetic_universe(t0, n_days=260, equity_trend=1.00, tlt_trend=1.00, gld_trend=1.00),
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

    total_iters = 5000
    violations = []

    for i in range(total_iters):
        regime = rng.choice(regimes)
        vol_scale = float(np_rng.uniform(0.01, 2.50))
        dd_gate = float(np_rng.uniform(0.0, 1.0))
        circuit_breaker = rng.choice([True, False])
        data_choice = rng.choice(dataset_keys)
        market_data = datasets[data_choice]

        snapshot = SignalSnapshot.model_construct(
            timestamp=t0 + timedelta(days=260),
            spy_price=500.0,
            spy_sma50=490.0,
            spy_sma200=470.0,
            realized_vol_20d=0.15,
            vol_scale_factor=vol_scale,
            drawdown_pct=-float(np_rng.uniform(0.0, 0.40)),
            circuit_breaker_active=circuit_breaker,
            regime=regime,
            indicators={"drawdown_gate": dd_gate},
        )

        try:
            alloc = compute_deterministic_allocation(snapshot, market_data)
        except Exception as exc:
            violations.append(f"Iter {i} crashed: {type(exc).__name__}: {exc}")
            continue

        w_sum = sum(alloc.weights.values())
        if not math.isclose(w_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            violations.append(f"Iter {i}: sum={w_sum:.8f} != 1.0 (regime={regime}, vol={vol_scale:.2f}, dd={dd_gate:.2f})")

        for sym, w in alloc.weights.items():
            if w < -1e-8:
                violations.append(f"Iter {i}: negative weight {sym}={w}")
            if w > 1.0 + 1e-5:
                violations.append(f"Iter {i}: weight {sym}={w} exceeds 1.0")

        if alloc.cash_weight < -1e-6 or alloc.cash_weight > 1.0 + 1e-5:
            violations.append(f"Iter {i}: invalid cash_weight {alloc.cash_weight}")

        # Check cash consistency
        cash_syms = sum(w for sym, w in alloc.weights.items() if sym in ("SHV", "BIL", "CASH"))
        if not math.isclose(alloc.cash_weight, cash_syms, abs_tol=1e-4):
            violations.append(f"Iter {i}: cash_weight {alloc.cash_weight} != cash_syms {cash_syms}")

    assert len(violations) == 0, f"Found {len(violations)} invariant violations in {total_iters} iterations:\n" + "\n".join(violations[:10])
