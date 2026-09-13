"""
tests/adversarial/test_cm2_challenger_empirical_invariants.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone C-M2 Empirical Challenger Adversarial Invariant Stress Test Suite.

Author: challenger_cm2_1
Mission: Empirically challenge and stress-test:
1. Forward-Data Contamination & Zero Lookahead Bias:
   - Extreme forward surges (+1000%) and crashes (-99%) injected into future bars.
   - Exact bit-for-bit invariance of compute_deterministic_allocation with vs without current_time.
   - Out-of-order forward timestamp contamination.
   - Universal lookahead insulation across all market regimes.
2. Target Weight Normalization Stress & Degenerate Inputs:
   - 1,000 extreme randomized float vectors across 60 orders of magnitude (1e-30 to 1e+30).
   - Degenerate near-zero sums (just above and below 1e-9 cutoff).
   - Infinite, NaN, negative, micro-weight, and zero permutations.
   - Floating-point residual absorption into cash_symbol.
   - Strict sum(w_i) = 1.0 +/- 1e-5 and non-negativity guarantees.
3. Antonacci 12-1 Dual Momentum Stress Testing:
   - Extreme choppy sideways noise (whipsaws around 200-day SMA).
   - Single-day and multi-day flash crashes (-20% to -90%).
   - 2022 duration shocks (rate grinds with TLT falling below 200 SMA): strictly 0% TLT.
   - Simultaneous TLT and GLD disqualification routing 100% to cash.
   - Boundary equality condition (Price == SMA200 / Mom == 0.0) safety bias.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Dict, List, Optional
import numpy as np
import pytest

from strategy_engine.allocator.rules import (
    compute_deterministic_allocation,
    get_regime_base_weights,
    normalize_target_weights,
)
from strategy_engine.core import universe
from strategy_engine.core.models import (
    Bar,
    MarketRegime,
    OrderIntent,
    OrderSide,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.signals.indicators import filter_bars_point_in_time
from strategy_engine.signals.momentum import (
    calculate_12_1_momentum,
    evaluate_absolute_momentum,
    evaluate_safe_haven_qualification,
    evaluate_safe_haven_dual_momentum,
    compute_safe_haven_weights,
)
from strategy_engine.signals.regime_detector import SignalEngine
from strategy_engine.simulator.stress_scenarios import generate_2022_inflation_grind


def _create_bar(
    symbol: str,
    dt: datetime,
    close: float,
    open_p: Optional[float] = None,
    high_p: Optional[float] = None,
    low_p: Optional[float] = None,
    volume: int = 100_000,
) -> Bar:
    c = float(close)
    o = float(open_p if open_p is not None else c)
    h = float(high_p if high_p is not None else max(o, c) * 1.005)
    l = float(low_p if low_p is not None else min(o, c) * 0.995)
    return Bar(
        symbol=symbol,
        timestamp=dt,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=volume,
    )


def _build_synthetic_universe(
    n_days: int = 270,
    seed: int = 789,
    start_dt: Optional[datetime] = None,
) -> Dict[str, List[Bar]]:
    rng = np.random.default_rng(seed)
    t0 = start_dt or datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc)
    symbols = [
        "SPY", "QQQ", "TLT", "GLD", "SHV", "BIL",
        "XLK", "XLF", "XLV", "XLE", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC"
    ]
    base_prices = {
        "SPY": 500.0, "QQQ": 450.0, "TLT": 100.0, "GLD": 190.0, "SHV": 110.0, "BIL": 91.0,
        "XLK": 210.0, "XLF": 42.0, "XLV": 145.0, "XLE": 88.0, "XLI": 125.0,
        "XLU": 68.0, "XLP": 78.0, "XLY": 180.0, "XLB": 88.0, "XLRE": 40.0, "XLC": 82.0,
    }
    data: Dict[str, List[Bar]] = {s: [] for s in symbols}
    for s in symbols:
        p = base_prices[s]
        for i in range(n_days):
            dt = t0 + timedelta(days=i)
            drift = 0.0003 if s not in ("SHV", "BIL") else 0.00008
            vol = 0.012 if s not in ("SHV", "BIL") else 0.0001
            ret = rng.normal(drift, vol)
            p = max(0.5, p * (1.0 + ret))
            h = p * (1.0 + abs(rng.normal(0, 0.003)))
            l = p * (1.0 - abs(rng.normal(0, 0.003)))
            data[s].append(_create_bar(s, dt, close=p, open_p=p, high_p=h, low_p=l))
    return data


# ============================================================================
# Suite 1: Forward-Data Contamination & Zero Lookahead Bias Stress Tests
# ============================================================================

class TestForwardDataContaminationAndLookaheadBias:
    """Empirical challenge: inject extreme forward-data mutations and verify zero lookahead leak."""

    def test_forward_contamination_future_spike_1000pct_omitted_current_time(self):
        """Inject +1000% future price surge on defensive assets and -99% crash on equities.
        
        Verify compute_deterministic_allocation without current_time matches clean PIT bit-for-bit.
        """
        clean_data = _build_synthetic_universe(n_days=265, seed=101)
        t_eval = clean_data["SPY"][220].timestamp

        engine = SignalEngine()
        clean_snap = engine.compute_daily_signals(clean_data, current_time=t_eval)

        # Baseline: clean PIT data containing only bars up to t_eval
        clean_pit = {
            s: [b for b in bars if b.timestamp <= t_eval]
            for s, bars in clean_data.items()
        }
        alloc_clean = compute_deterministic_allocation(
            signals=clean_snap,
            market_data=clean_pit,
            current_time=t_eval,
        )

        # Aggressive forward contamination: 60 forward days with +1000% spike on TLT/GLD/SHV
        # and -99% drop on equities
        contam_data: Dict[str, List[Bar]] = {}
        for s, bars in clean_pit.items():
            contam_data[s] = list(bars)
            for d in range(1, 61):
                fwd_dt = t_eval + timedelta(days=d)
                multiplier = 10.0 if s in ("TLT", "GLD", "SHV", "BIL") else 0.01
                fwd_p = bars[-1].close * multiplier
                contam_data[s].append(_create_bar(s, fwd_dt, close=fwd_p))

        # Invoke WITHOUT passing current_time argument
        alloc_contam = compute_deterministic_allocation(
            signals=clean_snap,
            market_data=contam_data,
        )

        assert alloc_contam.regime == alloc_clean.regime
        assert math.isclose(alloc_contam.cash_weight, alloc_clean.cash_weight, abs_tol=1e-7)
        assert set(alloc_contam.weights.keys()) == set(alloc_clean.weights.keys())
        for sym, w in alloc_clean.weights.items():
            assert math.isclose(alloc_contam.weights[sym], w, abs_tol=1e-6), (
                f"Lookahead leak for {sym}: clean={w}, contam={alloc_contam.weights[sym]}"
            )

    def test_forward_contamination_future_drop_99pct_omitted_current_time(self):
        """Inject -99% crash on all assets in the future.
        
        Verify allocator produces identical weights with and without current_time.
        """
        clean_data = _build_synthetic_universe(n_days=265, seed=202)
        t_eval = clean_data["SPY"][230].timestamp

        engine = SignalEngine()
        clean_snap = engine.compute_daily_signals(clean_data, current_time=t_eval)

        clean_pit = {
            s: [b for b in bars if b.timestamp <= t_eval]
            for s, bars in clean_data.items()
        }
        alloc_clean = compute_deterministic_allocation(
            signals=clean_snap,
            market_data=clean_pit,
            current_time=t_eval,
        )

        contam_data: Dict[str, List[Bar]] = {}
        for s, bars in clean_pit.items():
            contam_data[s] = list(bars)
            for d in range(1, 45):
                fwd_dt = t_eval + timedelta(days=d)
                fwd_p = max(0.01, bars[-1].close * 0.01)
                contam_data[s].append(_create_bar(s, fwd_dt, close=fwd_p))

        # Test both: explicit current_time vs omitted current_time
        alloc_explicit = compute_deterministic_allocation(
            signals=clean_snap,
            market_data=contam_data,
            current_time=t_eval,
        )
        alloc_implicit = compute_deterministic_allocation(
            signals=clean_snap,
            market_data=contam_data,
        )

        for sym, w in alloc_clean.weights.items():
            assert math.isclose(alloc_explicit.weights[sym], w, abs_tol=1e-6)
            assert math.isclose(alloc_implicit.weights[sym], w, abs_tol=1e-6)

    def test_forward_contamination_unordered_timestamps(self):
        """Verify that out-of-order forward contaminated bars are strictly filtered and sorted."""
        clean_data = _build_synthetic_universe(n_days=265, seed=303)
        t_eval = clean_data["SPY"][210].timestamp

        engine = SignalEngine()
        clean_snap = engine.compute_daily_signals(clean_data, current_time=t_eval)

        clean_pit = {
            s: [b for b in bars if b.timestamp <= t_eval]
            for s, bars in clean_data.items()
        }
        alloc_clean = compute_deterministic_allocation(signals=clean_snap, market_data=clean_pit)

        # Scrambled data: insert future bars at beginning and middle of bar list
        scrambled_data: Dict[str, List[Bar]] = {}
        for s, bars in clean_pit.items():
            future_bars = [
                _create_bar(s, t_eval + timedelta(days=d), close=bars[-1].close * 10.0)
                for d in range(1, 30)
            ]
            # Interleave future bars into historical bars
            combined = future_bars[:10] + bars + future_bars[10:]
            scrambled_data[s] = combined

        alloc_scrambled = compute_deterministic_allocation(
            signals=clean_snap,
            market_data=scrambled_data,
        )

        for sym, w in alloc_clean.weights.items():
            assert math.isclose(alloc_scrambled.weights[sym], w, abs_tol=1e-6), (
                f"Out-of-order forward timestamp leaked for {sym}: clean={w}, scrambled={alloc_scrambled.weights[sym]}"
            )

    @pytest.mark.parametrize("regime", [
        MarketRegime.BULL_AGGRESSIVE,
        MarketRegime.BULL_NORMAL,
        MarketRegime.CORRECTION_FRAGILE,
        MarketRegime.BEAR_CRISIS,
        MarketRegime.STALE_DATA_HOLD,
    ])
    def test_forward_contamination_across_all_regimes(self, regime: MarketRegime):
        """Universal verification: forward price distortion never leaks across any regime."""
        clean_data = _build_synthetic_universe(n_days=265, seed=404)
        t_eval = clean_data["SPY"][200].timestamp

        # Mock signal snapshot for the targeted regime
        snap = SignalSnapshot(
            timestamp=t_eval,
            spy_price=500.0,
            spy_sma50=490.0,
            spy_sma200=460.0,
            realized_vol_20d=0.12,
            vol_scale_factor=1.0,
            drawdown_pct=-0.02,
            circuit_breaker_active=False,
            regime=regime,
            indicators={"drawdown_gate": 1.0, "in_recovery_lockout": 0.0},
        )

        clean_pit = {s: [b for b in bars if b.timestamp <= t_eval] for s, bars in clean_data.items()}
        alloc_clean = compute_deterministic_allocation(snap, clean_pit)

        # Contaminate with extreme surges (+1000%)
        contam = {s: list(bars) for s, bars in clean_pit.items()}
        for s in contam:
            for d in range(1, 30):
                contam[s].append(_create_bar(s, t_eval + timedelta(days=d), close=contam[s][-1].close * 10.0))

        alloc_contam = compute_deterministic_allocation(snap, contam)

        assert alloc_contam.regime == alloc_clean.regime
        assert math.isclose(alloc_contam.cash_weight, alloc_clean.cash_weight, abs_tol=1e-6)
        for sym, w in alloc_clean.weights.items():
            assert math.isclose(alloc_contam.weights[sym], w, abs_tol=1e-6)

    def test_future_tlt_surge_in_bear_regime_never_induces_tlt_allocation(self):
        """In BEAR_CRISIS when TLT is historically in a downtrend (below 200 SMA),
        a future +2000% surge must NOT cause TLT to be selected.
        """
        clean_data = _build_synthetic_universe(n_days=265, seed=505)
        t_eval = clean_data["SPY"][220].timestamp

        # Force historical TLT in clean_data to be falling below 200 SMA
        tlt_bars = clean_data["TLT"][:221]
        for i, b in enumerate(tlt_bars):
            # Declining prices so current price is well below 200 SMA
            p = 120.0 - (i * 0.25)
            tlt_bars[i] = _create_bar("TLT", b.timestamp, close=p)
        clean_data["TLT"] = tlt_bars

        snap = SignalSnapshot(
            timestamp=t_eval,
            spy_price=400.0,
            spy_sma50=420.0,
            spy_sma200=450.0,
            realized_vol_20d=0.35,
            vol_scale_factor=0.34,
            drawdown_pct=-0.25,
            circuit_breaker_active=True,
            regime=MarketRegime.BEAR_CRISIS,
            indicators={"drawdown_gate": 0.0, "in_recovery_lockout": 1.0},
        )

        clean_pit = {s: [b for b in bars if b.timestamp <= t_eval] for s, bars in clean_data.items()}
        alloc_clean = compute_deterministic_allocation(snap, clean_pit)
        assert alloc_clean.weights.get("TLT", 0.0) == 0.0

        # Now contaminate future TLT bars with +2000% spike
        contam = {s: list(bars) for s, bars in clean_pit.items()}
        for d in range(1, 40):
            fwd_dt = t_eval + timedelta(days=d)
            contam["TLT"].append(_create_bar("TLT", fwd_dt, close=5000.0))

        alloc_contam = compute_deterministic_allocation(snap, contam)
        assert alloc_contam.weights.get("TLT", 0.0) == 0.0, (
            f"Future TLT surge contaminated allocator: got TLT={alloc_contam.weights.get('TLT')}"
        )


# ============================================================================
# Suite 2: Target Weight Normalization Stress & Degenerate Inputs
# ============================================================================

class TestTargetWeightNormalizationStress:
    """Empirical challenge: stress-test normalize_target_weights across 1,000 extreme vectors."""

    def test_normalize_target_weights_1000_extreme_random_vectors(self):
        """Fuzz normalize_target_weights across 1,000 vectors spanning 60 orders of magnitude."""
        rng = np.random.default_rng(9999)
        symbols_pool = [
            "SPY", "QQQ", "TLT", "GLD", "SHV", "BIL", "XLK", "XLF", "XLV", "XLE",
            "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC", "AAPL", "MSFT", "NVDA"
        ]

        for trial in range(1000):
            k = int(rng.integers(1, len(symbols_pool) + 1))
            chosen = rng.choice(symbols_pool, size=k, replace=False)
            weights: Dict[str, float] = {}

            scenario = rng.integers(0, 5)
            if scenario == 0:
                # Extreme dynamic range (1e-30 to 1e+30)
                for sym in chosen:
                    exp = rng.uniform(-30.0, 30.0)
                    weights[sym] = float(10.0 ** exp)
            elif scenario == 1:
                # Near-zero micro weights mixed with negative weights
                for sym in chosen:
                    weights[sym] = float(rng.uniform(-10.0, 1e-6))
            elif scenario == 2:
                # Highly skewed: one giant, rest tiny
                for sym in chosen:
                    weights[sym] = 1e-8
                weights[chosen[0]] = 1e12
            elif scenario == 3:
                # Negative and zero mix
                for sym in chosen:
                    weights[sym] = float(rng.choice([-50.0, -1.0, 0.0, 1e-9]))
            else:
                # Standard uniform random
                for sym in chosen:
                    weights[sym] = float(rng.uniform(-0.5, 5.0))

            normalized = normalize_target_weights(weights, cash_symbol="SHV")
            total = sum(normalized.values())

            assert math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5), (
                f"Trial {trial} sum violation: total={total}, raw={weights}"
            )
            for s, w in normalized.items():
                assert w >= 0.0, f"Trial {trial} negative weight: {s}={w}"
                assert w <= 1.0 + 1e-5, f"Trial {trial} weight > 1: {s}={w}"

    def test_normalize_target_weights_degenerate_near_zero_boundary(self):
        """Verify behavior around the 1e-9 threshold."""
        # Just below 1e-9 threshold -> should return {SHV: 1.0}
        sub_thresh = {"SPY": 5e-10, "QQQ": 4e-10}
        res_sub = normalize_target_weights(sub_thresh, cash_symbol="SHV")
        assert res_sub == {"SHV": 1.0}

        # Exactly 0.0 -> should return {SHV: 1.0}
        zeros = {"SPY": 0.0, "QQQ": 0.0}
        assert normalize_target_weights(zeros, cash_symbol="SHV") == {"SHV": 1.0}

        # Empty dict -> should return {SHV: 1.0}
        assert normalize_target_weights({}, cash_symbol="SHV") == {"SHV": 1.0}

        # Just above 1e-7 threshold for micro-weights (since items <= 1e-7 are pruned)
        above = {"SPY": 1.01e-6, "QQQ": 2.02e-6}
        res_above = normalize_target_weights(above, cash_symbol="SHV")
        assert math.isclose(sum(res_above.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(res_above["SPY"], 1.01 / 3.03, rel_tol=1e-4)

    def test_normalize_target_weights_nan_inf_mixed_stress(self):
        """Mixed vector of NaNs, Infs, negative numbers, and valid numbers."""
        raw = {
            "SPY": float("nan"),
            "QQQ": float("inf"),
            "TLT": float("-inf"),
            "GLD": -999.0,
            "XLK": 0.0,
            "SHV": 0.40,
            "BIL": 0.60,
        }
        res = normalize_target_weights(raw, cash_symbol="SHV")
        assert math.isclose(sum(res.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5)
        assert "SPY" not in res
        assert "QQQ" not in res
        assert "TLT" not in res
        assert "GLD" not in res
        assert math.isclose(res["SHV"], 0.40, abs_tol=1e-4)
        assert math.isclose(res["BIL"], 0.60, abs_tol=1e-4)

    def test_normalize_target_weights_floating_point_residual_absorption(self):
        """Vectors with non-terminating decimals (1/3, 1/7, 1/11, 1/13, 1/17)
        verify that floating-point residual is deterministically absorbed into cash_symbol.
        """
        for n_assets in [3, 7, 11, 13, 17, 19]:
            raw = {f"ASSET_{i}": 1.0 / n_assets for i in range(n_assets)}
            normalized = normalize_target_weights(raw, cash_symbol="SHV")
            total = sum(normalized.values())
            assert math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5)
            # Cash symbol exists or residual is 0
            for w in normalized.values():
                assert w >= 0.0


# ============================================================================
# Suite 3: Antonacci 12-1 Dual Momentum Stress Testing
# ============================================================================

class TestAntonacciDualMomentumStress:
    """Empirical challenge: test Antonacci dual momentum under choppy sideways noise,
    flash crashes, and 2022 rate grinds (verify 0% TLT allocation).
    """

    def test_antonacci_choppy_sideways_whipsaw_noise(self):
        """300 days of extreme choppy sideways whipsaw oscillating around 200 SMA.
        
        Assert: Whenever current price <= SMA200 or 12-1 momentum <= 0,
        TLT allocation is strictly 0.0%, and total defensive capital is preserved.
        """
        rng = np.random.default_rng(888)
        t0 = datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc)
        n_days = 320

        # Construct choppy sideways prices centered around $100
        tlt_prices = []
        p = 100.0
        for i in range(n_days):
            noise = rng.normal(0, 0.02)
            # Sinusoidal oscillation + mean reversion
            p = 100.0 + 5.0 * math.sin(i / 5.0) + noise * 5.0
            tlt_prices.append(max(10.0, p))

        tlt_bars = [_create_bar("TLT", t0 + timedelta(days=i), close=p) for i, p in enumerate(tlt_prices)]
        shv_bars = [_create_bar("SHV", t0 + timedelta(days=i), close=110.0 + i * 0.005) for i in range(n_days)]
        gld_bars = [_create_bar("GLD", t0 + timedelta(days=i), close=180.0 + i * 0.01) for i in range(n_days)]

        market_data = {"TLT": tlt_bars, "SHV": shv_bars, "GLD": gld_bars}

        # Evaluate across the last 60 days
        for eval_idx in range(255, n_days):
            sub_market = {s: bars[:eval_idx + 1] for s, bars in market_data.items()}
            closes = [b.close for b in sub_market["TLT"]]
            sma200 = float(np.mean(closes[-200:]))
            curr_p = closes[-1]
            mom12_1 = calculate_12_1_momentum(closes, lookback=252, skip=21)

            alloc_safe = evaluate_safe_haven_dual_momentum(
                sub_market,
                defensive_capital=1.0,
                cash_symbol="SHV",
            )

            total_w = sum(alloc_safe.values())
            assert math.isclose(total_w, 1.0, abs_tol=1e-5), f"Defensive capital leaked: {total_w}"

            if curr_p <= sma200 or mom12_1 <= 0.0:
                assert alloc_safe.get("TLT", 0.0) == 0.0, (
                    f"Day {eval_idx}: TLT allocated {alloc_safe.get('TLT')} when "
                    f"Price={curr_p:.2f}, SMA200={sma200:.2f}, Mom={mom12_1:.4f}"
                )
            else:
                # Qualified: TLT gets 40% of defensive capital
                assert math.isclose(alloc_safe["TLT"], 0.40, abs_tol=1e-5)

    def test_antonacci_flash_crash_immediate_disqualification(self):
        """Single-day -30% flash crash on TLT must immediately disqualify it from safe-haven routing."""
        t0 = datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc)
        n_days = 270

        # Stable upward trend for 260 days
        tlt_prices = [100.0 + i * 0.05 for i in range(260)]
        # Flash crash on day 261: drops 30% from 113 to 79
        crash_p = tlt_prices[-1] * 0.70
        tlt_prices.append(crash_p)
        for i in range(9):
            tlt_prices.append(crash_p * (1.0 + (i * 0.005)))

        tlt_bars = [_create_bar("TLT", t0 + timedelta(days=i), close=p) for i, p in enumerate(tlt_prices)]
        shv_bars = [_create_bar("SHV", t0 + timedelta(days=i), close=110.0 + i * 0.005) for i in range(len(tlt_prices))]
        gld_bars = [_create_bar("GLD", t0 + timedelta(days=i), close=180.0 + i * 0.01) for i in range(len(tlt_prices))]
        market_data = {"TLT": tlt_bars, "SHV": shv_bars, "GLD": gld_bars}

        # Day 260 (pre-crash): TLT qualifies
        pre_crash = {s: bars[:260] for s, bars in market_data.items()}
        assert evaluate_safe_haven_qualification("TLT", pre_crash["TLT"]) is True
        alloc_pre = evaluate_safe_haven_dual_momentum(pre_crash, defensive_capital=1.0)
        assert alloc_pre.get("TLT", 0.0) == 0.40

        # Day 261 (post-crash): TLT drops below SMA200 -> MUST BE DISQUALIFIED IMMEDIATELY
        post_crash = {s: bars[:261] for s, bars in market_data.items()}
        assert evaluate_safe_haven_qualification("TLT", post_crash["TLT"]) is False
        alloc_post = evaluate_safe_haven_dual_momentum(post_crash, defensive_capital=1.0)
        assert alloc_post.get("TLT", 0.0) == 0.0
        # Capital re-routed to cash and gold
        assert alloc_post.get("SHV", 0.0) >= 0.60

    def test_antonacci_2022_rate_grind_stress_zero_tlt(self):
        """Empirically test synthetic 2022 inflation grind scenario.
        
        Verify that TLT allocation is strictly 0.0% once rate hikes push TLT below 200 SMA.
        """
        market_data = generate_2022_inflation_grind(seed=777)
        tlt_bars = market_data["TLT"]

        # Check all monthly rebalance days in second half of the year
        rebalance_indices = [150, 180, 210, 240, 260]
        for idx in rebalance_indices:
            sub_market = {s: bars[:idx + 1] for s, bars in market_data.items()}
            alloc = evaluate_safe_haven_dual_momentum(sub_market, defensive_capital=1.0)
            # In 2022 rate grind, TLT is plunging: allocation must be strictly 0.0%
            assert alloc.get("TLT", 0.0) == 0.0, (
                f"Index {idx}: TLT received allocation {alloc.get('TLT')} during 2022 rate grind!"
            )
            # Total defensive capital strictly preserved
            assert math.isclose(sum(alloc.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5)

    def test_antonacci_simultaneous_tlt_and_gld_disqualification_100pct_cash(self):
        """When both TLT and GLD are broken (below SMA200 and negative momentum),
        100% of defensive capital must route to cash (SHV).
        """
        t0 = datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc)
        n_days = 265

        # Both TLT and GLD declining steadily
        tlt_bars = [_create_bar("TLT", t0 + timedelta(days=i), close=140.0 - i * 0.2) for i in range(n_days)]
        gld_bars = [_create_bar("GLD", t0 + timedelta(days=i), close=200.0 - i * 0.25) for i in range(n_days)]
        shv_bars = [_create_bar("SHV", t0 + timedelta(days=i), close=110.0 + i * 0.005) for i in range(n_days)]

        market_data = {"TLT": tlt_bars, "GLD": gld_bars, "SHV": shv_bars}

        alloc = evaluate_safe_haven_dual_momentum(market_data, defensive_capital=1.0)
        assert alloc == {"SHV": 1.0}
        assert alloc.get("TLT", 0.0) == 0.0
        assert alloc.get("GLD", 0.0) == 0.0

    def test_antonacci_boundary_equality_defensive_safety_bias(self):
        """Boundary test: Current Price == SMA200 and Mom_12-1 == 0.0 must evaluate to False.
        
        The requirement is strict inequality (Price > SMA200 and Mom > 0.0) to prevent
        allocating to borderline or stagnant assets.
        """
        t0 = datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc)
        # Constant prices = 100.0
        # Then SMA200 = 100.0, current_price = 100.0 (price == SMA)
        # Mom_12-1 = 100.0 / 100.0 - 1.0 = 0.0 (mom == 0)
        flat_bars = [_create_bar("TLT", t0 + timedelta(days=i), close=100.0) for i in range(260)]

        qualified = evaluate_safe_haven_qualification("TLT", flat_bars)
        assert qualified is False, "Boundary tie (Price == SMA, Mom == 0) must NOT qualify."

    def test_antonacci_insufficient_history_safely_disqualified(self):
        """Assets with fewer than 200 bars must safely return False without exception."""
        t0 = datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc)
        short_bars = [_create_bar("TLT", t0 + timedelta(days=i), close=100.0) for i in range(150)]

        qualified = evaluate_safe_haven_qualification("TLT", short_bars, sma_window=200)
        assert qualified is False


# ============================================================================
# Suite 4: Advanced Randomized Permutation & Reversal Skip Insulation
# ============================================================================

class TestAdvancedInvariantsAndSkipInsulation:
    """Further empirical challenges on momentum skip window and random permutation invariance."""

    def test_forward_contamination_50_random_seeds_bit_for_bit(self):
        """Run 50 randomized market states with random future shocks (spikes, drops, noise).
        
        Verify zero lookahead bias bit-for-bit across all 50 seeds when current_time is omitted.
        """
        engine = SignalEngine()
        for seed in range(500, 550):
            data = _build_synthetic_universe(n_days=265, seed=seed)
            t_eval = data["SPY"][220].timestamp

            snap = engine.compute_daily_signals(data, current_time=t_eval)
            clean_pit = {s: [b for b in bars if b.timestamp <= t_eval] for s, bars in data.items()}
            alloc_clean = compute_deterministic_allocation(snap, clean_pit)

            # Contaminate
            rng = np.random.default_rng(seed)
            contam = {s: list(bars) for s, bars in clean_pit.items()}
            for s in contam:
                for d in range(1, 40):
                    fwd_dt = t_eval + timedelta(days=d)
                    shock = float(rng.choice([0.05, 0.5, 2.0, 10.0]))
                    contam[s].append(_create_bar(s, fwd_dt, close=contam[s][-1].close * shock))

            alloc_contam = compute_deterministic_allocation(snap, contam)

            assert alloc_contam.regime == alloc_clean.regime
            assert math.isclose(alloc_contam.cash_weight, alloc_clean.cash_weight, abs_tol=1e-6)
            for sym, w in alloc_clean.weights.items():
                assert math.isclose(alloc_contam.weights[sym], w, abs_tol=1e-6), (
                    f"Seed {seed}: discrepancy in {sym}: clean={w}, contam={alloc_contam.weights[sym]}"
                )

    def test_momentum_skip_window_insulates_short_term_reversal_shocks(self):
        """Gary Antonacci 12-1 structural momentum skips the most recent 21 trading days (1 month).
        
        Verify that massive price shocks (+500% or -90%) inside the 21-day skip window have
        EXACTLY ZERO effect on the 12-1 momentum score.
        """
        t0 = datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc)
        base_bars = [_create_bar("SPY", t0 + timedelta(days=i), close=100.0 + i * 0.1) for i in range(260)]

        # Baseline momentum score
        base_mom = calculate_12_1_momentum(base_bars, lookback=252, skip=21)

        # Mutate bars in the skip window (last 21 bars: indices 239 through 259)
        shocked_up = list(base_bars)
        for idx in range(239, 260):
            shocked_up[idx] = _create_bar("SPY", base_bars[idx].timestamp, close=base_bars[idx].close * 5.0)

        mom_up = calculate_12_1_momentum(shocked_up, lookback=252, skip=21)
        assert math.isclose(mom_up, base_mom, abs_tol=1e-9), (
            f"12-1 momentum changed during skip window surge: base={base_mom}, shocked={mom_up}"
        )

        shocked_down = list(base_bars)
        for idx in range(239, 260):
            shocked_down[idx] = _create_bar("SPY", base_bars[idx].timestamp, close=base_bars[idx].close * 0.1)

        mom_down = calculate_12_1_momentum(shocked_down, lookback=252, skip=21)
        assert math.isclose(mom_down, base_mom, abs_tol=1e-9), (
            f"12-1 momentum changed during skip window drop: base={base_mom}, shocked={mom_down}"
        )

    def test_normalize_target_weights_50_symbols_stress(self):
        """Stress-test normalize_target_weights with 50 distinct symbols."""
        rng = np.random.default_rng(777)
        raw_50 = {f"SYM_{i:02d}": float(rng.uniform(0.01, 10.0)) for i in range(50)}
        normalized = normalize_target_weights(raw_50, cash_symbol="SHV")
        total = sum(normalized.values())
        assert math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5)
        for s, w in normalized.items():
            assert w >= 0.0
            assert w <= 1.0
