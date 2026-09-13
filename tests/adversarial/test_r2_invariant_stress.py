"""
tests/adversarial/test_r2_invariant_stress.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Requirement R2 & Milestone C-M2: Mathematical & Economic Invariant Stress Testing.

Comprehensive adversarial test suite covering:
1. Lookahead Bias Insulation:
   - Verify future contaminated bars do not leak when current_time is omitted in compute_deterministic_allocation.
2. Trailing Drawdown State Preservation:
   - Verify uninitialized/fresh SignalEngine across multi-period drops accurately computes peak equity and trailing drawdown.
   - Verify multi-day sequential tracking and DrawdownDefenseTracker.prime().
3. SDE Sub-Penny Candlestick Floor & Inversion Guard:
   - Generate bars with prices < $0.01 (e.g. 0.008, 0.005, 0.00005) and verify zero Pydantic validation errors.
   - Guarantee low <= min(open, close) and high >= max(open, close).
4. Volatility Scaling Fail-Safe:
   - Verify realized_vol of NaN, +Inf, -Inf returns fail-safe 0.0 exposure.
5. OrderIntent Direction Consistency:
   - Reject contradictory action and side pairs (BUY vs SELL, SELL vs BUY) with ValueError.
6. Mathematical & Economic Invariants Stress:
   - Strict sum-to-one normalization (sum w_i = 1.0 +/- 1e-5) and non-negativity across 1,000+ permutations.
   - 2022 Duration shock disqualification: 0.0% TLT allocation when P_TLT < SMA_200.
   - 90% single-day black swan drop triggers ATR circuit breaker and cuts equity risk by >= 50%.
   - Merton Jump-Diffusion synthetic black swan stress simulation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import numpy as np
import pytest

from strategy_engine.allocator.rules import (
    compute_deterministic_allocation,
    get_regime_base_weights,
    normalize_target_weights,
)
from strategy_engine.core import math_utils, universe
from strategy_engine.core.models import (
    Bar,
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderType,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.signals.indicators import (
    DrawdownDefenseTracker,
    compute_volatility_scale_factor,
    evaluate_drawdown_gate,
    filter_bars_point_in_time,
)
from strategy_engine.signals.momentum import evaluate_safe_haven_qualification
from strategy_engine.signals.regime_detector import SignalEngine
from strategy_engine.simulator.regime_sde import MertonJumpDiffusionSimulator
from strategy_engine.simulator.stress_scenarios import (
    generate_2008_liquidity_crisis,
    generate_2020_flash_crash,
    generate_2022_inflation_grind,
)


def _create_bar(
    symbol: str,
    dt: datetime,
    close: float,
    open_p: float | None = None,
    high_p: float | None = None,
    low_p: float | None = None,
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


def _build_synthetic_market_data(
    n_days: int = 265,
    seed: int = 456,
    t0: datetime | None = None,
) -> dict[str, list[Bar]]:
    rng = np.random.default_rng(seed)
    start_dt = t0 or datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc)
    symbols = [
        "SPY", "QQQ", "TLT", "GLD", "SHV", "BIL",
        "XLK", "XLF", "XLV", "XLE", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC"
    ]
    data: dict[str, list[Bar]] = {s: [] for s in symbols}
    base_prices = {
        "SPY": 500.0, "QQQ": 450.0, "TLT": 95.0, "GLD": 180.0, "SHV": 110.0, "BIL": 91.0,
        "XLK": 200.0, "XLF": 40.0, "XLV": 140.0, "XLE": 85.0, "XLI": 120.0,
        "XLU": 65.0, "XLP": 75.0, "XLY": 175.0, "XLB": 85.0, "XLRE": 38.0, "XLC": 80.0,
    }

    for s in symbols:
        p = base_prices[s]
        for i in range(n_days):
            dt = start_dt + timedelta(days=i)
            drift = 0.0004 if s not in ("SHV", "BIL") else 0.0001
            vol = 0.012 if s not in ("SHV", "BIL") else 0.0001
            ret = rng.normal(drift, vol)
            p = max(0.5, p * (1.0 + ret))
            h = p * (1.0 + abs(rng.normal(0, 0.004)))
            l = p * (1.0 - abs(rng.normal(0, 0.004)))
            data[s].append(_create_bar(s, dt, close=p, open_p=p, high_p=h, low_p=l))

    return data


# ============================================================================
# 1. Lookahead Bias Insulation Tests
# ============================================================================

class TestLookaheadBiasInsulation:
    """Verify that omitting current_time in compute_deterministic_allocation never leaks future data."""

    def test_allocator_omitted_current_time_filters_future_bars(self):
        """When current_time is omitted, compute_deterministic_allocation must use signals.timestamp

        to filter market_data point-in-time, preventing future bars from leaking.
        """
        clean_data = _build_synthetic_market_data(n_days=265, seed=456)
        t_eval = clean_data["SPY"][225].timestamp

        engine = SignalEngine()
        clean_snap = engine.compute_daily_signals(clean_data, current_time=t_eval)

        # Baseline: clean allocation where clean_data only contains bars up to t_eval
        clean_pit_data = {
            s: [b for b in bars if b.timestamp <= t_eval]
            for s, bars in clean_data.items()
        }
        alloc_clean = compute_deterministic_allocation(
            signals=clean_snap,
            market_data=clean_pit_data,
        )

        # Contaminated dataset: forward bars (t+1 through t+45) with +500% surge on defensive
        # and -90% crash on equities to heavily distort momentum if leaked
        contaminated_data: dict[str, list[Bar]] = {}
        for s, bars in clean_pit_data.items():
            contaminated_data[s] = list(bars)
            for day_offset in range(1, 46):
                fwd_dt = t_eval + timedelta(days=day_offset)
                multiplier = 5.0 if s in ("TLT", "GLD", "SHV") else 0.1
                fwd_p = bars[-1].close * multiplier
                contaminated_data[s].append(_create_bar(s, fwd_dt, close=fwd_p))

        # Invoke WITHOUT passing current_time argument
        alloc_contam = compute_deterministic_allocation(
            signals=clean_snap,
            market_data=contaminated_data,
            # current_time is omitted!
        )

        # Invariant: weights, cash, and regime must be bit-for-bit identical
        assert alloc_contam.regime == alloc_clean.regime
        assert math.isclose(alloc_contam.cash_weight, alloc_clean.cash_weight, abs_tol=1e-6)
        for sym, w in alloc_clean.weights.items():
            assert sym in alloc_contam.weights
            assert math.isclose(alloc_contam.weights[sym], w, abs_tol=1e-5), (
                f"Lookahead leak detected for {sym}: clean={w}, contam={alloc_contam.weights[sym]}"
            )

    def test_allocator_tlt_leak_prevention(self):
        """Specifically verify the survey explorer empirical finding where future TLT surge

        previously caused 34.3% TLT allocation instead of 0.0%.
        """
        clean_data = _build_synthetic_market_data(n_days=265, seed=456)
        t_eval = clean_data["SPY"][225].timestamp
        engine = SignalEngine()
        snap = engine.compute_daily_signals(clean_data, current_time=t_eval)

        # In clean data, TLT allocation is determined strictly by past bars
        clean_pit = {s: [b for b in bars if b.timestamp <= t_eval] for s, bars in clean_data.items()}
        clean_alloc = compute_deterministic_allocation(snap, clean_pit)
        clean_tlt = clean_alloc.weights.get("TLT", 0.0)

        # Contaminate future bars
        contam = {s: list(bars) for s, bars in clean_pit.items()}
        for s in contam:
            for d in range(1, 46):
                p = contam[s][-1].close * (5.0 if s in ("TLT", "GLD", "SHV") else 0.1)
                contam[s].append(_create_bar(s, t_eval + timedelta(days=d), p))

        contam_alloc = compute_deterministic_allocation(snap, contam)
        contam_tlt = contam_alloc.weights.get("TLT", 0.0)

        assert math.isclose(contam_tlt, clean_tlt, abs_tol=1e-5), (
            f"TLT weight leaked future data: expected {clean_tlt}, got {contam_tlt}"
        )


# ============================================================================
# 2. Drawdown State Preservation Tests
# ============================================================================

class TestDrawdownStatePreservation:
    """Verify that uninitialized or fresh SignalEngine preserves trailing drawdown from equity curve."""

    def test_fresh_signal_engine_detects_historical_drawdown(self):
        """A fresh SignalEngine must prime its tracker across portfolio_equity_curve

        rather than resetting peak equity to the final trough value.
        """
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spy_bars = [
            _create_bar("SPY", t0 + timedelta(days=i), close=500.0)
            for i in range(100)
        ]
        # On day 100, SPY drops to 250 (-50%)
        spy_bars.append(_create_bar("SPY", t0 + timedelta(days=100), close=250.0))

        market_data = {"SPY": spy_bars}
        equity_curve = [500.0] * 100 + [250.0]

        # Newly instantiated engine
        fresh_engine = SignalEngine()
        sig = fresh_engine.compute_daily_signals(
            market_data=market_data,
            current_time=spy_bars[-1].timestamp,
            portfolio_equity_curve=equity_curve,
        )

        # Peak was 500.0, current is 250.0 -> drawdown should be -50.0% (-0.50)
        assert math.isclose(sig.drawdown_pct, -0.50, abs_tol=1e-4), (
            f"Expected -0.50 drawdown, got {sig.drawdown_pct}"
        )
        assert sig.indicators["drawdown_gate"] == 0.00  # Level 3 circuit breaker
        assert sig.indicators["in_recovery_lockout"] == 1.0

    def test_multi_step_drawdown_trajectory_on_fresh_engine(self):
        """Verify trailing drawdown calculation across varying peak-trough trajectories."""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spy_bars = [
            _create_bar("SPY", t0 + timedelta(days=i), close=100.0)
            for i in range(60)
        ]
        market_data = {"SPY": spy_bars}

        # Equity rose from 100 -> 200 (peak), then fell to 180 (-10%), then 170 (-15%)
        equity_curve = [100.0, 150.0, 200.0, 190.0, 180.0, 170.0]

        fresh_engine = SignalEngine()
        sig = fresh_engine.compute_daily_signals(
            market_data=market_data,
            current_time=spy_bars[-1].timestamp,
            portfolio_equity_curve=equity_curve,
        )

        expected_dd = (170.0 - 200.0) / 200.0  # -0.15 (-15%)
        assert math.isclose(sig.drawdown_pct, expected_dd, abs_tol=1e-4)
        assert sig.indicators["drawdown_gate"] == 0.00  # <= -15% triggers 0.00 cash gate

    def test_sequential_updates_do_not_reprime(self):
        """When an engine has already tracked history, subsequent daily updates update incrementally."""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spy_bars = [
            _create_bar("SPY", t0 + timedelta(days=i), close=100.0)
            for i in range(60)
        ]
        market_data = {"SPY": spy_bars}
        engine = SignalEngine()

        # Day 1: prime with [100, 200, 180] -> peak 200, current 180 (-10%)
        sig1 = engine.compute_daily_signals(
            market_data=market_data,
            current_time=spy_bars[50].timestamp,
            portfolio_equity_curve=[100.0, 200.0, 180.0],
        )
        assert math.isclose(sig1.drawdown_pct, -0.10, abs_tol=1e-4)
        assert len(engine.drawdown_tracker.equity_history) == 3

        # Day 2: new point appended [100, 200, 180, 160] (-20% from peak 200)
        sig2 = engine.compute_daily_signals(
            market_data=market_data,
            current_time=spy_bars[51].timestamp,
            portfolio_equity_curve=[100.0, 200.0, 180.0, 160.0],
        )
        assert math.isclose(sig2.drawdown_pct, -0.20, abs_tol=1e-4)
        assert len(engine.drawdown_tracker.equity_history) == 4

    def test_tracker_prime_direct_method(self):
        """Directly verify DrawdownDefenseTracker.prime() sets peak, history, and gate correctly."""
        tracker = DrawdownDefenseTracker()
        tracker.prime([1000.0, 1200.0, 1500.0, 1350.0])

        assert tracker.peak_equity == 1500.0
        assert tracker.current_equity == 1350.0
        assert math.isclose(tracker.current_drawdown, -0.10, abs_tol=1e-5)
        assert tracker.active_gate_multiplier == 0.20
        assert tracker.in_recovery_lockout is True
        assert tracker.equity_history == [1000.0, 1200.0, 1500.0, 1350.0]


# ============================================================================
# 3. SDE Sub-Penny Candlestick Floor Inversion Tests
# ============================================================================

class TestSDESubPennyCandlestickGeneration:
    """Verify that MertonJumpDiffusionSimulator.paths_to_bars safely generates sub-penny candlesticks."""

    def test_paths_to_bars_sub_penny_prices(self):
        """Generate bars with prices below $0.01 (e.g. 0.008, 0.005) and verify zero validation errors."""
        sim = MertonJumpDiffusionSimulator(seed=42)
        # Price path with sub-penny prices
        paths = {
            "PENNY": np.array([0.008, 0.005, 0.003, 0.001, 0.0005, 0.0001]),
        }
        start_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars_dict = sim.paths_to_bars(paths, ["PENNY"], start_date)

        bars = bars_dict["PENNY"]
        assert len(bars) == 5
        for b in bars:
            assert b.open > 0.0
            assert b.high > 0.0
            assert b.low > 0.0
            assert b.close > 0.0
            assert b.low <= min(b.open, b.close) + 1e-5, f"Low {b.low} > min(O, C) {min(b.open, b.close)}"
            assert b.high >= max(b.open, b.close) - 1e-5, f"High {b.high} < max(O, C) {max(b.open, b.close)}"
            assert b.high >= b.low, f"High {b.high} < Low {b.low}"
            assert b.vwap is not None and b.vwap > 0.0

    def test_paths_to_bars_cash_sub_penny_spread(self):
        """Verify cash equivalents at sub-penny prices maintain valid spreads and zero inversion."""
        sim = MertonJumpDiffusionSimulator(seed=99)
        paths = {
            "SHV": np.array([0.005, 0.0049, 0.0048]),
        }
        bars = sim.paths_to_bars(paths, ["SHV"], datetime(2026, 1, 1, tzinfo=timezone.utc))["SHV"]
        for b in bars:
            assert b.low <= min(b.open, b.close) + 1e-5
            assert b.high >= max(b.open, b.close) - 1e-5

    def test_paths_to_bars_extreme_crash_regime(self):
        """Simulate a 99.9% crash where stock goes from $100 -> $0.001 without candlestick inversion."""
        sim = MertonJumpDiffusionSimulator(seed=123)
        # Rapid logarithmic drop
        crash_steps = np.logspace(np.log10(100.0), np.log10(0.001), 50)
        paths = {"DOOM": crash_steps}
        bars = sim.paths_to_bars(paths, ["DOOM"], datetime(2026, 1, 1, tzinfo=timezone.utc))["DOOM"]

        assert len(bars) == 49
        for b in bars:
            assert b.low <= min(b.open, b.close) + 1e-5
            assert b.high >= max(b.open, b.close) - 1e-5
            assert b.low > 0.0


# ============================================================================
# 4. Volatility Scaling Fail-Safe Tests
# ============================================================================

class TestVolatilityScalingFailsafe:
    """Verify that volatility scaling is fail-safe on NaN, Inf, and corrupted inputs."""

    def test_nan_realized_vol_returns_zero_exposure(self):
        """Passing NaN realized volatility must return 0.0 (fail-safe capital preservation)."""
        scale = math_utils.volatility_scale_factor(float("nan"))
        assert scale == 0.0

        scale_computed = compute_volatility_scale_factor(float("nan"))
        assert scale_computed == 0.0

    def test_infinite_realized_vol_returns_zero_exposure(self):
        """Passing positive or negative infinity must return 0.0."""
        assert math_utils.volatility_scale_factor(float("inf")) == 0.0
        assert math_utils.volatility_scale_factor(float("-inf")) == 0.0
        assert compute_volatility_scale_factor(float("inf")) == 0.0
        assert compute_volatility_scale_factor(float("-inf")) == 0.0

    def test_valid_volatility_scaling_preserved(self):
        """Normal volatility inputs scale accurately."""
        # Exact match to target 12% -> 1.00
        assert math.isclose(math_utils.volatility_scale_factor(0.12), 1.0)
        # Double target vol 24% -> 0.50
        assert math.isclose(math_utils.volatility_scale_factor(0.24), 0.5)
        # Four times target vol 48% -> 0.25
        assert math.isclose(math_utils.volatility_scale_factor(0.48), 0.25)
        # Massive 500% vol -> 0.024
        assert math.isclose(math_utils.volatility_scale_factor(5.0), 0.12 / 5.0, rel_tol=1e-3)
        # Flat returns (0.0 vol) -> clamped by min_vol 0.05 -> 1.00
        assert math.isclose(math_utils.volatility_scale_factor(0.0), 1.0)


# ============================================================================
# 5. OrderIntent Consistency Validation Tests
# ============================================================================

class TestOrderIntentConsistencyValidation:
    """Verify that OrderIntent enforces action/side consistency and rejects contradictions."""

    def test_reject_action_buy_with_side_sell(self):
        """OrderIntent with action='BUY' and side=OrderSide.SELL must raise ValueError."""
        with pytest.raises(ValueError, match="Contradictory action"):
            OrderIntent(symbol="SPY", action="BUY", side=OrderSide.SELL)

    def test_reject_action_sell_with_side_buy(self):
        """OrderIntent with action='SELL' and side=OrderSide.BUY must raise ValueError."""
        with pytest.raises(ValueError, match="Contradictory action"):
            OrderIntent(symbol="SPY", action="SELL", side=OrderSide.BUY)

    def test_reject_action_hold_with_side_buy(self):
        """OrderIntent with action='HOLD' and side=OrderSide.BUY must raise ValueError."""
        with pytest.raises(ValueError, match="Contradictory action"):
            OrderIntent(symbol="SPY", action="HOLD", side=OrderSide.BUY)

    def test_accept_consistent_action_and_side(self):
        """Consistent pairs must instantiate cleanly."""
        buy_order = OrderIntent(symbol="SPY", action="BUY", side=OrderSide.BUY)
        assert buy_order.action == "BUY"
        assert buy_order.side == OrderSide.BUY

        sell_order = OrderIntent(symbol="QQQ", action="SELL", side=OrderSide.SELL)
        assert sell_order.action == "SELL"
        assert sell_order.side == OrderSide.SELL

        hold_order = OrderIntent(symbol="TLT", action="HOLD", side=OrderSide.HOLD)
        assert hold_order.action == "HOLD"
        assert hold_order.side == OrderSide.HOLD

    def test_automatic_synchronization_when_one_provided(self):
        """When only action or only side is provided, the other is automatically inferred."""
        order_from_act = OrderIntent(symbol="SPY", action="BUY")
        assert order_from_act.side == OrderSide.BUY

        order_from_side = OrderIntent(symbol="SPY", side=OrderSide.SELL)
        assert order_from_side.action == "SELL"


# ============================================================================
# 6. Mathematical & Economic Invariants Stress Tests
# ============================================================================

class TestMathematicalAndEconomicInvariants:
    """Stress-test quantitative invariants across black swan regimes and boundary conditions."""

    @pytest.mark.parametrize("regime", [
        MarketRegime.BULL_AGGRESSIVE,
        MarketRegime.BULL_NORMAL,
        MarketRegime.CORRECTION_FRAGILE,
        MarketRegime.BEAR_CRISIS,
        MarketRegime.STALE_DATA_HOLD,
    ])
    def test_sum_to_one_invariant_all_regimes(self, regime: MarketRegime):
        """Target weights must strictly sum to 1.0 +/- 1e-5 across every market regime."""
        weights = get_regime_base_weights(regime)
        total = sum(weights.values())
        assert math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5)
        for sym, w in weights.items():
            assert 0.0 <= w <= 1.0 + 1e-6

    def test_sum_to_one_1000_random_weight_permutations(self):
        """Fuzz normalize_target_weights across 1,000 random vectors, verifying sum=1.0 and non-negativity."""
        rng = np.random.default_rng(2026)
        symbols = ["SPY", "QQQ", "TLT", "GLD", "XLK", "SHV", "BIL"]

        for _ in range(1000):
            k = rng.integers(1, len(symbols) + 1)
            chosen = rng.choice(symbols, size=k, replace=False)
            raw = {sym: float(rng.uniform(-0.5, 5.0)) for sym in chosen}
            # Occasionally inject 0.0 or micro-weights
            if rng.random() < 0.2:
                raw["SPY"] = 1e-8

            normalized = normalize_target_weights(raw, cash_symbol="SHV")
            total = sum(normalized.values())
            assert math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5)
            for s, w in normalized.items():
                assert w >= 0.0
                assert w <= 1.0 + 1e-5

    def test_normalize_target_weights_nan_inf_protection(self):
        """Verify normalize_target_weights filters NaN/Inf without unhandled assertions."""
        raw = {"SPY": 0.5, "QQQ": float("nan"), "TLT": float("inf"), "SHV": 0.5}
        normalized = normalize_target_weights(raw, cash_symbol="SHV")
        assert math.isclose(sum(normalized.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5)
        assert "QQQ" not in normalized or normalized["QQQ"] == 0.0
        assert "TLT" not in normalized or normalized["TLT"] == 0.0

    def test_2022_duration_shock_strictly_zero_tlt(self):
        """During 2022 duration shock (P_TLT < SMA_200 and Mom_12-1 <= 0), TLT allocation must be 0.0%."""
        market_data = generate_2022_inflation_grind(seed=42)
        tlt_bars = market_data["TLT"]

        # Evaluate qualification on the full year
        is_qualified = evaluate_safe_haven_qualification("TLT", tlt_bars, sma_window=200)
        assert is_qualified is False, "TLT must be disqualified during 2022 duration shock"

        engine = SignalEngine()
        t_eval = tlt_bars[-1].timestamp
        signals = engine.compute_daily_signals(market_data, current_time=t_eval)
        alloc = engine.compute_target_weights(signals, market_data=market_data)

        # Invariant: TLT must have strictly 0.0% weight
        tlt_weight = alloc.weights.get("TLT", 0.0)
        assert tlt_weight == 0.0, f"TLT received {tlt_weight:.4f} during 2022 rate grind!"
        # Defensive capital must be in cash or gold
        assert alloc.weights.get("SHV", 0.0) + alloc.weights.get("BIL", 0.0) + alloc.weights.get("GLD", 0.0) > 0.5

    def test_90_percent_single_day_drop_triggers_circuit_breaker(self):
        """A catastrophic 90% single-day crash must trigger the ATR circuit breaker

        and cut equity risk exposure by at least 50% immediately.
        """
        clean_data = _build_synthetic_market_data(n_days=265, seed=123)
        t_eval = clean_data["SPY"][200].timestamp

        # Inject 90% single-day crash into SPY on day 201
        crash_dt = t_eval + timedelta(days=1)
        prev_close = clean_data["SPY"][200].close
        crash_close = prev_close * 0.10  # 90% drop!

        crash_data = {s: list(bars[:201]) for s, bars in clean_data.items()}
        crash_data["SPY"].append(_create_bar("SPY", crash_dt, close=crash_close, low_p=crash_close * 0.99))

        engine = SignalEngine()
        sig = engine.compute_daily_signals(crash_data, current_time=crash_dt)

        assert sig.circuit_breaker_active is True
        assert sig.regime == MarketRegime.BEAR_CRISIS

        alloc = engine.compute_target_weights(sig, market_data=crash_data)
        # In BEAR_CRISIS with circuit breaker active: equity is 0%
        equity_weight = sum(
            alloc.weights.get(s, 0.0) for s in ("SPY", "QQQ", "XLK", "XLY", "XLF")
        )
        assert equity_weight == 0.0
        assert math.isclose(alloc.cash_weight, 1.0, rel_tol=1e-4) or alloc.weights.get("GLD", 0.0) > 0.0

    def test_synthetic_black_swan_merton_regimes(self):
        """Simulate 2008 liquidity crisis and 2020 flash crash datasets and verify invariant compliance."""
        scenarios = [
            ("2008_liquidity_crisis", generate_2008_liquidity_crisis(seed=42)),
            ("2020_flash_crash", generate_2020_flash_crash(seed=42)),
        ]

        engine = SignalEngine()
        for name, data in scenarios:
            eval_indices = [60, 120, 180]
            for idx in eval_indices:
                if idx >= len(data["SPY"]):
                    continue
                t = data["SPY"][idx].timestamp
                sig = engine.compute_daily_signals(data, current_time=t)
                alloc = engine.compute_target_weights(sig, market_data=data)

                # Invariant 1: sum to 1.0 +/- 1e-5
                assert math.isclose(sum(alloc.weights.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5)
                # Invariant 2: all weights in [0.0, 1.0]
                for s, w in alloc.weights.items():
                    assert 0.0 <= w <= 1.0 + 1e-5
                # Invariant 3: valid regime
                assert isinstance(alloc.regime, MarketRegime)
