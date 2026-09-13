"""
tests/adversarial/test_m3_adversarial_signals.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adversarial empirical stress tests for Milestone M3 (Signal Indicators & Regime Detector):
1. Lookahead Bias Challenge:
   - Injects forward bars (t+1, t+5, t+50) into market data.
   - Asserts calculated daily signals, monthly momentum, and regime snapshots at t
     are bit-for-bit identical with and without future bars.
   - Tests standalone functions and allocator rules for potential forward leakage.
2. Extreme Volatility Stress:
   - 200% annual volatility, 500% vol, 0% vol, and extreme price shocks.
   - Asserts S_vol scales down gracefully without division by zero, NaN, or inf.
   - Asserts regime detector transitions strictly to BEAR_CRISIS.
3. Sudden Flash Crash (30% drop over 2 days):
   - Asserts dynamic ATR Keltner lower band stops trigger immediately on Day 1.
   - Asserts trailing drawdown defense gates trigger Level 3 (100% cash) on Day 2.
   - Asserts instant emergency transition to BEAR_CRISIS.
4. Stateful Recovery Hysteresis:
   - Asserts drawdown gate does not prematurely re-risk before 3 consecutive daily closes above 50 SMA.
   - Tests counter reset on Day 3 dip below 50 SMA.
   - Tests lockout release strictly upon 3rd consecutive qualifying close.
5. Numerical Boundaries & Initial Value Stress:
   - Initial equity scaling (e.g. NAV=1.0 or price < 100.0) in DrawdownDefenseTracker.
   - Zero variance / flat price handling.
   - Alphabetical tie-breaking determinism in momentum ranking.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Dict, List
import numpy as np
import pytest

from strategy_engine.core.models import Bar, MarketRegime, SignalSnapshot
from strategy_engine.core.universe import CORE_COMPOUNDERS, SECTOR_ETFS
from strategy_engine.signals.indicators import (
    DrawdownDefenseTracker,
    check_circuit_breaker,
    compute_atr,
    compute_keltner_lower_band,
    compute_market_breadth,
    compute_realized_volatility,
    compute_sma,
    compute_volatility_scale_factor,
    evaluate_drawdown_gate,
    evaluate_trend_filter,
    filter_bars_point_in_time,
)
from strategy_engine.signals.momentum import (
    calculate_12_1_momentum,
    calculate_momentum_12_1,
    compute_universe_momentum,
    evaluate_absolute_momentum,
    evaluate_safe_haven_dual_momentum,
    rank_momentum_candidates,
    select_top_growth_leaders,
    select_top_sectors,
)
from strategy_engine.signals.regime_detector import (
    RegimeDetector,
    SignalEngine,
)


def _make_bar(
    symbol: str,
    dt: datetime,
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_price: float | None = None,
    volume: int = 100_000,
) -> Bar:
    c = float(close)
    o = float(open_price if open_price is not None else c)
    h = float(high if high is not None else max(o, c) * 1.01)
    l = float(low if low is not None else min(o, c) * 0.99)
    return Bar(
        symbol=symbol,
        timestamp=dt,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=volume,
    )


def _generate_synthetic_history(
    n_days: int = 260,
    base_price: float = 500.0,
    daily_drift: float = 0.0005,
    daily_vol: float = 0.008,
    seed: int = 42,
    start_dt: datetime | None = None,
) -> Dict[str, List[Bar]]:
    """Generate realistic deterministic multi-asset market data."""
    rng = np.random.default_rng(seed)
    t0 = start_dt or datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    
    symbols = ["SPY", "QQQ", "TLT", "GLD", "SHV", "XLK", "XLF", "XLV", "XLE", "XLI"]
    data: Dict[str, List[Bar]] = {sym: [] for sym in symbols}

    for sym in symbols:
        price = base_price * (1.2 if sym == "QQQ" else 0.3 if "XL" in sym else 0.2 if sym in ("TLT", "GLD") else 1.0)
        for i in range(n_days):
            dt = t0 + timedelta(days=i)
            shock = rng.normal(daily_drift, daily_vol)
            if sym == "SHV":
                shock = 0.00015  # Risk free cash drift
            price = max(1.0, price * (1.0 + shock))
            h = price * (1.0 + abs(rng.normal(0, 0.004)))
            l = price * (1.0 - abs(rng.normal(0, 0.004)))
            data[sym].append(_make_bar(sym, dt, price, high=h, low=l, open_price=price))

    return data


# ============================================================================
# 1. Lookahead Bias Challenge
# ============================================================================

class TestLookaheadBiasChallenge:
    """Rigorous verification that forward bars never contaminate signals at timestamp t."""

    def test_daily_signals_bit_for_bit_identical_with_future_bars(self):
        """Inject forward bars (t+1 to t+50) with extreme spikes/crashes.
        Assert that calculated signals for timestamp t are bit-for-bit identical.
        """
        clean_data = _generate_synthetic_history(n_days=250, seed=100)
        t_eval = clean_data["SPY"][200].timestamp  # Evaluate at day 200

        # Create fresh engine and compute baseline snapshot
        engine_clean = SignalEngine()
        baseline_snap = engine_clean.compute_daily_signals(clean_data, current_time=t_eval)

        # Create contaminated dataset with forward bars injected (t+1 through t+50)
        # We also inject pathological price movements (e.g. +1000% jump and -90% crash)
        contaminated_data: Dict[str, List[Bar]] = {}
        for sym, bars in clean_data.items():
            # Include bars up to day 200
            subset = [b for b in bars if b.timestamp <= t_eval]
            # Add forward bars with extreme corruptions
            for step in range(1, 51):
                fwd_dt = t_eval + timedelta(days=step)
                corrupt_close = subset[-1].close * (10.0 if step % 2 == 0 else 0.1)
                subset.append(_make_bar(sym, fwd_dt, corrupt_close, high=corrupt_close * 1.5, low=corrupt_close * 0.5))
            contaminated_data[sym] = subset

        # Compute on contaminated dataset
        engine_future = SignalEngine()
        test_snap = engine_future.compute_daily_signals(contaminated_data, current_time=t_eval)

        # Bit-for-bit assertions across all primary scalar fields
        assert baseline_snap.timestamp == test_snap.timestamp
        assert baseline_snap.spy_price == test_snap.spy_price
        assert baseline_snap.spy_sma50 == test_snap.spy_sma50
        assert baseline_snap.spy_sma200 == test_snap.spy_sma200
        assert baseline_snap.realized_vol_20d == test_snap.realized_vol_20d
        assert baseline_snap.vol_scale_factor == test_snap.vol_scale_factor
        assert baseline_snap.drawdown_pct == test_snap.drawdown_pct
        assert baseline_snap.circuit_breaker_active == test_snap.circuit_breaker_active
        assert baseline_snap.regime == test_snap.regime

        # Assert all 22 indicators match bit-for-bit
        assert set(baseline_snap.indicators.keys()) == set(test_snap.indicators.keys())
        for k in baseline_snap.indicators:
            assert baseline_snap.indicators[k] == test_snap.indicators[k], (
                f"Indicator {k} mismatch: clean={baseline_snap.indicators[k]} vs contaminated={test_snap.indicators[k]}"
            )

    def test_monthly_momentum_point_in_time_filtering(self):
        """Assert compute_monthly_momentum ignores future bars at t+1 and t+5."""
        clean_data = _generate_synthetic_history(n_days=270, seed=200)
        t_eval = clean_data["SPY"][255].timestamp

        engine = SignalEngine()
        clean_scores = engine.compute_monthly_momentum(clean_data, current_time=t_eval)

        # Contaminate data with forward bars
        contaminated_data = {sym: list(bars) for sym, bars in clean_data.items()}
        for sym in contaminated_data:
            last_dt = clean_data[sym][-1].timestamp
            for step in range(1, 10):
                fwd_dt = last_dt + timedelta(days=step)
                contaminated_data[sym].append(_make_bar(sym, fwd_dt, 99999.0))

        test_scores = engine.compute_monthly_momentum(contaminated_data, current_time=t_eval)
        for sym in clean_scores:
            assert math.isclose(clean_scores[sym], test_scores[sym], rel_tol=1e-12), (
                f"Lookahead leak in momentum for {sym}: {clean_scores[sym]} != {test_scores[sym]}"
            )

    def test_filter_bars_point_in_time_unsorted_and_boundary(self):
        """Verify PIT filtration handles unsorted inputs and microsecond boundaries."""
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        # Create bars deliberately shuffled out of order
        bars = [
            _make_bar("SPY", t0 + timedelta(days=5), 105.0),
            _make_bar("SPY", t0 + timedelta(days=1), 101.0),
            _make_bar("SPY", t0, 100.0),
            _make_bar("SPY", t0 + timedelta(days=3), 103.0),
            _make_bar("SPY", t0 + timedelta(days=2), 102.0),
            _make_bar("SPY", t0 + timedelta(days=4), 104.0),
        ]
        t_eval = t0 + timedelta(days=3)
        filtered = filter_bars_point_in_time(bars, t_eval)

        assert len(filtered) == 4
        # Verify chronological order
        assert [b.close for b in filtered] == [100.0, 101.0, 102.0, 103.0]
        assert filtered[-1].timestamp == t_eval

        # Microsecond boundary check: t_eval + 1 microsecond must not be included
        bars_micro = [
            _make_bar("SPY", t_eval, 100.0),
            _make_bar("SPY", t_eval + timedelta(microseconds=1), 999.0),
        ]
        filtered_micro = filter_bars_point_in_time(bars_micro, t_eval)
        assert len(filtered_micro) == 1
        assert filtered_micro[0].close == 100.0

    def test_filter_bars_point_in_time_naive_vs_aware_tz(self):
        """Vulnerability test: AlpacaRelay data feeds have timezone-aware UTC timestamps.
        If current_time is passed as timezone-naive datetime, filter_bars_point_in_time
        crashes with TypeError: can't compare offset-naive and offset-aware datetimes.
        """
        t_aware = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        bars = [_make_bar("SPY", t_aware, 100.0)]
        t_naive = datetime(2026, 1, 1, 12, 0)  # Naive datetime
        # Expect filter_bars_point_in_time to handle or normalize timezone gracefully
        try:
            filtered = filter_bars_point_in_time(bars, t_naive)
            assert len(filtered) == 1
        except TypeError as e:
            pytest.fail(f"Crash on naive datetime comparison: {e}")


    def test_target_weights_allocator_market_data_leakage(self):
        """Vulnerability test: Check whether compute_target_weights or select_top_sectors
        filters market_data point-in-time when passed contaminated market_data.
        If market_data has future bars (>21 days ahead), does target allocation shift due to future data?
        """
        clean_data = _generate_synthetic_history(n_days=265, seed=300)
        # Add XLU as a baseline defensive sector
        clean_data["XLU"] = [_make_bar("XLU", b.timestamp, 80.0) for b in clean_data["XLV"]]
        t_eval = clean_data["SPY"][255].timestamp

        engine = SignalEngine()
        snap = engine.compute_daily_signals(clean_data, current_time=t_eval)

        # Compute target weights with clean data (filtered up to t_eval)
        clean_pit_data = {sym: filter_bars_point_in_time(bars, t_eval) for sym, bars in clean_data.items()}
        weights_clean = engine.compute_target_weights(snap, market_data=clean_pit_data)

        # Contaminate market_data with forward bars on XLU (+5000 surge at t+1..t+35)
        contaminated_data = {sym: list(bars) for sym, bars in clean_pit_data.items()}
        for step in range(1, 36):
            fwd_dt = t_eval + timedelta(days=step)
            # Massive surge on XLU in future extending beyond 21-day skip window
            contaminated_data["XLU"].append(_make_bar("XLU", fwd_dt, 5000.0, high=5010.0, low=4990.0, open_price=5000.0))

        # Call compute_target_weights with contaminated_data (as an end-user might pass market_data)
        weights_contaminated = engine.compute_target_weights(snap, market_data=contaminated_data)

        # Assert zero lookahead bias: clean and contaminated weights must be identical
        assert weights_clean.weights == weights_contaminated.weights, (
            f"LOOKAHEAD LEAK DETECTED in compute_target_weights: "
            f"clean={weights_clean.weights} vs contaminated={weights_contaminated.weights}"
        )





# ============================================================================
# 2. Extreme Volatility Stress Challenge (200% vol & beyond)
# ============================================================================

class TestExtremeVolatilityStress:
    """Stress test realized vol and S_vol under pathological volatility regimes."""

    def test_vol_scale_factor_at_200_pct_vol(self):
        """At 200% annual volatility (sigma=2.0), S_vol must scale down gracefully to 0.06."""
        # Target 12%, min 5%
        scale = compute_volatility_scale_factor(2.0, target_vol=0.12, min_vol=0.05, max_scale=1.0)
        expected = 0.12 / 2.0  # 0.06
        assert math.isclose(scale, expected, rel_tol=1e-5)
        assert 0.0 < scale <= 1.0

    @pytest.mark.parametrize("extreme_vol,expected_scale", [
        (1.0, 0.12),      # 100% vol -> 0.12
        (2.0, 0.06),      # 200% vol -> 0.06
        (5.0, 0.024),     # 500% vol -> 0.024
        (12.0, 0.010),    # 1200% hyper-vol -> 0.010
        (100.0, 0.0012),  # 10000% crypto-meltdown -> 0.0012
    ])
    def test_vol_scale_factor_hyper_volatility(self, extreme_vol: float, expected_scale: float):
        """S_vol must remain strictly finite, positive, and non-NaN under extreme volatility."""
        scale = compute_volatility_scale_factor(extreme_vol, target_vol=0.12, min_vol=0.05, max_scale=1.0)
        assert not math.isnan(scale)
        assert not math.isinf(scale)
        assert math.isclose(scale, expected_scale, rel_tol=1e-4)

    def test_vol_scale_factor_zero_and_negative_vol(self):
        """At 0% or negative volatility, S_vol must clamp to min_vol (0.05) and cap at max_scale (1.0)."""
        scale_zero = compute_volatility_scale_factor(0.0, target_vol=0.12, min_vol=0.05, max_scale=1.0)
        assert scale_zero == 1.0  # 0.12 / 0.05 = 2.40 -> clamped to 1.0

        scale_neg = compute_volatility_scale_factor(-0.50, target_vol=0.12, min_vol=0.05, max_scale=1.0)
        assert scale_neg == 1.0

    def test_realized_volatility_pathological_price_swings(self):
        """Alternating price series simulating 200%+ realized vol produces finite, valid float."""
        prices = [100.0]
        for i in range(25):
            # Alternating +/- 12% daily swings (annual vol ~190%)
            price = prices[-1] * (1.12 if i % 2 == 0 else 0.88)
            prices.append(price)

        vol = compute_realized_volatility(prices, window=20, annualization_factor=252)
        assert vol > 1.50  # Above 150% annual vol
        assert not math.isnan(vol)
        assert not math.isinf(vol)

        scale = compute_volatility_scale_factor(vol)
        assert scale < 0.10
        assert scale > 0.0

    def test_regime_detector_200_pct_vol_triggers_bear_crisis(self):
        """Realized volatility of 200% must trigger BEAR_CRISIS override in RegimeDetector."""
        detector = RegimeDetector()
        bull_trend = {
            "price": 500.0, "sma50": 490.0, "sma200": 470.0,
            "price_above_sma50": True, "price_above_sma200": True,
            "is_golden_cross": True, "trend_score": 1.0,
        }

        regime, reason = detector.classify(
            spy_trend=bull_trend,
            qqq_trend=bull_trend,
            realized_vol_20d=2.0,  # 200% vol!
            circuit_breaker_active=False,
            drawdown_pct=-0.01,
            breadth_50=0.80,
        )
        assert regime == MarketRegime.BEAR_CRISIS
        assert "Volatility spike" in reason


# ============================================================================
# 3. Sudden Flash Crash Challenge (30% drop in 2 days)
# ============================================================================

class TestSuddenFlashCrash:
    """Stress test response to a sudden 30% crash over 2 trading days."""

    def test_flash_crash_day1_triggers_circuit_breaker(self):
        """Day 1 (-15% drop): SPY breaches dynamic ATR Keltner band immediately."""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # 60 calm days with SPY at 500.0, High=502, Low=498 (daily range ~4)
        bars = [
            _make_bar("SPY", t0 + timedelta(days=i), 500.0, high=502.0, low=498.0, open_price=500.0)
            for i in range(60)
        ]

        # Verify baseline calm state: ATR ~4.0, lower band = 500 - 2*4 = 492
        sma, atr, lower_band = compute_keltner_lower_band(bars, sma_window=50, atr_window=14, atr_multiplier=2.0)
        assert math.isclose(sma, 500.0, abs_tol=1e-3)
        assert math.isclose(atr, 4.0, abs_tol=1e-3)
        assert math.isclose(lower_band, 492.0, abs_tol=1e-3)

        # Day 61: Crash Day 1 (-15% drop from 500 to 425)
        dt_day1 = t0 + timedelta(days=61)
        bars.append(_make_bar("SPY", dt_day1, 425.0, high=500.0, low=420.0, open_price=500.0))

        is_cb_active, metrics = check_circuit_breaker(bars, sma_window=50, atr_window=14, atr_multiplier=2.0)
        assert is_cb_active is True
        assert metrics["close"] == 425.0
        assert metrics["close"] < metrics["lower_band"]

        # Assert RegimeDetector transitions immediately to BEAR_CRISIS
        detector = RegimeDetector()
        spy_trend = evaluate_trend_filter(bars, "SPY")
        regime, reason = detector.classify(
            spy_trend=spy_trend,
            qqq_trend=spy_trend,
            realized_vol_20d=0.15,
            circuit_breaker_active=is_cb_active,
            drawdown_pct=-0.15,
            breadth_50=0.50,
        )
        assert regime == MarketRegime.BEAR_CRISIS
        assert "Circuit breaker" in reason

    def test_flash_crash_day2_triggers_drawdown_gate_level3(self):
        """Day 2 (-30% cumulative drop from 500 to 350): Drawdown Gate L3 enforces 0.00 multiplier."""
        tracker = DrawdownDefenseTracker()

        # Day 0: Baseline peak at 500.0
        gate_d0 = tracker.update(current_equity=500.0, benchmark_price=500.0, benchmark_sma50=500.0)
        assert gate_d0 == 1.00
        assert tracker.peak_equity == 500.0

        # Day 1: Crash to 425.0 (-15% DD) -> Level 3 threshold reached (-15%)
        gate_d1 = tracker.update(current_equity=425.0, benchmark_price=425.0, benchmark_sma50=498.5)
        assert tracker.current_drawdown == -0.15
        assert gate_d1 == 0.00  # 100% Cash liquidation
        assert tracker.in_recovery_lockout is True

        # Day 2: Crash to 350.0 (-30% DD)
        gate_d2 = tracker.update(current_equity=350.0, benchmark_price=350.0, benchmark_sma50=495.0)
        assert math.isclose(tracker.current_drawdown, -0.30, abs_tol=1e-3)
        assert gate_d2 == 0.00  # Stays at 0.00 (100% Cash liquidation)
        assert tracker.in_recovery_lockout is True


# ============================================================================
# 4. Stateful Recovery Hysteresis Challenge
# ============================================================================

class TestStatefulRecoveryHysteresis:
    """Stress test 3-day recovery hysteresis confirmation and counter resets."""

    def test_recovery_hysteresis_strict_3_consecutive_days(self):
        """Drawdown gate MUST NOT re-risk before 3 consecutive daily closes above 50 SMA."""
        tracker = DrawdownDefenseTracker()

        # 1. Establish peak at 500.0
        tracker.update(current_equity=500.0, benchmark_price=500.0, benchmark_sma50=480.0)

        # 2. Trigger Level 2 drawdown (-12% DD to 440.0)
        gate_crash = tracker.update(current_equity=440.0, benchmark_price=440.0, benchmark_sma50=480.0)
        assert math.isclose(tracker.current_drawdown, -0.12, abs_tol=1e-3)
        assert gate_crash == 0.20
        assert tracker.in_recovery_lockout is True
        assert tracker.consecutive_recovery_days == 0

        # 3. Benchmark crosses above SMA50 (485 > 480) for Day 1
        # Equity recovers slightly to 480.0 (-4% DD -> raw gate would be 1.00!)
        gate_day1 = tracker.update(current_equity=480.0, benchmark_price=485.0, benchmark_sma50=480.0)
        assert tracker.consecutive_recovery_days == 1
        # Lockout must clamp multiplier to previous minimum (0.20), blocking re-risk to 1.00!
        assert gate_day1 == 0.20
        assert tracker.in_recovery_lockout is True

        # 4. Day 2 above SMA50 (487 > 480)
        gate_day2 = tracker.update(current_equity=485.0, benchmark_price=487.0, benchmark_sma50=480.0)
        assert tracker.consecutive_recovery_days == 2
        assert gate_day2 == 0.20
        assert tracker.in_recovery_lockout is True

        # 5. Day 3 DIP: Benchmark closes below SMA50 (478 <= 480)
        # COUNTER MUST RESET TO ZERO!
        gate_dip = tracker.update(current_equity=482.0, benchmark_price=478.0, benchmark_sma50=480.0)
        assert tracker.consecutive_recovery_days == 0
        assert gate_dip == 0.20
        assert tracker.in_recovery_lockout is True

        # 6. Fresh recovery attempt: Day 1 and Day 2
        tracker.update(current_equity=485.0, benchmark_price=485.0, benchmark_sma50=480.0)
        assert tracker.consecutive_recovery_days == 1
        tracker.update(current_equity=488.0, benchmark_price=489.0, benchmark_sma50=480.0)
        assert tracker.consecutive_recovery_days == 2
        assert tracker.active_gate_multiplier == 0.20

        # 7. Day 3 confirms: 3rd consecutive close above SMA50!
        gate_unlocked = tracker.update(current_equity=490.0, benchmark_price=492.0, benchmark_sma50=480.0)
        assert tracker.consecutive_recovery_days == 3
        # At 490.0 vs 500 peak, DD is -2.0% (> -5%), so gate re-risks fully to 1.00!
        assert gate_unlocked == 1.00
        assert tracker.in_recovery_lockout is False


# ============================================================================
# 5. Numerical Boundaries & Adversarial Edge Cases
# ============================================================================

class TestNumericalBoundariesAndEdgeCases:
    """Stress test boundary cases: flat series, normalized initial equity, tie-breaking."""

    def test_drawdown_tracker_initial_equity_below_100(self):
        """Vulnerability test: DrawdownDefenseTracker has peak_equity default = 100.0.
        If current_equity starts at 1.0 (normalized NAV) or 50.0 (sub-100 stock),
        verify whether tracker incorrectly calculates a -50% to -99% drawdown!
        """
        tracker = DrawdownDefenseTracker()
        # Suppose a user passes initial normalized equity of 1.0
        gate = tracker.update(current_equity=1.0, benchmark_price=50.0, benchmark_sma50=45.0)

        # Mathematical expectation: On initial valuation, drawdown from inception should be 0.0%!
        # If peak_equity was hardcoded to 100.0, raw_dd = (1.0 - 100.0)/100.0 = -0.99 (-99%)!
        assert tracker.current_drawdown == 0.0, (
            f"FLAW DETECTED: DrawdownDefenseTracker initialized with hardcoded peak_equity=100.0. "
            f"When current_equity=1.0, computed drawdown is {tracker.current_drawdown:.1%}, triggering gate={gate}!"
        )


    def test_drawdown_tracker_uninitialized_first_call_handling(self):
        """Verify that when peak_equity is explicitly initialized or when current_equity > 100, it works."""
        # When current_equity > 100.0, peak_equity updates to current_equity immediately:
        tracker = DrawdownDefenseTracker(peak_equity=1.0, current_equity=1.0)
        gate = tracker.update(current_equity=1.0, benchmark_price=500.0, benchmark_sma50=490.0)
        assert tracker.current_drawdown == 0.0
        assert gate == 1.00

    def test_momentum_alphabetical_tie_breaking_strictly_deterministic(self):
        """Assets with identical momentum score must break ties deterministically by ticker alphabetically."""
        scores = {
            "XLV": 0.123456,
            "XLF": 0.123456,
            "XLE": 0.123456,
            "XLK": 0.123456,
        }
        ranked = rank_momentum_candidates(scores)
        # Expected alphabetical ordering: XLE, XLF, XLK, XLV
        symbols = [sym for sym, _ in ranked]
        assert symbols == ["XLE", "XLF", "XLK", "XLV"]

    def test_zero_variance_flat_data_stability(self):
        """Completely flat price history does not produce NaN in indicators or regime detector."""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        flat_bars = [
            _make_bar("SPY", t0 + timedelta(days=i), 100.0, high=100.0, low=100.0, open_price=100.0)
            for i in range(210)
        ]
        market_data = {"SPY": flat_bars}

        engine = SignalEngine()
        snap = engine.compute_daily_signals(market_data, current_time=t0 + timedelta(days=205))

        assert snap.realized_vol_20d == 0.0
        assert snap.vol_scale_factor == 1.0
        assert not math.isnan(snap.vol_scale_factor)
        assert not math.isnan(snap.spy_sma50)
        assert snap.circuit_breaker_active is False
