"""
tests/adversarial/test_m3_challenger_stress.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial empirical challenge harness for Milestone M3 Iteration 2:
1. Normalized NAV Curves:
   - V_0 in {1.0, 0.5, 10.0, 0.01, 1_000_000.0}
   - Verify zero false -99% drawdown upon initialization
   - Verify drawdown gates (L1, L2, L3) trigger strictly upon genuine peak drop
   - Verify 3-day recovery hysteresis under normalized NAV
   - Verify sequential multi-day tracking via SignalEngine.compute_daily_signals
2. Forward Bar Leakage:
   - Inject future bars (t+1 through t+30 and t+50) with extreme regime-altering data
   - Verify SignalEngine.compute_target_weights produces identical weights, cash, and rationale
   - Verify compute_daily_signals and compute_monthly_momentum are strictly invariant
3. Timezone Mixing:
   - Cross-matrix of UTC aware, US/Eastern aware, timezone-naive, and custom offset
   - Mixed timestamps within the same bar list
   - Verify zero TypeErrors and exact chronological boundary filtering
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Dict, List
import zoneinfo
import numpy as np
import pytest

from strategy_engine.core.models import Bar, MarketRegime, SignalSnapshot
from strategy_engine.signals.indicators import (
    DrawdownDefenseTracker,
    evaluate_drawdown_gate,
    filter_bars_point_in_time,
)
from strategy_engine.signals.regime_detector import SignalEngine


def _create_bar(
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
    h = float(high if high is not None else max(o, c) * 1.005)
    l = float(low if low is not None else min(o, c) * 0.995)
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
    n_days: int = 260,
    seed: int = 777,
    start_dt: datetime | None = None,
) -> Dict[str, List[Bar]]:
    rng = np.random.default_rng(seed)
    t0 = start_dt or datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    symbols = [
        "SPY", "QQQ", "TLT", "GLD", "SHV", "BIL",
        "XLK", "XLF", "XLV", "XLE", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC"
    ]
    data: Dict[str, List[Bar]] = {sym: [] for sym in symbols}
    base_prices = {
        "SPY": 500.0, "QQQ": 450.0, "TLT": 95.0, "GLD": 180.0, "SHV": 110.0, "BIL": 91.0,
        "XLK": 200.0, "XLF": 40.0, "XLV": 140.0, "XLE": 85.0, "XLI": 120.0,
        "XLU": 65.0, "XLP": 75.0, "XLY": 175.0, "XLB": 85.0, "XLRE": 38.0, "XLC": 80.0,
    }

    for sym in symbols:
        price = base_prices[sym]
        for i in range(n_days):
            dt = t0 + timedelta(days=i)
            drift = 0.0003 if sym not in ("SHV", "BIL") else 0.0001
            vol = 0.01 if sym not in ("SHV", "BIL") else 0.0001
            ret = rng.normal(drift, vol)
            price = max(0.1, price * (1.0 + ret))
            h = price * (1.0 + abs(rng.normal(0, 0.003)))
            l = price * (1.0 - abs(rng.normal(0, 0.003)))
            data[sym].append(_create_bar(sym, dt, price, high=h, low=l, open_price=price))

    return data


# ============================================================================
# 1. Normalized NAV Curves Stress Test
# ============================================================================

class TestNormalizedNAVCurves:
    """Empirical challenge: DrawdownDefenseTracker behavior across arbitrary initial NAVs."""

    @pytest.mark.parametrize("v0", [1.0, 0.5, 10.0, 0.01, 100.0, 1_000_000.0])
    def test_initial_nav_no_false_drawdown(self, v0: float):
        """At day 0, uninitialized DrawdownDefenseTracker must record 0.0 drawdown, gate=1.00."""
        tracker = DrawdownDefenseTracker()
        gate = tracker.update(
            current_equity=v0,
            benchmark_price=500.0,
            benchmark_sma50=490.0,
        )
        assert tracker.peak_equity == v0
        assert tracker.current_drawdown == 0.0
        assert gate == 1.00
        assert tracker.in_recovery_lockout is False
        assert tracker.consecutive_recovery_days == 1  # 500 > 490

    @pytest.mark.parametrize("v0", [1.0, 0.5, 10.0])
    def test_normalized_nav_drawdown_defense_triggers_only_on_genuine_drop(self, v0: float):
        """Drawdown defense triggers must fire ONLY upon genuine peak drops regardless of scale."""
        tracker = DrawdownDefenseTracker()

        # 1. Initializing at V_0
        g0 = tracker.update(v0, benchmark_price=500.0, benchmark_sma50=490.0)
        assert g0 == 1.00
        assert tracker.current_drawdown == 0.0

        # 2. NAV compounds up: peak rises to 1.50 * V_0
        v_peak = v0 * 1.50
        g_peak = tracker.update(v_peak, benchmark_price=520.0, benchmark_sma50=500.0)
        assert g_peak == 1.00
        assert tracker.peak_equity == v_peak
        assert tracker.current_drawdown == 0.0

        # 3. Small pull back -3%: equity = 1.455 * V_0 (DD = -3.0%)
        # Should NOT trigger drawdown defense (gate remains 1.00)
        v_small_drop = v_peak * (1.0 - 0.03)
        g_small = tracker.update(v_small_drop, benchmark_price=515.0, benchmark_sma50=505.0)
        assert math.isclose(tracker.current_drawdown, -0.03, abs_tol=1e-5)
        assert g_small == 1.00
        assert tracker.in_recovery_lockout is False

        # 4. Genuine Level 1 drop -7%: equity = v_peak * (1 - 0.07)
        # Gate must drop to 0.50 (Level 1: -10% < DD <= -5%)
        v_l1 = v_peak * (1.0 - 0.07)
        g_l1 = tracker.update(v_l1, benchmark_price=490.0, benchmark_sma50=505.0)
        assert math.isclose(tracker.current_drawdown, -0.07, abs_tol=1e-5)
        assert g_l1 == 0.50
        assert tracker.in_recovery_lockout is True

        # 5. Genuine Level 2 drop -12%: equity = v_peak * (1 - 0.12)
        # Gate must drop to 0.20 (Level 2: -15% < DD <= -10%)
        v_l2 = v_peak * (1.0 - 0.12)
        g_l2 = tracker.update(v_l2, benchmark_price=470.0, benchmark_sma50=505.0)
        assert math.isclose(tracker.current_drawdown, -0.12, abs_tol=1e-5)
        assert g_l2 == 0.20
        assert tracker.in_recovery_lockout is True

        # 6. Genuine Level 3 drop -20%: equity = v_peak * (1 - 0.20)
        # Gate must drop to 0.00 (Level 3: DD <= -15%)
        v_l3 = v_peak * (1.0 - 0.20)
        g_l3 = tracker.update(v_l3, benchmark_price=440.0, benchmark_sma50=505.0)
        assert math.isclose(tracker.current_drawdown, -0.20, abs_tol=1e-5)
        assert g_l3 == 0.00
        assert tracker.in_recovery_lockout is True

    @pytest.mark.parametrize("v0", [1.0, 0.5, 10.0])
    def test_normalized_nav_recovery_hysteresis(self, v0: float):
        """Recovery hysteresis under normalized NAV requires 3 consecutive closes above SMA50."""
        tracker = DrawdownDefenseTracker()
        tracker.update(v0, benchmark_price=500.0, benchmark_sma50=490.0)

        # Trigger Level 2 drop (-12%)
        v_l2 = v0 * 0.88
        tracker.update(v_l2, benchmark_price=460.0, benchmark_sma50=490.0)
        assert tracker.active_gate_multiplier == 0.20
        assert tracker.in_recovery_lockout is True

        # Day 1 of recovery: NAV recovers to -3% DD (v0 * 0.97), price > SMA50
        # But lockout is active, so gate remains clamped to 0.20!
        g1 = tracker.update(v0 * 0.97, benchmark_price=495.0, benchmark_sma50=490.0)
        assert tracker.consecutive_recovery_days == 1
        assert g1 == 0.20
        assert tracker.in_recovery_lockout is True

        # Day 2: price > SMA50
        g2 = tracker.update(v0 * 0.98, benchmark_price=496.0, benchmark_sma50=490.0)
        assert tracker.consecutive_recovery_days == 2
        assert g2 == 0.20

        # Day 3 dip: price <= SMA50 (489 <= 490) -> Counter resets!
        g_dip = tracker.update(v0 * 0.98, benchmark_price=489.0, benchmark_sma50=490.0)
        assert tracker.consecutive_recovery_days == 0
        assert g_dip == 0.20
        assert tracker.in_recovery_lockout is True

        # Fresh attempt: 3 consecutive days above SMA50
        tracker.update(v0 * 0.98, benchmark_price=492.0, benchmark_sma50=490.0)
        tracker.update(v0 * 0.98, benchmark_price=493.0, benchmark_sma50=490.0)
        g_rec = tracker.update(v0 * 0.99, benchmark_price=495.0, benchmark_sma50=490.0)
        assert tracker.consecutive_recovery_days == 3
        assert g_rec == 1.00
        assert tracker.in_recovery_lockout is False

    @pytest.mark.parametrize("v0", [1.0, 0.5, 10.0])
    def test_signal_engine_portfolio_equity_curve_sequential_normalized(self, v0: float):
        """SignalEngine.compute_daily_signals in sequential production workflow tracks normalized NAV."""
        market_data = _build_synthetic_market_data(n_days=250, seed=123)

        engine = SignalEngine()
        eq_trajectory = [v0, v0 * 1.05, v0 * 1.10, v0 * 1.15, v0 * 1.08]

        snaps = []
        for i, current_val in enumerate(eq_trajectory):
            t_eval = market_data["SPY"][200 + i].timestamp
            snap = engine.compute_daily_signals(
                market_data=market_data,
                current_time=t_eval,
                portfolio_equity_curve=eq_trajectory[: i + 1],
            )
            snaps.append(snap)

        # On the final day (1.08 drop from peak 1.15), drawdown should be ~ -6.087%
        final_snap = snaps[-1]
        expected_dd = (v0 * 1.08 - v0 * 1.15) / (v0 * 1.15)
        assert math.isclose(final_snap.drawdown_pct, expected_dd, rel_tol=1e-4)
        assert final_snap.indicators["drawdown_gate"] == 0.50
        assert final_snap.indicators["in_recovery_lockout"] == 1.0


# ============================================================================
# 2. Forward Bar Leakage Stress Test (t+1, t+30)
# ============================================================================

class TestForwardBarLeakage:
    """Empirical challenge: Injecting future bars (t+1, t+30) must NOT alter compute_target_weights."""

    def test_compute_target_weights_immune_to_forward_bars_t1_and_t30(self):
        """Inject contaminated future bars at t+1 and t+30 with massive surges/crashes.
        Verify that compute_target_weights outputs bit-for-bit identical allocations.
        """
        clean_data = _build_synthetic_market_data(n_days=265, seed=456)
        t_eval = clean_data["SPY"][225].timestamp

        engine = SignalEngine()
        clean_snap = engine.compute_daily_signals(clean_data, current_time=t_eval)

        # Compute target allocation on clean dataset
        clean_alloc = engine.compute_target_weights(clean_snap, market_data=clean_data)

        # Construct contaminated dataset with future bars (t+1 through t+45)
        contaminated_data: Dict[str, List[Bar]] = {}
        for sym, bars in clean_data.items():
            bars_to_eval = [b for b in bars if b.timestamp <= t_eval]
            last_close = bars_to_eval[-1].close
            # Inject t+1 through t+45 future bars
            for day_offset in range(1, 46):
                fwd_dt = t_eval + timedelta(days=day_offset)
                # For defensive assets (TLT, GLD, SHV), create +500% spike
                # For equities (SPY, QQQ, XLK), create -90% crash
                multiplier = 5.0 if sym in ("TLT", "GLD", "SHV", "BIL", "XLU") else 0.10
                fwd_close = last_close * multiplier
                bars_to_eval.append(
                    _create_bar(
                        symbol=sym,
                        dt=fwd_dt,
                        close=fwd_close,
                        high=fwd_close * 1.05,
                        low=fwd_close * 0.95,
                        open_price=fwd_close,
                    )
                )
            contaminated_data[sym] = bars_to_eval

        # Re-compute signals on contaminated data at t_eval
        test_snap = engine.compute_daily_signals(contaminated_data, current_time=t_eval)
        assert clean_snap == test_snap, "compute_daily_signals leaked future bars!"

        # Call compute_target_weights with contaminated dataset
        contaminated_alloc = engine.compute_target_weights(test_snap, market_data=contaminated_data)

        # 1. Weights dictionary must match bit-for-bit
        assert clean_alloc.weights.keys() == contaminated_alloc.weights.keys()
        for k in clean_alloc.weights:
            assert math.isclose(clean_alloc.weights[k], contaminated_alloc.weights[k], abs_tol=1e-12), (
                f"Leak in weight for {k}: clean={clean_alloc.weights[k]} vs contaminated={contaminated_alloc.weights[k]}"
            )

        # 2. Cash weight, regime, rationale must match bit-for-bit
        assert math.isclose(clean_alloc.cash_weight, contaminated_alloc.cash_weight, abs_tol=1e-12)
        assert clean_alloc.regime == contaminated_alloc.regime
        assert clean_alloc.rationale == contaminated_alloc.rationale
        assert clean_alloc.timestamp == contaminated_alloc.timestamp

    def test_compute_monthly_momentum_immune_to_future_bars(self):
        """Monthly momentum scores must remain invariant under future bar injection."""
        clean_data = _build_synthetic_market_data(n_days=270, seed=888)
        t_eval = clean_data["SPY"][255].timestamp

        engine = SignalEngine()
        clean_mom = engine.compute_monthly_momentum(clean_data, current_time=t_eval)

        # Contaminate future bars
        contaminated = {sym: list(bars) for sym, bars in clean_data.items()}
        for sym in contaminated:
            last_dt = clean_data[sym][-1].timestamp
            for step in range(1, 35):
                contaminated[sym].append(_create_bar(sym, last_dt + timedelta(days=step), 9999.0))

        test_mom = engine.compute_monthly_momentum(contaminated, current_time=t_eval)
        for sym in clean_mom:
            assert math.isclose(clean_mom[sym], test_mom[sym], rel_tol=1e-12)


# ============================================================================
# 3. Timezone Mixing Stress Test
# ============================================================================

class TestTimezoneMixingRobustness:
    """Empirical challenge: filter_bars_point_in_time across all timezone permutations."""

    @pytest.fixture
    def eastern_tz(self):
        return zoneinfo.ZoneInfo("US/Eastern")

    @pytest.fixture
    def utc_tz(self):
        return timezone.utc

    def test_timezone_matrix_zero_type_errors(self, eastern_tz, utc_tz):
        """Exhaustive matrix of (current_time tz) x (bar tz) must never raise TypeError."""
        t_base = datetime(2026, 6, 1, 14, 0, 0)  # 14:00 naive = 10:00 EDT = 14:00 UTC
        custom_tz = timezone(timedelta(hours=-5))

        time_variants = [
            ("utc", t_base.replace(tzinfo=utc_tz)),
            ("eastern", t_base.replace(tzinfo=eastern_tz)),
            ("naive", t_base),
            ("custom", t_base.replace(tzinfo=custom_tz)),
        ]

        bar_tz_variants = [
            ("utc", utc_tz),
            ("eastern", eastern_tz),
            ("naive", None),
            ("custom", custom_tz),
        ]

        for ct_name, ct_val in time_variants:
            for b_name, b_tz in bar_tz_variants:
                dt_bar = t_base.replace(tzinfo=b_tz) if b_tz is not None else t_base
                bars = [
                    _create_bar("SPY", dt_bar - timedelta(hours=2), 500.0),
                    _create_bar("SPY", dt_bar, 505.0),
                    _create_bar("SPY", dt_bar + timedelta(hours=2), 510.0),
                ]
                try:
                    filtered = filter_bars_point_in_time(bars, ct_val)
                    assert isinstance(filtered, list)
                except TypeError as e:
                    pytest.fail(f"TypeError raised for current_time={ct_name} and bar_tz={b_name}: {e}")

    def test_mixed_timezones_in_same_bar_list(self, eastern_tz, utc_tz):
        """Single list containing a mixture of UTC, Eastern, and naive bars."""
        # 14:00 UTC == 10:00 Eastern
        t_eval_utc = datetime(2026, 6, 1, 14, 0, 0, tzinfo=utc_tz)

        bars = [
            _create_bar("SPY", datetime(2026, 6, 1, 13, 0, 0, tzinfo=utc_tz), 100.0),         # 13:00 UTC (past)
            _create_bar("SPY", datetime(2026, 6, 1, 9, 30, 0, tzinfo=eastern_tz), 101.0),      # 09:30 EDT = 13:30 UTC (past)
            _create_bar("SPY", datetime(2026, 6, 1, 14, 0, 0), 102.0),                         # 14:00 naive treated as UTC (exact match)
            _create_bar("SPY", datetime(2026, 6, 1, 10, 1, 0, tzinfo=eastern_tz), 999.0),      # 10:01 EDT = 14:01 UTC (future!)
            _create_bar("SPY", datetime(2026, 6, 1, 14, 0, 1, tzinfo=utc_tz), 999.0),          # 14:00:01 UTC (future!)
        ]

        filtered = filter_bars_point_in_time(bars, t_eval_utc)
        assert len(filtered) == 3
        closes = [b.close for b in filtered]
        assert closes == [100.0, 101.0, 102.0]

    def test_naive_current_time_point_in_time_filtering(self, utc_tz):
        """When current_time is naive, it defaults safely to UTC without error."""
        t_eval_naive = datetime(2026, 6, 1, 14, 0, 0)
        bars = [
            _create_bar("SPY", datetime(2026, 6, 1, 13, 0, 0, tzinfo=utc_tz), 100.0),
            _create_bar("SPY", datetime(2026, 6, 1, 14, 0, 0, tzinfo=utc_tz), 101.0),
            _create_bar("SPY", datetime(2026, 6, 1, 15, 0, 0, tzinfo=utc_tz), 102.0),
        ]
        filtered = filter_bars_point_in_time(bars, t_eval_naive)
        assert len(filtered) == 2
        assert [b.close for b in filtered] == [100.0, 101.0]
