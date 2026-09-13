"""
tests.adversarial.test_cm2_challenger_stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone C-M2 Empirical Challenger Stress Test Suite:
1. Drawdown defense gates:
   - Multi-period crashes (-5%, -10%, -15%, -50%, -90%) and boundary transitions.
   - Exact capital preservation cash routing across uninitialized and stateful engines.
   - Sum-to-one invariant and cash weight consistency under extreme crash conditions.
2. Recovery hysteresis:
   - Requires 3 consecutive closes above SMA50 to clear lockout.
   - Interruption (close <= SMA50) immediately resets recovery counter to 0.
   - Lockout prevents premature equity re-entry while drawdown is active.
   - Buffer multiplier validation.
3. Merton Jump-Diffusion SDE simulation:
   - Prices down to 1e-5, 1e-6, and sub-penny regimes.
   - Single-day 90% drops and logarithmic crashes without candlestick inversion.
   - Zero and negative base volume handling.
   - 1,000-step extreme jump diffusion stress test with 100% OHLC validity.
   - Bar model volume boundary enforcement (volume=0 valid, volume=-1 invalid).
4. Exhaustive OrderIntent fuzzing:
   - All contradictory action/side combinations rejected.
   - Action case normalization and invalid action string rejection.
   - Numerical boundary violations (target_weight, current_weight, delta_weight, estimated_price, target_shares).
   - Frozen immutability and extra field prohibition.
   - Action/side and reason/rationale bi-directional synchronization.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import numpy as np
import pytest
from pydantic import ValidationError

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
from strategy_engine.signals.regime_detector import SignalEngine
from strategy_engine.simulator.regime_sde import MertonJumpDiffusionSimulator
from tests.adversarial.test_r2_invariant_stress import _build_synthetic_market_data, _create_bar


# ============================================================================
# 1. Drawdown Defense Gates & Exact Cash Routing Stress Tests
# ============================================================================

class TestDrawdownDefenseGatesAndCrashes:
    """Stress-test drawdown defense gates, crash step-functions, and cash routing."""

    @pytest.mark.parametrize(
        "drawdown_pct, expected_gate",
        [
            (0.00, 1.00),
            (-0.01, 1.00),
            (-0.0499, 1.00),
            (-0.05, 0.50),     # Exact -5% threshold
            (-0.0501, 0.50),
            (-0.0999, 0.50),
            (-0.10, 0.20),     # Exact -10% threshold
            (-0.1001, 0.20),
            (-0.1499, 0.20),
            (-0.15, 0.00),     # Exact -15% circuit breaker
            (-0.25, 0.00),
            (-0.50, 0.00),     # Severe 50% crash
            (-0.90, 0.00),     # Extreme 90% black swan
            (-0.999, 0.00),
        ],
    )
    def test_drawdown_gate_step_function_exact_thresholds(
        self, drawdown_pct: float, expected_gate: float
    ):
        """Verify evaluate_drawdown_gate step function at exact thresholds and boundaries."""
        gate = evaluate_drawdown_gate(drawdown_pct)
        assert math.isclose(gate, expected_gate, abs_tol=1e-6), (
            f"Drawdown {drawdown_pct:.4f} expected gate {expected_gate}, got {gate}"
        )

    @pytest.mark.parametrize(
        "regime",
        [
            MarketRegime.BULL_AGGRESSIVE,
            MarketRegime.BULL_NORMAL,
            MarketRegime.CORRECTION_FRAGILE,
            MarketRegime.BEAR_CRISIS,
        ],
    )
    @pytest.mark.parametrize("crash_pct", [0.05, 0.10, 0.15, 0.50, 0.90])
    def test_cash_routing_across_all_crashes_and_regimes(
        self, regime: MarketRegime, crash_pct: float
    ):
        """Verify capital preservation and cash routing across crashes for all market regimes."""
        data = _build_synthetic_market_data(n_days=265, seed=123)
        t_eval = data["SPY"][220].timestamp

        dd_pct = -crash_pct
        gate = evaluate_drawdown_gate(dd_pct)

        signals = SignalSnapshot(
            timestamp=t_eval,
            spy_price=550.0,
            spy_sma50=520.0,
            spy_sma200=480.0,
            realized_vol_20d=0.12,
            vol_scale_factor=1.00,
            drawdown_pct=dd_pct,
            circuit_breaker_active=False,
            regime=regime,
            indicators={"drawdown_gate": gate},
        )

        alloc = compute_deterministic_allocation(signals, data, cash_symbol="SHV")

        # Invariant 1: Weights must sum strictly to 1.0 within 1e-5
        total_weight = sum(alloc.weights.values())
        assert math.isclose(total_weight, 1.0, rel_tol=1e-5, abs_tol=1e-5)

        # Invariant 2: All weights must be non-negative
        for sym, w in alloc.weights.items():
            assert w >= -1e-6, f"Negative weight for {sym}: {w}"
            assert w <= 1.0 + 1e-6, f"Weight > 1.0 for {sym}: {w}"

        # Invariant 3: Cash weight must match sum of cash-equivalents
        cash_sum = sum(w for s, w in alloc.weights.items() if s in ("SHV", "BIL", "CASH"))
        assert math.isclose(alloc.cash_weight, cash_sum, abs_tol=1e-4)

        # Invariant 4: Circuit breaker crashes (>= 15%) must route 100% of risk assets to defensive/cash
        if crash_pct >= 0.15:
            equity_weight = sum(
                alloc.weights.get(s, 0.0)
                for s in ("SPY", "QQQ", "AAPL", "MSFT", "NVDA", "XLK", "XLF", "XLV", "XLE", "XLI")
            )
            assert equity_weight == 0.0, (
                f"Equity exposure remaining during -{crash_pct*100:.0f}% crash: {equity_weight}"
            )
            # Entire portfolio must be defensive (Cash or qualified safe-havens TLT/GLD)
            defensive_weight = alloc.cash_weight + alloc.weights.get("TLT", 0.0) + alloc.weights.get("GLD", 0.0)
            assert math.isclose(defensive_weight, 1.0, rel_tol=1e-5, abs_tol=1e-5)

    def test_stateful_vs_uninitialized_engine_drawdown_retention(self):
        """Compare stateful incremental engine against fresh uninitialized engines across a crash trajectory."""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        spy_bars = [_create_bar("SPY", t0 + timedelta(days=i), close=100.0) for i in range(120)]
        market_data = {"SPY": spy_bars}

        # Multi-period trajectory: 100 -> 120 (peak) -> 114 (-5%) -> 108 (-10%) -> 102 (-15%) -> 60 (-50%) -> 12 (-90%)
        equity_trajectory = [100.0, 110.0, 120.0, 114.0, 108.0, 102.0, 60.0, 12.0]
        expected_drawdowns = [
            0.0,                                    # 100
            0.0,                                    # 110 (new peak)
            0.0,                                    # 120 (new peak)
            (114.0 - 120.0) / 120.0,                # -0.05 (-5%)
            (108.0 - 120.0) / 120.0,                # -0.10 (-10%)
            (102.0 - 120.0) / 120.0,                # -0.15 (-15%)
            (60.0 - 120.0) / 120.0,                 # -0.50 (-50%)
            (12.0 - 120.0) / 120.0,                 # -0.90 (-90%)
        ]

        # 1. Run incremental stateful engine
        stateful_engine = SignalEngine()
        stateful_dd = []
        stateful_gates = []

        for step, eq in enumerate(equity_trajectory):
            eval_time = spy_bars[60 + step].timestamp
            curve = equity_trajectory[: step + 1]
            snap = stateful_engine.compute_daily_signals(
                market_data, current_time=eval_time, portfolio_equity_curve=curve
            )
            stateful_dd.append(snap.drawdown_pct)
            stateful_gates.append(snap.indicators["drawdown_gate"])

        # 2. Run fresh uninitialized engines at each step with full historical equity curve
        for step, (eq, exp_dd) in enumerate(zip(equity_trajectory, expected_drawdowns)):
            eval_time = spy_bars[60 + step].timestamp
            curve = equity_trajectory[: step + 1]

            fresh_engine = SignalEngine()
            snap_fresh = fresh_engine.compute_daily_signals(
                market_data, current_time=eval_time, portfolio_equity_curve=curve
            )

            # Assert fresh engine matches stateful engine exactly
            assert math.isclose(snap_fresh.drawdown_pct, exp_dd, abs_tol=1e-4), (
                f"Step {step}: expected dd {exp_dd}, got {snap_fresh.drawdown_pct}"
            )
            assert math.isclose(snap_fresh.drawdown_pct, stateful_dd[step], abs_tol=1e-4), (
                f"Step {step}: fresh {snap_fresh.drawdown_pct} != stateful {stateful_dd[step]}"
            )
            assert snap_fresh.indicators["drawdown_gate"] == stateful_gates[step]

    def test_uninitialized_engine_omitted_equity_curve_empirical_behavior(self):
        """Empirically document that when portfolio_equity_curve is omitted, a fresh SignalEngine

        defaults current_equity to spy_bars[-1].close without priming from historical bars.
        """
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # SPY dropped from 500 to 250 over 100 days
        spy_bars = [_create_bar("SPY", t0 + timedelta(days=i), close=500.0) for i in range(100)]
        spy_bars.append(_create_bar("SPY", t0 + timedelta(days=100), close=250.0))

        # Fresh engine invoked WITHOUT portfolio_equity_curve
        fresh_engine = SignalEngine()
        sig = fresh_engine.compute_daily_signals(
            {"SPY": spy_bars},
            current_time=spy_bars[-1].timestamp,
            portfolio_equity_curve=None,  # Omitted
        )
        # Without portfolio_equity_curve, tracker has only 1 point (spy_bars[-1].close = 250.0)
        # and therefore drawdown evaluates to 0.0%
        assert sig.drawdown_pct == 0.0
        assert sig.indicators["drawdown_gate"] == 1.0


# ============================================================================
# 2. Recovery Hysteresis Stress Tests
# ============================================================================

class TestRecoveryHysteresisStress:
    """Stress-test 3-day recovery hysteresis, lockouts, interruptions, and threshold buffers."""

    def test_recovery_hysteresis_requires_strictly_3_consecutive_days(self):
        """Recovery lockout must persist through Day 1 and Day 2, clearing only on Day 3."""
        tracker = DrawdownDefenseTracker()
        tracker.prime([100.0, 100.0, 85.0])  # -15% drop -> gate 0.00, lockout True
        assert tracker.active_gate_multiplier == 0.00
        assert tracker.in_recovery_lockout is True
        assert tracker.consecutive_recovery_days == 0

        sma50 = 100.0

        # Day 1: Price > SMA50 (105 > 100)
        g1 = tracker.update(current_equity=85.0, benchmark_price=105.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 1
        assert g1 == 0.00
        assert tracker.in_recovery_lockout is True

        # Day 2: Price > SMA50 (106 > 100)
        g2 = tracker.update(current_equity=85.0, benchmark_price=106.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 2
        assert g2 == 0.00
        assert tracker.in_recovery_lockout is True

        # Day 3: Price > SMA50 (107 > 100) - 3 consecutive days reached!
        # If equity recovered to 96 (-4% drop, raw_gate = 1.00):
        g3 = tracker.update(current_equity=96.0, benchmark_price=107.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 3
        assert g3 == 1.00
        assert tracker.in_recovery_lockout is False

    def test_recovery_hysteresis_interrupted_by_single_dip(self):
        """A single close below SMA50 must reset consecutive_recovery_days to 0 immediately."""
        tracker = DrawdownDefenseTracker()
        tracker.prime([100.0, 100.0, 85.0])
        sma50 = 100.0

        # Day 1: above SMA50
        tracker.update(current_equity=85.0, benchmark_price=105.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 1

        # Day 2: above SMA50
        tracker.update(current_equity=85.0, benchmark_price=106.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 2

        # Day 3: DIP BELOW SMA50 (99.0 <= 100.0) -> resets to 0!
        g_dip = tracker.update(current_equity=96.0, benchmark_price=99.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 0
        assert g_dip == 0.00  # Lockout prevented re-entry despite equity recovery to 96!
        assert tracker.in_recovery_lockout is True

        # Day 4: above SMA50 (new streak begins: Day 1)
        g4 = tracker.update(current_equity=96.0, benchmark_price=105.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 1
        assert g4 == 0.00

        # Day 5: above SMA50 (Day 2)
        g5 = tracker.update(current_equity=96.0, benchmark_price=106.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 2
        assert g5 == 0.00

        # Day 6: above SMA50 (Day 3 -> Confirmed!)
        g6 = tracker.update(current_equity=96.0, benchmark_price=107.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 3
        assert g6 == 1.00
        assert tracker.in_recovery_lockout is False

    def test_recovery_lockout_monotonic_derisking_only(self):
        """During lockout before 3-day recovery, active_gate_multiplier can de-risk further (min)

        but can never increase equity risk.
        """
        tracker = DrawdownDefenseTracker()
        tracker.prime([100.0, 100.0, 92.0])  # -8% drop -> raw_gate = 0.50, lockout True
        assert tracker.active_gate_multiplier == 0.50
        assert tracker.in_recovery_lockout is True

        sma50 = 100.0

        # Day 1: benchmark > SMA50, but drawdown deepens to -12% (raw_gate = 0.20)
        # Gate must drop from 0.50 to 0.20
        g1 = tracker.update(current_equity=88.0, benchmark_price=105.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 1
        assert g1 == 0.20

        # Day 2: benchmark > SMA50, equity rebounds to 94 (-6%, raw_gate = 0.50)
        # Gate must NOT increase to 0.50 yet! It must remain min(0.20, 0.50) = 0.20
        g2 = tracker.update(current_equity=94.0, benchmark_price=106.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 2
        assert g2 == 0.20

        # Day 3: benchmark > SMA50 (3 days achieved), equity is 94 (-6%, raw_gate = 0.50)
        # Now can_reenter is True, gate steps up to raw_gate = 0.50
        g3 = tracker.update(current_equity=94.0, benchmark_price=107.0, benchmark_sma50=sma50)
        assert tracker.consecutive_recovery_days == 3
        assert g3 == 0.50
        # But lockout is STILL True because raw_gate < 1.00!
        assert tracker.in_recovery_lockout is True

    def test_recovery_buffer_multiplier(self):
        """Verify recovery_buffer_multiplier > 1.0 requires price above buffered SMA50."""
        tracker = DrawdownDefenseTracker()
        tracker.prime([100.0, 100.0, 85.0])
        sma50 = 100.0

        # With 2% buffer (1.02), threshold is 102.0
        # Price 101.0 > 100.0 (SMA50) but < 102.0 (threshold)
        tracker.update(
            current_equity=85.0,
            benchmark_price=101.0,
            benchmark_sma50=sma50,
            recovery_buffer_multiplier=1.02,
        )
        assert tracker.consecutive_recovery_days == 0

        # Price 102.5 > 102.0 (threshold) -> triggers increment
        tracker.update(
            current_equity=85.0,
            benchmark_price=102.5,
            benchmark_sma50=sma50,
            recovery_buffer_multiplier=1.02,
        )
        assert tracker.consecutive_recovery_days == 1


# ============================================================================
# 3. Merton Jump-Diffusion SDE Extreme Simulation Stress Tests
# ============================================================================

class TestMertonJumpDiffusionExtremeStress:
    """Stress-test SDE simulation across sub-penny regimes, single-day 90% drops, and volume extremes."""

    def test_sde_sub_penny_regimes_down_to_1e_5(self):
        """Simulate paths operating down to 1e-5 and verify zero candlestick validation crashes."""
        sim = MertonJumpDiffusionSimulator(seed=101)
        sub_penny_path = np.array([
            0.009, 0.005, 0.0025, 0.001, 0.0005, 0.0001, 0.00005, 0.00002, 0.00001
        ])
        paths = {"MICRO": sub_penny_path}
        start_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = sim.paths_to_bars(paths, ["MICRO"], start_date)["MICRO"]

        assert len(bars) == len(sub_penny_path) - 1
        for i, b in enumerate(bars):
            assert b.open > 0.0
            assert b.high > 0.0
            assert b.low > 0.0
            assert b.close > 0.0
            assert b.low <= min(b.open, b.close) + 1e-5, f"Bar {i} low inversion: {b}"
            assert b.high >= max(b.open, b.close) - 1e-5, f"Bar {i} high inversion: {b}"
            assert b.high >= b.low, f"Bar {i} high < low: {b}"
            assert b.vwap is not None and b.vwap > 0.0

    def test_sde_single_day_90_percent_drop_candlestick_validity(self):
        """Simulate consecutive 90% single-day crashes ($1000 -> $100 -> $10 -> $1 -> $0.1 -> $0.01)."""
        sim = MertonJumpDiffusionSimulator(seed=202)
        crash_steps = np.array([1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001])
        paths = {"CRASH": crash_steps}
        start_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = sim.paths_to_bars(paths, ["CRASH"], start_date)["CRASH"]

        assert len(bars) == 7
        for i, b in enumerate(bars):
            assert b.low <= min(b.open, b.close) + 1e-5
            assert b.high >= max(b.open, b.close) - 1e-5
            assert b.high >= b.low
            assert b.low > 0.0

    def test_sde_negative_and_zero_base_volumes(self):
        """Passing zero or negative base_volumes must be safely handled without throwing exceptions."""
        sim = MertonJumpDiffusionSimulator(seed=303)
        paths = {
            "SPY": np.array([500.0, 501.0, 502.0]),
            "QQQ": np.array([400.0, 401.0, 402.0]),
        }
        # Inject 0 and negative base volume
        bad_volumes = {"SPY": 0, "QQQ": -100_000}
        start_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars_dict = sim.paths_to_bars(paths, ["SPY", "QQQ"], start_date, base_volumes=bad_volumes)

        for sym in ("SPY", "QQQ"):
            bars = bars_dict[sym]
            assert len(bars) == 2
            for b in bars:
                assert b.volume >= 1000  # Enforces floor of 1000
                assert b.trade_count is not None and b.trade_count >= 50

    def test_sde_massive_extreme_jump_simulation_1000_steps(self):
        """Simulate 1,000 steps with extreme volatility (150%) and frequent large negative jumps."""
        sim = MertonJumpDiffusionSimulator(seed=777)
        symbols = ["VOLATILE"]
        corr = np.array([[1.0]])
        paths = sim.simulate_multivariate_paths(
            symbols=symbols,
            initial_prices={"VOLATILE": 100.0},
            drifts={"VOLATILE": -0.20},
            volatilities={"VOLATILE": 1.20},  # 120% annualized vol!
            correlation_matrix=corr,
            jump_lambda={"VOLATILE": 15.0},   # 15 jumps per year
            jump_mean={"VOLATILE": -0.30},    # -30% average jump
            jump_vol={"VOLATILE": 0.25},
            n_days=1000,
        )

        bars = sim.paths_to_bars(paths, symbols, datetime(2025, 1, 1, tzinfo=timezone.utc))["VOLATILE"]
        assert len(bars) == 1000
        for b in bars:
            assert b.low <= min(b.open, b.close) + 1e-5
            assert b.high >= max(b.open, b.close) - 1e-5
            assert b.low > 0.0
            assert b.high >= b.low

    def test_bar_model_volume_boundaries(self):
        """Verify Bar model requires volume >= 0 and strictly rejects negative volume."""
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # volume=0 is valid
        bar_zero_vol = Bar(symbol="SPY", timestamp=t0, open=100.0, high=101.0, low=99.0, close=100.0, volume=0)
        assert bar_zero_vol.volume == 0

        # volume=-1 must raise ValidationError
        with pytest.raises(ValidationError):
            Bar(symbol="SPY", timestamp=t0, open=100.0, high=101.0, low=99.0, close=100.0, volume=-1)


# ============================================================================
# 4. Exhaustive OrderIntent Fuzzing Stress Tests
# ============================================================================

class TestOrderIntentFuzzingStress:
    """Exhaustive fuzzing of OrderIntent data model."""

    @pytest.mark.parametrize(
        "act, sd",
        [
            ("BUY", OrderSide.SELL),
            ("BUY", OrderSide.HOLD),
            ("SELL", OrderSide.BUY),
            ("SELL", OrderSide.HOLD),
            ("HOLD", OrderSide.BUY),
            ("HOLD", OrderSide.SELL),
            ("buy", OrderSide.SELL),
            ("sell", OrderSide.BUY),
            ("hold", OrderSide.BUY),
        ],
    )
    def test_reject_all_contradictory_action_and_side_pairs(self, act: str, sd: OrderSide):
        """Every permutation of contradictory action and side must be rejected with ValueError."""
        with pytest.raises(ValueError, match="Contradictory action"):
            OrderIntent(symbol="SPY", action=act, side=sd)

    @pytest.mark.parametrize(
        "raw_action, expected_upper, expected_side",
        [
            ("buy", "BUY", OrderSide.BUY),
            ("Buy", "BUY", OrderSide.BUY),
            ("bUy", "BUY", OrderSide.BUY),
            ("sell", "SELL", OrderSide.SELL),
            ("sELL", "SELL", OrderSide.SELL),
            ("hold", "HOLD", OrderSide.HOLD),
            ("HoLd", "HOLD", OrderSide.HOLD),
        ],
    )
    def test_case_insensitive_action_normalization(
        self, raw_action: str, expected_upper: str, expected_side: OrderSide
    ):
        """Action string must be normalized to uppercase and synchronize side."""
        order = OrderIntent(symbol="SPY", action=raw_action)
        assert order.action == expected_upper
        assert order.side == expected_side

    @pytest.mark.parametrize(
        "invalid_action",
        [
            "SHORT",
            "COVER",
            "CANCEL",
            "FLIP",
            "123",
            "",
            "   ",
            "LONG",
        ],
    )
    def test_reject_invalid_action_strings(self, invalid_action: str):
        """Invalid actions not in {BUY, SELL, HOLD} must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid order action"):
            OrderIntent(symbol="SPY", action=invalid_action)

    @pytest.mark.parametrize(
        "field_name, invalid_val",
        [
            ("target_weight", -0.01),
            ("target_weight", 1.01),
            ("target_weight", float("nan")),
            ("current_weight", -0.01),
            ("current_weight", 1.01),
            ("current_weight", float("nan")),
            ("delta_weight", -1.01),
            ("delta_weight", 1.01),
            ("delta_weight", float("nan")),
            ("target_shares", -0.01),
            ("estimated_price", 0.0),
            ("estimated_price", -10.0),
        ],
    )
    def test_numerical_boundary_violations_raise_validation_error(
        self, field_name: str, invalid_val: float
    ):
        """Out-of-bounds numerical parameters must raise ValidationError."""
        kwargs = {"symbol": "SPY", "action": "BUY", field_name: invalid_val}
        with pytest.raises(ValidationError):
            OrderIntent(**kwargs)

    def test_immutability_frozen_model(self):
        """OrderIntent instances must be strictly immutable."""
        order = OrderIntent(symbol="SPY", action="BUY", target_weight=0.5)
        with pytest.raises(ValidationError):
            order.symbol = "QQQ"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            order.target_weight = 0.8  # type: ignore[misc]

    def test_extra_fields_forbidden(self):
        """Passing unexpected/rogue parameters must raise ValidationError."""
        with pytest.raises(ValidationError):
            OrderIntent(symbol="SPY", action="BUY", rogue_hacker_field=123)  # type: ignore[call-arg]

    def test_bi_directional_reason_rationale_sync(self):
        """Providing either rationale or reason must automatically populate the other."""
        o1 = OrderIntent(symbol="SPY", action="BUY", rationale="Rebalance QQQ/SPY")
        assert o1.rationale == "Rebalance QQQ/SPY"
        assert o1.reason == "Rebalance QQQ/SPY"

        o2 = OrderIntent(symbol="SPY", action="SELL", reason="ATR trailing stop")
        assert o2.rationale == "ATR trailing stop"
        assert o2.reason == "ATR trailing stop"
