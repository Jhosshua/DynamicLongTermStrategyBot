"""
tests/adversarial/test_tier5_adversarial.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tier 5 Adversarial Stress & Edge-Case Verification Suite for Milestone M1.
Covers:
1. Extreme price bounds ($10^{-8}$, $10^9$)
2. Inverted candles and OHLC boundary consistency (including epsilon bypass)
3. Target allocation weights edge cases (1.00001, 0.99999, negative, NaN, Inf)
4. Pydantic immutability under direct attribute mutation and container mutability
5. Division by zero, empty series, and NaN/Inf in math utilities
6. Floating point precision limits, CAGR overflow, and volatility scale factor edge cases
7. Universe reference leaking and contradictory OrderIntent handling
"""

from datetime import datetime, timezone
import math
import numpy as np
import pytest
from pydantic import ValidationError

from strategy_engine.core.models import (
    AssetClass,
    Bar,
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderType,
    Quote,
    SignalSnapshot,
    TargetAllocation,
    Trade,
)
from strategy_engine.core import math_utils
from strategy_engine.core import universe


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================================
# 1. Extreme Price Bounds ($10^{-8}$, $10^9$)
# ============================================================================

def test_bar_extreme_low_price_bound(now: datetime):
    """Bar must handle extreme sub-cent prices ($10^-8) without crashing."""
    bar = Bar(
        symbol="PENNY",
        timestamp=now,
        open=1.00e-8,
        high=1.05e-8,
        low=0.95e-8,
        close=1.02e-8,
        volume=100_000_000,
        trade_count=500,
        vwap=1.01e-8,
    )
    assert bar.open == 1.00e-8
    assert bar.high == 1.05e-8
    assert bar.low == 0.95e-8
    assert bar.close == 1.02e-8


def test_bar_extreme_high_price_bound(now: datetime):
    """Bar must handle Berkshire Hathaway / sovereign bond extreme prices ($10^9)."""
    bar = Bar(
        symbol="TITAN",
        timestamp=now,
        open=1.00e9,
        high=1.05e9,
        low=0.95e9,
        close=1.02e9,
        volume=10,
        trade_count=2,
        vwap=1.01e9,
    )
    assert bar.open == 1.00e9
    assert bar.high == 1.05e9
    assert bar.close == 1.02e9


def test_quote_and_trade_extreme_price_bounds(now: datetime):
    """Quotes and Trades must support both $10^-8 and $10^9 prices."""
    q_low = Quote(
        symbol="TINY", timestamp=now,
        bid_price=1.00e-8, bid_size=1000,
        ask_price=1.10e-8, ask_size=1000
    )
    assert q_low.bid_price == 1.00e-8

    q_high = Quote(
        symbol="MEGA", timestamp=now,
        bid_price=1.00e9, bid_size=1,
        ask_price=1.01e9, ask_size=1
    )
    assert q_high.ask_price == 1.01e9

    t_low = Trade(symbol="TINY", timestamp=now, price=1.05e-8, size=500, id=1)
    assert t_low.price == 1.05e-8

    t_high = Trade(symbol="MEGA", timestamp=now, price=1.005e9, size=1, id=2)
    assert t_high.price == 1.005e9


def test_math_utils_scale_invariance_on_extreme_prices():
    """Realized volatility, returns, and drawdown must remain scale invariant."""
    base_series = [100.0, 102.0, 99.0, 101.0, 103.0, 98.0, 97.0, 104.0] * 3
    tiny_series = [p * 1e-10 for p in base_series]
    huge_series = [p * 1e7 for p in base_series]

    vol_base = math_utils.calculate_realized_volatility(base_series, window=20)
    vol_tiny = math_utils.calculate_realized_volatility(tiny_series, window=20)
    vol_huge = math_utils.calculate_realized_volatility(huge_series, window=20)

    assert math.isclose(vol_base, vol_tiny, rel_tol=1e-5)
    assert math.isclose(vol_base, vol_huge, rel_tol=1e-5)

    dd_base_curr, dd_base_max = math_utils.calculate_drawdown(base_series)
    dd_tiny_curr, dd_tiny_max = math_utils.calculate_drawdown(tiny_series)
    dd_huge_curr, dd_huge_max = math_utils.calculate_drawdown(huge_series)

    assert math.isclose(dd_base_curr, dd_tiny_curr, rel_tol=1e-5)
    assert math.isclose(dd_base_max, dd_tiny_max, rel_tol=1e-5)
    assert math.isclose(dd_base_curr, dd_huge_curr, rel_tol=1e-5)
    assert math.isclose(dd_base_max, dd_huge_max, rel_tol=1e-5)


# ============================================================================
# 2. Inverted Candles and OHLC Boundary Consistency
# ============================================================================

@pytest.mark.parametrize("bad_ohlc", [
    {"open": 100.0, "high": 90.0, "low": 95.0, "close": 100.0},  # high < low
    {"open": 100.0, "high": 95.0, "low": 90.0, "close": 92.0},   # high < open
    {"open": 90.0, "high": 95.0, "low": 90.0, "close": 100.0},   # high < close
    {"open": 90.0, "high": 100.0, "low": 95.0, "close": 98.0},   # low > open
    {"open": 98.0, "high": 100.0, "low": 95.0, "close": 90.0},   # low > close
    {"open": 0.0, "high": 10.0, "low": 0.0, "close": 5.0},       # open <= 0
    {"open": 10.0, "high": 0.0, "low": 0.0, "close": 5.0},       # high <= 0
    {"open": 10.0, "high": 15.0, "low": 0.0, "close": 5.0},      # low <= 0
    {"open": 10.0, "high": 15.0, "low": 5.0, "close": 0.0},      # close <= 0
    {"open": 10.0, "high": 15.0, "low": 5.0, "close": 10.0, "volume": -1},  # negative volume
])
def test_inverted_candle_rejection_standard_prices(now: datetime, bad_ohlc: dict):
    """Standard price inverted candles must be strictly rejected with ValidationError."""
    kwargs = {"symbol": "SPY", "timestamp": now, "volume": 100}
    kwargs.update(bad_ohlc)
    with pytest.raises(ValidationError):
        Bar(**kwargs)


def test_quote_crossed_market_rejection(now: datetime):
    """Crossed quote where ask < bid must be rejected for standard prices."""
    with pytest.raises(ValidationError):
        Quote(
            symbol="SPY", timestamp=now,
            bid_price=501.0, bid_size=10,
            ask_price=500.0, ask_size=10
        )


def test_ohlc_epsilon_bypass_vulnerability(now: datetime):
    """EMPIRICAL FINDING: Epsilon bypass vulnerability in models.py line 92-100.
    
    Because eps = 1e-5 is hardcoded as an absolute difference, any inverted candle
    with price scale <= 1e-5 bypasses OHLC validation.
    For example: open=1e-8, high=1e-8, low=2e-8 (low is 2x high!).
    """
    # Low (2e-8) is greater than High (1e-8)
    # The condition `self.high < self.low - eps` computes:
    # 1e-8 < 2e-8 - 1e-5  =>  1e-8 < -9.98e-6 (False!)
    # Hence validation PASSES instead of raising!
    inverted_micro_bar = Bar(
        symbol="MICRO",
        timestamp=now,
        open=1.0e-8,
        high=1.0e-8,
        low=2.0e-8,  # Low > High!
        close=1.0e-8,
        volume=100,
    )
    assert inverted_micro_bar.high < inverted_micro_bar.low  # Inverted candle was accepted!


# ============================================================================
# 3. Target Allocation Weights Edge Cases
# ============================================================================

def test_target_allocation_weights_sum_to_0_99999(now: datetime):
    """Weights summing to 0.99999 must be accepted (within 1e-5 tolerance)."""
    alloc = TargetAllocation(
        timestamp=now,
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 0.99999},
        cash_weight=0.0,
    )
    assert math.isclose(sum(alloc.weights.values()), 0.99999)


def test_target_allocation_split_weights_sum_to_1_00001(now: datetime):
    """Split weights summing to 1.00001 must be accepted within 1e-5 tolerance."""
    alloc = TargetAllocation(
        timestamp=now,
        regime=MarketRegime.BULL_NORMAL,
        weights={"SPY": 0.500005, "QQQ": 0.500005},
        cash_weight=0.0,
    )
    assert math.isclose(sum(alloc.weights.values()), 1.00001)


def test_target_allocation_single_weight_1_00001_rejected_by_bounds(now: datetime):
    """A single weight of 1.00001 is rejected because individual weight bound is 1.0 + 1e-6."""
    with pytest.raises(ValidationError, match="out of valid bounds"):
        TargetAllocation(
            timestamp=now,
            regime=MarketRegime.BULL_NORMAL,
            weights={"SPY": 1.00001},
            cash_weight=0.0,
        )


def test_target_allocation_weights_outside_1e5_tolerance_rejected(now: datetime):
    """Weights summing to 1.0001 or 0.999 must be rejected."""
    # Sum = 1.0001
    with pytest.raises(ValidationError, match="Portfolio weights must sum to 1.0"):
        TargetAllocation(
            timestamp=now,
            regime=MarketRegime.BULL_NORMAL,
            weights={"SPY": 0.50005, "QQQ": 0.50005},
            cash_weight=0.0,
        )

    # Sum = 0.999
    with pytest.raises(ValidationError, match="Portfolio weights must sum to 1.0"):
        TargetAllocation(
            timestamp=now,
            regime=MarketRegime.BULL_NORMAL,
            weights={"SPY": 0.999},
            cash_weight=0.0,
        )


def test_target_allocation_negative_and_nan_inf_weights(now: datetime):
    """Negative weights, NaN, inf, and empty weights must all be rejected."""
    # Negative weight
    with pytest.raises(ValidationError, match="out of valid bounds"):
        TargetAllocation(
            timestamp=now,
            regime=MarketRegime.BULL_NORMAL,
            weights={"SPY": -0.05, "QQQ": 1.05},
            cash_weight=0.0,
        )

    # NaN weight
    with pytest.raises(ValidationError):
        TargetAllocation(
            timestamp=now,
            regime=MarketRegime.BULL_NORMAL,
            weights={"SPY": float("nan"), "CASH": 1.0},
            cash_weight=1.0,
        )

    # Inf weight
    with pytest.raises(ValidationError):
        TargetAllocation(
            timestamp=now,
            regime=MarketRegime.BULL_NORMAL,
            weights={"SPY": float("inf"), "CASH": 0.0},
            cash_weight=0.0,
        )

    # Empty weights
    with pytest.raises(ValidationError, match="Portfolio weights must sum to 1.0"):
        TargetAllocation(
            timestamp=now,
            regime=MarketRegime.BULL_NORMAL,
            weights={},
            cash_weight=0.0,
        )


def test_target_allocation_cash_weight_inconsistency(now: datetime):
    """When SHV/BIL/CASH is present, cash_weight must match sum of cash symbols."""
    with pytest.raises(ValidationError, match="cash_weight .* must match sum of cash"):
        TargetAllocation(
            timestamp=now,
            regime=MarketRegime.BEAR_CRISIS,
            weights={"SPY": 0.5, "SHV": 0.5},
            cash_weight=0.2,  # Inconsistent with SHV=0.5
        )


# ============================================================================
# 4. Pydantic Immutability Under Direct Attribute Mutation
# ============================================================================

def test_pydantic_direct_attribute_mutation_rejected(now: datetime):
    """Direct attribute reassignment must raise ValidationError on frozen models."""
    bar = Bar(symbol="SPY", timestamp=now, open=500, high=505, low=495, close=502, volume=100)
    with pytest.raises(ValidationError):
        bar.close = 510.0  # type: ignore

    quote = Quote(symbol="SPY", timestamp=now, bid_price=500, bid_size=10, ask_price=501, ask_size=10)
    with pytest.raises(ValidationError):
        quote.ask_price = 505.0  # type: ignore

    trade = Trade(symbol="SPY", timestamp=now, price=500, size=100, id=1)
    with pytest.raises(ValidationError):
        trade.price = 505.0  # type: ignore

    snapshot = SignalSnapshot(
        timestamp=now, spy_price=500, spy_sma50=495, spy_sma200=480,
        realized_vol_20d=0.15, vol_scale_factor=0.8, drawdown_pct=-0.02,
        regime=MarketRegime.BULL_NORMAL
    )
    with pytest.raises(ValidationError):
        snapshot.spy_price = 510.0  # type: ignore

    alloc = TargetAllocation(timestamp=now, regime=MarketRegime.BULL_NORMAL, weights={"SPY": 1.0}, cash_weight=0.0)
    with pytest.raises(ValidationError):
        alloc.regime = MarketRegime.BEAR_CRISIS  # type: ignore

    order = OrderIntent(symbol="SPY", action="BUY", target_weight=0.5)
    with pytest.raises(ValidationError):
        order.symbol = "QQQ"  # type: ignore


def test_extra_fields_forbidden(now: datetime):
    """Extra undeclared fields must be rejected (extra='forbid')."""
    with pytest.raises(ValidationError):
        Bar(symbol="SPY", timestamp=now, open=500, high=505, low=495, close=502, volume=100, rogue_field=42)  # type: ignore


def test_nested_mutable_containers_vulnerability(now: datetime):
    """EMPIRICAL FINDING: Dict attributes are mutable in-place despite frozen=True.
    
    TargetAllocation.weights and SignalSnapshot.indicators are standard Python dicts.
    Callers can mutate internal weights and indicators post-validation.
    """
    alloc = TargetAllocation(
        timestamp=now, regime=MarketRegime.BULL_NORMAL, weights={"SPY": 1.0}, cash_weight=0.0
    )
    # Mutating weights in-place bypasses frozen=True
    alloc.weights["SPY"] = 999.0
    assert alloc.weights["SPY"] == 999.0  # Demonstrates mutable container vulnerability

    snapshot = SignalSnapshot(
        timestamp=now, spy_price=500, spy_sma50=495, spy_sma200=480,
        realized_vol_20d=0.15, vol_scale_factor=0.8, drawdown_pct=-0.02,
        regime=MarketRegime.BULL_NORMAL, indicators={"atr": 5.0}
    )
    snapshot.indicators["atr"] = -100.0
    assert snapshot.indicators["atr"] == -100.0


# ============================================================================
# 5. Division by Zero and Empty Price Series in Math Utilities
# ============================================================================

def test_realized_volatility_zero_and_empty():
    """calculate_realized_volatility must return 0.0 on empty, short, or invalid series."""
    assert math_utils.calculate_realized_volatility([]) == 0.0
    assert math_utils.calculate_realized_volatility([100.0]) == 0.0
    assert math_utils.calculate_realized_volatility([100.0] * 10, window=20) == 0.0
    # Constant prices -> zero variance -> 0.0
    assert math_utils.calculate_realized_volatility([100.0] * 25, window=20) == 0.0
    # Zero in window -> returns 0.0 safely without ZeroDivisionError
    assert math_utils.calculate_realized_volatility([100.0] * 20 + [0.0], window=20) == 0.0
    # Negative in window -> returns 0.0 safely
    assert math_utils.calculate_realized_volatility([100.0] * 20 + [-10.0], window=20) == 0.0
    # Non-positive window -> returns 0.0 safely
    assert math_utils.calculate_realized_volatility([100.0] * 25, window=0) == 0.0
    assert math_utils.calculate_realized_volatility([100.0] * 25, window=1) == 0.0


def test_calculate_drawdown_zero_and_empty():
    """calculate_drawdown must return (0.0, 0.0) on empty, zero, or negative equity curves."""
    assert math_utils.calculate_drawdown([]) == (0.0, 0.0)
    assert math_utils.calculate_drawdown([0.0, 0.0, 0.0]) == (0.0, 0.0)
    assert math_utils.calculate_drawdown([-50.0, -100.0]) == (0.0, 0.0)
    assert math_utils.calculate_drawdown([100.0]) == (0.0, 0.0)

    # Valid drawdown
    curr_dd, max_dd = math_utils.calculate_drawdown([100.0, 80.0, 90.0, 70.0])
    assert math.isclose(curr_dd, -0.30)
    assert math.isclose(max_dd, -0.30)


def test_calculate_atr_empty_and_mismatched_lengths():
    """calculate_atr on empty series returns 0.0. Mismatched lengths crash with IndexError."""
    assert math_utils.calculate_atr([], [], []) == 0.0
    assert math_utils.calculate_atr([105.0], [95.0], [100.0], window=14) == 10.0

    # EMPIRICAL FINDING: calculate_atr crashes with IndexError if len(highs) < len(closes)
    with pytest.raises(IndexError):
        math_utils.calculate_atr(
            highs=[10.0, 11.0],
            lows=[9.0, 10.0],
            closes=[10.0] * 20,
            window=14,
        )


def test_log_returns_and_simple_returns_empty_and_zero():
    """Returns functions must handle empty and zero boundary inputs."""
    assert len(math_utils.log_returns([])) == 0
    assert len(math_utils.log_returns([100.0])) == 0
    with pytest.raises(ValueError, match="strictly positive"):
        math_utils.log_returns([100.0, 0.0])
    with pytest.raises(ValueError, match="strictly positive"):
        math_utils.log_returns([100.0, -10.0])

    assert len(math_utils.simple_returns([])) == 0
    assert len(math_utils.simple_returns([100.0])) == 0
    with pytest.raises(ZeroDivisionError):
        math_utils.simple_returns([0.0, 100.0])


def test_ratios_empty_and_zero():
    """Sharpe and Calmar ratios must return 0.0 on degenerate inputs."""
    assert math_utils.sharpe_ratio([]) == 0.0
    assert math_utils.sharpe_ratio([0.05]) == 0.0
    assert math_utils.sharpe_ratio([0.01, 0.01, 0.01]) == 0.0  # Zero variance

    assert math_utils.calmar_ratio([]) == 0.0
    assert math_utils.calmar_ratio([100.0, 100.0]) == 0.0
    assert math_utils.calmar_ratio([100.0, 105.0, 110.0]) == 0.0  # Zero drawdown


# ============================================================================
# 6. Floating Point Precision & Numerical Edge Cases
# ============================================================================

def test_cagr_overflow_on_extreme_short_jump():
    """EMPIRICAL FINDING: cagr produces inf on extreme multi-bagger jumps over tiny windows."""
    # Price jumps 100x over 1 day (years = 1/252) -> (100)**252 overflows float64
    with pytest.warns(RuntimeWarning, match="overflow encountered in scalar power"):
        c = math_utils.cagr([10.0, 1000.0], periods_per_year=252)
        assert math.isinf(c)


def test_volatility_scale_factor_nan_inf_quirks():
    """Fail-safe: volatility_scale_factor with NaN produces 0.0 to prevent unintended leverage."""
    # When realized_vol is NaN, fail-safe returns 0.0
    assert math_utils.volatility_scale_factor(float("nan")) == 0.0

    # When realized_vol is Inf, scale is 0.0
    assert math_utils.volatility_scale_factor(float("inf")) == 0.0

    # Normal vol
    assert math.isclose(math_utils.volatility_scale_factor(0.12), 1.0)
    assert math.isclose(math_utils.volatility_scale_factor(0.24), 0.5)


def test_drawdown_gate_exact_step_boundaries():
    """Step function exact boundaries: -0.05, -0.10, -0.15."""
    assert math_utils.evaluate_drawdown_gate(-0.0499) == 1.00
    assert math_utils.evaluate_drawdown_gate(-0.0500) == 0.50  # Exactly at -5%
    assert math_utils.evaluate_drawdown_gate(-0.0999) == 0.50
    assert math_utils.evaluate_drawdown_gate(-0.1000) == 0.20  # Exactly at -10%
    assert math_utils.evaluate_drawdown_gate(-0.1499) == 0.20
    assert math_utils.evaluate_drawdown_gate(-0.1500) == 0.00  # Exactly at -15%
    assert math_utils.evaluate_drawdown_gate(-0.5000) == 0.00


# ============================================================================
# 7. Universe & OrderIntent Adversarial Cases
# ============================================================================

def test_universe_direct_dictionary_mutability():
    """EMPIRICAL FINDING: get_universe() returns mutable reference to UNIVERSE_ASSETS."""
    u = universe.get_universe()
    initial_len = len(u)
    dummy_asset = universe.UniverseAsset("ROGUE", "Rogue", AssetClass.EQUITY_INDEX, 1)
    u["ROGUE"] = dummy_asset
    assert "ROGUE" in universe.get_universe()
    assert universe.is_valid_symbol("ROGUE") is True
    # Cleanup
    del u["ROGUE"]
    assert len(universe.get_universe()) == initial_len


def test_order_intent_contradictory_action_and_side():
    """OrderIntent must reject contradictory action='BUY' with side=OrderSide.SELL."""
    with pytest.raises(ValueError, match="Contradictory action"):
        OrderIntent(symbol="SPY", action="BUY", side=OrderSide.SELL)
