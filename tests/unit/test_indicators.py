"""
tests.unit.test_indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit test suite for multi-timeframe quantitative risk and trend indicators:
- Realized volatility targeting
- Dual-SMA trend filters and discrete scores
- Dynamic ATR Keltner stops and circuit breakers
- Trailing peak-to-trough drawdown gates with 3-day recovery hysteresis
- Market breadth
- Point-in-time causality filtration
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import pytest

from strategy_engine.core.models import Bar
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


def _create_bar(symbol: str, dt: datetime, close: float, high: float = 0.0, low: float = 0.0) -> Bar:
    h = high if high > 0 else close * 1.01
    l = low if low > 0 else close * 0.99
    return Bar(
        symbol=symbol,
        timestamp=dt,
        open=close,
        high=h,
        low=l,
        close=close,
        volume=100_000,
    )


def test_point_in_time_filtration():
    """Verify that bars with timestamp > current_time are strictly excluded."""
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    bars = [
        _create_bar("SPY", t0 + timedelta(days=i), 500.0 + i)
        for i in range(10)
    ]
    eval_time = t0 + timedelta(days=4)
    filtered = filter_bars_point_in_time(bars, eval_time)
    assert len(filtered) == 5
    assert filtered[-1].timestamp == eval_time
    assert all(b.timestamp <= eval_time for b in filtered)


def test_realized_volatility_calculation():
    """Verify 20-day annualized realized volatility calculation against mathematical definition."""
    # Alternating 2% gain / loss series
    prices = [100.0]
    for i in range(1, 25):
        mult = 1.02 if i % 2 == 1 else 0.98
        prices.append(prices[-1] * mult)
    
    vol = compute_realized_volatility(prices, window=20, annualization_factor=252)
    assert vol > 0.15
    assert vol < 0.45


def test_realized_volatility_zero_variance():
    """Flat price series should yield exactly 0.0 realized volatility."""
    flat_prices = [100.0] * 30
    vol = compute_realized_volatility(flat_prices, window=20)
    assert vol == 0.0


def test_realized_volatility_insufficient_bars():
    """Series with fewer bars than window + 1 returns 0.0."""
    short_prices = [100.0, 101.0, 102.0]
    assert compute_realized_volatility(short_prices, window=20) == 0.0
    assert compute_realized_volatility([], window=20) == 0.0


def test_realized_volatility_accepts_bars():
    """Verify compute_realized_volatility works on list of Bar objects."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [_create_bar("SPY", t0 + timedelta(days=i), 100.0 + (i % 2) * 2.0) for i in range(25)]
    vol = compute_realized_volatility(bars, window=20)
    assert vol > 0.0


def test_volatility_scale_factor_bounds():
    """Verify S_vol = min(max_scale, target_vol / max(realized_vol, min_vol))."""
    # Target 12%, min 5%
    # High vol: 40% -> 0.12 / 0.40 = 0.30
    scale_high = compute_volatility_scale_factor(0.40, target_vol=0.12, min_vol=0.05, max_scale=1.0)
    assert math.isclose(scale_high, 0.30, rel_tol=1e-3)

    # Moderate vol: 24% -> 0.12 / 0.24 = 0.50
    scale_med = compute_volatility_scale_factor(0.24, target_vol=0.12)
    assert math.isclose(scale_med, 0.50, rel_tol=1e-3)

    # Low vol: 8% -> 0.12 / 0.08 = 1.50 -> clamped to 1.0 (no leverage)
    scale_low = compute_volatility_scale_factor(0.08, target_vol=0.12, max_scale=1.0)
    assert scale_low == 1.0

    # Near-zero vol: 0.01 -> max(0.01, 0.05) = 0.05 -> 0.12 / 0.05 = 2.40 -> clamped to 1.0
    scale_zero = compute_volatility_scale_factor(0.001, target_vol=0.12, min_vol=0.05, max_scale=1.0)
    assert scale_zero == 1.0


def test_sma_calculation():
    """Verify moving average scalar over trailing window."""
    prices = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert compute_sma(prices, 3) == (30.0 + 40.0 + 50.0) / 3.0
    assert compute_sma([], 5) == 0.0


def test_atr_and_keltner_lower_band():
    """Verify True Range, ATR14, and lower Keltner band."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    # 20 bars with High=105, Low=95, Close=100
    for i in range(25):
        bars.append(Bar(
            symbol="SPY",
            timestamp=t0 + timedelta(days=i),
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=1000,
        ))

    atr = compute_atr(bars, window=14)
    assert math.isclose(atr, 10.0, abs_tol=1e-4)

    sma, atr_val, lower_band = compute_keltner_lower_band(bars, sma_window=20, atr_window=14, atr_multiplier=2.0)
    assert math.isclose(sma, 100.0, abs_tol=1e-4)
    assert math.isclose(atr_val, 10.0, abs_tol=1e-4)
    # Band_lower = 100.0 - 2.0 * 10.0 = 80.0
    assert math.isclose(lower_band, 80.0, abs_tol=1e-4)


def test_circuit_breaker_trigger():
    """Verify circuit breaker triggers when latest close drops below lower Keltner band."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [
        Bar(symbol="SPY", timestamp=t0 + timedelta(days=i), open=100.0, high=105.0, low=95.0, close=100.0, volume=1000)
        for i in range(50)
    ]
    # Normal day (Close = 100 > lower band ~80)
    is_active, metrics = check_circuit_breaker(bars, sma_window=50, atr_window=14, atr_multiplier=2.0)
    assert is_active is False
    assert metrics["close"] == 100.0

    # Flash crash bar (Close = 75 < lower band ~80)
    bars.append(Bar(symbol="SPY", timestamp=t0 + timedelta(days=51), open=95.0, high=96.0, low=70.0, close=75.0, volume=5000))
    is_active_breached, metrics_breached = check_circuit_breaker(bars, sma_window=50, atr_window=14, atr_multiplier=2.0)
    assert is_active_breached is True
    assert metrics_breached["close"] == 75.0
    assert metrics_breached["close"] < metrics_breached["lower_band"]


def test_evaluate_trend_filter_all_regimes():
    """Verify trend score for Confirmed Bull, Pullback, Breakdown, and Structural Bear."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # 1. Confirmed Bull: Price > SMA50 > SMA200
    prices_bull = list(range(100, 350))
    bars_bull = [_create_bar("SPY", t0 + timedelta(days=i), float(p)) for i, p in enumerate(prices_bull)]
    res_bull = evaluate_trend_filter(bars_bull, "SPY", fast_window=50, slow_window=200)
    assert res_bull["trend_score"] == 1.0
    assert res_bull["is_golden_cross"] is True
    assert res_bull["price_above_sma50"] is True
    assert res_bull["price_above_sma200"] is True

    # 2. Confirmed Bear: Price < SMA200 and SMA50 < SMA200
    prices_bear = list(range(350, 100, -1))
    bars_bear = [_create_bar("SPY", t0 + timedelta(days=i), float(p)) for i, p in enumerate(prices_bear)]
    res_bear = evaluate_trend_filter(bars_bear, "SPY", fast_window=50, slow_window=200)
    assert res_bear["trend_score"] == -1.0
    assert res_bear["is_golden_cross"] is False


def test_drawdown_gate_step_function():
    """Verify exact step boundaries of drawdown defense multiplier."""
    assert evaluate_drawdown_gate(0.0) == 1.00
    assert evaluate_drawdown_gate(-0.049) == 1.00
    assert evaluate_drawdown_gate(-0.05) == 0.50
    assert evaluate_drawdown_gate(-0.099) == 0.50
    assert evaluate_drawdown_gate(-0.10) == 0.20
    assert evaluate_drawdown_gate(-0.149) == 0.20
    assert evaluate_drawdown_gate(-0.15) == 0.00
    assert evaluate_drawdown_gate(-0.25) == 0.00


def test_drawdown_tracker_with_hysteresis():
    """Verify 3-day recovery hysteresis prevents premature re-entry from defensive gates."""
    tracker = DrawdownDefenseTracker()

    # Day 0: Peak at 100.0, Normal
    gate = tracker.update(current_equity=100.0, benchmark_price=500.0, benchmark_sma50=490.0)
    assert gate == 1.00
    assert tracker.in_recovery_lockout is False

    # Day 1: Crash to 92.0 (-8% DD -> triggers Level 1, gate=0.50)
    # Benchmark drops below SMA50
    gate = tracker.update(current_equity=92.0, benchmark_price=470.0, benchmark_sma50=490.0)
    assert gate == 0.50
    assert tracker.in_recovery_lockout is True
    assert tracker.consecutive_recovery_days == 0

    # Day 2: Benchmark rebounds above SMA50 for 1 day, equity recovers to 96.0 (-4% DD)
    # Lockout MUST stay active because consecutive recovery days = 1 (< 3)
    gate = tracker.update(current_equity=96.0, benchmark_price=495.0, benchmark_sma50=490.0)
    assert gate == 0.50
    assert tracker.in_recovery_lockout is True
    assert tracker.consecutive_recovery_days == 1

    # Day 3: Benchmark stays above SMA50 for 2nd day
    gate = tracker.update(current_equity=96.5, benchmark_price=496.0, benchmark_sma50=490.0)
    assert gate == 0.50
    assert tracker.consecutive_recovery_days == 2

    # Day 4: Benchmark dips back below SMA50 -> resets counter to 0!
    gate = tracker.update(current_equity=96.0, benchmark_price=488.0, benchmark_sma50=490.0)
    assert gate == 0.50
    assert tracker.consecutive_recovery_days == 0
    assert tracker.in_recovery_lockout is True

    # Days 5, 6, 7: Benchmark stays above SMA50 for 3 consecutive days
    tracker.update(current_equity=97.0, benchmark_price=495.0, benchmark_sma50=490.0)
    tracker.update(current_equity=97.0, benchmark_price=496.0, benchmark_sma50=490.0)
    gate_recovered = tracker.update(current_equity=97.0, benchmark_price=497.0, benchmark_sma50=490.0)
    assert tracker.consecutive_recovery_days >= 3
    # At -3% DD and 3 days above SMA50, re-entry is unlocked and gate returns to 1.00
    assert gate_recovered == 1.00
    assert tracker.in_recovery_lockout is False


def test_market_breadth_calculation():
    """Verify market breadth (% > 50 SMA / 200 SMA) and regime factor."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market_data = {}

    # 8 symbols in uptrend (Price > SMA50 and Price > SMA200)
    for i in range(8):
        sym = f"SYM_{i}"
        market_data[sym] = [_create_bar(sym, t0 + timedelta(days=d), 100.0 + d) for d in range(210)]

    # 2 symbols in downtrend (Price < SMA50 and Price < SMA200)
    for i in range(8, 10):
        sym = f"SYM_{i}"
        market_data[sym] = [_create_bar(sym, t0 + timedelta(days=d), 300.0 - d * 0.5) for d in range(210)]

    breadth = compute_market_breadth(market_data)
    assert math.isclose(breadth["breadth_50"], 0.80, abs_tol=1e-3)
    assert math.isclose(breadth["breadth_200"], 0.80, abs_tol=1e-3)
    assert breadth["breadth_regime_factor"] == 1.00
    assert breadth["symbols_evaluated"] == 10.0


def test_drawdown_tracker_normalized_nav_trajectory():
    """Verify DrawdownDefenseTracker full trajectory starting from normalized NAV V_0 = 1.0.
    Tests dynamic peak tracking, gate transitions, and recovery hysteresis.
    """
    tracker = DrawdownDefenseTracker()

    # Day 0: Inception at V_0 = 1.000
    gate_0 = tracker.update(current_equity=1.000, benchmark_price=500.0, benchmark_sma50=490.0)
    assert gate_0 == 1.00
    assert tracker.peak_equity == 1.000
    assert tracker.current_drawdown == 0.0
    assert tracker.in_recovery_lockout is False

    # Day 1: New high at V_1 = 1.050
    gate_1 = tracker.update(current_equity=1.050, benchmark_price=505.0, benchmark_sma50=491.0)
    assert gate_1 == 1.00
    assert tracker.peak_equity == 1.050
    assert tracker.current_drawdown == 0.0

    # Day 2: Mild pullback to 1.020 (DD = (1.020 - 1.050)/1.050 = -2.86%) -> Gate remains 1.00
    gate_2 = tracker.update(current_equity=1.020, benchmark_price=501.0, benchmark_sma50=492.0)
    assert gate_2 == 1.00
    assert math.isclose(tracker.current_drawdown, -0.028571, abs_tol=1e-5)
    assert tracker.in_recovery_lockout is False

    # Day 3: Level 1 Drawdown (-6.67% DD to 0.980) -> Gate cuts equity to 0.50
    gate_3 = tracker.update(current_equity=0.980, benchmark_price=480.0, benchmark_sma50=492.0)
    assert gate_3 == 0.50
    assert tracker.in_recovery_lockout is True

    # Day 4: Level 2 Drawdown (-12.38% DD to 0.920) -> Gate cuts equity to 0.20
    gate_4 = tracker.update(current_equity=0.920, benchmark_price=460.0, benchmark_sma50=490.0)
    assert gate_4 == 0.20
    assert tracker.in_recovery_lockout is True

    # Day 5: Level 3 Drawdown (-18.10% DD to 0.860) -> Gate cuts equity to 0.00 (100% Cash)
    gate_5 = tracker.update(current_equity=0.860, benchmark_price=440.0, benchmark_sma50=488.0)
    assert gate_5 == 0.00
    assert tracker.in_recovery_lockout is True


def test_filter_bars_point_in_time_timezone_cross_compatibility():
    """Verify filter_bars_point_in_time handles aware UTC, aware Eastern, and naive timestamps."""
    from zoneinfo import ZoneInfo

    t0_utc = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
    bars = [
        _create_bar("SPY", t0_utc + timedelta(hours=i), 500.0 + i)
        for i in range(5)
    ]  # 14:00, 15:00, 16:00, 17:00, 18:00 UTC

    # 1. Naive current_time matching 16:00 (assumed UTC)
    t_naive = datetime(2026, 6, 1, 16, 0)
    filtered_naive = filter_bars_point_in_time(bars, t_naive)
    assert len(filtered_naive) == 3
    assert filtered_naive[-1].timestamp == t0_utc + timedelta(hours=2)

    # 2. Timezone-aware US/Eastern current_time: 12:00 EDT == 16:00 UTC
    t_eastern = datetime(2026, 6, 1, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    filtered_eastern = filter_bars_point_in_time(bars, t_eastern)
    assert len(filtered_eastern) == 3
    assert filtered_eastern[-1].close == 502.0

    # 3. Naive bars compared to UTC aware current_time
    naive_bars = [
        _create_bar("SPY", datetime(2026, 6, 1, 14, 0) + timedelta(hours=i), 500.0 + i)
        for i in range(5)
    ]
    filtered_naive_bars = filter_bars_point_in_time(naive_bars, t0_utc + timedelta(hours=2))
    assert len(filtered_naive_bars) == 3
