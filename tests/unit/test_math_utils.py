"""
tests/unit/test_math_utils.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for deterministic numerical, statistical, and financial risk utilities.
"""

import math
import numpy as np
import pytest

from strategy_engine.core.math_utils import (
    annualized_volatility,
    average_true_range,
    cagr,
    calculate_atr,
    calculate_drawdown,
    calculate_ema,
    calculate_momentum_12_1,
    calculate_realized_volatility,
    calculate_sma,
    calmar_ratio,
    compound_returns,
    drawdown_series,
    evaluate_drawdown_gate,
    exponential_moving_average,
    log_returns,
    maximum_drawdown,
    sharpe_ratio,
    simple_moving_average,
    simple_returns,
    volatility_scale_factor,
)


def test_realized_volatility_flat_and_alternating():
    flat_prices = [100.0] * 25
    assert calculate_realized_volatility(flat_prices, window=20) == 0.0

    alt_prices = [100.0 if i % 2 == 0 else 102.0 for i in range(25)]
    vol = calculate_realized_volatility(alt_prices, window=20)
    assert vol > 0.05

    # Short price series returns 0.0 safely
    assert calculate_realized_volatility([100.0, 101.0], window=20) == 0.0


def test_annualized_volatility_returns():
    # Deterministic normal returns with known std
    rng = np.random.default_rng(42)
    daily_std = 0.01
    sample_rets = rng.normal(0.0, daily_std, 1000)
    ann_vol = annualized_volatility(sample_rets, is_returns=True, annualization_factor=252)
    expected_vol = daily_std * math.sqrt(252)
    assert math.isclose(ann_vol, expected_vol, rel_tol=0.10)


def test_simple_moving_average_scalar_and_series():
    prices = [10.0, 20.0, 30.0, 40.0, 50.0]
    sma3 = calculate_sma(prices, window=3)
    assert sma3 == (30.0 + 40.0 + 50.0) / 3.0

    # Test series
    series = simple_moving_average(prices, window=3)
    assert np.isnan(series[0])
    assert np.isnan(series[1])
    assert math.isclose(series[2], 20.0)
    assert math.isclose(series[3], 30.0)
    assert math.isclose(series[4], 40.0)

    # Empty and short series
    assert calculate_sma([], window=5) == 0.0
    assert calculate_sma([10.0, 20.0], window=5) == 20.0


def test_exponential_moving_average():
    prices = [10.0, 12.0, 14.0, 16.0]
    ema_series = exponential_moving_average(prices, span=3)
    assert ema_series[0] == 10.0
    alpha = 2.0 / (3.0 + 1.0)  # 0.5
    assert ema_series[1] == 0.5 * 12.0 + 0.5 * 10.0  # 11.0
    assert ema_series[2] == 0.5 * 14.0 + 0.5 * 11.0  # 12.5
    assert calculate_ema(prices, span=3) == ema_series[-1]


def test_average_true_range():
    highs = [102.0 + i for i in range(20)]
    lows = [98.0 + i for i in range(20)]
    closes = [100.0 + i for i in range(20)]

    atr = calculate_atr(highs, lows, closes, window=14)
    alias_atr = average_true_range(highs, lows, closes, window=14)
    assert atr > 0.0
    assert atr == alias_atr

    # Edge cases
    assert calculate_atr([], [], []) == 0.0
    assert calculate_atr([105.0], [95.0], [100.0], window=14) == 10.0


def test_drawdown_calculation():
    # Path: 100 -> 120 -> 90 -> 110
    prices = [100.0, 120.0, 90.0, 110.0]
    dds = drawdown_series(prices)
    assert dds[0] == 0.0
    assert dds[1] == 0.0
    assert math.isclose(dds[2], (90.0 - 120.0) / 120.0)  # -0.25 (-25%)
    assert math.isclose(dds[3], (110.0 - 120.0) / 120.0)

    max_dd = maximum_drawdown(prices)
    assert math.isclose(max_dd, -0.25)

    cur_dd, calc_max_dd = calculate_drawdown(prices)
    assert math.isclose(cur_dd, (110.0 - 120.0) / 120.0)
    assert math.isclose(calc_max_dd, -0.25)


def test_momentum_12_1():
    prices = [100.0 + i for i in range(260)]
    mom = calculate_momentum_12_1(prices, lookback=252, skip=21)
    p_base = prices[-(252 + 1)]
    p_skip = prices[-(21 + 1)]
    expected = (p_skip / p_base) - 1.0
    assert math.isclose(mom, expected, rel_tol=1e-5)

    # Insufficient length returns 0.0
    assert calculate_momentum_12_1([100.0, 105.0], lookback=252) == 0.0


def test_volatility_scale_factor():
    # Target 12%, realized 40% -> scale 0.30
    scale_high = volatility_scale_factor(realized_vol=0.40, target_vol=0.12)
    assert math.isclose(scale_high, 0.30, rel_tol=1e-3)

    # Realized 8% -> capped at 1.0 (no leverage)
    scale_low = volatility_scale_factor(realized_vol=0.08, target_vol=0.12, max_scale=1.0)
    assert scale_low == 1.0

    # Floor at min_vol (0.05)
    scale_floor = volatility_scale_factor(realized_vol=0.01, target_vol=0.12, min_vol=0.05, max_scale=3.0)
    assert math.isclose(scale_floor, 0.12 / 0.05)


def test_evaluate_drawdown_gate():
    assert evaluate_drawdown_gate(-0.02) == 1.00   # Normal
    assert evaluate_drawdown_gate(-0.07) == 0.50   # Level 1 Caution
    assert evaluate_drawdown_gate(-0.12) == 0.20   # Level 2 Defensive
    assert evaluate_drawdown_gate(-0.18) == 0.00   # Level 3 Circuit Breaker


def test_returns_and_cagr():
    prices = [100.0, 110.0, 121.0]
    log_ret = log_returns(prices)
    simp_ret = simple_returns(prices)

    assert len(log_ret) == 2
    assert math.isclose(simp_ret[0], 0.10)
    assert math.isclose(simp_ret[1], 0.10)
    assert math.isclose(compound_returns(simp_ret), 0.21)

    # CAGR over 253 days (approx 1 year)
    ann_prices = np.linspace(100.0, 110.0, 253)
    assert math.isclose(cagr(ann_prices, periods_per_year=252), 0.10, rel_tol=0.02)


def test_sharpe_and_calmar_ratios():
    # 252 days with steady 0.05% return daily
    daily_rets = np.full(252, 0.0005)
    # Zero variance -> Sharpe should be 0.0 safely
    assert sharpe_ratio(daily_rets) == 0.0

    # Normal distribution
    rng = np.random.default_rng(42)
    fluctuating_rets = rng.normal(0.0005, 0.01, 252)
    sr = sharpe_ratio(fluctuating_rets)
    assert sr > 0.0

    prices = [100.0, 110.0, 95.0, 115.0]
    cr = calmar_ratio(prices)
    assert cr > 0.0
