"""
strategy_engine.core.math_utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic numerical and statistical utilities for quantitative signals,
risk management, and portfolio evaluation. Free of lookahead bias.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np


def log_returns(prices: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Calculate logarithmic returns: r_t = ln(P_t / P_{t-1})."""
    arr = np.asarray(prices, dtype=np.float64)
    if len(arr) < 2:
        return np.array([], dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError("Prices must be strictly positive for log returns calculation")
    return np.diff(np.log(arr))


def simple_returns(prices: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Calculate simple arithmetic returns: R_t = (P_t - P_{t-1}) / P_{t-1}."""
    arr = np.asarray(prices, dtype=np.float64)
    if len(arr) < 2:
        return np.array([], dtype=np.float64)
    if np.any(arr[:-1] == 0):
        raise ZeroDivisionError("Preceding price cannot be zero for return calculation")
    return np.diff(arr) / arr[:-1]


def compound_returns(returns: Union[Sequence[float], np.ndarray]) -> float:
    """Calculate cumulative compound return: prod(1 + R_t) - 1."""
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def calculate_realized_volatility(
    prices: Union[Sequence[float], np.ndarray],
    window: int = 20,
    annualization_factor: int = 252,
) -> float:
    """Calculate annualized realized volatility from price series over the last `window` periods."""
    arr = np.asarray(prices, dtype=np.float64)
    if len(arr) < window + 1 or window <= 1:
        return 0.0
    
    sub_prices = arr[-(window + 1):]
    if np.any(sub_prices <= 0):
        return 0.0
    
    rets = np.diff(np.log(sub_prices))
    variance = float(np.var(rets, ddof=1))
    return float(math.sqrt(max(0.0, variance * annualization_factor)))


def annualized_volatility(
    returns_or_prices: Union[Sequence[float], np.ndarray],
    is_returns: bool = True,
    annualization_factor: int = 252,
) -> float:
    """Compute annualized standard deviation from return series or price series."""
    arr = np.asarray(returns_or_prices, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    if not is_returns:
        arr = log_returns(arr)
        if len(arr) < 2:
            return 0.0
    std = float(np.std(arr, ddof=1))
    return float(std * math.sqrt(annualization_factor))


def calculate_sma(prices: Union[Sequence[float], np.ndarray], window: int) -> float:
    """Calculate the scalar Simple Moving Average over the trailing `window` prices."""
    arr = np.asarray(prices, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    if len(arr) < window or window <= 0:
        return float(arr[-1])
    return float(np.mean(arr[-window:]))


def simple_moving_average(prices: Union[Sequence[float], np.ndarray], window: int) -> np.ndarray:
    """Compute moving average series of length N. Early elements (t < window - 1) are np.nan."""
    arr = np.asarray(prices, dtype=np.float64)
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or n < window:
        return result
    
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    result[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    return result


def calculate_ema(prices: Union[Sequence[float], np.ndarray], span: int) -> float:
    """Calculate current Exponential Moving Average scalar value."""
    ema_arr = exponential_moving_average(prices, span)
    return float(ema_arr[-1]) if len(ema_arr) > 0 else 0.0


def exponential_moving_average(prices: Union[Sequence[float], np.ndarray], span: int) -> np.ndarray:
    """Vectorized exponential moving average with alpha = 2 / (span + 1)."""
    arr = np.asarray(prices, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=np.float64)
    if span <= 1:
        return arr.copy()

    alpha = 2.0 / (span + 1.0)
    ema = np.zeros(n, dtype=np.float64)
    ema[0] = arr[0]
    for t in range(1, n):
        ema[t] = alpha * arr[t] + (1.0 - alpha) * ema[t - 1]
    return ema


def calculate_atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    window: int = 14,
) -> float:
    """Calculate Average True Range over trailing `window` bars."""
    h_arr = np.asarray(highs, dtype=np.float64)
    l_arr = np.asarray(lows, dtype=np.float64)
    c_arr = np.asarray(closes, dtype=np.float64)
    n = len(c_arr)

    if n == 0:
        return 0.0
    if n < window + 1:
        return float(h_arr[-1] - l_arr[-1]) if (len(h_arr) > 0 and len(l_arr) > 0) else 0.0

    trs = np.zeros(window, dtype=np.float64)
    start_idx = n - window
    for idx, i in enumerate(range(start_idx, n)):
        h = h_arr[i]
        l = l_arr[i]
        prev_c = c_arr[i - 1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs[idx] = tr
    return float(np.mean(trs))


def average_true_range(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    window: int = 14,
) -> float:
    """Alias for calculate_atr."""
    return calculate_atr(highs, lows, closes, window=window)


def drawdown_series(prices: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Compute the drawdown series: (P_t - HWM_t) / HWM_t, bounded in [-1.0, 0.0]."""
    arr = np.asarray(prices, dtype=np.float64)
    if len(arr) == 0:
        return np.array([], dtype=np.float64)
    hwm = np.maximum.accumulate(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(hwm > 0, (arr - hwm) / hwm, 0.0)
    return np.minimum(dd, 0.0)


def maximum_drawdown(prices: Union[Sequence[float], np.ndarray]) -> float:
    """Compute the maximum peak-to-trough drawdown (non-positive float, e.g. -0.25)."""
    dds = drawdown_series(prices)
    if len(dds) == 0:
        return 0.0
    return float(np.min(dds))


def calculate_drawdown(equity_curve: Union[Sequence[float], np.ndarray]) -> Tuple[float, float]:
    """Calculate current drawdown and maximum drawdown from equity curve.
    
    Returns: (current_drawdown, max_drawdown), both <= 0.0.
    """
    arr = np.asarray(equity_curve, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    peak = arr[0]
    max_dd = 0.0
    for val in arr:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (val - peak) / peak
            if dd < max_dd:
                max_dd = dd
    current_dd = (arr[-1] - peak) / peak if peak > 0 else 0.0
    return float(current_dd), float(max_dd)


def calculate_momentum_12_1(
    prices: Union[Sequence[float], np.ndarray],
    lookback: int = 252,
    skip: int = 21,
) -> float:
    """Calculate 12-1 structural momentum: return between (t - lookback) and (t - skip).
    
    Skips the most recent month (21 trading days) to avoid short-term reversal noise.
    Formula: P_{-(skip + 1)} / P_{-(lookback + 1)} - 1.0.
    """
    arr = np.asarray(prices, dtype=np.float64)
    if len(arr) < lookback + 1:
        return 0.0
    p_base = arr[-(lookback + 1)]
    p_skip = arr[-(skip + 1)]
    if p_base <= 0:
        return 0.0
    return float((p_skip / p_base) - 1.0)


def volatility_scale_factor(
    realized_vol: float,
    target_vol: float = 0.12,
    min_vol: float = 0.05,
    max_scale: float = 1.0,
) -> float:
    """Calculate position scaling factor S_vol = min(max_scale, target_vol / max(realized_vol, min_vol)).
    
    Fail-safe: returns 0.0 on NaN or infinite volatility inputs to prevent unintended leverage.
    """
    if math.isnan(realized_vol) or math.isinf(realized_vol):
        return 0.0
    vol = max(realized_vol, min_vol, 1e-6)
    scale = target_vol / vol
    return float(min(max_scale, scale))


def evaluate_drawdown_gate(drawdown_pct: float) -> float:
    """Step function for trailing peak-to-trough drawdown defense:
    - DD > -5%: 1.00 (Normal)
    - -10% < DD <= -5%: 0.50 (Caution: 50% equity cut)
    - -15% < DD <= -10%: 0.20 (Defensive: 80% safe haven / cash)
    - DD <= -15%: 0.00 (Circuit breaker: 100% cash)
    """
    if drawdown_pct > -0.05:
        return 1.00
    elif drawdown_pct > -0.10:
        return 0.50
    elif drawdown_pct > -0.15:
        return 0.20
    else:
        return 0.00


def cagr(prices: Union[Sequence[float], np.ndarray], periods_per_year: int = 252) -> float:
    """Calculate Compound Annual Growth Rate (CAGR)."""
    arr = np.asarray(prices, dtype=np.float64)
    n = len(arr)
    if n < 2 or arr[0] <= 0 or arr[-1] <= 0:
        return 0.0
    years = (n - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((arr[-1] / arr[0]) ** (1.0 / years) - 1.0)


def sharpe_ratio(
    returns: Union[Sequence[float], np.ndarray],
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio."""
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    excess = arr - (rf / periods_per_year)
    mean = np.mean(excess)
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(math.sqrt(periods_per_year) * (mean / std))


def calmar_ratio(
    prices: Union[Sequence[float], np.ndarray],
    periods_per_year: int = 252,
) -> float:
    """Calculate Calmar ratio: CAGR / abs(MaxDD)."""
    ann_ret = cagr(prices, periods_per_year)
    max_dd = abs(maximum_drawdown(prices))
    if max_dd < 1e-6:
        return 0.0
    return float(ann_ret / max_dd)
