"""
strategy_engine.signals.indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-timeframe quantitative risk and trend indicators:
- Rolling realized volatility targeting (target 12%, min 5%, max scale 1.0)
- Dual-SMA trend filters (50-day and 200-day) for SPY and QQQ
- Dynamic ATR Keltner lower band stops (Close < SMA50 - 2.0 * ATR14)
- Trailing peak-to-trough drawdown defense gates with stateful 3-day recovery hysteresis
- Market breadth (% > 50 SMA / 200 SMA)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from strategy_engine.core import math_utils
from strategy_engine.core.models import Bar


def filter_bars_point_in_time(
    bars: Sequence[Bar],
    current_time: datetime,
) -> List[Bar]:
    """Filter and sort bars strictly up to current_time to prevent lookahead bias."""
    if not bars:
        return []
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    filtered = [
        b
        for b in bars
        if (b.timestamp.replace(tzinfo=timezone.utc) if b.timestamp.tzinfo is None else b.timestamp)
        <= current_time
    ]
    filtered.sort(
        key=lambda b: (
            b.timestamp.replace(tzinfo=timezone.utc) if b.timestamp.tzinfo is None else b.timestamp
        )
    )
    return filtered


def extract_closes(prices_or_bars: Union[Sequence[float], Sequence[Bar]]) -> List[float]:
    """Extract float close prices from a sequence of float prices or Bar objects."""
    if not prices_or_bars:
        return []
    if isinstance(prices_or_bars[0], Bar):
        return [b.close for b in prices_or_bars]  # type: ignore[union-attr]
    return [float(p) for p in prices_or_bars]  # type: ignore[arg-type]


def compute_realized_volatility(
    prices_or_bars: Union[Sequence[float], Sequence[Bar]],
    window: int = 20,
    annualization_factor: int = 252,
) -> float:
    """Calculate annualized realized volatility over trailing window.
    
    Returns 0.0 if insufficient bars or non-positive prices.
    """
    closes = extract_closes(prices_or_bars)
    if len(closes) < window + 1 or window <= 1:
        return 0.0
    return math_utils.calculate_realized_volatility(
        closes, window=window, annualization_factor=annualization_factor
    )


def compute_volatility_scale_factor(
    realized_vol: float,
    target_vol: float = 0.12,
    min_vol: float = 0.05,
    max_scale: float = 1.0,
) -> float:
    """Calculate volatility scaling factor S_vol = min(max_scale, target_vol / max(realized_vol, min_vol))."""
    return math_utils.volatility_scale_factor(
        realized_vol=realized_vol,
        target_vol=target_vol,
        min_vol=min_vol,
        max_scale=max_scale,
    )


def compute_sma(
    prices_or_bars: Union[Sequence[float], Sequence[Bar]],
    window: int,
) -> float:
    """Calculate simple moving average scalar over trailing window."""
    closes = extract_closes(prices_or_bars)
    if not closes:
        return 0.0
    return math_utils.calculate_sma(closes, window=window)


def compute_atr(
    bars: Sequence[Bar],
    window: int = 14,
) -> float:
    """Calculate Average True Range over trailing window."""
    if not bars:
        return 0.0
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    closes = [b.close for b in bars]
    return math_utils.calculate_atr(highs, lows, closes, window=window)


def compute_keltner_lower_band(
    bars: Sequence[Bar],
    sma_window: int = 50,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
) -> Tuple[float, float, float]:
    """Calculate Keltner lower band: Band_lower = SMA50 - atr_multiplier * ATR14.
    
    Returns: (sma, atr, lower_band)
    """
    if not bars:
        return 0.0, 0.0, 0.0
    closes = [b.close for b in bars]
    sma = compute_sma(closes, window=sma_window)
    atr = compute_atr(bars, window=atr_window)
    lower_band = sma - (atr_multiplier * atr)
    return float(sma), float(atr), float(lower_band)


def check_circuit_breaker(
    bars: Sequence[Bar],
    sma_window: int = 50,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
) -> Tuple[bool, Dict[str, float]]:
    """Check if latest close breached the lower Keltner band.
    
    Condition: Close < SMA50 - atr_multiplier * ATR14
    Returns: (is_active, metrics_dict)
    """
    if not bars:
        return False, {"sma50": 0.0, "atr14": 0.0, "lower_band": 0.0, "close": 0.0, "stop_distance_pct": 0.0}
    sma, atr, lower_band = compute_keltner_lower_band(
        bars, sma_window=sma_window, atr_window=atr_window, atr_multiplier=atr_multiplier
    )
    latest_close = bars[-1].close
    is_active = bool(latest_close < lower_band)
    metrics = {
        "sma50": sma,
        "atr14": atr,
        "lower_band": lower_band,
        "close": float(latest_close),
        "stop_distance_pct": float((latest_close - lower_band) / lower_band) if lower_band > 0 else 0.0,
    }
    return is_active, metrics


def evaluate_trend_filter(
    bars: Sequence[Bar],
    symbol: str,
    fast_window: int = 50,
    slow_window: int = 200,
) -> Dict[str, Any]:
    """Evaluate dual-SMA trend filter and golden/death cross state.
    
    Discrete Trend Score T_t:
    +1.0: Confirmed Bull (Price > SMA50 and SMA50 > SMA200)
    +0.5: Pullback in Bull (Price > SMA200 and Price <= SMA50)
    -0.5: Breakdown Warning (Price <= SMA200 and SMA50 > SMA200)
    -1.0: Confirmed Bear (Price <= SMA200 and SMA50 <= SMA200)
    """
    if not bars:
        return {
            "symbol": symbol,
            "price": 0.0,
            "sma50": 0.0,
            "sma200": 0.0,
            "price_above_sma50": False,
            "price_above_sma200": False,
            "is_golden_cross": False,
            "trend_score": 0.0,
            "pct_above_sma50": 0.0,
            "pct_above_sma200": 0.0,
        }
    closes = [b.close for b in bars]
    current_price = closes[-1]
    sma50 = compute_sma(closes, window=fast_window)
    sma200 = compute_sma(closes, window=slow_window)

    above_50 = bool(current_price > sma50)
    above_200 = bool(current_price > sma200)
    golden_cross = bool(sma50 > sma200)

    if above_50 and golden_cross:
        trend_score = 1.0
    elif above_200 and not above_50:
        trend_score = 0.5
    elif not above_200 and golden_cross:
        trend_score = -0.5
    else:
        trend_score = -1.0

    pct_above_50 = float((current_price - sma50) / sma50) if sma50 > 0 else 0.0
    pct_above_200 = float((current_price - sma200) / sma200) if sma200 > 0 else 0.0

    return {
        "symbol": symbol,
        "price": float(current_price),
        "sma50": float(sma50),
        "sma200": float(sma200),
        "price_above_sma50": above_50,
        "price_above_sma200": above_200,
        "is_golden_cross": golden_cross,
        "trend_score": trend_score,
        "pct_above_sma50": pct_above_50,
        "pct_above_sma200": pct_above_200,
    }


def evaluate_drawdown_gate(drawdown_pct: float) -> float:
    """Step function for trailing peak-to-trough drawdown defense:
    - DD > -5%: 1.00 (Normal)
    - -10% < DD <= -5%: 0.50 (Caution: 50% equity cut)
    - -15% < DD <= -10%: 0.20 (Defensive: 80% safe haven / cash)
    - DD <= -15%: 0.00 (Circuit breaker: 100% cash)
    """
    return math_utils.evaluate_drawdown_gate(drawdown_pct)


@dataclass
class DrawdownDefenseTracker:
    """Stateful tracker for trailing drawdown defense with 3-day recovery hysteresis."""
    peak_equity: Optional[float] = None
    current_equity: Optional[float] = None
    current_drawdown: float = 0.0
    active_gate_multiplier: float = 1.00
    consecutive_recovery_days: int = 0
    recovery_target_days: int = 3
    in_recovery_lockout: bool = False
    equity_history: List[float] = field(default_factory=list)

    def prime(self, equity_curve: Sequence[float]) -> None:
        """Prime tracker with historical equity curve to establish peak equity and history."""
        if not equity_curve:
            return
        self.equity_history = [float(x) for x in equity_curve]
        max_eq = max(self.equity_history)
        if self.peak_equity is None or max_eq > self.peak_equity:
            self.peak_equity = float(max_eq)
        self.current_equity = float(self.equity_history[-1])
        raw_dd = (
            (self.current_equity - self.peak_equity) / self.peak_equity
            if self.peak_equity > 0.0
            else 0.0
        )
        self.current_drawdown = min(0.0, float(raw_dd))
        self.active_gate_multiplier = evaluate_drawdown_gate(self.current_drawdown)
        if self.active_gate_multiplier < 1.00:
            self.in_recovery_lockout = True

    def update(
        self,
        current_equity: float,
        benchmark_price: float,
        benchmark_sma50: float,
        recovery_buffer_multiplier: float = 1.00,
    ) -> float:
        """Update tracker state with latest portfolio equity and benchmark SMA50 close.
        
        Parameters:
            current_equity: Current portfolio equity (or benchmark close if uninitialized)
            benchmark_price: Current closing price of benchmark (e.g. SPY)
            benchmark_sma50: 50-day SMA of benchmark
            recovery_buffer_multiplier: Buffer multiplier required above SMA50 (default 1.00)
            
        Returns:
            Effective drawdown gate multiplier in {1.00, 0.50, 0.20, 0.00}
        """
        self.current_equity = current_equity
        self.equity_history.append(current_equity)
        if self.peak_equity is None or self.peak_equity <= 0.0:
            self.peak_equity = current_equity
        elif current_equity > self.peak_equity:
            self.peak_equity = current_equity

        raw_dd = (
            (current_equity - self.peak_equity) / self.peak_equity
            if (self.peak_equity is not None and self.peak_equity > 0.0)
            else 0.0
        )
        self.current_drawdown = min(0.0, float(raw_dd))
        raw_gate = evaluate_drawdown_gate(self.current_drawdown)

        # Hysteresis confirmation: benchmark close above SMA50 * recovery_buffer_multiplier
        recovery_threshold = benchmark_sma50 * recovery_buffer_multiplier
        if benchmark_price > recovery_threshold:
            self.consecutive_recovery_days += 1
        else:
            self.consecutive_recovery_days = 0

        can_reenter = bool(self.consecutive_recovery_days >= self.recovery_target_days)

        # If drawdown triggered defensive gate (Level 1, 2, or 3), activate recovery lockout
        if raw_gate < 1.00:
            self.in_recovery_lockout = True

        if self.in_recovery_lockout:
            if can_reenter:
                # Recovery confirmed by consecutive closes above threshold
                self.active_gate_multiplier = raw_gate
                if raw_gate >= 1.00:
                    self.in_recovery_lockout = False
            else:
                # Lockout active: cannot increase equity multiplier; can only de-risk further
                self.active_gate_multiplier = min(self.active_gate_multiplier, raw_gate)
        else:
            self.active_gate_multiplier = raw_gate

        return self.active_gate_multiplier


def compute_market_breadth(
    market_data: Dict[str, List[Bar]],
    symbols: Optional[List[str]] = None,
    current_time: Optional[datetime] = None,
) -> Dict[str, float]:
    """Compute market breadth (% > 50 SMA and % > 200 SMA) across universe symbols."""
    target_symbols = symbols or list(market_data.keys())
    if not target_symbols:
        return {"breadth_50": 1.0, "breadth_200": 1.0, "breadth_regime_factor": 1.0, "symbols_evaluated": 0.0}

    above_50 = 0
    above_200 = 0
    total = 0

    for sym in target_symbols:
        bars = market_data.get(sym, [])
        if current_time is not None:
            bars = filter_bars_point_in_time(bars, current_time)
        if not bars:
            continue
        closes = [b.close for b in bars]
        price = closes[-1]
        sma50 = compute_sma(closes, 50)
        sma200 = compute_sma(closes, 200)

        if price > sma50:
            above_50 += 1
        if price > sma200:
            above_200 += 1
        total += 1

    if total == 0:
        return {"breadth_50": 1.0, "breadth_200": 1.0, "breadth_regime_factor": 1.0, "symbols_evaluated": 0.0}

    b50 = above_50 / total
    b200 = above_200 / total

    if b50 >= 0.60:
        factor = 1.00
    elif b50 >= 0.40:
        factor = 0.75
    else:
        factor = 0.50

    return {
        "breadth_50": float(b50),
        "breadth_200": float(b200),
        "breadth_regime_factor": float(factor),
        "symbols_evaluated": float(total),
    }
