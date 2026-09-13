"""
strategy_engine.signals.momentum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

12-1 structural relative momentum and Gary Antonacci safe-haven dual momentum.
Implements Feature 8 with strict 2022 duration trap defense and zero lookahead bias.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from strategy_engine.core.models import Bar
from strategy_engine.core import universe
from strategy_engine.core.math_utils import calculate_sma, calculate_momentum_12_1
from strategy_engine.signals.indicators import filter_bars_point_in_time


def extract_closes(prices_or_bars: Union[Sequence[float], Sequence[Bar]]) -> List[float]:
    """Extract float close prices from a sequence of float prices or Bar objects."""
    if not prices_or_bars:
        return []
    if isinstance(prices_or_bars[0], Bar):
        return [b.close for b in prices_or_bars]  # type: ignore[union-attr]
    return [float(p) for p in prices_or_bars]  # type: ignore[arg-type]


def calculate_12_1_momentum(
    prices_or_bars: Union[Sequence[float], Sequence[Bar]],
    lookback: int = 252,
    skip: int = 21,
) -> float:
    """Calculate 12-1 structural momentum: return between (t - lookback) and (t - skip).
    
    Skips the most recent month (21 trading days) to eliminate short-term reversal noise.
    Formula: P[-(skip + 1)] / P[-(lookback + 1)] - 1.0.
    Returns 0.0 if insufficient history (< lookback + 1).
    """
    closes = extract_closes(prices_or_bars)
    return calculate_momentum_12_1(closes, lookback=lookback, skip=skip)


def compute_universe_momentum(
    market_data: Dict[str, List[Bar]],
    symbols: Optional[List[str]] = None,
    lookback: int = 252,
    skip: int = 21,
) -> Dict[str, float]:
    """Compute 12-1 structural momentum scores across universe symbols."""
    target_symbols = symbols if symbols is not None else list(market_data.keys())
    scores: Dict[str, float] = {}
    for sym in target_symbols:
        bars = market_data.get(sym, [])
        scores[sym] = calculate_12_1_momentum(bars, lookback=lookback, skip=skip)
    return scores


def evaluate_absolute_momentum(
    symbol: str,
    market_data: Dict[str, List[Bar]],
    cash_symbol: str = "SHV",
    lookback: int = 252,
    skip: int = 21,
    sma_window: int = 200,
) -> bool:
    """Evaluate Gary Antonacci absolute momentum hurdle.
    
    Returns True if:
    1. Mom_12-1(symbol) > Mom_12-1(cash_symbol)
    2. Current Price(symbol) > SMA_200(symbol)
    """
    asset_bars = market_data.get(symbol, [])
    if len(asset_bars) < sma_window:
        return False

    cash_bars = market_data.get(cash_symbol, [])
    asset_closes = extract_closes(asset_bars)
    cash_closes = extract_closes(cash_bars)

    mom_asset = calculate_12_1_momentum(asset_closes, lookback=lookback, skip=skip)
    mom_cash = calculate_12_1_momentum(cash_closes, lookback=lookback, skip=skip) if cash_closes else 0.0

    sma200 = calculate_sma(asset_closes, sma_window)
    current_price = asset_closes[-1]

    return bool(mom_asset > mom_cash and current_price > sma200)


def rank_momentum_candidates(
    momentum_scores: Dict[str, float],
    eligible_symbols: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """Rank assets descending by momentum score with deterministic alphabetical tie-breaking."""
    pool = (
        {sym: momentum_scores.get(sym, 0.0) for sym in eligible_symbols}
        if eligible_symbols is not None
        else momentum_scores
    )
    # Primary sort: momentum score descending (-score)
    # Secondary sort: ticker symbol ascending alphabetically
    return sorted(pool.items(), key=lambda item: (-item[1], item[0]))


def select_top_sectors(
    market_data: Dict[str, List[Bar]],
    n_top: int = 2,
    require_absolute_hurdle: bool = True,
    cash_symbol: str = "SHV",
) -> List[str]:
    """Select top N SPDR Sector ETFs by 12-1 structural momentum passing absolute hurdle."""
    sector_symbols = universe.get_sector_symbols()
    scores = compute_universe_momentum(market_data, symbols=sector_symbols)
    ranked = rank_momentum_candidates(scores, eligible_symbols=sector_symbols)

    selected: List[str] = []
    for sym, _ in ranked:
        if require_absolute_hurdle:
            if evaluate_absolute_momentum(sym, market_data, cash_symbol=cash_symbol):
                selected.append(sym)
        else:
            selected.append(sym)
        if len(selected) >= n_top:
            break
    return selected


def select_top_growth_leaders(
    market_data: Dict[str, List[Bar]],
    n_top: int = 1,
    require_absolute_hurdle: bool = True,
    cash_symbol: str = "SHV",
) -> List[str]:
    """Select top N Megacap Growth Leaders by 12-1 structural momentum."""
    megacap_symbols = universe.get_megacap_symbols()
    scores = compute_universe_momentum(market_data, symbols=megacap_symbols)
    ranked = rank_momentum_candidates(scores, eligible_symbols=megacap_symbols)

    selected: List[str] = []
    for sym, _ in ranked:
        if require_absolute_hurdle:
            if evaluate_absolute_momentum(sym, market_data, cash_symbol=cash_symbol):
                selected.append(sym)
        else:
            selected.append(sym)
        if len(selected) >= n_top:
            break
    return selected


def evaluate_safe_haven_qualification(
    symbol: str,
    bars: Sequence[Bar],
    sma_window: int = 200,
    lookback: int = 252,
    skip: int = 21,
) -> bool:
    """Evaluate qualification hurdle for safe-haven assets.
    
    - CASH / SHV / BIL: Always qualifies (True).
    - TLT / GLD: Requires Price > SMA200 AND Mom_12-1 > 0.0.
    """
    sym = symbol.upper()
    if sym in ("SHV", "BIL", "CASH", "USD"):
        return True

    closes = extract_closes(bars)
    if len(closes) < sma_window:
        return False

    sma = calculate_sma(closes, sma_window)
    mom = calculate_12_1_momentum(closes, lookback=lookback, skip=skip)
    current_p = closes[-1]
    return bool(current_p > sma and mom > 0.0)


def compute_safe_haven_weights(
    defensive_capital: float,
    qualifications: Dict[str, bool],
    tlt_symbol: str = "TLT",
    gld_symbol: str = "GLD",
    cash_symbol: str = "SHV",
) -> Dict[str, float]:
    """Compute safe haven weights given qualification booleans.
    
    If TLT qualifies: 40% of defensive capital.
    If GLD qualifies: 40% of defensive capital.
    Remainder: routed to cash_symbol (SHV).
    """
    if defensive_capital <= 1e-7:
        return {cash_symbol: 0.0}

    tlt_pass = qualifications.get(tlt_symbol, False)
    gld_pass = qualifications.get(gld_symbol, False)

    w_tlt = 0.40 * defensive_capital if tlt_pass else 0.0
    w_gld = 0.40 * defensive_capital if gld_pass else 0.0
    w_cash = defensive_capital - w_tlt - w_gld

    weights: Dict[str, float] = {}
    if w_tlt > 1e-6:
        weights[tlt_symbol] = w_tlt
    if w_gld > 1e-6:
        weights[gld_symbol] = w_gld
    weights[cash_symbol] = max(0.0, w_cash)
    return weights


def evaluate_safe_haven_dual_momentum(
    market_data: Dict[str, List[Bar]],
    defensive_capital: float = 1.0,
    tlt_symbol: str = "TLT",
    gld_symbol: str = "GLD",
    cash_symbol: str = "SHV",
    sma_window: int = 200,
) -> Dict[str, float]:
    """Antonacci Safe-Haven Dual Momentum Allocator with 2022 Duration Trap Protection.
    
    Evaluates TLT and GLD against trend (> 200 SMA) and absolute momentum (> 0).
    If TLT fails (e.g. 2022 rate shock), eliminates TLT (strictly 0.0 weight)
    and routes 100% of duration defensive capital to ultra-short cash (SHV/BIL) and GLD.
    """
    if defensive_capital <= 1e-7:
        return {cash_symbol: 0.0}

    tlt_bars = market_data.get(tlt_symbol, [])
    tlt_pass = evaluate_safe_haven_qualification(tlt_symbol, tlt_bars, sma_window=sma_window)

    gld_bars = market_data.get(gld_symbol, [])
    gld_pass = evaluate_safe_haven_qualification(gld_symbol, gld_bars, sma_window=sma_window)

    quals = {tlt_symbol: tlt_pass, gld_symbol: gld_pass}
    return compute_safe_haven_weights(
        defensive_capital=defensive_capital,
        qualifications=quals,
        tlt_symbol=tlt_symbol,
        gld_symbol=gld_symbol,
        cash_symbol=cash_symbol,
    )
