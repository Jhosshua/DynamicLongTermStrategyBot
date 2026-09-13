"""
strategy_engine.allocator.rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic regime allocation matrix, continuous risk overlay integration,
and strict target weight normalization (= 1.0 within 1e-5).
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Dict, List, Optional

from strategy_engine.core.models import Bar, MarketRegime, SignalSnapshot, TargetAllocation
from strategy_engine.core import universe
from strategy_engine.signals.indicators import filter_bars_point_in_time
from strategy_engine.signals.momentum import (
    compute_universe_momentum,
    evaluate_safe_haven_dual_momentum,
    select_top_sectors,
)


def normalize_target_weights(
    weights: Dict[str, float],
    cash_symbol: str = "SHV",
) -> Dict[str, float]:
    """Strictly normalize target weights to sum to 1.0 within 1e-5.
    
    Filters out near-zero micro-weights (< 1e-7), clamps negative values to 0.0,
    and absorbs floating point residual deterministically into cash_symbol.
    """
    cleaned = {
        sym: max(0.0, float(w))
        for sym, w in weights.items()
        if w > 1e-7 and not math.isnan(w) and not math.isinf(w)
    }
    total = sum(cleaned.values())
    if total <= 1e-9:
        return {cash_symbol: 1.0}

    normalized = {sym: w / total for sym, w in cleaned.items()}
    residual = 1.0 - sum(normalized.values())
    normalized[cash_symbol] = max(0.0, normalized.get(cash_symbol, 0.0) + residual)

    final_sum = sum(normalized.values())
    assert math.isclose(final_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5), f"Sum violated: {final_sum}"
    return normalized


def get_regime_base_weights(
    regime: MarketRegime,
    top_sectors: Optional[List[str]] = None,
    safe_haven_weights: Optional[Dict[str, float]] = None,
    cash_symbol: str = "SHV",
) -> Dict[str, float]:
    """Get nominal unscaled base allocation weights for a specific MarketRegime."""
    if regime == MarketRegime.BULL_AGGRESSIVE:
        sectors = top_sectors or ["XLK", "XLY"]
        w = {"QQQ": 0.50, "SPY": 0.20}
        if len(sectors) >= 2:
            w[sectors[0]] = 0.15
            w[sectors[1]] = 0.15
        elif len(sectors) == 1:
            w[sectors[0]] = 0.15
            w[cash_symbol] = 0.15
        else:
            w[cash_symbol] = 0.30
        return normalize_target_weights(w, cash_symbol=cash_symbol)

    elif regime == MarketRegime.BULL_NORMAL:
        sectors = top_sectors or ["XLK"]
        w = {"QQQ": 0.35, "SPY": 0.25}
        if len(sectors) >= 1:
            w[sectors[0]] = 0.20
        else:
            w[cash_symbol] = 0.20
        sh = safe_haven_weights or {cash_symbol: 1.0}
        for s, weight in sh.items():
            w[s] = w.get(s, 0.0) + weight * 0.20
        return normalize_target_weights(w, cash_symbol=cash_symbol)

    elif regime == MarketRegime.CORRECTION_FRAGILE:
        def_sector = top_sectors[0] if (top_sectors and len(top_sectors) > 0) else "XLV"
        w = {def_sector: 0.20}
        sh = safe_haven_weights or {cash_symbol: 1.0}
        for s, weight in sh.items():
            w[s] = w.get(s, 0.0) + weight * 0.80
        return normalize_target_weights(w, cash_symbol=cash_symbol)

    elif regime == MarketRegime.BEAR_CRISIS:
        sh = safe_haven_weights or {cash_symbol: 1.0}
        return normalize_target_weights(sh, cash_symbol=cash_symbol)

    else:  # STALE_DATA_HOLD
        return {cash_symbol: 1.0}


def compute_deterministic_allocation(
    signals: SignalSnapshot,
    market_data: Dict[str, List[Bar]],
    current_time: Optional[datetime] = None,
    cash_symbol: str = "SHV",
) -> TargetAllocation:
    """Compute exact TargetAllocation based on MarketRegime, risk overlays, and safe-haven momentum."""
    if current_time is None and signals is not None:
        current_time = signals.timestamp
    ts = current_time or (signals.timestamp if signals is not None else datetime.now(timezone.utc))
    regime = signals.regime
    s_vol = signals.vol_scale_factor
    g_dd = signals.indicators.get("drawdown_gate", 1.0)
    circuit_breaker = signals.circuit_breaker_active

    # Strict point-in-time filtration of market data
    if current_time is not None and market_data:
        pit_data = {
            sym: filter_bars_point_in_time(bars, current_time)
            for sym, bars in market_data.items()
        }
    else:
        pit_data = market_data

    # Safety override: Stale hold freezes new allocations into cash
    if regime == MarketRegime.STALE_DATA_HOLD:
        return TargetAllocation(
            timestamp=ts,
            regime=regime,
            weights={cash_symbol: 1.0},
            cash_weight=1.0,
            rationale="STALE_DATA_HOLD: Upstream disconnected or data stale, capital preserved in cash.",
        )

    # Effective risk multiplier compounding volatility scale factor and drawdown gate
    m_risk = min(1.0, s_vol * g_dd)
    if circuit_breaker:
        m_risk = min(m_risk, 0.50)  # ATR stop cuts equity exposure by 50% immediately

    raw_weights: Dict[str, float] = {}

    if regime == MarketRegime.BULL_AGGRESSIVE:
        top_sectors = select_top_sectors(
            pit_data, n_top=2, require_absolute_hurdle=True, cash_symbol=cash_symbol
        )
        eq_weights = {"QQQ": 0.50, "SPY": 0.20}
        if len(top_sectors) == 2:
            eq_weights[top_sectors[0]] = 0.15
            eq_weights[top_sectors[1]] = 0.15
        elif len(top_sectors) == 1:
            eq_weights[top_sectors[0]] = 0.15
            raw_weights[cash_symbol] = 0.15
        else:
            raw_weights[cash_symbol] = 0.30

        for sym, w in eq_weights.items():
            raw_weights[sym] = w * m_risk

        defensive_capital = 1.0 - sum(raw_weights.values())
        if defensive_capital > 1e-6:
            safe_havens = evaluate_safe_haven_dual_momentum(
                pit_data, defensive_capital=defensive_capital, cash_symbol=cash_symbol
            )
            for s, w in safe_havens.items():
                raw_weights[s] = raw_weights.get(s, 0.0) + w

        rationale = (
            f"BULL_AGGRESSIVE: Low vol, broad breadth. QQQ=50%, Sectors={top_sectors}, "
            f"SPY=20%, m_risk={m_risk:.2f}"
        )

    elif regime == MarketRegime.BULL_NORMAL:
        top_sectors = select_top_sectors(
            pit_data, n_top=1, require_absolute_hurdle=True, cash_symbol=cash_symbol
        )
        eq_weights = {"QQQ": 0.35, "SPY": 0.25}
        if len(top_sectors) >= 1:
            eq_weights[top_sectors[0]] = 0.20
        else:
            raw_weights[cash_symbol] = 0.20

        for sym, w in eq_weights.items():
            raw_weights[sym] = w * m_risk

        defensive_capital = 1.0 - sum(raw_weights.values())
        safe_havens = evaluate_safe_haven_dual_momentum(
            pit_data, defensive_capital=defensive_capital, cash_symbol=cash_symbol
        )
        for s, w in safe_havens.items():
            raw_weights[s] = raw_weights.get(s, 0.0) + w

        rationale = (
            f"BULL_NORMAL: Primary trend intact. QQQ=35%, SPY=25%, TopSector={top_sectors}, "
            f"Defensive={defensive_capital:.2f}"
        )

    elif regime == MarketRegime.CORRECTION_FRAGILE:
        defensive_sectors = universe.get_defensive_sectors()
        scores = compute_universe_momentum(pit_data, symbols=defensive_sectors)
        ranked_def = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        best_def = ranked_def[0][0] if ranked_def else "XLV"

        raw_weights[best_def] = 0.20 * m_risk
        defensive_capital = 1.0 - raw_weights[best_def]
        safe_havens = evaluate_safe_haven_dual_momentum(
            pit_data, defensive_capital=defensive_capital, cash_symbol=cash_symbol
        )
        for s, w in safe_havens.items():
            raw_weights[s] = raw_weights.get(s, 0.0) + w

        rationale = (
            f"CORRECTION_FRAGILE: Elevated vol/pullback. DefensiveEq={best_def}({raw_weights[best_def]:.2f}), "
            f"SafeHaven/Cash={defensive_capital:.2f}"
        )

    elif regime == MarketRegime.BEAR_CRISIS:
        safe_havens = evaluate_safe_haven_dual_momentum(
            pit_data, defensive_capital=1.0, cash_symbol=cash_symbol
        )
        raw_weights = safe_havens
        rationale = (
            "BEAR_CRISIS: Structural breakdown / vol spike. 0% equity, "
            "100% defensive capital routed via Antonacci dual momentum."
        )

    else:
        raw_weights = {cash_symbol: 1.0}
        rationale = f"Unknown regime {regime}: defaulting to 100% cash preservation."

    normalized_weights = normalize_target_weights(raw_weights, cash_symbol=cash_symbol)
    computed_cash = (
        normalized_weights.get("SHV", 0.0)
        + normalized_weights.get("BIL", 0.0)
        + normalized_weights.get("CASH", 0.0)
    )

    return TargetAllocation(
        timestamp=ts,
        regime=regime,
        weights=normalized_weights,
        cash_weight=computed_cash,
        rationale=rationale,
    )
