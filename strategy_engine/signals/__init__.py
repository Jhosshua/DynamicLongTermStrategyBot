"""
strategy_engine.signals
~~~~~~~~~~~~~~~~~~~~~~~

Multi-timeframe quantitative signal engine, momentum ranking,
and regime detection for the AlpacaRelay Strategy Engine.
"""

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
    compute_safe_haven_weights,
    compute_universe_momentum,
    evaluate_absolute_momentum,
    evaluate_safe_haven_dual_momentum,
    evaluate_safe_haven_qualification,
    rank_momentum_candidates,
    select_top_growth_leaders,
    select_top_sectors,
)
from strategy_engine.signals.regime_detector import (
    RegimeDetector,
    SignalEngine,
)

__all__ = [
    # Indicators
    "DrawdownDefenseTracker",
    "check_circuit_breaker",
    "compute_atr",
    "compute_keltner_lower_band",
    "compute_market_breadth",
    "compute_realized_volatility",
    "compute_sma",
    "compute_volatility_scale_factor",
    "evaluate_drawdown_gate",
    "evaluate_trend_filter",
    "filter_bars_point_in_time",
    # Momentum
    "calculate_12_1_momentum",
    "calculate_momentum_12_1",
    "compute_safe_haven_weights",
    "compute_universe_momentum",
    "evaluate_absolute_momentum",
    "evaluate_safe_haven_dual_momentum",
    "evaluate_safe_haven_qualification",
    "rank_momentum_candidates",
    "select_top_growth_leaders",
    "select_top_sectors",
    # Regime Detector
    "RegimeDetector",
    "SignalEngine",
]
