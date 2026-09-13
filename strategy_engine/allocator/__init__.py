"""
strategy_engine.allocator
~~~~~~~~~~~~~~~~~~~~~~~~~

Portfolio allocation optimization, deterministic weight normalization,
drift band filtering, and sequenced rebalance order generation.
"""

from strategy_engine.allocator.rules import (
    compute_deterministic_allocation,
    get_regime_base_weights,
    normalize_target_weights,
)
from strategy_engine.allocator.rebalancer import (
    DRIFT_TOLERANCE,
    MIN_ORDER_THRESHOLD,
    PortfolioRebalancer,
    generate_rebalance_orders,
)

__all__ = [
    "DRIFT_TOLERANCE",
    "MIN_ORDER_THRESHOLD",
    "PortfolioRebalancer",
    "compute_deterministic_allocation",
    "generate_rebalance_orders",
    "get_regime_base_weights",
    "normalize_target_weights",
]
