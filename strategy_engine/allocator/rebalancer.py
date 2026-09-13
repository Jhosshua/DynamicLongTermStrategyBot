"""
strategy_engine.allocator.rebalancer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drift band filtering (+/- 2.5%), micro-order suppression (< 0.5%),
and sequenced OrderIntent generation (SELLs before BUYs).
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Dict, List, Optional, Union

from strategy_engine.core.models import (
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderType,
    TargetAllocation,
)

DRIFT_TOLERANCE: float = 0.025         # +/- 2.5% drift threshold
MIN_ORDER_THRESHOLD: float = 0.005     # 0.5% micro-order filter floor


class PortfolioRebalancer:
    """Computes rebalancing deltas and emits validated OrderIntents with drift band filtering."""

    def __init__(
        self,
        drift_band: float = DRIFT_TOLERANCE,
        min_order_threshold: float = MIN_ORDER_THRESHOLD,
    ):
        self.drift_band = drift_band
        self.min_order_threshold = min_order_threshold

    def is_drift_exceeded(self, target_weight: float, current_weight: float) -> bool:
        """Check if deviation strictly exceeds drift band tolerance."""
        return abs(target_weight - current_weight) > self.drift_band

    def compute_rebalance_orders(
        self,
        target_allocation: TargetAllocation,
        current_weights: Dict[str, float],
        portfolio_equity: Optional[float] = None,
        current_prices: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> List[OrderIntent]:
        """Generate rebalancing OrderIntent list, filtering within-band drifts.
        
        Sequences SELL orders first (cash liberation) and BUY orders second.
        """
        if target_allocation.regime == MarketRegime.STALE_DATA_HOLD:
            return []

        now_ts = timestamp or datetime.now(timezone.utc)
        all_symbols = sorted(list(set(list(target_allocation.weights.keys()) + list(current_weights.keys()))))

        sell_orders: List[OrderIntent] = []
        buy_orders: List[OrderIntent] = []

        for sym in all_symbols:
            w_tgt = float(target_allocation.weights.get(sym, 0.0))
            w_cur = float(current_weights.get(sym, 0.0))
            delta_w = w_tgt - w_cur

            # Clamp weights to prevent OrderIntent Pydantic ValidationError
            w_tgt = min(1.0, max(0.0, w_tgt))
            w_cur = min(1.0, max(0.0, w_cur))
            delta_w = min(1.0, max(-1.0, delta_w))

            # 1. Drift Band Filtering (+/- 2.5%)
            if not self.is_drift_exceeded(w_tgt, w_cur):
                continue

            # 2. Micro-order Floor Filtering (0.5%)
            if abs(delta_w) < self.min_order_threshold:
                continue

            action = "BUY" if delta_w > 0.0 else "SELL"
            side = OrderSide.BUY if delta_w > 0.0 else OrderSide.SELL

            # Calculate dollar and share amounts if portfolio equity provided
            delta_dollars: Optional[float] = None
            notional: Optional[float] = None
            target_shares: Optional[float] = None
            delta_shares: Optional[float] = None
            est_price: Optional[float] = None

            if portfolio_equity is not None and portfolio_equity > 0.0:
                delta_dollars = delta_w * portfolio_equity
                notional = abs(delta_dollars)
                if current_prices and sym in current_prices and current_prices[sym] > 0.0:
                    est_price = float(current_prices[sym])
                    target_shares = float((w_tgt * portfolio_equity) / est_price)
                    delta_shares = float(delta_dollars / est_price)

            rationale = (
                f"Rebalance {sym}: current weight {w_cur:.4f} deviated from target {w_tgt:.4f} "
                f"by {delta_w:+.4f} (exceeds +/-{self.drift_band:.1%} drift band)"
            )

            order = OrderIntent(
                symbol=sym,
                action=action,
                side=side,
                target_weight=w_tgt,
                current_weight=w_cur,
                delta_weight=delta_w,
                target_shares=target_shares,
                delta_shares=delta_shares,
                delta_dollars=delta_dollars,
                estimated_price=est_price,
                notional=notional,
                order_type=OrderType.MARKET_ON_CLOSE,
                rationale=rationale,
                reason=rationale,
                timestamp=now_ts,
            )

            if action == "SELL":
                sell_orders.append(order)
            else:
                buy_orders.append(order)

        # Sequence: SELLs first (cash liberation), then BUYs
        return sell_orders + buy_orders


def generate_rebalance_orders(
    target: TargetAllocation,
    current_weights: Dict[str, float],
    portfolio_value: Optional[float] = None,
    current_prices: Optional[Dict[str, float]] = None,
    drift_band: float = DRIFT_TOLERANCE,
    min_order_threshold: float = MIN_ORDER_THRESHOLD,
    timestamp: Optional[datetime] = None,
) -> List[OrderIntent]:
    """Convenience function generating sequenced OrderIntents using PortfolioRebalancer."""
    rebalancer = PortfolioRebalancer(drift_band=drift_band, min_order_threshold=min_order_threshold)
    return rebalancer.compute_rebalance_orders(
        target_allocation=target,
        current_weights=current_weights,
        portfolio_equity=portfolio_value,
        current_prices=current_prices,
        timestamp=timestamp,
    )
