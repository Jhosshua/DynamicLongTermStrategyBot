"""
strategy_engine.signals.regime_detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-timeframe market regime detector and daily signal calculation engine.
Adheres to SignalEngineProtocol and outputs frozen SignalSnapshot instances.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from strategy_engine.core.models import Bar, MarketRegime, SignalSnapshot, TargetAllocation
from strategy_engine.core.universe import CORE_COMPOUNDERS, SECTOR_ETFS
from strategy_engine.signals.indicators import (
    DrawdownDefenseTracker,
    check_circuit_breaker,
    compute_keltner_lower_band,
    compute_market_breadth,
    compute_realized_volatility,
    compute_sma,
    compute_volatility_scale_factor,
    evaluate_drawdown_gate,
    evaluate_trend_filter,
    filter_bars_point_in_time,
)
from strategy_engine.signals.momentum import compute_universe_momentum


class RegimeDetector:
    """Configurable regime classification logic based on quantitative thresholds."""

    def __init__(
        self,
        vol_aggressive_threshold: float = 0.14,
        vol_normal_threshold: float = 0.22,
        vol_crisis_threshold: float = 0.30,
        breadth_bull_threshold: float = 0.60,
        breadth_fragile_threshold: float = 0.40,
        dd_l1_threshold: float = -0.05,
        dd_l2_threshold: float = -0.10,
        dd_l3_threshold: float = -0.15,
        atr_multiplier: float = 2.0,
    ):
        self.vol_aggressive = vol_aggressive_threshold
        self.vol_normal = vol_normal_threshold
        self.vol_crisis = vol_crisis_threshold
        self.breadth_bull = breadth_bull_threshold
        self.breadth_fragile = breadth_fragile_threshold
        self.dd_l1 = dd_l1_threshold
        self.dd_l2 = dd_l2_threshold
        self.dd_l3 = dd_l3_threshold
        self.atr_multiplier = atr_multiplier

    def classify(
        self,
        spy_trend: Dict[str, Any],
        qqq_trend: Dict[str, Any],
        realized_vol_20d: float,
        circuit_breaker_active: bool,
        drawdown_pct: float,
        breadth_50: float,
        in_recovery_lockout: bool = False,
        is_stale: bool = False,
        upstream_connected: bool = True,
    ) -> Tuple[MarketRegime, str]:
        """Classify market state into discrete MarketRegime enum with explanation."""
        # 1. Stale Data Hold Safety Override
        if is_stale or not upstream_connected:
            return MarketRegime.STALE_DATA_HOLD, "Upstream disconnected or market data stale"

        # 2. Crisis / Emergency Overrides
        if circuit_breaker_active:
            return MarketRegime.BEAR_CRISIS, "Circuit breaker triggered: SPY closed below ATR lower band"
        if drawdown_pct <= self.dd_l3:
            return MarketRegime.BEAR_CRISIS, f"Hard drawdown circuit breaker breached: DD {drawdown_pct:.1%}"
        if drawdown_pct <= self.dd_l2:
            return MarketRegime.BEAR_CRISIS, f"Drawdown defensive gate L2 breached: DD {drawdown_pct:.1%}"
        if realized_vol_20d > self.vol_crisis:
            return MarketRegime.BEAR_CRISIS, f"Volatility spike into crisis: 20d vol {realized_vol_20d:.1%}"
        if not spy_trend.get("price_above_sma200", False) and (
            not spy_trend.get("is_golden_cross", False) or realized_vol_20d > self.vol_normal
        ):
            return MarketRegime.BEAR_CRISIS, "Structural bear breakdown: SPY below 200 SMA with death cross or high vol"

        # 3. Fragile / Correction Mode
        if not spy_trend.get("price_above_sma50", False):
            return MarketRegime.CORRECTION_FRAGILE, "SPY pullback below 50 SMA in primary trend"
        if not qqq_trend.get("price_above_sma50", False):
            return MarketRegime.CORRECTION_FRAGILE, "QQQ pullback below 50 SMA in primary trend"
        if realized_vol_20d > self.vol_normal:
            return MarketRegime.CORRECTION_FRAGILE, f"Elevated volatility: 20d vol {realized_vol_20d:.1%}"
        if drawdown_pct <= self.dd_l1 or in_recovery_lockout:
            return MarketRegime.CORRECTION_FRAGILE, f"Drawdown caution/lockout active: DD {drawdown_pct:.1%}"
        if breadth_50 < self.breadth_fragile:
            return MarketRegime.CORRECTION_FRAGILE, f"Weak market participation: Breadth {breadth_50:.1%}"

        # 4. Aggressive Bull Mode
        is_bull_aggressive = (
            spy_trend.get("price_above_sma50", False)
            and spy_trend.get("is_golden_cross", False)
            and qqq_trend.get("price_above_sma50", False)
            and qqq_trend.get("is_golden_cross", False)
            and realized_vol_20d <= self.vol_aggressive
            and breadth_50 >= self.breadth_bull
            and drawdown_pct > self.dd_l1
        )
        if is_bull_aggressive:
            return MarketRegime.BULL_AGGRESSIVE, "Confirmed low-volatility broad-participation bull trend"

        # 5. Default: Normal Bull Mode
        return MarketRegime.BULL_NORMAL, "Normal primary bull uptrend"

    def classify_regime(
        self,
        spy_price: float,
        sma50: float,
        sma200: float,
        realized_vol: float,
        breadth_50: float = 0.50,
        drawdown_pct: float = 0.0,
        circuit_breaker_active: bool = False,
        upstream_connected: bool = True,
        is_stale: bool = False,
        qqq_price: Optional[float] = None,
        qqq_sma50: Optional[float] = None,
        qqq_sma200: Optional[float] = None,
        in_recovery_lockout: bool = False,
    ) -> MarketRegime:
        """Convenience method accepting raw indicator scalars."""
        spy_trend = {
            "price": spy_price,
            "sma50": sma50,
            "sma200": sma200,
            "price_above_sma50": spy_price > sma50,
            "price_above_sma200": spy_price > sma200,
            "is_golden_cross": sma50 > sma200,
            "trend_score": 1.0 if (spy_price > sma50 and sma50 > sma200) else 0.0,
        }
        q_p = qqq_price if qqq_price is not None else spy_price
        q_50 = qqq_sma50 if qqq_sma50 is not None else sma50
        q_200 = qqq_sma200 if qqq_sma200 is not None else sma200
        qqq_trend = {
            "price": q_p,
            "sma50": q_50,
            "sma200": q_200,
            "price_above_sma50": q_p > q_50,
            "price_above_sma200": q_p > q_200,
            "is_golden_cross": q_50 > q_200,
            "trend_score": 1.0 if (q_p > q_50 and q_50 > q_200) else 0.0,
        }

        regime, _ = self.classify(
            spy_trend=spy_trend,
            qqq_trend=qqq_trend,
            realized_vol_20d=realized_vol,
            circuit_breaker_active=circuit_breaker_active,
            drawdown_pct=drawdown_pct,
            breadth_50=breadth_50,
            in_recovery_lockout=in_recovery_lockout,
            is_stale=is_stale,
            upstream_connected=upstream_connected,
        )
        return regime


class SignalEngine:
    """Multi-timeframe signal engine implementing SignalEngineProtocol."""

    def __init__(self, detector: Optional[RegimeDetector] = None):
        self.detector = detector or RegimeDetector()
        self.drawdown_tracker = DrawdownDefenseTracker()

    def compute_daily_signals(
        self,
        market_data: Dict[str, List[Bar]],
        current_time: datetime,
        portfolio_equity_curve: Optional[List[float]] = None,
        is_stale: bool = False,
        upstream_connected: bool = True,
    ) -> SignalSnapshot:
        """Compute point-in-time quantitative risk signals and market regime.
        
        Zero lookahead bias guaranteed: bars strictly filtered to timestamp <= current_time.
        """
        if "SPY" not in market_data:
            raise KeyError("Market data must contain 'SPY' bars for daily signal evaluation")

        # Point-in-time bar extraction
        spy_bars = filter_bars_point_in_time(market_data["SPY"], current_time)
        if not spy_bars:
            raise ValueError(f"No SPY bars found on or before evaluation time {current_time.isoformat()}")

        qqq_bars = filter_bars_point_in_time(market_data.get("QQQ", spy_bars), current_time)
        if not qqq_bars:
            qqq_bars = spy_bars

        # 1. 20-day Rolling Realized Volatility Targeting
        vol_20d = compute_realized_volatility(spy_bars, window=20)
        vol_scale = compute_volatility_scale_factor(vol_20d, target_vol=0.12, min_vol=0.05, max_scale=1.0)

        # 2. Dual-SMA Trend Filters
        spy_trend = evaluate_trend_filter(spy_bars, "SPY", fast_window=50, slow_window=200)
        qqq_trend = evaluate_trend_filter(qqq_bars, "QQQ", fast_window=50, slow_window=200)

        # 3. Dynamic ATR Keltner Stop
        is_cb_active, cb_metrics = check_circuit_breaker(
            spy_bars, sma_window=50, atr_window=14, atr_multiplier=self.detector.atr_multiplier
        )

        # 4. Trailing Drawdown Defense & 3-Day Hysteresis
        if portfolio_equity_curve and len(portfolio_equity_curve) > 0:
            # Prime uninitialized or fresh tracker across historical equity curve
            if len(portfolio_equity_curve) > 1 and len(getattr(self.drawdown_tracker, "equity_history", [])) <= 1:
                if hasattr(self.drawdown_tracker, "prime"):
                    self.drawdown_tracker.prime(portfolio_equity_curve[:-1])
                else:
                    self.drawdown_tracker.equity_history = [float(x) for x in portfolio_equity_curve[:-1]]
                    self.drawdown_tracker.peak_equity = float(max(portfolio_equity_curve[:-1]))
            current_eq = portfolio_equity_curve[-1]
        else:
            current_eq = spy_bars[-1].close

        dd_gate = self.drawdown_tracker.update(
            current_equity=current_eq,
            benchmark_price=spy_trend["price"],
            benchmark_sma50=spy_trend["sma50"],
        )
        current_dd = self.drawdown_tracker.current_drawdown
        in_lockout = self.drawdown_tracker.in_recovery_lockout

        # 5. Market Breadth
        breadth_symbols = [s for s in SECTOR_ETFS if s in market_data]
        if not breadth_symbols:
            breadth_symbols = [s for s in CORE_COMPOUNDERS if s in market_data]
        breadth_info = compute_market_breadth(market_data, symbols=breadth_symbols, current_time=current_time)

        # 6. Regime Classification
        regime, rationale = self.detector.classify(
            spy_trend=spy_trend,
            qqq_trend=qqq_trend,
            realized_vol_20d=vol_20d,
            circuit_breaker_active=is_cb_active,
            drawdown_pct=current_dd,
            breadth_50=breadth_info["breadth_50"],
            in_recovery_lockout=in_lockout,
            is_stale=is_stale,
            upstream_connected=upstream_connected,
        )

        indicators: Dict[str, float] = {
            "spy_price": float(spy_trend["price"]),
            "spy_sma50": float(spy_trend["sma50"]),
            "spy_sma200": float(spy_trend["sma200"]),
            "qqq_price": float(qqq_trend["price"]),
            "qqq_sma50": float(qqq_trend["sma50"]),
            "qqq_sma200": float(qqq_trend["sma200"]),
            "realized_vol_20d": float(vol_20d),
            "vol_scale_factor": float(vol_scale),
            "atr_14": float(cb_metrics["atr14"]),
            "keltner_lower_band": float(cb_metrics["lower_band"]),
            "circuit_breaker_active": 1.0 if is_cb_active else 0.0,
            "drawdown_pct": float(current_dd),
            "drawdown_gate": float(dd_gate),
            "consecutive_recovery_days": float(self.drawdown_tracker.consecutive_recovery_days),
            "in_recovery_lockout": 1.0 if in_lockout else 0.0,
            "breadth_50": float(breadth_info["breadth_50"]),
            "breadth_200": float(breadth_info["breadth_200"]),
            "breadth_regime_factor": float(breadth_info["breadth_regime_factor"]),
            "spy_trend_score": float(spy_trend["trend_score"]),
            "qqq_trend_score": float(qqq_trend["trend_score"]),
            "is_golden_cross_spy": 1.0 if spy_trend["is_golden_cross"] else 0.0,
            "is_golden_cross_qqq": 1.0 if qqq_trend["is_golden_cross"] else 0.0,
        }

        return SignalSnapshot(
            timestamp=current_time,
            spy_price=float(spy_trend["price"]),
            spy_sma50=float(spy_trend["sma50"]),
            spy_sma200=float(spy_trend["sma200"]),
            realized_vol_20d=float(vol_20d),
            vol_scale_factor=float(vol_scale),
            drawdown_pct=float(current_dd),
            circuit_breaker_active=is_cb_active,
            regime=regime,
            indicators=indicators,
        )

    def compute_monthly_momentum(
        self,
        market_data: Dict[str, List[Bar]],
        current_time: datetime,
    ) -> Dict[str, float]:
        """Compute point-in-time 12-1 structural momentum scores across market data."""
        pit_data: Dict[str, List[Bar]] = {}
        for sym, bars in market_data.items():
            pit_data[sym] = filter_bars_point_in_time(bars, current_time)
        return compute_universe_momentum(pit_data)

    def compute_target_weights(
        self,
        signals: SignalSnapshot,
        momentum: Optional[Dict[str, float]] = None,
        market_data: Optional[Dict[str, List[Bar]]] = None,
    ) -> TargetAllocation:
        """Compute TargetAllocation matching SignalEngineProtocol."""
        from strategy_engine.allocator.rules import compute_deterministic_allocation
        data = market_data or {}
        pit_market_data = {
            s: filter_bars_point_in_time(b, signals.timestamp)
            for s, b in data.items()
        }
        return compute_deterministic_allocation(
            signals=signals, market_data=pit_market_data, current_time=signals.timestamp
        )
