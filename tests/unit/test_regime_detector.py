"""
tests.unit.test_regime_detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit test suite for multi-timeframe market regime classification and SignalEngine:
- Hierarchical decision tree classification
- Stale data hold safety override
- Crisis overrides: circuit breaker, drawdown levels, vol spikes, structural breakdowns
- Fragile / Correction mode: pullbacks, elevated vol, deteriorating breadth
- Bull Aggressive vs Bull Normal conditions
- Point-in-time SignalSnapshot generation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import pytest

from strategy_engine.core.models import Bar, MarketRegime, SignalSnapshot
from strategy_engine.signals.regime_detector import (
    RegimeDetector,
    SignalEngine,
)


def _create_bar(symbol: str, dt: datetime, close: float) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=dt,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=100_000,
    )


def test_classify_stale_data_hold():
    """Verify stale data or upstream disconnection triggers STALE_DATA_HOLD override."""
    detector = RegimeDetector()
    trend = {"price": 500.0, "sma50": 490.0, "sma200": 470.0, "price_above_sma50": True, "price_above_sma200": True, "is_golden_cross": True}

    regime_stale, _ = detector.classify(
        spy_trend=trend, qqq_trend=trend, realized_vol_20d=0.10, circuit_breaker_active=False,
        drawdown_pct=0.0, breadth_50=0.80, is_stale=True, upstream_connected=True,
    )
    assert regime_stale == MarketRegime.STALE_DATA_HOLD

    regime_disc, _ = detector.classify(
        spy_trend=trend, qqq_trend=trend, realized_vol_20d=0.10, circuit_breaker_active=False,
        drawdown_pct=0.0, breadth_50=0.80, is_stale=False, upstream_connected=False,
    )
    assert regime_disc == MarketRegime.STALE_DATA_HOLD


def test_classify_circuit_breaker_override():
    """Verify dynamic ATR lower band breach overrides immediately into BEAR_CRISIS."""
    detector = RegimeDetector()
    trend = {"price": 500.0, "sma50": 490.0, "sma200": 470.0, "price_above_sma50": True, "price_above_sma200": True, "is_golden_cross": True}

    regime, reason = detector.classify(
        spy_trend=trend, qqq_trend=trend, realized_vol_20d=0.10, circuit_breaker_active=True,
        drawdown_pct=-0.02, breadth_50=0.80,
    )
    assert regime == MarketRegime.BEAR_CRISIS
    assert "Circuit breaker" in reason


def test_classify_drawdown_hard_stop():
    """Verify deep drawdown breaches trigger BEAR_CRISIS."""
    detector = RegimeDetector()
    trend = {"price": 500.0, "sma50": 490.0, "sma200": 470.0, "price_above_sma50": True, "price_above_sma200": True, "is_golden_cross": True}

    # Level 2 breach (-11%)
    regime_l2, _ = detector.classify(
        spy_trend=trend, qqq_trend=trend, realized_vol_20d=0.10, circuit_breaker_active=False,
        drawdown_pct=-0.11, breadth_50=0.80,
    )
    assert regime_l2 == MarketRegime.BEAR_CRISIS

    # Level 3 breach (-16%)
    regime_l3, _ = detector.classify(
        spy_trend=trend, qqq_trend=trend, realized_vol_20d=0.10, circuit_breaker_active=False,
        drawdown_pct=-0.16, breadth_50=0.80,
    )
    assert regime_l3 == MarketRegime.BEAR_CRISIS


def test_classify_volatility_spike_crisis():
    """Verify realized vol > 30% triggers BEAR_CRISIS."""
    detector = RegimeDetector()
    trend = {"price": 500.0, "sma50": 490.0, "sma200": 470.0, "price_above_sma50": True, "price_above_sma200": True, "is_golden_cross": True}

    regime, _ = detector.classify(
        spy_trend=trend, qqq_trend=trend, realized_vol_20d=0.35, circuit_breaker_active=False,
        drawdown_pct=-0.02, breadth_50=0.80,
    )
    assert regime == MarketRegime.BEAR_CRISIS


def test_classify_structural_bear_breakdown():
    """Verify SPY below 200 SMA with death cross or high vol triggers BEAR_CRISIS."""
    detector = RegimeDetector()
    trend_bear = {"price": 460.0, "sma50": 465.0, "sma200": 470.0, "price_above_sma50": False, "price_above_sma200": False, "is_golden_cross": False}

    regime, _ = detector.classify(
        spy_trend=trend_bear, qqq_trend=trend_bear, realized_vol_20d=0.20, circuit_breaker_active=False,
        drawdown_pct=-0.03, breadth_50=0.30,
    )
    assert regime == MarketRegime.BEAR_CRISIS


def test_classify_correction_fragile():
    """Verify pullback below 50 SMA, elevated vol (22-30%), or weak breadth triggers CORRECTION_FRAGILE."""
    detector = RegimeDetector()
    trend_bull = {"price": 500.0, "sma50": 490.0, "sma200": 470.0, "price_above_sma50": True, "price_above_sma200": True, "is_golden_cross": True}
    trend_pullback = {"price": 485.0, "sma50": 490.0, "sma200": 470.0, "price_above_sma50": False, "price_above_sma200": True, "is_golden_cross": True}

    # 1. SPY Pullback below 50 SMA
    reg_pullback, _ = detector.classify(
        spy_trend=trend_pullback, qqq_trend=trend_bull, realized_vol_20d=0.12, circuit_breaker_active=False,
        drawdown_pct=-0.02, breadth_50=0.70,
    )
    assert reg_pullback == MarketRegime.CORRECTION_FRAGILE

    # 2. Elevated Volatility (25%)
    reg_high_vol, _ = detector.classify(
        spy_trend=trend_bull, qqq_trend=trend_bull, realized_vol_20d=0.25, circuit_breaker_active=False,
        drawdown_pct=-0.02, breadth_50=0.70,
    )
    assert reg_high_vol == MarketRegime.CORRECTION_FRAGILE

    # 3. Weak Breadth (30% < 40%)
    reg_breadth, _ = detector.classify(
        spy_trend=trend_bull, qqq_trend=trend_bull, realized_vol_20d=0.12, circuit_breaker_active=False,
        drawdown_pct=-0.02, breadth_50=0.30,
    )
    assert reg_breadth == MarketRegime.CORRECTION_FRAGILE


def test_classify_bull_aggressive():
    """Verify low-vol, broad-breadth uptrend triggers BULL_AGGRESSIVE."""
    detector = RegimeDetector()
    trend = {"price": 500.0, "sma50": 490.0, "sma200": 470.0, "price_above_sma50": True, "price_above_sma200": True, "is_golden_cross": True}

    regime, _ = detector.classify(
        spy_trend=trend, qqq_trend=trend, realized_vol_20d=0.10, circuit_breaker_active=False,
        drawdown_pct=-0.01, breadth_50=0.75,
    )
    assert regime == MarketRegime.BULL_AGGRESSIVE


def test_classify_bull_normal():
    """Verify normal bull trend (moderate vol 18% or moderate breadth 50%) defaults to BULL_NORMAL."""
    detector = RegimeDetector()
    trend = {"price": 500.0, "sma50": 490.0, "sma200": 470.0, "price_above_sma50": True, "price_above_sma200": True, "is_golden_cross": True}

    # Moderate vol: 18% (> 14% aggressive threshold, but <= 22% normal threshold)
    regime, _ = detector.classify(
        spy_trend=trend, qqq_trend=trend, realized_vol_20d=0.18, circuit_breaker_active=False,
        drawdown_pct=-0.02, breadth_50=0.75,
    )
    assert regime == MarketRegime.BULL_NORMAL


def test_signal_engine_daily_signals_creation():
    """Verify SignalEngine computes point-in-time SignalSnapshot with zero lookahead bias."""
    engine = SignalEngine()
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    market_data = {
        "SPY": [_create_bar("SPY", t0 + timedelta(days=i), 500.0 + i) for i in range(250)],
        "QQQ": [_create_bar("QQQ", t0 + timedelta(days=i), 450.0 + i) for i in range(250)],
        "XLK": [_create_bar("XLK", t0 + timedelta(days=i), 200.0 + i) for i in range(250)],
    }

    eval_time = t0 + timedelta(days=210)
    snapshot = engine.compute_daily_signals(market_data, current_time=eval_time)

    assert isinstance(snapshot, SignalSnapshot)
    assert snapshot.timestamp == eval_time
    assert snapshot.spy_price == 500.0 + 210
    assert snapshot.regime in (MarketRegime.BULL_AGGRESSIVE, MarketRegime.BULL_NORMAL)
    assert "vol_scale_factor" in snapshot.indicators
    assert "keltner_lower_band" in snapshot.indicators
    assert snapshot.circuit_breaker_active is False

    # Immutability check
    with pytest.raises(Exception):
        snapshot.spy_price = 999.0


def test_signal_engine_missing_spy_raises_keyerror():
    """SignalEngine requires SPY in market data; missing SPY must raise KeyError."""
    engine = SignalEngine()
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(KeyError):
        engine.compute_daily_signals({"QQQ": []}, current_time=t0)
