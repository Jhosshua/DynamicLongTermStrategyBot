"""
tests/unit/test_models.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for domain models, Pydantic v2 immutability, constraints, and Alpaca parsing.
"""

from datetime import datetime, timezone
import math
import pytest
from pydantic import ValidationError

from strategy_engine.core.models import (
    AssetClass,
    Bar,
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderType,
    Quote,
    SignalSnapshot,
    TargetAllocation,
    Trade,
)


# --- 1. Bar Tests ---

def test_bar_valid_creation(sample_bar: Bar):
    assert sample_bar.symbol == "SPY"
    assert sample_bar.open == 500.0
    assert sample_bar.high == 505.0
    assert sample_bar.low == 498.0
    assert sample_bar.close == 502.5
    assert sample_bar.volume == 50_000_000
    assert sample_bar.trade_count == 450_000
    assert sample_bar.vwap == 501.8


def test_bar_immutability(sample_bar: Bar):
    with pytest.raises((TypeError, ValidationError)):
        sample_bar.close = 510.0  # type: ignore

    with pytest.raises((TypeError, ValidationError)):
        sample_bar.symbol = "QQQ"  # type: ignore


@pytest.mark.parametrize("bad_kwargs", [
    {"high": 490.0, "open": 500.0, "close": 495.0, "low": 485.0},  # high < open
    {"high": 499.0, "open": 495.0, "close": 502.0, "low": 490.0},  # high < close
    {"low": 501.0, "open": 500.0, "close": 505.0, "high": 510.0},   # low > open
    {"low": 503.0, "open": 505.0, "close": 502.0, "high": 510.0},   # low > close
    {"low": -1.0, "open": 500.0, "high": 505.0, "close": 502.0},    # low <= 0
    {"volume": -100},                                                # volume < 0
    {"open": -50.0},                                                 # open <= 0
    {"close": 0.0},                                                  # close <= 0
])
def test_bar_validation_constraints(base_timestamp: datetime, bad_kwargs: dict):
    standard_kwargs = {
        "symbol": "SPY",
        "timestamp": base_timestamp,
        "open": 500.0,
        "high": 505.0,
        "low": 498.0,
        "close": 502.0,
        "volume": 1000,
    }
    standard_kwargs.update(bad_kwargs)
    with pytest.raises(ValidationError):
        Bar(**standard_kwargs)


def test_bar_flat_boundary(base_timestamp: datetime):
    bar = Bar(
        symbol="SPY",
        timestamp=base_timestamp,
        open=500.0,
        high=500.0,
        low=500.0,
        close=500.0,
        volume=0,
    )
    assert bar.high == bar.low == 500.0
    assert bar.volume == 0


def test_bar_from_alpaca_rest(base_timestamp: datetime):
    alpaca_data = {
        "t": "2026-01-05T14:30:00Z",
        "o": 500.0,
        "h": 505.0,
        "l": 498.0,
        "c": 502.5,
        "v": 50000000,
        "n": 450000,
        "vw": 501.8,
        "S": "SPY",
    }
    bar = Bar.from_alpaca(alpaca_data)
    assert bar.symbol == "SPY"
    assert bar.open == 500.0
    assert bar.trade_count == 450000
    assert bar.vwap == 501.8
    assert bar.timestamp == base_timestamp


def test_bar_serialization_roundtrip(sample_bar: Bar):
    dumped = sample_bar.model_dump()
    assert dumped["symbol"] == "SPY"
    assert dumped["volume"] == 50_000_000

    reconstructed = Bar.model_validate(dumped)
    assert reconstructed == sample_bar

    json_str = sample_bar.model_dump_json()
    from_json = Bar.model_validate_json(json_str)
    assert from_json == sample_bar


# --- 2. Quote Tests ---

def test_quote_immutability(sample_quote: Quote):
    with pytest.raises((TypeError, ValidationError)):
        sample_quote.bid_price = 503.0  # type: ignore


def test_quote_validation_crossed_market(base_timestamp: datetime):
    with pytest.raises(ValidationError):
        Quote(
            symbol="SPY",
            timestamp=base_timestamp,
            bid_price=502.50,
            bid_size=10,
            ask_price=502.40,
            ask_size=10,
        )


def test_quote_from_alpaca_ws():
    ws_msg = {
        "T": "q",
        "S": "AAPL",
        "bp": 180.25,
        "bs": 100,
        "bx": "V",
        "ap": 180.30,
        "as": 200,
        "ax": "V",
        "t": 1704465000000000000,  # nanoseconds
    }
    quote = Quote.from_alpaca(ws_msg)
    assert quote.symbol == "AAPL"
    assert quote.bid_price == 180.25
    assert quote.ask_price == 180.30
    assert quote.bid_size == 100
    assert quote.ask_size == 200
    assert quote.bid_exchange == "V"


# --- 3. Trade Tests ---

def test_trade_immutability(sample_trade: Trade):
    with pytest.raises((TypeError, ValidationError)):
        sample_trade.price = 505.0  # type: ignore


def test_trade_validation_positive(base_timestamp: datetime):
    with pytest.raises(ValidationError):
        Trade(symbol="SPY", timestamp=base_timestamp, price=0.0, size=100)
    with pytest.raises(ValidationError):
        Trade(symbol="SPY", timestamp=base_timestamp, price=500.0, size=-5)


def test_trade_from_alpaca_ws():
    ws_trade = {
        "T": "t",
        "S": "MSFT",
        "p": 400.50,
        "s": 50,
        "i": 98765,
        "x": "Q",
        "t": "2026-01-05T15:00:00Z",
    }
    trade = Trade.from_alpaca(ws_trade)
    assert trade.symbol == "MSFT"
    assert trade.price == 400.50
    assert trade.size == 50
    assert trade.id == 98765
    assert trade.trade_id == 98765
    assert trade.exchange == "Q"


# --- 4. SignalSnapshot Tests ---

def test_signal_snapshot_vol_scaler_bounds(base_timestamp: datetime):
    with pytest.raises(ValidationError):
        SignalSnapshot(
            timestamp=base_timestamp,
            spy_price=500.0,
            spy_sma50=490.0,
            spy_sma200=470.0,
            realized_vol_20d=0.10,
            vol_scale_factor=1.5,  # > 1.0
            drawdown_pct=-0.02,
            circuit_breaker_active=False,
            regime=MarketRegime.BULL_AGGRESSIVE,
            indicators={},
        )


def test_signal_snapshot_drawdown_bounds(base_timestamp: datetime):
    with pytest.raises(ValidationError):
        SignalSnapshot(
            timestamp=base_timestamp,
            spy_price=500.0,
            spy_sma50=490.0,
            spy_sma200=470.0,
            realized_vol_20d=0.10,
            vol_scale_factor=1.0,
            drawdown_pct=0.05,  # positive drawdown
            circuit_breaker_active=False,
            regime=MarketRegime.BULL_AGGRESSIVE,
            indicators={},
        )


# --- 5. TargetAllocation Tests ---

def test_target_allocation_valid(sample_target_allocation: TargetAllocation):
    assert math.isclose(sum(sample_target_allocation.weights.values()), 1.0, abs_tol=1e-5)
    assert sample_target_allocation.regime == MarketRegime.BULL_AGGRESSIVE


def test_target_allocation_weights_sum_validation(base_timestamp: datetime):
    # Under-allocated (0.90)
    with pytest.raises(ValidationError):
        TargetAllocation(
            timestamp=base_timestamp,
            regime=MarketRegime.BULL_NORMAL,
            weights={"QQQ": 0.50, "SPY": 0.40},
            cash_weight=0.0,
            rationale="Under-allocated",
        )

    # Over-allocated (1.10)
    with pytest.raises(ValidationError):
        TargetAllocation(
            timestamp=base_timestamp,
            regime=MarketRegime.BULL_NORMAL,
            weights={"QQQ": 0.60, "SPY": 0.50},
            cash_weight=0.0,
            rationale="Over-allocated",
        )


def test_target_allocation_negative_weight_rejected(base_timestamp: datetime):
    with pytest.raises(ValidationError):
        TargetAllocation(
            timestamp=base_timestamp,
            regime=MarketRegime.BULL_NORMAL,
            weights={"QQQ": 1.10, "SPY": -0.10},
            cash_weight=0.0,
            rationale="Short selling prohibited",
        )


# --- 6. OrderIntent Tests ---

def test_order_intent_valid_and_immutable(sample_order_intent: OrderIntent):
    assert sample_order_intent.action == "BUY"
    assert sample_order_intent.side == OrderSide.BUY
    assert sample_order_intent.reason == "Monthly momentum rebalance"
    assert sample_order_intent.rationale == "Monthly momentum rebalance"
    with pytest.raises((TypeError, ValidationError)):
        sample_order_intent.delta_shares = 50.0  # type: ignore


def test_order_intent_action_validation():
    with pytest.raises(ValidationError):
        OrderIntent(
            symbol="QQQ",
            action="SHORT",
            target_shares=10.0,
            delta_shares=10.0,
            delta_dollars=4000.0,
            estimated_price=400.0,
            reason="Unauthorized short",
        )


# --- 7. Enums Tests ---

def test_market_regime_enum_variants():
    expected = {
        "BULL_AGGRESSIVE",
        "BULL_NORMAL",
        "CORRECTION_FRAGILE",
        "BEAR_CRISIS",
        "STALE_DATA_HOLD",
    }
    actual = {r.value for r in MarketRegime}
    assert expected == actual


def test_asset_class_enum_variants():
    expected = {
        "EQUITY_INDEX",
        "EQUITY_SECTOR",
        "EQUITY_LEADER",
        "FIXED_INCOME_LONG",
        "CASH_EQUIVALENT",
        "COMMODITY",
    }
    actual = {a.value for a in AssetClass}
    assert expected == actual
