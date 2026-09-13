"""
strategy_engine.core.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Immutable domain models and enums for the AlpacaRelay Strategy Engine.
Built with Pydantic v2 with frozen=True for immutability and thread safety.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AssetClass(str, Enum):
    """Asset class classification for universe assets."""
    EQUITY_INDEX = "EQUITY_INDEX"          # Broad market index ETFs (SPY, QQQ)
    EQUITY_LEADER = "EQUITY_LEADER"        # Megacap tech & momentum leaders (AAPL, MSFT, NVDA, etc.)
    EQUITY_SECTOR = "EQUITY_SECTOR"        # SPDR Sector ETFs (XLK, XLE, XLF, etc.)
    FIXED_INCOME_LONG = "FIXED_INCOME_LONG"# Long Treasury bond ETFs (TLT)
    CASH_EQUIVALENT = "CASH_EQUIVALENT"    # Ultra-short T-bill ETFs / cash proxy (SHV, BIL, CASH)
    COMMODITY = "COMMODITY"                # Precious metals / commodities (GLD)


class MarketRegime(str, Enum):
    """Market regime classification for allocation state machine."""
    BULL_AGGRESSIVE = "BULL_AGGRESSIVE"    # Confirmed bull: low vol, strong trend, broad participation
    BULL_NORMAL = "BULL_NORMAL"            # Normal bull: price > 200 SMA, moderate vol
    CORRECTION_FRAGILE = "CORRECTION_FRAGILE" # Pullback / distribution: price < 50 SMA or elevated vol
    BEAR_CRISIS = "BEAR_CRISIS"            # Severe bear / crisis: price < 200 SMA, vol spike, breakdown
    STALE_DATA_HOLD = "STALE_DATA_HOLD"    # Upstream disconnected / stale data safety hold (no new risk)


class OrderSide(str, Enum):
    """Trading order direction."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(str, Enum):
    """Order execution type."""
    MARKET = "MARKET"
    MARKET_ON_CLOSE = "MOC"
    LIMIT = "LIMIT"


def _parse_timestamp(raw_t: Any) -> datetime:
    """Parse string, unix timestamp, or datetime into timezone-aware UTC datetime."""
    if isinstance(raw_t, datetime):
        if raw_t.tzinfo is None:
            return raw_t.replace(tzinfo=timezone.utc)
        return raw_t
    if isinstance(raw_t, str):
        cleaned = raw_t.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    if isinstance(raw_t, (int, float)):
        val = float(raw_t)
        if val > 1e16:
            val /= 1e9  # nanoseconds
        elif val > 1e11:
            val /= 1e3  # milliseconds
        return datetime.fromtimestamp(val, tz=timezone.utc)
    raise ValueError(f"Invalid timestamp value: {raw_t}")


class Bar(BaseModel):
    """Immutable OHLCV Bar representation.
    
    Compatible with standard Alpaca REST API and WebSocket stream frames.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timestamp: datetime
    open: float = Field(..., gt=0.0, description="Opening price")
    high: float = Field(..., gt=0.0, description="Highest price")
    low: float = Field(..., gt=0.0, description="Lowest price")
    close: float = Field(..., gt=0.0, description="Closing price")
    volume: int = Field(..., ge=0, description="Total volume traded")
    trade_count: Optional[int] = Field(default=None, ge=0, description="Number of trades ('n')")
    vwap: Optional[float] = Field(default=None, gt=0.0, description="Volume-weighted average price ('vw')")

    @model_validator(mode="after")
    def validate_ohlc_consistency(self) -> Bar:
        eps = 1e-5
        if self.high < self.low - eps:
            raise ValueError(f"Bar high ({self.high}) cannot be less than low ({self.low})")
        max_oc = max(self.open, self.close)
        min_oc = min(self.open, self.close)
        if self.high < max_oc - eps:
            raise ValueError(f"Bar high ({self.high}) must be >= max(open, close) ({max_oc})")
        if self.low > min_oc + eps:
            raise ValueError(f"Bar low ({self.low}) must be <= min(open, close) ({min_oc})")
        return self

    @classmethod
    def from_alpaca(cls, data: Dict[str, Any], symbol: Optional[str] = None) -> Bar:
        """Parse raw Alpaca JSON bar (REST or WS frame)."""
        sym = symbol or data.get("S") or data.get("symbol")
        if not sym:
            raise ValueError("Symbol must be provided in data ('S'/'symbol') or explicitly")

        raw_t = data.get("t") or data.get("timestamp")
        ts = _parse_timestamp(raw_t)

        return cls(
            symbol=sym,
            timestamp=ts,
            open=float(data.get("o", data.get("open"))),
            high=float(data.get("h", data.get("high"))),
            low=float(data.get("l", data.get("low"))),
            close=float(data.get("c", data.get("close"))),
            volume=int(data.get("v", data.get("volume", 0))),
            trade_count=int(data["n"]) if ("n" in data and data["n"] is not None) else data.get("trade_count"),
            vwap=float(data["vw"]) if ("vw" in data and data["vw"] is not None) else data.get("vwap"),
        )


class Quote(BaseModel):
    """Immutable Quote representation (Level 1 top-of-book)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timestamp: datetime
    bid_price: float = Field(..., gt=0.0, description="Best bid price")
    bid_size: int = Field(..., ge=0, description="Best bid size")
    ask_price: float = Field(..., gt=0.0, description="Best ask price")
    ask_size: int = Field(..., ge=0, description="Best ask size")
    bid_exchange: Optional[str] = None
    ask_exchange: Optional[str] = None
    conditions: Optional[List[str]] = None
    tape: Optional[str] = None

    @model_validator(mode="after")
    def validate_quote_bounds(self) -> Quote:
        if self.ask_price < self.bid_price - 1e-5:
            raise ValueError(f"Ask price ({self.ask_price}) cannot be less than bid price ({self.bid_price})")
        return self

    @classmethod
    def from_alpaca(cls, data: Dict[str, Any], symbol: Optional[str] = None) -> Quote:
        """Parse raw Alpaca JSON quote ('q' message)."""
        sym = symbol or data.get("S") or data.get("symbol")
        if not sym:
            raise ValueError("Symbol must be provided in data ('S'/'symbol') or explicitly")

        raw_t = data.get("t") or data.get("timestamp")
        ts = _parse_timestamp(raw_t)

        return cls(
            symbol=sym,
            timestamp=ts,
            bid_price=float(data.get("bp", data.get("bid_price"))),
            bid_size=int(data.get("bs", data.get("bid_size", 0))),
            bid_exchange=data.get("bx", data.get("bid_exchange")),
            ask_price=float(data.get("ap", data.get("ask_price"))),
            ask_size=int(data.get("as", data.get("ask_size", 0))),
            ask_exchange=data.get("ax", data.get("ask_exchange")),
            conditions=data.get("c", data.get("conditions")),
            tape=data.get("z", data.get("tape")),
        )


class Trade(BaseModel):
    """Immutable Trade representation ('t' message)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timestamp: datetime
    price: float = Field(..., gt=0.0, description="Trade execution price")
    size: int = Field(..., ge=0, description="Trade volume / size")
    id: Optional[Union[str, int]] = Field(default=None, description="Trade sequence identifier")
    trade_id: Optional[Union[str, int]] = Field(default=None, description="Alpaca trade ID ('i')")
    exchange: Optional[str] = None
    conditions: Optional[List[str]] = None
    tape: Optional[str] = None

    @model_validator(mode="after")
    def sync_ids(self) -> Trade:
        if self.id is not None and self.trade_id is None:
            object.__setattr__(self, "trade_id", self.id)
        elif self.trade_id is not None and self.id is None:
            object.__setattr__(self, "id", self.trade_id)
        return self

    @classmethod
    def from_alpaca(cls, data: Dict[str, Any], symbol: Optional[str] = None) -> Trade:
        """Parse raw Alpaca JSON trade ('t' message)."""
        sym = symbol or data.get("S") or data.get("symbol")
        if not sym:
            raise ValueError("Symbol must be provided in data ('S'/'symbol') or explicitly")

        raw_t = data.get("t") or data.get("timestamp")
        ts = _parse_timestamp(raw_t)

        tid = data.get("i", data.get("trade_id", data.get("id")))

        return cls(
            symbol=sym,
            timestamp=ts,
            price=float(data.get("p", data.get("price"))),
            size=int(data.get("s", data.get("size", 0))),
            id=tid,
            trade_id=tid,
            exchange=data.get("x", data.get("exchange")),
            conditions=data.get("c", data.get("conditions")),
            tape=data.get("z", data.get("tape")),
        )


class SignalSnapshot(BaseModel):
    """Point-in-time calculation snapshot of quantitative signals."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    spy_price: float = Field(..., gt=0.0)
    spy_sma50: float = Field(..., gt=0.0)
    spy_sma200: float = Field(..., gt=0.0)
    realized_vol_20d: float = Field(..., ge=0.0)
    vol_scale_factor: float = Field(..., ge=0.0, le=1.0)
    drawdown_pct: float = Field(..., le=0.0, description="Trailing peak-to-trough drawdown (non-positive)")
    circuit_breaker_active: bool = False
    regime: MarketRegime
    indicators: Dict[str, float] = Field(default_factory=dict)


class TargetAllocation(BaseModel):
    """Target portfolio weights for rebalancing."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    regime: MarketRegime
    weights: Dict[str, float] = Field(..., description="Target weights mapping symbol -> float, summing to 1.0")
    cash_weight: float = Field(..., ge=0.0, le=1.0, description="Explicit cash / cash-equivalent weight")
    rationale: str = Field(default="", description="Human-readable rationale for this allocation")

    @model_validator(mode="after")
    def validate_weights_sum_and_bounds(self) -> TargetAllocation:
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"Portfolio weights must sum to 1.0 within 1e-5 (got {total:.7f})")
        for sym, w in self.weights.items():
            if w < -1e-6 or w > 1.0 + 1e-6:
                raise ValueError(f"Weight for {sym} ({w}) out of valid bounds [0.0, 1.0]")

        # Check cash consistency if explicit cash/SHV/BIL present
        cash_syms_sum = sum(w for sym, w in self.weights.items() if sym in ("SHV", "BIL", "CASH"))
        if cash_syms_sum > 0 and not math.isclose(self.cash_weight, cash_syms_sum, rel_tol=1e-4, abs_tol=1e-4):
            # If cash symbols are present, cash_weight should be consistent
            if not math.isclose(self.cash_weight, cash_syms_sum, abs_tol=1e-4):
                raise ValueError(f"cash_weight ({self.cash_weight:.5f}) must match sum of cash/SHV/BIL ({cash_syms_sum:.5f})")
        return self


class OrderIntent(BaseModel):
    """Specific rebalancing order instruction."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    action: Optional[str] = Field(default=None, description="'BUY', 'SELL', or 'HOLD'")
    side: Optional[OrderSide] = Field(default=None, description="OrderSide enum")
    target_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    current_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    delta_weight: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    target_shares: Optional[float] = Field(default=None, ge=0.0)
    delta_shares: Optional[float] = Field(default=None)
    delta_dollars: Optional[float] = Field(default=None)
    estimated_price: Optional[float] = Field(default=None, gt=0.0)
    notional: Optional[float] = Field(default=None)
    order_type: Optional[Union[OrderType, str]] = Field(default=OrderType.MARKET_ON_CLOSE)
    rationale: str = ""
    reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_action_and_sync(self) -> OrderIntent:
        act = self.action
        sd = self.side

        if act is not None:
            act_upper = act.upper()
            if act_upper not in ("BUY", "SELL", "HOLD"):
                raise ValueError(f"Invalid order action '{act}'; must be BUY, SELL, or HOLD")
            if sd is None:
                object.__setattr__(self, "side", OrderSide(act_upper))
            else:
                sd_val = sd.value if isinstance(sd, OrderSide) else str(sd).upper()
                if act_upper != sd_val:
                    raise ValueError(f"Contradictory action '{act}' and side '{sd}'")
            object.__setattr__(self, "action", act_upper)
        elif sd is not None:
            object.__setattr__(self, "action", sd.value)

        # Synchronize reason and rationale
        if self.rationale and not self.reason:
            object.__setattr__(self, "reason", self.rationale)
        elif self.reason and not self.rationale:
            object.__setattr__(self, "rationale", self.reason)

        return self
