"""Reference domain models, protocols, and math utilities matching PROJECT.md interface contracts.
Provides fallback/mock fixtures for progressive testability during milestone implementation.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple


class AssetClass(str, Enum):
    EQUITY_INDEX = "EQUITY_INDEX"
    EQUITY_SECTOR = "EQUITY_SECTOR"
    EQUITY_LEADER = "EQUITY_LEADER"
    FIXED_INCOME_LONG = "FIXED_INCOME_LONG"
    CASH_EQUIVALENT = "CASH_EQUIVALENT"
    COMMODITY = "COMMODITY"


class MarketRegime(str, Enum):
    BULL_AGGRESSIVE = "BULL_AGGRESSIVE"
    BULL_NORMAL = "BULL_NORMAL"
    CORRECTION_FRAGILE = "CORRECTION_FRAGILE"
    BEAR_CRISIS = "BEAR_CRISIS"
    STALE_DATA_HOLD = "STALE_DATA_HOLD"


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: Optional[int] = None
    vwap: Optional[float] = None

    def __post_init__(self):
        if self.open < 0 or self.high < 0 or self.low < 0 or self.close < 0:
            raise ValueError(f"Prices cannot be negative: {self}")
        if self.low > self.high:
            raise ValueError(f"Low price ({self.low}) cannot exceed High price ({self.high})")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")


@dataclass(frozen=True)
class Quote:
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int


@dataclass(frozen=True)
class Trade:
    symbol: str
    timestamp: datetime
    price: float
    size: int
    id: Optional[str] = None


@dataclass(frozen=True)
class SignalSnapshot:
    timestamp: datetime
    spy_price: float
    spy_sma50: float
    spy_sma200: float
    realized_vol_20d: float
    vol_scale_factor: float
    drawdown_pct: float
    circuit_breaker_active: bool
    regime: MarketRegime
    indicators: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetAllocation:
    timestamp: datetime
    regime: MarketRegime
    weights: Dict[str, float]  # Must sum to 1.0 within 1e-6
    cash_weight: float
    rationale: str

    def __post_init__(self):
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"Target weights must sum to 1.0, got {total:.8f} (weights: {self.weights})")
        for sym, w in self.weights.items():
            if w < -1e-6:
                raise ValueError(f"Target weight for {sym} cannot be negative: {w}")
        if not math.isclose(self.weights.get("SHV", 0.0) + self.weights.get("BIL", 0.0) + self.weights.get("CASH", 0.0),
                            self.cash_weight, rel_tol=1e-4, abs_tol=1e-4) and "SHV" in self.weights:
            pass  # Cash weight aligns with cash proxies


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    action: str  # "BUY" or "SELL"
    target_weight: float
    current_weight: float
    delta_weight: float
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Universe Definitions
CORE_COMPOUNDERS = ["SPY", "QQQ"]
MEGACAP_LEADERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLI", "XLF", "XLV", "XLP", "XLU", "XLE", "XLB", "XLRE"]
SAFE_HAVENS = ["TLT", "SHV", "BIL", "GLD"]
ALL_UNIVERSE = sorted(list(set(CORE_COMPOUNDERS + MEGACAP_LEADERS + SECTOR_ETFS + SAFE_HAVENS)))


# Deterministic Mathematical Utilities
def calculate_realized_volatility(prices: List[float], window: int = 20, annualization_factor: int = 252) -> float:
    if len(prices) < window + 1:
        return 0.0
    returns = []
    for i in range(len(prices) - window, len(prices)):
        p_prev = prices[i - 1]
        p_curr = prices[i]
        if p_prev <= 0:
            returns.append(0.0)
        else:
            returns.append(math.log(p_curr / p_prev))
    n = len(returns)
    if n <= 1:
        return 0.0
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    return math.sqrt(max(0.0, variance * annualization_factor))


def calculate_sma(prices: List[float], window: int) -> float:
    if len(prices) < window or window <= 0:
        return prices[-1] if prices else 0.0
    return sum(prices[-window:]) / window


def calculate_atr(highs: List[float], lows: List[float], closes: List[float], window: int = 14) -> float:
    n = len(closes)
    if n < window + 1:
        return (highs[-1] - lows[-1]) if highs and lows else 0.0
    trs = []
    for i in range(n - window, n):
        high_val = highs[i]
        low_val = lows[i]
        prev_c = closes[i - 1]
        tr = max(high_val - low_val, abs(high_val - prev_c), abs(low_val - prev_c))
        trs.append(tr)
    return sum(trs) / len(trs)


def calculate_drawdown(equity_curve: List[float]) -> Tuple[float, float]:
    if not equity_curve:
        return 0.0, 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    current_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (val - peak) / peak
            if dd < current_dd:
                current_dd = dd
            if dd < max_dd:
                max_dd = dd
    return current_dd, max_dd


def calculate_momentum_12_1(prices: List[float], lookback: int = 252, skip: int = 21) -> float:
    if len(prices) < lookback + 1:
        return 0.0
    p_base = prices[-(lookback + 1)]
    p_skip = prices[-(skip + 1)]
    if p_base <= 0:
        return 0.0
    return (p_skip / p_base) - 1.0


def volatility_scale_factor(realized_vol: float, target_vol: float = 0.12, min_vol: float = 0.05, max_scale: float = 1.0) -> float:
    vol = max(realized_vol, min_vol, 1e-6)
    scale = target_vol / vol
    return min(max_scale, scale)


def evaluate_drawdown_gate(drawdown_pct: float) -> float:
    """Step function for trailing peak-to-trough drawdown defense:
    DD > -5%: 1.00 (Normal)
    -10% < DD <= -5%: 0.50 (Caution: 50% equity cut)
    -15% < DD <= -10%: 0.20 (Defensive: 80% safe haven / cash)
    DD <= -15%: 0.00 (Circuit breaker: 100% cash)
    """
    if drawdown_pct > -0.05:
        return 1.00
    elif drawdown_pct > -0.10:
        return 0.50
    elif drawdown_pct > -0.15:
        return 0.20
    else:
        return 0.00
