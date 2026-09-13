"""
bot.paper_account
~~~~~~~~~~~~~~~~~

Virtual Paper Trading Engine ($50,000.00 starting balance) with SQLite WAL
persistence, precision cost-basis accounting, atomic order fill simulation,
and pristine reset capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
import json
import logging
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strategy_engine.core.models import OrderIntent, OrderSide
from strategy_engine.storage.database import Database

logger = logging.getLogger("bot.paper_account")


def _resolve_order_side_str(o: Any) -> str:
    """Safely extract uppercase BUY or SELL side string from Enum, string, or action attribute."""
    side_attr = getattr(o, "side", None)
    if side_attr is not None:
        val = getattr(side_attr, "value", side_attr)
        return str(val).upper()
    action_attr = getattr(o, "action", None)
    if action_attr is not None:
        val = getattr(action_attr, "value", action_attr)
        return str(val).upper()
    return "BUY"


def _synchronized(func: Any) -> Any:
    """Decorator ensuring PaperAccountManager method execution is serialized by self._lock."""
    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


def _to_iso(dt: datetime) -> str:
    """Format datetime as UTC ISO-8601 string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(iso_str: str) -> datetime:
    """Parse ISO-8601 string to timezone-aware UTC datetime."""
    cleaned = iso_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class PaperOrderSide(str, Enum):
    """Order direction for simulated paper executions."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class PaperAccountConfig:
    """Configuration options for virtual paper trading engine."""
    initial_cash: float = 50000.00
    slippage_bps: float = 0.0          # Basis points of price slippage (e.g., 2.0 = 0.02%)
    fee_per_share: float = 0.0         # Commission / fee per share (e.g., 0.005)
    min_fee_per_order: float = 0.0     # Minimum fee per executed ticket
    cash_buffer_pct: float = 0.001     # 0.1% cash buffer to absorb price gaps on BUY orders
    allow_fractional: bool = True      # Whether fractional shares are permitted


class PositionDetail(BaseModel):
    """Granular detail for a single active portfolio holding."""
    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(..., description="Asset ticker symbol (e.g., SPY, QQQ, SHV)")
    qty: float = Field(..., ge=0.0, description="Quantity of shares held")
    shares: float = Field(..., ge=0.0, description="Alias for qty for backwards compatibility")
    avg_entry_price: float = Field(..., ge=0.0, description="Volume-weighted average purchase price")
    cost_basis: float = Field(..., ge=0.0, description="Total invested capital (qty * avg_entry_price)")
    current_price: float = Field(..., ge=0.0, description="Latest market price used for mark-to-market")
    market_value: float = Field(..., ge=0.0, description="Current market value (qty * current_price)")
    unrealized_pnl: float = Field(..., description="Unrealized dollar P&L (market_value - cost_basis)")
    unrealized_pnl_pct: float = Field(default=0.0, description="Unrealized percentage P&L")
    weight: float = Field(default=0.0, description="Fraction of Total NAV (market_value / total_nav)")

    @model_validator(mode="before")
    @classmethod
    def sync_qty_and_shares(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "qty" in values and "shares" not in values:
                values["shares"] = values["qty"]
            elif "shares" in values and "qty" not in values:
                values["qty"] = values["shares"]
        return values

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "shares": self.shares,
            "avg_entry_price": round(self.avg_entry_price, 4),
            "cost_basis": round(self.cost_basis, 2),
            "current_price": round(self.current_price, 4),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 4),
            "weight": round(self.weight, 4),
        }

    def __repr__(self) -> str:
        return (
            f"PositionDetail(symbol='{self.symbol}', qty={self.qty}, "
            f"avg_entry_price={self.avg_entry_price:.2f}, market_value={self.market_value:.2f}, "
            f"weight={self.weight:.2%})"
        )


class PortfolioSummary(BaseModel):
    """Institutional portfolio snapshot satisfying PROJECT.md interface contract."""
    model_config = ConfigDict(extra="ignore")

    cash: float = Field(..., description="Settled virtual cash balance")
    equity: float = Field(..., ge=0.0, description="Total market value of open positions")
    total_nav: float = Field(..., description="Net Asset Value (cash + equity)")
    realized_pnl: float = Field(default=0.0, description="Cumulative realized P&L since inception")
    unrealized_pnl: float = Field(default=0.0, description="Total unrealized P&L across open holdings")
    cumulative_fees: float = Field(default=0.0, description="Cumulative trading commissions/fees paid")
    positions: List[PositionDetail] = Field(default_factory=list, description="List of active positions")
    cash_weight: float = Field(default=0.0, description="Cash allocation weight (cash / total_nav)")
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Snapshot timestamp")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cash": round(self.cash, 2),
            "equity": round(self.equity, 2),
            "total_nav": round(self.total_nav, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "cumulative_fees": round(self.cumulative_fees, 2),
            "cash_weight": round(self.cash_weight, 4),
            "as_of": _to_iso(self.as_of),
            "positions": [p.to_dict() for p in self.positions],
        }

    def __repr__(self) -> str:
        return (
            f"PortfolioSummary(NAV=${self.total_nav:,.2f}, Cash=${self.cash:,.2f}, "
            f"Equity=${self.equity:,.2f}, Realized=${self.realized_pnl:,.2f}, "
            f"Unrealized=${self.unrealized_pnl:,.2f}, Positions={len(self.positions)})"
        )


class PaperTrade(BaseModel):
    """Immutable execution record for a filled trade."""
    model_config = ConfigDict(extra="ignore")

    trade_id: str = Field(..., description="Unique fill identifier (UUID hex)")
    order_id: str = Field(..., description="Correlating RebalanceOrder or OrderIntent ID")
    timestamp: datetime = Field(..., description="Fill execution UTC timestamp")
    symbol: str = Field(..., description="Instrument symbol")
    side: str = Field(..., description="BUY or SELL")
    shares: float = Field(..., gt=0.0, description="Executed share quantity")
    price: float = Field(..., gt=0.0, description="Effective fill price including slippage")
    notional: float = Field(..., gt=0.0, description="Gross trade value (shares * price)")
    fee: float = Field(default=0.0, ge=0.0, description="Simulated fee / transaction cost")
    slippage: float = Field(default=0.0, ge=0.0, description="Simulated price slippage per share")
    realized_pnl: float = Field(default=0.0, description="Realized P&L produced by this fill (0.0 for BUY)")
    cash_after: float = Field(..., description="Account cash balance immediately post-trade")
    nav_after: float = Field(..., description="Account Total NAV immediately post-trade")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "timestamp": _to_iso(self.timestamp),
            "symbol": self.symbol,
            "side": self.side,
            "shares": round(self.shares, 4),
            "price": round(self.price, 4),
            "notional": round(self.notional, 2),
            "fee": round(self.fee, 4),
            "slippage": round(self.slippage, 4),
            "realized_pnl": round(self.realized_pnl, 2),
            "cash_after": round(self.cash_after, 2),
            "nav_after": round(self.nav_after, 2),
        }


class ExecutionReport(BaseModel):
    """Aggregate result from executing a batch of rebalance orders."""
    model_config = ConfigDict(extra="ignore")

    batch_id: str = Field(..., description="Unique batch execution identifier")
    timestamp: datetime = Field(..., description="Batch completion timestamp")
    orders_received: int = Field(default=0, ge=0)
    orders_filled: int = Field(default=0, ge=0)
    orders_rejected: int = Field(default=0, ge=0)
    trades: List[PaperTrade] = Field(default_factory=list)
    total_bought_dollars: float = Field(default=0.0, ge=0.0)
    total_sold_dollars: float = Field(default=0.0, ge=0.0)
    total_fees: float = Field(default=0.0, ge=0.0)
    batch_realized_pnl: float = Field(default=0.0)
    portfolio_state_after: PortfolioSummary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "timestamp": _to_iso(self.timestamp),
            "orders_received": self.orders_received,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "trades": [t.to_dict() for t in self.trades],
            "total_bought_dollars": round(self.total_bought_dollars, 2),
            "total_sold_dollars": round(self.total_sold_dollars, 2),
            "total_fees": round(self.total_fees, 2),
            "batch_realized_pnl": round(self.batch_realized_pnl, 2),
            "portfolio_state_after": self.portfolio_state_after.to_dict(),
        }


PAPER_ACCOUNT_SCHEMA_DDL = """
-- 1. Account State Singleton Table (Enforces id = 1)
CREATE TABLE IF NOT EXISTS paper_account_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    initial_balance REAL NOT NULL DEFAULT 50000.00,
    cash REAL NOT NULL DEFAULT 50000.00 CHECK (cash >= 0.0),
    realized_pnl REAL NOT NULL DEFAULT 0.00,
    cumulative_fees REAL NOT NULL DEFAULT 0.00,
    cumulative_slippage REAL NOT NULL DEFAULT 0.00,
    equity REAL NOT NULL DEFAULT 0.00,
    total_nav REAL NOT NULL DEFAULT 50000.00,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- 2. Positions Ledger Table
CREATE TABLE IF NOT EXISTS paper_positions (
    symbol TEXT PRIMARY KEY,
    shares REAL NOT NULL CHECK (shares > 0.0),
    avg_entry_price REAL NOT NULL CHECK (avg_entry_price > 0.0),
    cost_basis REAL NOT NULL CHECK (cost_basis > 0.0),
    current_price REAL NOT NULL DEFAULT 0.0,
    market_value REAL NOT NULL DEFAULT 0.0,
    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
    weight REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_paper_positions_symbol ON paper_positions(symbol);

-- 3. Trade Execution History Table
CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    shares REAL NOT NULL CHECK (shares > 0.0),
    price REAL NOT NULL CHECK (price > 0.0),
    notional REAL NOT NULL CHECK (notional > 0.0),
    fee REAL NOT NULL DEFAULT 0.0,
    slippage REAL NOT NULL DEFAULT 0.0,
    realized_pnl REAL NOT NULL DEFAULT 0.0,
    cash_after REAL NOT NULL,
    nav_after REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_timestamp ON paper_trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol ON paper_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_paper_trades_order ON paper_trades(order_id);

-- 4. Historical Equity Snapshots Table
CREATE TABLE IF NOT EXISTS paper_equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cash REAL NOT NULL,
    equity REAL NOT NULL,
    total_nav REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    cumulative_fees REAL NOT NULL DEFAULT 0.0,
    positions_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_paper_snapshots_timestamp ON paper_equity_snapshots(timestamp DESC);
"""


class PaperAccountManager:
    """Virtual paper trading account manager providing persistent $50k WAL ledger."""

    def __init__(
        self,
        db: Optional[Union[Database, str, Path]] = None,
        config: Optional[PaperAccountConfig] = None,
        auto_init: bool = True,
    ):
        self._lock = threading.RLock()
        if db is None:
            self.db = Database("strategy_engine.db")
        elif isinstance(db, Database):
            self.db = db
        else:
            self.db = Database(str(db))

        self.config = config or PaperAccountConfig()

        if auto_init:
            self.init_schema()

    def init_schema(self) -> None:
        """Create tables if not existing and bootstrap $50,000.00 pristine account if empty."""
        with self._lock:
            with self.db.transaction() as conn:
                conn.executescript(PAPER_ACCOUNT_SCHEMA_DDL)
                cursor = conn.cursor()
                cursor.execute("SELECT id, cash FROM paper_account_state WHERE id = 1;")
                row = cursor.fetchone()
                if row is None:
                    now_iso = datetime.now(timezone.utc).isoformat()
                    cursor.execute(
                        """
                        INSERT INTO paper_account_state (
                            id, initial_balance, cash, realized_pnl, cumulative_fees,
                            cumulative_slippage, equity, total_nav, created_at, updated_at
                        ) VALUES (1, ?, ?, 0.0, 0.0, 0.0, 0.0, ?, ?, ?);
                        """,
                        (
                            self.config.initial_cash,
                            self.config.initial_cash,
                            self.config.initial_cash,
                            now_iso,
                            now_iso,
                        ),
                    )
                    cursor.execute(
                        """
                        INSERT INTO paper_equity_snapshots (
                            timestamp, cash, equity, total_nav, realized_pnl,
                            unrealized_pnl, cumulative_fees, positions_json
                        ) VALUES (?, ?, 0.0, ?, 0.0, 0.0, 0.0, '{}');
                        """,
                        (now_iso, self.config.initial_cash, self.config.initial_cash),
                    )
                    logger.info(
                        "Initialized pristine paper trading account with $%.2f cash balance.",
                        self.config.initial_cash,
                    )

    def get_portfolio_state(
        self,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> PortfolioSummary:
        """Fetch current portfolio state matching PROJECT.md interface contract."""
        with self._lock:
            current_prices = current_prices or {}
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT cash, realized_pnl, cumulative_fees, cumulative_slippage FROM paper_account_state WHERE id = 1;"
                )
                state_row = cursor.fetchone()
                if not state_row:
                    self.init_schema()
                    cash = self.config.initial_cash
                    realized_pnl = 0.0
                    cumulative_fees = 0.0
                else:
                    cash = float(state_row["cash"])
                    realized_pnl = float(state_row["realized_pnl"])
                    cumulative_fees = float(state_row["cumulative_fees"])

                cursor.execute("SELECT * FROM paper_positions ORDER BY symbol ASC;")
                pos_rows = cursor.fetchall()

                positions: List[PositionDetail] = []
                total_equity = 0.0
                total_unrealized_pnl = 0.0

                # First pass: calculate equity and unrealized P&L
                for r in pos_rows:
                    sym = r["symbol"]
                    shares = float(r["shares"])
                    avg_price = float(r["avg_entry_price"])
                    cost_basis = float(r["cost_basis"])

                    # Determine current price
                    c_price = current_prices.get(sym, float(r["current_price"]))
                    if c_price <= 0.0:
                        c_price = avg_price

                    mkt_val = shares * c_price
                    u_pnl = mkt_val - cost_basis
                    u_pnl_pct = (u_pnl / cost_basis) if cost_basis > 0 else 0.0

                    total_equity += mkt_val
                    total_unrealized_pnl += u_pnl

                    positions.append(
                        PositionDetail(
                            symbol=sym,
                            qty=shares,
                            shares=shares,
                            avg_entry_price=avg_price,
                            cost_basis=cost_basis,
                            current_price=c_price,
                            market_value=mkt_val,
                            unrealized_pnl=u_pnl,
                            unrealized_pnl_pct=u_pnl_pct,
                            weight=0.0,  # Computed in second pass
                        )
                    )

                total_nav = cash + total_equity
                cash_weight = (cash / total_nav) if total_nav > 0 else 1.0

                # Second pass: compute weights
                for p in positions:
                    p.weight = (p.market_value / total_nav) if total_nav > 0 else 0.0

                return PortfolioSummary(
                    cash=cash,
                    equity=total_equity,
                    total_nav=total_nav,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=total_unrealized_pnl,
                    cumulative_fees=cumulative_fees,
                    positions=positions,
                    cash_weight=cash_weight,
                    as_of=datetime.now(timezone.utc),
                )


    @_synchronized
    def execute_rebalance_orders(
        self,
        orders: List[OrderIntent],
        current_prices: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> ExecutionReport:
        """Atomically execute rebalance orders (SELLs before BUYs) within an ACID transaction."""
        now_ts = timestamp or datetime.now(timezone.utc)
        batch_id = uuid.uuid4().hex

        if not orders:
            curr_state = self.get_portfolio_state(current_prices)
            return ExecutionReport(
                batch_id=batch_id,
                timestamp=now_ts,
                orders_received=0,
                orders_filled=0,
                orders_rejected=0,
                trades=[],
                total_bought_dollars=0.0,
                total_sold_dollars=0.0,
                total_fees=0.0,
                batch_realized_pnl=0.0,
                portfolio_state_after=curr_state,
            )

        # 1. Partition orders: SELLs first (liberating cash), then BUYs
        sell_orders: List[OrderIntent] = []
        buy_orders: List[OrderIntent] = []
        for o in orders:
            side_str = _resolve_order_side_str(o)
            if side_str == "SELL":
                sell_orders.append(o)
            elif side_str == "BUY":
                buy_orders.append(o)

        sequenced_orders = sell_orders + buy_orders

        trades_executed: List[PaperTrade] = []
        orders_filled = 0
        orders_rejected = 0
        total_bought = 0.0
        total_sold = 0.0
        total_fees = 0.0
        batch_realized_pnl = 0.0

        with self.db.transaction() as conn:
            cursor = conn.cursor()

            # Read current settled account state
            cursor.execute(
                "SELECT cash, realized_pnl, cumulative_fees, cumulative_slippage FROM paper_account_state WHERE id = 1;"
            )
            acct_row = cursor.fetchone()
            cash = float(acct_row["cash"])
            cumulative_realized_pnl = float(acct_row["realized_pnl"])
            cumulative_fees = float(acct_row["cumulative_fees"])
            cumulative_slippage = float(acct_row["cumulative_slippage"])

            # Read existing active positions into local dict
            cursor.execute("SELECT * FROM paper_positions;")
            pos_dict: Dict[str, Dict[str, float]] = {}
            for r in cursor.fetchall():
                pos_dict[r["symbol"]] = {
                    "shares": float(r["shares"]),
                    "avg_entry_price": float(r["avg_entry_price"]),
                    "cost_basis": float(r["cost_basis"]),
                    "current_price": float(r["current_price"]),
                }

            # Process sequenced orders
            for o in sequenced_orders:
                sym = o.symbol
                side_str = _resolve_order_side_str(o)
                order_id = getattr(o, "id", None) or uuid.uuid4().hex

                # Resolve price
                raw_price = current_prices.get(sym) or o.estimated_price or 0.0
                if raw_price <= 0.0:
                    if sym in pos_dict and pos_dict[sym]["current_price"] > 0:
                        raw_price = pos_dict[sym]["current_price"]
                    else:
                        fallbacks = {
                            "SPY": 500.0, "QQQ": 440.0, "XLK": 210.0, "XLE": 85.0,
                            "XLV": 140.0, "XLI": 120.0, "XLU": 65.0, "TLT": 95.0,
                            "SHV": 110.0, "GLD": 215.0, "XLP": 75.0, "XLF": 40.0,
                            "XLY": 180.0, "XLC": 80.0, "XLB": 90.0, "XLRE": 40.0,
                            "AAPL": 220.0, "MSFT": 420.0, "NVDA": 120.0, "AMZN": 180.0,
                        }
                        raw_price = fallbacks.get(sym, 100.0)

                # Compute slippage
                slippage_mult = self.config.slippage_bps / 10000.0
                if side_str == "BUY":
                    fill_price = raw_price * (1.0 + slippage_mult)
                    slippage_per_share = raw_price * slippage_mult
                else:
                    fill_price = raw_price * (1.0 - slippage_mult)
                    slippage_per_share = raw_price * slippage_mult

                # Determine target shares
                if o.delta_shares is not None and abs(o.delta_shares) > 0:
                    shares_target = abs(o.delta_shares)
                elif o.notional is not None and o.notional > 0:
                    shares_target = o.notional / fill_price
                elif o.delta_dollars is not None and abs(o.delta_dollars) > 0:
                    shares_target = abs(o.delta_dollars) / fill_price
                else:
                    logger.warning("Rejecting order %s for %s: zero or missing share size", order_id, sym)
                    orders_rejected += 1
                    continue

                if not self.config.allow_fractional:
                    shares_target = float(int(shares_target))
                else:
                    shares_target = round(shares_target, 4)

                if shares_target <= 0.00001:
                    orders_rejected += 1
                    continue

                # --- EXECUTE SELL ---
                if side_str == "SELL":
                    if sym not in pos_dict or pos_dict[sym]["shares"] <= 0.00001:
                        if sym in ("SHV", "BIL", "CASH"):
                            # Cash proxy: account already holds uninvested cash
                            orders_filled += 1
                            continue
                        logger.warning("Rejecting SELL order for %s: no open position held", sym)
                        orders_rejected += 1
                        continue

                    held_shares = pos_dict[sym]["shares"]
                    shares_to_sell = min(shares_target, held_shares)
                    avg_entry = pos_dict[sym]["avg_entry_price"]

                    fee = max(self.config.min_fee_per_order, shares_to_sell * self.config.fee_per_share)
                    gross_notional = shares_to_sell * fill_price
                    net_proceeds = gross_notional - fee

                    # Realized P&L
                    pnl_realized = (fill_price - avg_entry) * shares_to_sell - fee

                    # Update cash and accumulators
                    cash += net_proceeds
                    cumulative_realized_pnl += pnl_realized
                    batch_realized_pnl += pnl_realized
                    cumulative_fees += fee
                    total_fees += fee
                    cumulative_slippage += (slippage_per_share * shares_to_sell)
                    total_sold += gross_notional

                    remaining_shares = held_shares - shares_to_sell
                    if remaining_shares <= 0.0001:
                        # Position fully closed
                        del pos_dict[sym]
                        cursor.execute("DELETE FROM paper_positions WHERE symbol = ?;", (sym,))
                    else:
                        remaining_cost_basis = remaining_shares * avg_entry
                        pos_dict[sym]["shares"] = remaining_shares
                        pos_dict[sym]["cost_basis"] = remaining_cost_basis
                        pos_dict[sym]["current_price"] = fill_price
                        cursor.execute(
                            """
                            UPDATE paper_positions 
                            SET shares = ?, cost_basis = ?, current_price = ?, updated_at = ?
                            WHERE symbol = ?;
                            """,
                            (remaining_shares, remaining_cost_basis, fill_price, _to_iso(now_ts), sym),
                        )

                    # Compute NAV after trade
                    current_equity = sum(p["shares"] * p["current_price"] for p in pos_dict.values())
                    nav_after = cash + current_equity

                    trade = PaperTrade(
                        trade_id=uuid.uuid4().hex,
                        order_id=order_id,
                        timestamp=now_ts,
                        symbol=sym,
                        side="SELL",
                        shares=shares_to_sell,
                        price=fill_price,
                        notional=gross_notional,
                        fee=fee,
                        slippage=slippage_per_share * shares_to_sell,
                        realized_pnl=pnl_realized,
                        cash_after=cash,
                        nav_after=nav_after,
                    )
                    trades_executed.append(trade)
                    orders_filled += 1

                # --- EXECUTE BUY ---
                elif side_str == "BUY":
                    # Available spendable cash check with buffer
                    spendable_cash = max(0.0, cash * (1.0 - self.config.cash_buffer_pct))
                    fee_estimate = max(self.config.min_fee_per_order, shares_target * self.config.fee_per_share)
                    required_cost = (shares_target * fill_price) + fee_estimate

                    if required_cost > spendable_cash:
                        # Adaptive clamping to protect cash invariant
                        max_affordable_notional = max(0.0, spendable_cash - self.config.min_fee_per_order)
                        if max_affordable_notional < 10.0:  # Less than $10 remaining, reject order
                            logger.warning("Rejecting BUY order for %s: insufficient spendable cash ($%.2f)", sym, cash)
                            orders_rejected += 1
                            continue

                        effective_per_share = fill_price + self.config.fee_per_share
                        if effective_per_share <= 0:
                            orders_rejected += 1
                            continue

                        clamped_shares = max_affordable_notional / effective_per_share
                        if not self.config.allow_fractional:
                            clamped_shares = float(math.floor(clamped_shares))
                        else:
                            clamped_shares = math.floor(clamped_shares * 10000.0) / 10000.0

                        if clamped_shares <= 0.0001:
                            orders_rejected += 1
                            continue

                        logger.info(
                            "Clamping BUY order %s for %s from %.4f to %.4f shares to preserve cash ($%.2f)",
                            order_id,
                            sym,
                            shares_target,
                            clamped_shares,
                            cash,
                        )
                        shares_to_buy = clamped_shares
                    else:
                        shares_to_buy = shares_target

                    fee = max(self.config.min_fee_per_order, shares_to_buy * self.config.fee_per_share)
                    gross_notional = shares_to_buy * fill_price
                    total_outflow = gross_notional + fee

                    while total_outflow > spendable_cash and shares_to_buy > 0.0001:
                        shares_to_buy = math.floor((shares_to_buy - 0.0001) * 10000.0) / 10000.0
                        fee = max(self.config.min_fee_per_order, shares_to_buy * self.config.fee_per_share)
                        gross_notional = shares_to_buy * fill_price
                        total_outflow = gross_notional + fee

                    # Deduct cash with floor protection
                    cash = max(0.0, cash - total_outflow)
                    if abs(cash) < 1e-9:
                        cash = 0.0
                    cumulative_fees += fee
                    total_fees += fee
                    cumulative_slippage += (slippage_per_share * shares_to_buy)
                    total_bought += gross_notional

                    # Update position in ledger
                    if sym in pos_dict:
                        existing_shares = pos_dict[sym]["shares"]
                        existing_cost = pos_dict[sym]["cost_basis"]
                        new_shares = existing_shares + shares_to_buy
                        new_cost = existing_cost + gross_notional
                        new_avg_price = new_cost / new_shares

                        pos_dict[sym]["shares"] = new_shares
                        pos_dict[sym]["avg_entry_price"] = new_avg_price
                        pos_dict[sym]["cost_basis"] = new_cost
                        pos_dict[sym]["current_price"] = fill_price
                    else:
                        pos_dict[sym] = {
                            "shares": shares_to_buy,
                            "avg_entry_price": fill_price,
                            "cost_basis": gross_notional,
                            "current_price": fill_price,
                        }

                    cursor.execute(
                        """
                        INSERT INTO paper_positions (
                            symbol, shares, avg_entry_price, cost_basis, current_price,
                            market_value, unrealized_pnl, weight, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 0.0, 0.0, ?)
                        ON CONFLICT(symbol) DO UPDATE SET
                            shares = paper_positions.shares + excluded.shares,
                            cost_basis = paper_positions.cost_basis + excluded.cost_basis,
                            avg_entry_price = (paper_positions.cost_basis + excluded.cost_basis) / (paper_positions.shares + excluded.shares),
                            current_price = excluded.current_price,
                            market_value = (paper_positions.shares + excluded.shares) * excluded.current_price,
                            updated_at = excluded.updated_at;
                        """,
                        (
                            sym,
                            shares_to_buy,
                            fill_price,
                            gross_notional,
                            fill_price,
                            gross_notional,
                            _to_iso(now_ts),
                        ),
                    )

                    # Compute NAV after trade
                    current_equity = sum(p["shares"] * p["current_price"] for p in pos_dict.values())
                    nav_after = cash + current_equity

                    trade = PaperTrade(
                        trade_id=uuid.uuid4().hex,
                        order_id=order_id,
                        timestamp=now_ts,
                        symbol=sym,
                        side="BUY",
                        shares=shares_to_buy,
                        price=fill_price,
                        notional=gross_notional,
                        fee=fee,
                        slippage=slippage_per_share * shares_to_buy,
                        realized_pnl=0.0,
                        cash_after=cash,
                        nav_after=nav_after,
                    )
                    trades_executed.append(trade)
                    orders_filled += 1

            # Insert all trade execution records
            if trades_executed:
                trade_rows = [
                    (
                        t.trade_id,
                        t.order_id,
                        _to_iso(t.timestamp),
                        t.symbol,
                        t.side,
                        t.shares,
                        t.price,
                        t.notional,
                        t.fee,
                        t.slippage,
                        t.realized_pnl,
                        t.cash_after,
                        t.nav_after,
                    )
                    for t in trades_executed
                ]
                cursor.executemany(
                    """
                    INSERT INTO paper_trades (
                        trade_id, order_id, timestamp, symbol, side, shares, price,
                        notional, fee, slippage, realized_pnl, cash_after, nav_after
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    trade_rows,
                )

            # Update final account state
            final_equity = sum(p["shares"] * p["current_price"] for p in pos_dict.values())
            final_nav = cash + final_equity

            cursor.execute(
                """
                UPDATE paper_account_state
                SET cash = ?, realized_pnl = ?, cumulative_fees = ?, cumulative_slippage = ?,
                    equity = ?, total_nav = ?, updated_at = ?
                WHERE id = 1;
                """,
                (
                    cash,
                    cumulative_realized_pnl,
                    cumulative_fees,
                    cumulative_slippage,
                    final_equity,
                    final_nav,
                    _to_iso(now_ts),
                ),
            )

            # Record final equity snapshot
            pos_json = json.dumps(
                {
                    sym: {
                        "shares": round(p["shares"], 4),
                        "price": round(p["current_price"], 4),
                        "market_value": round(p["shares"] * p["current_price"], 2),
                    }
                    for sym, p in pos_dict.items()
                }
            )
            cursor.execute(
                """
                INSERT INTO paper_equity_snapshots (
                    timestamp, cash, equity, total_nav, realized_pnl,
                    unrealized_pnl, cumulative_fees, positions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    _to_iso(now_ts),
                    cash,
                    final_equity,
                    final_nav,
                    cumulative_realized_pnl,
                    sum(p["shares"] * (p["current_price"] - p["avg_entry_price"]) for p in pos_dict.values()),
                    cumulative_fees,
                    pos_json,
                ),
            )

        post_state = self.get_portfolio_state(current_prices)
        return ExecutionReport(
            batch_id=batch_id,
            timestamp=now_ts,
            orders_received=len(orders),
            orders_filled=orders_filled,
            orders_rejected=orders_rejected,
            trades=trades_executed,
            total_bought_dollars=total_bought,
            total_sold_dollars=total_sold,
            total_fees=total_fees,
            batch_realized_pnl=batch_realized_pnl,
            portfolio_state_after=post_state,
        )

    @_synchronized
    def execute_single_order(
        self,
        symbol: str,
        side: Union[PaperOrderSide, OrderSide, str],
        shares: float,
        price: float,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> PaperTrade:
        """Execute a single atomic simulated order."""
        side_raw = getattr(side, "value", side)
        side_str = str(side_raw).upper()
        side_enum = OrderSide(side_str)
        intent = OrderIntent(
            symbol=symbol,
            action=side_enum.value,
            side=side_enum,
            delta_shares=shares if side_enum == OrderSide.BUY else -shares,
            estimated_price=price,
            notional=shares * price,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        report = self.execute_rebalance_orders(
            orders=[intent],
            current_prices={symbol: price},
            timestamp=timestamp,
        )
        if not report.trades:
            raise RuntimeError(f"Failed to execute single order for {symbol}: order rejected or invalid size")
        return report.trades[0]

    @_synchronized
    def update_market_prices(
        self,
        current_prices: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> PortfolioSummary:
        """Mark-to-market all active positions and update unrealized P&L."""
        now_ts = timestamp or datetime.now(timezone.utc)
        summary = self.get_portfolio_state(current_prices)

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            for p in summary.positions:
                cursor.execute(
                    """
                    UPDATE paper_positions
                    SET current_price = ?, market_value = ?, unrealized_pnl = ?, weight = ?, updated_at = ?
                    WHERE symbol = ?;
                    """,
                    (p.current_price, p.market_value, p.unrealized_pnl, p.weight, _to_iso(now_ts), p.symbol),
                )
            cursor.execute(
                """
                UPDATE paper_account_state
                SET equity = ?, total_nav = ?, updated_at = ?
                WHERE id = 1;
                """,
                (summary.equity, summary.total_nav, _to_iso(now_ts)),
            )

        return summary

    @_synchronized
    def take_equity_snapshot(
        self,
        current_prices: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Record point-in-time equity snapshot in paper_equity_snapshots and portfolio_states."""
        now_ts = timestamp or datetime.now(timezone.utc)
        summary = self.get_portfolio_state(current_prices)
        pos_json = json.dumps([p.to_dict() for p in summary.positions])

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO paper_equity_snapshots (
                    timestamp, cash, equity, total_nav, realized_pnl,
                    unrealized_pnl, cumulative_fees, positions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    _to_iso(now_ts),
                    summary.cash,
                    summary.equity,
                    summary.total_nav,
                    summary.realized_pnl,
                    summary.unrealized_pnl,
                    summary.cumulative_fees,
                    pos_json,
                ),
            )
            # Dual write for backward compatibility with StrategyEngine's portfolio_states table if present
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO portfolio_states (timestamp, cash, equity, total_nav, positions_json)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (_to_iso(now_ts), summary.cash, summary.equity, summary.total_nav, pos_json),
                )
            except sqlite3.OperationalError:
                pass

        return summary.to_dict()

    @_synchronized
    def get_positions(self) -> List[PositionDetail]:
        """Fetch current list of active position details."""
        return self.get_portfolio_state().positions

    @_synchronized
    def get_trade_history(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = 100,
    ) -> List[PaperTrade]:
        """Fetch chronologically ordered execution trade records."""
        conditions = []
        params: List[Any] = []
        if start:
            conditions.append("timestamp >= ?")
            params.append(_to_iso(start))
        if end:
            conditions.append("timestamp <= ?")
            params.append(_to_iso(end))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit else ""
        query = f"SELECT * FROM paper_trades {where_clause} ORDER BY timestamp ASC {limit_clause};"

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [
                PaperTrade(
                    trade_id=r["trade_id"],
                    order_id=r["order_id"],
                    timestamp=_from_iso(r["timestamp"]),
                    symbol=r["symbol"],
                    side=r["side"],
                    shares=float(r["shares"]),
                    price=float(r["price"]),
                    notional=float(r["notional"]),
                    fee=float(r["fee"]),
                    slippage=float(r["slippage"]),
                    realized_pnl=float(r["realized_pnl"]),
                    cash_after=float(r["cash_after"]),
                    nav_after=float(r["nav_after"]),
                )
                for r in rows
            ]

    @_synchronized
    def get_equity_history(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = 500,
    ) -> List[Dict[str, Any]]:
        """Fetch historical equity snapshots."""
        conditions = []
        params: List[Any] = []
        if start:
            conditions.append("timestamp >= ?")
            params.append(_to_iso(start))
        if end:
            conditions.append("timestamp <= ?")
            params.append(_to_iso(end))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit else ""
        query = f"SELECT * FROM paper_equity_snapshots {where_clause} ORDER BY timestamp ASC {limit_clause};"

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["positions"] = json.loads(d["positions_json"])
                results.append(d)
            return results

    @_synchronized
    def reset_to_pristine(self) -> PortfolioSummary:
        """Atomic purge of all smoke test data and restore exact $50,000.00 cash balance."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM paper_positions;")
            cursor.execute("DELETE FROM paper_trades;")
            cursor.execute("DELETE FROM paper_equity_snapshots;")
            try:
                cursor.execute("DELETE FROM rebalance_orders;")
            except sqlite3.OperationalError:
                pass

            cursor.execute(
                """
                INSERT OR REPLACE INTO paper_account_state (
                    id, initial_balance, cash, realized_pnl, cumulative_fees,
                    cumulative_slippage, equity, total_nav, created_at, updated_at
                ) VALUES (1, ?, ?, 0.0, 0.0, 0.0, 0.0, ?, ?, ?);
                """,
                (
                    self.config.initial_cash,
                    self.config.initial_cash,
                    self.config.initial_cash,
                    now_iso,
                    now_iso,
                ),
            )

            cursor.execute(
                """
                INSERT INTO paper_equity_snapshots (
                    timestamp, cash, equity, total_nav, realized_pnl,
                    unrealized_pnl, cumulative_fees, positions_json
                ) VALUES (?, ?, 0.0, ?, 0.0, 0.0, 0.0, '{}');
                """,
                (now_iso, self.config.initial_cash, self.config.initial_cash),
            )

        logger.info("Successfully reset paper trading account to pristine $50,000.00 state.")
        return self.get_portfolio_state()

    def close(self) -> None:
        """Close database connection."""
        self.db.close()
