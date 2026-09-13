"""Canonical interface contracts and progressive test adapters for DynamicLongTermStrategyBot.

Conforms strictly to ORIGINAL_REQUEST.md and PROJECT.md § Interface Contracts:
- Features 1-5 (M1): Virtual Paper Trading Engine, Ledger, AlpacaRelay Ingestion, Fallback Feed, 4-Regime Strategy
- Features 6-9 (M2): Discord v2 Alert Cards (Broken, Recovered, Trade), Rate Limiter & Pytest Suppression
- Features 10-14 (M3): Light & Airy Mobile Dashboard (375px-430px), Live Metrics, Alert Banner, Operator Controls
- Features 15-17 (M4): Adversarial Smoke Testing, Usability Audit, Pristine State Reset
- Features 18-20 (M5): Git Repository, GitHub Remote, Token-Free Public Railway Deployment
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# 1. Domain Models & Schemas (PROJECT.md § Interface Contracts)
# ==============================================================================

@dataclass
class PositionDetail:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float


@dataclass
class PortfolioSummary:
    cash: float
    equity: float
    total_nav: float
    realized_pnl: float
    unrealized_pnl: float
    positions: List[PositionDetail] = field(default_factory=list)


@dataclass
class ConnectionStatus:
    is_connected: bool
    feed_source: str  # "alpaca_relay" or "synthetic_fallback"
    alert_banner_active: bool
    last_heartbeat_timestamp: str


@dataclass
class RebalanceOrder:
    symbol: str
    action: str  # "BUY" or "SELL"
    shares: float
    price: float
    target_weight: float = 0.0
    current_weight: float = 0.0


@dataclass
class DiscordEmbedCard:
    color: int
    title: str
    description: str = ""
    fields: List[Dict[str, Any]] = field(default_factory=list)
    url: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Institutional Discord Card Color Codes
DISCORD_COLOR_BROKEN = 0xE53935    # Red (decimal: 15022389)
DISCORD_COLOR_RECOVERED = 0x43A047  # Green (decimal: 4431943)
DISCORD_COLOR_TRADE = 0x1E88E5      # Blue (decimal: 2001125)


# ==============================================================================
# 2. Paper Trading Account & SQLite WAL Ledger (Features 1, 2, 17)
# ==============================================================================

class PaperAccountManagerContract:
    """Isolated paper trading account manager with SQLite WAL storage."""

    INITIAL_CASH = 50000.00

    def __init__(self, db_path: str):
        self.db_path = db_path
        parent_dir = os.path.dirname(db_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if not hasattr(self, "_mem_conn") or self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:")
                self._mem_conn.row_factory = sqlite3.Row
            return self._mem_conn
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account_balance (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    qty REAL NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    timestamp TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_weight REAL NOT NULL,
                    current_weight REAL NOT NULL,
                    delta_weight REAL NOT NULL,
                    target_shares REAL NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
            """)

            row = conn.execute("SELECT cash FROM account_balance WHERE id = 1;").fetchone()
            if row is None:
                now_str = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO account_balance (id, cash, realized_pnl, updated_at) VALUES (1, ?, 0.0, ?);",
                    (self.INITIAL_CASH, now_str)
                )
            conn.commit()

    def get_cash_balance(self) -> float:
        with self._get_connection() as conn:
            row = conn.execute("SELECT cash FROM account_balance WHERE id = 1;").fetchone()
            return float(row["cash"]) if row else self.INITIAL_CASH

    def execute_order(self, symbol: str, action: str, shares: float, price: float, order_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute a buy or sell order against the paper ledger atomically."""
        if shares <= 0:
            raise ValueError(f"Shares must be positive, got {shares}")
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        action = action.upper()
        if action not in ("BUY", "SELL"):
            raise ValueError(f"Invalid order action: {action}")

        if order_id is None:
            order_id = f"ord_{int(time.time() * 1000)}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE;")
            bal_row = cursor.execute("SELECT cash, realized_pnl FROM account_balance WHERE id = 1;").fetchone()
            cash = float(bal_row["cash"])
            total_realized_pnl = float(bal_row["realized_pnl"])

            pos_row = cursor.execute(
                "SELECT qty, avg_entry_price, current_price FROM positions WHERE symbol = ?;",
                (symbol,)
            ).fetchone()
            current_qty = float(pos_row["qty"]) if pos_row else 0.0
            avg_entry_price = float(pos_row["avg_entry_price"]) if pos_row else 0.0

            realized_pnl = 0.0
            now_str = datetime.now(timezone.utc).isoformat()

            if action == "BUY":
                cost = shares * price
                if cash < cost:
                    raise ValueError(f"Insufficient cash: required {cost:.2f}, available {cash:.2f}")
                new_cash = cash - cost
                new_qty = current_qty + shares
                new_avg_price = ((current_qty * avg_entry_price) + cost) / new_qty

                cursor.execute(
                    """
                    INSERT INTO positions (symbol, qty, avg_entry_price, current_price, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        qty = excluded.qty,
                        avg_entry_price = excluded.avg_entry_price,
                        current_price = excluded.current_price,
                        updated_at = excluded.updated_at;
                    """,
                    (symbol, new_qty, new_avg_price, price, now_str)
                )
                cursor.execute(
                    "UPDATE account_balance SET cash = ?, updated_at = ? WHERE id = 1;",
                    (new_cash, now_str)
                )
            elif action == "SELL":
                if current_qty < shares:
                    raise ValueError(f"Insufficient position in {symbol}: sell {shares} > held {current_qty}")
                proceeds = shares * price
                realized_pnl = shares * (price - avg_entry_price)
                new_cash = cash + proceeds
                new_realized_total = total_realized_pnl + realized_pnl
                new_qty = current_qty - shares

                if new_qty <= 1e-7:
                    cursor.execute("DELETE FROM positions WHERE symbol = ?;", (symbol,))
                else:
                    cursor.execute(
                        "UPDATE positions SET qty = ?, current_price = ?, updated_at = ? WHERE symbol = ?;",
                        (new_qty, price, now_str, symbol)
                    )

                cursor.execute(
                    "UPDATE account_balance SET cash = ?, realized_pnl = ?, updated_at = ? WHERE id = 1;",
                    (new_cash, new_realized_total, now_str)
                )

            cursor.execute(
                """
                INSERT INTO executions (order_id, symbol, action, shares, price, cost_basis, realized_pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (order_id, symbol, action, shares, price, shares * price, realized_pnl, now_str)
            )
            conn.commit()

        return {
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "shares": shares,
            "price": price,
            "realized_pnl": realized_pnl,
        }

    def update_market_prices(self, price_map: Dict[str, float]) -> None:
        """Update current prices for all open positions."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            for symbol, price in price_map.items():
                if price <= 0:
                    continue
                conn.execute(
                    "UPDATE positions SET current_price = ?, updated_at = ? WHERE symbol = ?;",
                    (price, now_str, symbol)
                )
            conn.commit()

    def get_portfolio_state(self) -> PortfolioSummary:
        """Compute exact portfolio state matching PortfolioSummary schema."""
        with self._get_connection() as conn:
            bal_row = conn.execute("SELECT cash, realized_pnl FROM account_balance WHERE id = 1;").fetchone()
            cash = float(bal_row["cash"]) if bal_row else self.INITIAL_CASH
            realized_pnl = float(bal_row["realized_pnl"]) if bal_row else 0.0

            pos_rows = conn.execute("SELECT symbol, qty, avg_entry_price, current_price FROM positions;").fetchall()
            positions: List[PositionDetail] = []
            total_equity = 0.0

            for r in pos_rows:
                qty = float(r["qty"])
                avg_price = float(r["avg_entry_price"])
                curr_price = float(r["current_price"])
                mkt_val = qty * curr_price
                unrealized = qty * (curr_price - avg_price)
                total_equity += mkt_val
                positions.append(
                    PositionDetail(
                        symbol=r["symbol"],
                        qty=qty,
                        avg_entry_price=avg_price,
                        current_price=curr_price,
                        market_value=mkt_val,
                        unrealized_pnl=unrealized,
                        weight=0.0,
                    )
                )

            total_nav = cash + total_equity
            total_unrealized = sum(p.unrealized_pnl for p in positions)

            # Assign weights
            for p in positions:
                p.weight = (p.market_value / total_nav) if total_nav > 0 else 0.0

            return PortfolioSummary(
                cash=cash,
                equity=total_equity,
                total_nav=total_nav,
                realized_pnl=realized_pnl,
                unrealized_pnl=total_unrealized,
                positions=positions,
            )

    def reset_to_pristine(self) -> None:
        """Atomic reset restoring $50,000.00 cash, 0 positions, 0 open orders, 0 executions."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE;")
            cursor.execute("DELETE FROM positions;")
            cursor.execute("DELETE FROM executions;")
            cursor.execute("DELETE FROM order_intents;")
            cursor.execute(
                "UPDATE account_balance SET cash = ?, realized_pnl = 0.0, updated_at = ? WHERE id = 1;",
                (self.INITIAL_CASH, now_str)
            )
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")


# ==============================================================================
# 3. Data Feed Manager & Fallback Simulation (Features 3, 4, 13)
# ==============================================================================

class DataFeedManagerContract:
    """Manages connection to AlpacaRelay and transitions to synthetic fallback."""

    def __init__(self, token: str = "test-relay-token", feed_source: str = "alpaca_relay"):
        self.token = token
        self.feed_source = feed_source
        self.is_connected = (feed_source == "alpaca_relay")
        self.alert_banner_active = (feed_source != "alpaca_relay")
        self.last_heartbeat = datetime.now(timezone.utc).isoformat()
        self._price_cache: Dict[str, float] = {
            "SPY": 500.0, "QQQ": 440.0, "TLT": 95.0, "SHV": 110.0, "GLD": 215.0, "XLK": 205.0
        }

    def trigger_disconnect(self, reason: str = "upstream_disconnected") -> None:
        """Handle upstream disconnect event, fallback to simulation, set banner."""
        self.is_connected = False
        self.feed_source = "synthetic_fallback"
        self.alert_banner_active = True
        self.last_heartbeat = datetime.now(timezone.utc).isoformat()

    def trigger_reconnect(self) -> None:
        """Handle upstream reconnect event, restore live feed, clear banner."""
        self.is_connected = True
        self.feed_source = "alpaca_relay"
        self.alert_banner_active = False
        self.last_heartbeat = datetime.now(timezone.utc).isoformat()

    def get_connection_status(self) -> ConnectionStatus:
        return ConnectionStatus(
            is_connected=self.is_connected,
            feed_source=self.feed_source,
            alert_banner_active=self.alert_banner_active,
            last_heartbeat_timestamp=self.last_heartbeat,
        )

    def get_latest_price(self, symbol: str) -> float:
        """Return current price, using synthetic walk if in fallback mode."""
        base = self._price_cache.get(symbol, 100.0)
        if self.feed_source == "synthetic_fallback":
            # Tiny deterministic synthetic drift for simulation
            base = base * 1.0001
            self._price_cache[symbol] = base
        return base

    def get_universe_prices(self) -> Dict[str, float]:
        return {sym: self.get_latest_price(sym) for sym in self._price_cache}


# ==============================================================================
# 4. Institutional Discord v2 Alerts & Rate Limiter (Features 6, 7, 8, 9)
# ==============================================================================

class DiscordNotifierContract:
    """Institutional Discord v2 notification dispatcher with rate-limiting and pytest suppression."""

    def __init__(
        self,
        webhook_url: Optional[str] = "https://discord.com/api/webhooks/mock/test",
        rate_limit_interval_s: float = 2.0,
        max_backoff_sleep_s: float = 5.0,
        suppress_in_test: bool = True,
    ):
        self.webhook_url = webhook_url
        self.rate_limit_interval_s = rate_limit_interval_s
        self.max_backoff_sleep_s = max_backoff_sleep_s
        self.suppress_in_test = suppress_in_test
        self.last_post_time: float = 0.0
        self.dispatched_cards: List[DiscordEmbedCard] = []

    def is_pytest_environment(self) -> bool:
        return self.suppress_in_test or ("PYTEST_CURRENT_TEST" in os.environ)

    def _apply_rate_limit(self) -> float:
        now = time.time()
        elapsed = now - self.last_post_time
        sleep_needed = max(0.0, self.rate_limit_interval_s - elapsed)
        if sleep_needed > 0 and not self.is_pytest_environment():
            time.sleep(min(sleep_needed, self.max_backoff_sleep_s))
        self.last_post_time = time.time()
        return sleep_needed

    def post_broken_alert(
        self,
        component: str,
        error_message: str,
        evidence: str,
        dashboard_url: str,
    ) -> bool:
        """Dispatch institutional red embed card (0xE53935) with error evidence and dashboard link."""
        self._apply_rate_limit()
        # Truncate evidence if over 1024 characters
        safe_evidence = evidence[:1000] + "..." if len(evidence) > 1000 else evidence

        card = DiscordEmbedCard(
            color=DISCORD_COLOR_BROKEN,
            title=f"🚨 [BROKEN] System Failure: {component}",
            description=f"**Error**: {error_message}",
            fields=[
                {"name": "Component", "value": component, "inline": True},
                {"name": "Evidence", "value": f"```{safe_evidence}```", "inline": False},
                {"name": "Operator Dashboard", "value": f"[Open Dashboard]({dashboard_url})", "inline": True},
            ],
            url=dashboard_url,
        )
        self.dispatched_cards.append(card)
        if self.is_pytest_environment():
            return True
        return True

    def post_recovered_alert(
        self,
        component: str,
        downtime_duration_s: float,
        status_info: str,
        dashboard_url: str,
    ) -> bool:
        """Dispatch institutional green embed card (0x43A047) with recovery duration and telemetry."""
        self._apply_rate_limit()
        safe_duration = max(0.0, downtime_duration_s)
        duration_fmt = f"{safe_duration:.1f}s" if safe_duration < 60 else f"{safe_duration/60:.1f}m"

        card = DiscordEmbedCard(
            color=DISCORD_COLOR_RECOVERED,
            title=f"✅ [RECOVERED] System Operational: {component}",
            description=f"Service successfully restored after {duration_fmt} downtime.",
            fields=[
                {"name": "Component", "value": component, "inline": True},
                {"name": "Downtime Duration", "value": duration_fmt, "inline": True},
                {"name": "Telemetry Status", "value": status_info, "inline": False},
                {"name": "Operator Dashboard", "value": f"[Open Dashboard]({dashboard_url})", "inline": True},
            ],
            url=dashboard_url,
        )
        self.dispatched_cards.append(card)
        if self.is_pytest_environment():
            return True
        return True

    def post_trade_execution(
        self,
        orders: List[RebalanceOrder],
        nav: float,
        regime: str,
        dashboard_url: str,
    ) -> bool:
        """Dispatch institutional blue embed card (0x1E88E5) on rebalance execution."""
        if not orders:
            raise ValueError("Orders list cannot be empty for trade execution notification")
        self._apply_rate_limit()

        order_lines = [
            f"`{o.action.upper()}` **{o.symbol}**: {o.shares:.2f} shs @ ${o.price:.2f}"
            for o in orders[:10]
        ]
        if len(orders) > 10:
            order_lines.append(f"... and {len(orders) - 10} more orders")

        card = DiscordEmbedCard(
            color=DISCORD_COLOR_TRADE,
            title=f"⚖️ [TRADE EXECUTION] Portfolio Rebalance ({regime})",
            description=f"Executed {len(orders)} rebalance orders at NAV **${nav:,.2f}**.",
            fields=[
                {"name": "Regime", "value": regime, "inline": True},
                {"name": "Portfolio NAV", "value": f"${nav:,.2f}", "inline": True},
                {"name": "Orders Detail", "value": "\n".join(order_lines), "inline": False},
                {"name": "Operator Dashboard", "value": f"[Open Dashboard]({dashboard_url})", "inline": True},
            ],
            url=dashboard_url,
        )
        self.dispatched_cards.append(card)
        if self.is_pytest_environment():
            return True
        return True


# ==============================================================================
# 5. Mobile Operator Web App & REST/SSE Controls (Features 10, 11, 12, 14, 20)
# ==============================================================================

HTML_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="h-full bg-slate-50">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynamic Long-Term Strategy Bot - Operator Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .touch-btn { min-height: 44px; min-width: 44px; display: inline-flex; align-items: center; justify-content: center; }
        .card-shadow { box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1); }
    </style>
</head>
<body class="h-full text-slate-800 antialiased p-4 max-w-lg mx-auto bg-slate-50">
    <header class="mb-4 flex items-center justify-between">
        <div>
            <h1 class="text-xl font-bold tracking-tight text-slate-900">Operator Dashboard</h1>
            <p class="text-xs text-slate-500">Institutional Strategy Daemon</p>
        </div>
        <div id="status-badge" class="px-2.5 py-1 text-xs font-semibold rounded-full bg-emerald-100 text-emerald-800">
            RUNNING
        </div>
    </header>

    <!-- Disconnect Warning Banner (Feature 13) -->
    <div id="alert-banner" role="alert" class="hidden mb-4 p-3 rounded-lg border border-amber-300 bg-amber-50 text-amber-900 text-sm">
        <div class="flex items-center font-medium">
            <span class="mr-2">⚠️</span>
            <span>AlpacaRelay Disconnected — Operating in Synthetic Fallback</span>
        </div>
    </div>

    <!-- Portfolio NAV Card (Feature 10, 12) -->
    <section class="card-shadow bg-white rounded-xl p-4 mb-4 border border-slate-200">
        <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Total NAV</span>
        <div class="text-3xl font-extrabold text-slate-900 mt-1" id="portfolio-nav">$50,000.00</div>
        <div class="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-slate-100 text-sm">
            <div>
                <span class="text-slate-400 text-xs">Cash Balance:</span>
                <span class="font-semibold block text-slate-700" id="portfolio-cash">$50,000.00</span>
            </div>
            <div>
                <span class="text-slate-400 text-xs">Realized P&L:</span>
                <span class="font-semibold block text-emerald-600" id="portfolio-pnl">$0.00</span>
            </div>
        </div>
    </section>

    <!-- Operator Real-Time Controls (Feature 11, 14) -->
    <section class="card-shadow bg-white rounded-xl p-4 mb-4 border border-slate-200">
        <h2 class="text-sm font-semibold text-slate-700 mb-3">Operator Controls</h2>
        <div class="grid grid-cols-3 gap-2">
            <button id="btn-pause" class="touch-btn bg-amber-500 hover:bg-amber-600 text-white font-medium text-xs rounded-lg px-3 py-2.5">
                Pause
            </button>
            <button id="btn-resume" class="touch-btn bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-xs rounded-lg px-3 py-2.5">
                Resume
            </button>
            <button id="btn-rebalance" class="touch-btn bg-blue-600 hover:bg-blue-700 text-white font-medium text-xs rounded-lg px-3 py-2.5">
                Rebalance
            </button>
        </div>
    </section>

    <!-- Active Positions Section (Feature 12) -->
    <section class="card-shadow bg-white rounded-xl p-4 border border-slate-200">
        <h2 class="text-sm font-semibold text-slate-700 mb-2">Active Positions</h2>
        <div class="overflow-x-auto">
            <table class="w-full text-left text-xs text-slate-600" id="positions-table">
                <thead>
                    <tr class="border-b border-slate-200 text-slate-400">
                        <th class="py-1.5">Symbol</th>
                        <th class="py-1.5">Qty</th>
                        <th class="py-1.5">Price</th>
                        <th class="py-1.5 text-right">Value</th>
                    </tr>
                </thead>
                <tbody id="positions-body">
                    <tr><td colspan="4" class="py-3 text-center text-slate-400">No open positions ($50k cash pristine)</td></tr>
                </tbody>
            </table>
        </div>
    </section>
</body>
</html>
"""


class OperatorAppContract:
    """Mock/Reference ASGI-compatible web handler for FastAPI testing."""

    def __init__(
        self,
        paper_account: Optional[PaperAccountManagerContract] = None,
        feed_manager: Optional[DataFeedManagerContract] = None,
        daemon_state: str = "RUNNING",
    ):
        self.paper_account = paper_account or PaperAccountManagerContract(":memory:")
        self.feed_manager = feed_manager or DataFeedManagerContract()
        self.daemon_state = daemon_state

    def handle_request(self, method: str, path: str, body: Optional[dict] = None) -> Tuple[int, Dict[str, str], Any]:
        """Synchronous HTTP dispatch for test evaluation."""
        method = method.upper()

        if path == "/" and method == "GET":
            return (
                200,
                {"Content-Type": "text/html; charset=utf-8"},
                HTML_DASHBOARD_TEMPLATE,
            )

        if path == "/health" and method == "GET":
            nav = self.paper_account.get_portfolio_state().total_nav
            conn = self.feed_manager.get_connection_status()
            payload = {
                "status": "ok",
                "service": "DynamicLongTermStrategyBot",
                "relay": {
                    "is_connected": conn.is_connected,
                    "feed_source": conn.feed_source,
                    "alert_banner_active": conn.alert_banner_active,
                },
                "portfolio": {"nav": nav},
                "state": self.daemon_state,
            }
            return (200, {"Content-Type": "application/json"}, json.dumps(payload))

        if path == "/api/portfolio" and method == "GET":
            summary = self.paper_account.get_portfolio_state()
            conn = self.feed_manager.get_connection_status()
            if hasattr(summary, "model_dump"):
                summary_dict = summary.model_dump()
            elif hasattr(summary, "to_dict"):
                summary_dict = summary.to_dict()
            else:
                summary_dict = asdict(summary)
            if hasattr(conn, "model_dump"):
                conn_dict = conn.model_dump()
            elif hasattr(conn, "to_dict"):
                conn_dict = conn.to_dict()
            else:
                conn_dict = asdict(conn)
            payload = {
                "portfolio": summary_dict,
                "connection": conn_dict,
                "daemon_state": self.daemon_state,
                "regime": "BULL_NORMAL",
            }
            return (200, {"Content-Type": "application/json"}, json.dumps(payload))

        if path == "/api/operator/pause" and method == "POST":
            self.daemon_state = "PAUSED"
            return (200, {"Content-Type": "application/json"}, json.dumps({"status": "ok", "state": "PAUSED"}))

        if path == "/api/operator/resume" and method == "POST":
            self.daemon_state = "RUNNING"
            return (200, {"Content-Type": "application/json"}, json.dumps({"status": "ok", "state": "RUNNING"}))

        if path == "/api/operator/rebalance" and method == "POST":
            # Immediate out-of-cadence rebalance evaluation
            return (200, {"Content-Type": "application/json"}, json.dumps({"status": "ok", "action": "rebalanced"}))

        return (404, {"Content-Type": "application/json"}, json.dumps({"error": "Not Found"}))


# ==============================================================================
# 6. Progressive Resolution Helpers
# ==============================================================================

def resolve_paper_account_cls():
    """Returns canonical reference paper account manager contract."""
    return PaperAccountManagerContract


def resolve_feed_manager_cls():
    """Returns canonical reference feed manager contract."""
    return DataFeedManagerContract


def resolve_discord_notifier_cls():
    """Returns canonical reference discord notifier contract."""
    try:
        from bot.discord_alerts import DiscordNotifier
        return DiscordNotifier
    except ImportError:
        return DiscordNotifierContract
