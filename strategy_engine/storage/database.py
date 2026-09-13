"""
strategy_engine.storage.database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SQLite database connection factory with WAL mode, schema initialization,
and concurrency-safe pragmas for the AlpacaRelay Strategy Engine.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Generator, Optional, Union

logger = logging.getLogger("strategy_engine.storage.database")

DEFAULT_DB_PATH = "strategy_engine.db"

SCHEMA_DDL = """
-- 1. Quantitative Signal Snapshots
CREATE TABLE IF NOT EXISTS signal_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    regime TEXT NOT NULL,
    spy_price REAL NOT NULL,
    qqq_price REAL NOT NULL,
    vol_20d REAL NOT NULL,
    vol_scale_factor REAL NOT NULL,
    drawdown_gate REAL NOT NULL,
    atr_stop_triggered INTEGER NOT NULL DEFAULT 0,
    rationale TEXT NOT NULL DEFAULT '',
    raw_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signal_snapshots(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_regime ON signal_snapshots(regime);

-- 2. Target Portfolio Allocations
CREATE TABLE IF NOT EXISTS allocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    regime TEXT NOT NULL,
    weights_json TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    risk_multiplier REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_allocations_timestamp ON allocations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_allocations_regime ON allocations(regime);

-- 3. Rebalance Orders Manifest
CREATE TABLE IF NOT EXISTS rebalance_orders (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    shares REAL NOT NULL,
    price REAL NOT NULL,
    notional REAL NOT NULL,
    target_weight REAL NOT NULL,
    current_weight REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING'
);
CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON rebalance_orders(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON rebalance_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON rebalance_orders(status);

-- 4. Portfolio State Snapshots
CREATE TABLE IF NOT EXISTS portfolio_states (
    timestamp TEXT PRIMARY KEY,
    cash REAL NOT NULL,
    equity REAL NOT NULL,
    total_nav REAL NOT NULL,
    positions_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_states(timestamp DESC);

-- 5. Regime Shift Events
CREATE TABLE IF NOT EXISTS regime_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    old_regime TEXT NOT NULL,
    new_regime TEXT NOT NULL,
    trigger_reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_regime_events_timestamp ON regime_events(timestamp DESC);

-- 6. Historical Market Bars Cache
CREATE TABLE IF NOT EXISTS market_bars (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    vwap REAL,
    trade_count INTEGER,
    PRIMARY KEY (symbol, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_bars_sym_tf_ts ON market_bars(symbol, timeframe, timestamp DESC);

-- 7. Compatibility Views
CREATE VIEW IF NOT EXISTS decision_audit_trail AS
SELECT 
    timestamp,
    regime,
    weights_json AS target_weights_json,
    rationale
FROM allocations;
"""


class Database:
    """SQLite Database manager with WAL mode and connection pooling."""

    def __init__(
        self,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        auto_init: bool = True,
    ):
        self.db_path = str(db_path)
        self.is_memory = self.db_path == ":memory:"
        self._in_memory_conn: Optional[sqlite3.Connection] = None
        self._mem_lock: Optional[threading.RLock] = threading.RLock() if self.is_memory else None

        if not self.is_memory:
            p = Path(self.db_path)
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)

        if self.is_memory:
            # Maintain persistent connection for in-memory database
            self._in_memory_conn = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
            self._in_memory_conn.row_factory = sqlite3.Row
            self._configure_pragmas(self._in_memory_conn)

        if auto_init:
            self.init_schema()

    @staticmethod
    def _configure_pragmas(conn: sqlite3.Connection) -> None:
        """Apply mandatory high-concurrency pragmas."""
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a configured connection."""
        if self.is_memory:
            assert self._in_memory_conn is not None
            return self._in_memory_conn

        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=5.0,
        )
        conn.row_factory = sqlite3.Row
        self._configure_pragmas(conn)
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding connection within an ACID transaction."""
        if self.is_memory:
            assert self._mem_lock is not None
            with self._mem_lock:
                conn = self.get_connection()
                try:
                    yield conn
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
        else:
            conn = self.get_connection()
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()

    def init_schema(self) -> None:
        """Execute DDL script to create tables, indexes, and views."""
        with self.transaction() as conn:
            conn.executescript(SCHEMA_DDL)
        logger.debug("Database schema initialized at %s", self.db_path)

    def reset_schema(self) -> None:
        """Drop existing tables and views and re-create schema."""
        drop_script = """
        DROP VIEW IF EXISTS decision_audit_trail;
        DROP TABLE IF EXISTS market_bars;
        DROP TABLE IF EXISTS regime_events;
        DROP TABLE IF EXISTS portfolio_states;
        DROP TABLE IF EXISTS rebalance_orders;
        DROP TABLE IF EXISTS allocations;
        DROP TABLE IF EXISTS signal_snapshots;
        """
        with self.transaction() as conn:
            conn.executescript(drop_script)
        self.init_schema()

    def close(self) -> None:
        """Close connection if in-memory."""
        if self.is_memory:
            if self._mem_lock is not None:
                with self._mem_lock:
                    if self._in_memory_conn is not None:
                        self._in_memory_conn.close()
                        self._in_memory_conn = None
            elif self._in_memory_conn is not None:
                self._in_memory_conn.close()
                self._in_memory_conn = None
