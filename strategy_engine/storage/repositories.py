"""
strategy_engine.storage.repositories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Point-in-time relational repositories for signals, allocations, orders,
portfolio state, regime shifts, and market bars.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Union
import uuid

from strategy_engine.core.models import (
    Bar,
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderType,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.storage.database import Database

logger = logging.getLogger("strategy_engine.storage.repositories")


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


class BaseRepository:
    """Base repository providing database execution helpers."""

    def __init__(self, db: Database):
        self.db = db

    def _row_to_dict(self, row: Any) -> Dict[str, Any]:
        """Convert sqlite3.Row to standard dictionary."""
        return dict(row) if row is not None else {}


class SignalSnapshotRepository(BaseRepository):
    """Repository for SignalSnapshot persistence and point-in-time querying."""

    def save(
        self,
        snapshot: SignalSnapshot,
        qqq_price: Optional[float] = None,
        drawdown_gate: Optional[float] = None,
        rationale: str = "",
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """Persist a SignalSnapshot model to SQLite."""
        q_p = qqq_price if qqq_price is not None else snapshot.indicators.get("qqq_price", snapshot.spy_price)
        dd_g = drawdown_gate if drawdown_gate is not None else snapshot.indicators.get("drawdown_gate", 1.0)
        atr_trig = 1 if snapshot.circuit_breaker_active or snapshot.indicators.get("atr_stop_triggered", 0.0) >= 1.0 else 0
        raw_j = snapshot.model_dump_json()

        query = """
        INSERT INTO signal_snapshots (
            timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
            drawdown_gate, atr_stop_triggered, rationale, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            _to_iso(snapshot.timestamp),
            snapshot.regime.value if isinstance(snapshot.regime, MarketRegime) else str(snapshot.regime),
            snapshot.spy_price,
            float(q_p),
            snapshot.realized_vol_20d,
            snapshot.vol_scale_factor,
            float(dd_g),
            atr_trig,
            rationale,
            raw_j,
        )
        if conn is not None:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid or 0

        with self.db.transaction() as txn_conn:
            cursor = txn_conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid or 0

    def save_raw(
        self,
        timestamp: datetime,
        regime: str,
        spy_price: float,
        qqq_price: float,
        vol_20d: float,
        vol_scale_factor: float,
        drawdown_gate: float,
        atr_stop_triggered: int = 0,
        rationale: str = "",
        raw_json: str = "{}",
    ) -> int:
        """Low-level insert for raw signal values."""
        query = """
        INSERT INTO signal_snapshots (
            timestamp, regime, spy_price, qqq_price, vol_20d, vol_scale_factor,
            drawdown_gate, atr_stop_triggered, rationale, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (
                    _to_iso(timestamp),
                    regime,
                    spy_price,
                    qqq_price,
                    vol_20d,
                    vol_scale_factor,
                    drawdown_gate,
                    atr_stop_triggered,
                    rationale,
                    raw_json,
                ),
            )
            return cursor.lastrowid or 0

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Query the most recent signal snapshot."""
        query = "SELECT * FROM signal_snapshots ORDER BY timestamp DESC LIMIT 1;"
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_as_of(self, as_of: datetime) -> Optional[Dict[str, Any]]:
        """Point-in-time query: returns latest record on or before as_of."""
        query = """
        SELECT * FROM signal_snapshots 
        WHERE timestamp <= ? 
        ORDER BY timestamp DESC LIMIT 1;
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (_to_iso(as_of),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query snapshots across an ISO timestamp range."""
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
        query = f"SELECT * FROM signal_snapshots {where_clause} ORDER BY timestamp ASC {limit_clause};"

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def to_model(self, record: Dict[str, Any]) -> SignalSnapshot:
        """Reconstruct SignalSnapshot domain model from database record."""
        raw_json = record.get("raw_json")
        if raw_json and raw_json != "{}":
            return SignalSnapshot.model_validate_json(raw_json)

        # Fallback reconstruction from column values
        regime = MarketRegime(record["regime"])
        ts = _from_iso(record["timestamp"])
        return SignalSnapshot(
            timestamp=ts,
            spy_price=float(record["spy_price"]),
            spy_sma50=float(record.get("spy_sma50", record["spy_price"])),
            spy_sma200=float(record.get("spy_sma200", record["spy_price"])),
            realized_vol_20d=float(record["vol_20d"]),
            vol_scale_factor=float(record["vol_scale_factor"]),
            drawdown_pct=float(record.get("drawdown_pct", 0.0)),
            circuit_breaker_active=bool(record.get("atr_stop_triggered", 0)),
            regime=regime,
            indicators={
                "qqq_price": float(record["qqq_price"]),
                "drawdown_gate": float(record["drawdown_gate"]),
            },
        )


class AllocationRepository(BaseRepository):
    """Repository for TargetAllocation persistence."""

    def save(
        self,
        allocation: TargetAllocation,
        risk_multiplier: float = 1.0,
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """Persist a TargetAllocation to SQLite."""
        weights_j = json.dumps(allocation.weights)
        query = """
        INSERT INTO allocations (timestamp, regime, weights_json, rationale, risk_multiplier)
        VALUES (?, ?, ?, ?, ?)
        """
        params = (
            _to_iso(allocation.timestamp),
            allocation.regime.value if isinstance(allocation.regime, MarketRegime) else str(allocation.regime),
            weights_j,
            allocation.rationale,
            float(risk_multiplier),
        )
        if conn is not None:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid or 0

        with self.db.transaction() as txn_conn:
            cursor = txn_conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid or 0

    def save_raw(
        self,
        timestamp: datetime,
        regime: str,
        weights: Dict[str, float],
        rationale: str = "",
        risk_multiplier: float = 1.0,
    ) -> int:
        """Low-level insert for raw allocation weights."""
        query = """
        INSERT INTO allocations (timestamp, regime, weights_json, rationale, risk_multiplier)
        VALUES (?, ?, ?, ?, ?)
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (
                    _to_iso(timestamp),
                    regime,
                    json.dumps(weights),
                    rationale,
                    float(risk_multiplier),
                ),
            )
            return cursor.lastrowid or 0

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Query the most recent allocation."""
        query = "SELECT * FROM allocations ORDER BY timestamp DESC LIMIT 1;"
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["weights"] = json.loads(res["weights_json"])
            return res

    def get_as_of(self, as_of: datetime) -> Optional[Dict[str, Any]]:
        """Point-in-time query: returns latest allocation on or before as_of."""
        query = """
        SELECT * FROM allocations 
        WHERE timestamp <= ? 
        ORDER BY timestamp DESC LIMIT 1;
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (_to_iso(as_of),))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["weights"] = json.loads(res["weights_json"])
            return res

    def get_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query allocations across timestamp range."""
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
        query = f"SELECT * FROM allocations {where_clause} ORDER BY timestamp ASC {limit_clause};"

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["weights"] = json.loads(d["weights_json"])
                results.append(d)
            return results

    def to_model(self, record: Dict[str, Any]) -> TargetAllocation:
        """Reconstruct TargetAllocation domain model."""
        weights = record.get("weights") or json.loads(record["weights_json"])
        cash_wt = sum(w for sym, w in weights.items() if sym in ("SHV", "BIL", "CASH"))
        return TargetAllocation(
            timestamp=_from_iso(record["timestamp"]),
            regime=MarketRegime(record["regime"]),
            weights=weights,
            cash_weight=cash_wt,
            rationale=record.get("rationale", ""),
        )


class RebalanceOrderRepository(BaseRepository):
    """Repository for RebalanceOrder manifests and status tracking."""

    def save(
        self,
        order: OrderIntent,
        status: str = "PENDING",
        order_id: Optional[str] = None,
    ) -> str:
        """Persist an OrderIntent to SQLite."""
        oid = order_id or uuid.uuid4().hex
        side_val = order.side.value if order.side else (order.action or "BUY")
        shares_val = abs(order.delta_shares) if order.delta_shares is not None else 0.0
        price_val = order.estimated_price if order.estimated_price is not None else 0.0
        notional_val = (
            order.notional
            if order.notional is not None
            else (abs(order.delta_dollars) if order.delta_dollars is not None else 0.0)
        )
        t_wt = order.target_weight if order.target_weight is not None else 0.0
        c_wt = order.current_weight if order.current_weight is not None else 0.0

        query = """
        INSERT INTO rebalance_orders (
            id, timestamp, symbol, side, shares, price, notional,
            target_weight, current_weight, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (
                    oid,
                    _to_iso(order.timestamp),
                    order.symbol,
                    side_val,
                    float(shares_val),
                    float(price_val),
                    float(notional_val),
                    float(t_wt),
                    float(c_wt),
                    status,
                ),
            )
            return oid

    def save_raw(
        self,
        id: str,
        timestamp: datetime,
        symbol: str,
        side: str,
        shares: float,
        price: float,
        notional: float,
        target_weight: float,
        current_weight: float,
        status: str = "PENDING",
    ) -> str:
        """Low-level insert for raw order values."""
        query = """
        INSERT INTO rebalance_orders (
            id, timestamp, symbol, side, shares, price, notional,
            target_weight, current_weight, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (
                    id,
                    _to_iso(timestamp),
                    symbol,
                    side,
                    shares,
                    price,
                    notional,
                    target_weight,
                    current_weight,
                    status,
                ),
            )
            return id

    def save_batch(
        self,
        orders: List[OrderIntent],
        status: str = "PENDING",
    ) -> List[str]:
        """Atomically persist a batch of OrderIntent instances."""
        if not orders:
            return []
        ids: List[str] = []
        rows = []
        for o in orders:
            oid = uuid.uuid4().hex
            ids.append(oid)
            side_val = o.side.value if o.side else (o.action or "BUY")
            shares_val = abs(o.delta_shares) if o.delta_shares is not None else 0.0
            price_val = o.estimated_price if o.estimated_price is not None else 0.0
            notional_val = (
                o.notional
                if o.notional is not None
                else (abs(o.delta_dollars) if o.delta_dollars is not None else 0.0)
            )
            t_wt = o.target_weight if o.target_weight is not None else 0.0
            c_wt = o.current_weight if o.current_weight is not None else 0.0
            rows.append(
                (
                    oid,
                    _to_iso(o.timestamp),
                    o.symbol,
                    side_val,
                    float(shares_val),
                    float(price_val),
                    float(notional_val),
                    float(t_wt),
                    float(c_wt),
                    status,
                )
            )

        query = """
        INSERT INTO rebalance_orders (
            id, timestamp, symbol, side, shares, price, notional,
            target_weight, current_weight, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, rows)
        return ids

    def update_status(self, order_id: str, new_status: str) -> bool:
        """Update order lifecycle status."""
        query = "UPDATE rebalance_orders SET status = ? WHERE id = ?;"
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (new_status, order_id))
            return cursor.rowcount > 0

    def get_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Query single order by ID."""
        query = "SELECT * FROM rebalance_orders WHERE id = ?;"
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (order_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Query orders filtered by status."""
        query = "SELECT * FROM rebalance_orders WHERE status = ? ORDER BY timestamp DESC;"
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (status,))
            return [dict(r) for r in cursor.fetchall()]

    def get_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Query orders for a symbol."""
        query = "SELECT * FROM rebalance_orders WHERE symbol = ? ORDER BY timestamp DESC;"
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (symbol,))
            return [dict(r) for r in cursor.fetchall()]

    def get_as_of(self, as_of: datetime) -> List[Dict[str, Any]]:
        """Point-in-time query: orders on or before as_of."""
        query = "SELECT * FROM rebalance_orders WHERE timestamp <= ? ORDER BY timestamp DESC;"
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (_to_iso(as_of),))
            return [dict(r) for r in cursor.fetchall()]

    def get_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query orders across timestamp range."""
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
        query = f"SELECT * FROM rebalance_orders {where_clause} ORDER BY timestamp ASC {limit_clause};"

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]


class PortfolioStateRepository(BaseRepository):
    """Repository for Portfolio State snapshots."""

    def save(
        self,
        timestamp: datetime,
        cash: float,
        equity: float,
        total_nav: float,
        positions: Union[Dict, List, str],
    ) -> str:
        """Persist portfolio NAV and position snapshot."""
        pos_str = positions if isinstance(positions, str) else json.dumps(positions)
        iso_ts = _to_iso(timestamp)
        query = """
        INSERT OR REPLACE INTO portfolio_states (timestamp, cash, equity, total_nav, positions_json)
        VALUES (?, ?, ?, ?, ?)
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (iso_ts, float(cash), float(equity), float(total_nav), pos_str))
            return iso_ts

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Query latest portfolio snapshot."""
        query = "SELECT * FROM portfolio_states ORDER BY timestamp DESC LIMIT 1;"
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["positions"] = json.loads(res["positions_json"])
            return res

    def get_as_of(self, as_of: datetime) -> Optional[Dict[str, Any]]:
        """Point-in-time query: portfolio state on or before as_of."""
        query = """
        SELECT * FROM portfolio_states 
        WHERE timestamp <= ? 
        ORDER BY timestamp DESC LIMIT 1;
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (_to_iso(as_of),))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["positions"] = json.loads(res["positions_json"])
            return res

    def get_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Query portfolio states across timestamp range."""
        conditions = []
        params: List[Any] = []
        if start:
            conditions.append("timestamp >= ?")
            params.append(_to_iso(start))
        if end:
            conditions.append("timestamp <= ?")
            params.append(_to_iso(end))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM portfolio_states {where_clause} ORDER BY timestamp ASC;"

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


class RegimeEventRepository(BaseRepository):
    """Repository for tracking discrete MarketRegime transition events."""

    def record_event(
        self,
        timestamp: datetime,
        old_regime: Union[MarketRegime, str],
        new_regime: Union[MarketRegime, str],
        trigger_reason: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """Record regime change event."""
        old_str = old_regime.value if isinstance(old_regime, MarketRegime) else str(old_regime)
        new_str = new_regime.value if isinstance(new_regime, MarketRegime) else str(new_regime)

        query = """
        INSERT INTO regime_events (timestamp, old_regime, new_regime, trigger_reason)
        VALUES (?, ?, ?, ?)
        """
        params = (_to_iso(timestamp), old_str, new_str, trigger_reason)
        if conn is not None:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid or 0

        with self.db.transaction() as txn_conn:
            cursor = txn_conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid or 0

    def record_transition_if_changed(
        self,
        timestamp: datetime,
        current_regime: Union[MarketRegime, str],
        trigger_reason: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Optional[int]:
        """Record event only if current_regime differs from latest known regime."""
        latest = self.get_latest(conn=conn)
        cur_str = current_regime.value if isinstance(current_regime, MarketRegime) else str(current_regime)

        if latest is not None and latest["new_regime"] == cur_str:
            return None  # No regime transition

        old_str = latest["new_regime"] if latest is not None else "UNKNOWN"
        return self.record_event(
            timestamp=timestamp,
            old_regime=old_str,
            new_regime=cur_str,
            trigger_reason=trigger_reason,
            conn=conn,
        )

    def get_latest(self, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
        """Query latest regime shift event."""
        query = "SELECT * FROM regime_events ORDER BY timestamp DESC, id DESC LIMIT 1;"
        if conn is not None:
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            return dict(row) if row else None

        with self.db.transaction() as txn_conn:
            cursor = txn_conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_as_of(self, as_of: datetime) -> Optional[Dict[str, Any]]:
        """Point-in-time query: latest regime event on or before as_of."""
        query = """
        SELECT * FROM regime_events 
        WHERE timestamp <= ? 
        ORDER BY timestamp DESC, id DESC LIMIT 1;
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (_to_iso(as_of),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_events(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query regime shift events across range."""
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
        query = f"SELECT * FROM regime_events {where_clause} ORDER BY timestamp ASC {limit_clause};"

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]


class MarketBarRepository(BaseRepository):
    """Repository for storing and caching OHLCV market bars."""

    def save_bar(self, bar: Bar, timeframe: str = "1Day") -> None:
        """Insert or replace an OHLCV bar."""
        query = """
        INSERT OR REPLACE INTO market_bars (
            symbol, timeframe, timestamp, open, high, low, close, volume, vwap, trade_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (
                    bar.symbol,
                    timeframe,
                    _to_iso(bar.timestamp),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    bar.vwap,
                    bar.trade_count,
                ),
            )

    def save_bars(self, bars: List[Bar], timeframe: str = "1Day") -> int:
        """Batch insert or replace bars."""
        if not bars:
            return 0
        rows = [
            (
                b.symbol,
                timeframe,
                _to_iso(b.timestamp),
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                b.vwap,
                b.trade_count,
            )
            for b in bars
        ]
        query = """
        INSERT OR REPLACE INTO market_bars (
            symbol, timeframe, timestamp, open, high, low, close, volume, vwap, trade_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, rows)
            return len(rows)

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Bar]:
        """Fetch chronologically ordered Bar models."""
        conditions = ["symbol = ?", "timeframe = ?"]
        params: List[Any] = [symbol, timeframe]
        if start:
            conditions.append("timestamp >= ?")
            params.append(_to_iso(start))
        if end:
            conditions.append("timestamp <= ?")
            params.append(_to_iso(end))

        where_clause = f"WHERE {' AND '.join(conditions)}"
        limit_clause = f"LIMIT {int(limit)}" if limit else ""
        query = f"SELECT * FROM market_bars {where_clause} ORDER BY timestamp ASC {limit_clause};"

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            bars: List[Bar] = []
            for r in rows:
                bars.append(
                    Bar(
                        symbol=r["symbol"],
                        timestamp=_from_iso(r["timestamp"]),
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                        volume=int(r["volume"]),
                        vwap=float(r["vwap"]) if r["vwap"] is not None else None,
                        trade_count=int(r["trade_count"]) if r["trade_count"] is not None else None,
                    )
                )
            return bars

    def get_latest_bar(self, symbol: str, timeframe: str = "1Day") -> Optional[Bar]:
        """Query latest bar for a symbol."""
        bars = self.get_bars(symbol=symbol, timeframe=timeframe, limit=1)
        # Note: to get latest we need DESC order
        query = """
        SELECT * FROM market_bars 
        WHERE symbol = ? AND timeframe = ? 
        ORDER BY timestamp DESC LIMIT 1;
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (symbol, timeframe))
            row = cursor.fetchone()
            if not row:
                return None
            return Bar(
                symbol=row["symbol"],
                timestamp=_from_iso(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                vwap=float(row["vwap"]) if row["vwap"] is not None else None,
                trade_count=int(row["trade_count"]) if row["trade_count"] is not None else None,
            )

    def get_symbols(self, timeframe: str = "1Day") -> List[str]:
        """List distinct symbols available in market_bars."""
        query = "SELECT DISTINCT symbol FROM market_bars WHERE timeframe = ? ORDER BY symbol ASC;"
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (timeframe,))
            return [r[0] for r in cursor.fetchall()]


# Ergonomic Aliases
SignalRepository = SignalSnapshotRepository
OrderRepository = RebalanceOrderRepository
PortfolioRepository = PortfolioStateRepository


class StorageService:
    """Unified container for all storage repositories and SQLite operations."""

    def __init__(self, db_path: Union[str, Database] = "strategy_engine.db"):
        if isinstance(db_path, Database):
            self.db = db_path
        else:
            self.db = Database(db_path)
        self.signals = SignalSnapshotRepository(self.db)
        self.allocations = AllocationRepository(self.db)
        self.orders = RebalanceOrderRepository(self.db)
        self.portfolio = PortfolioStateRepository(self.db)
        self.regimes = RegimeEventRepository(self.db)
        self.bars = MarketBarRepository(self.db)

    def record_decision_audit(
        self,
        trigger: str,
        regime: Union[MarketRegime, str],
        status: str,
        rationale: str = "",
        timestamp: Optional[datetime] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> int:
        """Record decision audit event across allocations and regime tables."""
        ts = timestamp or datetime.now(timezone.utc)
        regime_str = regime.value if isinstance(regime, MarketRegime) else str(regime)
        w = weights or {"SHV": 1.0}
        return self.allocations.save_raw(
            timestamp=ts,
            regime=regime_str,
            weights=w,
            rationale=f"[{trigger} - {status}] {rationale}",
            risk_multiplier=1.0,
        )

    def save_daily_close(
        self,
        signals: SignalSnapshot,
        allocation: TargetAllocation,
        timestamp: datetime,
        rationale: str = "",
    ) -> None:
        """Persist signals, allocations, and regime transitions atomically in a single transaction."""
        with self.db.transaction() as conn:
            self.signals.save(signals, rationale=rationale or allocation.rationale, conn=conn)
            self.allocations.save(allocation, conn=conn)
            self.regimes.record_transition_if_changed(
                timestamp=timestamp,
                current_regime=signals.regime,
                trigger_reason=rationale or allocation.rationale,
                conn=conn,
            )
