"""
strategy_engine.storage.audit_logger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Daily append-only structured JSONL audit logger for strategy rebalance decisions.
Partitioned by UTC date (decisions_YYYY-MM-DD.jsonl) with thread-safe flushing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Union
import uuid

from strategy_engine.core.models import (
    MarketRegime,
    OrderIntent,
    SignalSnapshot,
    TargetAllocation,
)

logger = logging.getLogger("strategy_engine.storage.audit_logger")


@dataclass
class AuditLogEntry:
    """Structured container for decision audit record."""
    timestamp: str
    trigger: str
    regime: str
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    version: str = "1.0"
    signals: Dict[str, Any] = field(default_factory=dict)
    portfolio: Dict[str, Any] = field(default_factory=dict)
    allocations: Dict[str, Any] = field(default_factory=dict)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JSONLAuditLogger:
    """Thread-safe append-only daily JSONL decision audit logger."""

    def __init__(self, log_dir: Union[str, Path] = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_log_path_for_date(self, d: Union[date, str]) -> Path:
        """Derive filepath decisions_YYYY-MM-DD.jsonl for given date."""
        if isinstance(d, str):
            date_str = d[:10]
        elif isinstance(d, datetime):
            date_str = d.astimezone(timezone.utc).strftime("%Y-%m-%d")
        elif isinstance(d, date):
            date_str = d.strftime("%Y-%m-%d")
        else:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"decisions_{date_str}.jsonl"

    def log_rebalance_decision(
        self,
        trigger: str,
        regime: Union[MarketRegime, str],
        timestamp: Optional[datetime] = None,
        signals: Optional[Union[SignalSnapshot, Dict[str, Any]]] = None,
        portfolio: Optional[Dict[str, Any]] = None,
        allocations: Optional[Union[TargetAllocation, Dict[str, Any]]] = None,
        orders: Optional[List[Union[OrderIntent, Dict[str, Any]]]] = None,
        rationale: str = "",
        decision_id: Optional[str] = None,
    ) -> str:
        """Record rebalance decision event to daily JSONL file."""
        now_dt = timestamp or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        ts_str = now_dt.astimezone(timezone.utc).isoformat()
        did = decision_id or uuid.uuid4().hex

        # Format regime
        regime_str = regime.value if isinstance(regime, MarketRegime) else str(regime)

        # Format signals
        signals_dict: Dict[str, Any] = {}
        if isinstance(signals, SignalSnapshot):
            signals_dict = {
                "spy_price": signals.spy_price,
                "spy_sma50": signals.spy_sma50,
                "spy_sma200": signals.spy_sma200,
                "realized_vol_20d": signals.realized_vol_20d,
                "vol_scale_factor": signals.vol_scale_factor,
                "drawdown_pct": signals.drawdown_pct,
                "circuit_breaker_active": signals.circuit_breaker_active,
                "indicators": signals.indicators,
            }
        elif isinstance(signals, dict):
            signals_dict = signals

        # Format portfolio
        portfolio_dict = portfolio or {}

        # Format allocations
        allocations_dict: Dict[str, Any] = {}
        if isinstance(allocations, TargetAllocation):
            allocations_dict = {
                "weights": allocations.weights,
                "cash_weight": allocations.cash_weight,
                "rationale": allocations.rationale,
            }
        elif isinstance(allocations, dict):
            allocations_dict = allocations

        # Format orders
        orders_list: List[Dict[str, Any]] = []
        if orders:
            for o in orders:
                if isinstance(o, OrderIntent):
                    orders_list.append(o.model_dump(mode="json"))
                elif isinstance(o, dict):
                    orders_list.append(o)

        record = {
            "version": "1.0",
            "decision_id": did,
            "timestamp": ts_str,
            "trigger": trigger,
            "regime": regime_str,
            "signals": signals_dict,
            "portfolio": portfolio_dict,
            "allocations": allocations_dict,
            "orders": orders_list,
            "rationale": rationale or allocations_dict.get("rationale", ""),
        }

        log_path = self._get_log_path_for_date(now_dt)
        line = json.dumps(record, sort_keys=True) + "\n"

        with self._lock:
            # Guarantee clean line boundary if previous write was unflushed/truncated
            if log_path.exists() and log_path.stat().st_size > 0:
                try:
                    with open(log_path, "rb+") as bf:
                        bf.seek(-1, os.SEEK_END)
                        last_byte = bf.read(1)
                        if last_byte != b"\n":
                            bf.write(b"\n")
                            bf.flush()
                            try:
                                os.fsync(bf.fileno())
                            except OSError:
                                pass
                except OSError as e:
                    logger.warning("Failed checking trailing newline on %s: %s", log_path, e)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

        return did

    def read_date(self, d: Union[date, str, datetime]) -> List[Dict[str, Any]]:
        """Read all valid JSON records for a given date, skipping corrupted lines."""
        log_path = self._get_log_path_for_date(d)
        if not log_path.exists():
            return []

        records: List[Dict[str, Any]] = []
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f, 1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    parsed = json.loads(clean)
                    if isinstance(parsed, dict):
                        records.append(parsed)
                except Exception as e:
                    logger.warning("Skipping corrupted line %d in %s: %s", line_idx, log_path.name, e)
                    continue

        return records

    def read_decisions(self, d: Union[date, str, datetime]) -> List[Dict[str, Any]]:
        """Alias for read_date to satisfy API specification contract."""
        return self.read_date(d)

    def read_latest(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Read latest decisions across all daily files."""
        log_files = sorted(self.log_dir.glob("decisions_*.jsonl"), reverse=True)
        results: List[Dict[str, Any]] = []
        for lf in log_files:
            file_records = []
            with open(lf, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    clean = line.strip()
                    if not clean:
                        continue
                    try:
                        parsed = json.loads(clean)
                        if isinstance(parsed, dict):
                            file_records.append(parsed)
                    except Exception:
                        continue
            # Most recent lines in file are at bottom
            results.extend(reversed(file_records))
            if len(results) >= limit:
                break
        # Sort recovered records explicitly by ISO timestamp descending
        results.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)
        return results[:limit]

    def find_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Search across log files for specific decision_id."""
        log_files = sorted(self.log_dir.glob("decisions_*.jsonl"), reverse=True)
        for lf in log_files:
            with open(lf, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    clean = line.strip()
                    if not clean:
                        continue
                    try:
                        parsed = json.loads(clean)
                        if parsed.get("decision_id") == decision_id:
                            return parsed
                    except Exception:
                        continue
        return None
