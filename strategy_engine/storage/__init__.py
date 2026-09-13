"""
strategy_engine.storage
~~~~~~~~~~~~~~~~~~~~~~~

Dual-persistence engine for the AlpacaRelay Strategy Engine:
SQLite with WAL mode and daily append-only JSONL decision audit logger.
"""

from strategy_engine.storage.audit_logger import AuditLogEntry, JSONLAuditLogger
from strategy_engine.storage.database import Database
from strategy_engine.storage.repositories import (
    AllocationRepository,
    BaseRepository,
    MarketBarRepository,
    OrderRepository,
    PortfolioRepository,
    PortfolioStateRepository,
    RebalanceOrderRepository,
    RegimeEventRepository,
    SignalRepository,
    SignalSnapshotRepository,
    StorageService,
)

__all__ = [
    "Database",
    "BaseRepository",
    "SignalSnapshotRepository",
    "SignalRepository",
    "AllocationRepository",
    "RebalanceOrderRepository",
    "OrderRepository",
    "PortfolioStateRepository",
    "PortfolioRepository",
    "RegimeEventRepository",
    "MarketBarRepository",
    "StorageService",
    "JSONLAuditLogger",
    "AuditLogEntry",
]
