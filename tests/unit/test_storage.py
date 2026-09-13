"""
tests.unit.test_storage
~~~~~~~~~~~~~~~~~~~~~~~

Unit test suite for SQLite WAL persistence, point-in-time repositories,
and daily JSONL audit logging.
"""

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import threading
import pytest

from strategy_engine.core.models import (
    Bar,
    MarketRegime,
    OrderIntent,
    OrderSide,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.storage.audit_logger import JSONLAuditLogger
from strategy_engine.storage.database import Database
from strategy_engine.storage.repositories import (
    AllocationRepository,
    MarketBarRepository,
    PortfolioStateRepository,
    RebalanceOrderRepository,
    RegimeEventRepository,
    SignalSnapshotRepository,
    StorageService,
)


def test_database_wal_and_pragmas(tmp_path):
    db_file = tmp_path / "test_wal.db"
    db = Database(db_file)
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode;")
    journal_mode = cursor.fetchone()[0]
    assert journal_mode.upper() == "WAL"

    cursor.execute("PRAGMA synchronous;")
    sync_mode = cursor.fetchone()[0]
    # In SQLite, NORMAL is 1
    assert sync_mode in (1, "1", "NORMAL")

    cursor.execute("PRAGMA busy_timeout;")
    timeout = cursor.fetchone()[0]
    assert timeout == 5000

    cursor.execute("PRAGMA foreign_keys;")
    fk = cursor.fetchone()[0]
    assert fk in (1, "1", "ON")

    cursor.close()
    conn.close()
    db.close()


def test_database_in_memory_persistence():
    db = Database(":memory:")
    with db.transaction() as conn:
        conn.execute("CREATE TABLE test_mem (id INT, val TEXT);")
        conn.execute("INSERT INTO test_mem VALUES (1, 'alpha');")

    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT val FROM test_mem WHERE id = 1;")
        row = cursor.fetchone()
        assert row is not None
        assert row["val"] == "alpha"
    db.close()


def test_signal_snapshot_crud_and_point_in_time():
    db = Database(":memory:")
    repo = SignalSnapshotRepository(db)

    t1 = datetime(2026, 9, 1, 15, 50, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 2, 15, 50, tzinfo=timezone.utc)
    t3 = datetime(2026, 9, 3, 15, 50, tzinfo=timezone.utc)

    snap1 = SignalSnapshot(
        timestamp=t1,
        spy_price=500.0,
        spy_sma50=495.0,
        spy_sma200=480.0,
        realized_vol_20d=0.10,
        vol_scale_factor=1.0,
        drawdown_pct=-0.01,
        regime=MarketRegime.BULL_AGGRESSIVE,
        indicators={"qqq_price": 420.0, "drawdown_gate": 1.0},
    )
    snap2 = SignalSnapshot(
        timestamp=t2,
        spy_price=505.0,
        spy_sma50=496.0,
        spy_sma200=481.0,
        realized_vol_20d=0.11,
        vol_scale_factor=1.0,
        drawdown_pct=-0.005,
        regime=MarketRegime.BULL_NORMAL,
        indicators={"qqq_price": 425.0, "drawdown_gate": 1.0},
    )
    snap3 = SignalSnapshot(
        timestamp=t3,
        spy_price=490.0,
        spy_sma50=496.0,
        spy_sma200=482.0,
        realized_vol_20d=0.25,
        vol_scale_factor=0.48,
        drawdown_pct=-0.06,
        regime=MarketRegime.CORRECTION_FRAGILE,
        indicators={"qqq_price": 410.0, "drawdown_gate": 0.5},
    )

    repo.save(snap1, rationale="Day 1 bull")
    repo.save(snap2, rationale="Day 2 bull")
    repo.save(snap3, rationale="Day 3 pullback")

    latest = repo.get_latest()
    assert latest is not None
    assert latest["regime"] == MarketRegime.CORRECTION_FRAGILE.value
    assert latest["spy_price"] == 490.0

    # Point-in-time: query as of between t1 and t2
    as_of_mid = datetime(2026, 9, 1, 20, 0, tzinfo=timezone.utc)
    res_pit = repo.get_as_of(as_of_mid)
    assert res_pit is not None
    assert res_pit["spy_price"] == 500.0
    assert res_pit["regime"] == MarketRegime.BULL_AGGRESSIVE.value

    # Point-in-time before any records
    res_before = repo.get_as_of(datetime(2026, 8, 30, tzinfo=timezone.utc))
    assert res_before is None

    # Model reconstruction
    model = repo.to_model(res_pit)
    assert isinstance(model, SignalSnapshot)
    assert model.spy_price == 500.0
    assert model.regime == MarketRegime.BULL_AGGRESSIVE
    db.close()


def test_allocations_crud_and_to_model():
    db = Database(":memory:")
    repo = AllocationRepository(db)

    t1 = datetime(2026, 9, 1, 15, 50, tzinfo=timezone.utc)
    alloc = TargetAllocation(
        timestamp=t1,
        regime=MarketRegime.BULL_AGGRESSIVE,
        weights={"QQQ": 0.5, "XLK": 0.3, "SPY": 0.2},
        cash_weight=0.0,
        rationale="Aggressive tech weighting",
    )
    repo.save(alloc, risk_multiplier=1.0)

    latest = repo.get_latest()
    assert latest is not None
    assert latest["regime"] == "BULL_AGGRESSIVE"
    assert latest["weights"]["QQQ"] == 0.5
    assert latest["risk_multiplier"] == 1.0

    reconstructed = repo.to_model(latest)
    assert isinstance(reconstructed, TargetAllocation)
    assert reconstructed.weights["QQQ"] == 0.5
    assert reconstructed.cash_weight == 0.0

    # Test point-in-time
    pit = repo.get_as_of(t1 + timedelta(hours=1))
    assert pit is not None
    assert pit["weights"]["XLK"] == 0.3
    db.close()


def test_rebalance_orders_crud_and_status():
    db = Database(":memory:")
    repo = RebalanceOrderRepository(db)

    t1 = datetime(2026, 9, 1, 15, 50, tzinfo=timezone.utc)
    order1 = OrderIntent(
        symbol="QQQ",
        action="BUY",
        side=OrderSide.BUY,
        target_weight=0.50,
        current_weight=0.30,
        delta_weight=0.20,
        delta_shares=25.0,
        delta_dollars=10000.0,
        estimated_price=400.0,
        notional=10000.0,
        timestamp=t1,
        rationale="Increase tech",
    )
    order2 = OrderIntent(
        symbol="SHV",
        action="SELL",
        side=OrderSide.SELL,
        target_weight=0.0,
        current_weight=0.20,
        delta_weight=-0.20,
        delta_shares=-100.0,
        delta_dollars=-10000.0,
        estimated_price=100.0,
        notional=10000.0,
        timestamp=t1,
        rationale="Deploy cash",
    )

    ids = repo.save_batch([order1, order2], status="PENDING")
    assert len(ids) == 2

    # Query pending orders
    pending = repo.get_by_status("PENDING")
    assert len(pending) == 2

    # Update status of first order
    updated = repo.update_status(ids[0], "FILLED")
    assert updated is True

    fetched1 = repo.get_by_id(ids[0])
    assert fetched1 is not None
    assert fetched1["status"] == "FILLED"
    assert fetched1["symbol"] == "QQQ"

    by_symbol = repo.get_by_symbol("SHV")
    assert len(by_symbol) == 1
    assert by_symbol[0]["side"] == "SELL"
    db.close()


def test_portfolio_states_crud_and_point_in_time():
    db = Database(":memory:")
    repo = PortfolioStateRepository(db)

    t1 = datetime(2026, 9, 1, 16, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 2, 16, 0, tzinfo=timezone.utc)

    repo.save(
        timestamp=t1,
        cash=10000.0,
        equity=90000.0,
        total_nav=100000.0,
        positions={"SPY": 100, "QQQ": 50},
    )
    repo.save(
        timestamp=t2,
        cash=5000.0,
        equity=97000.0,
        total_nav=102000.0,
        positions={"SPY": 110, "QQQ": 55},
    )

    latest = repo.get_latest()
    assert latest is not None
    assert latest["total_nav"] == 102000.0
    assert latest["positions"]["SPY"] == 110

    pit = repo.get_as_of(t1 + timedelta(hours=2))
    assert pit is not None
    assert pit["total_nav"] == 100000.0
    db.close()


def test_regime_events_deduplication():
    db = Database(":memory:")
    repo = RegimeEventRepository(db)

    t1 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc)

    ev1 = repo.record_transition_if_changed(t1, MarketRegime.BULL_NORMAL, "Initial state")
    assert ev1 is not None

    # Same regime -> no duplicate event
    ev2 = repo.record_transition_if_changed(t2, MarketRegime.BULL_NORMAL, "Still normal bull")
    assert ev2 is None

    # Changed regime -> new event
    ev3 = repo.record_transition_if_changed(t3, MarketRegime.CORRECTION_FRAGILE, "Vol spiked")
    assert ev3 is not None

    events = repo.get_events()
    assert len(events) == 2
    assert events[1]["new_regime"] == MarketRegime.CORRECTION_FRAGILE.value
    db.close()


def test_market_bars_crud_and_upsert():
    db = Database(":memory:")
    repo = MarketBarRepository(db)

    t1 = datetime(2026, 9, 1, 4, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 2, 4, 0, tzinfo=timezone.utc)

    b1 = Bar(symbol="SPY", timestamp=t1, open=500.0, high=505.0, low=498.0, close=503.0, volume=10000)
    b2 = Bar(symbol="SPY", timestamp=t2, open=503.0, high=510.0, low=501.0, close=508.0, volume=12000)
    b3 = Bar(symbol="QQQ", timestamp=t1, open=450.0, high=455.0, low=448.0, close=452.0, volume=8000)

    count = repo.save_bars([b1, b2, b3])
    assert count == 3

    symbols = repo.get_symbols()
    assert "QQQ" in symbols and "SPY" in symbols

    spy_bars = repo.get_bars("SPY")
    assert len(spy_bars) == 2
    assert spy_bars[1].close == 508.0

    latest_spy = repo.get_latest_bar("SPY")
    assert latest_spy is not None
    assert latest_spy.timestamp == t2
    assert latest_spy.close == 508.0

    # Upsert test
    b2_revised = Bar(symbol="SPY", timestamp=t2, open=503.0, high=512.0, low=501.0, close=511.0, volume=12500)
    repo.save_bar(b2_revised)
    latest_revised = repo.get_latest_bar("SPY")
    assert latest_revised.close == 511.0
    db.close()


def test_storage_service_unified_facade():
    service = StorageService(":memory:")
    t = datetime(2026, 9, 3, 15, 50, tzinfo=timezone.utc)
    alloc_id = service.record_decision_audit(
        trigger="WEEKLY_REBALANCE",
        regime=MarketRegime.BULL_AGGRESSIVE,
        status="EXECUTED",
        rationale="Rotated to top momentum",
        timestamp=t,
        weights={"QQQ": 0.5, "SPY": 0.5},
    )
    assert alloc_id > 0
    latest = service.allocations.get_latest()
    assert latest["regime"] == "BULL_AGGRESSIVE"
    service.db.close()


def test_audit_logger_daily_partitioning_and_corrupted_lines(tmp_path):
    logger = JSONLAuditLogger(log_dir=tmp_path)
    t1 = datetime(2026, 9, 1, 15, 50, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 2, 15, 50, tzinfo=timezone.utc)

    did1 = logger.log_rebalance_decision(
        trigger="DAILY_CLOSE",
        regime=MarketRegime.BULL_AGGRESSIVE,
        timestamp=t1,
        rationale="Day 1 log",
    )
    did2 = logger.log_rebalance_decision(
        trigger="DAILY_CLOSE",
        regime=MarketRegime.BULL_NORMAL,
        timestamp=t2,
        rationale="Day 2 log",
    )

    file1 = tmp_path / "decisions_2026-09-01.jsonl"
    file2 = tmp_path / "decisions_2026-09-02.jsonl"
    assert file1.exists()
    assert file2.exists()

    # Corrupt file1 by injecting malformed line
    with open(file1, "a", encoding="utf-8") as f:
        f.write("{MALFORMED JSON LINE\n")

    # Resilient reader should read valid line and skip corrupted line
    records1 = logger.read_date(t1.date())
    assert len(records1) == 1
    assert records1[0]["decision_id"] == did1

    records2 = logger.read_date(t2.date())
    assert len(records2) == 1
    assert records2[0]["decision_id"] == did2

    # Find decision
    found = logger.find_decision(did1)
    assert found is not None
    assert found["regime"] == "BULL_AGGRESSIVE"


def test_concurrent_wal_readers_and_writers(tmp_path):
    db_path = tmp_path / "concurrent_wal.db"
    db = Database(db_path)
    repo = SignalSnapshotRepository(db)

    def writer_task():
        for i in range(20):
            t = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
            repo.save_raw(
                timestamp=t,
                regime="BULL_NORMAL",
                spy_price=500.0 + i,
                qqq_price=400.0 + i,
                vol_20d=0.12,
                vol_scale_factor=1.0,
                drawdown_gate=1.0,
            )

    def reader_task():
        for _ in range(30):
            repo.get_latest()

    t_writer = threading.Thread(target=writer_task)
    t_reader1 = threading.Thread(target=reader_task)
    t_reader2 = threading.Thread(target=reader_task)

    t_writer.start()
    t_reader1.start()
    t_reader2.start()

    t_writer.join()
    t_reader1.join()
    t_reader2.join()

    latest = repo.get_latest()
    assert latest is not None
    assert latest["spy_price"] == 519.0
    db.close()
