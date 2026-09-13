"""
tests.adversarial.test_m1_paper_account_stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Milestone 1 Empirical Stress and Adversarial Challenge Suite for PaperAccountManager.

Attacks and stress-tests:
1. Concurrency Stress (10+ threads on SQLite WAL):
   - Readers, price-updaters, and rebalance executors running simultaneously.
   - Zero database locks and zero corrupted records (cash & position accounting invariants).
   - Multi-threaded rebalancing on shared/identical symbols.
2. Non-Negative Cash Floor Invariant:
   - Oversized BUY orders exceeding cash.
   - Low/exhausted cash boundary conditions ($0, $5).
   - Fee-inclusive clamping (fee_per_share > 0, min_fee_per_order > 0).
   - Share round-up boundary conditions.
3. Strict SELL-before-BUY Sequencing:
   - Rebalance batches with $0 starting cash where BUYs precede SELLs in the input list.
   - Proceeds from SELLs must fund subsequent BUYs within the same rebalance transaction.
4. Sudden Crash Resilience & reset_to_pristine():
   - Abrupt process crash (os._exit) during persistence.
   - Pristine state restoration: exact $50,000.00 cash, 0 positions, 0 trades.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional
import pytest

from strategy_engine.core.models import OrderIntent, OrderSide
from bot.paper_account import (
    PaperAccountConfig,
    PaperAccountManager,
    PaperOrderSide,
    PaperTrade,
    PortfolioSummary,
    PositionDetail,
)


@pytest.fixture
def temp_wal_db() -> str:
    """Provide a real file-based SQLite WAL database path and cleanup afterwards."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for ext in ["", "-wal", "-shm"]:
        p = f"{path}{ext}"
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


# ==============================================================================
# 1. CONCURRENCY STRESS & WAL PERSISTENCE TESTS
# ==============================================================================


def test_concurrency_stress_readers_writers_wal(temp_wal_db: str):
    """
    Stress test with 12+ concurrent threads performing mixed operations on SQLite WAL:
    - 5 Portfolio reader threads
    - 3 Market price update threads
    - 4 Rebalancer threads executing BUY/SELL orders
    Assert zero unhandled database lock exceptions and strict accounting consistency.
    """
    mgr = PaperAccountManager(db=temp_wal_db)
    symbols = ["SPY", "QQQ", "TLT", "GLD", "SHV", "XLK", "XLE", "XLV"]
    errors: List[Exception] = []
    stop_event = threading.Event()

    def reader_worker():
        while not stop_event.is_set():
            try:
                st = mgr.get_portfolio_state()
                assert st.total_nav > 0
                time.sleep(0.002)
            except Exception as e:
                errors.append(e)

    def price_updater_worker():
        while not stop_event.is_set():
            try:
                prices = {s: random.uniform(50.0, 500.0) for s in symbols}
                mgr.update_market_prices(prices)
                time.sleep(0.003)
            except Exception as e:
                errors.append(e)

    def rebalancer_worker(worker_id: int):
        for step in range(8):
            if stop_event.is_set():
                break
            try:
                sym = symbols[worker_id % len(symbols)]
                side = OrderSide.BUY if step % 2 == 0 else OrderSide.SELL
                delta_shares = 1.0 if side == OrderSide.BUY else -1.0
                intent = OrderIntent(
                    symbol=sym,
                    action=side.value,
                    side=side,
                    delta_shares=delta_shares,
                    estimated_price=100.0,
                    notional=100.0,
                )
                mgr.execute_rebalance_orders([intent], {sym: 100.0})
                time.sleep(0.005)
            except Exception as e:
                errors.append(e)

    threads = []
    # 5 readers
    for _ in range(5):
        threads.append(threading.Thread(target=reader_worker))
    # 3 price updaters
    for _ in range(3):
        threads.append(threading.Thread(target=price_updater_worker))
    # 4 rebalancers
    for i in range(4):
        threads.append(threading.Thread(target=rebalancer_worker, args=(i,)))

    for t in threads:
        t.start()

    time.sleep(0.5)
    stop_event.set()

    for t in threads:
        t.join(timeout=5.0)

    # 1. Zero database lock errors or exceptions allowed
    assert len(errors) == 0, f"Encountered {len(errors)} concurrency errors: {errors[:5]}"

    # 2. Verify account state consistency
    st = mgr.get_portfolio_state()
    trades = mgr.get_trade_history(limit=1000)

    # Calculate expected cash from trade history
    net_bought = sum(t.notional for t in trades if t.side == "BUY")
    net_sold = sum(t.notional for t in trades if t.side == "SELL")
    net_fees = sum(t.fee for t in trades)
    expected_cash = round(50000.00 - net_bought + net_sold - net_fees, 2)

    # Final cash must match executed trade ledger with zero discrepancy
    assert round(st.cash, 2) == pytest.approx(expected_cash, abs=1e-2), (
        f"Cash corruption detected: Recorded cash ${st.cash:.2f} != Expected cash ${expected_cash:.2f} "
        f"(Discrepancy: ${st.cash - expected_cash:.2f})"
    )
    mgr.close()


def test_concurrent_same_symbol_rebalance_no_unique_constraint(temp_wal_db: str):
    """
    Stress test 10 concurrent threads executing BUY orders for the SAME unheld symbol.
    Verifies that simultaneous initialization of a position does not crash with
    sqlite3.IntegrityError: UNIQUE constraint failed: paper_positions.symbol.
    """
    mgr = PaperAccountManager(db=temp_wal_db)
    errors: List[Exception] = []

    def buy_worker(worker_id: int):
        try:
            intent = OrderIntent(
                symbol="NVDA",
                action="BUY",
                side=OrderSide.BUY,
                delta_shares=1.0,
                estimated_price=100.0,
                notional=100.0,
            )
            mgr.execute_rebalance_orders([intent], {"NVDA": 100.0})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=buy_worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    # All 10 threads must complete without UNIQUE constraint failures
    assert len(errors) == 0, f"Threads crashed on concurrent position insertion: {errors}"
    st = mgr.get_portfolio_state({"NVDA": 100.0})
    assert len(st.positions) == 1
    assert st.positions[0].symbol == "NVDA"
    assert st.positions[0].qty == pytest.approx(10.0, abs=1e-3)
    mgr.close()


# ==============================================================================
# 2. NON-NEGATIVE CASH FLOOR INVARIANT TESTS
# ==============================================================================


def test_cash_floor_invariant_oversized_buy(temp_wal_db: str):
    """
    Attempt to execute an extreme BUY order far exceeding total cash.
    Assert that cash never becomes negative, clamping preserves remaining cash.
    """
    mgr = PaperAccountManager(db=temp_wal_db)
    st_initial = mgr.get_portfolio_state()
    assert st_initial.cash == 50000.00

    # Request $1,000,000.00 purchase of SPY when account only has $50,000.00
    huge_order = OrderIntent(
        symbol="SPY",
        action="BUY",
        side=OrderSide.BUY,
        delta_shares=2000.0,
        estimated_price=500.0,
        notional=1000000.0,
    )
    report = mgr.execute_rebalance_orders([huge_order], {"SPY": 500.0})

    assert report.orders_filled == 1
    st_post = mgr.get_portfolio_state({"SPY": 500.0})

    # Invariant: Cash MUST be non-negative
    assert st_post.cash >= 0.00, f"Cash invariant breached: cash={st_post.cash}"
    assert st_post.total_nav == pytest.approx(50000.00, abs=1.0)
    mgr.close()


def test_cash_floor_invariant_exhausted_cash(temp_wal_db: str):
    """
    Deploy virtually all cash, then attempt additional BUY orders.
    Verify that subsequent orders are safely rejected and cash never drops below 0.
    """
    mgr = PaperAccountManager(db=temp_wal_db)
    # 1. Deploy cash: buy 100 shares of SPY ($49,950 spent, $50 cash remaining)
    mgr.execute_single_order("SPY", "BUY", shares=100.0, price=500.0)
    st1 = mgr.get_portfolio_state({"SPY": 500.0})
    assert st1.cash == pytest.approx(50.00, abs=1e-2)

    # 2. Deploy cash below $10 threshold: buy QQQ ($49.95 spent, ~$0.05 cash remaining)
    mgr.execute_single_order("QQQ", "BUY", shares=1.0, price=500.0)
    st2 = mgr.get_portfolio_state({"SPY": 500.0, "QQQ": 500.0})
    assert 0.0 <= st2.cash < 1.0

    # 3. With cash < $10.0, any additional BUY order must be rejected to protect cash floor
    intent = OrderIntent(
        symbol="GLD",
        action="BUY",
        side=OrderSide.BUY,
        delta_shares=1.0,
        estimated_price=200.0,
        notional=200.0,
    )
    report = mgr.execute_rebalance_orders([intent], {"GLD": 200.0})
    st3 = mgr.get_portfolio_state({"SPY": 500.0, "QQQ": 500.0, "GLD": 200.0})

    assert report.orders_rejected == 1
    assert report.orders_filled == 0
    assert st3.cash >= 0.00, f"Cash became negative: {st3.cash}"
    assert st3.cash == pytest.approx(st2.cash, abs=1e-4)
    mgr.close()



def test_cash_floor_invariant_with_fee_per_share():
    """
    Adversarial attack on cash clamping:
    Configure fee_per_share > 0 (e.g. $0.50/share commission) and min_fee_per_order > 0.
    Execute BUY order when cash is low.
    Verify that clamping accounts for the per-share fee and does NOT cause negative cash.
    """
    cfg = PaperAccountConfig(
        initial_cash=25.00,
        fee_per_share=0.50,
        min_fee_per_order=1.00,
        cash_buffer_pct=0.001,
    )
    mgr = PaperAccountManager(db=":memory:", config=cfg)

    # Cash is $25.00. Request BUY of $100.00 worth of XYZ at $2.00/share (50 shares)
    intent = OrderIntent(
        symbol="XYZ",
        action="BUY",
        side=OrderSide.BUY,
        delta_shares=50.0,
        estimated_price=2.00,
        notional=100.00,
    )
    report = mgr.execute_rebalance_orders([intent], {"XYZ": 2.00})
    st = mgr.get_portfolio_state({"XYZ": 2.00})

    assert st.cash >= 0.00, (
        f"Cash invariant breached! Cash became negative ({st.cash}) because "
        f"clamping omitted fee_per_share in share calculation."
    )
    mgr.close()


def test_cash_floor_invariant_round_up_boundary():
    """
    Adversarial boundary test:
    When cash_buffer_pct is 0.0 and division produces a recurring decimal where
    round(clamped_shares, 4) rounds UP, verify cash does not drop below 0.0.
    """
    cfg = PaperAccountConfig(
        initial_cash=11.00,
        cash_buffer_pct=0.0,
        min_fee_per_order=1.00,
        allow_fractional=True,
    )
    mgr = PaperAccountManager(db=":memory:", config=cfg)

    # 11 - 1 = 10 spendable cash. Price 7.0 -> 10/7 = 1.4285714 -> rounds UP to 1.4286
    # 1.4286 * 7.0 + 1.0 = 10.0002 + 1.0 = 11.0002 -> cash = -0.0002
    intent = OrderIntent(
        symbol="XYZ",
        action="BUY",
        side=OrderSide.BUY,
        delta_shares=10.0,
        estimated_price=7.00,
        notional=70.00,
    )
    report = mgr.execute_rebalance_orders([intent], {"XYZ": 7.00})
    st = mgr.get_portfolio_state({"XYZ": 7.00})

    assert st.cash >= 0.00, (
        f"Cash invariant breached: Cash is negative ({st.cash}) due to round-up of clamped_shares."
    )
    mgr.close()


# ==============================================================================
# 3. STRICT SELL-BEFORE-BUY ORDERING TESTS
# ==============================================================================


def test_strict_sell_before_buy_funds_subsequent_buys():
    """
    Verify strict SELL-before-BUY ordering within execute_rebalance_orders:
    Start with $0 uninvested cash (100% invested in SPY).
    Submit rebalance list where BUY orders appear FIRST in the list, followed by SELL orders.
    Assert:
    - SELL executes first, liberating cash.
    - Subsequent BUY uses newly liberated cash and fills completely.
    - No BUY order is prematurely rejected due to initial lack of cash.
    """
    mgr = PaperAccountManager(db=":memory:")
    # Deploy all cash into SPY
    mgr.execute_single_order("SPY", "BUY", shares=100.0, price=500.0)
    st0 = mgr.get_portfolio_state({"SPY": 500.0})

    # Prepare rebalance: BUY QQQ ($30,000) followed by SELL SPY ($40,000)
    buy_qqq = OrderIntent(
        symbol="QQQ",
        action="BUY",
        side=OrderSide.BUY,
        delta_shares=75.0,
        estimated_price=400.0,
        notional=30000.0,
    )
    sell_spy = OrderIntent(
        symbol="SPY",
        action="SELL",
        side=OrderSide.SELL,
        delta_shares=-80.0,
        estimated_price=500.0,
        notional=40000.0,
    )

    # Intentionally pass BUY before SELL in list
    report = mgr.execute_rebalance_orders([buy_qqq, sell_spy], {"SPY": 500.0, "QQQ": 400.0})

    assert report.orders_received == 2
    assert report.orders_filled == 2
    assert report.orders_rejected == 0
    assert len(report.trades) == 2

    # Assert SELL was executed before BUY
    assert report.trades[0].side == "SELL"
    assert report.trades[0].symbol == "SPY"
    assert report.trades[1].side == "BUY"
    assert report.trades[1].symbol == "QQQ"

    # Both positions must be reflected in state
    st1 = mgr.get_portfolio_state({"SPY": 500.0, "QQQ": 400.0})
    pos_symbols = {p.symbol for p in st1.positions}
    assert "SPY" in pos_symbols
    assert "QQQ" in pos_symbols
    mgr.close()


def test_order_side_string_compatibility():
    """
    Verify that execute_rebalance_orders accepts duck-typed order objects or
    OrderIntent where `side` is passed as a string ('BUY' or 'SELL') without throwing
    AttributeError: 'str' object has no attribute 'value'.
    """
    mgr = PaperAccountManager(db=":memory:")

    class DuckOrder:
        def __init__(self, sym, side_str, shares, price):
            self.id = "duck_1"
            self.symbol = sym
            self.side = side_str  # string, not Enum!
            self.action = side_str
            self.delta_shares = shares if side_str == "BUY" else -shares
            self.estimated_price = price
            self.notional = shares * price

    order = DuckOrder("SPY", "BUY", 10.0, 500.0)
    try:
        report = mgr.execute_rebalance_orders([order], {"SPY": 500.0})
        assert report.orders_filled == 1
    except AttributeError as err:
        pytest.fail(f"PaperAccountManager crashed on string order.side: {err}")
    mgr.close()


# ==============================================================================
# 4. RESTART RESILIENCE & PRISTINE RESET TESTS
# ==============================================================================


def test_unclean_process_crash_wal_recovery(temp_wal_db: str):
    """
    Simulate a sudden unclean process crash (SIGKILL / os._exit without closing db):
    1. Worker process writes BUY orders to WAL db and crashes via os._exit(99).
    2. Challenger process re-opens database and verifies full state integrity.
    """
    child_script = f"""
import os
from bot.paper_account import PaperAccountManager
mgr = PaperAccountManager(db="{temp_wal_db}")
mgr.execute_single_order("SPY", "BUY", 20.0, 500.0)
mgr.execute_single_order("GLD", "BUY", 50.0, 200.0)
# Crash abruptly without calling mgr.close() or letting Python cleanup
os._exit(99)
"""
    proc = subprocess.run([sys.executable, "-c", child_script], capture_output=True, text=True)
    assert proc.returncode == 99, f"Child process failed to exit with code 99: {proc.stderr}"

    # Reopen database in parent process
    mgr = PaperAccountManager(db=temp_wal_db)
    st = mgr.get_portfolio_state({"SPY": 500.0, "GLD": 200.0})

    assert st.cash == pytest.approx(30000.00, abs=1e-2)
    assert st.equity == pytest.approx(20000.00, abs=1e-2)
    assert st.total_nav == pytest.approx(50000.00, abs=1e-2)
    assert len(st.positions) == 2
    assert len(mgr.get_trade_history()) == 2
    mgr.close()


def test_reset_to_pristine_invariants(temp_wal_db: str):
    """
    Verify reset_to_pristine() contract:
    - Purges all positions, trades, orders, and equity snapshots.
    - Restores cash to exactly $50,000.00.
    - Restores total_nav to exactly $50,000.00.
    - Leaves 0 positions.
    - Persists across subsequent process restarts.
    """
    mgr = PaperAccountManager(db=temp_wal_db)

    # Create messy state with several trades
    mgr.execute_single_order("SPY", "BUY", 10.0, 500.0)
    mgr.execute_single_order("QQQ", "BUY", 20.0, 400.0)
    mgr.execute_single_order("SPY", "SELL", 5.0, 550.0)

    pre_state = mgr.get_portfolio_state()
    assert len(pre_state.positions) > 0
    assert len(mgr.get_trade_history()) > 0

    # Execute pristine reset
    clean_state = mgr.reset_to_pristine()

    assert clean_state.cash == 50000.00
    assert clean_state.equity == 0.00
    assert clean_state.total_nav == 50000.00
    assert clean_state.realized_pnl == 0.00
    assert clean_state.unrealized_pnl == 0.00
    assert clean_state.cumulative_fees == 0.00
    assert len(clean_state.positions) == 0
    assert len(mgr.get_trade_history()) == 0
    mgr.close()

    # Reopen to verify persistence of pristine state
    mgr_reopened = PaperAccountManager(db=temp_wal_db)
    reopened_state = mgr_reopened.get_portfolio_state()
    assert reopened_state.cash == 50000.00
    assert reopened_state.equity == 0.00
    assert reopened_state.total_nav == 50000.00
    assert len(reopened_state.positions) == 0
    mgr_reopened.close()
