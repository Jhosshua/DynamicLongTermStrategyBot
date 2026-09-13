"""
tests.unit.test_paper_account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive unit test suite for Virtual Paper Trading Engine (bot.paper_account).
Verifies:
- $50,000.00 pristine initial balance
- Precision accounting: cost basis, unrealized/realized P&L, NAV, weights
- Rebalancing sequencing (SELLs before BUYs)
- Cash guard & non-negative cash enforcement
- Slippage and fee modeling
- SQLite WAL persistence across simulated service restarts
- Atomic reset_to_pristine() functionality
- Concurrency and interface compliance
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import sqlite3
import tempfile
import threading
from typing import Generator
import pytest

from strategy_engine.core.models import OrderIntent, OrderSide
from bot.paper_account import (
    PaperAccountConfig,
    PaperAccountManager,
    PaperTrade,
    PortfolioSummary,
    PositionDetail,
)


@pytest.fixture
def mem_manager() -> Generator[PaperAccountManager, None, None]:
    """In-memory PaperAccountManager fixture."""
    mgr = PaperAccountManager(db=":memory:")
    yield mgr
    mgr.close()


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Temporary file path for disk-based SQLite WAL persistence tests."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for p in [path, f"{path}-wal", f"{path}-shm"]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def test_init_pristine_account(mem_manager: PaperAccountManager):
    """Verify clean initialization with exact $50,000.00 cash balance and 0 positions."""
    state = mem_manager.get_portfolio_state()
    assert state.cash == 50000.00
    assert state.equity == 0.00
    assert state.total_nav == 50000.00
    assert state.realized_pnl == 0.00
    assert state.unrealized_pnl == 0.00
    assert state.cumulative_fees == 0.00
    assert state.cash_weight == 1.0
    assert len(state.positions) == 0


def test_idempotent_initialization(mem_manager: PaperAccountManager):
    """Calling init_schema repeatedly should not overwrite or corrupt existing state."""
    mem_manager.execute_single_order("SPY", "BUY", shares=10.0, price=500.0)
    s1 = mem_manager.get_portfolio_state()
    assert s1.cash == 45000.00

    # Re-run initialization
    mem_manager.init_schema()
    s2 = mem_manager.get_portfolio_state()
    assert s2.cash == 45000.00
    assert len(s2.positions) == 1
    assert s2.positions[0].symbol == "SPY"


def test_single_buy_execution(mem_manager: PaperAccountManager):
    """Verify single BUY order execution reduces cash and creates position."""
    trade = mem_manager.execute_single_order("SPY", "BUY", shares=10.0, price=500.0)
    assert trade.symbol == "SPY"
    assert trade.side == "BUY"
    assert trade.shares == 10.0
    assert trade.price == 500.0
    assert trade.notional == 5000.0
    assert trade.cash_after == 45000.00
    assert trade.nav_after == 50000.00

    state = mem_manager.get_portfolio_state({"SPY": 500.0})
    assert state.cash == 45000.00
    assert state.equity == 5000.00
    assert state.total_nav == 50000.00
    assert len(state.positions) == 1

    pos = state.positions[0]
    assert pos.symbol == "SPY"
    assert pos.qty == 10.0
    assert pos.avg_entry_price == 500.0
    assert pos.market_value == 5000.0
    assert pos.weight == pytest.approx(0.10, abs=1e-4)


def test_multiple_buy_orders(mem_manager: PaperAccountManager):
    """Verify multi-asset purchases update cash, positions, and total NAV."""
    mem_manager.execute_single_order("SPY", "BUY", shares=20.0, price=500.0)   # $10,000
    mem_manager.execute_single_order("QQQ", "BUY", shares=25.0, price=400.0)   # $10,000
    mem_manager.execute_single_order("GLD", "BUY", shares=25.0, price=200.0)   # $5,000

    state = mem_manager.get_portfolio_state({"SPY": 500.0, "QQQ": 400.0, "GLD": 200.0})
    assert state.cash == 25000.00
    assert state.equity == 25000.00
    assert state.total_nav == 50000.00
    assert len(state.positions) == 3

    weights_sum = sum(p.weight for p in state.positions) + state.cash_weight
    assert weights_sum == pytest.approx(1.0, abs=1e-5)


def test_weighted_average_entry_price(mem_manager: PaperAccountManager):
    """Verify average entry price adjusts on multiple purchases at varying prices."""
    mem_manager.execute_single_order("SPY", "BUY", shares=10.0, price=500.0)  # $5,000
    mem_manager.execute_single_order("SPY", "BUY", shares=10.0, price=550.0)  # $5,500

    state = mem_manager.get_portfolio_state({"SPY": 525.0})
    assert len(state.positions) == 1
    pos = state.positions[0]
    assert pos.qty == 20.0
    assert pos.cost_basis == 10500.00
    assert pos.avg_entry_price == pytest.approx(525.0, abs=1e-4)
    assert pos.unrealized_pnl == 0.00


def test_partial_sell_with_gain(mem_manager: PaperAccountManager):
    """Verify partial liquidation with profit records realized P&L and updates cash."""
    mem_manager.execute_single_order("SPY", "BUY", shares=20.0, price=500.0)  # $10,000, cash=$40,000
    sell_trade = mem_manager.execute_single_order("SPY", "SELL", shares=10.0, price=600.0)  # $6,000 proceeds

    assert sell_trade.realized_pnl == 1000.00
    assert sell_trade.cash_after == 46000.00

    state = mem_manager.get_portfolio_state({"SPY": 600.0})
    assert state.cash == 46000.00
    assert state.equity == 6000.00
    assert state.total_nav == 52000.00
    assert state.realized_pnl == 1000.00
    assert state.unrealized_pnl == 1000.00  # remaining 10 shares purchased @ $500 marked @ $600
    assert len(state.positions) == 1
    assert state.positions[0].qty == 10.0
    assert state.positions[0].avg_entry_price == 500.0


def test_partial_sell_with_loss(mem_manager: PaperAccountManager):
    """Verify partial liquidation with loss updates realized P&L negatively."""
    mem_manager.execute_single_order("SPY", "BUY", shares=20.0, price=500.0)  # cash=$40,000
    sell_trade = mem_manager.execute_single_order("SPY", "SELL", shares=10.0, price=450.0)  # $4,500 proceeds

    assert sell_trade.realized_pnl == -500.00
    assert sell_trade.cash_after == 44500.00

    state = mem_manager.get_portfolio_state({"SPY": 450.0})
    assert state.cash == 44500.00
    assert state.equity == 4500.00
    assert state.total_nav == 49000.00
    assert state.realized_pnl == -500.00
    assert state.unrealized_pnl == -500.00


def test_complete_sell_closed_position(mem_manager: PaperAccountManager):
    """Verify selling entire position completely removes it from the positions ledger."""
    mem_manager.execute_single_order("SPY", "BUY", shares=10.0, price=500.0)
    mem_manager.execute_single_order("SPY", "SELL", shares=10.0, price=550.0)

    state = mem_manager.get_portfolio_state()
    assert state.cash == 50500.00
    assert state.equity == 0.00
    assert state.total_nav == 50500.00
    assert state.realized_pnl == 500.00
    assert len(state.positions) == 0


def test_rebalance_sequencing_sells_fund_buys(mem_manager: PaperAccountManager):
    """Verify that in a rebalancing batch, SELLs execute before BUYs to liberate cash."""
    # Deploy $49,000 into SPY, leaving only $1,000 cash
    mem_manager.execute_single_order("SPY", "BUY", shares=98.0, price=500.0)  # $49,000, cash=$1,000

    # Rebalance: Sell 40 SPY ($20,000) and Buy 50 QQQ ($20,000)
    sell_order = OrderIntent(
        symbol="SPY",
        action="SELL",
        side=OrderSide.SELL,
        delta_shares=-40.0,
        estimated_price=500.0,
        notional=20000.0,
    )
    buy_order = OrderIntent(
        symbol="QQQ",
        action="BUY",
        side=OrderSide.BUY,
        delta_shares=50.0,
        estimated_price=400.0,
        notional=20000.0,
    )

    # Pass in BUY first to test sorting/sequencing logic
    report = mem_manager.execute_rebalance_orders(
        orders=[buy_order, sell_order],
        current_prices={"SPY": 500.0, "QQQ": 400.0},
    )

    assert report.orders_filled == 2
    assert report.orders_rejected == 0
    assert len(report.trades) == 2
    assert report.trades[0].side == "SELL"
    assert report.trades[1].side == "BUY"

    state = report.portfolio_state_after
    assert state.cash == pytest.approx(1000.00, abs=1e-2)
    assert len(state.positions) == 2


def test_cash_guard_clamping(mem_manager: PaperAccountManager):
    """Verify that when requested BUY exceeds spendable cash, shares are adaptively clamped."""
    # Cash is $50,000. Request BUY of $60,000 worth of SPY (120 shares @ $500)
    buy_order = OrderIntent(
        symbol="SPY",
        action="BUY",
        side=OrderSide.BUY,
        delta_shares=120.0,
        estimated_price=500.0,
        notional=60000.0,
    )

    report = mem_manager.execute_rebalance_orders(
        orders=[buy_order],
        current_prices={"SPY": 500.0},
    )

    assert report.orders_filled == 1
    state = report.portfolio_state_after
    assert state.cash >= 0.00  # Cash must never be negative!
    assert state.positions[0].qty < 120.0


def test_mark_to_market_unrealized_pnl(mem_manager: PaperAccountManager):
    """Verify unrealized P&L and weights update dynamically as prices fluctuate."""
    mem_manager.execute_single_order("SPY", "BUY", shares=50.0, price=500.0)  # $25,000

    # Bullish price rise: SPY -> $550
    s_bull = mem_manager.update_market_prices({"SPY": 550.0})
    assert s_bull.equity == 27500.00
    assert s_bull.unrealized_pnl == 2500.00
    assert s_bull.total_nav == 52500.00

    # Pullback: SPY -> $480
    s_bear = mem_manager.update_market_prices({"SPY": 480.0})
    assert s_bear.equity == 24000.00
    assert s_bear.unrealized_pnl == -1000.00
    assert s_bear.total_nav == 49000.00


def test_accounting_invariant(mem_manager: PaperAccountManager):
    """Verify accounting identity invariant: NAV == Initial + Realized + Unrealized - Fees."""
    prices = {"SPY": 500.0, "QQQ": 400.0}
    mem_manager.execute_single_order("SPY", "BUY", shares=20.0, price=500.0)
    mem_manager.execute_single_order("QQQ", "BUY", shares=25.0, price=400.0)
    mem_manager.execute_single_order("SPY", "SELL", shares=10.0, price=550.0)

    state = mem_manager.get_portfolio_state({"SPY": 540.0, "QQQ": 410.0})
    expected_nav = 50000.00 + state.realized_pnl + state.unrealized_pnl - state.cumulative_fees
    assert state.total_nav == pytest.approx(expected_nav, abs=1e-2)


def test_slippage_modeling():
    """Verify slippage increases BUY fill price and decreases SELL fill price."""
    cfg = PaperAccountConfig(slippage_bps=10.0)  # 10 bps = 0.1%
    mgr = PaperAccountManager(db=":memory:", config=cfg)

    trade_buy = mgr.execute_single_order("SPY", "BUY", shares=10.0, price=500.0)
    assert trade_buy.price == pytest.approx(500.50, abs=1e-4)

    trade_sell = mgr.execute_single_order("SPY", "SELL", shares=5.0, price=600.0)
    assert trade_sell.price == pytest.approx(599.40, abs=1e-4)
    mgr.close()


def test_fee_modeling():
    """Verify fee deductions are debited from cash and accumulated."""
    cfg = PaperAccountConfig(fee_per_share=0.01, min_fee_per_order=1.00)
    mgr = PaperAccountManager(db=":memory:", config=cfg)

    trade = mgr.execute_single_order("SPY", "BUY", shares=200.0, price=100.0)
    assert trade.fee == 2.00  # 200 * $0.01 = $2.00
    assert trade.cash_after == 50000.00 - 20000.00 - 2.00

    state = mgr.get_portfolio_state({"SPY": 100.0})
    assert state.cumulative_fees == 2.00
    mgr.close()


def test_persistence_across_restart(temp_db_path: str):
    """Verify account state, positions, and history survive process restart."""
    mgr1 = PaperAccountManager(db=temp_db_path)
    mgr1.execute_single_order("SPY", "BUY", shares=20.0, price=500.0)
    mgr1.execute_single_order("SPY", "SELL", shares=5.0, price=600.0)
    s1 = mgr1.get_portfolio_state({"SPY": 600.0})
    mgr1.close()

    # Restart service / re-open from same database file
    mgr2 = PaperAccountManager(db=temp_db_path)
    s2 = mgr2.get_portfolio_state({"SPY": 600.0})

    assert s2.cash == s1.cash
    assert s2.equity == s1.equity
    assert s2.total_nav == s1.total_nav
    assert s2.realized_pnl == s1.realized_pnl
    assert len(s2.positions) == 1
    assert s2.positions[0].symbol == "SPY"
    assert s2.positions[0].qty == 15.0
    assert s2.positions[0].avg_entry_price == 500.0

    trades = mgr2.get_trade_history()
    assert len(trades) == 2
    mgr2.close()


def test_reset_to_pristine(temp_db_path: str):
    """Verify reset_to_pristine wipes all data and restores exact $50,000.00."""
    mgr = PaperAccountManager(db=temp_db_path)
    mgr.execute_single_order("SPY", "BUY", shares=20.0, price=500.0)
    mgr.execute_single_order("QQQ", "BUY", shares=20.0, price=400.0)

    # Perform pristine reset
    clean_state = mgr.reset_to_pristine()
    assert clean_state.cash == 50000.00
    assert clean_state.equity == 0.00
    assert clean_state.total_nav == 50000.00
    assert clean_state.realized_pnl == 0.00
    assert clean_state.unrealized_pnl == 0.00
    assert len(clean_state.positions) == 0
    assert len(mgr.get_trade_history()) == 0

    # Verify persistence of clean state after re-opening
    mgr.close()
    mgr_reopened = PaperAccountManager(db=temp_db_path)
    reopened_state = mgr_reopened.get_portfolio_state()
    assert reopened_state.cash == 50000.00
    assert reopened_state.total_nav == 50000.00
    assert len(reopened_state.positions) == 0
    mgr_reopened.close()


def test_concurrent_reads_wal(temp_db_path: str):
    """Verify concurrent reads from secondary thread execute cleanly during WAL transactions."""
    mgr = PaperAccountManager(db=temp_db_path)
    errors = []

    def reader_loop():
        try:
            reader_mgr = PaperAccountManager(db=temp_db_path)
            for _ in range(50):
                st = reader_mgr.get_portfolio_state()
                assert st.total_nav > 0
            reader_mgr.close()
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=reader_loop)
    t.start()

    for i in range(10):
        mgr.execute_single_order("SPY", "BUY", shares=1.0, price=500.0)

    t.join(timeout=10.0)
    assert len(errors) == 0
    mgr.close()


def test_portfolio_summary_interface_compliance(mem_manager: PaperAccountManager):
    """Verify PortfolioSummary conforms to PROJECT.md interface contract."""
    summary = mem_manager.get_portfolio_state()
    assert hasattr(summary, "cash")
    assert hasattr(summary, "equity")
    assert hasattr(summary, "total_nav")
    assert hasattr(summary, "realized_pnl")
    assert hasattr(summary, "unrealized_pnl")
    assert hasattr(summary, "positions")
    assert isinstance(summary.cash, float)
    assert isinstance(summary.equity, float)
    assert isinstance(summary.total_nav, float)
    assert isinstance(summary.positions, list)
