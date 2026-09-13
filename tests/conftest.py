"""Global pytest configuration and fixtures for AlpacaRelay Strategy Engine."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os
import sqlite3
import tempfile
from typing import Dict, Generator, List
import pytest
import pytest_asyncio

from strategy_engine.core.models import (
    AssetClass,
    Bar,
    MarketRegime,
    OrderIntent,
    OrderSide,
    OrderType,
    Quote,
    SignalSnapshot,
    TargetAllocation,
    Trade,
)
from strategy_engine.core.universe import (
    ALL_SYMBOLS,
    CORE_COMPOUNDERS,
    CYCLICAL_SECTORS,
    DEFENSIVE_SECTORS,
    MEGACAP_LEADERS,
    SAFE_HAVENS,
    SECTOR_ETFS,
    UNIVERSE_GROWTH,
    UNIVERSE_SAFE_HAVENS,
    UNIVERSE_SECTORS,
)
from strategy_engine.simulator.stress_scenarios import (
    generate_2008_liquidity_crisis,
    generate_2017_low_vol_bull,
    generate_2020_flash_crash,
    generate_2022_inflation_grind,
)
from tests.mocks.mock_relay_server import MockAlpacaRelayServer
from tests.mocks.synthetic_data import MertonJumpDiffusionSimulator
import tests.mocks.contracts as mock_contracts


# --- M1 Unit Test Fixtures ---

@pytest.fixture
def base_timestamp() -> datetime:
    """Standardized UTC anchor timestamp."""
    return datetime(2026, 1, 5, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_bar(base_timestamp: datetime) -> Bar:
    """Valid frozen Bar instance for SPY."""
    return Bar(
        symbol="SPY",
        timestamp=base_timestamp,
        open=500.0,
        high=505.0,
        low=498.0,
        close=502.5,
        volume=50_000_000,
        trade_count=450_000,
        vwap=501.8,
    )


@pytest.fixture
def sample_quote(base_timestamp: datetime) -> Quote:
    """Valid frozen Quote instance for SPY."""
    return Quote(
        symbol="SPY",
        timestamp=base_timestamp,
        bid_price=502.40,
        bid_size=10,
        ask_price=502.50,
        ask_size=15,
    )


@pytest.fixture
def sample_trade(base_timestamp: datetime) -> Trade:
    """Valid frozen Trade instance for SPY."""
    return Trade(
        symbol="SPY",
        timestamp=base_timestamp,
        price=502.45,
        size=100,
        id="1001",
        trade_id="1001",
        conditions=["@"],
        tape="C",
    )


@pytest.fixture
def sample_signal_snapshot(base_timestamp: datetime) -> SignalSnapshot:
    """Valid frozen SignalSnapshot instance."""
    return SignalSnapshot(
        timestamp=base_timestamp,
        spy_price=502.5,
        spy_sma50=490.0,
        spy_sma200=470.0,
        realized_vol_20d=0.11,
        vol_scale_factor=1.0,
        drawdown_pct=-0.015,
        circuit_breaker_active=False,
        regime=MarketRegime.BULL_AGGRESSIVE,
        indicators={"realized_vol_20d": 0.11, "breadth_50": 0.75},
    )


@pytest.fixture
def sample_target_allocation(base_timestamp: datetime) -> TargetAllocation:
    """Valid TargetAllocation summing to 1.0."""
    return TargetAllocation(
        timestamp=base_timestamp,
        regime=MarketRegime.BULL_AGGRESSIVE,
        weights={"QQQ": 0.50, "XLK": 0.15, "XLI": 0.15, "SPY": 0.20},
        cash_weight=0.0,
        rationale="Confirmed Bull regime with low realized volatility.",
    )


@pytest.fixture
def sample_order_intent() -> OrderIntent:
    """Valid OrderIntent instance."""
    return OrderIntent(
        symbol="QQQ",
        action="BUY",
        target_weight=0.50,
        current_weight=0.30,
        delta_weight=0.20,
        target_shares=100.0,
        delta_shares=20.0,
        delta_dollars=8000.0,
        estimated_price=400.0,
        reason="Monthly momentum rebalance",
    )


@pytest.fixture
def universe_lists() -> Dict[str, List[str]]:
    """Complete dictionary of universe tier symbol lists."""
    return {
        "growth": UNIVERSE_GROWTH,
        "sectors": UNIVERSE_SECTORS,
        "safe_havens": UNIVERSE_SAFE_HAVENS,
        "all": ALL_SYMBOLS,
    }


@pytest.fixture(scope="session")
def scenario_2008_data() -> Dict[str, List[Bar]]:
    """Cached session-level 2008 liquidity crisis dataset (seed=42)."""
    return generate_2008_liquidity_crisis(seed=42)


@pytest.fixture(scope="session")
def scenario_2020_data() -> Dict[str, List[Bar]]:
    """Cached session-level 2020 flash crash dataset (seed=42)."""
    return generate_2020_flash_crash(seed=42)


@pytest.fixture(scope="session")
def scenario_2022_data() -> Dict[str, List[Bar]]:
    """Cached session-level 2022 inflation grind dataset (seed=42)."""
    return generate_2022_inflation_grind(seed=42)


@pytest.fixture(scope="session")
def scenario_2017_data() -> Dict[str, List[Bar]]:
    """Cached session-level 2017 low-vol bull dataset (seed=42)."""
    return generate_2017_low_vol_bull(seed=42)


# --- Backward Compatible E2E Test Fixtures ---

@pytest.fixture
def synthetic_simulator() -> MertonJumpDiffusionSimulator:
    return MertonJumpDiffusionSimulator(seed=42)


@pytest.fixture
def calibrated_2008_data(synthetic_simulator):
    return synthetic_simulator.generate_2008_crisis(n_days=252, seed=42)


@pytest.fixture
def calibrated_2020_data(synthetic_simulator):
    return synthetic_simulator.generate_2020_flash_crash(n_days=60, seed=42)


@pytest.fixture
def calibrated_2022_data(synthetic_simulator):
    return synthetic_simulator.generate_2022_inflation_grind(n_days=252, seed=42)


@pytest.fixture
def calibrated_2017_data(synthetic_simulator):
    return synthetic_simulator.generate_2017_low_vol_bull(n_days=252, seed=42)


@pytest_asyncio.fixture
async def mock_relay_server():
    server = MockAlpacaRelayServer(token="test-relay-token-xyz", feed="sip")
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest.fixture
def temp_sqlite_db() -> Generator[str, None, None]:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.commit()
    conn.close()
    yield path
    for p in [path, f"{path}-wal", f"{path}-shm"]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


@pytest.fixture
def progressive_contracts():
    """Provides access to domain models, preferring strategy_engine if implemented."""
    class ContractResolver:
        def __init__(self):
            try:
                import strategy_engine.core.models as real_models
                self.models = real_models
            except ImportError:
                self.models = mock_contracts

            try:
                import strategy_engine.core.math_utils as real_math
                self.math_utils = real_math
            except ImportError:
                self.math_utils = mock_contracts

            try:
                import strategy_engine.core.universe as real_universe
                self.universe = real_universe
            except ImportError:
                self.universe = mock_contracts

    return ContractResolver()
