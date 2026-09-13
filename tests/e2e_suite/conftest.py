"""Pytest fixtures for 4-Tier Opaque-Box E2E Test Suite (tests/e2e_suite/)."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Generator
import pytest

from tests.e2e_suite.contracts import (
    DataFeedManagerContract,
    DiscordNotifierContract,
    OperatorAppContract,
    PaperAccountManagerContract,
    RebalanceOrder,
    resolve_discord_notifier_cls,
    resolve_feed_manager_cls,
    resolve_paper_account_cls,
)


@pytest.fixture
def temp_paper_db() -> Generator[str, None, None]:
    """Provides an isolated SQLite WAL database path in a temporary directory."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="paper_test_")
    os.close(fd)
    yield path
    for p in [path, f"{path}-wal", f"{path}-shm"]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


@pytest.fixture
def paper_account(temp_paper_db: str):
    """Instantiates a PaperAccountManager resolved dynamically."""
    cls = resolve_paper_account_cls()
    mgr = cls(temp_paper_db)
    return mgr


@pytest.fixture
def feed_manager():
    """Instantiates a DataFeedManager resolved dynamically."""
    cls = resolve_feed_manager_cls()
    return cls(token="test-relay-token")


@pytest.fixture
def discord_notifier():
    """Instantiates an institutional DiscordNotifier with pytest suppression enabled."""
    cls = resolve_discord_notifier_cls()
    return cls(
        webhook_url="https://discord.com/api/webhooks/12345/mock_token",
        rate_limit_interval_s=2.0,
        max_backoff_sleep_s=5.0,
        suppress_in_test=True,
    )


@pytest.fixture
def operator_app(paper_account, feed_manager):
    """Instantiates operator web application for HTTP endpoint testing."""
    return OperatorAppContract(
        paper_account=paper_account,
        feed_manager=feed_manager,
    )


@pytest.fixture
def sample_rebalance_orders():
    """Provides sample RebalanceOrder instances."""
    return [
        RebalanceOrder(symbol="QQQ", action="BUY", shares=25.0, price=440.0, target_weight=0.22, current_weight=0.0),
        RebalanceOrder(symbol="XLK", action="BUY", shares=40.0, price=205.0, target_weight=0.16, current_weight=0.0),
        RebalanceOrder(symbol="SPY", action="BUY", shares=30.0, price=500.0, target_weight=0.30, current_weight=0.0),
    ]
