"""
tests.unit.test_dashboard_api
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive unit and API test suite for Milestone 3 FastAPI Operator Dashboard.
Verifies:
- Root dashboard HTML serving (HTTP 200)
- Unauthenticated Railway health check endpoint (/health)
- Portfolio summary schema adherence (PortfolioSummary contract)
- Real-time connection and alert banner state (/api/status)
- Operator controls: Pause, Resume, Manual Rebalance, Pristine Reset
- Edge cases, error responses (404, 405, 422), and RLock concurrency.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from bot.service import (
    DynamicStrategyService,
    ManualRebalanceResult,
    ServiceConfig,
    ServiceState,
    ServiceStatus,
)
from bot.paper_account import PaperAccountManager, PortfolioSummary
from bot.feed_manager import FeedManager, FeedManagerConfig, FeedSource
from web.app import create_app


@pytest.fixture(autouse=True)
def ensure_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield


@pytest.fixture
def service_instance(temp_sqlite_db) -> Generator[DynamicStrategyService, None, None]:
    """Provides an isolated DynamicStrategyService with pristine $50k paper ledger."""
    config = ServiceConfig(
        db_path=temp_sqlite_db,
        relay_base_url="http://127.0.0.1:9999",
        relay_ws_url="ws://127.0.0.1:9999",
        initial_cash=50000.00,
        dry_run=False,
    )
    service = DynamicStrategyService(config=config)
    service.paper_account.init_schema()
    service._service_state = ServiceState.RUNNING
    yield service
    service.paper_account.close()


@pytest.fixture
def app_client(service_instance: DynamicStrategyService) -> Generator[TestClient, None, None]:
    """TestClient bound to isolated service."""
    app = create_app(service=service_instance)
    with TestClient(app) as client:
        yield client


# ==============================================================================
# 1. Root & Health Endpoints
# ==============================================================================

def test_get_root_dashboard_html_200(app_client: TestClient):
    """GET / serves HTML operator dashboard with HTTP 200 without requiring auth."""
    resp = app_client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    content = resp.text
    assert "<!DOCTYPE html>" in content or "<html" in content
    assert "<meta" in content and "viewport" in content
    assert "Operator Dashboard" in content or "Dynamic Long-Term" in content


def test_get_health_unauthenticated_200(app_client: TestClient):
    """GET /health returns HTTP 200 JSON unauthenticated for Railway monitoring."""
    # Strictly no Authorization header
    resp = app_client.get("/health")
    assert resp.status_code == 200
    assert "application/json" in resp.headers.get("content-type", "")

    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "DynamicLongTermStrategyBot"
    assert data["state"] in ("RUNNING", "PAUSED", "INITIALIZING")
    assert "relay" in data
    assert "portfolio" in data
    assert math.isclose(data["portfolio"]["nav"], 50000.00, rel_tol=1e-3)
    assert data["portfolio"]["cash"] == 50000.00
    assert data["portfolio"]["equity"] == 0.00
    assert data["portfolio"]["positions_count"] == 0
    assert data["uptime_s"] >= 0.0


# ==============================================================================
# 2. Portfolio & Telemetry Endpoints
# ==============================================================================

def test_get_portfolio_summary_schema(app_client: TestClient):
    """GET /api/portfolio conforms strictly to PortfolioSummary schema contract."""
    resp = app_client.get("/api/portfolio")
    assert resp.status_code == 200
    assert "application/json" in resp.headers.get("content-type", "")

    data = resp.json()
    assert "cash" in data
    assert "equity" in data
    assert "total_nav" in data
    assert "realized_pnl" in data
    assert "unrealized_pnl" in data
    assert "positions" in data
    assert "cash_weight" in data
    assert "as_of" in data

    # Numerical invariants
    assert math.isclose(data["cash"], 50000.00, rel_tol=1e-3)
    assert math.isclose(data["total_nav"], data["cash"] + data["equity"], rel_tol=1e-3)
    assert data["cash_weight"] == 1.0
    assert isinstance(data["positions"], list)
    assert len(data["positions"]) == 0


def test_get_status_connection_and_banner_flag(app_client: TestClient, service_instance: DynamicStrategyService):
    """GET /api/status returns operational telemetry, connection source, and alert banner."""
    resp = app_client.get("/api/status")
    assert resp.status_code == 200

    data = resp.json()
    assert data["service_name"] == "DynamicLongTermStrategyBot"
    assert data["is_running"] is True
    assert data["is_paused"] is False
    assert "alert_banner_active" in data
    assert isinstance(data["alert_banner_active"], bool)
    assert data["feed_source"] in ("alpaca_relay", "synthetic_fallback")
    assert data["total_nav"] == 50000.00


# ==============================================================================
# 3. Operator Real-Time Controls
# ==============================================================================

def test_post_operator_pause(app_client: TestClient, service_instance: DynamicStrategyService):
    """POST /api/operator/pause freezes strategy evaluations and transitions to PAUSED."""
    resp = app_client.post("/api/operator/pause")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "PAUSED"
    assert data["is_paused"] is True

    # Verify service internal state updated
    assert service_instance.get_service_status().is_paused is True

    # Verify reflected in /api/status and /health
    status_resp = app_client.get("/api/status").json()
    assert status_resp["is_paused"] is True
    assert status_resp["state"] == "PAUSED"

    health_resp = app_client.get("/health").json()
    assert health_resp["state"] == "PAUSED"


def test_post_operator_resume(app_client: TestClient, service_instance: DynamicStrategyService):
    """POST /api/operator/resume unfreezes strategy evaluations and transitions to RUNNING."""
    # Pause first
    app_client.post("/api/operator/pause")

    resp = app_client.post("/api/operator/resume")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "RUNNING"
    assert data["is_paused"] is False

    # Verify service internal state updated
    assert service_instance.get_service_status().is_paused is False

    status_resp = app_client.get("/api/status").json()
    assert status_resp["is_paused"] is False
    assert status_resp["state"] == "RUNNING"


@pytest.mark.asyncio
async def test_post_operator_rebalance_running(app_client: TestClient, service_instance: DynamicStrategyService):
    """POST /api/operator/rebalance executes immediate rebalance when RUNNING."""
    # Ensure warmup bars exist
    await service_instance._warmup_historical_bars()

    resp = app_client.post("/api/operator/rebalance", json={"force": True})
    assert resp.status_code == 200
    data = resp.json()

    assert data["success"] is True
    assert data["status"] in ("EXECUTED", "SKIPPED_WITHIN_BAND")
    assert "orders_count" in data
    assert "executed_trades_count" in data
    assert "rationale" in data
    assert data["portfolio_state_after"] is not None


def test_post_operator_rebalance_paused_rejection(app_client: TestClient):
    """POST /api/operator/rebalance is rejected when PAUSED unless force=True."""
    # Pause service
    app_client.post("/api/operator/pause")

    resp = app_client.post("/api/operator/rebalance", json={"force": False})
    # Either 200 with success=False or 409 Conflict
    assert resp.status_code in (200, 409)
    data = resp.json()
    assert data["success"] is False
    assert "REJECTED_PAUSED" in data["status"]
    assert data["orders_count"] == 0


@pytest.mark.asyncio
async def test_post_operator_rebalance_force_override(app_client: TestClient, service_instance: DynamicStrategyService):
    """POST /api/operator/rebalance with force=True executes even when PAUSED."""
    await service_instance._warmup_historical_bars()
    app_client.post("/api/operator/pause")

    resp = app_client.post("/api/operator/rebalance", json={"force": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True


def test_post_operator_reset(app_client: TestClient, service_instance: DynamicStrategyService):
    """POST /api/operator/reset clears paper account and resets to pristine $50,000.00 balance."""
    resp = app_client.post("/api/operator/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "portfolio" in data
    assert math.isclose(data["portfolio"]["cash"], 50000.00)
    assert data["portfolio"]["positions"] == []


def test_get_trades_and_equity_history(app_client: TestClient):
    """GET /api/trades and GET /api/equity-history return valid historical records."""
    t_resp = app_client.get("/api/trades?limit=10")
    assert t_resp.status_code == 200
    assert "trades" in t_resp.json()

    e_resp = app_client.get("/api/equity-history?limit=10")
    assert e_resp.status_code == 200
    assert "history" in e_resp.json()


# ==============================================================================
# 4. Error Handling & Edge Cases
# ==============================================================================

def test_api_404_not_found(app_client: TestClient):
    """Unknown API routes return HTTP 404 with standard JSON detail."""
    resp = app_client.get("/api/unknown_route_xyz")
    assert resp.status_code == 404
    data = resp.json()
    assert "detail" in data


def test_api_405_method_not_allowed(app_client: TestClient):
    """Calling POST endpoints with GET returns HTTP 405."""
    resp = app_client.get("/api/operator/pause")
    assert resp.status_code == 405


def test_api_422_validation_error(app_client: TestClient):
    """Sending malformed parameter types returns HTTP 422 Unprocessable Entity."""
    resp = app_client.post("/api/operator/rebalance", json={"force": "invalid-non-boolean"})
    assert resp.status_code == 422


def test_thread_safety_and_rlock_protection(app_client: TestClient):
    """Rapid concurrent requests across threads complete without SQLite lock errors."""
    def make_request(i: int):
        endpoint = "/api/portfolio" if i % 2 == 0 else "/api/status"
        res = app_client.get(endpoint)
        return res.status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(make_request, i) for i in range(20)]
        results = [f.result() for f in futures]

    assert all(code == 200 for code in results)
