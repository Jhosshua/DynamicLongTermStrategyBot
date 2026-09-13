"""
tests.integration.test_e2e_smoke_workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pytest integration test suite executing the 7-phase E2E smoke test workflow:
1. Boot & Health Check (GET /health, GET /)
2. Synthetic Market Regime Transitions (BULL -> CORRECTION_FRAGILE -> BEAR_CRISIS)
3. Out-of-Cadence Rebalance & Order Execution (POST /api/operator/rebalance)
4. Network Chaos & Alert Banner Verification (Disconnect -> Recover)
5. Operator Controls (Pause -> Resume)
6. Institutional Discord v2 Alerts (Trade, Broken, Recovered)
7. Guaranteed Clean Reset Execution (Wipe -> $50k Pristine & 17 Invariants)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import math
from pathlib import Path
import pytest
import httpx

from bot.discord_alerts import DiscordNotifier
from bot.service import DynamicStrategyService, ServiceConfig
from scripts.e2e_smoke_test import SmokeTestRunner
from scripts.reset_pristine_for_monday import PristineResetManager
from web.app import create_app


@pytest.fixture
async def smoke_harness(tmp_path: Path):
    """Fixture providing in-process ASGI test harness for E2E smoke workflow."""
    db_file = str(tmp_path / "smoke_integration.db")
    config = ServiceConfig(db_path=db_file, initial_cash=50000.00)
    notifier = DiscordNotifier(suppress_in_test=True)
    service = DynamicStrategyService(config=config, discord_notifier=notifier)
    service.paper_account.init_schema()

    await service.client.state_machine.handle_upstream_connected("Harness boot")
    await service.feed_manager._transition_to_live("Harness boot")

    app = create_app(service=service)
    app.state.service = service
    app.state._service_injected = True

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000", timeout=10.0) as client:
        runner = SmokeTestRunner(
            client=client,
            service=service,
            db_path=db_file,
            timeout=10.0,
            target_display_url="http://127.0.0.1:8000 (Integration Pytest Harness)",
        )
        yield runner, client, service, db_file

    await service.shutdown()


@pytest.mark.asyncio
async def test_full_e2e_smoke_workflow(smoke_harness):
    """Full 7-phase E2E smoke test runs to 100% completion with clean $50k reset."""
    runner, client, service, db_file = smoke_harness

    all_passed = await runner.run_all(skip_reset=False)
    assert all_passed is True, f"Smoke test failed: {[r for r in runner.results if r.status != 'PASS']}"
    assert runner.summary.total_phases == 7
    assert runner.summary.passed_phases == 7
    assert runner.summary.failed_phases == 0
    assert runner.summary.success_rate_pct == 100.0
    assert runner.summary.verdict == "READY_FOR_MONDAY_OPEN"

    # Terminal report output validity
    report_text = runner.format_terminal_report()
    assert "DYNAMIC LONG-TERM STRATEGY BOT — E2E SMOKE TEST HARNESS REPORT" in report_text
    assert "RESULT: 7 OF 7 PHASES PASSED (100% SUCCESS)" in report_text
    assert "PRISTINE READINESS FOR MONDAY MARKET OPEN" in report_text

    # JSON report dictionary validity
    json_dict = runner.to_json_dict()
    assert json_dict["summary"]["passed_phases"] == 7
    assert len(json_dict["phases"]) == 7


@pytest.mark.asyncio
async def test_phase1_boot_and_health_check(smoke_harness):
    """Phase 1: Validates /health JSON and / HTML response."""
    runner, client, service, _ = smoke_harness
    p1 = await runner.phase_1_service_boot()
    assert p1.status == "PASS"
    assert p1.details.get("health_status") == "ok"


@pytest.mark.asyncio
async def test_phase2_regime_transitions(smoke_harness):
    """Phase 2: Validates Bull, Fragile, and Bear Crisis transitions."""
    runner, client, service, _ = smoke_harness
    p2 = await runner.phase_2_synthetic_regimes()
    assert p2.status == "PASS"
    assert "BULL_AGGRESSIVE" in p2.details.get("transitions_verified", [])
    assert "CORRECTION_FRAGILE" in p2.details.get("transitions_verified", [])
    assert "BEAR_CRISIS" in p2.details.get("transitions_verified", [])


@pytest.mark.asyncio
async def test_phase3_rebalance_execution(smoke_harness):
    """Phase 3: Validates out-of-cadence rebalance, cash debiting, and positions tracking."""
    runner, client, service, _ = smoke_harness
    await runner.phase_2_synthetic_regimes()
    p3 = await runner.phase_3_rebalance_execution()
    assert p3.status == "PASS"
    assert p3.details.get("orders_executed") > 0
    assert p3.details.get("total_bought") > 0.0
    assert p3.details.get("positions_held") >= 1


@pytest.mark.asyncio
async def test_phase4_network_chaos_alert_banner(smoke_harness):
    """Phase 4: Validates disconnect triggers alert banner, and recovery clears it."""
    runner, client, service, _ = smoke_harness
    p4 = await runner.phase_4_network_chaos()
    assert p4.status == "PASS"

    st = (await client.get("/api/status")).json()
    assert st["connection"]["is_connected"] is True
    assert st["connection"]["alert_banner_active"] is False


@pytest.mark.asyncio
async def test_phase5_operator_pause_resume(smoke_harness):
    """Phase 5: Validates operator pause freezes rebalance and resume unfreezes."""
    runner, client, service, _ = smoke_harness
    p5 = await runner.phase_5_operator_controls()
    assert p5.status == "PASS"

    h = (await client.get("/health")).json()
    assert h["state"] == "RUNNING"


@pytest.mark.asyncio
async def test_phase6_discord_cards(smoke_harness):
    """Phase 6: Validates Trade, Broken, and Recovered Discord v2 cards."""
    runner, client, service, _ = smoke_harness
    # Execute actions that emit alerts
    await runner.phase_2_synthetic_regimes()
    await runner.phase_3_rebalance_execution()
    await runner.phase_4_network_chaos()

    p6 = await runner.phase_6_discord_alerts()
    assert p6.status == "PASS"
    assert p6.details.get("trade_card_hex") == "0x1e88e5"
    assert p6.details.get("broken_card_hex") == "0xe53935"
    assert p6.details.get("recovered_card_hex") == "0x43a047"


@pytest.mark.asyncio
async def test_phase7_clean_reset_and_17_invariants(smoke_harness):
    """Phase 7: Validates clean reset returns to $50,000.00 cash and all 17 invariants pass."""
    runner, client, service, db_file = smoke_harness

    # Contaminate state first
    await runner.phase_2_synthetic_regimes()
    await runner.phase_3_rebalance_execution()

    p7 = await runner.phase_7_pristine_reset()
    assert p7.status == "PASS"
    assert p7.details.get("final_cash") == 50000.00
    assert p7.details.get("final_equity") == 0.00
    assert p7.details.get("final_nav") == 50000.00
    assert p7.details.get("positions_count") == 0
    assert p7.details.get("invariants_passed") == 17

    # Directly verify invariants using PristineResetManager
    mgr = PristineResetManager(db_path=db_file)
    invariants = mgr.verify_invariants()
    assert len(invariants) == 17
    assert all(inv.passed for inv in invariants)


@pytest.mark.asyncio
async def test_skip_reset_flag(smoke_harness):
    """Test skip_reset=True skips Phase 7 while passing all other phases."""
    runner, client, service, _ = smoke_harness
    all_passed = await runner.run_all(skip_reset=True)
    assert all_passed is True
    assert runner.results[-1].status == "SKIPPED"
    assert runner.results[-1].phase_num == 7
