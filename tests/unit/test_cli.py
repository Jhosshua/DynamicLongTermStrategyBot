"""
tests.unit.test_cli
~~~~~~~~~~~~~~~~~~~

Unit test suite for production Typer CLI application and ExplainabilityEngine.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from strategy_engine.cli.explain import ExplainabilityEngine
from strategy_engine.cli.main import app
from strategy_engine.core.models import (
    MarketRegime,
    OrderIntent,
    OrderSide,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.storage.database import Database
from strategy_engine.storage.repositories import StorageService

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    help_text = result.stdout.lower()
    for cmd in ["dry-run", "daemon", "rebalance", "backtest", "status", "export-metrics"]:
        assert cmd in help_text


def test_cli_dry_run_scenarios():
    for scen in ["2008", "2020", "2022", "2017"]:
        res = runner.invoke(app, ["dry-run", "-s", scen])
        assert res.exit_code == 0, f"dry-run failed for scenario {scen}: {res.stdout}"
        assert "TARGET ALLOCATION WEIGHTS" in res.stdout
        assert "100.00%" in res.stdout
        assert "DECISION RATIONALE" in res.stdout


def test_cli_dry_run_json():
    res = runner.invoke(app, ["dry-run", "-s", "2008", "--json"])
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)
    assert "signals" in data
    assert "target_allocation" in data
    assert "rebalance_orders" in data
    assert "regime" in data
    # Target weights sum must be 1.0
    weights = data["target_allocation"]["weights"]
    assert pytest.approx(sum(weights.values()), abs=1e-4) == 1.0


def test_cli_daemon_once(tmp_path):
    db_file = tmp_path / "daemon_cli.db"
    res = runner.invoke(app, ["daemon", "--once", "--db-path", str(db_file)])
    assert res.exit_code == 0, res.stdout
    assert "completed successfully" in res.stdout.lower() or "completed" in res.stdout.lower()


def test_cli_rebalance_drift_tolerance():
    # Portfolio already aligned with target (no orders expected)
    current_w = json.dumps({"QQQ": 0.50, "SPY": 0.50})
    res = runner.invoke(app, ["rebalance", "-w", current_w, "-s", "2017"])
    assert res.exit_code == 0, res.stdout
    assert "within drift tolerance" in res.stdout.lower() or "orders" in res.stdout.lower()


def test_cli_rebalance_force():
    current_w = json.dumps({"QQQ": 0.50, "SPY": 0.50})
    res = runner.invoke(app, ["rebalance", "-w", current_w, "-s", "2017", "--force"])
    assert res.exit_code == 0, res.stdout
    assert "rebalancing orders" in res.stdout.lower() or "order" in res.stdout.lower()


def test_cli_rebalance_json():
    current_w = json.dumps({"SPY": 1.0})
    res = runner.invoke(app, ["rebalance", "-w", current_w, "-s", "2008", "--json"])
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)
    assert "target_weights" in data
    assert "orders" in data
    assert len(data["orders"]) > 0


def test_cli_backtest_single_scenario():
    res = runner.invoke(app, ["backtest", "-s", "2008"])
    assert res.exit_code == 0, res.stdout
    assert "BACKTEST PERFORMANCE COMPARISON" in res.stdout
    assert "Cumulative Return" in res.stdout
    assert "Maximum Drawdown" in res.stdout
    assert "PASS" in res.stdout


def test_cli_backtest_json():
    res = runner.invoke(app, ["backtest", "-s", "2020", "--json"])
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)
    assert "2020" in data
    scen_data = data["2020"]
    assert "strategy" in scen_data
    assert "benchmark_spy" in scen_data
    assert scen_data["strategy"]["max_drawdown"] < 0
    assert scen_data["criteria_passed"] is True


def test_cli_status_fresh_db(tmp_path):
    db_file = tmp_path / "fresh_status.db"
    res = runner.invoke(app, ["status", "--db-path", str(db_file)])
    assert res.exit_code == 0, res.stdout
    assert "SYSTEM VITALS" in res.stdout
    assert "Table Name" in res.stdout


def test_cli_status_json(tmp_path):
    db_file = tmp_path / "status.db"
    res = runner.invoke(app, ["status", "--db-path", str(db_file), "--json"])
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)
    assert data["service"] == "strategy_engine"
    assert data["status"] == "active"
    assert "database" in data


def test_cli_export_metrics_json_and_csv(tmp_path):
    db_file = tmp_path / "export_test.db"
    db = Database(db_file)
    storage = StorageService(db)

    # Seed data
    t = datetime(2026, 9, 1, 15, 50, tzinfo=timezone.utc)
    snap = SignalSnapshot(
        timestamp=t,
        spy_price=500.0,
        spy_sma50=495.0,
        spy_sma200=480.0,
        realized_vol_20d=0.10,
        vol_scale_factor=1.0,
        drawdown_pct=-0.01,
        regime=MarketRegime.BULL_NORMAL,
    )
    storage.signals.save(snap, rationale="Test snap")
    db.close()

    # JSON export
    json_out = tmp_path / "metrics.json"
    res_j = runner.invoke(app, ["export-metrics", "--format", "json", "--table", "signals", "--output", str(json_out), "--db-path", str(db_file)])
    assert res_j.exit_code == 0, res_j.stdout
    assert json_out.exists()
    with open(json_out, "r") as f:
        j_data = json.load(f)
        assert len(j_data) == 1
        assert j_data[0]["regime"] == "BULL_NORMAL"

    # CSV export
    csv_out = tmp_path / "metrics.csv"
    res_c = runner.invoke(app, ["export-metrics", "--format", "csv", "--table", "signals", "--output", str(csv_out), "--db-path", str(db_file)])
    assert res_c.exit_code == 0, res_c.stdout
    assert csv_out.exists()
    with open(csv_out, "r") as f:
        lines = f.readlines()
        assert len(lines) >= 2  # Header + 1 row
        assert "timestamp" in lines[0]


def test_explain_engine_direct_rendering():
    engine = ExplainabilityEngine()
    t = datetime(2026, 9, 3, 15, 50, tzinfo=timezone.utc)
    snap = SignalSnapshot(
        timestamp=t,
        spy_price=500.0,
        spy_sma50=495.0,
        spy_sma200=480.0,
        realized_vol_20d=0.10,
        vol_scale_factor=1.0,
        drawdown_pct=-0.01,
        regime=MarketRegime.BULL_AGGRESSIVE,
        indicators={"breadth_50": 0.75, "keltner_lower_band": 490.0},
    )
    target = TargetAllocation(
        timestamp=t,
        regime=MarketRegime.BULL_AGGRESSIVE,
        weights={"QQQ": 0.50, "SPY": 0.50},
        cash_weight=0.0,
        rationale="Strong bull market",
    )
    order = OrderIntent(
        symbol="QQQ",
        action="BUY",
        side=OrderSide.BUY,
        target_weight=0.50,
        current_weight=0.30,
        delta_weight=0.20,
        delta_shares=10.0,
        estimated_price=400.0,
        notional=4000.0,
    )

    t_risk = engine.render_risk_diagnostics_table(snap)
    assert t_risk.row_count == 6

    t_alloc = engine.render_allocation_table(target, {"SHV": 1.0}, 100000.0)
    assert t_alloc.row_count >= 3

    t_plan = engine.render_rebalance_plan_table([order], 100000.0)
    assert t_plan.row_count == 1
