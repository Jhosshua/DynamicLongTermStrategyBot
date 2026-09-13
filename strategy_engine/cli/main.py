"""
strategy_engine.cli.main
~~~~~~~~~~~~~~~~~~~~~~~~

Production Typer CLI application for the AlpacaRelay Strategy Engine.
Supports dry-run, daemon, rebalance, backtest, status, and export-metrics.
"""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
import io
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import urllib.request

import numpy as np
import typer

from strategy_engine.allocator.rebalancer import PortfolioRebalancer
from strategy_engine.cli.explain import ExplainabilityEngine
from strategy_engine.core.models import (
    Bar,
    MarketRegime,
    OrderIntent,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.daemon.daemon import DaemonConfig, DecisionDaemon
from strategy_engine.daemon.scheduler import MarketCalendar
from strategy_engine.signals.indicators import filter_bars_point_in_time
from strategy_engine.signals.regime_detector import SignalEngine
from strategy_engine.simulator.stress_scenarios import (
    StressScenarioType,
    generate_stress_scenario,
)
from strategy_engine.storage.database import Database
from strategy_engine.storage.repositories import StorageService

logger = logging.getLogger("strategy_engine.cli")

app = typer.Typer(
    name="strategy-engine",
    help="AlpacaRelay Systematic US Equity Strategy Engine CLI",
    add_completion=False,
)

explain = ExplainabilityEngine()


def _parse_interval_str(interval: str) -> float:
    """Parse interval string (e.g. '60s', '1.0', '5m') to float seconds."""
    try:
        clean = interval.strip().lower()
        if clean.endswith("s"):
            val = float(clean[:-1])
        elif clean.endswith("m"):
            val = float(clean[:-1]) * 60.0
        elif clean.endswith("h"):
            val = float(clean[:-1]) * 3600.0
        else:
            val = float(clean)
        if val <= 0 or math.isnan(val) or math.isinf(val):
            raise ValueError(f"Interval must be a positive finite number, got {val}")
        return val
    except Exception as e:
        typer.secho(f"Error: Invalid interval '{interval}': {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


# ============================================================================
# Subcommand 1: dry-run
# ============================================================================
@app.command("dry-run")
def dry_run(
    scenario: str = typer.Option("2008", "--scenario", "-s", help="Stress scenario ('2008', '2020', '2022', '2017', 'none')"),
    as_of: Optional[str] = typer.Option(None, "--as-of", help="Point-in-time ISO timestamp"),
    equity: float = typer.Option(100000.0, "--equity", "-e", help="Portfolio total NAV in USD"),
    current_weights: Optional[str] = typer.Option(None, "--current-weights", "-w", help="JSON string of current portfolio weights"),
    db_path: str = typer.Option("strategy_engine.db", "--db-path", help="Path to SQLite database"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
    symbols: Optional[str] = typer.Option(None, "--symbols", help="Comma-separated symbols list"),
    simulate: bool = typer.Option(False, "--simulate", help="Use synthetic simulation data"),
):
    """Evaluate signals, classify regime, and output allocation & rebalancing plan without trades."""
    try:
        # Validate equity
        if equity <= 0.0 or math.isnan(equity) or math.isinf(equity):
            typer.secho(
                f"Error: Portfolio equity must be a positive finite number (got {equity})",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        # 1. Acquire Market Data
        market_data: Dict[str, List[Bar]] = {}

        if scenario != "none" or simulate:
            scen_name = scenario if scenario != "none" else "2008"
            try:
                ds = generate_stress_scenario(scen_name, seed=42)
                market_data = ds.bars
            except Exception as e:
                typer.secho(f"Error: Invalid scenario '{scenario}': {e}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)
        else:
            # Load from SQLite database
            if not db_path or not Path(db_path).exists():
                typer.secho(f"Error: Database path '{db_path}' does not exist", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)

            try:
                db = Database(db_path, auto_init=False)
                storage = StorageService(db)
                syms = [s.strip() for s in symbols.split(",")] if symbols else storage.bars.get_symbols()
                for s in syms:
                    bars = storage.bars.get_bars(s)
                    if bars:
                        market_data[s] = bars
            except Exception as e:
                typer.secho(f"Error reading database '{db_path}': {e}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)

            if not market_data or "SPY" not in market_data:
                typer.secho("Error: No SPY bar data found in database.", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)

        # 2. Determine evaluation timestamp
        if as_of:
            try:
                eval_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
                if eval_dt.tzinfo is None:
                    eval_dt = eval_dt.replace(tzinfo=timezone.utc)
            except Exception as e:
                typer.secho(f"Error: Invalid --as-of timestamp '{as_of}': {e}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)
        else:
            # Use latest timestamp of SPY bars
            eval_dt = market_data["SPY"][-1].timestamp

        # 3. Compute Signals, Momentum & Target Allocation
        engine = SignalEngine()
        snapshot = engine.compute_daily_signals(market_data=market_data, current_time=eval_dt)
        momentum = engine.compute_monthly_momentum(market_data=market_data, current_time=eval_dt)
        target = engine.compute_target_weights(signals=snapshot, momentum=momentum, market_data=market_data)

        # 4. Parse Current Portfolio Weights & Rebalance Orders
        curr_w: Dict[str, float] = {}
        if current_weights:
            try:
                parsed = json.loads(current_weights)
                if not isinstance(parsed, dict):
                    typer.secho("Error: --current-weights must be a JSON dictionary", err=True, fg=typer.colors.RED)
                    raise typer.Exit(code=1)
                curr_w = {str(k): float(v) for k, v in parsed.items()}
            except typer.Exit:
                raise
            except Exception as e:
                typer.secho(f"Error parsing --current-weights '{current_weights}': {e}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)
        else:
            curr_w = {"SHV": 1.0}

        # Extract current prices
        curr_prices: Dict[str, float] = {}
        for sym, b_list in market_data.items():
            pit_b = filter_bars_point_in_time(b_list, eval_dt)
            if pit_b:
                curr_prices[sym] = pit_b[-1].close

        rebalancer = PortfolioRebalancer(drift_band=0.025, min_order_threshold=0.005)
        orders = rebalancer.compute_rebalance_orders(
            target_allocation=target,
            current_weights=curr_w,
            portfolio_equity=equity,
            current_prices=curr_prices,
            timestamp=eval_dt,
        )

        # 5. Output Presentation
        if json_output:
            payload = explain.build_json_payload(
                snapshot=snapshot,
                target=target,
                orders=orders,
                momentum=momentum,
                equity=equity,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            explain.render_dry_run(
                snapshot=snapshot,
                target=target,
                orders=orders,
                momentum=momentum,
                equity=equity,
                current_weights=curr_w,
            )
    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"Error during dry-run: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


# ============================================================================
# Subcommand 2: daemon
# ============================================================================
@app.command("daemon")
def daemon_cmd(
    relay_url: str = typer.Option("https://alpacarelay-production.up.railway.app", "--relay-url", help="AlpacaRelay base URL"),
    ws_url: str = typer.Option("wss://alpacarelay-production.up.railway.app", "--ws-url", help="AlpacaRelay WebSocket URL"),
    token: str = typer.Option("", "--token", envvar="RELAY_TOKEN", help="Relay auth token"),
    db_path: str = typer.Option("strategy_engine.db", "--db-path", help="Path to SQLite DB"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Dry run mode (no live trades)"),
    once: bool = typer.Option(False, "--once", help="Execute single evaluation tick and exit"),
    interval: str = typer.Option("1.0", "--interval", help="Tick interval in seconds (e.g. '1.0', '60s')"),
    intraday_check: bool = typer.Option(True, "--intraday-check/--no-intraday-check", help="Enable intraday circuit breaker"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON status"),
):
    """Start the production decision daemon with market calendar scheduling."""
    try:
        interval_seconds = _parse_interval_str(interval)

        config = DaemonConfig(
            relay_base_url=relay_url,
            ws_url=ws_url,
            relay_token=token,
            db_path=db_path,
            dry_run=dry_run,
            tick_interval_seconds=interval_seconds,
        )

        daemon = DecisionDaemon(config=config)

        if once:
            async def run_once():
                # Warm up historical bars
                await daemon._warmup_historical_bars()
                # Execute single evaluation tick
                now = datetime.now(timezone.utc)
                await daemon._handle_daily_close(now)
                await daemon.shutdown("Run once completed")

            asyncio.run(run_once())
            if json_output:
                print(json.dumps({"status": "completed", "mode": "once", "timestamp": datetime.now(timezone.utc).isoformat()}))
            else:
                typer.secho("Single daemon evaluation completed successfully.", fg=typer.colors.GREEN)
            return

        if not json_output:
            typer.secho(f"Starting Production Decision Daemon (interval: {interval_seconds}s, dry_run: {dry_run})...", fg=typer.colors.CYAN)

        try:
            asyncio.run(daemon.run())
        except KeyboardInterrupt:
            if not json_output:
                typer.secho("\nDaemon shutdown requested by operator.", fg=typer.colors.YELLOW)
    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"Error starting daemon: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


# ============================================================================
# Subcommand 3: rebalance
# ============================================================================
@app.command("rebalance")
def rebalance_cmd(
    equity: float = typer.Option(100000.0, "--equity", "-e", help="Total portfolio equity ($)"),
    current_weights: Optional[str] = typer.Option(None, "--current-weights", "-w", help="JSON current weights"),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebalance regardless of drift band"),
    scenario: str = typer.Option("2008", "--scenario", "-s", help="Scenario for target weights"),
    db_path: str = typer.Option("strategy_engine.db", "--db-path", help="Path to SQLite DB"),
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Execution mode"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON manifest"),
):
    """Evaluate current portfolio against target allocation, enforcing +/- 2.5% drift bands."""
    try:
        # Validate equity
        if equity <= 0.0 or math.isnan(equity) or math.isinf(equity):
            typer.secho(
                f"Error: Portfolio equity must be a positive finite number (got {equity})",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        # 1. Load market data
        try:
            ds = generate_stress_scenario(scenario, seed=42)
            market_data = ds.bars
        except Exception as e:
            typer.secho(f"Error: Invalid scenario '{scenario}': {e}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

        eval_dt = market_data["SPY"][-1].timestamp

        # 2. Compute signals & target allocation
        engine = SignalEngine()
        snapshot = engine.compute_daily_signals(market_data=market_data, current_time=eval_dt)
        target = engine.compute_target_weights(signals=snapshot, market_data=market_data)

        # 3. Parse current portfolio
        curr_w: Dict[str, float] = {}
        if current_weights:
            try:
                parsed = json.loads(current_weights)
                if not isinstance(parsed, dict):
                    typer.secho("Error: --current-weights must be a JSON dictionary", err=True, fg=typer.colors.RED)
                    raise typer.Exit(code=1)
                curr_w = {str(k): float(v) for k, v in parsed.items()}
            except typer.Exit:
                raise
            except Exception as e:
                typer.secho(f"Error parsing --current-weights: {e}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)
        else:
            curr_w = {"SHV": 1.0}

        # Extract prices
        curr_prices: Dict[str, float] = {
            sym: b_list[-1].close for sym, b_list in market_data.items() if b_list
        }

        drift_band = 0.0 if force else 0.025
        rebalancer = PortfolioRebalancer(drift_band=drift_band, min_order_threshold=0.005)
        orders = rebalancer.compute_rebalance_orders(
            target_allocation=target,
            current_weights=curr_w,
            portfolio_equity=equity,
            current_prices=curr_prices,
            timestamp=eval_dt,
        )

        if json_output:
            res = {
                "timestamp": eval_dt.isoformat(),
                "regime": target.regime.value,
                "force": force,
                "target_weights": target.weights,
                "current_weights": curr_w,
                "orders": [o.model_dump(mode="json") for o in orders],
                "orders_count": len(orders),
            }
            print(json.dumps(res, indent=2))
        else:
            explain.console.print(explain.render_header(mode="REBALANCE EVALUATION", timestamp=eval_dt))
            explain.console.print(explain.render_allocation_table(target, curr_w, equity))
            explain.console.print(explain.render_rebalance_plan_table(orders, equity))
            if not orders:
                typer.secho("✓ Portfolio is within drift tolerance (+/- 2.5%). No rebalancing required.", fg=typer.colors.GREEN)
            else:
                typer.secho(f"✓ Generated {len(orders)} rebalancing orders (SELLs sequenced before BUYs).", fg=typer.colors.GREEN)
    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"Error during rebalance: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


# ============================================================================
# Subcommand 4: backtest
# ============================================================================
@app.command("backtest")
def backtest_cmd(
    scenario: str = typer.Option("all", "--scenario", "-s", help="Scenario ('all', '2008', '2020', '2022', '2017')"),
    capital: float = typer.Option(100000.0, "--capital", "-c", help="Initial capital in USD"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON metrics"),
    export_csv: Optional[str] = typer.Option(None, "--export-csv", help="Filepath to export backtest CSV"),
):
    """Simulate strategy across historical stress scenarios and benchmark vs SPY."""
    try:
        # Validate capital
        if capital <= 0.0 or math.isnan(capital) or math.isinf(capital):
            typer.secho(
                f"Error: Initial capital must be a positive finite number (got {capital})",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        scenarios_to_run = (
            ["2008", "2020", "2022", "2017"]
            if scenario.lower() in ("all", "all_scenarios")
            else [scenario]
        )

        all_results: Dict[str, Any] = {}

        for scen in scenarios_to_run:
            try:
                ds = generate_stress_scenario(scen, seed=42)
            except Exception as e:
                typer.secho(f"Error: Invalid scenario '{scen}': {e}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)

            market_data = ds.bars
            spy_bars = market_data["SPY"]
            n_bars = len(spy_bars)

            # Day-by-day simulation loop
            strat_equity = [capital]
            spy_equity = [capital]
            strat_returns = []
            spy_returns = []

            engine = SignalEngine()
            current_alloc: Optional[TargetAllocation] = None

            # Warmup period of 30 bars minimum
            warmup_len = min(30, n_bars // 4)
            for i in range(warmup_len, n_bars):
                t = spy_bars[i].timestamp
                pit_data = {
                    s: filter_bars_point_in_time(b_list, t)
                    for s, b_list in market_data.items()
                }
                signals = engine.compute_daily_signals(pit_data, current_time=t)
                current_alloc = engine.compute_target_weights(signals, market_data=pit_data)

                # Asset returns on day i vs day i-1
                day_return_strat = 0.0
                for sym, wt in current_alloc.weights.items():
                    if sym in pit_data and len(pit_data[sym]) >= 2:
                        p_now = pit_data[sym][-1].close
                        p_prev = pit_data[sym][-2].close
                        ret = (p_now - p_prev) / p_prev if p_prev > 0 else 0.0
                        day_return_strat += wt * ret
                    elif sym in ("SHV", "BIL", "CASH"):
                        # Cash proxy yield: ~4% annual = ~0.015% per day
                        day_return_strat += wt * (0.04 / 252.0)

                # SPY return
                spy_ret = (spy_bars[i].close - spy_bars[i - 1].close) / spy_bars[i - 1].close
                strat_returns.append(day_return_strat)
                spy_returns.append(spy_ret)

                strat_equity.append(strat_equity[-1] * (1.0 + day_return_strat))
                spy_equity.append(spy_equity[-1] * (1.0 + spy_ret))

            # Metrics calculation
            strat_arr = np.array(strat_equity)
            spy_arr = np.array(spy_equity)
            strat_ret_arr = np.array(strat_returns)
            spy_ret_arr = np.array(spy_returns)

            cum_strat = (strat_arr[-1] / strat_arr[0]) - 1.0
            cum_spy = (spy_arr[-1] / spy_arr[0]) - 1.0

            n_days = len(strat_returns)
            cagr_strat = ((strat_arr[-1] / strat_arr[0]) ** (252.0 / max(1, n_days))) - 1.0 if strat_arr[-1] > 0 else -1.0
            cagr_spy = ((spy_arr[-1] / spy_arr[0]) ** (252.0 / max(1, n_days))) - 1.0 if spy_arr[-1] > 0 else -1.0

            # Drawdowns
            strat_peaks = np.maximum.accumulate(strat_arr)
            strat_dds = (strat_arr - strat_peaks) / strat_peaks
            max_dd_strat = float(np.min(strat_dds))

            spy_peaks = np.maximum.accumulate(spy_arr)
            spy_dds = (spy_arr - spy_peaks) / spy_peaks
            max_dd_spy = float(np.min(spy_dds))

            vol_strat = float(np.std(strat_ret_arr) * np.sqrt(252.0)) if len(strat_ret_arr) > 1 else 0.0
            vol_spy = float(np.std(spy_ret_arr) * np.sqrt(252.0)) if len(spy_ret_arr) > 1 else 0.0

            sharpe_strat = (cagr_strat / vol_strat) if vol_strat > 0 else 0.0
            sharpe_spy = (cagr_spy / vol_spy) if vol_spy > 0 else 0.0

            # Verification criteria: strategy max drawdown must be lower than SPY
            pass_criteria = abs(max_dd_strat) < abs(max_dd_spy)
            if "2008" in scen:
                pass_criteria = pass_criteria and abs(max_dd_strat) <= 0.15
            elif "2020" in scen:
                pass_criteria = pass_criteria and abs(max_dd_strat) <= 0.10
            elif "2022" in scen:
                pass_criteria = pass_criteria and abs(max_dd_strat) <= 0.07

            res = {
                "scenario": scen,
                "description": ds.description,
                "strategy": {
                    "cumulative_return": cum_strat,
                    "cagr": cagr_strat,
                    "max_drawdown": max_dd_strat,
                    "volatility": vol_strat,
                    "sharpe": sharpe_strat,
                },
                "benchmark_spy": {
                    "cumulative_return": cum_spy,
                    "cagr": cagr_spy,
                    "max_drawdown": max_dd_spy,
                    "volatility": vol_spy,
                    "sharpe": sharpe_spy,
                },
                "criteria_passed": pass_criteria,
            }
            all_results[scen] = res

            if not json_output:
                table = explain.render_backtest_summary(
                    scenario_name=scen,
                    strat_metrics=res["strategy"],
                    spy_metrics=res["benchmark_spy"],
                    pass_criteria=pass_criteria,
                )
                explain.console.print(table)

        if export_csv:
            try:
                out_path = Path(export_csv)
                if not out_path.parent.exists():
                    raise FileNotFoundError(f"Parent directory '{out_path.parent}' does not exist")
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["scenario", "strat_cum_ret", "spy_cum_ret", "strat_max_dd", "spy_max_dd", "strat_cagr", "strat_sharpe", "pass"])
                    for sc, r in all_results.items():
                        writer.writerow([
                            sc,
                            r["strategy"]["cumulative_return"],
                            r["benchmark_spy"]["cumulative_return"],
                            r["strategy"]["max_drawdown"],
                            r["benchmark_spy"]["max_drawdown"],
                            r["strategy"]["cagr"],
                            r["strategy"]["sharpe"],
                            r["criteria_passed"],
                        ])
                if not json_output:
                    typer.secho(f"Exported backtest metrics to CSV: {export_csv}", fg=typer.colors.GREEN)
            except typer.Exit:
                raise
            except Exception as e:
                typer.secho(f"Error exporting CSV to '{export_csv}': {e}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)

        if json_output:
            print(json.dumps(all_results, indent=2))
    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"Error during backtest: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


# ============================================================================
# Subcommand 5: status
# ============================================================================
@app.command("status")
def status_cmd(
    db_path: str = typer.Option("strategy_engine.db", "--db-path", help="Path to SQLite DB"),
    relay_url: str = typer.Option("https://alpacarelay-production.up.railway.app", "--relay-url", help="AlpacaRelay base URL"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose details"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON status"),
):
    """Display system vitals, active regime, database counts, and relay health."""
    try:
        db_stats: Dict[str, Any] = {"path": db_path, "exists": False, "wal": False, "tables": {}}
        latest_snap = None
        latest_alloc = None
        active_regime = MarketRegime.BULL_NORMAL.value

        p = Path(db_path)
        if p.exists():
            db_stats["exists"] = True
            try:
                db = Database(db_path, auto_init=False)
                with db.transaction() as conn:
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA journal_mode;")
                    db_stats["wal"] = cursor.fetchone()[0].upper() == "WAL"

                    tables = [
                        "signal_snapshots", "allocations", "rebalance_orders",
                        "portfolio_states", "regime_events", "market_bars",
                    ]
                    for tbl in tables:
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {tbl};")
                            db_stats["tables"][tbl] = cursor.fetchone()[0]
                        except Exception:
                            db_stats["tables"][tbl] = 0

                if db_stats["tables"].get("signal_snapshots", 0) > 0:
                    storage = StorageService(db)
                    latest_snap = storage.signals.get_latest()
                    latest_alloc = storage.allocations.get_latest()
                    if latest_snap:
                        active_regime = latest_snap["regime"]
            except Exception as e:
                logger.debug("Error reading DB status: %s", e)

        # Relay health probe
        relay_health: Optional[Dict[str, Any]] = None
        try:
            req = urllib.request.Request(f"{relay_url}/health", headers={"User-Agent": "StrategyEngine/1.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    relay_health = json.loads(resp.read().decode("utf-8"))
        except Exception:
            relay_health = {"status": "unreachable", "upstream": "disconnected"}

        if json_output:
            status_payload = {
                "service": "strategy_engine",
                "status": "active",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "regime": active_regime,
                "database": db_stats,
                "relay_health": relay_health,
                "latest_snapshot": latest_snap,
                "latest_allocation": latest_alloc,
            }
            print(json.dumps(status_payload, indent=2))
        else:
            dashboard = explain.render_status_dashboard(
                db_stats=db_stats,
                relay_health=relay_health,
                active_regime=active_regime,
                last_snapshot=latest_snap,
            )
            explain.console.print(dashboard)
    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"Error getting status: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


# ============================================================================
# Subcommand 6: export-metrics
# ============================================================================
@app.command("export-metrics")
def export_metrics_cmd(
    format: str = typer.Option("json", "--format", help="'json' or 'csv'"),
    table: str = typer.Option("signals", "--table", help="'bars', 'regimes', 'signals', 'allocations', 'decisions', or 'orders'"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Filepath or stdout"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="Start ISO timestamp"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="End ISO timestamp"),
    db_path: str = typer.Option("strategy_engine.db", "--db-path", help="Path to SQLite DB"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON to stdout"),
):
    """Export historical signal snapshots, allocations, and order history to JSON/CSV."""
    try:
        ALLOWED_TABLES = {"bars", "regimes", "signals", "allocations", "decisions", "orders"}
        if table.lower() not in ALLOWED_TABLES:
            typer.secho(
                f"Error: Invalid table '{table}'. Allowed tables are: {', '.join(sorted(ALLOWED_TABLES))}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        ALLOWED_FORMATS = {"json", "csv"}
        if format.lower() not in ALLOWED_FORMATS:
            typer.secho(
                f"Error: Invalid format '{format}'. Allowed formats are: {', '.join(sorted(ALLOWED_FORMATS))}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        if not db_path or not Path(db_path).exists():
            typer.secho(f"Error: Database file not found at '{db_path}'", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

        try:
            db = Database(db_path, auto_init=False)
            storage = StorageService(db)
        except Exception as e:
            typer.secho(f"Error accessing database '{db_path}': {e}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

        start_dt = None
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            except Exception as e:
                typer.secho(f"Error: Invalid --start-date '{start_date}': {e}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)

        end_dt = None
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            except Exception as e:
                typer.secho(f"Error: Invalid --end-date '{end_date}': {e}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)

        data: Dict[str, Any] = {}
        tbl = table.lower()
        try:
            if tbl == "signals":
                data["signals"] = storage.signals.get_range(start=start_dt, end=end_dt)
            elif tbl == "allocations":
                data["allocations"] = storage.allocations.get_range(start=start_dt, end=end_dt)
            elif tbl == "decisions":
                data["decisions"] = storage.allocations.get_range(start=start_dt, end=end_dt)
            elif tbl == "orders":
                data["orders"] = storage.orders.get_range(start=start_dt, end=end_dt)
            elif tbl == "regimes":
                data["regimes"] = storage.regimes.get_events(start=start_dt, end=end_dt)
            elif tbl == "bars":
                query = "SELECT * FROM market_bars"
                conds = []
                params = []
                if start_dt:
                    conds.append("timestamp >= ?")
                    params.append(start_dt.isoformat())
                if end_dt:
                    conds.append("timestamp <= ?")
                    params.append(end_dt.isoformat())
                if conds:
                    query += f" WHERE {' AND '.join(conds)}"
                query += " ORDER BY timestamp ASC"
                with storage.db.transaction() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    data["bars"] = [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            typer.secho(f"Error reading database tables from '{db_path}': {e}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

        # Serialize output
        out_content = ""
        fmt = "json" if json_output else format.lower()

        if fmt == "json":
            out_content = json.dumps(data.get(tbl, []), indent=2, sort_keys=True, default=str)
        else:  # CSV format
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            rows = data.get(tbl, [])
            if rows:
                headers = list(rows[0].keys())
                writer.writerow(headers)
                for r in rows:
                    writer.writerow([r.get(h, "") for h in headers])
            else:
                writer.writerow(["timestamp", "value", "status"])
            out_content = csv_buffer.getvalue()

        if output and output != "-":
            try:
                out_path = Path(output)
                if not out_path.parent.exists():
                    raise FileNotFoundError(f"Parent directory '{out_path.parent}' does not exist")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(out_content)
                if not json_output:
                    typer.secho(f"Exported metrics to {output} ({len(out_content)} bytes).", fg=typer.colors.GREEN)
            except typer.Exit:
                raise
            except Exception as e:
                typer.secho(f"Error writing to output '{output}': {e}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)
        else:
            print(out_content)
    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"Error during export-metrics: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
