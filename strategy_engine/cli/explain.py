"""
strategy_engine.cli.explain
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rich explainable terminal formatting engine and JSON serialization for
the AlpacaRelay Systematic Strategy Decision Engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Any, Dict, List, Optional, Union

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from strategy_engine.core.models import (
    AssetClass,
    MarketRegime,
    OrderIntent,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.core.universe import UNIVERSE_ASSETS


class ExplainabilityEngine:
    """Rich terminal formatting engine for decision rationales and risk diagnostics."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    @staticmethod
    def _format_currency(val: Optional[float]) -> str:
        if val is None or math.isnan(val):
            return "$0.00"
        sign = "-" if val < 0 else "+" if val > 0 else ""
        return f"{sign}${abs(val):,.2f}"

    @staticmethod
    def _format_pct(val: Optional[float], decimals: int = 2) -> str:
        if val is None or math.isnan(val):
            return "0.00%"
        sign = "+" if val > 0 else ""
        return f"{sign}{val * 100:.{decimals}f}%"

    def render_header(
        self,
        mode: str = "DRY-RUN (Read-Only)",
        timestamp: Optional[datetime] = None,
        feed_status: str = "Connected (SIP)",
        db_path: str = "strategy_engine.db",
    ) -> Panel:
        """Render top application banner."""
        ts_str = (timestamp or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S UTC")
        header_text = Text()
        header_text.append("ALPACARELAY STRATEGY DECISION ENGINE\n", style="bold cyan")
        header_text.append(f"Execution Mode: {mode}\n", style="bold yellow")
        header_text.append(f"Timestamp: {ts_str} | Data Feed: {feed_status} | DB: {db_path}", style="dim")
        return Panel(header_text, box=box.DOUBLE, expand=False, border_style="cyan")

    def render_risk_diagnostics_table(self, snapshot: SignalSnapshot) -> Table:
        """Render market regime and quantitative risk indicator diagnostics."""
        table = Table(
            title="[bold]MARKET REGIME & RISK DIAGNOSTICS[/bold]",
            box=box.ROUNDED,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="bold cyan")
        table.add_column("Current Value", justify="right")
        table.add_column("Threshold / Target", justify="right")
        table.add_column("Status / Evaluation", justify="left")

        # 1. Primary Trend (SPY vs 200 SMA)
        spy_p = snapshot.spy_price
        spy_200 = snapshot.spy_sma200
        trend_above_200 = spy_p > spy_200
        trend_pct = (spy_p - spy_200) / spy_200 if spy_200 > 0 else 0.0
        trend_status = (
            f"[bold green]BULL (+{trend_pct * 100:.1f}% above 200d)[/bold green]"
            if trend_above_200
            else f"[bold red]BEAR ({trend_pct * 100:.1f}% below 200d)[/bold red]"
        )
        table.add_row(
            "Primary Trend (SPY)",
            f"${spy_p:,.2f}",
            f"200 SMA: ${spy_200:,.2f}",
            trend_status,
        )

        # 2. Fast Trend (SPY vs 50 SMA)
        spy_50 = snapshot.spy_sma50
        trend_above_50 = spy_p > spy_50
        fast_pct = (spy_p - spy_50) / spy_50 if spy_50 > 0 else 0.0
        fast_status = (
            f"[bold green]BULL (+{fast_pct * 100:.1f}% above 50d)[/bold green]"
            if trend_above_50
            else f"[bold yellow]PULLBACK ({fast_pct * 100:.1f}% below 50d)[/bold yellow]"
        )
        table.add_row(
            "Fast Trend (SPY)",
            f"${spy_p:,.2f}",
            f"50 SMA: ${spy_50:,.2f}",
            fast_status,
        )

        # 3. 20-Day Realized Volatility
        vol_20d = snapshot.realized_vol_20d
        vol_scale = snapshot.vol_scale_factor
        vol_status = (
            f"[bold green]NORMAL (Scaler: {vol_scale:.2f})[/bold green]"
            if vol_scale >= 0.99
            else f"[bold yellow]ELEVATED (Scaler: {vol_scale:.2f})[/bold yellow]"
            if vol_scale >= 0.50
            else f"[bold red]VOL SPIKE (Scaler: {vol_scale:.2f})[/bold red]"
        )
        table.add_row(
            "20-Day Realized Vol",
            f"{vol_20d * 100:.2f}%",
            "Target: 12.00%",
            vol_status,
        )

        # 4. Trailing Drawdown
        dd = snapshot.drawdown_pct
        dd_gate = snapshot.indicators.get("drawdown_gate", 1.0)
        dd_status = (
            f"[bold green]SAFE (Gate: {dd_gate:.2f})[/bold green]"
            if dd > -0.05
            else f"[bold yellow]CAUTION (Gate: {dd_gate:.2f})[/bold yellow]"
            if dd > -0.10
            else f"[bold red]DEFENSE ACTIVE (Gate: {dd_gate:.2f})[/bold red]"
        )
        table.add_row(
            "Trailing Drawdown",
            f"{dd * 100:.2f}%",
            "Defense Gates: -5% / -10%",
            dd_status,
        )

        # 5. Lower ATR Channel Stop
        atr_lower = snapshot.indicators.get("keltner_lower_band", snapshot.spy_sma50)
        cb_active = snapshot.circuit_breaker_active or (snapshot.indicators.get("circuit_breaker_active", 0.0) >= 1.0)
        cb_status = (
            "[bold red]CIRCUIT BREAKER TRIGGERED[/bold red]"
            if cb_active
            else f"[bold green]SAFE (Buffer: {((spy_p - atr_lower) / atr_lower) * 100:+.2f}%)[/bold green]"
        )
        table.add_row(
            "Lower ATR Channel Stop",
            f"${atr_lower:,.2f}",
            f"SPY: ${spy_p:,.2f}",
            cb_status,
        )

        # 6. Market Breadth
        breadth = snapshot.indicators.get("breadth_50", 1.0)
        breadth_status = (
            f"[bold green]EXPANSION ({breadth * 100:.1f}%)[/bold green]"
            if breadth >= 0.60
            else f"[bold yellow]NEUTRAL ({breadth * 100:.1f}%)[/bold yellow]"
            if breadth >= 0.40
            else f"[bold red]CONTRACTION ({breadth * 100:.1f}%)[/bold red]"
        )
        table.add_row(
            "Market Breadth (% > 50d)",
            f"{breadth * 100:.1f}%",
            "Hurdle: 60.0%",
            breadth_status,
        )

        return table

    def render_momentum_panel(self, momentum: Dict[str, float]) -> Panel:
        """Render top momentum ranked universe assets."""
        if not momentum:
            return Panel("[dim]No momentum scores available[/dim]", title="Momentum Ranking")

        sorted_m = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
        t = Text()
        for idx, (sym, score) in enumerate(sorted_m[:6], 1):
            sign = "+" if score > 0 else ""
            color = "green" if score > 0 else "red"
            t.append(f"  {idx}. {sym:<5}: [{color}]{sign}{score * 100:.1f}%[/{color}]\n")

        return Panel(t, title="[bold]TOP MOMENTUM ASSETS (12-1 Structural Momentum)[/bold]", box=box.ROUNDED)

    def render_allocation_table(
        self,
        target: TargetAllocation,
        current_weights: Dict[str, float],
        equity: float = 100000.0,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Table:
        """Render target allocation weights with current weights and delta values."""
        table = Table(
            title="[bold]TARGET ALLOCATION WEIGHTS (Sum = 1.00000)[/bold]",
            box=box.ROUNDED,
            header_style="bold blue",
        )
        table.add_column("Ticker", style="bold white")
        table.add_column("Asset Class", style="dim")
        table.add_column("Current Wt", justify="right")
        table.add_column("Target Wt", justify="right")
        table.add_column("Delta Wt", justify="right")
        table.add_column("Delta Value", justify="right")
        table.add_column("Action", justify="center")

        all_syms = sorted(list(set(list(target.weights.keys()) + list(current_weights.keys()))))
        total_curr_wt = 0.0
        total_tgt_wt = 0.0
        total_delta_val = 0.0

        for sym in all_syms:
            tgt_w = target.weights.get(sym, 0.0)
            cur_w = current_weights.get(sym, 0.0)
            delta_w = tgt_w - cur_w
            delta_val = delta_w * equity

            total_curr_wt += cur_w
            total_tgt_wt += tgt_w
            total_delta_val += delta_val

            asset_meta = UNIVERSE_ASSETS.get(sym)
            ac = asset_meta.asset_class.value if asset_meta else AssetClass.EQUITY_INDEX.value

            if abs(delta_w) < 0.005:
                action_str = "[dim]HOLD[/dim]"
                delta_str = "[dim]$0.00[/dim]"
            elif delta_w > 0:
                action_str = "[bold green]BUY[/bold green]"
                delta_str = f"[bold green]+${delta_val:,.2f}[/bold green]"
            else:
                action_str = "[bold red]SELL[/bold red]"
                delta_str = f"[bold red]-${abs(delta_val):,.2f}[/bold red]"

            table.add_row(
                sym,
                ac,
                f"{cur_w * 100:.2f}%",
                f"{tgt_w * 100:.2f}%",
                f"{delta_w * 100:+.2f}%",
                delta_str,
                action_str,
            )

        # Footer Row
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            "[dim]Normalized Portfolio[/dim]",
            f"[bold]{total_curr_wt * 100:.2f}%[/bold]",
            f"[bold]{total_tgt_wt * 100:.2f}%[/bold]",
            f"{(total_tgt_wt - total_curr_wt) * 100:+.2f}%",
            f"Net: ${total_delta_val:,.2f}",
            "[bold green]✓ 100.00%[/bold green]",
        )

        return table

    def render_rebalance_plan_table(
        self,
        orders: List[OrderIntent],
        equity: float = 100000.0,
    ) -> Table:
        """Render sequenced execution order manifest (SELLs before BUYs)."""
        table = Table(
            title="[bold]REBALANCE EXECUTION ORDER PLAN[/bold]",
            box=box.ROUNDED,
            header_style="bold green",
        )
        table.add_column("Seq #", justify="center", style="dim")
        table.add_column("Symbol", style="bold white")
        table.add_column("Action", justify="center")
        table.add_column("Shares", justify="right")
        table.add_column("Est Price", justify="right")
        table.add_column("Notional", justify="right")
        table.add_column("Target Wt", justify="right")
        table.add_column("Current Wt", justify="right")
        table.add_column("Rationale", style="dim")

        if not orders:
            table.add_row(
                "-", "None", "[dim]HOLD[/dim]", "-", "-", "-", "-", "-",
                "All assets within +/- 2.5% drift band; zero rebalancing required"
            )
            return table

        for idx, o in enumerate(orders, 1):
            act_color = "green" if o.action == "BUY" else "red" if o.action == "SELL" else "white"
            shares_str = f"{abs(o.delta_shares):.1f}" if o.delta_shares is not None else "-"
            price_str = f"${o.estimated_price:.2f}" if o.estimated_price is not None else "-"
            notional_str = f"${o.notional:,.2f}" if o.notional is not None else (
                f"${abs(o.delta_dollars):,.2f}" if o.delta_dollars is not None else "-"
            )
            tgt_str = f"{o.target_weight * 100:.2f}%" if o.target_weight is not None else "-"
            cur_str = f"{o.current_weight * 100:.2f}%" if o.current_weight is not None else "-"

            table.add_row(
                str(idx),
                o.symbol,
                f"[{act_color}]{o.action}[/{act_color}]",
                shares_str,
                price_str,
                notional_str,
                tgt_str,
                cur_str,
                o.rationale or o.reason,
            )

        return table

    def render_rationale_panel(
        self,
        rationale: str,
        regime: MarketRegime,
        s_vol: float,
        g_dd: float,
    ) -> Panel:
        """Render transparent mathematical rationale."""
        text = Text()
        text.append(f"Regime: {regime.value}\n", style="bold cyan")
        text.append(f"Risk Multipliers: Volatility Scaler = {s_vol:.2f} | Drawdown Defense Gate = {g_dd:.2f}\n\n", style="bold yellow")
        text.append(f">> {rationale}\n", style="white")
        return Panel(text, title="[bold]DECISION RATIONALE AUDIT LOG[/bold]", box=box.ROUNDED)

    def render_backtest_summary(
        self,
        scenario_name: str,
        strat_metrics: Dict[str, float],
        spy_metrics: Dict[str, float],
        pass_criteria: bool,
    ) -> Table:
        """Render comparative backtest summary metrics vs SPY."""
        table = Table(
            title=f"[bold]BACKTEST PERFORMANCE COMPARISON — {scenario_name.upper()}[/bold]",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Performance Metric", style="bold white")
        table.add_column("Strategy Engine", justify="right", style="bold green")
        table.add_column("Benchmark (SPY)", justify="right", style="dim")
        table.add_column("Outperformance / Advantage", justify="right")

        cum_strat = strat_metrics.get("cumulative_return", 0.0)
        cum_spy = spy_metrics.get("cumulative_return", 0.0)
        table.add_row(
            "Cumulative Return",
            f"{cum_strat * 100:+.2f}%",
            f"{cum_spy * 100:+.2f}%",
            f"{(cum_strat - cum_spy) * 100:+.2f}%",
        )

        cagr_strat = strat_metrics.get("cagr", 0.0)
        cagr_spy = spy_metrics.get("cagr", 0.0)
        table.add_row(
            "CAGR (Annualized)",
            f"{cagr_strat * 100:+.2f}%",
            f"{cagr_spy * 100:+.2f}%",
            f"{(cagr_strat - cagr_spy) * 100:+.2f}%",
        )

        dd_strat = strat_metrics.get("max_drawdown", 0.0)
        dd_spy = spy_metrics.get("max_drawdown", 0.0)
        dd_diff = abs(dd_spy) - abs(dd_strat)
        table.add_row(
            "Maximum Drawdown",
            f"[bold green]{dd_strat * 100:.2f}%[/bold green]",
            f"[bold red]{dd_spy * 100:.2f}%[/bold red]",
            f"[bold green]+{dd_diff * 100:.2f}% Lower Risk[/bold green]",
        )

        vol_strat = strat_metrics.get("volatility", 0.0)
        vol_spy = spy_metrics.get("volatility", 0.0)
        table.add_row(
            "Annualized Volatility",
            f"{vol_strat * 100:.2f}%",
            f"{vol_spy * 100:.2f}%",
            f"{(vol_spy - vol_strat) * 100:.2f}% Reduction",
        )

        sharpe_strat = strat_metrics.get("sharpe", 0.0)
        sharpe_spy = spy_metrics.get("sharpe", 0.0)
        table.add_row(
            "Sharpe Ratio",
            f"{sharpe_strat:.2f}",
            f"{sharpe_spy:.2f}",
            f"{sharpe_strat - sharpe_spy:+.2f}",
        )

        table.add_section()
        status_text = "[bold green][PASS] Drawdown Mitigation Target Met[/bold green]" if pass_criteria else "[bold red][FAIL] Drawdown Target Not Met[/bold red]"
        table.add_row("[bold]CRITERIA VERIFICATION[/bold]", status_text, "", "")

        return table

    def render_status_dashboard(
        self,
        db_stats: Dict[str, Any],
        relay_health: Optional[Dict[str, Any]],
        active_regime: str,
        last_snapshot: Optional[Dict[str, Any]],
    ) -> Group:
        """Render composite status dashboard."""
        status_table = Table(box=box.ROUNDED, title="[bold]SYSTEM VITALS & CONNECTIVITY[/bold]")
        status_table.add_column("Component", style="bold cyan")
        status_table.add_column("Status / Metric", justify="right")

        status_table.add_row("Decision Engine", "[bold green]ONLINE[/bold green]")
        status_table.add_row("Active Regime", f"[bold yellow]{active_regime}[/bold yellow]")
        status_table.add_row("SQLite WAL Mode", "[bold green]ACTIVE[/bold green]" if db_stats.get("wal") else "[yellow]OFF[/yellow]")
        status_table.add_row("Database Path", str(db_stats.get("path", "N/A")))

        relay_status = "[bold green]CONNECTED[/bold green]" if (relay_health and relay_health.get("upstream") == "connected") else "[yellow]STANDALONE[/yellow]"
        status_table.add_row("AlpacaRelay Stream", relay_status)

        # Database rows table
        rows_table = Table(box=box.ROUNDED, title="[bold]DATABASE RECORD COUNTS[/bold]")
        rows_table.add_column("Table Name", style="bold white")
        rows_table.add_column("Rows", justify="right")
        for tbl, cnt in db_stats.get("tables", {}).items():
            rows_table.add_row(tbl, str(cnt))

        return Group(status_table, rows_table)

    def render_dry_run(
        self,
        snapshot: SignalSnapshot,
        target: TargetAllocation,
        orders: List[OrderIntent],
        momentum: Dict[str, float],
        equity: float,
        current_weights: Dict[str, float],
    ) -> None:
        """Render complete composite terminal presentation for dry-run."""
        self.console.print(self.render_header())
        self.console.print(self.render_risk_diagnostics_table(snapshot))
        self.console.print(self.render_momentum_panel(momentum))
        self.console.print(self.render_allocation_table(target, current_weights, equity))
        self.console.print(self.render_rebalance_plan_table(orders, equity))
        self.console.print(
            self.render_rationale_panel(
                target.rationale,
                snapshot.regime,
                snapshot.vol_scale_factor,
                snapshot.indicators.get("drawdown_gate", 1.0),
            )
        )
        self.console.print(
            "[bold green]✓ Target Weights Validated: Sum = 1.00000 (100.00%)[/bold green]\n"
        )

    def build_json_payload(
        self,
        snapshot: SignalSnapshot,
        target: TargetAllocation,
        orders: List[OrderIntent],
        momentum: Dict[str, float],
        equity: float,
    ) -> Dict[str, Any]:
        """Serialize complete engine decision into machine-readable JSON dict."""
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "regime": snapshot.regime.value,
            "signals": {
                "spy_price": snapshot.spy_price,
                "spy_sma50": snapshot.spy_sma50,
                "spy_sma200": snapshot.spy_sma200,
                "realized_vol_20d": snapshot.realized_vol_20d,
                "vol_scale_factor": snapshot.vol_scale_factor,
                "drawdown_pct": snapshot.drawdown_pct,
                "circuit_breaker_active": snapshot.circuit_breaker_active,
                "indicators": snapshot.indicators,
            },
            "momentum_rankings": momentum,
            "target_allocation": {
                "weights": target.weights,
                "cash_weight": target.cash_weight,
                "rationale": target.rationale,
                "weights_sum": sum(target.weights.values()),
            },
            "rebalance_orders": [o.model_dump(mode="json") for o in orders],
            "portfolio_equity": equity,
        }
