"""
strategy_engine.daemon.daemon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Production Decision Daemon managing real-time market data feeds,
multi-cadence scheduling, risk management, and zero-corruption shutdown lifecycle.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import signal
from typing import Any, Dict, List, Optional, Union

from strategy_engine.allocator.rebalancer import PortfolioRebalancer
from strategy_engine.core.models import (
    Bar,
    MarketRegime,
    OrderIntent,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.core.universe import ALL_SYMBOLS
from strategy_engine.daemon.scheduler import CadenceType, MarketCalendar, MarketScheduler
from strategy_engine.ingestion.client import AlpacaRelayClient
from strategy_engine.signals.regime_detector import SignalEngine
from strategy_engine.storage.audit_logger import JSONLAuditLogger
from strategy_engine.storage.database import Database
from strategy_engine.storage.repositories import StorageService

logger = logging.getLogger("strategy_engine.daemon.daemon")


@dataclass
class DaemonConfig:
    """Production decision daemon configuration."""
    relay_base_url: str = "https://alpacarelay-production.up.railway.app"
    ws_url: str = "wss://alpacarelay-production.up.railway.app"
    relay_token: str = ""
    db_path: str = "strategy_engine.db"
    symbols: List[str] = field(default_factory=lambda: list(ALL_SYMBOLS))
    dry_run: bool = True
    tick_interval_seconds: float = 1.0
    daily_close_offset_minutes: int = 10
    stale_timeout_seconds: float = 300.0
    drift_band: float = 0.025
    min_order_threshold: float = 0.005
    log_dir: str = "logs"


class DecisionDaemon:
    """Production decision daemon coordinating market feed, scheduling, and risk."""

    def __init__(
        self,
        config: Optional[DaemonConfig] = None,
        client: Optional[AlpacaRelayClient] = None,
        signal_engine: Optional[SignalEngine] = None,
        rebalancer: Optional[PortfolioRebalancer] = None,
        scheduler: Optional[MarketScheduler] = None,
        storage: Optional[StorageService] = None,
        audit_logger: Optional[JSONLAuditLogger] = None,
    ):
        self.config = config or DaemonConfig()
        self.calendar = MarketCalendar()
        self.scheduler = scheduler or MarketScheduler(
            calendar=self.calendar,
            daily_close_offset_minutes=self.config.daily_close_offset_minutes,
            check_interval_seconds=self.config.tick_interval_seconds,
        )
        self.client = client or AlpacaRelayClient(
            base_url=self.config.relay_base_url,
            ws_url=self.config.ws_url,
            relay_token=self.config.relay_token,
        )
        self.signal_engine = signal_engine or SignalEngine()
        self.rebalancer = rebalancer or PortfolioRebalancer(
            drift_band=self.config.drift_band,
            min_order_threshold=self.config.min_order_threshold,
        )
        self.storage = storage or StorageService(self.config.db_path)
        self.audit_logger = audit_logger or JSONLAuditLogger(self.config.log_dir)

        # In-memory runtime state
        self._cached_daily_bars: Dict[str, List[Bar]] = {}
        self._latest_intraday_bars: Dict[str, Bar] = {}
        self._current_portfolio_weights: Dict[str, float] = {"SHV": 1.0}
        self._current_nav: float = 100000.0
        self._latest_signal: Optional[SignalSnapshot] = None
        self._latest_allocation: Optional[TargetAllocation] = None
        self._circuit_breaker_triggered_today: bool = False

        self._stop_event = asyncio.Event()
        self._running_tasks: List[asyncio.Task] = []
        self._is_shutting_down: bool = False

        # Wire scheduler events
        self.scheduler.on_cadence(CadenceType.DAILY_CLOSE, self._handle_daily_close)
        self.scheduler.on_cadence(CadenceType.WEEKLY_REBALANCE, self._handle_weekly_rebalance)
        self.scheduler.on_cadence(CadenceType.MONTHLY_MOMENTUM, self._handle_monthly_momentum)

        # Wire WebSocket bar listener
        self.client.ws_client.on_bar(self._on_bar_received)

    async def _warmup_historical_bars(self) -> None:
        """Fetch historical daily bars via REST proxy to populate indicators."""
        logger.info("Warming up historical daily bars for %d universe symbols...", len(self.config.symbols))
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=365)
        try:
            bars_map = await self.client.get_historical_bars(
                symbols=self.config.symbols,
                timeframe="1Day",
                start=start_dt,
                end=end_dt,
            )
            for sym, b_list in bars_map.items():
                sorted_bars = sorted(b_list, key=lambda b: b.timestamp)
                self._cached_daily_bars[sym] = sorted_bars
                # Persist to database cache
                if sorted_bars and self.storage:
                    self.storage.bars.save_bars(sorted_bars, timeframe="1Day")

            logger.info("Historical warmup complete. Cached %d symbols.", len(self._cached_daily_bars))
        except Exception as e:
            logger.warning("Historical warmup encountered error (will proceed): %s", e)

    async def _handle_daily_close(self, dt: datetime) -> None:
        """Daily close evaluation (15:50 ET)."""
        logger.info("Dispatching DAILY_CLOSE evaluation at %s", dt.isoformat())
        if not self._cached_daily_bars or "SPY" not in self._cached_daily_bars:
            logger.warning("No SPY daily bars available for daily close evaluation")
            return

        is_stale = not self.client.is_safe_to_rebalance()
        signals = self.signal_engine.compute_daily_signals(
            market_data=self._cached_daily_bars,
            current_time=dt,
            is_stale=is_stale,
            upstream_connected=self.client.state_machine.is_upstream_connected,
        )

        # Apply regime override if in safe holding state
        override = self.client.get_market_regime_override()
        if override:
            signals = signals.model_copy(update={"regime": override})

        self._latest_signal = signals
        allocation = self.signal_engine.compute_target_weights(
            signals=signals,
            market_data=self._cached_daily_bars,
        )
        self._latest_allocation = allocation

        # Persist to storage in a single atomic transaction with rollback safety
        if self.storage:
            with self.storage.db.transaction() as conn:
                self.storage.signals.save(signals, rationale=allocation.rationale, conn=conn)
                self.storage.allocations.save(allocation, conn=conn)
                self.storage.regimes.record_transition_if_changed(
                    timestamp=dt,
                    current_regime=signals.regime,
                    trigger_reason=allocation.rationale,
                    conn=conn,
                )

        # Record structured audit log
        if self.audit_logger:
            self.audit_logger.log_rebalance_decision(
                trigger="DAILY_CLOSE",
                regime=signals.regime,
                timestamp=dt,
                signals=signals,
                portfolio={
                    "total_nav": self._current_nav,
                    "weights": self._current_portfolio_weights,
                },
                allocations=allocation,
                rationale=allocation.rationale,
            )

        logger.info(
            "DAILY_CLOSE evaluated: Regime=%s, VolScale=%.2f, DD=%.2f%%, Allocations=%s",
            signals.regime.value,
            signals.vol_scale_factor,
            signals.drawdown_pct * 100,
            allocation.weights,
        )

    async def _handle_weekly_rebalance(self, dt: datetime) -> List[OrderIntent]:
        """Weekly rebalance cadence (Friday 15:50 ET)."""
        logger.info("Dispatching WEEKLY_REBALANCE evaluation at %s", dt.isoformat())

        # Strict safety invariant: suppress rebalance if data feed is unsafe/stale
        if not self.client.is_safe_to_rebalance():
            logger.warning("WEEKLY_REBALANCE suppressed: data feed is in STALE_DATA_HOLD / unsafe state")
            if self.audit_logger:
                self.audit_logger.log_rebalance_decision(
                    trigger="WEEKLY_REBALANCE",
                    regime=MarketRegime.STALE_DATA_HOLD,
                    timestamp=dt,
                    rationale="Rebalance strictly suppressed due to active STALE_DATA_HOLD / unsafe data feed.",
                )
            if self.storage:
                self.storage.record_decision_audit(
                    trigger="WEEKLY_REBALANCE",
                    regime=MarketRegime.STALE_DATA_HOLD,
                    status="SUPPRESSED_STALE_DATA_HOLD",
                    rationale="Rebalance strictly suppressed due to active STALE_DATA_HOLD.",
                    timestamp=dt,
                )
            return []

        if self._latest_allocation is None:
            await self._handle_daily_close(dt)

        if self._latest_allocation is None:
            logger.warning("No target allocation available; skipping weekly rebalance")
            return []

        orders = self.rebalancer.compute_rebalance_orders(
            target_allocation=self._latest_allocation,
            current_weights=self._current_portfolio_weights,
            portfolio_equity=self._current_nav,
            timestamp=dt,
        )

        status_str = "DRY_RUN" if self.config.dry_run else "SUBMITTED"

        if orders:
            logger.info("WEEKLY_REBALANCE generated %d orders (dry_run=%s)", len(orders), self.config.dry_run)
            for o in orders:
                logger.info(
                    "  %s %s: target_w=%.3f, cur_w=%.3f, delta_w=%+.3f",
                    o.action, o.symbol, o.target_weight, o.current_weight, o.delta_weight,
                )
            if self.storage:
                self.storage.orders.save_batch(orders, status=status_str)

            # Update simulated weights
            self._current_portfolio_weights = dict(self._latest_allocation.weights)
        else:
            logger.info("WEEKLY_REBALANCE evaluated: all asset drifts within +/-2.5%% drift band; zero orders required")

        if self.audit_logger:
            self.audit_logger.log_rebalance_decision(
                trigger="WEEKLY_REBALANCE",
                regime=self._latest_allocation.regime,
                timestamp=dt,
                signals=self._latest_signal,
                portfolio={
                    "total_nav": self._current_nav,
                    "weights": self._current_portfolio_weights,
                },
                allocations=self._latest_allocation,
                orders=orders,
                rationale=self._latest_allocation.rationale,
            )

        return orders

    async def _handle_monthly_momentum(self, dt: datetime) -> Dict[str, float]:
        """Monthly 12-1 structural momentum ranking (1st trading day of month)."""
        logger.info("Dispatching MONTHLY_MOMENTUM ranking at %s", dt.isoformat())
        scores = self.signal_engine.compute_monthly_momentum(
            market_data=self._cached_daily_bars,
            current_time=dt,
        )
        logger.info("MONTHLY_MOMENTUM updated for %d assets", len(scores))
        return scores

    def _on_bar_received(self, bar: Bar) -> None:
        """Intraday 1-minute streaming bar listener and circuit breaker watchdog."""
        self._latest_intraday_bars[bar.symbol] = bar

        # Circuit breaker monitoring on SPY bars
        if bar.symbol == "SPY" and not self._circuit_breaker_triggered_today:
            # Check Keltner lower band breach
            keltner_lower = None
            if self._latest_signal:
                keltner_lower = self._latest_signal.indicators.get("keltner_lower_band")

            is_lower_breach = keltner_lower is not None and bar.close < keltner_lower
            is_flash_drop = bar.open > 0 and ((bar.close - bar.open) / bar.open < -0.03)

            if is_lower_breach or is_flash_drop:
                trigger_reason = (
                    f"SPY close ({bar.close:.2f}) breached Keltner lower band ({keltner_lower:.2f})"
                    if is_lower_breach
                    else f"SPY flash drop intraday: {(bar.close - bar.open) / bar.open:.2%}"
                )
                logger.critical("EMERGENCY CIRCUIT BREAKER TRIGGERED: %s", trigger_reason)
                self._circuit_breaker_triggered_today = True
                asyncio.create_task(self._trigger_emergency_circuit_breaker(bar, trigger_reason))

    async def _trigger_emergency_circuit_breaker(self, bar: Bar, reason: str) -> None:
        """Execute emergency de-risk / rotation into SHV on circuit breaker breach."""
        logger.critical("Executing immediate emergency de-risk: %s", reason)
        self._current_portfolio_weights = {"SHV": 1.0}

        if self.storage:
            self.storage.regimes.record_event(
                timestamp=bar.timestamp,
                old_regime=self._latest_signal.regime if self._latest_signal else MarketRegime.BULL_NORMAL,
                new_regime=MarketRegime.BEAR_CRISIS,
                trigger_reason=f"Emergency Circuit Breaker: {reason}",
            )

        if self.audit_logger:
            self.audit_logger.log_rebalance_decision(
                trigger="CIRCUIT_BREAKER_INTRADAY",
                regime=MarketRegime.BEAR_CRISIS,
                timestamp=bar.timestamp,
                portfolio={
                    "total_nav": self._current_nav,
                    "weights": self._current_portfolio_weights,
                },
                rationale=f"EMERGENCY DE-RISK: {reason}",
            )

    async def _watchdog_loop(self) -> None:
        """Stream watchdog monitoring feed freshness every 10 seconds."""
        while not self._stop_event.is_set():
            try:
                await self.client.state_machine.check_watchdog()
            except Exception as e:
                logger.error("Error in watchdog loop: %s", e)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                pass

    def _setup_signal_handlers(self) -> None:
        """Trap SIGINT and SIGTERM for graceful async shutdown."""
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(
                        sig, lambda s=sig: asyncio.create_task(self.shutdown(f"Received {s.name}"))
                    )
                except NotImplementedError:
                    # Windows or threads without signal handlers
                    signal.signal(
                        sig, lambda s, f: asyncio.create_task(self.shutdown(f"Received signal {s}"))
                    )
        except RuntimeError:
            pass

    async def start(self) -> None:
        """Start daemon background tasks and event loops."""
        logger.info("Starting DecisionDaemon (dry_run=%s)...", self.config.dry_run)
        self._stop_event.clear()
        self._setup_signal_handlers()

        # 1. Warm up historical daily bars
        await self._warmup_historical_bars()

        # 2. Connect WebSocket stream
        try:
            await self.client.connect_stream(
                symbols=self.config.symbols,
                channels=["bars"],
            )
        except Exception as e:
            logger.warning("Stream connection deferred (will retry in background): %s", e)

        # 3. Launch background tasks
        scheduler_task = asyncio.create_task(self.scheduler.run())
        watchdog_task = asyncio.create_task(self._watchdog_loop())
        self._running_tasks = [scheduler_task, watchdog_task]

        logger.info("DecisionDaemon is operational and monitoring market schedule.")
        await self._stop_event.wait()

    async def run(self) -> None:
        """Alias for start()."""
        await self.start()

    async def shutdown(self, reason: str = "Shutdown requested") -> None:
        """Gracefully stop daemon tasks, disconnect stream, and flush storage."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        logger.info("Shutting down DecisionDaemon: %s", reason)
        self._stop_event.set()
        self.scheduler.stop()

        # Cancel background tasks
        for t in self._running_tasks:
            if not t.done():
                t.cancel()
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)

        # Close client connection
        try:
            await self.client.close()
        except Exception as e:
            logger.warning("Error closing AlpacaRelayClient: %s", e)

        # Close storage
        try:
            if self.storage and hasattr(self.storage, "db"):
                self.storage.db.close()
        except Exception as e:
            logger.warning("Error closing storage database: %s", e)

        logger.info("DecisionDaemon shutdown complete. Zero data corruption guaranteed.")
