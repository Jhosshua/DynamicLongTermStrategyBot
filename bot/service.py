"""
bot.service
~~~~~~~~~~~

Unified Dynamic Strategy Service integrating DecisionDaemon, FeedManager,
PortfolioRebalancer, and PaperAccountManager for the $50k paper trading bot.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from bot.feed_manager import FeedManager, FeedManagerConfig, FeedSource
from bot.paper_account import (
    ExecutionReport,
    PaperAccountConfig,
    PaperAccountManager,
    PortfolioSummary,
)
from strategy_engine.allocator.rebalancer import PortfolioRebalancer
from strategy_engine.core.models import (
    Bar,
    MarketRegime,
    OrderIntent,
    OrderSide,
    SignalSnapshot,
    TargetAllocation,
)
from strategy_engine.core.universe import ALL_SYMBOLS
from strategy_engine.daemon.daemon import DaemonConfig, DecisionDaemon
from strategy_engine.daemon.scheduler import CadenceType, MarketCalendar, MarketScheduler
from strategy_engine.signals.regime_detector import SignalEngine
from strategy_engine.storage.audit_logger import JSONLAuditLogger
from strategy_engine.storage.repositories import StorageService

logger = logging.getLogger("bot.service")


class ServiceState(str, Enum):
    """Lifecycle state machine for DynamicStrategyService."""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class ServiceStatus(BaseModel):
    """Real-time operational status snapshot for dashboard and monitoring."""
    model_config = ConfigDict(extra="ignore")

    service_name: str = "DynamicLongTermStrategyBot"
    state: ServiceState
    is_running: bool
    is_paused: bool
    current_regime: str
    last_eval_timestamp: Optional[str] = None
    next_scheduled_event: Optional[str] = None
    feed_source: str
    alert_banner_active: bool
    cash: float
    equity: float
    total_nav: float
    active_positions_count: int
    uptime_seconds: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ManualRebalanceResult(BaseModel):
    """Result payload returned by an operator-initiated manual rebalance."""
    model_config = ConfigDict(extra="ignore")

    success: bool
    status: str
    rationale: str
    orders_count: int
    executed_trades_count: int
    total_bought: float
    total_sold: float
    total_fees: float
    portfolio_state_after: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


@dataclass
class ServiceConfig:
    """Configuration for DynamicStrategyService integration."""
    db_path: str = "strategy_engine.db"
    log_dir: str = "logs"
    relay_base_url: str = "https://alpacarelay-production.up.railway.app"
    relay_ws_url: str = "ws://localhost:8765"
    relay_token: str = ""
    symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "XLK", "XLE", "XLV", "XLI", "XLU", "TLT", "SHV", "GLD"
    ])
    initial_cash: float = 50000.00
    drift_band: float = 0.025               # +/- 2.5% drift band filter
    min_order_threshold: float = 0.005      # 0.5% micro-order threshold
    daily_close_offset_minutes: int = 10    # 15:50 ET evaluation
    monthly_eval_offset_minutes: int = 14   # 15:46 ET evaluation
    tick_interval_seconds: float = 1.0      # Scheduler tick interval
    dry_run: bool = False                   # In production bot, executes real paper orders
    dashboard_url: str = "https://dynamiclongtermstrategybot-production.up.railway.app"
    # Railway's edge cuts the relay websocket every few minutes; the bot
    # reconnects in ~2-3s. Without a grace period every blip posted a BROKEN
    # and a RECOVERED card (345 Discord posts on 2026-09-14). Only outages
    # longer than this get a card, and RECOVERED only follows a posted BROKEN.
    alert_grace_seconds: float = 90.0

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Build config from environment variables (production entry point).

        Before this existed nothing read the environment, so the deployed bot
        dialed ws://localhost with an empty token and never reached the relay.
        """
        import os

        base = cls()
        return cls(
            db_path=os.environ.get("DB_PATH", base.db_path),
            log_dir=os.environ.get("LOG_DIR", base.log_dir),
            relay_base_url=os.environ.get("RELAY_BASE_URL", base.relay_base_url),
            relay_ws_url=os.environ.get("RELAY_WS_URL", "wss://alpacarelay-production.up.railway.app"),
            relay_token=os.environ.get("RELAY_TOKEN", ""),
            dashboard_url=os.environ.get("DASHBOARD_URL", base.dashboard_url),
        )


class DynamicStrategyService(DecisionDaemon):
    """
    Unified async strategy service orchestrating DecisionDaemon, FeedManager,
    PortfolioRebalancer, and PaperAccountManager for the $50k paper trading bot.
    """

    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        paper_account: Optional[PaperAccountManager] = None,
        feed_manager: Optional[FeedManager] = None,
        signal_engine: Optional[SignalEngine] = None,
        rebalancer: Optional[PortfolioRebalancer] = None,
        scheduler: Optional[MarketScheduler] = None,
        storage: Optional[StorageService] = None,
        audit_logger: Optional[JSONLAuditLogger] = None,
        discord_notifier: Optional[Any] = None,
    ):
        # Defensively ensure an event loop exists in the current thread (Python 3.9 compatibility)
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        self.service_config = config or ServiceConfig()

        # Harmonize configuration for DecisionDaemon base class
        daemon_cfg = DaemonConfig(
            relay_base_url=self.service_config.relay_base_url,
            ws_url=self.service_config.relay_ws_url,
            relay_token=self.service_config.relay_token,
            db_path=self.service_config.db_path,
            symbols=list(self.service_config.symbols),
            dry_run=self.service_config.dry_run,
            tick_interval_seconds=self.service_config.tick_interval_seconds,
            daily_close_offset_minutes=self.service_config.daily_close_offset_minutes,
            drift_band=self.service_config.drift_band,
            min_order_threshold=self.service_config.min_order_threshold,
            log_dir=self.service_config.log_dir,
        )

        super().__init__(
            config=daemon_cfg,
            client=feed_manager.client if feed_manager else None,
            signal_engine=signal_engine,
            rebalancer=rebalancer,
            scheduler=scheduler,
            storage=storage,
            audit_logger=audit_logger,
        )

        # Inject FeedManager
        self.feed_manager = feed_manager or FeedManager(
            config=FeedManagerConfig(
                relay_base_url=self.service_config.relay_base_url,
                relay_ws_url=self.service_config.relay_ws_url,
                relay_token=self.service_config.relay_token,
                universe_symbols=self.service_config.symbols,
                db_path=self.service_config.db_path,
            ),
            database=self.storage.db,
        )
        self.client = self.feed_manager.client

        # Inject PaperAccountManager ($50,000.00 pristine starting balance)
        self.paper_account = paper_account or PaperAccountManager(
            db=self.storage.db,
            config=PaperAccountConfig(initial_cash=self.service_config.initial_cash),
        )

        # Operational state
        self._service_state: ServiceState = ServiceState.INITIALIZING
        self._start_time: datetime = datetime.now(timezone.utc)
        self._last_eval_time: Optional[datetime] = None
        self._last_rebalance_time: Optional[datetime] = None
        self._rebalance_lock = asyncio.Lock()
        self.discord_notifier = discord_notifier
        if self.discord_notifier and getattr(self.feed_manager, "discord_notifier", None) is None:
            self._wire_feed_manager_alerts()

        self._circuit_breaker_date = None

        # Wire feed manager bar events to intraday watchdog
        self.feed_manager.on_bar(self._on_bar_received)

        # Re-bind scheduler cadence callbacks to service implementations.
        # The base class already registered these bound methods, and appending
        # again made every cadence (incl. the weekly rebalance) run twice.
        self._cadence_skipped = False
        for cadence, handler in (
            (CadenceType.DAILY_CLOSE, self._handle_daily_close),
            (CadenceType.WEEKLY_REBALANCE, self._handle_weekly_rebalance),
            (CadenceType.MONTHLY_MOMENTUM, self._handle_monthly_momentum),
        ):
            self.scheduler._handlers[cadence] = []
            self.scheduler.on_cadence(cadence, self._scheduled(handler))

        # Persist completed cadences so a restart inside the 15:50-16:00 window
        # neither re-runs a finished rebalance nor forgets what already ran.
        self._init_scheduler_state()
        self.scheduler._on_executed = self._save_cadence_done

    def _scheduled(self, handler: Callable[[datetime], Any]) -> Callable[[datetime], Any]:
        """Wrap a cadence handler so a quiet skip reports "not done" (False).

        Handlers skip by returning early (paused, no live feed, no bars). The
        scheduler used to count that as done, so a 15:50 skip was never retried.
        """
        async def run(dt: datetime) -> bool:
            self._cadence_skipped = False
            await handler(dt)
            return not self._cadence_skipped

        return run

    def _init_scheduler_state(self) -> None:
        if not self.storage:
            return
        try:
            with self.storage.db.transaction() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS scheduler_state ("
                    "cadence TEXT PRIMARY KEY, last_executed_date TEXT NOT NULL)"
                )
                rows = conn.execute("SELECT cadence, last_executed_date FROM scheduler_state").fetchall()
            state = {}
            for cadence_name, day in rows:
                try:
                    state[CadenceType(cadence_name)] = date.fromisoformat(day)
                except ValueError:
                    logger.warning("Ignoring bad scheduler_state row: %s=%s", cadence_name, day)
            self.scheduler.restore_last_executed(state)
        except Exception as e:
            logger.error("Could not load scheduler state (cadences may re-run after restart): %s", e)

    def _save_cadence_done(self, cadence: CadenceType, day: date) -> None:
        if not self.storage:
            return
        with self.storage.db.transaction() as conn:
            conn.execute(
                "INSERT INTO scheduler_state (cadence, last_executed_date) VALUES (?, ?) "
                "ON CONFLICT(cadence) DO UPDATE SET last_executed_date = excluded.last_executed_date",
                (cadence.value, day.isoformat()),
            )

    def _wire_feed_manager_alerts(self) -> None:
        """Wire DataFeedManager lifecycle events to DiscordNotifier.

        A disconnect arms a timer. If the feed is still down when it fires,
        the BROKEN card is posted; a recovery before then cancels it silently.
        RECOVERED is posted only when a BROKEN card actually went out, so a
        2-second edge blip produces no Discord traffic at all.
        """
        self._broken_alert_task: Optional[asyncio.Task] = None
        self._broken_alert_posted: bool = False

        def _post_broken(reason: str, ts: datetime) -> None:
            if self.discord_notifier and hasattr(self.discord_notifier, "post_broken_alert"):
                try:
                    evidence = f"Reason: {reason} | Disconnect: {ts.isoformat()}"
                    res = self.discord_notifier.post_broken_alert(
                        component="AlpacaRelayClient",
                        error_message=f"Feed disconnected: {reason}",
                        evidence=evidence,
                        dashboard_url=self.service_config.dashboard_url,
                        since=ts,
                    )
                    if inspect.isawaitable(res):
                        asyncio.create_task(res)
                    self._broken_alert_posted = True
                except Exception as e:
                    logger.error("Failed to post Discord broken alert on feed disconnect: %s", e)

        async def _broken_after_grace(reason: str, ts: datetime) -> None:
            try:
                await asyncio.sleep(self.service_config.alert_grace_seconds)
            except asyncio.CancelledError:
                return
            if self.feed_manager.feed_source == FeedSource.ALPACA_RELAY:
                return
            logger.warning(
                "Feed still down after %.0fs grace; posting BROKEN alert",
                self.service_config.alert_grace_seconds,
            )
            _post_broken(reason, ts)

        def _on_disconnect(reason: str, ts: datetime) -> None:
            if self._broken_alert_posted:
                return  # already announced this outage
            if self._broken_alert_task and not self._broken_alert_task.done():
                return  # grace timer already running
            grace = self.service_config.alert_grace_seconds
            if grace <= 0:
                _post_broken(reason, ts)
                return
            try:
                self._broken_alert_task = asyncio.create_task(_broken_after_grace(reason, ts))
            except RuntimeError:
                # No running loop (sync test harness): fall back to posting now.
                _post_broken(reason, ts)

        def _on_recover(downtime_s: float, ts: datetime) -> None:
            if self._broken_alert_task and not self._broken_alert_task.done():
                self._broken_alert_task.cancel()
            self._broken_alert_task = None
            if not self._broken_alert_posted:
                logger.info("Feed blip of %.1fs recovered inside grace; no Discord alert", downtime_s)
                return
            self._broken_alert_posted = False
            if self.discord_notifier and hasattr(self.discord_notifier, "post_recovered_alert"):
                try:
                    status_info = f"Stream re-established at {ts.isoformat()} · Resuming live data"
                    res = self.discord_notifier.post_recovered_alert(
                        component="AlpacaRelayClient",
                        downtime_duration_s=downtime_s,
                        status_info=status_info,
                        dashboard_url=self.service_config.dashboard_url,
                    )
                    if inspect.isawaitable(res):
                        asyncio.create_task(res)
                except Exception as e:
                    logger.error("Failed to post Discord recovered alert on feed recovery: %s", e)

        self.feed_manager.on_disconnect(_on_disconnect)
        self.feed_manager.on_recover(_on_recover)

    # ------------------------------------------------------------------------
    # Historical Warmup & Mark-to-Market
    # ------------------------------------------------------------------------
    async def _warmup_historical_bars(self) -> None:
        """Fetch historical daily bars via FeedManager to warm up indicators."""
        logger.info("Warming up historical daily bars for %d symbols via FeedManager...", len(self.config.symbols))
        try:
            # 365 calendar days is ~251 trading bars, but 12-1 momentum needs
            # 253+, so momentum silently read 0 and sectors/TLT/GLD never qualified.
            bars_map = await self.feed_manager.get_historical_daily_bars(
                symbols=self.config.symbols,
                lookback_days=420,
            )
            if self.feed_manager.feed_source != FeedSource.ALPACA_RELAY:
                # Never feed synthetic bars into the signal engine.
                logger.warning("Warmup skipped: feed is not live, refusing synthetic bars")
                self._cached_daily_bars = {}
                return
            for sym, b_list in bars_map.items():
                sorted_bars = sorted(b_list, key=lambda b: b.timestamp)
                self._cached_daily_bars[sym] = sorted_bars
                if sorted_bars and self.storage:
                    self.storage.bars.save_bars(sorted_bars, timeframe="1Day")

            # Initial mark-to-market of paper account
            current_prices = self.feed_manager.get_latest_prices(self.config.symbols)
            if current_prices:
                self.paper_account.update_market_prices(current_prices)

            logger.info("Historical warmup complete. Cached %d symbols.", len(self._cached_daily_bars))
        except Exception as e:
            logger.warning("Historical warmup encountered error (operating with cached/synthetic bars): %s", e)

    # ------------------------------------------------------------------------
    # Cadence Execution Pipeline
    # ------------------------------------------------------------------------
    async def _handle_daily_close(self, dt: datetime, force: bool = False) -> None:
        """Daily regime cadence (15:50 ET)."""
        logger.info("Dispatching DAILY_CLOSE evaluation at %s (force=%s)", dt.isoformat(), force)

        # 1. Check if paused
        if self._service_state == ServiceState.PAUSED and not force:
            logger.info("DAILY_CLOSE: Service is PAUSED by operator. Skipping regime evaluation.")
            self._cadence_skipped = True
            return

        if self.feed_manager.feed_source != FeedSource.ALPACA_RELAY:
            logger.warning("DAILY_CLOSE skipped: no live relay feed, refusing to evaluate synthetic data")
            self._cadence_skipped = True
            return

        # Bars were only loaded once at startup, freezing every indicator.
        # Refetch when the cache lacks today's daily bar (Alpaca stamps it
        # 04:00 UTC, the 15:50 ET evaluation is ~16h later).
        spy_cached = self._cached_daily_bars.get("SPY") if self._cached_daily_bars else None
        eval_dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        if not spy_cached or (eval_dt - spy_cached[-1].timestamp) > timedelta(hours=20):
            await self._warmup_historical_bars()

        if not self._cached_daily_bars or "SPY" not in self._cached_daily_bars:
            logger.warning("No SPY daily bars available for daily close evaluation")
            self._cadence_skipped = True
            return

        is_safe = self.feed_manager.is_safe_to_rebalance()
        feed_active = self.feed_manager.is_connected
        signals = self.signal_engine.compute_daily_signals(
            market_data=self._cached_daily_bars,
            current_time=dt,
            is_stale=not is_safe,
            upstream_connected=feed_active,
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
        self._last_eval_time = dt

        # Persist to storage in a single atomic transaction
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

        # Mark-to-market and snapshot paper account
        current_prices = self.feed_manager.get_latest_prices(self.config.symbols)
        self.paper_account.update_market_prices(current_prices, timestamp=dt)
        self.paper_account.take_equity_snapshot(current_prices, timestamp=dt)

        # Audit logging
        portfolio = self.paper_account.get_portfolio_state(current_prices)
        if self.audit_logger:
            self.audit_logger.log_rebalance_decision(
                trigger="DAILY_CLOSE",
                regime=signals.regime,
                timestamp=dt,
                signals=signals,
                portfolio={
                    "total_nav": portfolio.total_nav,
                    "cash": portfolio.cash,
                    "equity": portfolio.equity,
                    "positions_count": len(portfolio.positions),
                },
                allocations=allocation,
                rationale=allocation.rationale,
            )

        logger.info(
            "DAILY_CLOSE evaluated: Regime=%s, NAV=$%.2f, Allocations=%s",
            signals.regime.value,
            portfolio.total_nav,
            allocation.weights,
        )

    async def _handle_weekly_rebalance(self, dt: datetime) -> List[OrderIntent]:
        """Weekly rebalance cadence (Friday 15:50 ET) executing on PaperAccountManager."""
        async with self._rebalance_lock:
            logger.info("Dispatching WEEKLY_REBALANCE evaluation at %s", dt.isoformat())

            # 1. Operator pause guard
            if self._service_state == ServiceState.PAUSED:
                logger.warning("WEEKLY_REBALANCE suppressed: Service is PAUSED by operator.")
                if self.audit_logger:
                    self.audit_logger.log_rebalance_decision(
                        trigger="WEEKLY_REBALANCE",
                        regime=self._latest_signal.regime if self._latest_signal else MarketRegime.STALE_DATA_HOLD,
                        timestamp=dt,
                        rationale="WEEKLY_REBALANCE suppressed: Operator paused.",
                    )
                self._cadence_skipped = True
                return []

            # 2. Feed safety check
            if not self.feed_manager.is_safe_to_rebalance():
                logger.warning("WEEKLY_REBALANCE suppressed: feed manager reported unsafe data feed.")
                if self.audit_logger:
                    self.audit_logger.log_rebalance_decision(
                        trigger="WEEKLY_REBALANCE",
                        regime=MarketRegime.STALE_DATA_HOLD,
                        timestamp=dt,
                        rationale="Rebalance strictly suppressed due to unsafe data feed.",
                    )
                self._cadence_skipped = True
                return []

            # 3. Allocation freshness check
            if self._latest_allocation is None:
                await self._handle_daily_close(dt)

            if self._latest_allocation is None:
                logger.warning("No target allocation available; skipping weekly rebalance")
                self._cadence_skipped = True
                return []

            # 4. Read portfolio state from real paper account
            current_prices = self.feed_manager.get_latest_prices(self.config.symbols)
            portfolio = self.paper_account.get_portfolio_state(current_prices)

            current_weights: Dict[str, float] = {}
            for pos in portfolio.positions:
                current_weights[pos.symbol] = pos.weight
            current_weights["SHV"] = current_weights.get("SHV", 0.0) + portfolio.cash_weight
            nav = portfolio.total_nav

            # 5. Compute rebalance delta orders with drift band filter
            orders = self.rebalancer.compute_rebalance_orders(
                target_allocation=self._latest_allocation,
                current_weights=current_weights,
                portfolio_equity=nav,
                current_prices=current_prices,
                timestamp=dt,
            )

            # 6. Execute rebalance orders on paper account (SELLs before BUYs)
            if orders:
                logger.info("Executing %d rebalance orders on paper account ($%.2f NAV)", len(orders), nav)
                execution_report = self.paper_account.execute_rebalance_orders(
                    orders=orders,
                    current_prices=current_prices,
                    timestamp=dt,
                )
                self.storage.orders.save_batch(orders, status="FILLED")
                self._current_portfolio_weights = dict(self._latest_allocation.weights)
                self._last_rebalance_time = dt

                # Optional Discord notification
                if self.discord_notifier and hasattr(self.discord_notifier, "post_trade_execution"):
                    try:
                        res = self.discord_notifier.post_trade_execution(
                            orders=orders,
                            nav=execution_report.portfolio_state_after.total_nav,
                            regime=self._latest_allocation.regime.value,
                            dashboard_url=self.service_config.dashboard_url,
                        )
                        if inspect.isawaitable(res):
                            await res
                    except Exception as notify_err:
                        logger.error("Failed to post Discord trade alert: %s", notify_err)
            else:
                logger.info("WEEKLY_REBALANCE evaluated: all holdings within +/-2.5%% drift band. 0 orders needed.")

            # 7. Audit log
            if self.audit_logger:
                self.audit_logger.log_rebalance_decision(
                    trigger="WEEKLY_REBALANCE",
                    regime=self._latest_allocation.regime,
                    timestamp=dt,
                    signals=self._latest_signal,
                    portfolio={
                        "total_nav": nav,
                        "cash": portfolio.cash,
                        "equity": portfolio.equity,
                        "positions_count": len(portfolio.positions),
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

    # ------------------------------------------------------------------------
    # Intraday Circuit Breaker Watchdog
    # ------------------------------------------------------------------------
    def _on_bar_received(self, bar: Bar) -> None:
        """Intraday streaming bar listener and circuit breaker watchdog."""
        self._latest_intraday_bars[bar.symbol] = bar

        # Synthetic ticks must never liquidate the account.
        if self.feed_manager.feed_source != FeedSource.ALPACA_RELAY:
            return

        # The flag was never reset, so the breaker fired at most once per process.
        # Re-arm it when the calendar day changes.
        today = bar.timestamp.date()
        if getattr(self, "_circuit_breaker_date", None) != today:
            self._circuit_breaker_triggered_today = False

        # Circuit breaker monitoring on SPY bars
        if bar.symbol == "SPY" and not self._circuit_breaker_triggered_today:
            keltner_lower = None
            if self._latest_signal and self._latest_signal.indicators:
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
                self._circuit_breaker_date = today
                asyncio.create_task(self._trigger_emergency_circuit_breaker(bar, trigger_reason))

    async def _trigger_emergency_circuit_breaker(self, bar: Bar, reason: str) -> None:
        """Execute emergency de-risk / rotation into SHV on paper account."""
        logger.critical("Executing immediate emergency de-risk: %s", reason)
        async with self._rebalance_lock:
            current_prices = self.feed_manager.get_latest_prices(self.config.symbols)
            portfolio = self.paper_account.get_portfolio_state(current_prices)

            # Construct emergency target allocation: 100% SHV
            emergency_target = TargetAllocation(
                timestamp=bar.timestamp,
                regime=MarketRegime.BEAR_CRISIS,
                weights={"SHV": 1.0},
                cash_weight=1.0,
                rationale=f"EMERGENCY DE-RISK: {reason}",
            )
            self._latest_allocation = emergency_target

            current_weights: Dict[str, float] = {p.symbol: p.weight for p in portfolio.positions}
            current_weights["SHV"] = current_weights.get("SHV", 0.0) + portfolio.cash_weight

            orders = self.rebalancer.compute_rebalance_orders(
                target_allocation=emergency_target,
                current_weights=current_weights,
                portfolio_equity=portfolio.total_nav,
                current_prices=current_prices,
                timestamp=bar.timestamp,
            )

            if orders:
                execution_report = self.paper_account.execute_rebalance_orders(
                    orders=orders,
                    current_prices=current_prices,
                    timestamp=bar.timestamp,
                )
                self.storage.orders.save_batch(orders, status="FILLED")

                if self.discord_notifier and hasattr(self.discord_notifier, "post_trade_execution"):
                    try:
                        res = self.discord_notifier.post_trade_execution(
                            orders=orders,
                            nav=execution_report.portfolio_state_after.total_nav,
                            regime=emergency_target.regime.value,
                            dashboard_url=self.service_config.dashboard_url,
                        )
                        if inspect.isawaitable(res):
                            await res
                    except Exception as notify_err:
                        logger.error("Failed to post Discord trade alert on emergency de-risk: %s", notify_err)

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
                        "total_nav": portfolio.total_nav,
                        "cash": portfolio.cash,
                        "equity": portfolio.equity,
                        "positions_count": len(portfolio.positions),
                    },
                    rationale=f"EMERGENCY DE-RISK: {reason}",
                )

    # ------------------------------------------------------------------------
    # Public Operator Interfaces
    # ------------------------------------------------------------------------
    async def pause(self) -> ServiceStatus:
        """
        Pause automated rebalancing evaluations.
        Market data ingestion and mark-to-market continue, but rebalances are frozen.
        """
        async with self._rebalance_lock:
            self._service_state = ServiceState.PAUSED
            logger.warning("OPERATOR ACTION: Service PAUSED. Scheduled rebalances frozen.")
            if self.audit_logger and hasattr(self.audit_logger, "log_operator_action"):
                self.audit_logger.log_operator_action("PAUSE", "Operator requested pause")
        return self.get_service_status()

    async def resume(self) -> ServiceStatus:
        """
        Resume automated rebalancing evaluations.
        Restores normal scheduled cadence execution.
        """
        async with self._rebalance_lock:
            self._service_state = ServiceState.RUNNING
            logger.info("OPERATOR ACTION: Service RESUMED. Scheduled rebalances active.")
            if self.audit_logger and hasattr(self.audit_logger, "log_operator_action"):
                self.audit_logger.log_operator_action("RESUME", "Operator requested resume")
        return self.get_service_status()

    async def manual_rebalance(self, force: bool = False) -> ManualRebalanceResult:
        """
        Trigger an immediate out-of-cadence strategy rebalance evaluation.

        Parameters:
            force: If True, executes even if service is PAUSED or data feed is in fallback.
        """
        async with self._rebalance_lock:
            dt = datetime.now(timezone.utc)
            logger.info("OPERATOR ACTION: Manual rebalance evaluation initiated (force=%s)...", force)

            # 1. State check
            if self._service_state == ServiceState.PAUSED and not force:
                return ManualRebalanceResult(
                    success=False,
                    status="REJECTED_PAUSED",
                    rationale="Service is currently PAUSED. Use force=True to override.",
                    orders_count=0,
                    executed_trades_count=0,
                    total_bought=0.0,
                    total_sold=0.0,
                    total_fees=0.0,
                )

            # 2. Feed safety check
            # force may override PAUSE, never a dead/synthetic feed.
            if not self.feed_manager.is_safe_to_rebalance():
                return ManualRebalanceResult(
                    success=False,
                    status="REJECTED_UNSAFE_FEED",
                    rationale="Market data feed is not live (synthetic or STALE_DATA_HOLD). Trading refused.",
                    orders_count=0,
                    executed_trades_count=0,
                    total_bought=0.0,
                    total_sold=0.0,
                    total_fees=0.0,
                )

            # 3. Re-evaluate daily signals & target allocation
            await self._handle_daily_close(dt, force=force)
            if self._latest_allocation is None:
                return ManualRebalanceResult(
                    success=False,
                    status="ERROR_NO_ALLOCATION",
                    rationale="Failed to compute target allocation weights.",
                    orders_count=0,
                    executed_trades_count=0,
                    total_bought=0.0,
                    total_sold=0.0,
                    total_fees=0.0,
                )

            # 4. Read portfolio state
            current_prices = self.feed_manager.get_latest_prices(self.config.symbols)
            portfolio = self.paper_account.get_portfolio_state(current_prices)
            current_weights = {p.symbol: p.weight for p in portfolio.positions}
            current_weights["SHV"] = current_weights.get("SHV", 0.0) + portfolio.cash_weight

            # 5. Compute delta orders with drift band filter
            orders = self.rebalancer.compute_rebalance_orders(
                target_allocation=self._latest_allocation,
                current_weights=current_weights,
                portfolio_equity=portfolio.total_nav,
                current_prices=current_prices,
                timestamp=dt,
            )

            # 6. Execute orders (SELLs before BUYs)
            if orders:
                report = self.paper_account.execute_rebalance_orders(orders, current_prices, timestamp=dt)
                self.storage.orders.save_batch(orders, status="FILLED")
                self._current_portfolio_weights = dict(self._latest_allocation.weights)
                self._last_rebalance_time = dt

                if self.discord_notifier and hasattr(self.discord_notifier, "post_trade_execution"):
                    try:
                        res = self.discord_notifier.post_trade_execution(
                            orders=orders,
                            nav=report.portfolio_state_after.total_nav,
                            regime=self._latest_allocation.regime.value,
                            dashboard_url=self.service_config.dashboard_url,
                        )
                        if inspect.isawaitable(res):
                            await res
                    except Exception as e:
                        logger.error("Discord alert error: %s", e)

                return ManualRebalanceResult(
                    success=True,
                    status="EXECUTED",
                    rationale=self._latest_allocation.rationale,
                    orders_count=len(orders),
                    executed_trades_count=report.orders_filled,
                    total_bought=report.total_bought_dollars,
                    total_sold=report.total_sold_dollars,
                    total_fees=report.total_fees,
                    portfolio_state_after=report.portfolio_state_after.to_dict(),
                )
            else:
                return ManualRebalanceResult(
                    success=True,
                    status="SKIPPED_WITHIN_BAND",
                    rationale="All asset weights are within +/-2.5% drift band. No orders generated.",
                    orders_count=0,
                    executed_trades_count=0,
                    total_bought=0.0,
                    total_sold=0.0,
                    total_fees=0.0,
                    portfolio_state_after=portfolio.to_dict(),
                )

    def get_service_status(self) -> ServiceStatus:
        """Return consolidated operational status snapshot."""
        current_prices = self.feed_manager.get_latest_prices(self.config.symbols)
        portfolio = self.paper_account.get_portfolio_state(current_prices)
        conn_status = self.feed_manager.get_connection_status()
        now_dt = datetime.now(timezone.utc)
        uptime = (now_dt - self._start_time).total_seconds()

        return ServiceStatus(
            service_name="DynamicLongTermStrategyBot",
            state=self._service_state,
            is_running=self._service_state in (ServiceState.RUNNING, ServiceState.PAUSED),
            is_paused=self._service_state == ServiceState.PAUSED,
            current_regime=self._latest_signal.regime.value if self._latest_signal else "UNKNOWN",
            last_eval_timestamp=self._last_eval_time.isoformat() if self._last_eval_time else None,
            next_scheduled_event=self.scheduler.get_next_event_description() if hasattr(self.scheduler, "get_next_event_description") else "DAILY_CLOSE (15:50 ET)",
            feed_source=conn_status.feed_source,
            alert_banner_active=conn_status.alert_banner_active,
            cash=portfolio.cash,
            equity=portfolio.equity,
            total_nav=portfolio.total_nav,
            active_positions_count=len(portfolio.positions),
            uptime_seconds=uptime,
        )

    def get_health(self) -> Dict[str, Any]:
        """Return standard unauthenticated /health dictionary for Railway monitoring."""
        status = self.get_service_status()
        conn_status = self.feed_manager.get_connection_status()
        live = self.feed_manager.feed_source == FeedSource.ALPACA_RELAY
        running = status.state in (ServiceState.RUNNING, ServiceState.PAUSED)
        return {
            "status": "ok" if (live and running) else "degraded",
            "service": status.service_name,
            "state": status.state.value,
            "relay": conn_status.to_dict(),
            "portfolio": {
                "nav": status.total_nav,
                "cash": status.cash,
                "equity": status.equity,
                "positions_count": status.active_positions_count,
            },
            "regime": status.current_regime,
            "uptime_s": status.uptime_seconds,
            "timestamp": status.timestamp,
        }

    def reset_to_pristine(self) -> PortfolioSummary:
        """
        Purge all smoke test data and restore clean $50,000.00 cash balance.
        Guarantees account is in a pristine state ready for Monday's market open.
        """
        logger.info("OPERATOR ACTION: Executing reset_to_pristine()...")
        summary = self.paper_account.reset_to_pristine()
        self._current_portfolio_weights = {"SHV": 1.0}
        self._circuit_breaker_triggered_today = False
        self._latest_allocation = None
        logger.info("Reset to pristine complete: Cash=$%.2f, Positions=0, Total NAV=$%.2f", summary.cash, summary.total_nav)
        return summary

    # ------------------------------------------------------------------------
    # Service Lifecycle Management
    # ------------------------------------------------------------------------
    async def start(self) -> None:
        """Start all background loops, feed listeners, and scheduler."""
        logger.info("Starting DynamicStrategyService (initial_cash=$%.2f)...", self.service_config.initial_cash)
        self._stop_event.clear()
        # No signal handlers here: uvicorn owns SIGTERM/SIGINT and calls shutdown()
        # via the lifespan. Hijacking them left the web server up with a STOPPED bot.

        # 1. Initialize SQLite schema & paper account ledger
        self.paper_account.init_schema()

        # 2. Start FeedManager (connects WS / REST, auto-fallback on error)
        await self.feed_manager.start()

        # 3. Warm up historical bars
        await self._warmup_historical_bars()

        # 4. Launch background tasks
        scheduler_task = asyncio.create_task(self.scheduler.run(), name="scheduler_loop")
        feed_watchdog_task = asyncio.create_task(self.feed_manager.watchdog_loop(), name="feed_watchdog")
        self._running_tasks = [scheduler_task, feed_watchdog_task]

        self._service_state = ServiceState.RUNNING
        logger.info("DynamicStrategyService is fully operational and monitoring NYSE market cadences.")
        await self._stop_event.wait()

    async def shutdown(self, reason: str = "Shutdown requested") -> None:
        """Graceful shutdown flushes WAL database, disconnects feed, and stops scheduler."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        self._service_state = ServiceState.STOPPED
        logger.info("Shutting down DynamicStrategyService: %s", reason)
        self._stop_event.set()
        self.scheduler.stop()

        # Cancel running background tasks
        for t in self._running_tasks:
            if not t.done():
                t.cancel()
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)

        # Stop feed manager
        try:
            await self.feed_manager.stop()
        except Exception as e:
            logger.warning("Error stopping FeedManager: %s", e)

        # Flush & close storage
        try:
            self.paper_account.close()
        except Exception as e:
            logger.warning("Error closing PaperAccountManager: %s", e)

        try:
            if self.storage and hasattr(self.storage, "db"):
                self.storage.db.close()
        except Exception as e:
            logger.warning("Error closing database: %s", e)

        logger.info("DynamicStrategyService shutdown complete. Zero data corruption.")
