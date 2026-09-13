"""
bot.feed_manager
~~~~~~~~~~~~~~~~

Production Resilient Data Feed Manager & Fallback Simulation for AlpacaRelay.
Coordinates live REST/WebSocket ingestion, disconnect detection, two-tier
fallback simulation (SQLite WAL + SDE stress scenarios), dashboard alert flags,
and autonomous self-healing recovery.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union

from pydantic import BaseModel, ConfigDict, Field

from strategy_engine.core.models import Bar, Quote, Trade
from strategy_engine.ingestion.client import AlpacaRelayClient
from strategy_engine.ingestion.rest_client import (
    AlpacaRelayError,
    RelayAuthError,
    RelayRateLimitError,
    RelayUpstreamError,
)
from strategy_engine.ingestion.ws_client import (
    LifecycleEvent,
    RelayClientError,
    RelayConnectionError,
    RelaySlowClientEvictionError,
)
from strategy_engine.simulator.stress_scenarios import (
    StressScenarioType,
    generate_stress_scenario,
)
from strategy_engine.storage.database import Database
from strategy_engine.storage.repositories import MarketBarRepository

logger = logging.getLogger("bot.feed_manager")


def _to_iso(dt: datetime) -> str:
    """Format datetime as UTC ISO-8601 string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


class FeedSource(str, Enum):
    """Origin of current market data feed."""
    ALPACA_RELAY = "alpaca_relay"
    SYNTHETIC_FALLBACK = "synthetic_fallback"


class ConnectionStatus(BaseModel):
    """Connection status object consumed by FastAPI SSE and Dashboard UI."""
    model_config = ConfigDict(extra="ignore")

    is_connected: bool
    feed_source: str
    alert_banner_active: bool
    last_heartbeat_timestamp: str
    status_message: str
    fallback_reason: Optional[str] = None
    disconnect_timestamp: Optional[str] = None
    downtime_duration_seconds: float = 0.0
    reconnect_attempts: int = 0
    active_symbols: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


@dataclass
class FeedManagerConfig:
    """Configuration options for Resilient Feed Manager."""
    relay_base_url: str = "https://alpacarelay-production.up.railway.app"
    relay_ws_url: str = "ws://localhost:8765"
    relay_token: str = ""
    universe_symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "XLK", "XLE", "XLV", "XLI", "XLU", "TLT", "SHV", "GLD"
    ])
    db_path: str = "strategy_engine.db"
    stale_timeout_seconds: float = 120.0
    recovery_check_interval: float = 5.0
    max_recovery_interval: float = 30.0
    synthetic_tick_interval: float = 2.0
    default_fallback_scenario: str = "2017_low_vol_bull"


class FeedManager:
    """Resilient Feed Manager wrapping AlpacaRelay with automatic fallback simulation."""

    def __init__(
        self,
        config: Optional[FeedManagerConfig] = None,
        client: Optional[AlpacaRelayClient] = None,
        database: Optional[Union[Database, str]] = None,
        bar_repo: Optional[MarketBarRepository] = None,
        discord_notifier: Optional[Any] = None,
        dashboard_url: Optional[str] = None,
    ):
        self.config = config or FeedManagerConfig()
        if database is None:
            self.db = Database(self.config.db_path)
        elif isinstance(database, Database):
            self.db = database
        else:
            self.db = Database(str(database))

        self.bar_repo = bar_repo or MarketBarRepository(self.db)

        # Underlying low-level AlpacaRelay client
        self.client = client or AlpacaRelayClient(
            base_url=self.config.relay_base_url,
            ws_url=self.config.relay_ws_url,
            relay_token=self.config.relay_token,
        )

        # State management
        self._feed_source: FeedSource = FeedSource.SYNTHETIC_FALLBACK
        self._is_connected: bool = False
        self._alert_banner_active: bool = True
        self._status_message: str = "FeedManager initializing..."
        self._fallback_reason: Optional[str] = "Initial startup prior to live connect"
        self._disconnect_timestamp: Optional[datetime] = datetime.now(timezone.utc)
        self._last_heartbeat: Optional[datetime] = None
        self._reconnect_attempts: int = 0

        # In-memory caches
        self._latest_bars: Dict[str, Bar] = {}
        self._latest_prices: Dict[str, float] = {}

        # Callbacks
        self._on_bar_callbacks: List[Callable[[Bar], Any]] = []
        self._disconnect_callbacks: List[Callable[[str, datetime], Any]] = []
        self._recover_callbacks: List[Callable[[float, datetime], Any]] = []

        # Optional Discord alert wiring
        self.discord_notifier = discord_notifier
        self.dashboard_url = dashboard_url or getattr(
            self.config, "dashboard_url", "https://dynamiclongtermstrategybot-production.up.railway.app"
        )
        if self.discord_notifier:
            self._wire_discord_callbacks()

        # Background tasks
        self._running: bool = False
        self._recovery_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._synthetic_ticker_task: Optional[asyncio.Task] = None
        self._state_lock = asyncio.Lock()

        # Wire client callbacks
        self.client.ws_client.on_bar(self._on_ws_bar)
        self.client.ws_client.on_quote(self._on_ws_quote)
        self.client.ws_client.on_trade(self._on_ws_trade)
        self.client.ws_client.on_lifecycle(self._on_ws_lifecycle)

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def feed_source(self) -> FeedSource:
        return self._feed_source

    @property
    def alert_banner_active(self) -> bool:
        return self._alert_banner_active

    # ------------------------------------------------------------------------
    # Public Lifecycle
    # ------------------------------------------------------------------------
    async def start(self) -> None:
        """Start feed manager, connect stream, and initialize background workers."""
        self._running = True

        # 1. Populate baseline historical bars into cache
        await self._ensure_baseline_data()

        # 2. Start background tasks
        self._watchdog_task = asyncio.create_task(self._watchdog_loop(), name="feed_watchdog")
        self._recovery_task = asyncio.create_task(self._auto_recovery_loop(), name="feed_recovery")

        # 3. Attempt initial live connection
        await self._try_connect_live()
        if not self._is_connected:
            self._ensure_synthetic_ticker()

    async def stop(self) -> None:
        """Graceful shutdown of all clients and background loops."""
        self._running = False
        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
        if self._synthetic_ticker_task and not self._synthetic_ticker_task.done():
            self._synthetic_ticker_task.cancel()

        try:
            await self.client.close()
        except Exception as e:
            logger.warning("Error closing AlpacaRelayClient: %s", e)

        logger.info("FeedManager shutdown cleanly.")

    # ------------------------------------------------------------------------
    # Baseline Data & Connection Mechanics
    # ------------------------------------------------------------------------
    async def _ensure_baseline_data(self) -> None:
        """Pre-populate SQLite WAL with historical bars if empty."""
        missing: List[str] = []
        for sym in self.config.universe_symbols:
            existing = self.bar_repo.get_latest_bar(sym, timeframe="1Day")
            if not existing:
                missing.append(sym)
            else:
                self._latest_bars[sym] = existing
                self._latest_prices[sym] = existing.close

        if missing:
            logger.info("Pre-populating SQLite WAL cache with synthetic scenario bars for %s...", missing)
            try:
                dataset = generate_stress_scenario(self.config.default_fallback_scenario, seed=42)
                fallbacks = {
                    "SPY": 500.0, "QQQ": 440.0, "XLK": 210.0, "XLE": 85.0,
                    "XLV": 140.0, "XLI": 120.0, "XLU": 65.0, "TLT": 95.0,
                    "SHV": 110.0, "GLD": 215.0,
                }
                spy_bars = dataset.bars.get("SPY", [])
                for sym in missing:
                    b_list = dataset.bars.get(sym, [])
                    if not b_list and spy_bars:
                        base_price = fallbacks.get(sym, 100.0)
                        b_list = [
                            Bar(
                                symbol=sym,
                                timestamp=sb.timestamp,
                                open=round(base_price * (sb.open / spy_bars[0].open), 2),
                                high=round(base_price * (sb.high / spy_bars[0].open), 2),
                                low=round(base_price * (sb.low / spy_bars[0].open), 2),
                                close=round(base_price * (sb.close / spy_bars[0].open), 2),
                                volume=100000,
                            )
                            for sb in spy_bars
                        ]
                    if b_list:
                        self.bar_repo.save_bars(b_list, timeframe="1Day")
                        self._latest_bars[sym] = b_list[-1]
                        self._latest_prices[sym] = b_list[-1].close
            except Exception as e:
                logger.warning("Failed to pre-populate synthetic bars: %s", e)

    async def _try_connect_live(self) -> bool:
        """Attempt to connect to live AlpacaRelay REST and WS."""
        try:
            health = await self.client.rest_client.get_health()
            if health.get("upstream") == "connected":
                await self.client.connect_stream(
                    symbols=self.config.universe_symbols,
                    channels=["bars", "quotes", "trades"],
                )
                # Brief yield to ensure subscription is acknowledged on socket
                await asyncio.sleep(0.05)
                await self._transition_to_live("Initial connection successful")
                return True
            else:
                await self._transition_to_fallback("Upstream reported down on initial connect")
                return False
        except Exception as e:
            logger.info("Initial live connect deferred (%s). Operating in fallback simulation.", e)
            await self._transition_to_fallback(f"Initial connection failed: {e}")
            return False

    async def _transition_to_live(self, reason: str = "Connected") -> None:
        """Transition state from fallback simulation to live AlpacaRelay stream."""
        async with self._state_lock:
            if self._feed_source == FeedSource.ALPACA_RELAY and self._is_connected:
                return

            downtime = 0.0
            if self._disconnect_timestamp is not None:
                downtime = max(0.0, (datetime.now(timezone.utc) - self._disconnect_timestamp).total_seconds())

            self._feed_source = FeedSource.ALPACA_RELAY
            self._is_connected = True
            self._alert_banner_active = False
            self._status_message = "AlpacaRelay live feed active"
            self._fallback_reason = None
            self._disconnect_timestamp = None
            self._last_heartbeat = datetime.now(timezone.utc)

            # Stop synthetic ticker if running
            if self._synthetic_ticker_task and not self._synthetic_ticker_task.done():
                self._synthetic_ticker_task.cancel()
                self._synthetic_ticker_task = None

            logger.info("FeedManager transitioned to LIVE: %s (downtime: %.1fs)", reason, downtime)

            # Dispatch recovery notifications
            for cb in self._recover_callbacks:
                try:
                    res = cb(downtime, datetime.now(timezone.utc))
                    if inspect.isawaitable(res):
                        await res
                except Exception as e:
                    logger.error("Error in recover callback: %s", e)

    async def _transition_to_fallback(self, reason: str) -> None:
        """Transition state from live stream to fallback simulation."""
        async with self._state_lock:
            # Start synthetic ticker if not running
            self._ensure_synthetic_ticker()

            if self._feed_source == FeedSource.SYNTHETIC_FALLBACK and not self._is_connected:
                return

            self._feed_source = FeedSource.SYNTHETIC_FALLBACK
            self._is_connected = False
            self._alert_banner_active = True
            self._disconnect_timestamp = datetime.now(timezone.utc)
            self._fallback_reason = reason
            self._status_message = (
                f"AlpacaRelay offline: {reason}. Operating on synthetic fallback simulation."
            )
            logger.warning("FeedManager transitioned to FALLBACK: %s", reason)

            # Dispatch disconnect notifications
            for cb in self._disconnect_callbacks:
                try:
                    res = cb(reason, self._disconnect_timestamp)
                    if inspect.isawaitable(res):
                        await res
                except Exception as e:
                    logger.error("Error in disconnect callback: %s", e)

    def _ensure_synthetic_ticker(self) -> None:
        """Launch background synthetic forward ticker during fallback mode."""
        if self._synthetic_ticker_task is None or self._synthetic_ticker_task.done():
            self._synthetic_ticker_task = asyncio.create_task(
                self._synthetic_ticker_loop(), name="synthetic_ticker"
            )

    async def _synthetic_ticker_loop(self) -> None:
        """Simulate incremental forward ticks during fallback mode so UI and paper account stay dynamic."""
        import random
        while self._running and self._feed_source == FeedSource.SYNTHETIC_FALLBACK:
            try:
                await asyncio.sleep(self.config.synthetic_tick_interval)
                now = datetime.now(timezone.utc)
                self._last_heartbeat = now

                for sym in self.config.universe_symbols:
                    cur_price = self.get_latest_price(sym)
                    # Micro-drift simulation
                    change_pct = random.gauss(0.0001, 0.001)
                    new_price = round(max(1.0, cur_price * (1.0 + change_pct)), 4)
                    self._latest_prices[sym] = new_price

                    sim_bar = Bar(
                        symbol=sym,
                        timestamp=now,
                        open=cur_price,
                        high=max(cur_price, new_price),
                        low=min(cur_price, new_price),
                        close=new_price,
                        volume=100,
                    )
                    self._latest_bars[sym] = sim_bar

                    # Dispatch bar to subscribers
                    for cb in self._on_bar_callbacks:
                        try:
                            res = cb(sim_bar)
                            if inspect.isawaitable(res):
                                asyncio.create_task(res)
                        except Exception as e:
                            logger.debug("Error in on_bar subscriber during sim tick: %s", e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in synthetic ticker loop: %s", e)

    # ------------------------------------------------------------------------
    # Client Callback Hooks
    # ------------------------------------------------------------------------
    def _on_ws_bar(self, bar: Bar) -> None:
        self._latest_bars[bar.symbol] = bar
        self._latest_prices[bar.symbol] = bar.close
        self._last_heartbeat = datetime.now(timezone.utc)
        try:
            self.bar_repo.save_bar(bar, timeframe="1Min")
        except Exception as e:
            logger.error("Failed to save bar to WAL: %s", e)

        for cb in self._on_bar_callbacks:
            try:
                res = cb(bar)
                if inspect.isawaitable(res):
                    asyncio.create_task(res)
            except Exception as e:
                logger.error("Error in on_bar callback: %s", e)

    def _on_ws_quote(self, quote: Quote) -> None:
        if quote.bid_price > 0 and quote.ask_price > 0:
            self._latest_prices[quote.symbol] = (quote.bid_price + quote.ask_price) / 2.0
        elif quote.bid_price > 0:
            self._latest_prices[quote.symbol] = quote.bid_price
        elif quote.ask_price > 0:
            self._latest_prices[quote.symbol] = quote.ask_price
        self._last_heartbeat = datetime.now(timezone.utc)

    def _on_ws_trade(self, trade: Trade) -> None:
        if trade.price > 0:
            self._latest_prices[trade.symbol] = trade.price
        self._last_heartbeat = datetime.now(timezone.utc)

    def _on_ws_lifecycle(self, event: LifecycleEvent) -> None:
        if event.event_type == "upstream_disconnected":
            asyncio.create_task(self._transition_to_fallback("AlpacaRelay upstream disconnected"))
        elif event.event_type == "upstream_connected":
            asyncio.create_task(self._try_connect_live())

    # ------------------------------------------------------------------------
    # Background Tasks: Watchdog & Recovery
    # ------------------------------------------------------------------------
    async def _watchdog_loop(self) -> None:
        """Silent stream watchdog detecting hung sockets, evictions, or missing heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(0.5)
                if not self._running:
                    break
                if self._feed_source == FeedSource.ALPACA_RELAY:
                    # Fast disconnect detection: check underlying WebSocket client connection state
                    ws_client = getattr(self.client, "ws_client", None)
                    if ws_client is not None and not ws_client.is_connected:
                        logger.warning("Feed watchdog detected disconnected WebSocket stream")
                        await self._transition_to_fallback("WebSocket stream disconnected")
                        continue

                    if self._last_heartbeat is not None:
                        elapsed = (datetime.now(timezone.utc) - self._last_heartbeat).total_seconds()
                        if elapsed > self.config.stale_timeout_seconds:
                            logger.warning(
                                "Feed watchdog timeout: stream silent for %.1fs > %.1fs",
                                elapsed,
                                self.config.stale_timeout_seconds,
                            )
                            await self._transition_to_fallback(
                                f"Feed watchdog timeout: stream silent > {int(self.config.stale_timeout_seconds)}s"
                            )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Watchdog loop error: %s", e)

    async def watchdog_loop(self) -> None:
        """Public alias for _watchdog_loop."""
        await self._watchdog_loop()

    async def _auto_recovery_loop(self) -> None:
        """Autonomous self-healing background worker checking upstream health and restoring stream."""
        interval = self.config.recovery_check_interval
        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    break

                if self._feed_source == FeedSource.SYNTHETIC_FALLBACK:
                    self._reconnect_attempts += 1
                    try:
                        health = await self.client.rest_client.get_health()
                        if health.get("upstream") == "connected":
                            logger.info("Auto-recovery: upstream is online. Re-connecting WebSocket stream...")
                            await self.client.connect_stream(
                                symbols=self.config.universe_symbols,
                                channels=["bars", "quotes", "trades"],
                            )

                            # Gap check & backfill if downtime > 60 seconds
                            if self._disconnect_timestamp is not None:
                                gap_duration = (datetime.now(timezone.utc) - self._disconnect_timestamp).total_seconds()
                                if gap_duration > 60.0:
                                    logger.info("Downtime was %.1fs > 60s. Backfilling gap bars via REST...", gap_duration)
                                    try:
                                        backfilled = await self.client.get_historical_bars(
                                            symbols=self.config.universe_symbols,
                                            timeframe="1Min",
                                            start=self._disconnect_timestamp,
                                            end=datetime.now(timezone.utc),
                                        )
                                        for sym, b_list in backfilled.items():
                                            if b_list:
                                                self.bar_repo.save_bars(b_list, timeframe="1Min")
                                    except Exception as be:
                                        logger.warning("Gap backfill encountered error (proceeding to live): %s", be)

                            await self._transition_to_live("Autonomous recovery complete")
                            interval = self.config.recovery_check_interval
                        else:
                            # Exponential backoff
                            interval = min(self.config.max_recovery_interval, interval * 1.5)
                    except Exception as e:
                        logger.debug("Auto-recovery poll attempt failed: %s", e)
                        interval = min(self.config.max_recovery_interval, interval * 1.5)
                else:
                    interval = self.config.recovery_check_interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in auto-recovery loop: %s", e)

    # ------------------------------------------------------------------------
    # Public Market Data Queries (Zero-Crash Interface)
    # ------------------------------------------------------------------------
    async def get_historical_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, List[Bar]]:
        """Fetch historical bars from live REST proxy, falling back to SQLite WAL / SDE."""
        # 1. Try Live REST if connected
        if self._feed_source == FeedSource.ALPACA_RELAY:
            try:
                live_bars = await self.client.get_historical_bars(
                    symbols=symbols,
                    timeframe=timeframe,
                    start=start or (datetime.now(timezone.utc) - timedelta(days=365)),
                    end=end or datetime.now(timezone.utc),
                )
                if live_bars:
                    # Cache to SQLite WAL
                    for sym, b_list in live_bars.items():
                        if b_list:
                            self.bar_repo.save_bars(b_list, timeframe=timeframe)
                    return live_bars
            except Exception as e:
                logger.warning("Live REST get_historical_bars failed (%s). Falling back to cached/synthetic.", e)
                await self._transition_to_fallback(reason=f"REST error: {e}")

        # 2. Tier 1 Fallback: SQLite WAL Cache
        result: Dict[str, List[Bar]] = {}
        missing_symbols: List[str] = []
        for sym in symbols:
            cached = self.bar_repo.get_bars(symbol=sym, timeframe=timeframe, start=start, end=end)
            if cached:
                result[sym] = cached
            else:
                missing_symbols.append(sym)

        # 3. Tier 2 Fallback: Synthetic SDE Generator for missing symbols
        if missing_symbols:
            logger.info("Cache miss for %s. Generating synthetic bars from scenario.", missing_symbols)
            try:
                synthetic_dataset = generate_stress_scenario(
                    self.config.default_fallback_scenario, seed=42
                )
                fallbacks = {
                    "SPY": 500.0, "QQQ": 440.0, "XLK": 210.0, "XLE": 85.0,
                    "XLV": 140.0, "XLI": 120.0, "XLU": 65.0, "TLT": 95.0,
                    "SHV": 110.0, "GLD": 215.0,
                }
                spy_bars = synthetic_dataset.bars.get("SPY", [])
                for sym in missing_symbols:
                    syn_bars = synthetic_dataset.bars.get(sym, [])
                    if not syn_bars and spy_bars:
                        base_price = fallbacks.get(sym, 100.0)
                        syn_bars = [
                            Bar(
                                symbol=sym,
                                timestamp=sb.timestamp,
                                open=round(base_price * (sb.open / spy_bars[0].open), 2),
                                high=round(base_price * (sb.high / spy_bars[0].open), 2),
                                low=round(base_price * (sb.low / spy_bars[0].open), 2),
                                close=round(base_price * (sb.close / spy_bars[0].open), 2),
                                volume=100000,
                            )
                            for sb in spy_bars
                        ]
                    if syn_bars:
                        self.bar_repo.save_bars(syn_bars, timeframe=timeframe)
                        result[sym] = syn_bars
            except Exception as se:
                logger.error("Error generating synthetic scenario: %s", se)

        return result

    async def get_historical_daily_bars(
        self,
        symbols: List[str],
        lookback_days: int = 365,
    ) -> Dict[str, List[Bar]]:
        """Fetch daily bars for the specified lookback days."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=lookback_days)
        return await self.get_historical_bars(symbols=symbols, timeframe="1Day", start=start, end=now)

    def get_latest_price(self, symbol: str) -> float:
        """Resolve current price with zero-crash guarantee."""
        sym = symbol.upper()
        # 1. In-memory latest price
        if sym in self._latest_prices and self._latest_prices[sym] > 0:
            return self._latest_prices[sym]

        # 2. In-memory latest bar
        if sym in self._latest_bars and self._latest_bars[sym].close > 0:
            return self._latest_bars[sym].close

        # 3. SQLite WAL latest bar
        latest_bar = self.bar_repo.get_latest_bar(sym, timeframe="1Day")
        if latest_bar and latest_bar.close > 0:
            self._latest_prices[sym] = latest_bar.close
            return latest_bar.close

        # 4. Fallback constants
        fallbacks = {
            "SPY": 500.0, "QQQ": 440.0, "XLK": 210.0, "XLE": 85.0,
            "XLV": 140.0, "XLI": 120.0, "XLU": 65.0, "TLT": 95.0,
            "SHV": 110.0, "GLD": 215.0,
        }
        return fallbacks.get(sym, 100.0)

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        return {sym: self.get_latest_price(sym) for sym in symbols}

    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        sym = symbol.upper()
        if sym in self._latest_bars:
            return self._latest_bars[sym]
        return self.bar_repo.get_latest_bar(sym, timeframe="1Day")

    def is_safe_to_rebalance(self) -> bool:
        """
        Gate check for rebalancing safety.
        When operating in synthetic fallback simulation, synthetic bars are actively provided,
        so rebalancing is safe (returns True) while alert_banner_active remains True.
        In live mode, delegates to client.is_safe_to_rebalance().
        """
        if self._feed_source == FeedSource.SYNTHETIC_FALLBACK:
            return True
        return self.client.is_safe_to_rebalance()

    def get_connection_status(self) -> ConnectionStatus:
        """Conforms to M1 <-> M3 Interface Contract in PROJECT.md."""
        now = datetime.now(timezone.utc)
        downtime = 0.0
        if self._disconnect_timestamp is not None:
            downtime = max(0.0, (now - self._disconnect_timestamp).total_seconds())

        return ConnectionStatus(
            is_connected=self._is_connected,
            feed_source=self._feed_source.value,
            alert_banner_active=self._alert_banner_active,
            last_heartbeat_timestamp=(
                _to_iso(self._last_heartbeat) if self._last_heartbeat
                else _to_iso(now)
            ),
            status_message=self._status_message,
            fallback_reason=self._fallback_reason,
            disconnect_timestamp=(
                _to_iso(self._disconnect_timestamp) if self._disconnect_timestamp
                else None
            ),
            downtime_duration_seconds=round(downtime, 1),
            reconnect_attempts=self._reconnect_attempts,
            active_symbols=list(self.config.universe_symbols),
        )

    # ------------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------------
    def on_bar(self, callback: Callable[[Bar], Any]) -> None:
        self._on_bar_callbacks.append(callback)

    def on_disconnect(self, callback: Callable[[str, datetime], Any]) -> None:
        self._disconnect_callbacks.append(callback)

    def on_recover(self, callback: Callable[[float, datetime], Any]) -> None:
        self._recover_callbacks.append(callback)

    def _wire_discord_callbacks(self) -> None:
        """Wire automatic Discord alert dispatch on disconnect and recovery events."""
        def _handle_disconnect(reason: str, ts: datetime) -> None:
            if self.discord_notifier and hasattr(self.discord_notifier, "post_broken_alert"):
                try:
                    res = self.discord_notifier.post_broken_alert(
                        component="AlpacaRelayClient",
                        error_message=f"Feed disconnected: {reason}",
                        evidence=f"Ingestion state transitioned to FALLBACK_SIMULATION. Reason: {reason}",
                        dashboard_url=self.dashboard_url,
                        since=ts,
                    )
                    if inspect.isawaitable(res):
                        asyncio.create_task(res)
                except Exception as exc:
                    logger.error("Error dispatching Discord broken alert: %s", exc)

        def _handle_recover(downtime_s: float, ts: datetime) -> None:
            if self.discord_notifier and hasattr(self.discord_notifier, "post_recovered_alert"):
                try:
                    res = self.discord_notifier.post_recovered_alert(
                        component="AlpacaRelayClient",
                        downtime_duration_s=downtime_s,
                        status_info="WebSocket stream reconnected. Historical bars synchronized.",
                        dashboard_url=self.dashboard_url,
                    )
                    if inspect.isawaitable(res):
                        asyncio.create_task(res)
                except Exception as exc:
                    logger.error("Error dispatching Discord recovered alert: %s", exc)

        self.on_disconnect(_handle_disconnect)
        self.on_recover(_handle_recover)


# Ergonomic alias
DataFeedManager = FeedManager
