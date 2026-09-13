"""
strategy_engine.ingestion.client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unified AlpacaRelay Client coordinating REST ingestion, WebSocket streaming,
and the Ingestion State Machine to satisfy RelayClientProtocol.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from strategy_engine.core.models import Bar, MarketRegime
from strategy_engine.ingestion.rest_client import AlpacaRelayRestClient
from strategy_engine.ingestion.state_machine import IngestionStateMachine, LifecycleState
from strategy_engine.ingestion.ws_client import AlpacaRelayWSClient, LifecycleEvent

logger = logging.getLogger("strategy_engine.ingestion.client")


class AlpacaRelayClient:
    """Unified client conforming to RelayClientProtocol."""

    def __init__(
        self,
        base_url: str = "https://alpacarelay-production.up.railway.app",
        ws_url: str = "ws://localhost:8765",
        relay_token: str = "",
        rate_limit_per_minute: float = 30.0,
        burst_capacity: float = 10.0,
    ):
        self.rest_client = AlpacaRelayRestClient(
            base_url=base_url,
            relay_token=relay_token,
            rate_limit_per_minute=rate_limit_per_minute,
            burst_capacity=burst_capacity,
        )
        self.ws_client = AlpacaRelayWSClient(
            url=ws_url,
            token=relay_token,
        )
        self.state_machine = IngestionStateMachine(
            gap_backfill_callback=self._backfill_gap,
        )
        # Wire WebSocket lifecycle events to state machine
        self.ws_client.on_lifecycle(self._on_ws_lifecycle)
        self.ws_client.on_bar(self._on_ws_bar)

    async def _on_ws_lifecycle(self, event: LifecycleEvent) -> None:
        await self.handle_lifecycle_event(event.event_type)

    def _on_ws_bar(self, bar: Bar) -> None:
        self.state_machine.record_bar(bar)

    async def _backfill_gap(self, start: datetime, end: datetime) -> int:
        """Backfill missing bars across the disconnect gap window."""
        symbols = list(self.ws_client._desired_subscriptions.get("bars", []))
        if not symbols:
            return 0
        try:
            bars_map = await self.rest_client.get_historical_bars(
                symbols=symbols,
                timeframe="1Min",
                start=start,
                end=end,
            )
            total = sum(len(b_list) for b_list in bars_map.values())
            return total
        except Exception as e:
            logger.warning("Failed to backfill gap bars: %s", e)
            return 0

    async def get_historical_bars(
        self, symbols: List[str], timeframe: str, start: datetime, end: datetime
    ) -> Dict[str, List[Bar]]:
        """Fetch historical bars via REST proxy."""
        return await self.rest_client.get_historical_bars(
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
        )

    async def connect_stream(self, symbols: List[str], channels: List[str]) -> None:
        """Connect WebSocket stream and subscribe to specified channels."""
        # Check initial health first
        try:
            health = await self.rest_client.get_health()
            await self.state_machine.check_initial_health(health)
        except Exception as e:
            logger.warning("Initial health check failed: %s", e)
            await self.state_machine.handle_upstream_disconnected("Health check failed on connect_stream")

        await self.ws_client.connect()
        await self.ws_client.connect_stream(symbols=symbols, channels=channels)

    async def handle_lifecycle_event(self, event_type: str) -> None:
        """Handle upstream lifecycle events from relay."""
        if event_type == "upstream_disconnected":
            await self.state_machine.handle_upstream_disconnected()
        elif event_type == "upstream_connected":
            await self.state_machine.handle_upstream_connected()
        else:
            logger.info("Received unknown lifecycle event: %s", event_type)

    def get_connection_status(self) -> str:
        """Returns the current state of the ingestion state machine."""
        return self.state_machine.current_state.value

    def is_safe_to_rebalance(self) -> bool:
        """Gate check for rebalancing allocation safety."""
        return self.state_machine.is_safe_to_rebalance()

    def get_market_regime_override(self) -> Optional[MarketRegime]:
        """Override quantitative regime when in safe holding state."""
        return self.state_machine.get_market_regime_override()

    async def close(self) -> None:
        """Graceful shutdown of both REST and WS connections."""
        await self.ws_client.disconnect()
        await self.rest_client.close()
