"""
strategy_engine.ingestion.state_machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ingestion lifecycle finite-state coordinator and resilience manager.
Handles upstream connection events, initial health checks, safe stale-data
holding state (STALE_DATA_HOLD), allocation safety guards, and automated
historical REST gap backfilling upon stream reconnection.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import inspect
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from strategy_engine.core.models import Bar, MarketRegime

logger = logging.getLogger("strategy_engine.ingestion.state_machine")


# ============================================================================
# Lifecycle States & Transitions
# ============================================================================

class LifecycleState(str, Enum):
    """Lifecycle states for market data feed health and risk management."""
    INITIALIZING = "INITIALIZING"
    CONNECTING = "CONNECTING"
    SYNCING_HISTORY = "SYNCING_HISTORY"
    MONITORING_STREAM = "MONITORING_STREAM"
    STALE_DATA_HOLD = "STALE_DATA_HOLD"
    DISCONNECTED = "DISCONNECTED"
    SHUTDOWN = "SHUTDOWN"


@dataclass(frozen=True)
class StateTransitionRecord:
    """Immutable audit record of an ingestion state transition."""
    timestamp: datetime
    from_state: LifecycleState
    to_state: LifecycleState
    reason: str


# ============================================================================
# Ingestion State Machine
# ============================================================================

class IngestionStateMachine:
    """Deterministic finite-state coordinator governing feed resiliency and allocation safety."""

    def __init__(
        self,
        stale_timeout_seconds: float = 300.0,
        gap_backfill_callback: Optional[Callable[[datetime, datetime], Coroutine[Any, Any, int]]] = None,
    ):
        self.stale_timeout_seconds = stale_timeout_seconds
        self.gap_backfill_callback = gap_backfill_callback
        self._current_state: LifecycleState = LifecycleState.INITIALIZING
        self._is_upstream_connected: bool = False
        self._last_msg_timestamp: Optional[datetime] = None
        self._last_bar_timestamp: Optional[datetime] = None
        self._disconnect_timestamp: Optional[datetime] = None
        self._transition_history: List[StateTransitionRecord] = []
        self._state_change_callbacks: List[Callable[[LifecycleState, LifecycleState, str], Any]] = []
        self._lock = asyncio.Lock()

    # ========================================================================
    # Status & Properties
    # ========================================================================

    @property
    def current_state(self) -> LifecycleState:
        return self._current_state

    @property
    def is_upstream_connected(self) -> bool:
        return self._is_upstream_connected

    @property
    def last_bar_timestamp(self) -> Optional[datetime]:
        return self._last_bar_timestamp

    @property
    def last_msg_timestamp(self) -> Optional[datetime]:
        return self._last_msg_timestamp

    @property
    def disconnect_timestamp(self) -> Optional[datetime]:
        return self._disconnect_timestamp

    @property
    def transition_history(self) -> List[StateTransitionRecord]:
        return list(self._transition_history)

    def is_safe_to_rebalance(self) -> bool:
        """Strict safety gate: allocations allowed ONLY during MONITORING_STREAM.
        
        When in STALE_DATA_HOLD, DISCONNECTED, or SYNCING_HISTORY, no new portfolio
        allocations or risk-increasing trades may be committed.
        """
        return self._current_state == LifecycleState.MONITORING_STREAM

    def get_market_regime_override(self) -> Optional[MarketRegime]:
        """Returns MarketRegime.STALE_DATA_HOLD when data is stale/disconnected, else None."""
        if self._current_state == LifecycleState.STALE_DATA_HOLD:
            return MarketRegime.STALE_DATA_HOLD
        return None

    def register_state_change_callback(
        self, cb: Callable[[LifecycleState, LifecycleState, str], Any]
    ) -> None:
        """Register a callback invoked on every state transition."""
        self._state_change_callbacks.append(cb)

    # ========================================================================
    # State Transition Logic
    # ========================================================================

    async def _transition_to(self, new_state: LifecycleState, reason: str) -> None:
        """Executes state transition, logs audit record, and triggers callbacks."""
        if self._current_state == new_state:
            return

        prev = self._current_state
        self._current_state = new_state
        rec = StateTransitionRecord(
            timestamp=datetime.now(timezone.utc),
            from_state=prev,
            to_state=new_state,
            reason=reason,
        )
        self._transition_history.append(rec)
        logger.info("Lifecycle transition: %s -> %s (reason: %s)", prev.value, new_state.value, reason)

        for cb in list(self._state_change_callbacks):
            try:
                res = cb(prev, new_state, reason)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                logger.warning("Error in state change callback %s: %s", cb, e)

    async def check_initial_health(self, health_data: Dict[str, Any]) -> LifecycleState:
        """Determines initial state based on GET /health response."""
        async with self._lock:
            upstream_status = health_data.get("upstream")
            if upstream_status == "connected":
                self._is_upstream_connected = True
                await self._transition_to(
                    LifecycleState.MONITORING_STREAM,
                    "Initial health check: upstream connected",
                )
            else:
                self._is_upstream_connected = False
                self._disconnect_timestamp = datetime.now(timezone.utc)
                await self._transition_to(
                    LifecycleState.STALE_DATA_HOLD,
                    f"Initial health check: upstream status '{upstream_status}'",
                )
            return self._current_state

    async def handle_upstream_disconnected(
        self, reason: str = "upstream_disconnected event received"
    ) -> None:
        """Transitions to STALE_DATA_HOLD safely with zero unhandled exceptions."""
        async with self._lock:
            try:
                self._is_upstream_connected = False
                self._disconnect_timestamp = datetime.now(timezone.utc)
                await self._transition_to(LifecycleState.STALE_DATA_HOLD, reason)
            except Exception as e:
                logger.error("Error handling upstream disconnect: %s", e)

    async def handle_upstream_connected(
        self, reason: str = "upstream_connected event received"
    ) -> None:
        """Triggers REST gap backfill if needed, then transitions to MONITORING_STREAM."""
        async with self._lock:
            try:
                self._is_upstream_connected = True
                now = datetime.now(timezone.utc)
                start_gap = self._last_bar_timestamp or self._disconnect_timestamp

                # Trigger backfill if gap > 60 seconds and callback available
                if start_gap and (now - start_gap).total_seconds() > 60 and self.gap_backfill_callback:
                    await self._transition_to(
                        LifecycleState.SYNCING_HISTORY,
                        f"Performing REST gap backfill for gap {(now - start_gap).total_seconds():.1f}s",
                    )
                    try:
                        backfilled = await self.gap_backfill_callback(start_gap, now)
                        logger.info("Gap backfill completed: %s bars restored", backfilled)
                    except Exception as e:
                        logger.warning("Gap backfill encountered error (resuming stream): %s", e)

                self._disconnect_timestamp = None
                await self._transition_to(LifecycleState.MONITORING_STREAM, reason)
            except Exception as e:
                logger.error("Error handling upstream connect: %s", e)

    # ========================================================================
    # Tracking & Watchdog
    # ========================================================================

    def record_bar(self, bar: Bar) -> None:
        """Records a received Bar to track latest bar timestamp and stream freshness."""
        self.record_message(bar.timestamp)
        if self._last_bar_timestamp is None or bar.timestamp > self._last_bar_timestamp:
            self._last_bar_timestamp = bar.timestamp

    def record_message(self, timestamp: Optional[datetime] = None) -> None:
        """Records any received stream message for watchdog tracking."""
        ts = timestamp or datetime.now(timezone.utc)
        self._last_msg_timestamp = ts

    async def check_watchdog(self, current_time: Optional[datetime] = None) -> bool:
        """Detects dead/silent stream (> stale_timeout_seconds) and triggers STALE_DATA_HOLD."""
        if self._current_state != LifecycleState.MONITORING_STREAM:
            return False

        now = current_time or datetime.now(timezone.utc)
        if self._last_msg_timestamp is not None:
            age = (now - self._last_msg_timestamp).total_seconds()
            if age > self.stale_timeout_seconds:
                async with self._lock:
                    await self._transition_to(
                        LifecycleState.STALE_DATA_HOLD,
                        f"Watchdog timeout: no stream messages for {age:.1f}s (> {self.stale_timeout_seconds}s)",
                    )
                return True
        return False

    def get_status_diagnostics(self) -> Dict[str, Any]:
        """Provides operational diagnostics for monitoring and audit logging."""
        now = datetime.now(timezone.utc)
        msg_age = (now - self._last_msg_timestamp).total_seconds() if self._last_msg_timestamp else None
        return {
            "state": self._current_state.value,
            "upstream_connected": self._is_upstream_connected,
            "is_safe_to_rebalance": self.is_safe_to_rebalance(),
            "last_bar_time": self._last_bar_timestamp.isoformat() if self._last_bar_timestamp else None,
            "last_msg_age_seconds": msg_age,
            "disconnected_at": self._disconnect_timestamp.isoformat() if self._disconnect_timestamp else None,
            "total_transitions": len(self._transition_history),
        }
