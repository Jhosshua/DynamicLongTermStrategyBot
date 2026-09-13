"""
strategy_engine.ingestion
~~~~~~~~~~~~~~~~~~~~~~~~~

Market data ingestion clients, token-bucket rate limiting, decoupled WebSocket
streaming, and connection resiliency state machine for AlpacaRelay.
"""

from strategy_engine.ingestion.client import AlpacaRelayClient
from strategy_engine.ingestion.rest_client import (
    AlpacaRelayError,
    AlpacaRelayRestClient,
    RelayAuthError,
    RelayAuthenticationError,
    RelayRateLimitError,
    RelayRequestError,
    RelayUpstreamError,
    TokenBucketRateLimiter,
)
from strategy_engine.ingestion.state_machine import (
    IngestionStateMachine,
    LifecycleState,
    StateTransitionRecord,
)
from strategy_engine.ingestion.ws_client import (
    AlpacaRelayWSClient,
    LifecycleEvent,
    RelayClientError,
    RelayConnectionError,
    RelayHandshakeError,
    RelaySlowClientEvictionError,
    RelaySubscriptionError,
    StreamEvent,
    SubscriptionAckEvent,
    WSConnectionState,
)

__all__ = [
    # Unified client
    "AlpacaRelayClient",
    # REST client & components
    "AlpacaRelayRestClient",
    "TokenBucketRateLimiter",
    "AlpacaRelayError",
    "RelayAuthError",
    "RelayAuthenticationError",
    "RelayRateLimitError",
    "RelayUpstreamError",
    "RelayRequestError",
    # WS client & components
    "AlpacaRelayWSClient",
    "WSConnectionState",
    "LifecycleEvent",
    "SubscriptionAckEvent",
    "StreamEvent",
    "RelayClientError",
    "RelayConnectionError",
    "RelayHandshakeError",
    "RelaySubscriptionError",
    "RelaySlowClientEvictionError",
    # State machine
    "IngestionStateMachine",
    "LifecycleState",
    "StateTransitionRecord",
]
