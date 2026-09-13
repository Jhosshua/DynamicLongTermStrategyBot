"""
bot
~~~

Systematic Long-Term Holding & Rebalancing Bot Package.
Provides Virtual Paper Account Engine, Resilient Feed Manager, and Dynamic Strategy Service.
"""

from bot.feed_manager import (
    ConnectionStatus,
    DataFeedManager,
    FeedManager,
    FeedManagerConfig,
    FeedSource,
)
from bot.paper_account import (
    ExecutionReport,
    PaperAccountConfig,
    PaperAccountManager,
    PaperOrderSide,
    PaperTrade,
    PortfolioSummary,
    PositionDetail,
)
from bot.service import (
    DynamicStrategyService,
    ManualRebalanceResult,
    ServiceConfig,
    ServiceState,
    ServiceStatus,
)

__all__ = [
    # Paper Account
    "PaperAccountManager",
    "PaperAccountConfig",
    "PaperTrade",
    "PortfolioSummary",
    "PositionDetail",
    "ExecutionReport",
    "PaperOrderSide",
    # Feed Manager
    "FeedManager",
    "DataFeedManager",
    "ConnectionStatus",
    "FeedSource",
    "FeedManagerConfig",
    # Strategy Service
    "DynamicStrategyService",
    "ServiceConfig",
    "ServiceStatus",
    "ServiceState",
    "ManualRebalanceResult",
]
