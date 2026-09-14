"""Put a service into a simulated LIVE relay feed without touching the network.

Production refuses to evaluate or trade unless the feed is a live AlpacaRelay
stream. Tests that exercise trading must therefore simulate a genuinely live
feed: state machine in MONITORING_STREAM, feed manager on ALPACA_RELAY, and
historical bars served from a local stub instead of a real HTTP call.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from bot.feed_manager import FeedSource
from strategy_engine.core.models import Bar


async def simulate_live_feed(service, bars: Optional[Dict[str, List[Bar]]] = None) -> Dict[str, List[Bar]]:
    fm = service.feed_manager
    symbols = list(service.service_config.symbols)

    if bars is None:
        # Build deterministic daily bars while still in fallback (memory only, no network).
        assert fm.feed_source == FeedSource.SYNTHETIC_FALLBACK
        bars = await fm.get_historical_bars(symbols=symbols, timeframe="1Day")

    async def _stub_historical_bars(symbols, timeframe="1Day", start=None, end=None):
        return {s: list(bars[s]) for s in symbols if s in bars}

    fm.get_historical_bars = _stub_historical_bars

    await service.client.state_machine.handle_upstream_connected("test: simulated live relay")
    await fm._transition_to_live("test: simulated live relay")
    # Live mode clears in-memory prices; seed them from the stubbed bars.
    fm._latest_prices.update({s: b[-1].close for s, b in bars.items() if b})
    return bars
