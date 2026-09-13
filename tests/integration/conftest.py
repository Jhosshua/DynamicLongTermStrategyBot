"""
tests.integration.conftest
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pytest fixtures and Extended Mock Relay Server for M2 integration tests.
Extends MockAlpacaRelayServer with pagination, status code injection (429/502),
latest bar/quote/trade endpoints, and upstream status toggling.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional
from aiohttp import web
import pytest

from tests.mocks.mock_relay_server import MockAlpacaRelayServer

logger = logging.getLogger("test_conftest")


class ExtendedMockRelayServer(MockAlpacaRelayServer):
    """Extended mock server for comprehensive M2 integration testing."""

    def __init__(self, token: str = "test-relay-token", feed: str = "sip"):
        super().__init__(token=token, feed=feed)
        self.injected_status: Optional[int] = None
        self.injected_countdown: int = 0
        self.injected_retry_after: Optional[str] = None
        self.quotes_db: Dict[str, List[dict]] = {}
        self.trades_db: Dict[str, List[dict]] = {}

    def _setup_routes(self):
        super()._setup_routes()
        # Override single and multi bars with paginated & injectable versions
        self.app.router.add_get("/data/v2/stocks/bars/latest", self._handle_latest_bars)
        self.app.router.add_get("/data/v2/stocks/quotes/latest", self._handle_latest_quotes)
        self.app.router.add_get("/data/v2/stocks/trades/latest", self._handle_latest_trades)

    def set_simulate_status(self, status: int, count: int = 1, retry_after: Optional[str] = None):
        """Inject HTTP status code for next N requests."""
        self.injected_status = status
        self.injected_countdown = count
        self.injected_retry_after = retry_after

    def set_upstream_status(self, connected: bool):
        """Toggle upstream connectivity reported in /health."""
        self.upstream_connected = connected

    def add_mock_quotes(self, symbol: str, quotes: List[dict]):
        if symbol not in self.quotes_db:
            self.quotes_db[symbol] = []
        self.quotes_db[symbol].extend(quotes)

    def add_mock_trades(self, symbol: str, trades: List[dict]):
        if symbol not in self.trades_db:
            self.trades_db[symbol] = []
        self.trades_db[symbol].extend(trades)

    async def _handle_single_bars(self, request: web.Request):
        if not self._check_auth(request):
            return web.json_response({"relay_error": "missing or bad relay token"}, status=401)

        # Injected status check
        if self.injected_countdown > 0 and self.injected_status is not None:
            self.injected_countdown -= 1
            headers = {}
            if self.injected_retry_after:
                headers["Retry-After"] = str(self.injected_retry_after)
            return web.json_response(
                {"message": f"Injected status {self.injected_status}"},
                status=self.injected_status,
                headers=headers,
            )

        symbol = request.match_info["symbol"]
        limit = int(request.query.get("limit", 1000))
        page_token = request.query.get("page_token")
        offset = int(page_token) if (page_token and page_token.isdigit()) else 0

        all_bars = self.bars_db.get(symbol, [])
        sliced = all_bars[offset : offset + limit]
        next_token = str(offset + limit) if (offset + limit) < len(all_bars) else None

        return web.json_response({
            "bars": sliced,
            "symbol": symbol,
            "next_page_token": next_token,
        })

    async def _handle_multi_bars(self, request: web.Request):
        if not self._check_auth(request):
            return web.json_response({"relay_error": "missing or bad relay token"}, status=401)

        if self.injected_countdown > 0 and self.injected_status is not None:
            self.injected_countdown -= 1
            headers = {}
            if self.injected_retry_after:
                headers["Retry-After"] = str(self.injected_retry_after)
            return web.json_response(
                {"message": f"Injected status {self.injected_status}"},
                status=self.injected_status,
                headers=headers,
            )

        symbols_raw = request.query.get("symbols", "")
        requested = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        limit = int(request.query.get("limit", 1000))
        page_token = request.query.get("page_token")
        offset = int(page_token) if (page_token and page_token.isdigit()) else 0

        result = {}
        has_more = False
        for sym in requested:
            all_b = self.bars_db.get(sym, [])
            sliced = all_b[offset : offset + limit]
            result[sym] = sliced
            if (offset + limit) < len(all_b):
                has_more = True

        next_token = str(offset + limit) if has_more else None
        return web.json_response({
            "bars": result,
            "next_page_token": next_token,
        })

    async def _handle_latest_bars(self, request: web.Request):
        if not self._check_auth(request):
            return web.json_response({"relay_error": "missing or bad relay token"}, status=401)
        symbols_raw = request.query.get("symbols", "")
        requested = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        result = {}
        for sym in requested:
            bars = self.bars_db.get(sym, [])
            result[sym] = bars[-1] if bars else None
        return web.json_response({"bars": result})

    async def _handle_latest_quotes(self, request: web.Request):
        if not self._check_auth(request):
            return web.json_response({"relay_error": "missing or bad relay token"}, status=401)
        symbols_raw = request.query.get("symbols", "")
        requested = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        result = {}
        for sym in requested:
            quotes = self.quotes_db.get(sym, [])
            result[sym] = quotes[-1] if quotes else None
        return web.json_response({"quotes": result})

    async def _handle_latest_trades(self, request: web.Request):
        if not self._check_auth(request):
            return web.json_response({"relay_error": "missing or bad relay token"}, status=401)
        symbols_raw = request.query.get("symbols", "")
        requested = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        result = {}
        for sym in requested:
            trades = self.trades_db.get(sym, [])
            result[sym] = trades[-1] if trades else None
        return web.json_response({"trades": result})

    async def broadcast_quote(self, symbol: str, quote_dict: dict):
        """Broadcasts a quote update to subscribed clients."""
        msg = [{"T": "q", "S": symbol, **quote_dict}]
        for ws in list(self.clients):
            subs = self.subscriptions.get(ws, {}).get("quotes", set())
            if "*" in subs or symbol in subs:
                try:
                    await ws.send_json(msg)
                except Exception:
                    pass

    async def broadcast_trade(self, symbol: str, trade_dict: dict):
        """Broadcasts a trade update to subscribed clients."""
        msg = [{"T": "t", "S": symbol, **trade_dict}]
        for ws in list(self.clients):
            subs = self.subscriptions.get(ws, {}).get("trades", set())
            if "*" in subs or symbol in subs:
                try:
                    await ws.send_json(msg)
                except Exception:
                    pass


@pytest.fixture
async def mock_relay():
    """Provides a started, dynamic in-process mock relay server."""
    server = ExtendedMockRelayServer(token="secret-relay-token-123")
    await server.start()
    try:
        yield server
    finally:
        await server.stop()
