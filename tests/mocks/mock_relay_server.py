"""In-process mock AlpacaRelay server (Feature 6).
Simulates both REST proxy endpoints (/data/v2/...) and WebSocket streaming
with token authentication, channel subscriptions, upstream lifecycle events,
and slow-client evictions.
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
from aiohttp import web

logger = logging.getLogger("mock_relay_server")


class MockAlpacaRelayServer:
    def __init__(self, token: str = "test-relay-token", feed: str = "sip"):
        self.token = token
        self.feed = feed
        self.upstream_connected = True
        self.clients: Set[web.WebSocketResponse] = set()
        self.subscriptions: Dict[web.WebSocketResponse, Dict[str, Set[str]]] = {}
        self.bars_db: Dict[str, List[dict]] = {}
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.host: str = "127.0.0.1"
        self.port: int = 0
        self._setup_routes()

    def _setup_routes(self):
        self.app.router.add_get("/health", self._handle_health)
        self.app.router.add_get("/data/v2/stocks/bars", self._handle_multi_bars)
        self.app.router.add_get("/data/v2/stocks/{symbol}/bars", self._handle_single_bars)
        self.app.router.add_post("/data/{tail:.*}", self._handle_invalid_method)
        self.app.router.add_put("/data/{tail:.*}", self._handle_invalid_method)
        self.app.router.add_delete("/data/{tail:.*}", self._handle_invalid_method)
        self.app.router.add_get("/", self._handle_root_or_ws)

    def add_mock_bars(self, symbol: str, bars: List[dict]):
        if symbol not in self.bars_db:
            self.bars_db[symbol] = []
        self.bars_db[symbol].extend(bars)

    def _check_auth(self, request: web.Request) -> bool:
        supplied = (
            request.headers.get("X-Relay-Token")
            or request.headers.get("APCA-API-KEY-ID")
            or ""
        )
        return supplied == self.token

    async def _handle_health(self, request: web.Request):
        status = "connected" if self.upstream_connected else "down"
        return web.json_response({
            "upstream": status,
            "feed": self.feed,
            "clients": len(self.clients),
            "symbols": {"bars": len(self.bars_db)},
            "last_upstream_msg_age_s": 0 if self.upstream_connected else 120,
        })

    async def _handle_invalid_method(self, request: web.Request):
        return web.Response(
            status=400,
            text="Failed to open a WebSocket connection: did not receive a valid HTTP request."
        )

    async def _handle_single_bars(self, request: web.Request):
        if not self._check_auth(request):
            return web.json_response({"relay_error": "missing or bad relay token"}, status=401)
        symbol = request.match_info["symbol"]
        limit = int(request.query.get("limit", 1000))
        bars = self.bars_db.get(symbol, [])
        # Slice according to limit
        return web.json_response({
            "bars": bars[:limit],
            "symbol": symbol,
            "next_page_token": None,
        })

    async def _handle_multi_bars(self, request: web.Request):
        if not self._check_auth(request):
            return web.json_response({"relay_error": "missing or bad relay token"}, status=401)
        symbols_raw = request.query.get("symbols", "")
        requested = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        result = {}
        for sym in requested:
            result[sym] = self.bars_db.get(sym, [])
        return web.json_response({
            "bars": result,
            "next_page_token": None,
        })

    async def _handle_root_or_ws(self, request: web.Request):
        if request.headers.get("Upgrade", "").lower() != "websocket":
            return web.Response(
                status=200,
                text="AlpacaRelay: websocket endpoint. GET /health for status.\n"
            )

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)
        self.subscriptions[ws] = {"trades": set(), "quotes": set(), "bars": set()}

        # 1. Send connected banner
        await ws.send_json([{"T": "success", "msg": "connected"}])

        # 2. Wait for auth within 10s
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    action = data.get("action")
                    token = data.get("token") or data.get("key")
                    if action == "auth" and token == self.token:
                        await ws.send_json([{"T": "success", "msg": "authenticated"}])
                    else:
                        await ws.send_json([{"T": "error", "code": 402, "msg": "auth failed"}])
                        await ws.close(code=1008)
                        return ws
                except json.JSONDecodeError:
                    await ws.close(code=1008)
                    return ws
            else:
                await ws.close(code=1008)
                return ws
        except asyncio.TimeoutError:
            await ws.close(code=1000)
            return ws

        # 3. Handle subscriptions and streaming
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        action = data.get("action")
                        if action == "subscribe":
                            for ch in ("trades", "quotes", "bars"):
                                for sym in data.get(ch, []):
                                    self.subscriptions[ws][ch].add(sym)
                            ack = {"T": "subscription"}
                            for ch in ("trades", "quotes", "bars"):
                                ack[ch] = sorted(list(self.subscriptions[ws][ch]))
                            await ws.send_json([ack])
                        elif action == "unsubscribe":
                            for ch in ("trades", "quotes", "bars"):
                                for sym in data.get(ch, []):
                                    self.subscriptions[ws][ch].discard(sym)
                            ack = {"T": "subscription"}
                            for ch in ("trades", "quotes", "bars"):
                                ack[ch] = sorted(list(self.subscriptions[ws][ch]))
                            await ws.send_json([ack])
                    except json.JSONDecodeError:
                        pass  # Silently ignore malformed non-JSON
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.ERROR):
                    break
        finally:
            self.clients.discard(ws)
            self.subscriptions.pop(ws, None)

        return ws

    async def broadcast_lifecycle(self, event_type: str):
        """Broadcasts upstream_connected or upstream_disconnected to all connected clients."""
        if event_type == "upstream_connected":
            self.upstream_connected = True
        elif event_type == "upstream_disconnected":
            self.upstream_connected = False
        payload = [{"T": "relay", "msg": event_type}]
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)
            self.subscriptions.pop(ws, None)

    async def broadcast_bar(self, symbol: str, bar_dict: dict):
        """Broadcasts a bar update to clients subscribed to bars for symbol."""
        msg = [{"T": "b", "S": symbol, **bar_dict}]
        dead = []
        for ws in list(self.clients):
            subs = self.subscriptions.get(ws, {}).get("bars", set())
            if "*" in subs or symbol in subs:
                try:
                    await ws.send_json(msg)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)
            self.subscriptions.pop(ws, None)

    async def simulate_slow_client_eviction(self, ws: web.WebSocketResponse):
        """Evicts a slow client with close code 1013 (too slow)."""
        await ws.close(code=1013, message=b"too slow")
        self.clients.discard(ws)
        self.subscriptions.pop(ws, None)

    async def start(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, host, port)
        await self.site.start()
        # Find assigned port
        for s in self.runner.sites:
            if isinstance(s, web.TCPSite):
                # Retrieve actual socket port
                for sock in s._server.sockets:
                    self.port = sock.getsockname()[1]
                    break
        logger.info("MockAlpacaRelayServer listening on %s:%d", self.host, self.port)

    async def stop(self):
        for ws in list(self.clients):
            try:
                await ws.close(code=1000)
            except Exception:
                pass
        self.clients.clear()
        self.subscriptions.clear()
        if self.runner:
            await self.runner.cleanup()

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
