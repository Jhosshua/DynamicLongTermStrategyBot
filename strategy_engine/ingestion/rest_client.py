"""
strategy_engine.ingestion.rest_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Async REST proxy client for AlpacaRelay with proactive token-bucket rate limiting,
exponential backoff with full jitter for 429/502/503/504 errors, dual auth headers,
auto-pagination generators, and immutable domain model mapping.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import random
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

import httpx

from strategy_engine.core.models import Bar, Quote, Trade

logger = logging.getLogger("strategy_engine.ingestion.rest_client")


# ============================================================================
# Exceptions Hierarchy
# ============================================================================

class AlpacaRelayError(Exception):
    """Base exception for all AlpacaRelay REST client errors."""
    pass


class RelayAuthError(AlpacaRelayError):
    """Raised when authentication fails (HTTP 401). Never retried."""
    pass


# Alias for compatibility with varying test naming conventions
RelayAuthenticationError = RelayAuthError


class RelayRateLimitError(AlpacaRelayError):
    """Raised when rate limit is exceeded (HTTP 429) after retries are exhausted."""
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        response: Optional[httpx.Response] = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.response = response


class RelayUpstreamError(AlpacaRelayError):
    """Raised when upstream relay/Alpaca fails (HTTP 502/503/504) after retries exhausted."""
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        response: Optional[httpx.Response] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class RelayRequestError(AlpacaRelayError):
    """Raised on non-retryable 4xx client errors (400, 404, 422, etc.)."""
    def __init__(
        self,
        message: str,
        status_code: int,
        response: Optional[httpx.Response] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


# ============================================================================
# Token Bucket Rate Limiter
# ============================================================================

class TokenBucketRateLimiter:
    """Thread-safe and coroutine-safe asynchronous Token Bucket Rate Limiter.
    
    Enforces a fleet-safe client-side rate ceiling (default 30 req/min) to prevent
    exhausting the shared 200 req/min upstream Alpaca key across all 13+ bots.
    """

    def __init__(
        self,
        rate_limit_per_minute: float = 30.0,
        burst_capacity: float = 10.0,
    ):
        if rate_limit_per_minute <= 0:
            raise ValueError(f"rate_limit_per_minute must be positive, got {rate_limit_per_minute}")
        if burst_capacity < 1.0:
            raise ValueError(f"burst_capacity must be >= 1.0, got {burst_capacity}")

        self.rate_per_sec = rate_limit_per_minute / 60.0
        self.rate_limit_per_minute = float(rate_limit_per_minute)
        self.capacity = float(burst_capacity)
        self.tokens = float(burst_capacity)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def available_tokens(self) -> float:
        """Current available tokens (non-locking estimate)."""
        now = time.monotonic()
        elapsed = now - self._last_update
        return min(self.capacity, self.tokens + elapsed * self.rate_per_sec)

    async def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens, sleeping asynchronously if token reserve is insufficient.
        
        Returns the duration waited in seconds (0.0 if tokens were immediately available).
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            needed = tokens - self.tokens
            wait_time = needed / self.rate_per_sec
            self.tokens = 0.0
            self._last_update = now + wait_time

        await asyncio.sleep(wait_time)
        return wait_time

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking attempt to acquire tokens. Returns True if acquired, False otherwise."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def reset(self) -> None:
        """Reset token bucket to full capacity."""
        self.tokens = self.capacity
        self._last_update = time.monotonic()


# ============================================================================
# AlpacaRelay REST Client
# ============================================================================

class AlpacaRelayRestClient:
    """Production-grade asynchronous REST client for AlpacaRelay proxy.
    
    Features:
    - URL routing for /data/v2/... and /health
    - Dual auth headers (X-Relay-Token and APCA-API-KEY-ID)
    - Auto-pagination generators traversing next_page_token
    - Token bucket rate limiting (default 30 req/min)
    - Full-jitter exponential backoff for 429 and 502/503/504
    - Parsing responses into immutable Bar, Quote, and Trade domain models
    """

    RETRYABLE_STATUS_CODES: Set[int] = {429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str = "https://alpacarelay-production.up.railway.app",
        relay_token: str = "",
        auth_header_mode: str = "relay",  # "relay", "apca", or "both"
        rate_limit_per_minute: float = 30.0,
        burst_capacity: float = 10.0,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = 20.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.raw_base_url = base_url.rstrip("/")
        # Normalize server base URL
        if self.raw_base_url.endswith("/data"):
            self.server_url = self.raw_base_url[:-5]
        else:
            self.server_url = self.raw_base_url

        self.relay_token = relay_token
        self.auth_header_mode = auth_header_mode.lower()
        self.rate_limiter = TokenBucketRateLimiter(
            rate_limit_per_minute=rate_limit_per_minute,
            burst_capacity=burst_capacity,
        )
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self._custom_client = client
        self._client: Optional[httpx.AsyncClient] = client

    def get_auth_headers(self) -> Dict[str, str]:
        """Generate authentication headers based on configured mode."""
        headers: Dict[str, str] = {}
        if not self.relay_token:
            return headers

        if self.auth_header_mode in ("relay", "both"):
            headers["X-Relay-Token"] = self.relay_token
        if self.auth_header_mode in ("apca", "both"):
            headers["APCA-API-KEY-ID"] = self.relay_token
        return headers

    def _build_url(self, path: str) -> str:
        """Construct full URL cleanly mapping /health and /data/v2/... routes."""
        clean_path = path.strip("/")
        if clean_path == "health":
            return f"{self.server_url}/health"
        if clean_path.startswith("data/"):
            return f"{self.server_url}/{clean_path}"
        # Prepend /data for market data routes
        return f"{self.server_url}/data/{clean_path}"

    async def _get_client(self) -> httpx.AsyncClient:
        """Retrieve or create the underlying httpx.AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client and not self._client.is_closed and self._client is not self._custom_client:
            await self._client.aclose()

    async def __aenter__(self) -> AlpacaRelayRestClient:
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute rate-limited HTTP request with exponential backoff and jitter."""
        url = self._build_url(path)
        req_headers = {"Accept": "application/json"}
        req_headers.update(self.get_auth_headers())
        if headers:
            req_headers.update(headers)

        clean_params = {k: v for k, v in (params or {}).items() if v is not None}

        attempt = 0
        while True:
            # 1. Proactive Rate Limiting (acquire token bucket slot)
            await self.rate_limiter.acquire(1.0)

            client = await self._get_client()
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    params=clean_params,
                    headers=req_headers,
                )

                # 2. Immediate Success
                if response.status_code == 200:
                    try:
                        return response.json()
                    except Exception as e:
                        raise RelayRequestError(f"Failed to parse JSON response: {e}", 200, response)

                # 3. Non-Retryable Authentication Error (401)
                if response.status_code == 401:
                    err_msg = "Missing or bad relay token"
                    try:
                        err_msg = response.json().get("relay_error", response.text)
                    except Exception:
                        err_msg = response.text
                    raise RelayAuthError(f"Authentication failed (HTTP 401): {err_msg}")

                # 4. Non-Retryable Client Errors (400, 404, 422)
                if response.status_code < 500 and response.status_code != 429:
                    raise RelayRequestError(
                        f"Request failed with client error {response.status_code}: {response.text}",
                        status_code=response.status_code,
                        response=response,
                    )

                # 5. Retryable Errors (429, 502, 503, 504)
                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    attempt += 1
                    if attempt > self.max_retries:
                        if response.status_code == 429:
                            retry_after_hdr = response.headers.get("Retry-After")
                            r_after = float(retry_after_hdr) if retry_after_hdr and retry_after_hdr.isdigit() else None
                            raise RelayRateLimitError(
                                f"Rate limit exceeded (HTTP 429) after {self.max_retries} retries",
                                retry_after=r_after,
                                response=response,
                            )
                        raise RelayUpstreamError(
                            f"Upstream server error (HTTP {response.status_code}) after {self.max_retries} retries",
                            status_code=response.status_code,
                            response=response,
                        )

                    backoff = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
                    jitter = random.uniform(0.0, 1.0)
                    delay = backoff + jitter

                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after) + random.uniform(0.0, 0.5))
                        except ValueError:
                            pass

                    logger.warning(
                        "AlpacaRelay request %s %s returned status %d. Retry %d/%d in %.2fs",
                        method, url, response.status_code, attempt, self.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise RelayUpstreamError(
                        f"AlpacaRelay server error: {e}",
                        status_code=response.status_code,
                        response=response,
                    ) from e

            except (httpx.TransportError, httpx.TimeoutException) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise RelayUpstreamError(
                        f"Transport error after {self.max_retries} retries: {exc}",
                        status_code=502,
                    ) from exc

                delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1))) + random.uniform(0.0, 1.0)
                logger.warning(
                    "AlpacaRelay transport error on %s %s: %s. Retry %d/%d in %.2fs",
                    method, url, exc, attempt, self.max_retries, delay,
                )
                await asyncio.sleep(delay)

    # ========================================================================
    # Public API Methods
    # ========================================================================

    async def get_health(self) -> Dict[str, Any]:
        """Fetch relay health check status (GET /health)."""
        client = await self._get_client()
        url = f"{self.server_url}/health"
        resp = await client.get(url)
        if resp.status_code != 200:
            raise RelayUpstreamError(f"Health check failed with status {resp.status_code}: {resp.text}")
        return resp.json()

    async def is_upstream_connected(self) -> bool:
        """Check if relay's upstream connection to Alpaca is connected."""
        try:
            health = await self.get_health()
            return health.get("upstream") == "connected"
        except Exception:
            return False

    async def iter_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        limit: int = 1000,
        adjustment: str = "all",
        feed: str = "sip",
        sort: str = "asc",
        asof: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> AsyncIterator[Bar]:
        """Asynchronous generator auto-paginating historical bars for a single symbol."""
        if symbol is None or not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"Symbol must be a non-empty string (got {symbol!r})")
        clean_symbol = symbol.strip()

        page_token: Optional[str] = None
        pages_fetched = 0

        while True:
            params: Dict[str, Any] = {
                "timeframe": timeframe,
                "limit": limit,
                "adjustment": adjustment,
                "feed": feed,
                "sort": sort,
            }
            if start:
                params["start"] = start.isoformat() if isinstance(start, datetime) else str(start)
            if end:
                params["end"] = end.isoformat() if isinstance(end, datetime) else str(end)
            if asof:
                params["asof"] = asof
            if page_token:
                params["page_token"] = page_token

            data = await self._request("GET", f"v2/stocks/{clean_symbol}/bars", params=params)
            raw_bars = data.get("bars") or []
            for bar_dict in raw_bars:
                yield Bar.from_alpaca(bar_dict, symbol=clean_symbol)

            pages_fetched += 1
            page_token = data.get("next_page_token")
            if not page_token or (max_pages is not None and pages_fetched >= max_pages):
                break

    async def get_bars(
        self,
        symbol: Union[str, List[str]] = None,
        timeframe: str = "1Day",
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        limit: int = 1000,
        adjustment: str = "all",
        feed: str = "sip",
        sort: str = "asc",
        asof: Optional[str] = None,
        max_pages: Optional[int] = None,
        symbols: Optional[Union[str, List[str]]] = None,
    ) -> Union[List[Bar], Dict[str, List[Bar]]]:
        """Fetch all historical bars for a single symbol (or multiple symbols), auto-paginating to completion."""
        target = symbols if symbols is not None else symbol
        if target is None:
            raise ValueError("Symbol must be a non-empty string (got None)")
        if isinstance(target, str):
            if not target.strip():
                raise ValueError("Symbol must be a non-empty string (got empty string)")
            clean_sym = target.strip()
            bars: List[Bar] = []
            async for bar in self.iter_bars(
                symbol=clean_sym,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit,
                adjustment=adjustment,
                feed=feed,
                sort=sort,
                asof=asof,
                max_pages=max_pages,
            ):
                bars.append(bar)
            return bars
        elif isinstance(target, (list, tuple, set)):
            if not target:
                raise ValueError("Symbols list cannot be empty")
            clean_syms: List[str] = []
            for s in target:
                if s is None or not isinstance(s, str) or not s.strip():
                    raise ValueError(f"All symbols must be non-empty strings (got {s!r})")
                clean_syms.append(s.strip())
            return await self.get_multi_bars(
                symbols=clean_syms,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit,
                adjustment=adjustment,
                feed=feed,
                sort=sort,
                asof=asof,
                max_pages=max_pages,
            )
        else:
            raise ValueError(f"Symbol must be a non-empty string (got {type(target).__name__})")

    async def get_multi_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        limit: int = 1000,
        adjustment: str = "all",
        feed: str = "sip",
        sort: str = "asc",
        asof: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> Dict[str, List[Bar]]:
        """Fetch historical bars for multiple symbols, auto-paginating through next_page_token."""
        if symbols is None or not isinstance(symbols, (list, tuple, set)):
            raise ValueError(f"Symbols must be a list of non-empty strings (got {symbols!r})")
        clean_symbols: List[str] = []
        for s in symbols:
            if s is None or not isinstance(s, str) or not s.strip():
                raise ValueError(f"All symbols must be non-empty strings (got {s!r})")
            clean_symbols.append(s.strip())
        symbols = clean_symbols

        results: Dict[str, List[Bar]] = {sym: [] for sym in symbols}
        if not symbols:
            return results

        page_token: Optional[str] = None
        pages_fetched = 0
        symbols_param = ",".join(symbols)

        while True:
            params: Dict[str, Any] = {
                "symbols": symbols_param,
                "timeframe": timeframe,
                "limit": limit,
                "adjustment": adjustment,
                "feed": feed,
                "sort": sort,
            }
            if start:
                params["start"] = start.isoformat() if isinstance(start, datetime) else str(start)
            if end:
                params["end"] = end.isoformat() if isinstance(end, datetime) else str(end)
            if asof:
                params["asof"] = asof
            if page_token:
                params["page_token"] = page_token

            data = await self._request("GET", "v2/stocks/bars", params=params)
            raw_bars_dict = data.get("bars") or {}
            for sym, bar_list in raw_bars_dict.items():
                if sym not in results:
                    results[sym] = []
                if isinstance(bar_list, list):
                    for raw_b in bar_list:
                        results[sym].append(Bar.from_alpaca(raw_b, symbol=sym))

            pages_fetched += 1
            page_token = data.get("next_page_token")
            if not page_token or (max_pages is not None and pages_fetched >= max_pages):
                break

        return results

    async def get_historical_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        limit: int = 1000,
        adjustment: str = "all",
        feed: str = "sip",
    ) -> Dict[str, List[Bar]]:
        """Protocol-compliant method matching RelayClientProtocol interface contract."""
        if symbols is None or not isinstance(symbols, (list, tuple, set)):
            raise ValueError(f"Symbols must be a list of non-empty strings (got {symbols!r})")
        clean_symbols: List[str] = []
        for s in symbols:
            if s is None or not isinstance(s, str) or not s.strip():
                raise ValueError(f"All symbols must be non-empty strings (got {s!r})")
            clean_symbols.append(s.strip())
        symbols = clean_symbols

        if len(symbols) == 1:
            bars = await self.get_bars(
                symbols[0],
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit,
                adjustment=adjustment,
                feed=feed,
            )
            return {symbols[0]: bars}
        return await self.get_multi_bars(
            symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            adjustment=adjustment,
            feed=feed,
        )

    async def get_latest_bars(
        self,
        symbols: Union[str, List[str]],
        feed: str = "sip",
    ) -> Dict[str, Bar]:
        """Fetch latest bar for one or more symbols (GET /data/v2/stocks/bars/latest)."""
        if isinstance(symbols, str):
            sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
        else:
            sym_list = list(symbols)

        if not sym_list:
            return {}

        params = {"symbols": ",".join(sym_list), "feed": feed}
        data = await self._request("GET", "v2/stocks/bars/latest", params=params)
        raw_dict = data.get("bars") or {}
        results: Dict[str, Bar] = {}
        for sym, bar_data in raw_dict.items():
            if bar_data:
                results[sym] = Bar.from_alpaca(bar_data, symbol=sym)
        return results

    async def get_latest_bar(self, symbol: str, feed: str = "sip") -> Optional[Bar]:
        """Convenience method to retrieve latest bar for a single symbol."""
        res = await self.get_latest_bars([symbol], feed=feed)
        return res.get(symbol)

    async def get_latest_quotes(
        self,
        symbols: Union[str, List[str]],
        feed: str = "sip",
    ) -> Dict[str, Quote]:
        """Fetch latest quote for one or more symbols (GET /data/v2/stocks/quotes/latest)."""
        if isinstance(symbols, str):
            sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
        else:
            sym_list = list(symbols)

        if not sym_list:
            return {}

        params = {"symbols": ",".join(sym_list), "feed": feed}
        data = await self._request("GET", "v2/stocks/quotes/latest", params=params)
        raw_dict = data.get("quotes") or {}
        results: Dict[str, Quote] = {}
        for sym, quote_data in raw_dict.items():
            if quote_data:
                results[sym] = Quote.from_alpaca(quote_data, symbol=sym)
        return results

    async def get_latest_quote(self, symbol: str, feed: str = "sip") -> Optional[Quote]:
        """Convenience method to retrieve latest quote for a single symbol."""
        res = await self.get_latest_quotes([symbol], feed=feed)
        return res.get(symbol)

    async def get_latest_trades(
        self,
        symbols: Union[str, List[str]],
        feed: str = "sip",
    ) -> Dict[str, Trade]:
        """Fetch latest trade for one or more symbols (GET /data/v2/stocks/trades/latest)."""
        if isinstance(symbols, str):
            sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
        else:
            sym_list = list(symbols)

        if not sym_list:
            return {}

        params = {"symbols": ",".join(sym_list), "feed": feed}
        data = await self._request("GET", "v2/stocks/trades/latest", params=params)
        raw_dict = data.get("trades") or {}
        results: Dict[str, Trade] = {}
        for sym, trade_data in raw_dict.items():
            if trade_data:
                results[sym] = Trade.from_alpaca(trade_data, symbol=sym)
        return results

    async def get_latest_trade(self, symbol: str, feed: str = "sip") -> Optional[Trade]:
        """Convenience method to retrieve latest trade for a single symbol."""
        res = await self.get_latest_trades([symbol], feed=feed)
        return res.get(symbol)
