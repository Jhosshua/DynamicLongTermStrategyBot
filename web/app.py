"""
web.app
~~~~~~~

FastAPI Application Server for Milestone 3 Mobile-Centric Operator Dashboard.
Provides unauthenticated public health, real-time portfolio snapshots,
operator control endpoints, and Server-Sent Events (SSE) streaming.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from bot.feed_manager import ConnectionStatus
from bot.paper_account import PaperAccountManager, PortfolioSummary
from bot.service import (
    DynamicStrategyService,
    ManualRebalanceResult,
    ServiceState,
    ServiceStatus,
)

logger = logging.getLogger("web.app")

DEFAULT_BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATES_DIR = DEFAULT_BASE_DIR / "templates"
DEFAULT_STATIC_DIR = DEFAULT_BASE_DIR / "static"


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------

class OperatorRebalanceRequest(BaseModel):
    """Payload for operator-triggered manual rebalance."""
    model_config = ConfigDict(extra="ignore")
    force: bool = Field(
        default=False,
        description="If True, forces rebalance execution even if PAUSED or feed is in fallback."
    )


class OperatorActionResponse(BaseModel):
    """Generic response for operator state changes."""
    model_config = ConfigDict(extra="ignore")
    success: bool
    state: str
    is_paused: bool
    message: str
    status: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ErrorDetail(BaseModel):
    code: str
    message: str
    status_code: int
    timestamp: str
    path: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
    detail: Optional[Any] = None


# ---------------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------------

def get_strategy_service(request: Request) -> DynamicStrategyService:
    """Dependency resolver for DynamicStrategyService from app.state."""
    service: Optional[DynamicStrategyService] = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Strategy service is not initialized or unavailable.",
        )
    return service


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

def create_app(
    service: Optional[DynamicStrategyService] = None,
    template_dir: Optional[Union[str, Path]] = None,
    static_dir: Optional[Union[str, Path]] = None,
) -> FastAPI:
    """
    Construct and configure the Milestone 3 FastAPI Dashboard application.
    """
    tmpl_dir = Path(template_dir) if template_dir else DEFAULT_TEMPLATES_DIR
    st_dir = Path(static_dir) if static_dir else DEFAULT_STATIC_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # --- Startup ---
        logger.info("Initializing Dashboard Server lifespan...")
        svc: Optional[DynamicStrategyService] = getattr(app.state, "service", None)
        svc_task = None
        if svc is None:
            logger.info("Instantiating default DynamicStrategyService instance...")
            svc = DynamicStrategyService()
            app.state.service = svc

        auto_start = getattr(app.state, "auto_start_service", None)
        if auto_start is None:
            auto_start = not getattr(app.state, "_service_injected", False)

        is_running = svc._service_state in (ServiceState.RUNNING, ServiceState.PAUSED)
        if auto_start and not is_running and svc._service_state == ServiceState.INITIALIZING:
            logger.info("Launching DynamicStrategyService in background task...")
            svc_task = asyncio.create_task(svc.start(), name="dynamic_strategy_service")

        yield

        # --- Shutdown ---
        logger.info("Executing Dashboard Server graceful shutdown...")
        if svc_task and not svc_task.done():
            svc_task.cancel()
            try:
                await svc_task
            except asyncio.CancelledError:
                pass
        if not getattr(app.state, "_service_injected", False) and svc:
            await svc.shutdown(reason="FastAPI application shutdown")
        logger.info("Dashboard Server shutdown complete.")

    app = FastAPI(
        title="DynamicLongTermStrategyBot Dashboard",
        description="Mobile-Centric Light & Airy Operator Dashboard & API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Attach state
    app.state.service = service
    app.state._service_injected = service is not None
    app.state.template_dir = tmpl_dir
    app.state.static_dir = st_dir

    # 1. CORS Middleware (Public access per R5)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Security & Cache-Control Headers Middleware
    @app.middleware("http")
    async def add_security_and_cache_headers(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Apply no-cache to dynamic API and health endpoints
        if request.url.path.startswith("/api/") or request.url.path == "/health":
            if request.url.path == "/api/events":
                response.headers["Cache-Control"] = "no-cache"
            else:
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    # 3. Exception Handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        err = ErrorDetail(
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            status_code=exc.status_code,
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": err.model_dump(),
                "detail": str(exc.detail),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        err = ErrorDetail(
            code="VALIDATION_ERROR",
            message="Invalid request parameters or payload.",
            status_code=422,
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=request.url.path,
            details=exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": err.model_dump(),
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server exception on %s: %s", request.url.path, exc)
        err = ErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected internal server error occurred.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=request.url.path,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": err.model_dump(),
                "detail": "Internal server error",
            },
        )

    # 4. Static Files Mounting
    if st_dir.exists():
        app.mount("/static", StaticFiles(directory=str(st_dir)), name="static")

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse, tags=["Dashboard"])
    async def get_dashboard(request: Request):
        """Render the Mobile-Centric Operator Dashboard."""
        index_path = tmpl_dir / "dashboard.html"
        if index_path.exists():
            html_raw = index_path.read_text(encoding="utf-8")
            # Attempt Jinja2 template rendering if dynamic context is available
            svc: Optional[DynamicStrategyService] = getattr(request.app.state, "service", None)
            try:
                import jinja2
                if svc:
                    status_obj = svc.get_service_status()
                    conn_obj = svc.feed_manager.get_connection_status()
                    symbols = svc.service_config.symbols
                    prices = svc.feed_manager.get_latest_prices(symbols)
                    portfolio = svc.paper_account.get_portfolio_state(prices)
                    ctx = {
                        "alert_banner_active": conn_obj.alert_banner_active,
                        "daemon_state": status_obj.state.value,
                        "unrealized_pnl": portfolio.unrealized_pnl,
                        "portfolio": portfolio,
                        "regime": status_obj.current_regime,
                    }
                else:
                    ctx = {
                        "alert_banner_active": False,
                        "daemon_state": "RUNNING",
                        "unrealized_pnl": 0.0,
                        "portfolio": None,
                        "regime": "BULL_NORMAL",
                    }
                rendered = jinja2.Template(html_raw).render(**ctx)
                return HTMLResponse(content=rendered, status_code=200)
            except Exception as e:
                logger.debug("Jinja2 render fallback to raw template: %s", e)
                return HTMLResponse(content=html_raw, status_code=200)

        # Clean fallback HTML if template file is not yet deployed
        fallback_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <title>Dynamic Long-Term Strategy Bot — Operator Dashboard</title>
</head>
<body style="font-family: sans-serif; background: #f8fafc; color: #0f172a; padding: 2rem;">
    <h1>Dynamic Long-Term Strategy Bot — Operator Dashboard</h1>
    <div id="alert-banner" class="hidden">AlpacaRelay Disconnected</div>
    <div id="portfolio-nav">Total NAV: $50,000.00</div>
    <p>API Server is running. Dashboard template is loading...</p>
    <p><a href="/health">View Health Status</a> | <a href="/api/portfolio">View Portfolio</a></p>
</body>
</html>"""
        return HTMLResponse(content=fallback_html, status_code=200)

    @app.api_route("/health", methods=["GET", "HEAD"], tags=["Monitoring"])
    async def get_health(service: DynamicStrategyService = Depends(get_strategy_service)):
        """Unauthenticated public health check for Railway deployment."""
        try:
            return service.get_health()
        except Exception as e:
            logger.error("Health check evaluation error: %s", e)
            return {
                "status": "degraded",
                "service": "DynamicLongTermStrategyBot",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @app.get("/api/portfolio", tags=["Portfolio"])
    async def get_portfolio(service: DynamicStrategyService = Depends(get_strategy_service)):
        """Fetch current live portfolio state, holdings, and strategy regime."""
        symbols = service.service_config.symbols
        current_prices = service.feed_manager.get_latest_prices(symbols)

        # Thread-safe portfolio calculation offloaded to threadpool
        portfolio: PortfolioSummary = await asyncio.to_thread(
            service.paper_account.get_portfolio_state, current_prices
        )

        res = portfolio.to_dict()
        # Provide both flat fields and nested "portfolio" key for comprehensive client compatibility
        res["portfolio"] = portfolio.to_dict()
        res["current_regime"] = (
            service._latest_signal.regime.value
            if service._latest_signal
            else "UNKNOWN"
        )
        res["regime"] = res["current_regime"]
        res["connection"] = service.feed_manager.get_connection_status().to_dict()
        res["daemon_state"] = service._service_state.value
        return res

    @app.get("/api/status", tags=["Status"])
    async def get_status(service: DynamicStrategyService = Depends(get_strategy_service)):
        """Fetch consolidated operational and data feed status."""
        status_obj: ServiceStatus = service.get_service_status()
        conn_obj: ConnectionStatus = service.feed_manager.get_connection_status()

        data = status_obj.to_dict()
        data["connection"] = conn_obj.to_dict()
        data["alert_banner_active"] = conn_obj.alert_banner_active
        data["feed_source"] = conn_obj.feed_source
        data["state"] = status_obj.state.value
        return data

    @app.get("/api/events", tags=["SSE"])
    async def get_events(
        request: Request,
        service: DynamicStrategyService = Depends(get_strategy_service),
    ):
        """
        Server-Sent Events (SSE) streaming real-time heartbeat, price updates,
        and rebalance notifications to front-end clients.
        """
        async def event_generator():
            try:
                while True:
                    if await request.is_disconnected():
                        logger.debug("SSE client disconnected.")
                        break

                    # Collect status snapshot
                    status_obj = service.get_service_status()
                    conn_obj = service.feed_manager.get_connection_status()
                    symbols = service.service_config.symbols
                    prices = service.feed_manager.get_latest_prices(symbols)

                    payload = {
                        "total_nav": status_obj.total_nav,
                        "cash": status_obj.cash,
                        "equity": status_obj.equity,
                        "regime": status_obj.current_regime,
                        "is_paused": status_obj.is_paused,
                        "state": status_obj.state.value,
                        "alert_banner_active": conn_obj.alert_banner_active,
                        "feed_source": conn_obj.feed_source,
                        "status_message": conn_obj.status_message,
                        "prices": prices,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "service_name": status_obj.service_name,
                        "status": status_obj.to_dict(),
                        "portfolio": {
                            "total_nav": status_obj.total_nav,
                            "cash": status_obj.cash,
                            "equity": status_obj.equity,
                            "positions_count": status_obj.active_positions_count,
                        },
                    }

                    yield f"event: heartbeat\ndata: {json.dumps(payload)}\n\n"
                    interval = 0.1 if ("PYTEST_CURRENT_TEST" in os.environ) else 1.5
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logger.debug("SSE task cancelled on client disconnect.")
            except Exception as e:
                logger.error("SSE stream error: %s", e)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/operator/pause", response_model=OperatorActionResponse, tags=["Operator"])
    async def operator_pause(service: DynamicStrategyService = Depends(get_strategy_service)):
        """Pause automated rebalancing evaluations."""
        status_obj = await service.pause()
        return OperatorActionResponse(
            success=True,
            state=status_obj.state.value,
            is_paused=status_obj.is_paused,
            message="Strategy rebalancing paused by operator.",
            status=status_obj.to_dict(),
        )

    @app.post("/api/operator/resume", response_model=OperatorActionResponse, tags=["Operator"])
    async def operator_resume(service: DynamicStrategyService = Depends(get_strategy_service)):
        """Resume automated rebalancing evaluations."""
        status_obj = await service.resume()
        return OperatorActionResponse(
            success=True,
            state=status_obj.state.value,
            is_paused=status_obj.is_paused,
            message="Strategy rebalancing resumed by operator.",
            status=status_obj.to_dict(),
        )

    @app.post("/api/operator/rebalance", tags=["Operator"])
    async def operator_rebalance(
        payload: Optional[OperatorRebalanceRequest] = None,
        service: DynamicStrategyService = Depends(get_strategy_service),
    ):
        """Trigger immediate out-of-cadence strategy rebalance evaluation."""
        force = payload.force if payload else False
        result: ManualRebalanceResult = await service.manual_rebalance(force=force)
        return result.to_dict()

    @app.post("/api/operator/reset", tags=["Operator"])
    async def operator_reset(service: DynamicStrategyService = Depends(get_strategy_service)):
        """Pristine reset restoring $50,000.00 cash balance and purging test state."""
        summary = await asyncio.to_thread(service.reset_to_pristine)
        return {
            "success": True,
            "message": "Account reset to pristine $50,000.00 cash balance.",
            "portfolio": summary.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/trades", tags=["Activity"])
    async def get_trades(
        limit: int = 50,
        service: DynamicStrategyService = Depends(get_strategy_service),
    ):
        """Fetch chronologically ordered execution trade records."""
        trades = await asyncio.to_thread(service.paper_account.get_trade_history, limit=limit)
        return {"trades": [t.to_dict() for t in trades]}

    @app.get("/api/equity-history", tags=["Activity"])
    async def get_equity_history(
        limit: int = 100,
        service: DynamicStrategyService = Depends(get_strategy_service),
    ):
        """Fetch historical equity snapshots for performance charting."""
        history = await asyncio.to_thread(service.paper_account.get_equity_history, limit=limit)
        return {"history": history}

    return app


# Default singleton app instance for `uvicorn web.app:app`
app = create_app()
