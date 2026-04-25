"""FastAPI application factory.

Creates the Providence REST API with all routes, middleware,
exception handlers, and startup/shutdown lifecycle hooks.

Security features (Session 35):
- API key authentication via X-API-Key header (PROVIDENCE_API_KEY env var)
- In-memory sliding-window rate limiting (100 GET/min, 5 POST/min per IP)
- Request body size limiting (10 MB)
- Error message sanitization (no credential leakage)
- Restrictive CORS (explicit origins only in production)
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from providence.api.deps import AppState, get_state, set_state
from providence.api.routes import agents, chat, config, health, keys, perception, pipeline, portfolio, regime, seed, shadow, stores
from providence.api.security import (
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    sanitize_error,
)

logger = structlog.get_logger()


# ── Pipeline Scheduler ──────────────────────────────────────────────
# Configurable via environment variables:
#   SCHEDULER_ENABLED=true (default: true)
#   SCHEDULER_HOURS=10,15 (ET hours to run, default: 10,15)
#   SCHEDULER_DAYS=mon,tue,wed,thu,fri (default: weekdays)

_SCHEDULER_TASK: asyncio.Task | None = None

_DAY_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


async def _pipeline_scheduler() -> None:
    """Background task that triggers pipeline runs on a schedule.

    Runs at configured hours (US/Eastern) on configured days.
    Checks every 60 seconds if it's time to run.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")

    # Parse config from env
    hours_str = os.getenv("SCHEDULER_HOURS", "10,15")
    run_hours = {int(h.strip()) for h in hours_str.split(",") if h.strip()}

    days_str = os.getenv("SCHEDULER_DAYS", "mon,tue,wed,thu,fri")
    run_days = {_DAY_MAP[d.strip().lower()] for d in days_str.split(",") if d.strip().lower() in _DAY_MAP}

    logger.info(
        "Pipeline scheduler started",
        run_hours=sorted(run_hours),
        run_days=sorted(run_days),
    )

    triggered_today: set[int] = set()  # hours already triggered today
    last_date = None

    while True:
        try:
            await asyncio.sleep(60)  # check every minute

            now_et = datetime.now(et)

            # Reset triggered set on new day
            if last_date != now_et.date():
                triggered_today = set()
                last_date = now_et.date()

            # Check if we should run
            if (
                now_et.weekday() in run_days
                and now_et.hour in run_hours
                and now_et.minute < 5  # within first 5 minutes of the hour
                and now_et.hour not in triggered_today
            ):
                triggered_today.add(now_et.hour)
                logger.info(
                    "Scheduler triggering pipeline run",
                    time_et=now_et.strftime("%Y-%m-%d %H:%M ET"),
                )

                try:
                    state = get_state()
                    if state.runner:
                        result = await state.runner.run_once()
                        logger.info(
                            "Scheduled pipeline run completed",
                            status=result.status.value if result else "unknown",
                        )
                    else:
                        logger.warning("Scheduler: no runner available")
                except Exception as exc:
                    logger.error("Scheduled pipeline run failed", error=str(exc))

        except asyncio.CancelledError:
            logger.info("Pipeline scheduler stopped")
            break
        except Exception as exc:
            logger.error("Scheduler error (will retry)", error=str(exc))
            await asyncio.sleep(60)


def create_app(
    state: AppState | None = None,
    *,
    title: str = "Providence API",
    version: str = "0.1.0",
    cors_origins: list[str] | None = None,
    enable_auth: bool | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    state:
        Pre-built AppState. If None, a minimal default is created
        (useful for testing).
    title:
        OpenAPI title.
    version:
        API version string.
    cors_origins:
        Allowed CORS origins. In production, this MUST be set explicitly.
        Defaults to ["*"] only when PROVIDENCE_ENV != "production".
    enable_auth:
        Whether to require X-API-Key header. Defaults to True if
        PROVIDENCE_API_KEY env var is set, False otherwise.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        global _SCHEDULER_TASK
        logger.info("Providence API starting", version=version)

        # Start pipeline scheduler if enabled
        scheduler_enabled = os.getenv("SCHEDULER_ENABLED", "true").lower() in ("true", "1", "yes")
        if scheduler_enabled:
            _SCHEDULER_TASK = asyncio.create_task(_pipeline_scheduler())
            logger.info("Pipeline scheduler enabled")
        else:
            logger.info("Pipeline scheduler disabled (SCHEDULER_ENABLED=false)")

        yield

        # Stop scheduler on shutdown
        if _SCHEDULER_TASK and not _SCHEDULER_TASK.done():
            _SCHEDULER_TASK.cancel()
            try:
                await _SCHEDULER_TASK
            except asyncio.CancelledError:
                pass
        logger.info("Providence API shutting down")

    app = FastAPI(
        title=title,
        version=version,
        description="Providence AI-Native Hedge Fund — REST API",
        lifespan=lifespan,
    )

    # ── State injection ─────────────────────────────────────────────
    if state is not None:
        set_state(state)

    # ── Security middleware (outermost first) ────────────────────────
    # Request size limiting — reject oversized payloads early
    app.add_middleware(RequestSizeLimitMiddleware)

    # Rate limiting — sliding window per IP
    app.add_middleware(RateLimitMiddleware)

    # ── CORS (restricted in production) ──────────────────────────────
    is_production = os.getenv("PROVIDENCE_ENV", "").lower() == "production"
    if cors_origins:
        origins = cors_origins
    elif os.getenv("CORS_ORIGINS"):
        # Allow comma-separated origins from env var
        origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
        logger.info("CORS: origins from env", origins=origins)
    elif is_production:
        # In production, allow Railway portal origin by default
        origins = [
            "https://providence-portal-production.up.railway.app",
            "https://providence-portal.up.railway.app",
        ]
        logger.info("CORS: using default production origins", origins=origins)
    else:
        origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=bool(origins and origins != ["*"]),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )

    # ── API key authentication middleware ─────────────────────────────
    # Resolved: enable auth if PROVIDENCE_API_KEY is set, unless
    # explicitly overridden by the enable_auth parameter.
    api_key = os.getenv("PROVIDENCE_API_KEY", "")
    if enable_auth is None:
        _auth_enabled = bool(api_key)
    else:
        _auth_enabled = enable_auth

    if _auth_enabled:
        from providence.api.security import require_api_key

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Exempt health probes, docs, and root
            path = request.url.path
            exempt = {
                "/api/v1/health",
                "/api/v1/health/live",
                "/api/v1/health/ready",
                "/api/v1/config/keys",
                "/docs",
                "/openapi.json",
                "/redoc",
                "/",
                "/dashboard",
            }
            # Dashboard-served pages can access shadow/health data
            exempt_prefixes = (
                "/api/v1/shadow/",
            )
            # Skip auth for CORS preflight and exempt paths
            if request.method == "OPTIONS" or path in exempt or path.startswith(exempt_prefixes):
                return await call_next(request)
            await require_api_key(request)
            return await call_next(request)

        logger.info("API key authentication enabled")
    else:
        logger.warning("API key authentication DISABLED — set PROVIDENCE_API_KEY to enable")

    # ── Request logging middleware ───────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "HTTP request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response

    # ── Exception handlers (sanitized — no credential leakage) ──────
    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        safe_msg = sanitize_error(exc)
        logger.error("Runtime error", error=safe_msg, path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        safe_msg = sanitize_error(exc)
        logger.error("Unhandled exception", error=safe_msg, path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    # ── Routes ──────────────────────────────────────────────────────
    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(pipeline.router, prefix=api_prefix)
    app.include_router(agents.router, prefix=api_prefix)
    app.include_router(stores.router, prefix=api_prefix)
    app.include_router(shadow.router, prefix=api_prefix)
    app.include_router(portfolio.router, prefix=api_prefix)
    app.include_router(regime.router, prefix=api_prefix)
    app.include_router(config.router, prefix=api_prefix)
    app.include_router(keys.router, prefix=api_prefix)
    app.include_router(perception.router, prefix=api_prefix)
    app.include_router(seed.router, prefix=api_prefix)
    app.include_router(chat.router, prefix=api_prefix)

    # ── Marketing site (public root) ────────────────────────────────
    def _read_ui_kit(relative_path: str) -> str | None:
        base = Path(__file__).parent.parent.parent
        candidates = [
            base / relative_path,
            Path("/app") / relative_path,
        ]
        for p in candidates:
            if p.exists():
                return p.read_text()
        return None

    @app.get("/", include_in_schema=False)
    async def marketing_site():
        """Serve the public-facing marketing site."""
        content = _read_ui_kit("ui_kits/marketing/index.html")
        if content:
            return HTMLResponse(content=content)
        # Fallback JSON for API clients
        return JSONResponse({"name": "Providence API", "version": version, "docs": "/docs"})

    @app.get("/marketing", include_in_schema=False)
    async def marketing_alias():
        return RedirectResponse(url="/", status_code=301)

    # ── Shadow Dashboard (self-contained HTML) ─────────────────────
    @app.get("/dashboard", include_in_schema=False)
    async def dashboard():
        """Serve the shadow mode monitoring dashboard."""
        content = _read_ui_kit("dashboard/shadow_dashboard.html")
        if content:
            return HTMLResponse(content=content)
        return JSONResponse(
            status_code=404,
            content={"error": "Dashboard not found"},
        )

    # ── Static assets for UI kits ───────────────────────────────────
    _assets_base = Path(__file__).parent.parent.parent / "ui_kits"
    if not _assets_base.exists():
        _assets_base = Path("/app/ui_kits")
    if _assets_base.exists():
        app.mount("/static", StaticFiles(directory=str(_assets_base)), name="static")

    return app
