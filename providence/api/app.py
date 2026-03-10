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

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from providence.api.deps import AppState, get_state, set_state
from providence.api.routes import agents, config, health, pipeline, portfolio, shadow, stores
from providence.api.security import (
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    sanitize_error,
)

logger = structlog.get_logger()


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
        logger.info("Providence API starting", version=version)
        yield
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
    elif is_production:
        # In production, CORS must be explicitly configured
        origins = []
        logger.warning("CORS: no origins configured in production — all cross-origin requests blocked")
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
                "/api/v1/health/live",
                "/api/v1/health/ready",
                "/docs",
                "/openapi.json",
                "/redoc",
                "/",
                "/dashboard",
            }
            if path not in exempt:
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
    app.include_router(config.router, prefix=api_prefix)

    # ── Root endpoint ───────────────────────────────────────────────
    @app.get("/")
    async def root():
        return {
            "name": "Providence API",
            "version": version,
            "docs": "/docs",
            "health": f"{api_prefix}/health",
            "dashboard": "/dashboard",
        }

    # ── Dashboard (serves shadow_dashboard.html) ──────────────────
    _dashboard_path = Path(__file__).resolve().parent.parent.parent / "dashboard" / "shadow_dashboard.html"

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard():
        if _dashboard_path.exists():
            return HTMLResponse(content=_dashboard_path.read_text(encoding="utf-8"))
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>Place shadow_dashboard.html in the dashboard/ directory.</p>",
            status_code=404,
        )

    return app
