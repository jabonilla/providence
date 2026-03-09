"""
Security middleware and utilities for the Providence REST API.

Provides:
- API key authentication via X-API-Key header
- Rate limiting (sliding window, in-memory)
- Error sanitization (removes credentials from exception messages)
- Request size limiting
"""

import os
import re
import time
from collections import defaultdict
from typing import Callable, Optional

import structlog
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = structlog.get_logger(__name__)

# Configuration constants
DEFAULT_API_KEY = os.getenv("PROVIDENCE_API_KEY", "")
MAX_REQUEST_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
RATE_LIMIT_READ_REQ_MIN = 100  # requests per minute
RATE_LIMIT_WRITE_REQ_MIN = 5  # requests per minute

# Exempt paths from authentication and rate limiting
EXEMPT_PATHS = {"/health/live", "/health/ready", "/docs", "/openapi.json", "/redoc"}


# ============================================================================
# API Key Authentication
# ============================================================================


async def require_api_key(request: Request) -> str:
    """
    FastAPI dependency to validate X-API-Key header.

    Health probe endpoints (/health/live, /health/ready) are exempt from
    authentication to support Kubernetes liveness/readiness checks.

    Args:
        request: The incoming HTTP request.

    Returns:
        The API key if valid.

    Raises:
        HTTPException: 401 if key is missing, 403 if invalid.
    """
    # Exempt health probes from authentication
    if request.url.path in {"/health/live", "/health/ready"}:
        return "exempt"

    api_key = request.headers.get("X-API-Key")

    if not api_key:
        logger.warning("missing_api_key", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if api_key != DEFAULT_API_KEY:
        logger.warning("invalid_api_key", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key


# ============================================================================
# Rate Limiting (Sliding Window)
# ============================================================================


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    In-memory sliding window rate limiter for HTTP requests.

    Limits:
    - GET/HEAD/OPTIONS: 100 req/min per IP
    - POST/PUT/DELETE: 5 req/min per IP

    Health probes and exempt paths bypass rate limiting.
    Returns 429 Too Many Requests with Retry-After header on limit.
    """

    def __init__(self, app):
        super().__init__(app)
        # Track: ip -> list of (timestamp, method) tuples
        self._requests: dict[str, list[tuple[float, str]]] = defaultdict(list)
        self._lock_initialized = False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiting."""
        # Skip exempt paths
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        method = request.method

        # Determine rate limit based on method
        if method in {"GET", "HEAD", "OPTIONS"}:
            max_requests = RATE_LIMIT_READ_REQ_MIN
        else:
            max_requests = RATE_LIMIT_WRITE_REQ_MIN

        # Check rate limit
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Clean old requests outside the window
        if client_ip in self._requests:
            self._requests[client_ip] = [
                (ts, m) for ts, m in self._requests[client_ip] if ts > window_start
            ]

        # Count requests in current window
        request_count = len(self._requests[client_ip])

        if request_count >= max_requests:
            logger.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                method=method,
                limit=max_requests,
                window="1m",
            )
            return Response(
                content="Too Many Requests",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Record this request
        self._requests[client_ip].append((now, method))

        # Add rate limit headers to response
        response = await call_next(request)
        remaining = max_requests - request_count - 1
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(now + 60))

        return response


# ============================================================================
# Request Size Limiting
# ============================================================================


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce maximum request size limit.

    Rejects requests with Content-Length > 10 MB with 413 Payload Too Large.
    Exempt paths (health probes, docs) bypass this check.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with size validation."""
        # Skip exempt paths
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        content_length = request.headers.get("Content-Length")

        if content_length:
            try:
                size_bytes = int(content_length)
                if size_bytes > MAX_REQUEST_SIZE_BYTES:
                    logger.warning(
                        "request_too_large",
                        client_ip=request.client.host if request.client else "unknown",
                        content_length=size_bytes,
                        max_allowed=MAX_REQUEST_SIZE_BYTES,
                    )
                    return Response(
                        content="Request payload too large",
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        headers={
                            "X-Max-Content-Length": str(MAX_REQUEST_SIZE_BYTES),
                        },
                    )
            except ValueError:
                logger.warning("invalid_content_length", value=content_length)

        return await call_next(request)


# ============================================================================
# Error Sanitization
# ============================================================================


def sanitize_error(exc: Exception) -> str:
    """
    Sanitize an exception message by removing sensitive information.

    Removes:
    - API keys: sk-ant-*, sk-proj-*, Bearer tokens
    - Query parameters: api_key=*, token=*, password=*
    - URLs with credentials: http(s)://user:pass@...
    - Database connection strings
    - AWS credentials and region hints

    Args:
        exc: The exception to sanitize.

    Returns:
        A sanitized error message safe for external exposure.
    """
    error_msg = str(exc)

    # Patterns for sensitive data
    patterns = [
        # API keys and tokens
        (r"sk-ant-[a-zA-Z0-9/+=]{20,}", "<REDACTED_APIKEY>"),
        (r"sk-proj-[a-zA-Z0-9/+=]{20,}", "<REDACTED_APIKEY>"),
        (r"Bearer\s+[a-zA-Z0-9/+=.]+", "<REDACTED_TOKEN>"),
        (r"api[_-]?key\s*=\s*[^\s&]+", "api_key=<REDACTED>"),
        (r"token\s*=\s*[^\s&]+", "token=<REDACTED>"),
        (r"password\s*=\s*[^\s&]+", "password=<REDACTED>"),
        # URLs with credentials
        (r"https?://[^:]+:[^@]+@", "http(s)://<REDACTED>@"),
        # AWS credentials
        (r"AKIA[A-Z0-9]{16}", "<REDACTED_AWS_KEY>"),
        (r"aws_secret_access_key\s*=\s*[^\s]+", "aws_secret_access_key=<REDACTED>"),
        # Database connection strings
        (r"postgresql://[^@]+@", "postgresql://<REDACTED>@"),
        (r"mongodb://[^@]+@", "mongodb://<REDACTED>@"),
        # Generic credentials in query strings
        (r"[?&](auth|apikey|secret|credential)\s*=\s*[^\s&]+", r"?\1=<REDACTED>"),
    ]

    sanitized = error_msg
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    # If sanitization removed significant content or message is too technical,
    # return a generic message
    if len(sanitized) < 20 or "traceback" in error_msg.lower():
        return "An error occurred processing your request. Please contact support."

    return sanitized


# ============================================================================
# Utility Functions
# ============================================================================


def is_exempt_path(path: str) -> bool:
    """Check if a path is exempt from security checks."""
    return path in EXEMPT_PATHS or path.startswith("/api/v1/health")
