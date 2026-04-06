FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (cache layer)
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY providence/ providence/
COPY config/ config/
COPY scripts/ scripts/

# Create non-root user
RUN groupadd -r providence && useradd -r -g providence providence \
    && mkdir -p /app/data && chown -R providence:providence /app

USER providence

# ── API server target (default) ──────────────────────────────────
FROM base AS api
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health/live || exit 1

ENTRYPOINT ["python", "-m", "providence.api.server"]
CMD ["--host", "0.0.0.0", "--data-dir", "/app/data", "--skip-perception", "--skip-adaptive"]

# ── Pipeline runner target ───────────────────────────────────────
FROM base AS runner

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m providence --skip-perception --skip-adaptive --data-dir /app/data health || exit 1

ENTRYPOINT ["python", "-m", "providence"]
CMD ["--data-dir", "/app/data", "run-once"]
