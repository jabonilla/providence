FROM python:3.12-slim

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

# Create data directory for persistent storage
RUN mkdir -p /app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m providence --skip-perception --skip-adaptive --data-dir /app/data health || exit 1

# Default entry point
ENTRYPOINT ["python", "-m", "providence"]
CMD ["--skip-perception", "--data-dir", "/app/data", "run-once"]
