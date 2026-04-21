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
COPY scripts/ scripts/
COPY dashboard/ dashboard/

# Create non-root user
RUN groupadd -r providence && useradd -r -g providence providence \
    && mkdir -p /app/data && chown -R providence:providence /app

USER providence

EXPOSE 8000

ENTRYPOINT ["python", "-m", "providence.api.server"]
CMD ["--host", "0.0.0.0", "--data-dir", "/app/data"]
