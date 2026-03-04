# Providence Deployment Guide

## Quick Start (Development)

```bash
# 1. Install dependencies
make install

# 2. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Start API server
make api

# 4. In another terminal, run a pipeline cycle
curl -X POST http://localhost:8000/api/v1/pipeline/run
```

## Docker Deployment

### Build images

```bash
make docker-build
```

This creates two images:
- `providence-api:latest` — REST API server (FastAPI + uvicorn)
- `providence-runner:latest` — Pipeline runner (CLI-based)

### Run locally with Docker Compose

```bash
# API only (for development)
make docker-api

# API + continuous runner
make docker-full

# Full production stack (API + runner + nginx with TLS)
make docker-prod
```

### Production Setup

1. **TLS certificates**: Place `fullchain.pem` and `privkey.pem` in `deploy/nginx/certs/`
2. **Environment**: Copy `.env.example` to `.env` and fill in production values
3. **Start**: `docker compose --profile production up -d`
4. **Monitor**: `python scripts/monitor.py --url https://your-domain --interval 60`

## CI/CD (GitHub Actions)

The pipeline (`.github/workflows/ci.yml`) runs on every push:

1. **Lint** — ruff format + check
2. **Unit tests** — pytest with coverage
3. **Integration tests** — full pipeline (skip perception)
4. **Docker build** — both targets, with API health verification
5. **Deploy** (main branch only) — push images, SSH deploy, health check

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `REGISTRY_URL` | Container registry URL |
| `REGISTRY_USERNAME` | Registry login |
| `REGISTRY_PASSWORD` | Registry password/token |
| `DEPLOY_HOST` | Production server hostname |
| `DEPLOY_USER` | SSH username |
| `DEPLOY_SSH_KEY` | SSH private key |

## Architecture

```
Internet → nginx (TLS, rate limiting) → API server (FastAPI)
                                             ↓
                                    Pipeline Runner (scheduled)
                                             ↓
                              Stores (JSONL on /app/data volume)
```

- **API server**: Stateless, reads from shared stores volume
- **Runner**: Writes pipeline results to shared stores volume
- **nginx**: TLS termination, rate limiting (30 req/s general, 2 req/min for pipeline triggers)
- **Data volume**: Persistent JSONL storage at `/app/data`

## Monitoring

```bash
# One-shot health check
python scripts/monitor.py --url http://localhost:8000 --verbose

# Continuous monitoring (every 60s)
python scripts/monitor.py --url http://localhost:8000 --interval 60 --verbose
```

Output is structured JSON lines, suitable for log aggregation systems.
