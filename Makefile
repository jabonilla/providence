.PHONY: help install test lint run run-once run-learning health agents docker docker-run clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ──────────────────────────────────────────────────

install:  ## Install dependencies
	pip install -e ".[dev]"

test:  ## Run all tests
	python -m pytest tests/ -v --tb=short

test-unit:  ## Run unit tests only
	python -m pytest tests/unit/ -v --tb=short

test-integration:  ## Run integration tests only
	python -m pytest tests/integration/ -v --tb=short

coverage:  ## Run tests with coverage report
	python -m pytest tests/ --cov=providence --cov-report=term-missing --cov-report=html

lint:  ## Run linters (ruff)
	ruff check providence/ tests/
	ruff format --check providence/ tests/

format:  ## Auto-format code
	ruff format providence/ tests/

# ── Local execution ──────────────────────────────────────────────

run:  ## Run continuous mode (default)
	python -m providence run-continuous --log-level INFO

run-once:  ## Run a single pipeline cycle
	python -m providence run-once --with-exit --with-governance --log-level INFO

run-learning:  ## Run offline learning batch
	python -m providence run-learning --log-level INFO

health:  ## Check system health
	python -m providence --skip-perception --skip-adaptive health

agents:  ## List all agents
	python -m providence list-agents

frozen:  ## Run with frozen agents only (no API keys needed)
	python -m providence --skip-perception --skip-adaptive --log-level DEBUG run-once

# ── API server ───────────────────────────────────────────────────

api:  ## Start the REST API server locally
	python -m providence.api.server --port 8000 --data-dir data/ --log-level info

api-dev:  ## Start API with auto-reload (dev mode)
	uvicorn providence.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

monitor:  ## Run health monitor against local API
	python scripts/monitor.py --url http://localhost:8000 --verbose

monitor-loop:  ## Run health monitor in continuous loop (60s interval)
	python scripts/monitor.py --url http://localhost:8000 --interval 60 --verbose

# ── Docker ───────────────────────────────────────────────────────

docker-build:  ## Build all Docker images
	docker build --target api -t providence-api:latest .
	docker build --target runner -t providence-runner:latest .

docker-api:  ## Start API server with Docker Compose
	docker compose up -d api

docker-full:  ## Start API + runner with Docker Compose
	docker compose up -d api runner

docker-once:  ## Run single cycle with Docker Compose
	docker compose run --rm run-once

docker-learning:  ## Run learning batch with Docker Compose
	docker compose run --rm learning

docker-prod:  ## Start full production stack (API + runner + nginx)
	docker compose --profile production up -d

docker-down:  ## Stop all services
	docker compose --profile production down

docker-logs:  ## Tail all service logs
	docker compose logs -f

# ── Cleanup ──────────────────────────────────────────────────────

clean:  ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
