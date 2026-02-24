.PHONY: help install test lint run run-once run-learning health agents docker docker-run clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

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

run:  ## Run continuous mode (default)
	python -m providence run-continuous --log-level INFO

run-once:  ## Run a single pipeline cycle
	python -m providence run-once --with-exit --with-governance --log-level INFO

run-learning:  ## Run offline learning batch
	python -m providence run-learning --log-level INFO

health:  ## Check system health
	python -m providence health --skip-perception --skip-adaptive

agents:  ## List all agents
	python -m providence list-agents

frozen:  ## Run with frozen agents only (no API keys needed)
	python -m providence run-once --skip-perception --skip-adaptive --log-level DEBUG

docker:  ## Build Docker image
	docker build -t providence:latest .

docker-run:  ## Run with Docker Compose (continuous mode)
	docker compose up -d providence

docker-once:  ## Run single cycle with Docker Compose
	docker compose run --rm providence-once

docker-learning:  ## Run learning batch with Docker Compose
	docker compose run --rm providence-learning

clean:  ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
