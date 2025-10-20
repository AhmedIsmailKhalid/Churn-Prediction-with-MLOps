.PHONY: help setup install clean test lint format docker-build docker-up docker-down docker-logs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3.11
POETRY := poetry
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := mlops-churn-prediction

help: ## Show this help message
	@echo '$(PROJECT_NAME) - Makefile Commands'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Initial project setup
	@echo "Setting up $(PROJECT_NAME)..."
	@if [ ! -f .env ]; then cp .env.example .env; echo ".env file created"; fi
	@mkdir -p data/raw data/processed data/simulated logs
	@echo "Setup complete! Edit .env file with your configuration."

install: ## Install dependencies with Poetry
	@echo "Installing dependencies with Poetry..."
	$(POETRY) install

install-prod: ## Install production dependencies only
	$(POETRY) install --no-dev

update: ## Update dependencies
	$(POETRY) update

lock: ## Generate poetry.lock file
	$(POETRY) lock

export-requirements: ## Export requirements.txt from Poetry (for Docker)
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes

clean: ## Clean up generated files
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	rm -rf build dist htmlcov .coverage
	rm -rf .venv

data-download: ## Download Telco Churn dataset
	$(POETRY) run python scripts/download_data.py

data-validate: ## Validate raw data
	$(POETRY) run python scripts/validate_data.py

run-training: ## Run model training pipeline
	$(POETRY) run python -m src.models.train

test: ## Run tests
	$(POETRY) run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	$(POETRY) run pytest tests/unit/ -v

test-integration: ## Run integration tests only
	$(POETRY) run pytest tests/integration/ -v

test-watch: ## Run tests in watch mode
	$(POETRY) run ptw tests/ -- -v

lint: ## Run linters
	$(POETRY) run flake8 src/ tests/
	$(POETRY) run mypy src/
	$(POETRY) run bandit -r src/

format: ## Format code
	$(POETRY) run black src/ tests/
	$(POETRY) run isort src/ tests/

format-check: ## Check code formatting
	$(POETRY) run black --check src/ tests/
	$(POETRY) run isort --check src/ tests/

docker-build: ## Build Docker images
	$(DOCKER_COMPOSE) build

docker-up: ## Start all services
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop all services
	$(DOCKER_COMPOSE) down

docker-restart: ## Restart all services
	$(DOCKER_COMPOSE) restart

docker-logs: ## Show logs from all services
	$(DOCKER_COMPOSE) logs -f

docker-logs-api: ## Show API logs
	$(DOCKER_COMPOSE) logs -f api

docker-clean: docker-down ## Stop services and remove volumes
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

shell: ## Open Poetry shell
	$(POETRY) shell

run-api: ## Run API locally (without Docker)
	$(POETRY) run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

all: setup install docker-build docker-up ## Complete setup and start services
	@echo "All services started successfully!"
	@echo "MLflow: http://localhost:5000"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "API: http://localhost:8000/docs"
	@echo "Prometheus: http://localhost:9090"