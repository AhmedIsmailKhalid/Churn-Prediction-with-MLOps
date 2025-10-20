# MLOps Churn Prediction Pipeline

Production-grade MLOps pipeline for customer churn prediction demonstrating senior-level MLOps engineering practices.

## Project Overview

This project demonstrates end-to-end MLOps best practices including:
- Feature Store (Feast)
- Experiment Tracking (MLflow)
- Workflow Orchestration (Prefect)
- Model Serving (FastAPI)
- Monitoring (Prometheus + Grafana)
- CI/CD (GitHub Actions)

## Quick Start (Windows OS)

### Prerequisites

#### Required Software:
1. **Python 3.11+**
   - Download: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

2. **Poetry 1.7+**
   ```powershell
   # Install using PowerShell (run as Administrator)
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```
   - After installation, add Poetry to PATH:
     - Search "Environment Variables" in Windows
     - Add `C:\Users\<YourUsername>\AppData\Roaming\Python\Scripts` to PATH

3. **Docker Desktop**
   - Download: https://www.docker.com/products/docker-desktop/
   - Ensure WSL 2 is enabled (Docker Desktop will guide you)
   - Start Docker Desktop before running containers

4. **Git**
   - Download: https://git-scm.com/download/win
   - Or use GitHub Desktop: https://desktop.github.com/

5. **Make (Optional, but recommended)**
   ```powershell
   # Install using Chocolatey
   choco install make
   
   # Or using Scoop
   scoop install make
   ```
   - If you don't want to install Make, see "Without Make" section below

---

### Installation

#### Step 1: Clone the Repository
```powershell
git clone <your-repo-url>
cd churn-prediction-mlops
```

#### Step 2: Verify Prerequisites
```powershell
# Check Python version (should be 3.11+)
python --version

# Check Poetry installation
poetry --version

# Check Docker is running
docker --version
docker-compose --version
```

#### Step 3: Setup Project

**With Make:**
```powershell
make setup
make install
```

**Without Make:**
```powershell
# Create .env file
copy .env.example .env

# Create directories
mkdir data\raw, data\processed, data\simulated, logs

# Install dependencies
poetry install
```

#### Step 4: Run Tests
```powershell
# With Make
make test

# Without Make
poetry run pytest tests\ -v
```

#### Step 5: Start Docker Services
```powershell
# With Make
make docker-up

# Without Make
docker-compose up -d
```

#### Step 6: Verify Services Are Running
```powershell
# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

---

## 🖥️ Accessing Services

Once services are running, access them via:

| Service | URL | Credentials |
|---------|-----|-------------|
| **API Documentation** | http://localhost:8000/docs | - |
| **MLflow** | http://localhost:5000 | - |
| **Grafana** | http://localhost:3000 | admin/admin |
| **Prometheus** | http://localhost:9090 | - |
| **Prefect** | http://localhost:4200 | - |

---

## 📦 Development Setup (Windows)

### Using Poetry

```powershell
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run tests
poetry run pytest

# Run specific test file
poetry run pytest tests\unit\test_api.py -v

# Format code
poetry run black src\ tests\
poetry run isort src\ tests\

# Lint code
poetry run flake8 src\ tests\
poetry run mypy src\
```

### Common Make Commands

```powershell
make help          # Show all available commands
make install       # Install dependencies
make test          # Run all tests
make test-unit     # Run unit tests only
make lint          # Run linters
make format        # Format code
make docker-up     # Start all services
make docker-down   # Stop all services
make docker-logs   # View service logs
make clean         # Clean generated files
```

### Without Make (Windows PowerShell)

```powershell
# Setup
copy .env.example .env
mkdir data\raw, data\processed, data\simulated, logs

# Install dependencies
poetry install

# Run tests
poetry run pytest tests\ -v --cov=src --cov-report=html --cov-report=term

# Format code
poetry run black src\ tests\
poetry run isort src\ tests\

# Lint
poetry run flake8 src\ tests\
poetry run mypy src\

# Docker operations
docker-compose build
docker-compose up -d
docker-compose down
docker-compose logs -f

# Run API locally (without Docker)
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🧪 Testing

### Run All Tests
```powershell
# With Make
make test

# Without Make
poetry run pytest tests\ -v --cov=src
```

### Run Specific Test Categories
```powershell
# Unit tests only
poetry run pytest tests\unit\ -v

# Integration tests only
poetry run pytest tests\integration\ -v

# Specific test file
poetry run pytest tests\unit\test_api.py -v

# Run tests matching a pattern
poetry run pytest tests\ -k "test_predict" -v
```

### View Coverage Report
```powershell
# Generate HTML coverage report
poetry run pytest tests\ --cov=src --cov-report=html

# Open coverage report in browser
start htmlcov\index.html
```

---

## 🐳 Docker Commands (Windows)

### Managing Services

```powershell
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d api

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f api

# Rebuild services
docker-compose build

# Rebuild specific service
docker-compose build api

# Restart services
docker-compose restart

# Check service status
docker-compose ps
```

---

## 📁 Project Structure

```
churn-prediction-mlops/
├── src/
│   ├── api/              # FastAPI application
│   ├── models/           # Model training code
│   ├── features/         # Feast feature store
│   ├── pipelines/        # Prefect workflows
│   └── monitoring/       # Monitoring utilities
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── data/
│   ├── raw/             # Raw datasets
│   ├── processed/       # Processed features
│   └── simulated/       # Simulated data
├── docker/              # Dockerfiles
├── monitoring/          # Prometheus & Grafana configs
├── scripts/             # Utility scripts
├── notebooks/           # Jupyter notebooks
├── docs/               # Documentation
├── .github/            # GitHub Actions workflows
├── docker-compose.yml  # Docker orchestration
├── pyproject.toml      # Poetry dependencies
└── Makefile           # Common commands
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📝 License

MIT License