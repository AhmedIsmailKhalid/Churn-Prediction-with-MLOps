FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.7.1
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# Copy only MLflow-related dependencies
COPY pyproject.toml poetry.lock* ./

# Install only necessary dependencies for MLflow
RUN poetry install --no-dev --no-root --only main && \
    poetry run pip install mlflow==2.8.0 psycopg2-binary==2.9.9 boto3==1.28.85

# Create directory for artifacts
RUN mkdir -p /mlflow/artifacts

# Expose MLflow port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1