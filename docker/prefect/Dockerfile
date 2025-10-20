FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.7.1
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Prefect and related dependencies
RUN poetry install --no-dev --no-root --only main && \
    poetry run pip install prefect==2.14.0 asyncpg==0.29.0 psycopg2-binary==2.9.9

# Expose Prefect server port
EXPOSE 4200

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:4200/api/health || exit 1