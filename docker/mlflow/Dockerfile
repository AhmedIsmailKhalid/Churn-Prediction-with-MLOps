FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install MLflow directly (no Poetry complexity)
RUN pip install --no-cache-dir \
    mlflow==2.8.0 \
    psycopg2-binary==2.9.9 \
    boto3==1.28.85

# Create directory for artifacts
RUN mkdir -p /mlflow/artifacts

# Expose MLflow port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]