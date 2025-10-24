FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin prefect and griffe versions that are compatible
RUN pip install --no-cache-dir \
    "prefect==2.14.0" \
    "griffe==0.36.3" \
    asyncpg==0.29.0 \
    psycopg2-binary==2.9.9

EXPOSE 4200

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:4200/api/health || exit 1

CMD ["prefect", "server", "start", "--host", "0.0.0.0"]
