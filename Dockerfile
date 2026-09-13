FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (curl for health checks, ca-certificates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy build manifest and project files
COPY pyproject.toml README.md ./
COPY bot/ ./bot/
COPY strategy_engine/ ./strategy_engine/
COPY web/ ./web/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY *.md ./

# Install project and production runtime dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir .

EXPOSE 8000

# Run Uvicorn server with dynamic PORT fallback
CMD ["sh", "-c", "python -m uvicorn web.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
