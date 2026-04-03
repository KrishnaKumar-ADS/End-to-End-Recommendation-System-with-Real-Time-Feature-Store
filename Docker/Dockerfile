FROM python:3.10-slim AS builder

LABEL maintainer="DS19 Project"
LABEL description="Multi-stage builder for DS19 recommender system"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
# Copy requirements first for better Docker layer caching
# (if requirements.txt doesn't change, this layer is cached)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── STAGE 2: Runtime ──────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="DS19 Project"
LABEL description="DS19 Recommender System — FastAPI + MLflow + Feast"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # MLflow tracking
    MLFLOW_TRACKING_URI="sqlite:////app/mlops/experiments/mlflow.db" \
    # Redis (will be overridden by docker-compose env)
    REDIS_HOST="redis" \
    REDIS_PORT="6379" \
    # FastAPI
    HOST="0.0.0.0" \
    PORT="8000" \
    WORKERS="1" \
    LOG_LEVEL="info"

# Install only runtime system dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
# Copy in dependency order (least-changing first for cache efficiency)
COPY data/processed/         ./data/processed/
COPY data/raw/movies.csv     ./data/raw/movies.csv
COPY models/saved/           ./models/saved/
COPY retrieval/              ./retrieval/
COPY feature_store/          ./feature_store/
COPY mlops/                  ./mlops/
COPY backend/                ./backend/

# Create necessary runtime directories
RUN mkdir -p \
    /app/mlops/experiments/artifacts \
    /app/mlops/ab_testing \
    /app/mlops/reports \
    /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 ds19user && \
    chown -R ds19user:ds19user /app
USER ds19user

# Expose FastAPI port
EXPOSE 8000

# Health check — Docker will mark container as unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run FastAPI with uvicorn
# WORKERS=1 because multiple workers would each load the GPU model
CMD ["sh", "-c", \
     "uvicorn backend.app.main:app \
      --host ${HOST} \
      --port ${PORT} \
      --workers ${WORKERS} \
      --log-level ${LOG_LEVEL} \
      --access-log"]