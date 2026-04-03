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

# -- STAGE 2: Frontend Builder ----------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# -- STAGE 3: Runtime -------------------------------------------------
FROM python:3.10-slim AS runtime

LABEL maintainer="DS19 Project"
LABEL description="DS19 Recommender System - FastAPI + MLflow + Feast"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # MLflow tracking
    MLFLOW_TRACKING_URI="sqlite:////app/mlops/experiments/mlflow.db" \
    # Redis defaults to disabled in single-container Spaces runtime
    REDIS_HOST="localhost" \
    REDIS_PORT="6379" \
    REDIS_ENABLED="false" \
    # FastAPI
    HOST="0.0.0.0" \
    PORT="7860" \
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
COPY data/processed/item2idx.json    ./data/processed/item2idx.json
COPY data/processed/idx2item.json    ./data/processed/idx2item.json
COPY data/processed/dataset_meta.json ./data/processed/dataset_meta.json
COPY data/processed/user2idx.json    ./data/processed/user2idx.json
COPY data/features/user_features.parquet ./data/features/user_features.parquet
COPY data/features/item_features.parquet ./data/features/item_features.parquet
COPY data/sequences/sequences.pkl    ./data/sequences/sequences.pkl
COPY data/raw/movies.csv     ./data/raw/movies.csv
COPY models/__init__.py      ./models/__init__.py
COPY models/two_tower/       ./models/two_tower/
COPY models/saved/two_tower_best.pt ./models/saved/two_tower_best.pt
COPY models/saved/lgbm_ranker.pkl   ./models/saved/lgbm_ranker.pkl
COPY retrieval/faiss_item.index ./retrieval/faiss_item.index
COPY retrieval/item_idx_map.npy ./retrieval/item_idx_map.npy
COPY mlops/__init__.py       ./mlops/__init__.py
COPY mlops/ab_testing/       ./mlops/ab_testing/
COPY mlops/bandits/          ./mlops/bandits/
COPY mlops/mlflow_setup/     ./mlops/mlflow_setup/
COPY backend/                ./backend/
COPY --from=frontend-builder /frontend/dist ./frontend/dist

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
EXPOSE 7860

# Health check - Docker will mark container as unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD sh -c "curl -f http://localhost:${PORT}/health || exit 1"

# Default command: run FastAPI with uvicorn
# WORKERS=1 because multiple workers would each load the GPU model
CMD ["sh", "-c", \
    "uvicorn backend.app.main:app \
    --host ${HOST} \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level ${LOG_LEVEL} \
    --access-log"]
