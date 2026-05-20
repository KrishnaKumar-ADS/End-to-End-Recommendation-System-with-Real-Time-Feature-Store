#!/usr/bin/env bash

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info()  { echo -e "${GREEN}[worker-entrypoint]${NC} $*"; }
echo_warn()  { echo -e "${YELLOW}[worker-entrypoint]${NC} $*"; }
echo_error() { echo -e "${RED}[worker-entrypoint] ERROR:${NC} $*" >&2; }

validate_production_env() {
    if [[ "${APP_ENV:-development}" != "production" ]]; then
        return
    fi

    echo_info "Validating required production environment variables..."

    local required_vars=(
        "REDIS_URL"
        "POSTGRES_HOST"
        "POSTGRES_DB"
        "POSTGRES_USER"
        "POSTGRES_PASSWORD"
        "GEMINI_API_KEY"
        "OPENROUTER_API_KEY"
        "QDRANT_HOST"
        "NEO4J_URI"
        "NEO4J_USER"
        "NEO4J_PASSWORD"
    )

    local validation_failed=0
    local var
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            echo_error "Required environment variable '${var}' is not set."
            validation_failed=1
        fi
    done

    if [[ $validation_failed -eq 1 ]]; then
        echo_error "Missing required production environment variables. Worker cannot start."
        exit 1
    fi

    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
        echo_warn "DEEPSEEK_API_KEY is not set. DeepSeek direct fallback is disabled."
    fi

    echo_info "Production environment validation passed."
}

wait_for_redis() {
    local redis_host redis_port
    local max_retries=15
    local retry_count=0

    redis_host=$(echo "${REDIS_URL}" | sed -E 's#redis://([^@/]+@)?([^:/]+)(:[0-9]+)?(/.*)?#\2#')
    redis_port=$(echo "${REDIS_URL}" | sed -nE 's#redis://([^@/]+@)?[^:/]+:([0-9]+).*#\2#p')
    redis_port=${redis_port:-6379}

    echo_info "Waiting for Redis at ${redis_host}:${redis_port}..."

    until redis-cli -h "${redis_host}" -p "${redis_port}" ping 2>/dev/null | grep -q PONG; do
        retry_count=$((retry_count + 1))
        if [[ $retry_count -ge $max_retries ]]; then
            echo_error "Redis did not become available within $((max_retries * 2)) seconds."
            exit 1
        fi

        echo_warn "Redis not ready yet (attempt ${retry_count}/${max_retries}); retrying in 2s..."
        sleep 2
    done

    echo_info "Redis is ready."
}

wait_for_postgres() {
    local max_retries=30
    local retry_count=0

    echo_info "Waiting for PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT:-5432}..."

    until pg_isready -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -q; do
        retry_count=$((retry_count + 1))
        if [[ $retry_count -ge $max_retries ]]; then
            echo_error "PostgreSQL did not become available within $((max_retries * 2)) seconds."
            exit 1
        fi

        echo_warn "PostgreSQL not ready yet (attempt ${retry_count}/${max_retries}); retrying in 2s..."
        sleep 2
    done

    echo_info "PostgreSQL is ready."
}

validate_production_env
wait_for_redis
wait_for_postgres

if [[ "$#" -gt 0 ]]; then
    echo_info "Executing custom command: $*"
    exec "$@"
fi

echo_info "Starting Celery worker..."
exec celery -A tasks.celery_app worker \
    --loglevel=info \
    --concurrency=2 \
    --max-tasks-per-child=50 \
    --without-gossip \
    --without-mingle \
    --without-heartbeat
