#!/usr/bin/env bash

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info()  { echo -e "${GREEN}[entrypoint]${NC} $*"; }
echo_warn()  { echo -e "${YELLOW}[entrypoint]${NC} $*"; }
echo_error() { echo -e "${RED}[entrypoint] ERROR:${NC} $*" >&2; }

validate_production_env() {
    if [[ "${APP_ENV:-development}" != "production" ]]; then
        return
    fi

    echo_info "Validating required production environment variables..."

    declare -A REQUIRED_VARS=(
        ["GEMINI_API_KEY"]="Gemini text embedding provider"
        ["OPENROUTER_API_KEY"]="Qwen LLM provider via OpenRouter"
        ["POSTGRES_HOST"]="PostgreSQL connection"
        ["POSTGRES_DB"]="PostgreSQL database name"
        ["POSTGRES_USER"]="PostgreSQL authentication"
        ["POSTGRES_PASSWORD"]="PostgreSQL authentication"
        ["REDIS_URL"]="Celery broker and cache store"
        ["QDRANT_HOST"]="Qdrant vector database"
        ["NEO4J_URI"]="Neo4j graph database"
        ["NEO4J_USER"]="Neo4j authentication"
        ["NEO4J_PASSWORD"]="Neo4j authentication"
    )

    local validation_failed=0
    for var in "${!REQUIRED_VARS[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            echo_error "Required environment variable '${var}' is not set."
            echo_error "  This variable is required for: ${REQUIRED_VARS[$var]}"
            echo_error "  Set it in your .env.prod file and run: make prod-up"
            validation_failed=1
        fi
    done

    if [[ $validation_failed -eq 1 ]]; then
        exit 1
    fi

    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
        echo_warn "DEEPSEEK_API_KEY is not set. DeepSeek direct fallback is disabled."
    fi

    echo_info "Production environment validation passed."
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

validate_production_env
wait_for_postgres
wait_for_redis

# If a custom command is provided (for example in docker-compose.yml), run it.
if [[ "$#" -gt 0 ]]; then
    echo_info "Executing custom command: $*"
    exec "$@"
fi

echo_info "Running Alembic migrations..."
cd /app/backend

set +e
ALEMBIC_OUTPUT=$(alembic upgrade head 2>&1)
ALEMBIC_RC=$?
set -e

if [[ ${ALEMBIC_RC} -ne 0 ]]; then
    if grep -qi "Multiple head revisions are present" <<<"${ALEMBIC_OUTPUT}"; then
        echo_warn "Multiple Alembic heads detected; retrying with canonical production revision f163e94dc01f."
        alembic upgrade f163e94dc01f
    else
        echo_error "Alembic migration failed."
        echo "${ALEMBIC_OUTPUT}" >&2
        exit ${ALEMBIC_RC}
    fi
fi

echo_info "Starting Uvicorn with 4 workers on port 8000..."
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --no-access-log \
    --proxy-headers \
    --forwarded-allow-ips "*"
