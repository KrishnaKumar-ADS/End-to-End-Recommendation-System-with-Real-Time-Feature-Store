#!/usr/bin/env sh
set -eu

if [ -d "/app/project/frontend" ]; then
  FRONTEND_DIR="/app/project/frontend"
elif [ -d "/app/frontend" ]; then
  FRONTEND_DIR="/app/frontend"
else
  echo "Frontend directory not found under /app/project/frontend or /app/frontend" >&2
  exit 1
fi

export PORT="${PORT:-7860}"
export HOSTNAME="0.0.0.0"
export NODE_ENV="${NODE_ENV:-production}"
export NEXT_TELEMETRY_DISABLED="${NEXT_TELEMETRY_DISABLED:-1}"

echo "Starting frontend from ${FRONTEND_DIR} on ${HOSTNAME}:${PORT}"

if [ ! -d "${FRONTEND_DIR}/.next" ]; then
  echo "Missing build output at ${FRONTEND_DIR}/.next" >&2
  exit 1
fi

STANDALONE_DIR="${FRONTEND_DIR}/.next/standalone"
STATIC_SRC_DIR="${FRONTEND_DIR}/.next/static"
STATIC_DST_DIR="${STANDALONE_DIR}/.next/static"
PUBLIC_SRC_DIR="${FRONTEND_DIR}/public"
PUBLIC_DST_DIR="${STANDALONE_DIR}/public"

if [ ! -f "${STANDALONE_DIR}/server.js" ]; then
  echo "Missing standalone server at ${STANDALONE_DIR}/server.js" >&2
  exit 1
fi

mkdir -p "${STANDALONE_DIR}/.next"

if [ -d "${STATIC_SRC_DIR}" ]; then
  rm -rf "${STATIC_DST_DIR}"
  cp -R "${STATIC_SRC_DIR}" "${STATIC_DST_DIR}"
fi

if [ -d "${PUBLIC_SRC_DIR}" ]; then
  rm -rf "${PUBLIC_DST_DIR}"
  cp -R "${PUBLIC_SRC_DIR}" "${PUBLIC_DST_DIR}"
fi

exec node "${STANDALONE_DIR}/server.js"