#!/usr/bin/env bash

set -euo pipefail

CERTS_DIR="docker/nginx/certs"
CERT_FILE="${CERTS_DIR}/localhost.crt"
KEY_FILE="${CERTS_DIR}/localhost.key"
DAYS=825

echo "Creating certs directory at ${CERTS_DIR}..."
mkdir -p "${CERTS_DIR}"

echo "Generating self-signed certificate for localhost (${DAYS} days)..."

openssl req \
    -newkey rsa:2048 \
    -nodes \
    -keyout "${KEY_FILE}" \
    -x509 \
    -days "${DAYS}" \
    -out "${CERT_FILE}" \
    -subj "/C=IN/ST=AndhraPradesh/L=SriCity/O=IIITSriCity/OU=DataTeam/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
    2>/dev/null

echo "Certificates generated:"
echo "  ${CERT_FILE}"
echo "  ${KEY_FILE}"

openssl x509 -noout -subject -dates -in "${CERT_FILE}" 2>/dev/null || true
