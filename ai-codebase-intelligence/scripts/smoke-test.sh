#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${BASE_URL:-https://localhost}"
SKIP_TLS="${SKIP_TLS:-1}"
TIMEOUT="${TIMEOUT:-20}"
INGEST_GITHUB_URL="${INGEST_GITHUB_URL:-https://github.com/psf/requests}"
INGEST_BRANCH="${INGEST_BRANCH:-main}"
SEARCH_QUERY="${SEARCH_QUERY:-authentication}"
SMOKE_MAX_WAIT_SECONDS="${SMOKE_MAX_WAIT_SECONDS:-300}"
SMOKE_POLL_INTERVAL_SECONDS="${SMOKE_POLL_INTERVAL_SECONDS:-5}"

CURL_FLAGS=("--silent" "--show-error" "--max-time" "${TIMEOUT}")
if [[ "${SKIP_TLS}" == "1" ]]; then
    CURL_FLAGS+=("--insecure")
fi

PASS=0
FAIL=0
TOTAL=5

RESPONSE_STATUS=""
RESPONSE_BODY=""
TASK_ID=""
REPO_ID=""

extract_json_string() {
    local key="$1"
    local text="$2"
    printf '%s' "${text}" \
        | grep -Eo "\"${key}\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" \
        | head -n1 \
        | sed -E 's/.*:[[:space:]]*"([^"]*)".*/\1/' || true
}

extract_repo_context_for_url() {
    local repos_json="$1"
    local target_url="$2"

    local parser_script='\
import json
import sys

target_url = sys.argv[1]
raw = sys.stdin.read().strip()
if not raw:
    sys.exit(0)

try:
    repos = json.loads(raw)
except Exception:
    sys.exit(0)

if not isinstance(repos, list) or not repos:
    sys.exit(0)

matches = [r for r in repos if isinstance(r, dict) and r.get("github_url") == target_url]
candidates = matches if matches else [r for r in repos if isinstance(r, dict)]
if not candidates:
    sys.exit(0)

selected = None
for repo in candidates:
    if str(repo.get("status", "")).lower() == "completed":
        selected = repo
        break

if selected is None:
    selected = candidates[0]

repo_id = str(selected.get("id", ""))
task_id = str(selected.get("task_id", ""))
if repo_id or task_id:
    print(f"{repo_id}|{task_id}")
'

    if command -v python >/dev/null 2>&1; then
        printf '%s' "${repos_json}" | python -c "${parser_script}" "${target_url}" 2>/dev/null
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        printf '%s' "${repos_json}" | python3 -c "${parser_script}" "${target_url}" 2>/dev/null
        return
    fi

    if command -v py >/dev/null 2>&1; then
        printf '%s' "${repos_json}" | py -3 -c "${parser_script}" "${target_url}" 2>/dev/null
        return
    fi

    return 1
}

request() {
    local method="$1"
    local url="$2"
    local payload="${3:-}"

    local tmp_file
    tmp_file=$(mktemp)

    local status
    if [[ "${method}" == "POST" ]]; then
        status=$(curl "${CURL_FLAGS[@]}" -o "${tmp_file}" -w "%{http_code}" -X POST -H "Content-Type: application/json" -d "${payload}" "${url}" || echo "000")
    else
        status=$(curl "${CURL_FLAGS[@]}" -o "${tmp_file}" -w "%{http_code}" "${url}" || echo "000")
    fi

    RESPONSE_STATUS="${status}"
    RESPONSE_BODY=$(cat "${tmp_file}" 2>/dev/null || true)
    rm -f "${tmp_file}"
}

mark_pass() {
    echo " PASS"
    PASS=$((PASS + 1))
}

mark_fail() {
    local reason="$1"
    echo " FAIL (${reason})"
    FAIL=$((FAIL + 1))
}

lookup_existing_repo_context() {
    request "GET" "${BASE_URL}/api/v1/repos"
    if [[ "${RESPONSE_STATUS}" != "200" ]]; then
        return
    fi

    local resolved
    resolved=$(extract_repo_context_for_url "${RESPONSE_BODY}" "${INGEST_GITHUB_URL}" || true)
    if [[ -n "${resolved}" && "${resolved}" == *"|"* ]]; then
        local resolved_repo_id="${resolved%%|*}"
        local resolved_task_id="${resolved#*|}"
        if [[ -n "${resolved_repo_id}" ]]; then
            REPO_ID="${resolved_repo_id}"
        fi
        if [[ -n "${resolved_task_id}" ]]; then
            TASK_ID="${resolved_task_id}"
        fi
    fi

    if [[ -z "${REPO_ID}" ]]; then
        REPO_ID=$(extract_json_string "id" "${RESPONSE_BODY}")
    fi
    if [[ -z "${TASK_ID}" ]]; then
        TASK_ID=$(extract_json_string "task_id" "${RESPONSE_BODY}")
    fi
}

wait_for_ingestion_completion() {
    local waited=0

    if [[ -z "${TASK_ID}" ]]; then
        return 1
    fi

    while (( waited <= SMOKE_MAX_WAIT_SECONDS )); do
        request "GET" "${BASE_URL}/api/v1/status/${TASK_ID}"
        if [[ "${RESPONSE_STATUS}" == "200" ]]; then
            if grep -qi '"status"[[:space:]]*:[[:space:]]*"completed"' <<<"${RESPONSE_BODY}"; then
                return 0
            fi
            if grep -qi '"status"[[:space:]]*:[[:space:]]*"failed"' <<<"${RESPONSE_BODY}"; then
                return 1
            fi
        fi

        sleep "${SMOKE_POLL_INTERVAL_SECONDS}"
        waited=$((waited + SMOKE_POLL_INTERVAL_SECONDS))
    done

    return 1
}

printf "[1/%d] %-55s" "${TOTAL}" "GET /health returns 200"
request "GET" "${BASE_URL}/health"
if [[ "${RESPONSE_STATUS}" == "200" ]] && grep -qi "status" <<<"${RESPONSE_BODY}"; then
    mark_pass
else
    mark_fail "expected 200+status, got ${RESPONSE_STATUS}"
fi

printf "[2/%d] %-55s" "${TOTAL}" "POST /api/v1/ingest returns 202 or 409"
INGEST_PAYLOAD=$(printf '{"github_url":"%s","branch":"%s"}' "${INGEST_GITHUB_URL}" "${INGEST_BRANCH}")
request "POST" "${BASE_URL}/api/v1/ingest" "${INGEST_PAYLOAD}"

if [[ "${RESPONSE_STATUS}" == "202" || "${RESPONSE_STATUS}" == "409" ]]; then
    TASK_ID=$(extract_json_string "task_id" "${RESPONSE_BODY}")
    REPO_ID=$(extract_json_string "repo_id" "${RESPONSE_BODY}")
    if [[ -z "${TASK_ID}" || -z "${REPO_ID}" ]]; then
        lookup_existing_repo_context
    fi
    mark_pass
else
    mark_fail "expected 202/409, got ${RESPONSE_STATUS}"
fi

printf "[3/%d] %-55s" "${TOTAL}" "GET /api/v1/status/{task_id} returns 200"
if [[ -n "${TASK_ID}" ]]; then
    request "GET" "${BASE_URL}/api/v1/status/${TASK_ID}"
    if [[ "${RESPONSE_STATUS}" == "200" ]] && grep -qi "task_id" <<<"${RESPONSE_BODY}"; then
        mark_pass
    else
        mark_fail "expected 200+task_id, got ${RESPONSE_STATUS}"
    fi
else
    mark_fail "task_id unavailable"
fi

printf "[4/%d] %-55s" "${TOTAL}" "GET /api/v1/search returns 200"
if [[ -n "${REPO_ID}" ]]; then
    if wait_for_ingestion_completion; then
        request "GET" "${BASE_URL}/api/v1/search?q=${SEARCH_QUERY}&repo_id=${REPO_ID}&top_k=3"
        if [[ "${RESPONSE_STATUS}" == "200" ]] && grep -qi "results" <<<"${RESPONSE_BODY}"; then
            mark_pass
        else
            mark_fail "expected 200+results, got ${RESPONSE_STATUS}"
        fi
    else
        mark_fail "ingestion did not complete within ${SMOKE_MAX_WAIT_SECONDS}s"
    fi
else
    mark_fail "repo_id unavailable"
fi

printf "[5/%d] %-55s" "${TOTAL}" "GET / returns HTML"
request "GET" "${BASE_URL}/"
if [[ "${RESPONSE_STATUS}" == "200" ]] && grep -qi "html" <<<"${RESPONSE_BODY}"; then
    mark_pass
else
    mark_fail "expected 200+html, got ${RESPONSE_STATUS}"
fi

echo ""
if [[ ${FAIL} -eq 0 ]]; then
    echo "All ${PASS}/${TOTAL} smoke tests passed."
    exit 0
fi

echo "${FAIL}/${TOTAL} smoke tests failed."
exit 1
