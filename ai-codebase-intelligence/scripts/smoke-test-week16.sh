#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PASS=0
FAIL=0

pass() {
  echo " PASS"
  PASS=$((PASS + 1))
}

fail() {
  local reason="$1"
  echo " FAIL (${reason})"
  FAIL=$((FAIL + 1))
}

check_file_exists() {
  local label="$1"
  local path="$2"
  printf "%-70s" "${label}"
  if [[ -f "${path}" ]]; then
    pass
  else
    fail "missing ${path}"
  fi
}

check_contains() {
  local label="$1"
  local path="$2"
  local pattern="$3"
  printf "%-70s" "${label}"
  if grep -qE "${pattern}" "${path}"; then
    pass
  else
    fail "pattern not found in ${path}"
  fi
}

check_command() {
  local label="$1"
  shift
  printf "%-70s" "${label}"
  if "$@" >/dev/null 2>&1; then
    pass
  else
    fail "command failed"
  fi
}

run_provider_comparison() {
  if command -v python >/dev/null 2>&1; then
    python backend/evaluation/provider_comparison.py --no-html --output-dir data/benchmarks
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 backend/evaluation/provider_comparison.py --no-html --output-dir data/benchmarks
    return
  fi

  if command -v py >/dev/null 2>&1; then
    py -3 backend/evaluation/provider_comparison.py --no-html --output-dir data/benchmarks
    return
  fi

  return 1
}

check_file_exists "README present" "README.md"
check_file_exists "Week 16 deployment guide present" "docs/DEPLOYMENT.md"
check_file_exists "Week 16 architecture doc present" "docs/ARCHITECTURE.md"
check_file_exists "Week 16 lessons doc present" "docs/WHAT_I_LEARNED.md"
check_file_exists "Branch protection doc present" "docs/BRANCH_PROTECTION.md"
check_file_exists "Changelog present" "CHANGELOG.md"
check_file_exists "Demo script present" "scripts/demo-script.md"
check_file_exists "Recording helper present" "scripts/record-demo.sh"
check_file_exists "Provider comparison script present" "backend/evaluation/provider_comparison.py"
check_file_exists "Bug template present" ".github/ISSUE_TEMPLATE/bug_report.md"
check_file_exists "Feature template present" ".github/ISSUE_TEMPLATE/feature_request.md"
check_file_exists "PR template present" ".github/PULL_REQUEST_TEMPLATE.md"
check_file_exists "Dependabot config present" ".github/dependabot.yml"

check_contains "README has quick start section" "README.md" "## Quick start"
check_contains "README links deployment docs" "README.md" "docs/DEPLOYMENT.md"
check_contains "Main has custom_openapi" "backend/main.py" "def custom_openapi"
check_contains "Main has OpenAPI contact metadata" "backend/main.py" "contact"
check_contains "Ingest route has summary metadata" "backend/api/routes/ingest.py" "summary=\"Ingest a GitHub repository\""
check_contains "Search route has summary metadata" "backend/api/routes/search.py" "summary=\"Semantic and hybrid search"
check_contains "Ask route has summary metadata" "backend/api/routes/ask.py" "summary=\"Ask a natural language question"
check_contains "Graph route has summary metadata" "backend/api/routes/graph.py" "summary=\"Get repository graph data\""

check_command "Provider comparison script executes" run_provider_comparison
check_file_exists "Provider comparison markdown generated" "data/benchmarks/PROVIDER_COMPARISON.md"


echo ""
if [[ ${FAIL} -eq 0 ]]; then
  echo "Week 16 smoke tests passed: ${PASS} checks succeeded."
  exit 0
fi

echo "Week 16 smoke tests failed: ${FAIL} checks failed, ${PASS} checks passed."
exit 1
