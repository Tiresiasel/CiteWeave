#!/bin/bash
# ============================================================
# scripts/deployment_check.sh
# CiteWeave deployment smoke-test.
# Run this after `docker-compose up -d` to verify everything
# is healthy before uploading papers or running queries.
#
# Exit codes:
#   0  — all checks passed
#   1  — one or more checks failed (details printed below)
# ============================================================

# Exit on error is intentionally NOT set globally — we want to run all checks
# and report all failures. Explicitly exit 1 at the end if ERRORS > 0.
set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

# Load env vars (skip if .env is not present)
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

NEO4J_PASSWORD="${CITEWEAVE_NEO4J_PASSWORD:-0xC1735}"
NEO4J_HOST="${NEO4J_HOST:-localhost}"
QDRANT_HOST="${QDRANT_HOST:-localhost}"
GROBID_HOST="${GROBID_HOST:-localhost}"

# Detect docker access
DOCKER_AVAILABLE=0
if docker info &>/dev/null; then
  DOCKER_AVAILABLE=1
fi

# ── Helpers ────────────────────────────────────────────────

pass()  { echo -e "  ${GREEN}✓${RESET}  $1"; }
warn()  { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail()  { echo -e "  ${RED}✗${RESET}  $1"; }
info()  { echo -e "  ${BOLD}→${RESET}  $1"; }

header() {
  echo ""
  echo -e "${BOLD}[ CiteWeave Deployment Check ]${RESET}"
  echo "-----------------------------------"
}

section() {
  echo ""
  echo -e "${BOLD}▸ $1${RESET}"
}

# ── 1. Docker Compose services ─────────────────────────────

section "Docker Compose services"

SERVICES_OK=0

if [[ ${DOCKER_AVAILABLE} -eq 0 ]]; then
  warn "Docker daemon not accessible in this environment — skipping container checks"
  warn "Run this script on the host where Docker is available."
else
  check_docker_service() {
    local name=$1
    local port=$2
    local url=$3
    local expected_code=${4:-200}

    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" \
      --max-time 10 \
      "${url}" 2>/dev/null || echo "000")

    if [[ "${code}" == "${expected_code}" ]]; then
      pass "${name} is up (HTTP ${code})"
      return 0
    else
      fail "${name} is NOT responding (got HTTP ${code}, expected ${expected_code})"
      return 1
    fi
  }

  check_docker_container() {
    local container=$1
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
      pass "Container '${container}' is running"
      return 0
    else
      fail "Container '${container}' is NOT running"
      return 1
    fi
  }

  CONTAINER_CHECKS=0
  for c in citeweave-qdrant citeweave-grobid citeweave-neo4j; do
    check_docker_container "${c}" && ((CONTAINER_CHECKS++)) || true
  done

  if [[ ${CONTAINER_CHECKS} -eq 3 ]]; then
    pass "All 3 containers are running"
  else
    warn "Only ${CONTAINER_CHECKS}/3 containers running — run: docker-compose up -d"
  fi

  # Check service endpoints via HTTP
  check_docker_service "Qdrant" "6333" "http://localhost:6333/collections" || ((SERVICES_OK++))
  check_docker_service "GROBID" "8070" "http://localhost:8070/api/isalive" || ((SERVICES_OK++))
  check_docker_service "Neo4j HTTP" "7474" "http://localhost:7474" || ((SERVICES_OK++))
fi

# ── 2. Neo4j auth ──────────────────────────────────────────

section "Neo4j authentication"

NEO4J_AUTH_OK=0
# Try bolt connection using cypher-shell or just HTTP auth check
AUTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
  -u "neo4j:${NEO4J_PASSWORD}" \
  --max-time 10 \
  "http://localhost:7474/auth" 2>/dev/null || echo "000")

if [[ "${AUTH_RESPONSE}" =~ ^(200|201|204)$ ]]; then
  pass "Neo4j authentication successful (HTTP ${AUTH_RESPONSE})"
else
  # Also try the /db/neo4j endpoint which returns 200 on good auth
  DB_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -u "neo4j:${NEO4J_PASSWORD}" \
    --max-time 10 \
    "http://localhost:7474/db/neo4j/" 2>/dev/null || echo "000")
  if [[ "${DB_RESPONSE}" =~ ^(200|201)$ ]]; then
    pass "Neo4j database reachable with password (HTTP ${DB_RESPONSE})"
  else
    fail "Neo4j authentication FAILED (HTTP ${DB_RESPONSE}) — check CITEWEAVE_NEO4J_PASSWORD"
    NEO4J_AUTH_OK=1
  fi
fi

# ── 3. Python environment ───────────────────────────────────

section "Python environment"

PYENV_OK=0
if command -v python3 &>/dev/null; then
  pass "python3 found: $(python3 --version)"
else
  fail "python3 not found"
  PYENV_OK=1
fi

# Check key dependencies
for pkg in langchain langchain_openai langchain_ollama dotenv; do
  if python3 -c "import ${pkg}" 2>/dev/null; then
    pass "Python package '${pkg}' is installed"
  else
    warn "Python package '${pkg}' is NOT installed — run: pip install -r requirements.txt"
    ((PYENV_OK++))
  fi
done

# ── 4. Config files ─────────────────────────────────────────

section "Configuration files"

CONFIG_OK=0
for f in config/model_config.json config/neo4j_config.json; do
  if [[ -f "${f}" ]]; then
    pass "Found: ${f}"
  else
    fail "MISSING: ${f}"
    ((CONFIG_OK++))
  fi
done

if [[ ! -f .env ]]; then
  warn ".env not found — copy .env_template to .env and edit values"
  ((CONFIG_OK++))
else
  pass ".env found"
fi

# Check .env has a non-default Neo4j password
if grep -q "0xC1735" .env 2>/dev/null; then
  warn "Neo4j password is still the default '0xC1735' — change before production"
fi

# ── 5. CLI syntax check ────────────────────────────────────

section "CLI health check"

if [[ -f src/core/cli.py ]]; then
  CLI_CHECK=$(python3 -m py_compile src/core/cli.py 2>&1 && echo "ok" || echo "fail")
  if [[ "${CLI_CHECK}" == "ok" ]]; then
    pass "src/core/cli.py compiles cleanly"
  else
    fail "src/core/cli.py has syntax errors"
    ((PYENV_OK++))
  fi
else
  warn "src/core/cli.py not found — are you in the project root?"
fi

# ── 6. LLM mode ────────────────────────────────────────────

section "LLM mode"

LLM_MODE="${CITEWEAVE_LLM_PROVIDER:-not set}"
if [[ "${LLM_MODE}" == "openclaw" ]]; then
  API_BASE="${CITEWEAVE_LLM_API_BASE:-http://localhost:18789/v1}"
  MODEL="${CITEWEAVE_LLM_MODEL:-openai-codex/gpt-5.4}"
  pass "Mode: openclaw"
  info "  API base : ${API_BASE}"
  info "  Model    : ${MODEL}"

  # Quick connectivity check to gateway
  GATEWAY_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time 5 \
    "${API_BASE%}/v1/models" 2>/dev/null || echo "000")
  if [[ "${GATEWAY_CODE}" =~ ^(200|404)$ ]]; then
    pass "OpenClaw gateway reachable (HTTP ${GATEWAY_CODE})"
  else
    warn "OpenClaw gateway NOT reachable at ${API_BASE} (HTTP ${GATEWAY_CODE})"
    warn "Make sure the OpenClaw gateway is running on this host."
  fi
elif [[ "${LLM_MODE}" == "openai" ]]; then
  pass "Mode: openai"
  if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${CITEWEAVE_LLM_API_KEY:-}" ]]; then
    fail "OPENAI_API_KEY / CITEWEAVE_LLM_API_KEY not set"
  else
    pass "OpenAI API key is configured"
  fi
elif [[ "${LLM_MODE}" == "ollama" ]]; then
  pass "Mode: ollama"
  OLLAMA_BASE="${CITEWEAVE_LLM_API_BASE:-http://localhost:11434}"
  info "  API base : ${OLLAMA_BASE}"
else
  warn "CITEWEAVE_LLM_PROVIDER not set — defaulting to openai"
fi

# ── Summary ─────────────────────────────────────────────────

section "Summary"

ERRORS=$((SERVICES_OK + NEO4J_AUTH_OK + CONFIG_OK + PYENV_OK))
if [[ ${ERRORS} -eq 0 ]]; then
  echo -e "  ${GREEN}${BOLD}All checks passed!${RESET} CiteWeave is ready to use."
  echo ""
  echo "  Next steps:"
  echo "    1. Upload a paper:  python -m src.core.cli upload path/to/paper.pdf"
  echo "    2. Ask a question:  python -m src.core.cli query \"<your question>\""
  echo ""
  exit 0
else
  echo -e "  ${RED}${ERRORS} error(s) found.${RESET} Please fix the items marked ${RED}✗${RESET} above."
  echo ""
  exit 1
fi
