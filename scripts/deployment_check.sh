#!/bin/bash
# ============================================================
# scripts/deployment_check.sh
# CiteWeave deployment smoke-test.
# Run this after `bash scripts/deploy_local_stack.sh` to verify everything
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

NEO4J_PASSWORD="${CITEWEAVE_NEO4J_PASSWORD:-${NEO4J_PASSWORD:-}}"
NEO4J_HOST="${NEO4J_HOST:-localhost}"
QDRANT_HOST="${QDRANT_HOST:-localhost}"
GROBID_HOST="${GROBID_HOST:-localhost}"

if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

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
if [[ -z "${NEO4J_PASSWORD}" ]]; then
  fail "CITEWEAVE_NEO4J_PASSWORD is not set"
  NEO4J_AUTH_OK=1
elif [[ "${NEO4J_PASSWORD}" == "change-me-local-only" || "${NEO4J_PASSWORD}" == "CHANGE_ME_LOCAL_ONLY" ]]; then
  fail "CITEWEAVE_NEO4J_PASSWORD is still the template placeholder"
  NEO4J_AUTH_OK=1
else
  if [[ -z "${PYTHON_BIN}" ]]; then
    fail "Neo4j authentication could not be checked because python3 is not available"
    NEO4J_AUTH_OK=1
  else
    BOLT_AUTH_OUTPUT=$(NEO4J_PASSWORD="${NEO4J_PASSWORD}" NEO4J_HOST="${NEO4J_HOST}" "${PYTHON_BIN}" - <<'PY' 2>&1
import os
import sys

try:
    from neo4j import GraphDatabase
except Exception as exc:
    print(f"neo4j-driver-unavailable: {exc}")
    sys.exit(2)

host = os.environ.get("NEO4J_HOST", "localhost")
password = os.environ["NEO4J_PASSWORD"]
uri = f"bolt://{host}:7687"

driver = None
try:
    driver = GraphDatabase.driver(uri, auth=("neo4j", password))
    with driver.session(database="neo4j") as session:
        value = session.run("RETURN 1 AS ok").single()["ok"]
    if value != 1:
        raise RuntimeError(f"unexpected query result: {value!r}")
    print(f"{uri} RETURN 1")
except Exception as exc:
    print(f"bolt-failed: {type(exc).__name__}: {exc}")
    sys.exit(1)
finally:
    if driver is not None:
        driver.close()
PY
)
    BOLT_AUTH_CODE=$?

    if [[ ${BOLT_AUTH_CODE} -eq 0 ]]; then
      pass "Neo4j Bolt authentication successful (${BOLT_AUTH_OUTPUT})"
    else
      HTTP_AUTH_OUTPUT=$(NEO4J_PASSWORD="${NEO4J_PASSWORD}" NEO4J_HOST="${NEO4J_HOST}" "${PYTHON_BIN}" - <<'PY' 2>&1
import base64
import json
import os
import sys
import urllib.error
import urllib.request

host = os.environ.get("NEO4J_HOST", "localhost")
password = os.environ["NEO4J_PASSWORD"]
url = f"http://{host}:7474/db/neo4j/tx/commit"
body = json.dumps({"statements": [{"statement": "RETURN 1 AS ok"}]}).encode("utf-8")
token = base64.b64encode(f"neo4j:{password}".encode("utf-8")).decode("ascii")
request = urllib.request.Request(
    url,
    data=body,
    headers={
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
    },
)

try:
    with urllib.request.urlopen(request, timeout=10) as response:
        status = response.status
        payload = json.loads(response.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    print(f"http-failed: HTTP {exc.code}")
    sys.exit(1)
except Exception as exc:
    print(f"http-failed: {type(exc).__name__}: {exc}")
    sys.exit(1)

errors = payload.get("errors") or []
if status in (200, 201) and not errors:
    print(f"HTTP {status} tx commit")
    sys.exit(0)

print(f"http-failed: HTTP {status}, errors={errors!r}")
sys.exit(1)
PY
)
      HTTP_AUTH_CODE=$?

      if [[ ${HTTP_AUTH_CODE} -eq 0 ]]; then
        pass "Neo4j HTTP transaction authentication successful (${HTTP_AUTH_OUTPUT})"
      else
        fail "Neo4j authentication FAILED — Bolt: ${BOLT_AUTH_OUTPUT}; HTTP: ${HTTP_AUTH_OUTPUT}"
        NEO4J_AUTH_OK=1
      fi
    fi
  fi
fi

# ── 3. Python environment ───────────────────────────────────

section "Python environment"

PYENV_OK=0
if [[ -n "${PYTHON_BIN}" ]]; then
  pass "Python found: $(${PYTHON_BIN} --version) (${PYTHON_BIN})"
else
  fail "python3 not found"
  PYENV_OK=1
fi

# Check key dependencies
for pkg in langchain langchain_openai langchain_ollama dotenv sentence_transformers openai; do
  if [[ -n "${PYTHON_BIN}" ]] && ${PYTHON_BIN} -c "import ${pkg}" 2>/dev/null; then
    pass "Python package '${pkg}' is installed"
  else
    warn "Python package '${pkg}' is NOT installed — run: pip install -r requirements.txt"
    ((PYENV_OK++))
  fi
done

if [[ -n "${PYTHON_BIN}" ]]; then
  NLTK_CHECK_OUTPUT=$(${PYTHON_BIN} - <<'PY' 2>&1
import nltk

required_resources = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}
missing = []
for package, resource_path in required_resources.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        missing.append(package)

if missing:
    print(", ".join(missing))
    raise SystemExit(1)
PY
)
  NLTK_CHECK_CODE=$?
  if [[ ${NLTK_CHECK_CODE} -eq 0 ]]; then
    pass "NLTK resources 'punkt' and 'punkt_tab' are installed"
  else
    warn "NLTK resources missing: ${NLTK_CHECK_OUTPUT} — run: python -m nltk.downloader punkt punkt_tab"
    ((PYENV_OK++))
  fi
fi

# ── 4. Config files ─────────────────────────────────────────

section "Configuration files"

CONFIG_OK=0
for f in config/model_config.json config/qdrant_config.json config/neo4j_config.example.json; do
  if [[ -f "${f}" ]]; then
    pass "Found: ${f}"
  else
    fail "MISSING: ${f}"
    ((CONFIG_OK++))
  fi
done

if [[ -f config/neo4j_config.local.json ]]; then
  pass "Found ignored local override: config/neo4j_config.local.json"
elif [[ -f config/neo4j_config.json ]]; then
  warn "Found legacy config/neo4j_config.json; prefer ignored config/neo4j_config.local.json or .env"
else
  pass "Neo4j config will use environment variables plus the safe example template"
fi

if [[ ! -f .env ]]; then
  warn ".env not found — copy .env_template to .env and edit values"
  ((CONFIG_OK++))
else
  pass ".env found"
fi

# Check .env has the current Neo4j variable name.
if [[ -f .env ]] && ! grep -q '^CITEWEAVE_NEO4J_PASSWORD=' .env 2>/dev/null; then
  warn ".env does not set CITEWEAVE_NEO4J_PASSWORD"
fi
if [[ -f .env ]] && grep -q '^NEO4J_PASSWORD=' .env 2>/dev/null; then
  warn ".env uses legacy NEO4J_PASSWORD; rename it to CITEWEAVE_NEO4J_PASSWORD"
fi
if [[ -f .env ]] && grep -q '^CITEWEAVE_NEO4J_PASSWORD=\(change-me-local-only\|CHANGE_ME_LOCAL_ONLY\)$' .env 2>/dev/null; then
  warn ".env still uses the Neo4j template placeholder"
fi

# ── 5. Embedding mode ──────────────────────────────────────

section "Embedding mode"

EMBEDDING_OK=0
EMBEDDING_PROVIDER="${CITEWEAVE_EMBEDDING_PROVIDER:-local}"
EMBEDDING_MODEL="${CITEWEAVE_EMBEDDING_MODEL:-}"

if [[ "${EMBEDDING_PROVIDER}" == "local" ]]; then
  pass "Mode: local SentenceTransformers"
  info "  Model: ${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
  info "  Vector size: 384"
elif [[ "${EMBEDDING_PROVIDER}" == "openai" ]]; then
  pass "Mode: OpenAI embeddings"
  info "  Model: ${EMBEDDING_MODEL:-text-embedding-3-small}"
  info "  Vector size: ${CITEWEAVE_EMBEDDING_DIMENSIONS:-1536}"
  if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${CITEWEAVE_EMBEDDING_API_KEY:-}" ]]; then
    fail "OPENAI_API_KEY / CITEWEAVE_EMBEDDING_API_KEY not set for OpenAI embeddings"
    ((EMBEDDING_OK++))
  else
    pass "OpenAI embedding API key is configured"
  fi
else
  fail "Unsupported CITEWEAVE_EMBEDDING_PROVIDER='${EMBEDDING_PROVIDER}' — use local or openai"
  ((EMBEDDING_OK++))
fi

# ── 6. CLI syntax check ────────────────────────────────────

section "CLI health check"

if [[ -f src/core/cli.py ]]; then
  CLI_CHECK=$(${PYTHON_BIN:-python3} -m py_compile src/core/cli.py 2>&1 && echo "ok" || echo "fail")
  if [[ "${CLI_CHECK}" == "ok" ]]; then
    pass "src/core/cli.py compiles cleanly"
  else
    fail "src/core/cli.py has syntax errors"
    ((PYENV_OK++))
  fi
else
  warn "src/core/cli.py not found — are you in the project root?"
fi

# ── 7. LLM mode ────────────────────────────────────────────

section "LLM mode"

LLM_MODE="${CITEWEAVE_LLM_PROVIDER:-not set}"
if [[ "${LLM_MODE}" == "openclaw" ]]; then
  API_BASE="${CITEWEAVE_LLM_API_BASE:-http://localhost:18789/v1}"
  AGENT_TARGET="${CITEWEAVE_LLM_MODEL:-openclaw/default}"
  BACKEND_MODEL="${CITEWEAVE_OPENCLAW_BACKEND_MODEL:-${CITEWEAVE_LLM_BACKEND_MODEL:-}}"
  if [[ "${AGENT_TARGET}" != openclaw* ]]; then
    # Backwards compatibility: older .env files used CITEWEAVE_LLM_MODEL for
    # the provider/model. OpenClaw's OpenAI-compatible API expects that value
    # in x-openclaw-model and uses openclaw/default as the request model.
    BACKEND_MODEL="${BACKEND_MODEL:-${AGENT_TARGET}}"
    AGENT_TARGET="openclaw/default"
  fi

  pass "Mode: openclaw"
  info "  API base      : ${API_BASE}"
  info "  Agent target  : ${AGENT_TARGET}"
  if [[ -n "${BACKEND_MODEL}" ]]; then
    info "  Backend model : ${BACKEND_MODEL}"
  else
    info "  Backend model : gateway default"
  fi

  # Quick connectivity check to gateway. OpenClaw's OpenAI-compatible HTTP
  # surface is commonly protected by the gateway bearer token; treat 401 as a
  # configuration warning only when no token was supplied.
  CURL_AUTH_ARGS=()
  if [[ -n "${CITEWEAVE_LLM_API_KEY:-}" ]]; then
    CURL_AUTH_ARGS=(-H "Authorization: Bearer ${CITEWEAVE_LLM_API_KEY}")
  elif [[ -n "${OPENCLAW_GATEWAY_TOKEN:-}" ]]; then
    CURL_AUTH_ARGS=(-H "Authorization: Bearer ${OPENCLAW_GATEWAY_TOKEN}")
  fi

  GATEWAY_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time 5 \
    "${CURL_AUTH_ARGS[@]}" \
    "${API_BASE%/}/models" 2>/dev/null || echo "000")
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

ERRORS=$((SERVICES_OK + NEO4J_AUTH_OK + CONFIG_OK + PYENV_OK + EMBEDDING_OK))
if [[ ${ERRORS} -eq 0 ]]; then
  echo -e "  ${GREEN}${BOLD}All checks passed!${RESET} CiteWeave is ready to use."
  echo ""
  echo "  Next steps:"
  echo "    1. Upload a paper:  citeweave upload path/to/paper.pdf"
  echo "    2. Ask a question:  citeweave query \"<your question>\""
  echo ""
  exit 0
else
  echo -e "  ${RED}${ERRORS} error(s) found.${RESET} Please fix the items marked ${RED}✗${RESET} above."
  echo ""
  exit 1
fi
