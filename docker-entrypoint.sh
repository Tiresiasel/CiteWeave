#!/usr/bin/env bash
set -euo pipefail

echo "[CiteWeave] Entrypoint starting..."

# Defaults
: "${CITEWEAVE_DATA_DIR:=/app/data}"
: "${CITEWEAVE_SETTINGS_PATH:=${CITEWEAVE_DATA_DIR}/settings.json}"

# 1) Prepare data directories and default settings
mkdir -p "${CITEWEAVE_DATA_DIR}"
if [ ! -f "${CITEWEAVE_SETTINGS_PATH}" ]; then
  echo "[CiteWeave] Creating default settings at ${CITEWEAVE_SETTINGS_PATH}"
  cat > "${CITEWEAVE_SETTINGS_PATH}" <<EOF
{
  "api_base": "",
  "theme": "dark",
  "display_name": "",
  "openai_key": ""
}
EOF
fi

# 2) Optionally wait for dependent services if explicitly enabled
wait_for_url() {
  url="$1"; name="$2"; timeout_sec="${3:-60}"
  echo "[CiteWeave] Waiting for ${name} at ${url}..."
  end=$((SECONDS+timeout_sec))
  while [ $SECONDS -lt $end ]; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[CiteWeave] ${name} is up."
      return 0
    fi
    sleep 2
  done
  echo "[CiteWeave] WARNING: Timed out waiting for ${name} (${url})"
  return 1
}

if [ "${CITEWEAVE_WAIT_FOR_DEPS:-0}" = "1" ]; then
  if [ -n "${QDRANT_URL:-}" ]; then
    wait_for_url "${QDRANT_URL%/}/readyz" "Qdrant" 90 || true
  fi
  if [ -n "${GROBID_URL:-}" ]; then
    wait_for_url "${GROBID_URL%/}/api/isalive" "GROBID" 90 || true
  fi
  if [ -n "${NEO4J_URI:-}" ]; then
    neo4j_http="${NEO4J_HTTP:-}"
    if [ -n "$neo4j_http" ]; then
      wait_for_url "$neo4j_http" "Neo4j" 90 || true
    fi
  fi
else
  echo "[CiteWeave] Skipping dependency wait; UI will be available immediately."
fi

echo "[CiteWeave] Starting API server..."
exec "$@"


