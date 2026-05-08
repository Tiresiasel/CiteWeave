#!/usr/bin/env bash
# Deploy CiteWeave's local infrastructure stack.
#
# This is the Docker Compose boundary OpenClaw should use for databases and
# local services. It starts Neo4j, Qdrant, and GROBID; it does not ingest PDFs
# and does not run research queries.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

printf '\n[deploy_local_stack] project root: %s\n' "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "[deploy_local_stack] ERROR: docker is not installed or not on PATH" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[deploy_local_stack] ERROR: Docker daemon is not accessible for this user" >&2
  exit 1
fi

if command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE=(docker-compose)
elif docker compose version >/dev/null 2>&1; then
  DOCKER_COMPOSE=(docker compose)
else
  echo "[deploy_local_stack] ERROR: Docker Compose is not available" >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env_template .env
  printf '[deploy_local_stack] created .env from .env_template\n'
fi

python3 - <<'PY'
from pathlib import Path
import secrets
import stat

PLACEHOLDER_PASSWORDS = {"", "change-me-local-only", "CHANGE_ME_LOCAL_ONLY"}


def normalize(value: str | None) -> str:
    return (value or "").strip().strip('"').strip("'")


def is_placeholder(value: str | None) -> bool:
    return normalize(value) in PLACEHOLDER_PASSWORDS


env_path = Path(".env")
lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
legacy_password = None
current_index = None
current_password = None

for index, line in enumerate(lines):
    if "=" not in line or line.lstrip().startswith("#"):
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    if key == "CITEWEAVE_NEO4J_PASSWORD":
        current_index = index
        current_password = value.strip()
    elif key == "NEO4J_PASSWORD":
        legacy_password = value.strip()

if is_placeholder(current_password):
    if not is_placeholder(legacy_password):
        password = normalize(legacy_password)
        action = "migrated NEO4J_PASSWORD -> CITEWEAVE_NEO4J_PASSWORD"
    else:
        password = secrets.token_hex(24)
        action = "generated local CITEWEAVE_NEO4J_PASSWORD in .env"

    replacement = f"CITEWEAVE_NEO4J_PASSWORD={password}"
    if current_index is None:
        lines.append(replacement)
    else:
        lines[current_index] = replacement
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[deploy_local_stack] {action}")
else:
    print("[deploy_local_stack] CITEWEAVE_NEO4J_PASSWORD already set")

try:
    env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
except OSError:
    pass
PY

printf '[deploy_local_stack] validating docker-compose.yml...\n'
"${DOCKER_COMPOSE[@]}" config >/dev/null

printf '[deploy_local_stack] starting Neo4j, Qdrant, and GROBID...\n'
"${DOCKER_COMPOSE[@]}" up -d neo4j qdrant grobid

cat <<'EOF'

[deploy_local_stack] Docker Compose stack requested.

Local services:
  - Neo4j HTTP : http://127.0.0.1:7474
  - Neo4j Bolt : bolt://127.0.0.1:7687
  - Qdrant REST: http://127.0.0.1:6333
  - GROBID     : http://127.0.0.1:8070

Verify after startup:
  bash scripts/deployment_check.sh
EOF
