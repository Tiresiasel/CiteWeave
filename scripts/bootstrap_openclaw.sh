#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

printf '\n[bootstrap_openclaw] project root: %s\n' "$ROOT_DIR"

if [[ ! -f .env ]]; then
  cp .env_template .env
  printf '[bootstrap_openclaw] created .env from .env_template\n'
else
  printf '[bootstrap_openclaw] .env already exists, patching OpenClaw settings in place\n'
fi

python3 - <<'PY'
from pathlib import Path
env_path = Path('.env')
lines = env_path.read_text(encoding='utf-8').splitlines() if env_path.exists() else []
updates = {
    'CITEWEAVE_LLM_PROVIDER': 'openclaw',
    'CITEWEAVE_LLM_MODEL': 'openai-codex/gpt-5.4',
    'CITEWEAVE_LLM_API_BASE': 'http://localhost:18789/v1',
}
seen = set()
out = []
for line in lines:
    if '=' in line and not line.lstrip().startswith('#'):
        key = line.split('=', 1)[0].strip()
        if key in updates:
            out.append(f'{key}={updates[key]}')
            seen.add(key)
            continue
    out.append(line)
for key, value in updates.items():
    if key not in seen:
        out.append(f'{key}={value}')
env_path.write_text('\n'.join(out).rstrip() + '\n', encoding='utf-8')
PY

bash scripts/bootstrap_local.sh

printf '[bootstrap_openclaw] checking local OpenClaw gateway...\n'
openclaw gateway status || true

cat <<'EOF'

[bootstrap_openclaw] Done.

This project is now configured to route CiteWeave LLM calls through the local OpenClaw gateway.

Recommended verification:
  1. bash scripts/deployment_check.sh
  2. .venv/bin/python -m src.core.cli routes
  3. .venv/bin/python -m src.core.cli chat

Then call CiteWeave from your OpenClaw session.
EOF
