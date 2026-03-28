#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

printf '\n[bootstrap_local] project root: %s\n' "$ROOT_DIR"

if [[ ! -f .env ]]; then
  cp .env_template .env
  printf '[bootstrap_local] created .env from .env_template\n'
else
  printf '[bootstrap_local] .env already exists, leaving it untouched\n'
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  printf '[bootstrap_local] created virtualenv .venv\n'
else
  printf '[bootstrap_local] .venv already exists\n'
fi

. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader punkt

printf '[bootstrap_local] starting Docker services...\n'
docker-compose up -d

printf '[bootstrap_local] running deployment check...\n'
bash scripts/deployment_check.sh

cat <<'EOF'

[bootstrap_local] Done.

Next steps:
  1. Edit .env if needed (provider=openai by default for standalone CLI use)
  2. Upload a PDF:
       .venv/bin/python -m src.core.cli upload path/to/paper.pdf
  3. Start interactive research:
       .venv/bin/python -m src.core.cli chat
EOF
