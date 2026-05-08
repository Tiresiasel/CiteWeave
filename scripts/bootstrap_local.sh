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

python3 - <<'PY'
from pathlib import Path
import secrets
import stat

PLACEHOLDER_PASSWORDS = {"", "change-me-local-only", "CHANGE_ME_LOCAL_ONLY"}


def normalize(value):
    return value.strip().strip('"').strip("'")


def is_placeholder(value):
    return value is None or normalize(value) in PLACEHOLDER_PASSWORDS


env_path = Path('.env')
lines = env_path.read_text(encoding='utf-8').splitlines() if env_path.exists() else []
legacy_password = None
current_index = None
current_password = None

for index, line in enumerate(lines):
    if '=' not in line or line.lstrip().startswith('#'):
        continue
    key, value = line.split('=', 1)
    key = key.strip()
    if key == 'CITEWEAVE_NEO4J_PASSWORD':
        current_index = index
        current_password = value.strip()
    elif key == 'NEO4J_PASSWORD':
        legacy_password = value.strip()

if is_placeholder(current_password):
    if not is_placeholder(legacy_password):
        password = normalize(legacy_password)
        action = 'migrated NEO4J_PASSWORD -> CITEWEAVE_NEO4J_PASSWORD'
    else:
        password = secrets.token_hex(24)
        action = 'generated local CITEWEAVE_NEO4J_PASSWORD in .env'

    replacement = f'CITEWEAVE_NEO4J_PASSWORD={password}'
    if current_index is None:
        lines.append(replacement)
    else:
        lines[current_index] = replacement
    env_path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    print(f'[bootstrap_local] {action}')
else:
    print('[bootstrap_local] CITEWEAVE_NEO4J_PASSWORD already set')

try:
    env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
except OSError:
    pass
PY

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  printf '[bootstrap_local] created virtualenv .venv\n'
else
  printf '[bootstrap_local] .venv already exists\n'
fi

. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e . --no-deps
python - <<'PY'
import nltk

required_resources = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}

for package, resource_path in required_resources.items():
    try:
        nltk.data.find(resource_path)
        print(f"[bootstrap_local] NLTK resource already present: {package}")
    except LookupError:
        print(f"[bootstrap_local] downloading NLTK resource: {package}")
        nltk.download(package, quiet=True, raise_on_error=True)
PY

printf '[bootstrap_local] deploying local Docker Compose stack...\n'
bash scripts/deploy_local_stack.sh

printf '[bootstrap_local] running deployment check...\n'
bash scripts/deployment_check.sh

cat <<'EOF'

[bootstrap_local] Done.

Next steps:
  1. Edit .env if needed (provider=openclaw by default; set openai/ollama explicitly if needed)
  2. Upload a PDF:
       .venv/bin/citeweave upload path/to/paper.pdf
  3. Start interactive research:
       .venv/bin/citeweave chat
EOF
