#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRACKED = subprocess.check_output(['git', '-C', str(REPO), 'ls-files'], text=True).splitlines()

PATTERNS = [
    (re.compile(r'/home/tiresias|\.openclaw/workspace'), 'absolute local workspace path'),
    (re.compile(r'machespresso@gmail\.com', re.I), 'personal email'),
    (re.compile(r'\.secrets/|token file /home/', re.I), 'local secret/token path'),
    (re.compile(r'NEO4J_AUTH=neo4j/12345678'), 'hard-coded Neo4j password'),
    (re.compile(r'"password"\s*:\s*"12345678"'), 'hard-coded Neo4j password in config'),
]

violations = []
for rel in TRACKED:
    if rel.startswith('.venv/') or rel == 'scripts/repo_privacy_audit.py':
        continue
    p = REPO / rel
    if not p.is_file():
        continue
    try:
        text = p.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        continue
    for rx, label in PATTERNS:
        if rx.search(text):
            violations.append((rel, label))
            break

if violations:
    print('PRIVACY_AUDIT_FAIL')
    for rel, label in violations:
        print(f'{rel}\t{label}')
    sys.exit(1)

print('PRIVACY_AUDIT_OK')
