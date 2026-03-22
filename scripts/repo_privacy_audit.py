#!/usr/bin/env python3
import fnmatch
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Violation = Tuple[str, str]
REPO = Path(__file__).resolve().parents[1]

CONTENT_PATTERNS = [
    (re.compile(r'/home/tiresias|\.openclaw/workspace'), 'absolute local workspace path'),
    (re.compile(r'machespresso@gmail\.com', re.I), 'personal email'),
    (re.compile(r'\.secrets/|token file /home/', re.I), 'local secret/token path'),
    (re.compile(r'NEO4J_AUTH=neo4j/12345678'), 'hard-coded Neo4j password'),
    (re.compile(r'"password"\s*:\s*"12345678"'), 'hard-coded Neo4j password in config'),
    (re.compile(r'ghp_[A-Za-z0-9]{36}\b'), 'GitHub personal access token'),
    (re.compile(r'github_pat_[A-Za-z0-9_]{20,}\b'), 'GitHub fine-grained personal access token'),
    (re.compile(r'sk-(?:proj-)?[A-Za-z0-9_-]{24,}\b'), 'OpenAI API key-like secret'),
    (re.compile(r'-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----'), 'private key material'),
]

PATH_GUARDS = [
    ('*.local.json', 'tracked local override config'),
    ('*_local.json', 'tracked local override config'),
    ('*.local.yaml', 'tracked local override config'),
    ('*.local.yml', 'tracked local override config'),
    ('*.local.toml', 'tracked local override config'),
    ('*.local.ini', 'tracked local override config'),
    ('*.local.env', 'tracked local override config'),
    ('.env', 'tracked local environment file'),
    ('.env.*', 'tracked local environment file'),
]

EXCLUDED_PATHS = {
    'scripts/repo_privacy_audit.py',
}
EXCLUDED_PREFIXES = ('.venv/',)


def tracked_files(repo: Path) -> List[str]:
    return subprocess.check_output(['git', '-C', str(repo), 'ls-files'], text=True).splitlines()



def should_skip(rel_path: str) -> bool:
    return rel_path in EXCLUDED_PATHS or rel_path.startswith(EXCLUDED_PREFIXES)



def scan_repo(repo: Path, tracked: Sequence[str] | None = None) -> List[Violation]:
    tracked = tracked if tracked is not None else tracked_files(repo)
    violations: List[Violation] = []

    for rel in tracked:
        if should_skip(rel):
            continue

        for pattern, label in PATH_GUARDS:
            if fnmatch.fnmatch(rel, pattern):
                violations.append((rel, label))
                break
        else:
            path = repo / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            for rx, label in CONTENT_PATTERNS:
                if rx.search(text):
                    violations.append((rel, label))
                    break

    return violations



def emit_report(violations: Iterable[Violation]) -> int:
    violations = list(violations)
    if violations:
        print('PRIVACY_AUDIT_FAIL')
        for rel, label in violations:
            print(f'{rel}\t{label}')
        return 1

    print('PRIVACY_AUDIT_OK')
    return 0


if __name__ == '__main__':
    sys.exit(emit_report(scan_repo(REPO)))
