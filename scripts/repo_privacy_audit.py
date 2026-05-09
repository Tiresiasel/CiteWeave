#!/usr/bin/env python3
"""Fail fast when private paths, secrets, or runtime artifacts enter git.

This audit intentionally inspects tracked files, plus local git remotes because
remote URLs are not tracked but are easy to leak into logs.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRACKED = subprocess.check_output(["git", "-C", str(REPO), "ls-files"], text=True).splitlines()

TEXT_PATTERNS = [
    (re.compile(r"/home/tiresias|\.openclaw/workspace"), "absolute local workspace path"),
    (re.compile(r"machespresso@gmail\.com", re.I), "personal email"),
    (re.compile(r"\.secrets/|token file /home/", re.I), "local secret/token path"),
    (re.compile(r"x-access-token:[^@\s]+@github\.com", re.I), "GitHub token embedded in URL"),
    (re.compile(r"github_pat_[A-Za-z0-9_]+"), "GitHub fine-grained token"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"), "GitHub token"),
    (re.compile(r"NEO4J_AUTH=neo4j/(12345678|0xC1735)"), "hard-coded Neo4j password"),
    (re.compile(r"NEO4J_PASSWORD\s*=\s*[\"'](12345678|0xC1735)[\"']"), "hard-coded Neo4j password"),
    (re.compile(r'"password"\s*:\s*"(12345678|0xC1735)"'), "hard-coded Neo4j password in config"),
]

RUNTIME_ARTIFACT_PATTERNS = [
    re.compile(r"(^|/)data/(?!README\.md$)"),
    re.compile(r"^src/data/(?!README\.md$)"),
    re.compile(r"\.sqlite$"),
    re.compile(r"(^|/)\.lock$"),
]

violations: list[tuple[str, str]] = []

for rel in TRACKED:
    if rel.startswith(".venv/") or rel == "scripts/repo_privacy_audit.py":
        continue

    if any(rx.search(rel) for rx in RUNTIME_ARTIFACT_PATTERNS):
        violations.append((rel, "tracked runtime/generated artifact"))
        continue

    p = REPO / rel
    if not p.is_file():
        continue
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for rx, label in TEXT_PATTERNS:
        if rx.search(text):
            violations.append((rel, label))
            break

remote_output = subprocess.run(
    ["git", "-C", str(REPO), "remote", "-v"],
    text=True,
    capture_output=True,
    check=False,
).stdout
for rx, label in TEXT_PATTERNS:
    if rx.search(remote_output):
        violations.append((".git/config", label))
        break

if violations:
    print("PRIVACY_AUDIT_FAIL")
    for rel, label in violations:
        print(f"{rel}\t{label}")
    sys.exit(1)

print("PRIVACY_AUDIT_OK")
