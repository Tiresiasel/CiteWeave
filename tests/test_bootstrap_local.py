import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_LOCAL = REPO_ROOT / "scripts" / "bootstrap_local.sh"


def _bootstrap_env_python() -> str:
    script = BOOTSTRAP_LOCAL.read_text(encoding="utf-8")
    match = re.search(r"python3 - <<'PY'\n(?P<body>.*?)\nPY\n", script, re.DOTALL)
    assert match is not None
    return match.group("body")


def test_bootstrap_local_generates_password_before_services_start():
    script = BOOTSTRAP_LOCAL.read_text(encoding="utf-8")
    assert script.index("generated local CITEWEAVE_NEO4J_PASSWORD") < script.index("bash scripts/deploy_local_stack.sh")


def test_bootstrap_local_replaces_template_neo4j_password(tmp_path):
    (tmp_path / ".env").write_text(
        "CITEWEAVE_LLM_PROVIDER=openclaw\nCITEWEAVE_NEO4J_PASSWORD=change-me-local-only\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "-c", _bootstrap_env_python()],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "CITEWEAVE_NEO4J_PASSWORD=change-me-local-only" not in env_text
    match = re.search(r"^CITEWEAVE_NEO4J_PASSWORD=(\w+)$", env_text, re.MULTILINE)
    assert match is not None
    assert len(match.group(1)) == 48


def test_bootstrap_local_migrates_legacy_neo4j_password(tmp_path):
    (tmp_path / ".env").write_text(
        "NEO4J_PASSWORD=legacy-local-password\nCITEWEAVE_NEO4J_PASSWORD=change-me-local-only\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "-c", _bootstrap_env_python()],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CITEWEAVE_NEO4J_PASSWORD=legacy-local-password" in (tmp_path / ".env").read_text(encoding="utf-8")
