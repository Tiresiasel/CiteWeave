import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "apply_install_choices.py"


def parse_env(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def run_apply(tmp_path, *args):
    env_path = tmp_path / ".env"
    state_path = tmp_path / "config" / "install_session.local.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--env-path",
            str(env_path),
            "--state-path",
            str(state_path),
            "--json",
            *args,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout), env_path, state_path


def test_apply_install_choices_writes_codex_zotero_bge_sync(tmp_path):
    zotero = tmp_path / "Zotero"
    (zotero / "storage").mkdir(parents=True)

    report, env_path, state_path = run_apply(
        tmp_path,
        "--research-agent",
        "Codex",
        "--reference-manager",
        "zotero",
        "--source-location-mode",
        "custom",
        "--source-dir",
        str(zotero),
        "--embedding-mode",
        "local",
        "--embedding-profile",
        "bge_large_en",
        "--sync-schedule",
        "every_5_minutes",
        "--processors",
        "10",
        "--skip-failed",
    )

    env_values = parse_env(env_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert env_values["CITEWEAVE_RESEARCH_AGENT"] == "Codex"
    assert env_values["CITEWEAVE_QUERY_ORCHESTRATION"] == "agent_managed"
    assert env_values["CITEWEAVE_REFERENCE_MANAGER"] == "zotero"
    assert env_values["CITEWEAVE_LITERATURE_SOURCE_DIR"] == str(zotero.resolve())
    assert env_values["CITEWEAVE_ZOTERO_LIBRARY_DIR"] == str(zotero.resolve())
    assert env_values["CITEWEAVE_EMBEDDING_PROFILE"] == "bge_large_en"
    assert env_values["CITEWEAVE_EMBEDDING_MODEL"] == "BAAI/bge-large-en-v1.5"
    assert env_values["CITEWEAVE_EMBEDDING_DIMENSIONS"] == "1024"

    assert state["research_agent"] == "Codex"
    assert state["scheduler_adapter"] == "codex_heartbeat"
    assert state["reference_manager"] == "zotero"
    assert state["resolved_pdf_source"] == str((zotero / "storage").resolve())
    assert state["sync"]["schedule"] == "every_5_minutes"
    assert state["sync"]["resume_only"] is True
    assert "CITEWEAVE_EMBEDDING_API_KEY" not in report["applied_env_keys"]


def test_apply_install_choices_writes_openai_large_without_secret_in_state(tmp_path):
    report, env_path, state_path = run_apply(
        tmp_path,
        "--embedding-mode",
        "api",
        "--api-provider",
        "openai",
        "--embedding-model",
        "text-embedding-3-large",
        "--api-key",
        "sk-test-secret",
    )

    env_values = parse_env(env_path)
    state_text = state_path.read_text(encoding="utf-8")
    state = json.loads(state_text)

    assert env_values["CITEWEAVE_EMBEDDING_PROVIDER"] == "openai"
    assert env_values["CITEWEAVE_EMBEDDING_PROFILE"] == ""
    assert env_values["CITEWEAVE_EMBEDDING_MODEL"] == "text-embedding-3-large"
    assert env_values["CITEWEAVE_EMBEDDING_DIMENSIONS"] == "3072"
    assert env_values["CITEWEAVE_EMBEDDING_API_KEY"] == "sk-test-secret"
    assert state["embedding"]["api_provider"] == "openai"
    assert state["embedding"]["api_key_configured"] is True
    assert "sk-test-secret" not in state_text
    assert report["warnings"]


def test_apply_install_choices_single_pdf_disables_sync(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    _, _, state_path = run_apply(
        tmp_path,
        "--single-pdf",
        str(pdf),
        "--sync-schedule",
        "every_5_minutes",
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["install_mode"] == "single_pdf_test"
    assert state["single_pdf_path"] == str(pdf.resolve())
    assert state["sync"]["enabled"] is False
