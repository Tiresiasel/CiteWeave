import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_zotero_pdfs.py"


def test_sync_zotero_dry_run_resolves_zotero_storage(tmp_path):
    zotero = tmp_path / "Zotero"
    storage = zotero / "storage" / "ABCD1234"
    storage.mkdir(parents=True)
    (zotero / "zotero.sqlite").write_text("placeholder")
    (storage / "paper.pdf").write_bytes(b"%PDF-1.4\n")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(zotero), "--dry-run", "--json"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["resolved_pdf_source"] == str((zotero / "storage").resolve())
    assert payload["pdf_count"] == 1
    assert payload["command"][1] in {"batch-upload", "-m"}


def test_sync_zotero_forwards_resume_safety_flags(tmp_path):
    zotero = tmp_path / "Zotero"
    storage = zotero / "storage" / "ABCD1234"
    storage.mkdir(parents=True)
    (zotero / "zotero.sqlite").write_text("placeholder")
    (storage / "paper.pdf").write_bytes(b"%PDF-1.4\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(zotero),
            "--dry-run",
            "--json",
            "--sequential",
            "--skip-failed",
            "--file-timeout-seconds",
            "180",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "--sequential" in payload["command"]
    assert "--skip-failed" in payload["command"]
    assert payload["command"][-2:] == ["--file-timeout-seconds", "180"]


def test_sync_zotero_errors_without_source():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--json"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        env={},
    )

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert "CITEWEAVE_ZOTERO_LIBRARY_DIR" in payload["error"]
