import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_literature_pdfs.py"


def run_sync(*args, env=None):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    return result


def test_sync_literature_dry_run_resolves_zotero_storage(tmp_path):
    zotero = tmp_path / "Zotero"
    storage = zotero / "storage" / "ABCD1234"
    storage.mkdir(parents=True)
    (zotero / "zotero.sqlite").write_text("placeholder", encoding="utf-8")
    (storage / "paper.pdf").write_bytes(b"%PDF-1.4\n")

    result = run_sync("--source", str(zotero), "--reference-manager", "zotero", "--dry-run", "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["reference_manager"] == "zotero"
    assert payload["resolved_pdf_source"] == str((zotero / "storage").resolve())
    assert payload["pdf_count"] == 1
    assert payload["command"][1] in {"batch-upload", "-m"}
    assert "--resume" in payload["command"]


def test_sync_literature_dry_run_accepts_plain_pdf_folder(tmp_path):
    pdfs = tmp_path / "papers" / "nested"
    pdfs.mkdir(parents=True)
    (pdfs / "paper.pdf").write_bytes(b"%PDF-1.4\n")

    result = run_sync("--source", str(tmp_path / "papers"), "--reference-manager", "pdf_folder", "--dry-run", "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["reference_manager"] == "pdf_folder"
    assert payload["resolved_pdf_source"] == str((tmp_path / "papers").resolve())
    assert payload["pdf_count"] == 1


def test_sync_literature_dry_run_accepts_single_pdf(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    result = run_sync("--source", str(pdf), "--reference-manager", "pdf_folder", "--dry-run", "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["source_type"] == "single_pdf"
    assert payload["pdf_count"] == 1
    assert payload["command"][1] in {"upload", "-m"}


def test_sync_literature_default_detection_failure_requests_custom_location(tmp_path):
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "CITEWEAVE_LITERATURE_SOURCE_DIR": "",
        "CITEWEAVE_ZOTERO_LIBRARY_DIR": "",
    }

    result = run_sync("--reference-manager", "mendeley", "--dry-run", "--json", env=env)

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert "Custom Location" in payload["error"]
