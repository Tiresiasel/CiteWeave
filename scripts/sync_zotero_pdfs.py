#!/usr/bin/env python3
"""Continuously ingest PDFs from a Zotero data directory.

This is the Zotero-specific compatibility entrypoint. New AI-facing installers
should prefer scripts/sync_literature_pdfs.py, which handles Zotero, Mendeley,
EndNote, generic PDF folders, and single-PDF dry runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_zotero_pdf_source(source: str | None) -> Path:
    """Resolve a Zotero data directory or storage directory to a PDF source."""
    raw_source = source or os.environ.get("CITEWEAVE_ZOTERO_LIBRARY_DIR", "")
    if not raw_source.strip():
        raise ValueError(
            "Zotero source is not configured. Set CITEWEAVE_ZOTERO_LIBRARY_DIR "
            "or pass --source /path/to/Zotero."
        )

    candidate = Path(raw_source).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Zotero source does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Zotero source is not a directory: {candidate}")

    # Common layout: /path/to/Zotero/zotero.sqlite + /path/to/Zotero/storage/**.pdf
    storage_child = candidate / "storage"
    if storage_child.is_dir():
        return storage_child

    # Also accept the storage directory itself, or any exported PDF directory.
    return candidate


def discover_pdfs(source_dir: Path) -> List[Path]:
    return sorted(path for path in source_dir.rglob("*.pdf") if path.is_file())


def citeweave_command(
    root: Path,
    source_dir: Path,
    resume: bool,
    force_restart: bool,
    clear_progress: bool,
    processors: int | None,
    sequential: bool,
    skip_failed: bool,
    file_timeout_seconds: int | None,
) -> List[str]:
    citeweave_bin = root / ".venv" / "bin" / "citeweave"
    if citeweave_bin.exists():
        command = [str(citeweave_bin), "batch-upload", str(source_dir)]
    else:
        command = [sys.executable, "-m", "src.core.cli", "batch-upload", str(source_dir)]

    if resume and not force_restart:
        command.append("--resume")
    if force_restart:
        command.append("--force-restart")
    if clear_progress:
        command.append("--clear-progress")
    if sequential:
        command.append("--sequential")
    elif processors is not None:
        command.extend(["--processors", str(processors)])
    if skip_failed:
        command.append("--skip-failed")
    if file_timeout_seconds:
        command.extend(["--file-timeout-seconds", str(file_timeout_seconds)])
    return command


def build_report(source_arg: str | None, source_dir: Path, pdfs: List[Path], command: List[str], dry_run: bool) -> Dict[str, Any]:
    return {
        "status": "dry_run" if dry_run else "ready",
        "source_arg": source_arg or os.environ.get("CITEWEAVE_ZOTERO_LIBRARY_DIR", ""),
        "resolved_pdf_source": str(source_dir),
        "pdf_count": len(pdfs),
        "sample_pdfs": [str(path) for path in pdfs[:10]],
        "command": command,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Zotero PDFs into CiteWeave using resumable batch upload.")
    parser.add_argument("--source", help="Zotero data directory, Zotero storage directory, or exported PDF directory.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the source and list PDFs without running ingestion.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    parser.add_argument("--no-resume", action="store_true", help="Do not pass --resume to CiteWeave batch-upload.")
    parser.add_argument("--force-restart", action="store_true", help="Reprocess all PDFs, ignoring previous progress.")
    parser.add_argument("--clear-progress", action="store_true", help="Clear batch progress for this source before ingestion.")
    parser.add_argument("--processors", type=int, help="Number of CiteWeave batch-upload workers to use.")
    parser.add_argument("--sequential", action="store_true", help="Force sequential CiteWeave batch-upload processing.")
    parser.add_argument("--skip-failed", action="store_true", help="Do not retry files already marked failed by a previous run.")
    parser.add_argument("--file-timeout-seconds", type=int, help="Sequential mode only: fail and continue if one PDF exceeds this many seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()

    try:
        source_dir = resolve_zotero_pdf_source(args.source)
        pdfs = discover_pdfs(source_dir)
        command = citeweave_command(
            root,
            source_dir,
            resume=not args.no_resume,
            force_restart=args.force_restart,
            clear_progress=args.clear_progress,
            processors=args.processors,
            sequential=args.sequential,
            skip_failed=args.skip_failed,
            file_timeout_seconds=args.file_timeout_seconds,
        )
        report = build_report(args.source, source_dir, pdfs, command, args.dry_run)

        if args.dry_run:
            if args.json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print(f"Resolved PDF source: {source_dir}")
                print(f"PDF files found: {len(pdfs)}")
                for path in pdfs[:10]:
                    print(f"  - {path}")
                if len(pdfs) > 10:
                    print(f"  ... {len(pdfs) - 10} more")
                print("Command that would run:")
                print("  " + " ".join(command))
            return 0

        if not pdfs:
            report["status"] = "no_pdfs"
            if args.json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print(f"No PDF files found under {source_dir}; nothing to ingest.")
            return 0

        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))

        completed = subprocess.run(command, cwd=root, check=False)
        return completed.returncode
    except Exception as exc:
        error = {"status": "error", "error": str(exc)}
        if args.json:
            print(json.dumps(error, indent=2, ensure_ascii=False), file=sys.stderr)
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
