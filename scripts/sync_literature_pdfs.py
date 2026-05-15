#!/usr/bin/env python3
"""Recursively sync PDFs from a reference manager or plain PDF folder.

This is the agent-facing generic sync entrypoint. It accepts Zotero, Mendeley,
EndNote, and generic PDF directories while preserving CiteWeave's resumable
batch-upload behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_reference_manager(value: str) -> str:
    normalized = (value or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pdf": "pdf_folder",
        "pdfs": "pdf_folder",
        "folder": "pdf_folder",
        "generic": "pdf_folder",
    }
    return aliases.get(normalized, normalized)


def default_candidates(reference_manager: str) -> List[Path]:
    home = Path.home()
    candidates: List[Path] = []
    if reference_manager in {"auto", "zotero"}:
        candidates.extend(
            [
                home / "Zotero",
                home / "Documents" / "Zotero",
            ]
        )
        profiles_root = home / "Library" / "Application Support" / "Zotero" / "Profiles"
        if profiles_root.exists():
            candidates.extend(sorted(profiles_root.glob("*/zotero")))
    if reference_manager in {"auto", "mendeley"}:
        candidates.extend(
            [
                home / "Documents" / "Mendeley Desktop",
                home / "Library" / "Application Support" / "Mendeley Desktop",
                home / "Library" / "Application Support" / "Mendeley Reference Manager",
            ]
        )
    if reference_manager in {"auto", "endnote"}:
        candidates.extend(
            [
                home / "Documents" / "EndNote",
                home / "Documents" / "EndNote Libraries",
                home / "Library" / "Application Support" / "EndNote",
            ]
        )
    return candidates


def discover_pdfs(source_dir: Path) -> List[Path]:
    return sorted(path for path in source_dir.rglob("*.pdf") if path.is_file())


def resolve_source(source: Optional[str], reference_manager: str) -> Path:
    raw_source = (
        source
        or os.environ.get("CITEWEAVE_LITERATURE_SOURCE_DIR", "")
        or (os.environ.get("CITEWEAVE_ZOTERO_LIBRARY_DIR", "") if reference_manager in {"auto", "zotero"} else "")
    )

    if raw_source.strip():
        candidate = Path(raw_source).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Literature source does not exist: {candidate}")
        if candidate.is_file():
            if candidate.suffix.lower() != ".pdf":
                raise ValueError(f"Single-file source is not a PDF: {candidate}")
            return candidate
        if not candidate.is_dir():
            raise NotADirectoryError(f"Literature source is not a directory: {candidate}")
        storage_child = candidate / "storage"
        if reference_manager == "zotero" and storage_child.is_dir():
            return storage_child.resolve()
        return candidate

    for candidate in default_candidates(reference_manager):
        if not candidate.exists() or not candidate.is_dir():
            continue
        resolved = (candidate / "storage").resolve() if reference_manager == "zotero" and (candidate / "storage").is_dir() else candidate.resolve()
        if discover_pdfs(resolved):
            return resolved

    raise ValueError(
        "No default literature source was found. Ask the user for Custom Location or PDF Folder."
    )


def citeweave_command(
    root: Path,
    source_path: Path,
    resume: bool,
    force_restart: bool,
    clear_progress: bool,
    processors: Optional[int],
    sequential: bool,
    skip_failed: bool,
    file_timeout_seconds: Optional[int],
) -> List[str]:
    citeweave_bin = root / ".venv" / "bin" / "citeweave"
    if source_path.is_file():
        return [str(citeweave_bin), "upload", str(source_path)] if citeweave_bin.exists() else [sys.executable, "-m", "src.core.cli", "upload", str(source_path)]

    if citeweave_bin.exists():
        command = [str(citeweave_bin), "batch-upload", str(source_path)]
    else:
        command = [sys.executable, "-m", "src.core.cli", "batch-upload", str(source_path)]

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


def build_report(
    source_arg: Optional[str],
    reference_manager: str,
    source_path: Path,
    pdfs: List[Path],
    command: List[str],
    dry_run: bool,
) -> Dict[str, Any]:
    return {
        "status": "dry_run" if dry_run else "ready",
        "reference_manager": reference_manager,
        "source_arg": source_arg or os.environ.get("CITEWEAVE_LITERATURE_SOURCE_DIR", ""),
        "resolved_pdf_source": str(source_path),
        "source_type": "single_pdf" if source_path.is_file() else "directory",
        "pdf_count": len(pdfs),
        "sample_pdfs": [str(path) for path in pdfs[:10]],
        "command": command,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync literature PDFs into CiteWeave using resumable batch upload.")
    parser.add_argument("--source", help="Reference manager directory, PDF directory, or single PDF.")
    parser.add_argument(
        "--reference-manager",
        default=os.environ.get("CITEWEAVE_REFERENCE_MANAGER", "auto"),
        help="auto, zotero, mendeley, endnote, or pdf_folder.",
    )
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
    reference_manager = normalize_reference_manager(args.reference_manager)

    try:
        source_path = resolve_source(args.source, reference_manager)
        pdfs = [source_path] if source_path.is_file() else discover_pdfs(source_path)
        command = citeweave_command(
            root,
            source_path,
            resume=not args.no_resume,
            force_restart=args.force_restart,
            clear_progress=args.clear_progress,
            processors=args.processors,
            sequential=args.sequential,
            skip_failed=args.skip_failed,
            file_timeout_seconds=args.file_timeout_seconds,
        )
        report = build_report(args.source, reference_manager, source_path, pdfs, command, args.dry_run)

        if args.dry_run:
            if args.json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print(f"Resolved PDF source: {source_path}")
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
                print(f"No PDF files found under {source_path}; nothing to ingest.")
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
