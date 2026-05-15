"""Batch upload progress tracker.

Moved out of CLI so progress bookkeeping belongs to the kernel/application
layer rather than the terminal adapter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_STATUS_DUPLICATE = "duplicate"
_READ_CHUNK_BYTES = 1024 * 1024


class BatchUploadTracker:
    """Tracks batch upload progress to enable resuming interrupted uploads."""

    def __init__(self, directory: str, tracker_file: str | None = None):
        self.directory = directory
        if tracker_file is None:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            self.tracker_file = data_dir / "batch_upload_tracker.json"
        else:
            self.tracker_file = Path(tracker_file)

        self.progress_data = self._load_progress()
        self._content_hashes: Dict[str, str] = {}
        self._last_deduplication_summary: Dict[str, Any] | None = None

    def _load_progress(self) -> Dict[str, Any]:
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {k: v for k, v in data.items() if v.get("directory") == self.directory}
            except (json.JSONDecodeError, IOError) as e:
                logging.warning("Could not load progress tracker: %s", e)
                return {}
        return {}

    def _save_progress(self) -> None:
        try:
            all_data = {}
            if self.tracker_file.exists():
                with open(self.tracker_file, "r", encoding="utf-8") as f:
                    all_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            all_data = {}

        # Replace this directory's tracker entries atomically within the JSON
        # document. A plain ``dict.update`` resurrects entries after
        # clear_progress(directory), because the old on-disk entries remain in
        # all_data. That defeats force-restart/resume semantics in exactly the
        # way progress trackers enjoy doing at 2 AM.
        all_data = {
            path: entry
            for path, entry in all_data.items()
            if entry.get("directory") != self.directory
        }
        all_data.update(self.progress_data)
        with open(self.tracker_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _sha256_file(pdf_path: str) -> str:
        digest = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(_READ_CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def compute_content_hashes(self, pdf_files: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Return SHA256 hashes for files, plus any read errors keyed by path."""
        hashes: Dict[str, str] = {}
        errors: Dict[str, str] = {}
        for pdf_path in sorted(str(path) for path in pdf_files):
            try:
                hashes[pdf_path] = self._sha256_file(pdf_path)
            except OSError as exc:
                errors[pdf_path] = str(exc)

        self._content_hashes = hashes
        return hashes, errors

    def file_hash_for_path(self, pdf_path: str) -> str | None:
        """Return the current dedupe hash for a path when known."""
        path = str(pdf_path)
        if path in self._content_hashes:
            return self._content_hashes[path]
        entry = self.progress_data.get(path) or {}
        file_hash = entry.get("file_hash")
        return str(file_hash) if file_hash else None

    def apply_content_deduplication(self, pdf_files: Iterable[str]) -> Dict[str, Any]:
        """Mark byte-identical PDFs as duplicates and keep one canonical path.

        The tracker is path-keyed for resume, but the ingest workload is content-
        keyed.  This method computes SHA256 for the current file set, chooses one
        canonical path per content hash, and records the other paths as duplicate
        aliases so resume skips them without deleting anything from Zotero.
        """
        started_at = time.time()
        all_files = sorted(str(path) for path in pdf_files)
        hashes, hash_errors = self.compute_content_hashes(all_files)
        hash_to_paths: Dict[str, list[str]] = defaultdict(list)
        for pdf_path, file_hash in hashes.items():
            hash_to_paths[file_hash].append(pdf_path)

        # If a path was already tracked but the file contents changed, forget the
        # old status.  Path-based resume without this check is just optimism with
        # a JSON file.
        content_changed_paths = []
        for pdf_path in all_files:
            entry = self.progress_data.get(pdf_path)
            if not entry:
                continue
            current_hash = hashes.get(pdf_path)
            previous_hash = entry.get("file_hash")
            if previous_hash and current_hash and previous_hash != current_hash:
                content_changed_paths.append(pdf_path)
                self.progress_data.pop(pdf_path, None)
            elif current_hash and entry.get("status") in {_STATUS_COMPLETED, _STATUS_DUPLICATE}:
                entry["file_hash"] = current_hash

        duplicate_groups = []
        duplicate_paths_marked = 0
        stale_duplicate_paths_cleared = 0
        duplicate_completed_paths = 0

        for file_hash, paths in sorted(hash_to_paths.items(), key=lambda item: item[0]):
            paths = sorted(paths)
            if len(paths) == 1:
                only_path = paths[0]
                if self.progress_data.get(only_path, {}).get("status") == _STATUS_DUPLICATE:
                    self.progress_data.pop(only_path, None)
                    stale_duplicate_paths_cleared += 1
                continue

            completed_candidates = [
                path for path in paths
                if self.progress_data.get(path, {}).get("status") == _STATUS_COMPLETED
            ]
            if completed_candidates:
                canonical_path = sorted(
                    completed_candidates,
                    key=lambda path: (
                        self.progress_data.get(path, {}).get("processed_at") is None,
                        self.progress_data.get(path, {}).get("processed_at") or float("inf"),
                        path,
                    ),
                )[0]
            else:
                canonical_path = paths[0]

            # A canonical path cannot itself be a duplicate alias.
            if self.progress_data.get(canonical_path, {}).get("status") == _STATUS_DUPLICATE:
                self.progress_data.pop(canonical_path, None)
                stale_duplicate_paths_cleared += 1

            canonical_entry = self.progress_data.get(canonical_path, {})
            canonical_paper_id = canonical_entry.get("paper_id")
            group_duplicate_paths = []
            for pdf_path in paths:
                if pdf_path == canonical_path:
                    continue

                previous_entry = self.progress_data.get(pdf_path, {})
                was_completed = previous_entry.get("status") == _STATUS_COMPLETED
                if was_completed:
                    duplicate_completed_paths += 1

                duplicate_entry = {
                    "status": _STATUS_DUPLICATE,
                    "directory": self.directory,
                    "file_hash": file_hash,
                    "duplicate_of": canonical_path,
                    "processed_at": previous_entry.get("processed_at") or time.time(),
                }
                paper_id = canonical_paper_id or previous_entry.get("paper_id")
                if paper_id:
                    duplicate_entry["paper_id"] = paper_id
                if was_completed:
                    duplicate_entry["duplicate_completed_previously"] = True
                    duplicate_entry["previous_duration_seconds"] = previous_entry.get("duration_seconds")
                    duplicate_entry["previous_stats"] = previous_entry.get("stats", {})

                self.progress_data[pdf_path] = duplicate_entry
                group_duplicate_paths.append(pdf_path)
                duplicate_paths_marked += 1

            duplicate_groups.append({
                "file_hash": file_hash,
                "canonical_path": canonical_path,
                "duplicate_count": len(group_duplicate_paths),
                "duplicate_paths": group_duplicate_paths,
            })

        redundant_duplicate_files = sum(max(0, len(paths) - 1) for paths in hash_to_paths.values())
        summary = {
            "total_pdf_files": len(all_files),
            "hashed_files": len(hashes),
            "hash_error_count": len(hash_errors),
            "hash_errors": hash_errors,
            "unique_content_files": len(hash_to_paths),
            "duplicate_groups": len([paths for paths in hash_to_paths.values() if len(paths) > 1]),
            "duplicate_files": redundant_duplicate_files,
            "duplicate_paths_marked": duplicate_paths_marked,
            "duplicate_completed_paths": duplicate_completed_paths,
            "stale_duplicate_paths_cleared": stale_duplicate_paths_cleared,
            "content_changed_paths": content_changed_paths,
            "elapsed_seconds": round(time.time() - started_at, 3),
            "examples": sorted(duplicate_groups, key=lambda item: (-item["duplicate_count"], item["canonical_path"]))[:10],
        }
        self._last_deduplication_summary = summary
        self._save_progress()
        return summary

    def mark_file_completed(self, pdf_path: str, result_data: Dict[str, Any]) -> None:
        processed_at = result_data.get("processed_at", result_data.get("processing_time"))
        duration_seconds = result_data.get("duration_seconds")
        file_hash = result_data.get("file_hash") or self.file_hash_for_path(pdf_path)
        entry = {
            "status": _STATUS_COMPLETED,
            "directory": self.directory,
            "paper_id": result_data.get("paper_id"),
            "processed_at": processed_at,
            "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
            "stats": {
                "total_sentences": result_data.get("total_sentences", 0),
                "sentences_with_citations": result_data.get("sentences_with_citations", 0),
                "total_citations": result_data.get("total_citations", 0),
                "total_references": result_data.get("total_references", 0),
            },
        }
        if file_hash:
            entry["file_hash"] = file_hash
        self.progress_data[str(pdf_path)] = entry
        self._save_progress()

    def mark_file_failed(self, pdf_path: str, error_msg: str) -> None:
        file_hash = self.file_hash_for_path(pdf_path)
        entry = {
            "status": _STATUS_FAILED,
            "directory": self.directory,
            "error": str(error_msg),
        }
        if file_hash:
            entry["file_hash"] = file_hash
        self.progress_data[str(pdf_path)] = entry
        self._save_progress()

    def mark_file_duplicate(self, pdf_path: str, canonical_path: str, file_hash: str | None = None, paper_id: str | None = None) -> None:
        entry = {
            "status": _STATUS_DUPLICATE,
            "directory": self.directory,
            "duplicate_of": str(canonical_path),
            "processed_at": time.time(),
        }
        if file_hash:
            entry["file_hash"] = file_hash
        if paper_id:
            entry["paper_id"] = paper_id
        self.progress_data[str(pdf_path)] = entry
        self._save_progress()

    def is_file_completed(self, pdf_path: str) -> bool:
        return self.progress_data.get(str(pdf_path), {}).get("status") == _STATUS_COMPLETED

    def is_file_failed(self, pdf_path: str) -> bool:
        return self.progress_data.get(str(pdf_path), {}).get("status") == _STATUS_FAILED

    def is_file_duplicate(self, pdf_path: str) -> bool:
        return self.progress_data.get(str(pdf_path), {}).get("status") == _STATUS_DUPLICATE

    def get_pending_files(self, all_files, force_restart: bool = False, retry_failed: bool = True):
        if force_restart:
            return [pdf_path for pdf_path in all_files if not self.is_file_duplicate(pdf_path)]
        return [
            pdf_path for pdf_path in all_files
            if not self.is_file_completed(pdf_path)
            and not self.is_file_duplicate(pdf_path)
            and (retry_failed or not self.is_file_failed(pdf_path))
        ]

    def completed_entries(self) -> Dict[str, Any]:
        return {
            path: entry
            for path, entry in self.progress_data.items()
            if entry.get("status") == _STATUS_COMPLETED
        }

    def failed_entries(self) -> Dict[str, Any]:
        return {
            path: entry
            for path, entry in self.progress_data.items()
            if entry.get("status") == _STATUS_FAILED
        }

    def duplicate_entries(self) -> Dict[str, Any]:
        return {
            path: entry
            for path, entry in self.progress_data.items()
            if entry.get("status") == _STATUS_DUPLICATE
        }

    def get_progress_summary(self):
        completed_entries = self.completed_entries()
        failed_entries = self.failed_entries()
        duplicate_entries = self.duplicate_entries()
        total = len(self.progress_data)
        completed = len(completed_entries)
        failed = len(failed_entries)
        duplicates = len(duplicate_entries)

        aggregate_stats = {
            "total_sentences": 0,
            "sentences_with_citations": 0,
            "total_citations": 0,
            "total_references": 0,
        }
        last_completed = None
        total_duration_seconds = 0.0
        completed_with_duration = 0
        for path, entry in completed_entries.items():
            stats = entry.get("stats", {})
            aggregate_stats["total_sentences"] += int(stats.get("total_sentences", 0) or 0)
            aggregate_stats["sentences_with_citations"] += int(stats.get("sentences_with_citations", 0) or 0)
            aggregate_stats["total_citations"] += int(stats.get("total_citations", 0) or 0)
            aggregate_stats["total_references"] += int(stats.get("total_references", 0) or 0)

            duration_seconds = entry.get("duration_seconds")
            if duration_seconds is not None:
                total_duration_seconds += float(duration_seconds)
                completed_with_duration += 1

            processed_at = entry.get("processed_at")
            if processed_at is None:
                continue
            if last_completed is None or processed_at > last_completed["processed_at"]:
                last_completed = {
                    "pdf_path": path,
                    "paper_id": entry.get("paper_id"),
                    "processed_at": processed_at,
                    "duration_seconds": duration_seconds,
                    "stats": stats,
                    "file_hash": entry.get("file_hash"),
                }

        error_counter = Counter(
            entry.get("error", "") or "unknown error"
            for entry in failed_entries.values()
        )
        attempted = completed + failed

        return {
            "total_tracked": total,
            "completed": completed,
            "failed": failed,
            "duplicate": duplicates,
            "success_rate": (completed / attempted * 100) if attempted > 0 else 0,
            "completed_files": sorted(completed_entries.keys()),
            "failed_files": {
                path: entry.get("error", "")
                for path, entry in sorted(failed_entries.items())
            },
            "duplicate_files": {
                path: entry.get("duplicate_of", "")
                for path, entry in sorted(duplicate_entries.items())
            },
            "aggregate_stats": aggregate_stats,
            "total_completed_duration_seconds": round(total_duration_seconds, 3),
            "average_completed_duration_seconds": round(total_duration_seconds / completed_with_duration, 3) if completed_with_duration else None,
            "last_completed": last_completed,
            "failure_reasons": [
                {"error": error, "count": count}
                for error, count in error_counter.most_common()
            ],
            "last_deduplication_summary": self._last_deduplication_summary,
        }

    def clear_progress(self, directory: str | None = None) -> None:
        if directory:
            self.progress_data = {k: v for k, v in self.progress_data.items() if v.get("directory") != directory}
        else:
            self.progress_data = {}
        self._save_progress()
