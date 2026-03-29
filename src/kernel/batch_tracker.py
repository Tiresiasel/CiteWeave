"""Batch upload progress tracker.

Moved out of CLI so progress bookkeeping belongs to the kernel/application
layer rather than the terminal adapter.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any


_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"


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

        all_data.update(self.progress_data)
        with open(self.tracker_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

    def mark_file_completed(self, pdf_path: str, result_data: Dict[str, Any]) -> None:
        self.progress_data[pdf_path] = {
            "status": _STATUS_COMPLETED,
            "directory": self.directory,
            "paper_id": result_data.get("paper_id"),
            "processed_at": result_data.get("processing_time"),
            "stats": {
                "total_sentences": result_data.get("total_sentences", 0),
                "sentences_with_citations": result_data.get("sentences_with_citations", 0),
                "total_citations": result_data.get("total_citations", 0),
                "total_references": result_data.get("total_references", 0),
            },
        }
        self._save_progress()

    def mark_file_failed(self, pdf_path: str, error_msg: str) -> None:
        self.progress_data[pdf_path] = {
            "status": _STATUS_FAILED,
            "directory": self.directory,
            "error": error_msg,
        }
        self._save_progress()

    def is_file_completed(self, pdf_path: str) -> bool:
        return pdf_path in self.progress_data and self.progress_data[pdf_path]["status"] == _STATUS_COMPLETED

    def is_file_failed(self, pdf_path: str) -> bool:
        return pdf_path in self.progress_data and self.progress_data[pdf_path]["status"] == _STATUS_FAILED

    def get_pending_files(self, all_files, force_restart: bool = False):
        if force_restart:
            return all_files
        return [pdf_path for pdf_path in all_files if not self.is_file_completed(pdf_path)]

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

    def get_progress_summary(self):
        completed_entries = self.completed_entries()
        failed_entries = self.failed_entries()
        total = len(self.progress_data)
        completed = len(completed_entries)
        failed = len(failed_entries)
        return {
            "total_tracked": total,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total * 100) if total > 0 else 0,
            "completed_files": sorted(completed_entries.keys()),
            "failed_files": {
                path: entry.get("error", "")
                for path, entry in sorted(failed_entries.items())
            },
        }

    def clear_progress(self, directory: str | None = None) -> None:
        if directory:
            self.progress_data = {k: v for k, v in self.progress_data.items() if v.get("directory") != directory}
        else:
            self.progress_data = {}
        self._save_progress()
