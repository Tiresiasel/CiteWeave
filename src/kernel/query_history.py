"""Query history recorder for CiteWeave runtime telemetry.

This stays in the kernel layer so CLI/OpenClaw/other entrypoints can reuse a
single recorder without inventing their own file formats.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


class QueryHistoryRecorder:
    """Append query interaction records to a local JSONL log file."""

    def __init__(self, log_file: str | None = None):
        if log_file is None:
            log_file = os.environ.get("CITEWEAVE_QUERY_HISTORY_FILE", "data/query_history.jsonl")
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: Dict[str, Any]) -> None:
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_entries(self) -> List[Dict[str, Any]]:
        if not self.log_file.exists():
            return []

        entries: List[Dict[str, Any]] = []
        for line in self.log_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                entries.append({
                    "status": "corrupt",
                    "raw_line": line,
                })
        return entries

    def recent_entries(
        self,
        limit: int = 10,
        status: Optional[str] = None,
        source: Optional[str] = None,
        confirmation: Optional[str] = None,
        since_hours: Optional[float] = None,
        now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []

        entries = self.load_entries()
        if status and status != "all":
            entries = [entry for entry in entries if entry.get("status") == status]
        if source and source != "all":
            entries = [entry for entry in entries if (entry.get("source") or "unknown") == source]
        if confirmation and confirmation != "all":
            entries = [entry for entry in entries if (entry.get("confirmation") or "unspecified") == confirmation]
        if since_hours is not None and since_hours >= 0:
            cutoff = (time.time() if now is None else now) - (since_hours * 3600)
            entries = [
                entry for entry in entries
                if isinstance(entry.get("timestamp"), (int, float)) and entry["timestamp"] >= cutoff
            ]
        return list(reversed(entries[-limit:]))

    def summary(
        self,
        limit: int = 10,
        status: Optional[str] = None,
        source: Optional[str] = None,
        confirmation: Optional[str] = None,
        since_hours: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        status_filter = status or "all"
        source_filter = source or "all"
        confirmation_filter = confirmation or "all"
        recent = self.recent_entries(
            limit=limit,
            status=status_filter,
            source=source_filter,
            confirmation=confirmation_filter,
            since_hours=since_hours,
            now=now,
        )
        considered = [entry for entry in recent if entry.get("status") != "corrupt"]
        success_count = sum(1 for entry in considered if entry.get("status") == "success")
        error_count = sum(1 for entry in considered if entry.get("status") == "error")
        durations = [entry.get("duration_ms") for entry in considered if isinstance(entry.get("duration_ms"), int)]
        latest = considered[0] if considered else None
        latest_error = next((entry for entry in recent if entry.get("status") == "error"), None)
        source_counter = Counter((entry.get("source") or "unknown") for entry in considered)
        confirmation_counter = Counter((entry.get("confirmation") or "unspecified") for entry in considered)

        return {
            "log_file": str(self.log_file),
            "requested_limit": limit,
            "status_filter": status_filter,
            "source_filter": source_filter,
            "confirmation_filter": confirmation_filter,
            "entries_returned": len(recent),
            "entries_considered": len(considered),
            "since_hours": since_hours,
            "success_count": success_count,
            "error_count": error_count,
            "corrupt_count": len(recent) - len(considered),
            "average_duration_ms": round(sum(durations) / len(durations), 2) if durations else None,
            "max_duration_ms": max(durations) if durations else None,
            "latest_status": latest.get("status") if latest else None,
            "latest_question": latest.get("question") if latest else None,
            "latest_source": latest.get("source") if latest else None,
            "latest_error": latest_error.get("error") if latest_error else None,
            "source_breakdown": [
                {"source": source_name, "count": count}
                for source_name, count in source_counter.most_common()
            ],
            "confirmation_breakdown": [
                {"confirmation": confirmation, "count": count}
                for confirmation, count in confirmation_counter.most_common()
            ],
            "entries": recent,
        }
