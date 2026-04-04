"""Query history recorder for CiteWeave runtime telemetry.

This stays in the kernel layer so CLI/OpenClaw/other entrypoints can reuse a
single recorder without inventing their own file formats.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


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
