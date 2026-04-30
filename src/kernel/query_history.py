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


DATABASE_ROUTE_MAP = {
    "graph_db": "graph_analysis",
    "vector_db": "vector_search",
    "pdf_db": "pdf_analysis",
}

SATISFACTION_BUCKETS = {
    "satisfied",
    "neutral",
    "dissatisfied",
    "unrated",
}


def _normalize_satisfaction(value: Any) -> str:
    """Collapse raw satisfaction values into a stable diagnostic bucket."""
    if value is None:
        return "unrated"

    if isinstance(value, bool):
        return "satisfied" if value else "dissatisfied"

    if isinstance(value, (int, float)):
        if value >= 4:
            return "satisfied"
        if value <= 2:
            return "dissatisfied"
        return "neutral"

    if not isinstance(value, str):
        return "unrated"

    normalized = value.strip().casefold()
    if not normalized:
        return "unrated"

    if normalized in {"satisfied", "satisfaction_high", "positive", "yes", "true", "thumbs_up", "up", "good", "helpful"}:
        return "satisfied"
    if normalized in {"dissatisfied", "unsatisfied", "negative", "no", "false", "thumbs_down", "down", "bad", "unhelpful"}:
        return "dissatisfied"
    if normalized in {"neutral", "mixed", "partial", "unclear", "ok", "average"}:
        return "neutral"
    if normalized in {"unrated", "unknown", "none", "n/a", "na", "null"}:
        return "unrated"

    return normalized if normalized in SATISFACTION_BUCKETS else "unrated"


def _infer_query_plan_routes(entry: Dict[str, Any]) -> List[str]:
    """Return planned routes for an entry, inferring them from databases when needed."""
    explicit_routes = entry.get("query_plan_routes") or []
    routes: List[str] = []
    for route in explicit_routes:
        if isinstance(route, str) and route and route not in routes:
            routes.append(route)

    for database in entry.get("query_plan_databases") or []:
        if not isinstance(database, str):
            continue
        route = DATABASE_ROUTE_MAP.get(database)
        if route and route not in routes:
            routes.append(route)

    return routes


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

    def _apply_filters(
        self,
        entries: List[Dict[str, Any]],
        status: Optional[str] = None,
        source: Optional[str] = None,
        confirmation: Optional[str] = None,
        satisfaction: Optional[str] = None,
        since_hours: Optional[float] = None,
        contains: Optional[str] = None,
        question_contains: Optional[str] = None,
        error_contains: Optional[str] = None,
        response_contains: Optional[str] = None,
        planned_database: Optional[str] = None,
        planned_method: Optional[str] = None,
        planned_route: Optional[str] = None,
        min_duration_ms: Optional[int] = None,
        max_duration_ms: Optional[int] = None,
        min_response_chars: Optional[int] = None,
        max_response_chars: Optional[int] = None,
        now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        filtered = entries
        if status and status != "all":
            filtered = [entry for entry in filtered if entry.get("status") == status]
        if source and source != "all":
            filtered = [entry for entry in filtered if (entry.get("source") or "unknown") == source]
        if confirmation and confirmation != "all":
            filtered = [entry for entry in filtered if (entry.get("confirmation") or "unspecified") == confirmation]
        if satisfaction and satisfaction != "all":
            filtered = [entry for entry in filtered if _normalize_satisfaction(entry.get("satisfaction")) == satisfaction]
        if since_hours is not None and since_hours >= 0:
            cutoff = (time.time() if now is None else now) - (since_hours * 3600)
            filtered = [
                entry for entry in filtered
                if isinstance(entry.get("timestamp"), (int, float)) and entry["timestamp"] >= cutoff
            ]
        if contains:
            needle = contains.casefold()
            filtered = [
                entry for entry in filtered
                if needle in (entry.get("question") or "").casefold()
                or needle in (entry.get("error") or "").casefold()
                or needle in (entry.get("response_preview") or "").casefold()
                or needle in (entry.get("raw_line") or "").casefold()
            ]
        if question_contains:
            question_needle = question_contains.casefold()
            filtered = [
                entry for entry in filtered
                if question_needle in (entry.get("question") or "").casefold()
            ]
        if error_contains:
            error_needle = error_contains.casefold()
            filtered = [
                entry for entry in filtered
                if error_needle in (entry.get("error") or "").casefold()
            ]
        if response_contains:
            response_needle = response_contains.casefold()
            filtered = [
                entry for entry in filtered
                if response_needle in (entry.get("response_preview") or "").casefold()
            ]
        if planned_database and planned_database != "all":
            database_needle = planned_database.casefold()
            filtered = [
                entry for entry in filtered
                if any(
                    isinstance(database, str) and database.casefold() == database_needle
                    for database in (entry.get("query_plan_databases") or [])
                )
            ]
        if planned_method and planned_method != "all":
            method_needle = planned_method.casefold()
            filtered = [
                entry for entry in filtered
                if any(
                    isinstance(method, str) and method.casefold() == method_needle
                    for method in (entry.get("query_plan_methods") or [])
                )
            ]
        if planned_route and planned_route != "all":
            route_needle = planned_route.casefold()
            filtered = [
                entry for entry in filtered
                if any(route.casefold() == route_needle for route in _infer_query_plan_routes(entry))
            ]
        if min_duration_ms is not None and min_duration_ms >= 0:
            filtered = [
                entry for entry in filtered
                if isinstance(entry.get("duration_ms"), int) and entry["duration_ms"] >= min_duration_ms
            ]
        if max_duration_ms is not None and max_duration_ms >= 0:
            filtered = [
                entry for entry in filtered
                if isinstance(entry.get("duration_ms"), int) and entry["duration_ms"] <= max_duration_ms
            ]
        if min_response_chars is not None and min_response_chars >= 0:
            filtered = [
                entry for entry in filtered
                if isinstance(entry.get("response_chars"), int) and entry["response_chars"] >= min_response_chars
            ]
        if max_response_chars is not None and max_response_chars >= 0:
            filtered = [
                entry for entry in filtered
                if isinstance(entry.get("response_chars"), int) and entry["response_chars"] <= max_response_chars
            ]
        return filtered

    def recent_entries(
        self,
        limit: int = 10,
        status: Optional[str] = None,
        source: Optional[str] = None,
        confirmation: Optional[str] = None,
        satisfaction: Optional[str] = None,
        since_hours: Optional[float] = None,
        contains: Optional[str] = None,
        question_contains: Optional[str] = None,
        error_contains: Optional[str] = None,
        response_contains: Optional[str] = None,
        planned_database: Optional[str] = None,
        planned_method: Optional[str] = None,
        planned_route: Optional[str] = None,
        min_duration_ms: Optional[int] = None,
        max_duration_ms: Optional[int] = None,
        min_response_chars: Optional[int] = None,
        max_response_chars: Optional[int] = None,
        now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []

        entries = self._apply_filters(
            self.load_entries(),
            status=status,
            source=source,
            confirmation=confirmation,
            satisfaction=satisfaction,
            since_hours=since_hours,
            contains=contains,
            question_contains=question_contains,
            error_contains=error_contains,
            response_contains=response_contains,
            planned_database=planned_database,
            planned_method=planned_method,
            planned_route=planned_route,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            min_response_chars=min_response_chars,
            max_response_chars=max_response_chars,
            now=now,
        )
        return list(reversed(entries[-limit:]))

    def summary(
        self,
        limit: int = 10,
        status: Optional[str] = None,
        source: Optional[str] = None,
        confirmation: Optional[str] = None,
        satisfaction: Optional[str] = None,
        since_hours: Optional[float] = None,
        contains: Optional[str] = None,
        question_contains: Optional[str] = None,
        error_contains: Optional[str] = None,
        response_contains: Optional[str] = None,
        planned_database: Optional[str] = None,
        planned_method: Optional[str] = None,
        planned_route: Optional[str] = None,
        min_duration_ms: Optional[int] = None,
        max_duration_ms: Optional[int] = None,
        min_response_chars: Optional[int] = None,
        max_response_chars: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        status_filter = status or "all"
        source_filter = source or "all"
        confirmation_filter = confirmation or "all"
        satisfaction_filter = satisfaction or "all"
        contains_filter = contains or ""
        question_contains_filter = question_contains or ""
        error_contains_filter = error_contains or ""
        response_contains_filter = response_contains or ""
        planned_database_filter = planned_database or "all"
        planned_method_filter = planned_method or "all"
        planned_route_filter = planned_route or "all"
        min_duration_filter = min_duration_ms if isinstance(min_duration_ms, int) and min_duration_ms >= 0 else None
        max_duration_filter = max_duration_ms if isinstance(max_duration_ms, int) and max_duration_ms >= 0 else None
        min_response_filter = min_response_chars if isinstance(min_response_chars, int) and min_response_chars >= 0 else None
        max_response_filter = max_response_chars if isinstance(max_response_chars, int) and max_response_chars >= 0 else None
        matching_entries = self._apply_filters(
            self.load_entries(),
            status=status_filter,
            source=source_filter,
            confirmation=confirmation_filter,
            satisfaction=satisfaction_filter,
            since_hours=since_hours,
            contains=contains_filter,
            question_contains=question_contains_filter,
            error_contains=error_contains_filter,
            response_contains=response_contains_filter,
            planned_database=planned_database_filter,
            planned_method=planned_method_filter,
            planned_route=planned_route_filter,
            min_duration_ms=min_duration_filter,
            max_duration_ms=max_duration_filter,
            min_response_chars=min_response_filter,
            max_response_chars=max_response_filter,
            now=now,
        )
        recent = self.recent_entries(
            limit=limit,
            status=status_filter,
            source=source_filter,
            confirmation=confirmation_filter,
            satisfaction=satisfaction_filter,
            since_hours=since_hours,
            contains=contains_filter,
            question_contains=question_contains_filter,
            error_contains=error_contains_filter,
            response_contains=response_contains_filter,
            planned_database=planned_database_filter,
            planned_method=planned_method_filter,
            planned_route=planned_route_filter,
            min_duration_ms=min_duration_filter,
            max_duration_ms=max_duration_filter,
            min_response_chars=min_response_filter,
            max_response_chars=max_response_filter,
            now=now,
        )
        considered = [entry for entry in recent if entry.get("status") != "corrupt"]
        matching_considered = [entry for entry in matching_entries if entry.get("status") != "corrupt"]
        success_count = sum(1 for entry in considered if entry.get("status") == "success")
        error_count = sum(1 for entry in considered if entry.get("status") == "error")
        matching_success_count = sum(1 for entry in matching_considered if entry.get("status") == "success")
        matching_error_count = sum(1 for entry in matching_considered if entry.get("status") == "error")
        success_rate = round(success_count / len(considered), 4) if considered else None
        error_rate = round(error_count / len(considered), 4) if considered else None
        matching_success_rate = round(matching_success_count / len(matching_considered), 4) if matching_considered else None
        matching_error_rate = round(matching_error_count / len(matching_considered), 4) if matching_considered else None
        durations = [entry.get("duration_ms") for entry in considered if isinstance(entry.get("duration_ms"), int)]
        matching_durations = [entry.get("duration_ms") for entry in matching_considered if isinstance(entry.get("duration_ms"), int)]
        response_sizes = [entry.get("response_chars") for entry in considered if isinstance(entry.get("response_chars"), int)]
        matching_response_sizes = [entry.get("response_chars") for entry in matching_considered if isinstance(entry.get("response_chars"), int)]
        latest = considered[0] if considered else None
        latest_error = next((entry for entry in recent if entry.get("status") == "error"), None)
        def _source_counter(rows: List[Dict[str, Any]]) -> Counter:
            return Counter((entry.get("source") or "unknown") for entry in rows)

        def _confirmation_counter(rows: List[Dict[str, Any]]) -> Counter:
            return Counter((entry.get("confirmation") or "unspecified") for entry in rows)

        def _satisfaction_counter(rows: List[Dict[str, Any]]) -> Counter:
            return Counter(_normalize_satisfaction(entry.get("satisfaction")) for entry in rows)

        def _error_counter(rows: List[Dict[str, Any]]) -> Counter:
            return Counter(
                error
                for entry in rows
                for error in [entry.get("error")]
                if isinstance(error, str) and error
            )

        def _query_plan_database_counter(rows: List[Dict[str, Any]]) -> Counter:
            return Counter(
                database
                for entry in rows
                for database in (entry.get("query_plan_databases") or [])
                if isinstance(database, str) and database
            )

        def _query_plan_method_counter(rows: List[Dict[str, Any]]) -> Counter:
            return Counter(
                method
                for entry in rows
                for method in (entry.get("query_plan_methods") or [])
                if isinstance(method, str) and method
            )

        def _query_plan_route_counter(rows: List[Dict[str, Any]]) -> Counter:
            return Counter(
                route
                for entry in rows
                for route in _infer_query_plan_routes(entry)
            )

        source_counter = _source_counter(considered)
        confirmation_counter = _confirmation_counter(considered)
        satisfaction_counter = _satisfaction_counter(considered)
        error_counter = _error_counter(considered)
        query_plan_database_counter = _query_plan_database_counter(considered)
        query_plan_method_counter = _query_plan_method_counter(considered)
        query_plan_route_counter = _query_plan_route_counter(considered)
        matching_source_counter = _source_counter(matching_considered)
        matching_confirmation_counter = _confirmation_counter(matching_considered)
        matching_satisfaction_counter = _satisfaction_counter(matching_considered)
        matching_error_counter = _error_counter(matching_considered)
        matching_query_plan_database_counter = _query_plan_database_counter(matching_considered)
        matching_query_plan_method_counter = _query_plan_method_counter(matching_considered)
        matching_query_plan_route_counter = _query_plan_route_counter(matching_considered)

        return {
            "log_file": str(self.log_file),
            "requested_limit": limit,
            "status_filter": status_filter,
            "source_filter": source_filter,
            "confirmation_filter": confirmation_filter,
            "satisfaction_filter": satisfaction_filter,
            "contains_filter": contains_filter,
            "question_contains_filter": question_contains_filter,
            "error_contains_filter": error_contains_filter,
            "response_contains_filter": response_contains_filter,
            "planned_database_filter": planned_database_filter,
            "planned_method_filter": planned_method_filter,
            "planned_route_filter": planned_route_filter,
            "min_duration_ms_filter": min_duration_filter,
            "max_duration_ms_filter": max_duration_filter,
            "min_response_chars_filter": min_response_filter,
            "max_response_chars_filter": max_response_filter,
            "entries_returned": len(recent),
            "matching_entries_total": len(matching_entries),
            "entries_considered": len(considered),
            "matching_entries_considered": len(matching_considered),
            "since_hours": since_hours,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "matching_success_count": matching_success_count,
            "matching_error_count": matching_error_count,
            "matching_success_rate": matching_success_rate,
            "matching_error_rate": matching_error_rate,
            "corrupt_count": len(recent) - len(considered),
            "matching_corrupt_count": len(matching_entries) - len(matching_considered),
            "average_duration_ms": round(sum(durations) / len(durations), 2) if durations else None,
            "max_duration_ms": max(durations) if durations else None,
            "matching_average_duration_ms": round(sum(matching_durations) / len(matching_durations), 2) if matching_durations else None,
            "matching_max_duration_ms": max(matching_durations) if matching_durations else None,
            "average_response_chars": round(sum(response_sizes) / len(response_sizes), 2) if response_sizes else None,
            "max_response_chars": max(response_sizes) if response_sizes else None,
            "matching_average_response_chars": round(sum(matching_response_sizes) / len(matching_response_sizes), 2) if matching_response_sizes else None,
            "matching_max_response_chars": max(matching_response_sizes) if matching_response_sizes else None,
            "latest_status": latest.get("status") if latest else None,
            "latest_question": latest.get("question") if latest else None,
            "latest_source": latest.get("source") if latest else None,
            "latest_error": latest_error.get("error") if latest_error else None,
            "latest_response_preview": latest.get("response_preview") if latest else None,
            "query_plan_database_breakdown": [
                {"database": database, "count": count}
                for database, count in query_plan_database_counter.most_common()
            ],
            "query_plan_method_breakdown": [
                {"method": method, "count": count}
                for method, count in query_plan_method_counter.most_common()
            ],
            "query_plan_route_breakdown": [
                {"route": route, "count": count}
                for route, count in query_plan_route_counter.most_common()
            ],
            "source_breakdown": [
                {"source": source_name, "count": count}
                for source_name, count in source_counter.most_common()
            ],
            "confirmation_breakdown": [
                {"confirmation": confirmation, "count": count}
                for confirmation, count in confirmation_counter.most_common()
            ],
            "satisfaction_breakdown": [
                {"satisfaction": satisfaction_name, "count": count}
                for satisfaction_name, count in satisfaction_counter.most_common()
            ],
            "error_breakdown": [
                {"error": error_name, "count": count}
                for error_name, count in error_counter.most_common()
            ],
            "matching_query_plan_database_breakdown": [
                {"database": database, "count": count}
                for database, count in matching_query_plan_database_counter.most_common()
            ],
            "matching_query_plan_method_breakdown": [
                {"method": method, "count": count}
                for method, count in matching_query_plan_method_counter.most_common()
            ],
            "matching_query_plan_route_breakdown": [
                {"route": route, "count": count}
                for route, count in matching_query_plan_route_counter.most_common()
            ],
            "matching_source_breakdown": [
                {"source": source_name, "count": count}
                for source_name, count in matching_source_counter.most_common()
            ],
            "matching_confirmation_breakdown": [
                {"confirmation": confirmation, "count": count}
                for confirmation, count in matching_confirmation_counter.most_common()
            ],
            "matching_satisfaction_breakdown": [
                {"satisfaction": satisfaction_name, "count": count}
                for satisfaction_name, count in matching_satisfaction_counter.most_common()
            ],
            "matching_error_breakdown": [
                {"error": error_name, "count": count}
                for error_name, count in matching_error_counter.most_common()
            ],
            "entries": recent,
        }
