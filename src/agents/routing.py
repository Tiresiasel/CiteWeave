"""Routing constants and helpers for the multi-agent query workflow.

This module centralizes route names and normalization logic so route updates
can be made in one place.

It also supports optional OpenClaw addon overrides via environment variables:
`CITEWEAVE_ROUTE_PRIORITY_OVERRIDES` and `CITEWEAVE_ROUTE_ALIASES`.

Expected formats:
    CITEWEAVE_ROUTE_PRIORITY_OVERRIDES = {"priority_key": "route_name"}
    CITEWEAVE_ROUTE_ALIASES = {"alias_name": "route_name"}

Only known routes are accepted to keep routing safe.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

ROUTE_GRAPH_ANALYSIS = "graph_analysis"
ROUTE_VECTOR_SEARCH = "vector_search"
ROUTE_PDF_ANALYSIS = "pdf_analysis"
ROUTE_AUTHOR_COLLECTION = "author_collection"

VALID_ROUTES = (
    ROUTE_GRAPH_ANALYSIS,
    ROUTE_VECTOR_SEARCH,
    ROUTE_PDF_ANALYSIS,
    ROUTE_AUTHOR_COLLECTION,
)

DEFAULT_ROUTE = ROUTE_VECTOR_SEARCH

PRIORITY_TO_ROUTE = {
    "embedding_vector": ROUTE_VECTOR_SEARCH,
    "graph_database": ROUTE_GRAPH_ANALYSIS,
    "pdf_content": ROUTE_PDF_ANALYSIS,
    "author_index": ROUTE_AUTHOR_COLLECTION,
}

BASE_ROUTE_ALIASES = {
    ROUTE_GRAPH_ANALYSIS: ROUTE_GRAPH_ANALYSIS,
    "graph": ROUTE_GRAPH_ANALYSIS,
    ROUTE_VECTOR_SEARCH: ROUTE_VECTOR_SEARCH,
    "vector": ROUTE_VECTOR_SEARCH,
    ROUTE_PDF_ANALYSIS: ROUTE_PDF_ANALYSIS,
    "pdf": ROUTE_PDF_ANALYSIS,
    ROUTE_AUTHOR_COLLECTION: ROUTE_AUTHOR_COLLECTION,
    "author": ROUTE_AUTHOR_COLLECTION,
}


def _normalize_key(value: str) -> str:
    """Normalize addon/user-provided keys for stable matching."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_from_alias_map(route_value: str | None, alias_map: Dict[str, str]) -> str | None:
    """Resolve route name/alias from a provided alias map."""
    if not route_value or not isinstance(route_value, str):
        return None
    normalized = _normalize_key(route_value)
    return alias_map.get(normalized)


def _invalid_override(reason: str, alias_or_key: Any, route_name: Any) -> Dict[str, Any]:
    """Create a stable diagnostic payload for ignored addon overrides."""
    entry: Dict[str, Any] = {"reason": reason}
    if alias_or_key is not None:
        entry["key"] = alias_or_key
    if route_name is not None:
        entry["route"] = route_name
    return entry


def _parse_route_alias_overrides_with_diagnostics(raw_value: str | None) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Parse and validate addon-provided route alias overrides.

    Safety constraints:
    - payload must be a dict of string -> string
    - target routes must be canonical known routes
    - canonical route names cannot be remapped to different routes

    Returns a tuple of:
    - accepted alias overrides
    - ignored entries with reasons for addon diagnostics
    """
    if not raw_value:
        return {}, []

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}, [_invalid_override("invalid_json", None, None)]

    if not isinstance(parsed, dict):
        return {}, [_invalid_override("non_object_payload", None, None)]

    overrides: Dict[str, str] = {}
    ignored: List[Dict[str, Any]] = []

    for alias_name, route_name in parsed.items():
        if not isinstance(alias_name, str) or not isinstance(route_name, str):
            ignored.append(_invalid_override("non_string_entry", alias_name, route_name))
            continue

        normalized_alias = _normalize_key(alias_name)
        normalized_route = _resolve_from_alias_map(route_name, BASE_ROUTE_ALIASES)
        if not normalized_route:
            ignored.append(_invalid_override("unknown_route", alias_name, route_name))
            continue

        # Keep canonical route keys stable for safe behavior.
        if normalized_alias in VALID_ROUTES and normalized_alias != normalized_route:
            ignored.append(_invalid_override("canonical_route_locked", alias_name, route_name))
            continue

        overrides[normalized_alias] = normalized_route

    return overrides, ignored


def _parse_route_alias_overrides(raw_value: str | None) -> Dict[str, str]:
    """Backward-compatible alias override parser."""
    overrides, _ = _parse_route_alias_overrides_with_diagnostics(raw_value)
    return overrides


def _parse_route_priority_overrides_with_diagnostics(
    raw_value: str | None,
    alias_map: Dict[str, str],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Parse and validate route priority overrides from environment.

    Invalid JSON, non-dict payloads, or mappings to unknown routes are ignored.

    Returns a tuple of:
    - accepted priority overrides
    - ignored entries with reasons for addon diagnostics
    """
    if not raw_value:
        return {}, []

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}, [_invalid_override("invalid_json", None, None)]

    if not isinstance(parsed, dict):
        return {}, [_invalid_override("non_object_payload", None, None)]

    overrides: Dict[str, str] = {}
    ignored: List[Dict[str, Any]] = []

    for priority_key, route_name in parsed.items():
        if not isinstance(priority_key, str) or not isinstance(route_name, str):
            ignored.append(_invalid_override("non_string_entry", priority_key, route_name))
            continue

        normalized_route = _resolve_from_alias_map(route_name, alias_map)
        if normalized_route:
            overrides[_normalize_key(priority_key)] = normalized_route
        else:
            ignored.append(_invalid_override("unknown_route", priority_key, route_name))

    return overrides, ignored


def _parse_route_priority_overrides(raw_value: str | None, alias_map: Dict[str, str]) -> Dict[str, str]:
    """Backward-compatible priority override parser."""
    overrides, _ = _parse_route_priority_overrides_with_diagnostics(raw_value, alias_map)
    return overrides


@lru_cache(maxsize=8)
def _build_route_registry(alias_override_raw: str | None, priority_override_raw: str | None) -> Dict[str, Any]:
    """Build a cached route registry for the current addon override snapshot."""
    alias_overrides, ignored_alias_overrides = _parse_route_alias_overrides_with_diagnostics(alias_override_raw)

    alias_map = dict(BASE_ROUTE_ALIASES)
    alias_map.update(alias_overrides)

    priority_override_map, ignored_priority_overrides = _parse_route_priority_overrides_with_diagnostics(
        priority_override_raw,
        alias_map,
    )
    priority_map = dict(PRIORITY_TO_ROUTE)
    priority_map.update(priority_override_map)

    return {
        "valid_routes": list(VALID_ROUTES),
        "default_route": DEFAULT_ROUTE,
        "aliases": alias_map,
        "base_aliases": dict(BASE_ROUTE_ALIASES),
        "alias_overrides": alias_overrides,
        "ignored_alias_overrides": ignored_alias_overrides,
        "priority_map": priority_map,
        "priority_overrides": {
            key: value
            for key, value in priority_map.items()
            if PRIORITY_TO_ROUTE.get(key) != value
        },
        "ignored_priority_overrides": ignored_priority_overrides,
    }


def _current_route_registry() -> Dict[str, Any]:
    """Return the active route registry for the current environment."""
    return _build_route_registry(
        os.getenv("CITEWEAVE_ROUTE_ALIASES"),
        os.getenv("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"),
    )


def active_route_configuration() -> Dict[str, Any]:
    """Expose the active routing registry for addon diagnostics and tests.

    The returned snapshot includes:
    - canonical valid routes
    - the safe default route
    - full alias map after applying safe overrides
    - only the addon-provided alias/priority overrides that took effect
    - ignored addon override entries with stable diagnostic reasons
    - the final priority-to-route mapping
    """
    registry = _current_route_registry()
    return {
        "valid_routes": list(registry["valid_routes"]),
        "default_route": registry["default_route"],
        "aliases": dict(registry["aliases"]),
        "alias_overrides": dict(registry["alias_overrides"]),
        "ignored_alias_overrides": list(registry["ignored_alias_overrides"]),
        "priority_map": dict(registry["priority_map"]),
        "priority_overrides": dict(registry["priority_overrides"]),
        "ignored_priority_overrides": list(registry["ignored_priority_overrides"]),
    }


def resolve_route(route_value: str | None) -> str | None:
    """Resolve route name/alias to a canonical route, or None when invalid."""
    return _resolve_from_alias_map(route_value, _current_route_registry()["aliases"])


def normalize_routes(routes: Iterable[str]) -> List[str]:
    """Keep only valid routes, preserving order and removing duplicates."""
    normalized: List[str] = []
    seen = set()
    for route in routes:
        canonical_route = resolve_route(route)
        if canonical_route and canonical_route not in seen:
            normalized.append(canonical_route)
            seen.add(canonical_route)
    return normalized


def route_for_priority(priority_value: str) -> str:
    """Map retrieval priority value to route with a safe default.

    Supports flexible key formats (e.g., `graph-database`) and optional addon
    overrides through `CITEWEAVE_ROUTE_PRIORITY_OVERRIDES`.
    """
    normalized_priority = _normalize_key(priority_value)
    return _current_route_registry()["priority_map"].get(normalized_priority, DEFAULT_ROUTE)


def next_required_route(required_routes: Iterable[str], completed_routes: Iterable[str]) -> str | None:
    """Return the next unfinished required route, or None when complete."""
    completed = set(normalize_routes(completed_routes))
    for route in normalize_routes(required_routes):
        if route not in completed:
            return route
    return None
