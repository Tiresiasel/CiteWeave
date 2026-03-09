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
from typing import Dict, Iterable, List

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


def _parse_route_alias_overrides(raw_value: str | None) -> Dict[str, str]:
    """Parse and validate addon-provided route alias overrides.

    Safety constraints:
    - payload must be a dict of string -> string
    - target routes must be canonical known routes
    - canonical route names cannot be remapped to different routes
    """
    if not raw_value:
        return {}

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    overrides: Dict[str, str] = {}
    for alias_name, route_name in parsed.items():
        if not isinstance(alias_name, str) or not isinstance(route_name, str):
            continue

        normalized_alias = _normalize_key(alias_name)
        normalized_route = _resolve_from_alias_map(route_name, BASE_ROUTE_ALIASES)
        if not normalized_route:
            continue

        # Keep canonical route keys stable for safe behavior.
        if normalized_alias in VALID_ROUTES and normalized_alias != normalized_route:
            continue

        overrides[normalized_alias] = normalized_route

    return overrides


def _parse_route_priority_overrides(raw_value: str | None) -> Dict[str, str]:
    """Parse and validate route priority overrides from environment.

    Invalid JSON, non-dict payloads, or mappings to unknown routes are ignored.
    """
    if not raw_value:
        return {}

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    overrides: Dict[str, str] = {}
    for priority_key, route_name in parsed.items():
        if not isinstance(priority_key, str) or not isinstance(route_name, str):
            continue
        normalized_route = resolve_route(route_name)
        if normalized_route:
            overrides[_normalize_key(priority_key)] = normalized_route

    return overrides


def resolve_route(route_value: str | None) -> str | None:
    """Resolve route name/alias to a canonical route, or None when invalid."""
    alias_map = dict(BASE_ROUTE_ALIASES)
    alias_map.update(_parse_route_alias_overrides(os.getenv("CITEWEAVE_ROUTE_ALIASES")))
    return _resolve_from_alias_map(route_value, alias_map)


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
    override_raw = os.getenv("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES")
    overrides = _parse_route_priority_overrides(override_raw)

    if normalized_priority in overrides:
        return overrides[normalized_priority]

    return PRIORITY_TO_ROUTE.get(normalized_priority, DEFAULT_ROUTE)


def next_required_route(required_routes: Iterable[str], completed_routes: Iterable[str]) -> str | None:
    """Return the next unfinished required route, or None when complete."""
    completed = set(normalize_routes(completed_routes))
    for route in normalize_routes(required_routes):
        if route not in completed:
            return route
    return None
