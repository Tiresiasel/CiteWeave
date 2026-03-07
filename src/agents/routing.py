"""Routing constants and helpers for the multi-agent query workflow.

This module centralizes route names and normalization logic so route updates
can be made in one place.
"""

from __future__ import annotations

from typing import Iterable, List

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


def normalize_routes(routes: Iterable[str]) -> List[str]:
    """Keep only valid routes, preserving order and removing duplicates."""
    normalized: List[str] = []
    seen = set()
    for route in routes:
        if route in VALID_ROUTES and route not in seen:
            normalized.append(route)
            seen.add(route)
    return normalized


def route_for_priority(priority_value: str) -> str:
    """Map retrieval priority value to route with a safe default."""
    return PRIORITY_TO_ROUTE.get(priority_value, DEFAULT_ROUTE)


def next_required_route(required_routes: Iterable[str], completed_routes: Iterable[str]) -> str | None:
    """Return the next unfinished required route, or None when complete."""
    completed = set(completed_routes)
    for route in required_routes:
        if route not in completed:
            return route
    return None
