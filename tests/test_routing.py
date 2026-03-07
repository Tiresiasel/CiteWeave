import importlib.util
from pathlib import Path


_ROUTING_PATH = Path(__file__).resolve().parents[1] / "src" / "agents" / "routing.py"
_spec = importlib.util.spec_from_file_location("citeweave_routing", _ROUTING_PATH)
routing = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(routing)

DEFAULT_ROUTE = routing.DEFAULT_ROUTE
ROUTE_AUTHOR_COLLECTION = routing.ROUTE_AUTHOR_COLLECTION
ROUTE_GRAPH_ANALYSIS = routing.ROUTE_GRAPH_ANALYSIS
ROUTE_PDF_ANALYSIS = routing.ROUTE_PDF_ANALYSIS
ROUTE_VECTOR_SEARCH = routing.ROUTE_VECTOR_SEARCH

next_required_route = routing.next_required_route
normalize_routes = routing.normalize_routes
route_for_priority = routing.route_for_priority


def test_normalize_routes_keeps_order_and_removes_invalid_duplicates():
    raw_routes = [
        ROUTE_VECTOR_SEARCH,
        "invalid_route",
        ROUTE_GRAPH_ANALYSIS,
        ROUTE_VECTOR_SEARCH,
        ROUTE_PDF_ANALYSIS,
    ]

    assert normalize_routes(raw_routes) == [
        ROUTE_VECTOR_SEARCH,
        ROUTE_GRAPH_ANALYSIS,
        ROUTE_PDF_ANALYSIS,
    ]


def test_route_for_priority_maps_known_and_defaults_unknown():
    assert route_for_priority("embedding_vector") == ROUTE_VECTOR_SEARCH
    assert route_for_priority("graph_database") == ROUTE_GRAPH_ANALYSIS
    assert route_for_priority("pdf_content") == ROUTE_PDF_ANALYSIS
    assert route_for_priority("author_index") == ROUTE_AUTHOR_COLLECTION
    assert route_for_priority("unknown_priority") == DEFAULT_ROUTE


def test_next_required_route_returns_first_unfinished_or_none():
    required_routes = [ROUTE_GRAPH_ANALYSIS, ROUTE_VECTOR_SEARCH, ROUTE_AUTHOR_COLLECTION]

    assert next_required_route(required_routes, [ROUTE_GRAPH_ANALYSIS]) == ROUTE_VECTOR_SEARCH
    assert next_required_route(required_routes, required_routes) is None
