import importlib.util
import os
import uuid
from pathlib import Path


_ROUTING_PATH = Path(__file__).resolve().parents[1] / "src" / "agents" / "routing.py"


def _load_routing_module():
    module_name = f"citeweave_routing_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, _ROUTING_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_routes_keeps_order_and_supports_aliases():
    routing = _load_routing_module()

    raw_routes = [
        routing.ROUTE_VECTOR_SEARCH,
        "invalid_route",
        "GRAPH",
        routing.ROUTE_VECTOR_SEARCH,
        "pdf-analysis",
        "author",
    ]

    assert routing.normalize_routes(raw_routes) == [
        routing.ROUTE_VECTOR_SEARCH,
        routing.ROUTE_GRAPH_ANALYSIS,
        routing.ROUTE_PDF_ANALYSIS,
        routing.ROUTE_AUTHOR_COLLECTION,
    ]


def test_route_for_priority_maps_known_and_defaults_unknown():
    routing = _load_routing_module()

    assert routing.route_for_priority("embedding_vector") == routing.ROUTE_VECTOR_SEARCH
    assert routing.route_for_priority("graph_database") == routing.ROUTE_GRAPH_ANALYSIS
    assert routing.route_for_priority("graph-database") == routing.ROUTE_GRAPH_ANALYSIS
    assert routing.route_for_priority("pdf_content") == routing.ROUTE_PDF_ANALYSIS
    assert routing.route_for_priority("author_index") == routing.ROUTE_AUTHOR_COLLECTION
    assert routing.route_for_priority("unknown_priority") == routing.DEFAULT_ROUTE


def test_route_for_priority_honors_safe_env_overrides():
    routing = _load_routing_module()
    original = os.environ.get("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES")

    try:
        os.environ["CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"] = (
            '{"embedding_vector": "graph", "pdf content": "author_collection", "bad": "not_real"}'
        )

        assert routing.route_for_priority("embedding_vector") == routing.ROUTE_GRAPH_ANALYSIS
        assert routing.route_for_priority("pdf_content") == routing.ROUTE_AUTHOR_COLLECTION
        # Invalid override route should be ignored
        assert routing.route_for_priority("bad") == routing.DEFAULT_ROUTE
    finally:
        if original is None:
            os.environ.pop("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"] = original


def test_route_alias_overrides_allow_addon_aliases_but_keep_canonical_stable():
    routing = _load_routing_module()
    original = os.environ.get("CITEWEAVE_ROUTE_ALIASES")

    try:
        os.environ["CITEWEAVE_ROUTE_ALIASES"] = (
            '{"citation_map": "graph_analysis", "semantic": "vector", "vector_search": "graph"}'
        )

        # Additive aliases work
        assert routing.resolve_route("citation_map") == routing.ROUTE_GRAPH_ANALYSIS
        assert routing.resolve_route("semantic") == routing.ROUTE_VECTOR_SEARCH

        # Canonical route names are protected from remapping
        assert routing.resolve_route("vector_search") == routing.ROUTE_VECTOR_SEARCH
    finally:
        if original is None:
            os.environ.pop("CITEWEAVE_ROUTE_ALIASES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ALIASES"] = original


def test_route_alias_overrides_are_used_by_normalize_routes():
    routing = _load_routing_module()
    original = os.environ.get("CITEWEAVE_ROUTE_ALIASES")

    try:
        os.environ["CITEWEAVE_ROUTE_ALIASES"] = '{"citation_map": "graph_analysis"}'
        assert routing.normalize_routes(["citation_map", "graph"]) == [routing.ROUTE_GRAPH_ANALYSIS]
    finally:
        if original is None:
            os.environ.pop("CITEWEAVE_ROUTE_ALIASES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ALIASES"] = original


def test_next_required_route_returns_first_unfinished_or_none():
    routing = _load_routing_module()
    required_routes = ["graph", "vector_search", "author_collection"]

    assert routing.next_required_route(required_routes, [routing.ROUTE_GRAPH_ANALYSIS]) == routing.ROUTE_VECTOR_SEARCH
    assert routing.next_required_route(required_routes, required_routes) is None
