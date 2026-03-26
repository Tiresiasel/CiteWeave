import importlib.util
import os
import uuid
import json
import tempfile
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


def test_route_for_priority_accepts_route_names_and_aliases():
    routing = _load_routing_module()
    original_aliases = os.environ.get("CITEWEAVE_ROUTE_ALIASES")

    try:
        os.environ["CITEWEAVE_ROUTE_ALIASES"] = '{"citation_map": "graph_analysis"}'

        assert routing.route_for_priority("graph") == routing.ROUTE_GRAPH_ANALYSIS
        assert routing.route_for_priority("vector") == routing.ROUTE_VECTOR_SEARCH
        assert routing.route_for_priority("citation_map") == routing.ROUTE_GRAPH_ANALYSIS
    finally:
        if original_aliases is None:
            os.environ.pop("CITEWEAVE_ROUTE_ALIASES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ALIASES"] = original_aliases


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
            '{"citation_map": "graph_analysis", "semantic": "vector", "vector_search": "graph", "graph": "vector"}'
        )

        # Additive aliases work
        assert routing.resolve_route("citation_map") == routing.ROUTE_GRAPH_ANALYSIS
        assert routing.resolve_route("semantic") == routing.ROUTE_VECTOR_SEARCH

        # Canonical route names are protected from remapping
        assert routing.resolve_route("vector_search") == routing.ROUTE_VECTOR_SEARCH

        # Built-in short aliases are also protected from remapping
        assert routing.resolve_route("graph") == routing.ROUTE_GRAPH_ANALYSIS
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


def test_active_route_configuration_exposes_safe_effective_overrides():
    routing = _load_routing_module()
    original_aliases = os.environ.get("CITEWEAVE_ROUTE_ALIASES")
    original_priorities = os.environ.get("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES")

    try:
        os.environ["CITEWEAVE_ROUTE_ALIASES"] = '{"citation_map": "graph_analysis", "vector_search": "graph"}'
        os.environ["CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"] = '{"pdf_content": "citation_map", "bad": "not_real"}'

        config = routing.active_route_configuration()

        assert config["default_route"] == routing.DEFAULT_ROUTE
        assert config["aliases"]["citation_map"] == routing.ROUTE_GRAPH_ANALYSIS
        assert config["alias_overrides"] == {"citation_map": routing.ROUTE_GRAPH_ANALYSIS}
        assert config["priority_map"]["pdf_content"] == routing.ROUTE_GRAPH_ANALYSIS
        assert config["priority_overrides"] == {"pdf_content": routing.ROUTE_GRAPH_ANALYSIS}
    finally:
        if original_aliases is None:
            os.environ.pop("CITEWEAVE_ROUTE_ALIASES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ALIASES"] = original_aliases

        if original_priorities is None:
            os.environ.pop("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"] = original_priorities


def test_active_route_configuration_reports_ignored_override_reasons():
    routing = _load_routing_module()
    original_aliases = os.environ.get("CITEWEAVE_ROUTE_ALIASES")
    original_priorities = os.environ.get("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES")

    try:
        os.environ["CITEWEAVE_ROUTE_ALIASES"] = (
            '{"semantic": "vector", "vector_search": "graph", "graph": "vector", "broken": "nope"}'
        )
        os.environ["CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"] = '{"graph_database": "semantic", "bad": "missing_route"}'

        config = routing.active_route_configuration()

        assert config["alias_overrides"] == {"semantic": routing.ROUTE_VECTOR_SEARCH}
        assert config["ignored_alias_overrides"] == [
            {"reason": "canonical_route_locked", "key": "vector_search", "route": "graph"},
            {"reason": "built_in_alias_locked", "key": "graph", "route": "vector"},
            {"reason": "unknown_route", "key": "broken", "route": "nope"},
        ]
        assert config["priority_overrides"] == {"graph_database": routing.ROUTE_VECTOR_SEARCH}
        assert config["ignored_priority_overrides"] == [
            {"reason": "unknown_route", "key": "bad", "route": "missing_route"}
        ]
    finally:
        if original_aliases is None:
            os.environ.pop("CITEWEAVE_ROUTE_ALIASES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ALIASES"] = original_aliases

        if original_priorities is None:
            os.environ.pop("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"] = original_priorities


def test_active_route_configuration_reports_invalid_payload_shape():
    routing = _load_routing_module()
    original_aliases = os.environ.get("CITEWEAVE_ROUTE_ALIASES")
    original_priorities = os.environ.get("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES")

    try:
        os.environ["CITEWEAVE_ROUTE_ALIASES"] = "[]"
        os.environ["CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"] = "{"

        config = routing.active_route_configuration()

        assert config["alias_overrides"] == {}
        assert config["ignored_alias_overrides"] == [{"reason": "non_object_payload"}]
        assert config["priority_overrides"] == {}
        assert config["ignored_priority_overrides"] == [{"reason": "invalid_json"}]
    finally:
        if original_aliases is None:
            os.environ.pop("CITEWEAVE_ROUTE_ALIASES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ALIASES"] = original_aliases

        if original_priorities is None:
            os.environ.pop("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"] = original_priorities


def test_route_registry_refreshes_when_env_changes():
    routing = _load_routing_module()
    original = os.environ.get("CITEWEAVE_ROUTE_ALIASES")

    try:
        os.environ["CITEWEAVE_ROUTE_ALIASES"] = '{"semantic": "vector"}'
        first_config = routing.active_route_configuration()
        assert first_config["aliases"]["semantic"] == routing.ROUTE_VECTOR_SEARCH

        os.environ["CITEWEAVE_ROUTE_ALIASES"] = '{"semantic": "graph"}'
        second_config = routing.active_route_configuration()
        assert second_config["aliases"]["semantic"] == routing.ROUTE_GRAPH_ANALYSIS
    finally:
        if original is None:
            os.environ.pop("CITEWEAVE_ROUTE_ALIASES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ALIASES"] = original


def test_active_route_configuration_rejects_normalized_alias_collisions():
    routing = _load_routing_module()
    original_aliases = os.environ.get("CITEWEAVE_ROUTE_ALIASES")

    try:
        os.environ["CITEWEAVE_ROUTE_ALIASES"] = (
            '{"semantic-search": "vector", "semantic search": "graph", "citation map": "graph", "citation_map": "graph"}'
        )

        config = routing.active_route_configuration()

        assert config["alias_overrides"] == {
            "semantic_search": routing.ROUTE_VECTOR_SEARCH,
            "citation_map": routing.ROUTE_GRAPH_ANALYSIS,
        }
        assert config["ignored_alias_overrides"] == [
            {"reason": "normalized_key_conflict", "key": "semantic search", "route": "graph"},
            {"reason": "duplicate_normalized_key", "key": "citation_map", "route": "graph"},
        ]
    finally:
        if original_aliases is None:
            os.environ.pop("CITEWEAVE_ROUTE_ALIASES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ALIASES"] = original_aliases


def test_active_route_configuration_rejects_normalized_priority_collisions():
    routing = _load_routing_module()
    original_priorities = os.environ.get("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES")

    try:
        os.environ["CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"] = (
            '{"graph-database": "vector", "graph database": "author", "pdf content": "pdf", "pdf_content": "pdf"}'
        )

        config = routing.active_route_configuration()

        assert config["priority_overrides"] == {
            "graph_database": routing.ROUTE_VECTOR_SEARCH,
        }
        assert config["ignored_priority_overrides"] == [
            {"reason": "normalized_key_conflict", "key": "graph database", "route": "author"},
            {"reason": "duplicate_normalized_key", "key": "pdf_content", "route": "pdf"},
        ]
    finally:
        if original_priorities is None:
            os.environ.pop("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"] = original_priorities


def test_next_required_route_returns_first_unfinished_or_none():
    routing = _load_routing_module()
    required_routes = ["graph", "vector_search", "author_collection"]

    assert routing.next_required_route(required_routes, [routing.ROUTE_GRAPH_ANALYSIS]) == routing.ROUTE_VECTOR_SEARCH
    assert routing.next_required_route(required_routes, required_routes) is None


def test_active_route_configuration_loads_file_overrides():
    routing = _load_routing_module()
    original_path = os.environ.get("CITEWEAVE_ROUTE_ADDON_CONFIG")
    original_aliases = os.environ.get("CITEWEAVE_ROUTE_ALIASES")
    original_priorities = os.environ.get("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(
            {
                "aliases": {"citation_map": "graph_analysis"},
                "priority_overrides": {"author_index": "citation_map"},
            },
            tmp,
        )
        tmp_path = tmp.name

    try:
        os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = tmp_path

        config = routing.active_route_configuration()

        assert config["addon_config_path"] == tmp_path
        assert config["addon_config_paths"] == [tmp_path]
        assert config["addon_config_issues"] == []
        assert config["addon_alias_overrides"] == {
            "citation_map": routing.ROUTE_GRAPH_ANALYSIS
        }
        assert config["addon_priority_overrides"] == {
            "author_index": routing.ROUTE_GRAPH_ANALYSIS
        }
        assert config["alias_overrides"]["citation_map"] == routing.ROUTE_GRAPH_ANALYSIS
        assert config["priority_overrides"]["author_index"] == routing.ROUTE_GRAPH_ANALYSIS
        assert routing.resolve_route("citation_map") == routing.ROUTE_GRAPH_ANALYSIS
        assert routing.route_for_priority("author_index") == routing.ROUTE_GRAPH_ANALYSIS
    finally:
        if original_path is None:
            os.environ.pop("CITEWEAVE_ROUTE_ADDON_CONFIG", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = original_path
        if original_aliases is None:
            os.environ.pop("CITEWEAVE_ROUTE_ALIASES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ALIASES"] = original_aliases
        if original_priorities is None:
            os.environ.pop("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"] = original_priorities
        Path(tmp_path).unlink(missing_ok=True)


def test_env_overrides_take_precedence_over_addon_file_overrides():
    routing = _load_routing_module()
    original_path = os.environ.get("CITEWEAVE_ROUTE_ADDON_CONFIG")
    original_aliases = os.environ.get("CITEWEAVE_ROUTE_ALIASES")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump({"aliases": {"semantic": "vector"}}, tmp)
        tmp_path = tmp.name

    try:
        os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = tmp_path
        os.environ["CITEWEAVE_ROUTE_ALIASES"] = json.dumps({"semantic": "graph"})

        config = routing.active_route_configuration()

        assert config["addon_alias_overrides"] == {"semantic": routing.ROUTE_VECTOR_SEARCH}
        assert config["env_alias_overrides"] == {"semantic": routing.ROUTE_GRAPH_ANALYSIS}
        assert config["alias_overrides"] == {"semantic": routing.ROUTE_GRAPH_ANALYSIS}
        assert routing.resolve_route("semantic") == routing.ROUTE_GRAPH_ANALYSIS
    finally:
        if original_path is None:
            os.environ.pop("CITEWEAVE_ROUTE_ADDON_CONFIG", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = original_path
        if original_aliases is None:
            os.environ.pop("CITEWEAVE_ROUTE_ALIASES", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ALIASES"] = original_aliases
        Path(tmp_path).unlink(missing_ok=True)


def test_active_route_configuration_reports_missing_addon_config_file():
    routing = _load_routing_module()
    original_path = os.environ.get("CITEWEAVE_ROUTE_ADDON_CONFIG")

    try:
        os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = "/tmp/citeweave-missing-route-config.json"

        config = routing.active_route_configuration()

        assert config["addon_config_path"] == "/tmp/citeweave-missing-route-config.json"
        assert config["addon_config_paths"] == ["/tmp/citeweave-missing-route-config.json"]
        assert config["addon_config_issues"] == [
            {
                "reason": "addon_config_not_found",
                "path": "/tmp/citeweave-missing-route-config.json",
            }
        ]
    finally:
        if original_path is None:
            os.environ.pop("CITEWEAVE_ROUTE_ADDON_CONFIG", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = original_path


def test_active_route_configuration_accepts_legacy_priority_overrides_key():
    routing = _load_routing_module()
    original_path = os.environ.get("CITEWEAVE_ROUTE_ADDON_CONFIG")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump({"priorityOverrides": {"author_index": "graph"}}, tmp)
        tmp_path = tmp.name

    try:
        os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = tmp_path

        config = routing.active_route_configuration()

        assert config["addon_priority_overrides"] == {
            "author_index": routing.ROUTE_GRAPH_ANALYSIS
        }
        assert config["addon_config_paths"] == [tmp_path]
        assert config["addon_config_issues"] == []
        assert routing.route_for_priority("author_index") == routing.ROUTE_GRAPH_ANALYSIS
    finally:
        if original_path is None:
            os.environ.pop("CITEWEAVE_ROUTE_ADDON_CONFIG", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = original_path
        Path(tmp_path).unlink(missing_ok=True)


def test_active_route_configuration_merges_layered_addon_configs():
    routing = _load_routing_module()
    original_path = os.environ.get("CITEWEAVE_ROUTE_ADDON_CONFIG")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as base_tmp:
        json.dump(
            {
                "aliases": {"citation_map": "graph", "semantic": "vector"},
                "priority_overrides": {"author_index": "graph"},
            },
            base_tmp,
        )
        base_tmp_path = base_tmp.name

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as overlay_tmp:
        json.dump(
            {
                "aliases": {"semantic": "pdf"},
                "priority_overrides": {"author_index": "semantic", "pdf_content": "semantic"},
            },
            overlay_tmp,
        )
        overlay_tmp_path = overlay_tmp.name

    try:
        os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = os.pathsep.join([base_tmp_path, overlay_tmp_path])

        config = routing.active_route_configuration()

        assert config["addon_config_path"] == os.pathsep.join([base_tmp_path, overlay_tmp_path])
        assert config["addon_config_paths"] == [base_tmp_path, overlay_tmp_path]
        assert config["addon_config_issues"] == []
        assert config["addon_alias_overrides"] == {
            "citation_map": routing.ROUTE_GRAPH_ANALYSIS,
            "semantic": routing.ROUTE_PDF_ANALYSIS,
        }
        assert config["addon_priority_overrides"] == {
            "author_index": routing.ROUTE_PDF_ANALYSIS,
            "pdf_content": routing.ROUTE_PDF_ANALYSIS,
        }
        assert routing.resolve_route("semantic") == routing.ROUTE_PDF_ANALYSIS
        assert routing.route_for_priority("author_index") == routing.ROUTE_PDF_ANALYSIS
        assert routing.route_for_priority("pdf_content") == routing.ROUTE_PDF_ANALYSIS
    finally:
        if original_path is None:
            os.environ.pop("CITEWEAVE_ROUTE_ADDON_CONFIG", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = original_path
        Path(base_tmp_path).unlink(missing_ok=True)
        Path(overlay_tmp_path).unlink(missing_ok=True)


def test_active_route_configuration_reports_unknown_addon_config_keys():
    routing = _load_routing_module()
    original_path = os.environ.get("CITEWEAVE_ROUTE_ADDON_CONFIG")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump({"aliases": {"citation_map": "graph"}, "notes": {"owner": "addon-team"}}, tmp)
        tmp_path = tmp.name

    try:
        os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = tmp_path

        config = routing.active_route_configuration()

        assert config["addon_alias_overrides"] == {
            "citation_map": routing.ROUTE_GRAPH_ANALYSIS
        }
        assert config["addon_config_paths"] == [tmp_path]
        assert config["addon_config_issues"] == [
            {
                "reason": "addon_config_unknown_keys",
                "path": tmp_path,
                "detail": ["notes"],
            }
        ]
    finally:
        if original_path is None:
            os.environ.pop("CITEWEAVE_ROUTE_ADDON_CONFIG", None)
        else:
            os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = original_path
        Path(tmp_path).unlink(missing_ok=True)


def test_active_route_configuration_expands_directory_entries_in_load_order():
    routing = _load_routing_module()
    original_path = os.environ.get("CITEWEAVE_ROUTE_ADDON_CONFIG")

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir) / "00-base.json"
        overlay_path = Path(tmp_dir) / "10-overlay.json"
        ignored_path = Path(tmp_dir) / "README.txt"

        base_path.write_text(
            json.dumps(
                {
                    "aliases": {"semantic": "vector"},
                    "priority_overrides": {"author_index": "graph"},
                }
            ),
            encoding="utf-8",
        )
        overlay_path.write_text(
            json.dumps(
                {
                    "aliases": {"semantic": "pdf", "citation_map": "graph"},
                    "priority_overrides": {"author_index": "semantic"},
                }
            ),
            encoding="utf-8",
        )
        ignored_path.write_text("not json", encoding="utf-8")

        try:
            os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = tmp_dir

            config = routing.active_route_configuration()

            assert config["addon_config_path"] == tmp_dir
            assert config["addon_config_paths"] == [str(base_path), str(overlay_path)]
            assert config["addon_config_issues"] == []
            assert config["addon_alias_overrides"] == {
                "semantic": routing.ROUTE_PDF_ANALYSIS,
                "citation_map": routing.ROUTE_GRAPH_ANALYSIS,
            }
            assert config["addon_priority_overrides"] == {
                "author_index": routing.ROUTE_PDF_ANALYSIS,
            }
            assert routing.resolve_route("semantic") == routing.ROUTE_PDF_ANALYSIS
            assert routing.route_for_priority("author_index") == routing.ROUTE_PDF_ANALYSIS
        finally:
            if original_path is None:
                os.environ.pop("CITEWEAVE_ROUTE_ADDON_CONFIG", None)
            else:
                os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = original_path


def test_active_route_configuration_reports_empty_addon_config_directory():
    routing = _load_routing_module()
    original_path = os.environ.get("CITEWEAVE_ROUTE_ADDON_CONFIG")

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = tmp_dir

            config = routing.active_route_configuration()

            assert config["addon_config_paths"] == []
            assert config["addon_config_issues"] == [
                {
                    "reason": "addon_config_dir_empty",
                    "path": tmp_dir,
                }
            ]
        finally:
            if original_path is None:
                os.environ.pop("CITEWEAVE_ROUTE_ADDON_CONFIG", None)
            else:
                os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = original_path


def test_active_route_configuration_ignores_subdirectories_in_addon_config_path():
    """Subdirectories within an addon config directory path should be silently skipped."""
    routing = _load_routing_module()
    original_path = os.environ.get("CITEWEAVE_ROUTE_ADDON_CONFIG")

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "00-valid.json"
        json_path.write_text(
            json.dumps({"aliases": {"semantic": "vector"}}),
            encoding="utf-8",
        )
        # Nested subdirectory should be ignored (not walked recursively)
        nested_dir = Path(tmp_dir) / "subdir"
        nested_dir.mkdir()
        (nested_dir / "nested.json").write_text(
            json.dumps({"aliases": {"nested_alias": "author_collection"}}),
            encoding="utf-8",
        )

        try:
            os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = tmp_dir

            config = routing.active_route_configuration()

            # Only the top-level JSON should be picked up; nested files ignored
            assert config["addon_config_paths"] == [str(json_path)]
            assert config["addon_config_issues"] == []
            assert config["addon_alias_overrides"] == {"semantic": routing.ROUTE_VECTOR_SEARCH}
            # Nested alias should NOT appear
            assert "nested_alias" not in config["addon_alias_overrides"]
            assert routing.resolve_route("nested_alias") is None
        finally:
            if original_path is None:
                os.environ.pop("CITEWEAVE_ROUTE_ADDON_CONFIG", None)
            else:
                os.environ["CITEWEAVE_ROUTE_ADDON_CONFIG"] = original_path
