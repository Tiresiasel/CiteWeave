"""Routing constants and helpers for the multi-agent query workflow.

This module centralizes route names and normalization logic so route updates
can be made in one place.

It also supports optional OpenClaw addon overrides via environment variables:
`CITEWEAVE_ROUTE_PRIORITY_OVERRIDES` and `CITEWEAVE_ROUTE_ALIASES`,
and an optional config file via `CITEWEAVE_ROUTE_ADDON_CONFIG`.

Expected formats:
    CITEWEAVE_ROUTE_PRIORITY_OVERRIDES = {"priority_key": "route_name"}
    CITEWEAVE_ROUTE_ALIASES = {"alias_name": "route_name"}
    CITEWEAVE_ROUTE_ADDON_CONFIG = /path/to/route-config.json
    CITEWEAVE_ROUTE_ADDON_CONFIG = /path/base-route-config.json:/path/overlay-route-config.json
    CITEWEAVE_ROUTE_ADDON_CONFIG = /path/base-route-config.json:/path/route-overrides.d

When addon config file(s) are present, each accepts:
    {
      "aliases": {"alias_name": "route_name"},
      "priority_overrides": {"priority_key": "route_name"}
    }

When multiple config files are provided via the platform path separator,
files are loaded left-to-right and later files override earlier file entries.
Directory entries are also supported and expand to top-level `*.json` files in
sorted order, making layered addon config bundles easier to manage.

Only known routes are accepted to keep routing safe.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
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


def _addon_config_issue(reason: str, detail: Any = None, *, path: str | None = None) -> Dict[str, Any]:
    """Create a stable diagnostic payload for addon config file parsing."""
    entry: Dict[str, Any] = {"reason": reason}
    if path is not None:
        entry["path"] = path
    if detail is not None:
        entry["detail"] = detail
    return entry


def _coerce_mapping_payload(raw_value: Any) -> Tuple[Any | None, List[Dict[str, Any]]]:
    """Normalize alias/priority override payloads from both strings and mappings."""
    if not raw_value:
        return None, []

    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return None, []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None, [_invalid_override("invalid_json", None, None)]
    else:
        parsed = raw_value

    if not isinstance(parsed, dict):
        return None, [_invalid_override("non_object_payload", None, None)]

    return parsed, []


def _serialize_cache_payload(raw_value: Any) -> str | None:
    """Serialize override payload consistently for cache keys."""
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        return raw_value.strip() if raw_value.strip() else None
    return json.dumps(raw_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_single_addon_route_config(config_path: str) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None, List[Dict[str, Any]], str]:
    """Load route overrides from a single optional addon JSON config file."""
    expanded_path = str(Path(config_path).expanduser())
    issues: List[Dict[str, Any]] = []

    try:
        raw_text = Path(expanded_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        issues.append(_addon_config_issue("addon_config_not_found", path=expanded_path))
        return None, None, issues, expanded_path
    except OSError as exc:
        issues.append(
            _addon_config_issue(
                "addon_config_unreadable",
                str(exc),
                path=expanded_path,
            )
        )
        return None, None, issues, expanded_path

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        issues.append(
            _addon_config_issue(
                "addon_config_invalid_json",
                str(exc),
                path=expanded_path,
            )
        )
        return None, None, issues, expanded_path

    if not isinstance(payload, dict):
        issues.append(_addon_config_issue("addon_config_invalid_payload", "root_payload_not_object", path=expanded_path))
        return None, None, issues, expanded_path

    supported_top_level_keys = {"aliases", "priority_overrides", "priorityOverrides"}
    unknown_top_level_keys = sorted(key for key in payload if key not in supported_top_level_keys)
    if unknown_top_level_keys:
        issues.append(
            _addon_config_issue(
                "addon_config_unknown_keys",
                detail=unknown_top_level_keys,
                path=expanded_path,
            )
        )

    aliases_payload, alias_parse_issues = _coerce_mapping_payload(payload.get("aliases"))
    issues.extend(
        _addon_config_issue("addon_config_aliases_invalid", detail=str(issue), path=expanded_path)
        for issue in alias_parse_issues
    )

    priority_payload_source = payload.get("priority_overrides")
    legacy_priority_payload_source = payload.get("priorityOverrides")
    if priority_payload_source is None:
        priority_payload_source = legacy_priority_payload_source
    elif legacy_priority_payload_source is not None and priority_payload_source != legacy_priority_payload_source:
        issues.append(
            _addon_config_issue(
                "addon_config_priority_key_conflict",
                detail="both_priority_overrides_and_priorityOverrides_present_using_priority_overrides",
                path=expanded_path,
            )
        )

    priorities_payload, priority_parse_issues = _coerce_mapping_payload(priority_payload_source)
    issues.extend(
        _addon_config_issue("addon_config_priority_overrides_invalid", detail=str(issue), path=expanded_path)
        for issue in priority_parse_issues
    )

    return aliases_payload, priorities_payload, issues, expanded_path


def _expand_addon_route_config_paths(raw_paths: Iterable[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Expand addon config path entries into an ordered list of JSON files.

    File entries are kept as-is. Directory entries expand to top-level `*.json`
    files in sorted order so addon bundles can be layered without listing every
    file in the environment variable.
    """
    expanded_paths: List[str] = []
    issues: List[Dict[str, Any]] = []

    for raw_path in raw_paths:
        expanded_path = str(Path(raw_path).expanduser())
        path_obj = Path(expanded_path)

        if path_obj.is_dir():
            json_files = sorted(
                str(candidate)
                for candidate in path_obj.iterdir()
                if candidate.is_file() and candidate.suffix.lower() == ".json"
            )
            if json_files:
                expanded_paths.extend(json_files)
            else:
                issues.append(_addon_config_issue("addon_config_dir_empty", path=expanded_path))
            continue

        expanded_paths.append(expanded_path)

    return expanded_paths, issues


def _load_addon_route_config() -> Tuple[str | None, str | None, List[Dict[str, Any]], str | None, List[str]]:
    """Load route overrides from optional addon JSON config file(s).

    Returns:
        - cached merged alias override payload string
        - cached merged priority override payload string
        - parsed issue list across all config files
        - raw config path env value
        - expanded config path list in load order
    """
    config_path = os.getenv("CITEWEAVE_ROUTE_ADDON_CONFIG")
    if not config_path:
        return None, None, [], None, []

    raw_paths = [segment.strip() for segment in config_path.split(os.pathsep) if segment.strip()]
    if not raw_paths:
        return None, None, [], config_path, []

    expanded_paths, issues = _expand_addon_route_config_paths(raw_paths)

    merged_aliases: Dict[str, Any] = {}
    merged_priorities: Dict[str, Any] = {}

    for expanded_path in expanded_paths:
        aliases_payload, priorities_payload, file_issues, _ = _load_single_addon_route_config(expanded_path)
        issues.extend(file_issues)

        if aliases_payload:
            merged_aliases.update(aliases_payload)
        if priorities_payload:
            merged_priorities.update(priorities_payload)

    return (
        _serialize_cache_payload(merged_aliases or None),
        _serialize_cache_payload(merged_priorities or None),
        issues,
        config_path,
        expanded_paths,
    )


def _parse_route_alias_overrides_with_diagnostics(
    raw_value: str | Dict[str, Any] | None,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Parse and validate addon-provided route alias overrides.

    Safety constraints:
    - payload must be a dict of string -> string
    - target routes must be canonical known routes
    - canonical route names cannot be remapped to different routes
    - built-in aliases cannot be remapped to different routes

    Returns a tuple of:
    - accepted alias overrides
    - ignored entries with reasons for addon diagnostics
    """
    payload, parse_issues = _coerce_mapping_payload(raw_value)
    if parse_issues:
        return {}, parse_issues
    if payload is None:
        return {}, []

    overrides: Dict[str, str] = {}
    ignored: List[Dict[str, Any]] = []

    for alias_name, route_name in payload.items():
        if not isinstance(alias_name, str) or not isinstance(route_name, str):
            ignored.append(_invalid_override("non_string_entry", alias_name, route_name))
            continue

        normalized_alias = _normalize_key(alias_name)
        normalized_route = _resolve_from_alias_map(route_name, BASE_ROUTE_ALIASES)
        if not normalized_route:
            ignored.append(_invalid_override("unknown_route", alias_name, route_name))
            continue

        existing_target = overrides.get(normalized_alias)
        if existing_target is not None:
            if existing_target == normalized_route:
                ignored.append(_invalid_override("duplicate_normalized_key", alias_name, route_name))
            else:
                ignored.append(_invalid_override("normalized_key_conflict", alias_name, route_name))
            continue

        # Keep canonical route keys stable for safe behavior.
        if normalized_alias in VALID_ROUTES and normalized_alias != normalized_route:
            ignored.append(_invalid_override("canonical_route_locked", alias_name, route_name))
            continue

        base_alias_target = BASE_ROUTE_ALIASES.get(normalized_alias)
        if base_alias_target and base_alias_target != normalized_route:
            ignored.append(_invalid_override("built_in_alias_locked", alias_name, route_name))
            continue

        overrides[normalized_alias] = normalized_route

    return overrides, ignored


def _parse_route_alias_overrides(raw_value: str | Dict[str, Any] | None) -> Dict[str, str]:
    """Backward-compatible alias override parser."""
    overrides, _ = _parse_route_alias_overrides_with_diagnostics(raw_value)
    return overrides


def _parse_route_priority_overrides_with_diagnostics(
    raw_value: str | Dict[str, Any] | None,
    alias_map: Dict[str, str],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Parse and validate route priority overrides from environment.

    Invalid JSON, non-dict payloads, or mappings to unknown routes are ignored.

    Returns a tuple of:
    - accepted priority overrides
    - ignored entries with reasons for addon diagnostics
    """
    payload, parse_issues = _coerce_mapping_payload(raw_value)
    if parse_issues:
        return {}, parse_issues
    if payload is None:
        return {}, []

    overrides: Dict[str, str] = {}
    ignored: List[Dict[str, Any]] = []

    for priority_key, route_name in payload.items():
        if not isinstance(priority_key, str) or not isinstance(route_name, str):
            ignored.append(_invalid_override("non_string_entry", priority_key, route_name))
            continue

        normalized_key = _normalize_key(priority_key)
        normalized_route = _resolve_from_alias_map(route_name, alias_map)
        if not normalized_route:
            ignored.append(_invalid_override("unknown_route", priority_key, route_name))
            continue

        existing_target = overrides.get(normalized_key)
        if existing_target is not None:
            if existing_target == normalized_route:
                ignored.append(_invalid_override("duplicate_normalized_key", priority_key, route_name))
            else:
                ignored.append(_invalid_override("normalized_key_conflict", priority_key, route_name))
            continue

        overrides[normalized_key] = normalized_route

    return overrides, ignored


def _parse_route_priority_overrides(
    raw_value: str | Dict[str, Any] | None,
    alias_map: Dict[str, str],
) -> Dict[str, str]:
    """Backward-compatible priority override parser."""
    overrides, _ = _parse_route_priority_overrides_with_diagnostics(raw_value, alias_map)
    return overrides


@lru_cache(maxsize=8)
def _build_route_registry(
    addon_alias_override_raw: str | None,
    addon_priority_override_raw: str | None,
    env_alias_override_raw: str | None,
    env_priority_override_raw: str | None,
) -> Dict[str, Any]:
    """Build a cached route registry for the current addon override snapshot."""
    addon_alias_overrides, ignored_alias_overrides = _parse_route_alias_overrides_with_diagnostics(addon_alias_override_raw)
    env_alias_overrides, env_ignored_alias_overrides = _parse_route_alias_overrides_with_diagnostics(env_alias_override_raw)

    alias_map = dict(BASE_ROUTE_ALIASES)
    alias_map.update(addon_alias_overrides)
    alias_map.update(env_alias_overrides)

    ignored_alias_overrides.extend(env_ignored_alias_overrides)

    addon_alias_map_for_priority = dict(BASE_ROUTE_ALIASES)
    addon_alias_map_for_priority.update(addon_alias_overrides)

    addon_priority_overrides, ignored_priority_overrides = _parse_route_priority_overrides_with_diagnostics(
        addon_priority_override_raw,
        addon_alias_map_for_priority,
    )

    env_priority_overrides, env_ignored_priority_overrides = _parse_route_priority_overrides_with_diagnostics(
        env_priority_override_raw,
        alias_map,
    )
    ignored_priority_overrides.extend(env_ignored_priority_overrides)

    priority_map = dict(PRIORITY_TO_ROUTE)
    priority_map.update(addon_priority_overrides)
    priority_map.update(env_priority_overrides)

    alias_overrides = dict(addon_alias_overrides)
    alias_overrides.update(env_alias_overrides)

    priority_overrides = {
        key: value
        for key, value in priority_map.items()
        if PRIORITY_TO_ROUTE.get(key) != value
    }

    return {
        "valid_routes": list(VALID_ROUTES),
        "default_route": DEFAULT_ROUTE,
        "aliases": alias_map,
        "base_aliases": dict(BASE_ROUTE_ALIASES),
        "alias_overrides": alias_overrides,
        "addon_alias_overrides": dict(addon_alias_overrides),
        "env_alias_overrides": dict(env_alias_overrides),
        "ignored_alias_overrides": ignored_alias_overrides,
        "priority_map": priority_map,
        "addon_priority_overrides": dict(addon_priority_overrides),
        "env_priority_overrides": dict(env_priority_overrides),
        "priority_overrides": priority_overrides,
        "ignored_priority_overrides": ignored_priority_overrides,
    }


def _current_route_registry() -> Dict[str, Any]:
    """Return the active route registry for the current environment."""
    addon_alias_raw, addon_priority_raw, addon_issues, addon_config_path, addon_config_paths = _load_addon_route_config()
    registry = _build_route_registry(
        addon_alias_raw,
        addon_priority_raw,
        os.getenv("CITEWEAVE_ROUTE_ALIASES"),
        os.getenv("CITEWEAVE_ROUTE_PRIORITY_OVERRIDES"),
    )

    return {
        "valid_routes": list(registry["valid_routes"]),
        "default_route": registry["default_route"],
        "aliases": dict(registry["aliases"]),
        "base_aliases": dict(registry["base_aliases"]),
        "alias_overrides": dict(registry["alias_overrides"]),
        "addon_alias_overrides": dict(registry["addon_alias_overrides"]),
        "env_alias_overrides": dict(registry["env_alias_overrides"]),
        "ignored_alias_overrides": list(registry["ignored_alias_overrides"]),
        "priority_map": dict(registry["priority_map"]),
        "addon_priority_overrides": dict(registry["addon_priority_overrides"]),
        "env_priority_overrides": dict(registry["env_priority_overrides"]),
        "priority_overrides": dict(registry["priority_overrides"]),
        "ignored_priority_overrides": list(registry["ignored_priority_overrides"]),
        "addon_config_path": addon_config_path,
        "addon_config_paths": list(addon_config_paths),
        "addon_config_issues": list(addon_issues),
    }


def active_route_configuration() -> Dict[str, Any]:
    """Expose the active routing registry for addon diagnostics and tests.

    The returned snapshot includes:
    - canonical valid routes
    - the safe default route
    - full alias map after applying safe overrides
    - only the addon-provided alias/priority overrides that took effect
    - ignored addon override entries with stable diagnostic reasons
    - the final priority-to-route mapping
    - raw/expanded addon config path metadata for layered config debugging
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
        "addon_config_path": registry["addon_config_path"],
        "addon_config_paths": list(registry["addon_config_paths"]),
        "addon_config_issues": list(registry["addon_config_issues"]),
        "addon_alias_overrides": dict(registry["addon_alias_overrides"]),
        "env_alias_overrides": dict(registry["env_alias_overrides"]),
        "addon_priority_overrides": dict(registry["addon_priority_overrides"]),
        "env_priority_overrides": dict(registry["env_priority_overrides"]),
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

    Supports flexible key formats (e.g., `graph-database`),
    route aliases, and optional addon overrides through
    `CITEWEAVE_ROUTE_PRIORITY_OVERRIDES`.
    """
    registry = _current_route_registry()
    normalized_priority = _normalize_key(priority_value)

    if normalized_priority in registry["priority_map"]:
        return registry["priority_map"][normalized_priority]

    # Keep route_name resolution resilient when callers pass route aliases directly.
    return registry["aliases"].get(normalized_priority, DEFAULT_ROUTE)


def next_required_route(required_routes: Iterable[str], completed_routes: Iterable[str]) -> str | None:
    """Return the next unfinished required route, or None when complete."""
    completed = set(normalize_routes(completed_routes))
    for route in normalize_routes(required_routes):
        if route not in completed:
            return route
    return None
