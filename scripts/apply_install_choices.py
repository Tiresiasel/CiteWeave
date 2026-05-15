#!/usr/bin/env python3
"""Apply AI installation choices to CiteWeave local configuration.

This script is intentionally boring: it maps fixed wizard choices to `.env`
runtime settings and `config/install_session.local.json` installation state so
an installing agent does not need to hand-edit config files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_STATE_PATH = ROOT / "config" / "install_session.local.json"
ENV_TEMPLATE_PATH = ROOT / ".env_template"


AGENT_ALIASES = {
    "openclaw": "OpenClaw",
    "codex": "Codex",
    "claude": "Claude Code",
    "claude code": "Claude Code",
    "claude-code": "Claude Code",
    "others": "Others",
    "other": "Others",
    "unknown": "Unknown",
}

SCHEDULER_BY_AGENT = {
    "OpenClaw": "openclaw_automation",
    "Codex": "codex_heartbeat",
    "Claude Code": "launchd",
    "Others": "custom_agent",
    "Unknown": "custom_agent",
}

LOCAL_PROFILES = {
    "bge_large_en": {
        "CITEWEAVE_EMBEDDING_PROVIDER": "local",
        "CITEWEAVE_EMBEDDING_PROFILE": "bge_large_en",
        "CITEWEAVE_EMBEDDING_MODEL": "BAAI/bge-large-en-v1.5",
        "CITEWEAVE_EMBEDDING_DIMENSIONS": "1024",
        "CITEWEAVE_EMBEDDING_BATCH_SIZE": "16",
    },
    "mini_l6_compat": {
        "CITEWEAVE_EMBEDDING_PROVIDER": "local",
        "CITEWEAVE_EMBEDDING_PROFILE": "mini_l6_compat",
        "CITEWEAVE_EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "CITEWEAVE_EMBEDDING_DIMENSIONS": "384",
        "CITEWEAVE_EMBEDDING_BATCH_SIZE": "64",
    },
    "qwen3_embedding_4b_cuda": {
        "CITEWEAVE_EMBEDDING_PROVIDER": "local",
        "CITEWEAVE_EMBEDDING_PROFILE": "qwen3_embedding_4b_cuda",
        "CITEWEAVE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-4B",
        "CITEWEAVE_EMBEDDING_DIMENSIONS": "2560",
        "CITEWEAVE_EMBEDDING_BATCH_SIZE": "8",
        "CITEWEAVE_EMBEDDING_DEVICE": "cuda",
        "CITEWEAVE_EMBEDDING_REQUIRE_CUDA": "true",
        "CITEWEAVE_EMBEDDING_TRUST_REMOTE_CODE": "true",
    },
    "qwen3_embedding_8b_cuda": {
        "CITEWEAVE_EMBEDDING_PROVIDER": "local",
        "CITEWEAVE_EMBEDDING_PROFILE": "qwen3_embedding_8b_cuda",
        "CITEWEAVE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
        "CITEWEAVE_EMBEDDING_DIMENSIONS": "4096",
        "CITEWEAVE_EMBEDDING_BATCH_SIZE": "4",
        "CITEWEAVE_EMBEDDING_DEVICE": "cuda",
        "CITEWEAVE_EMBEDDING_REQUIRE_CUDA": "true",
        "CITEWEAVE_EMBEDDING_TRUST_REMOTE_CODE": "true",
    },
}

OPENAI_MODELS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def read_env_lines(path: Path) -> List[str]:
    if path.exists():
        return path.read_text(encoding="utf-8").splitlines()
    if ENV_TEMPLATE_PATH.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ENV_TEMPLATE_PATH, path)
        return path.read_text(encoding="utf-8").splitlines()
    return []


def parse_env_values(lines: Iterable[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in lines:
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def write_env(path: Path, updates: Dict[str, str]) -> List[str]:
    lines = read_env_lines(path)
    seen = set()
    out = []

    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        out.append(line)

    if updates:
        if out and out[-1].strip():
            out.append("")
        out.append("# CiteWeave AI installation choices")
        for key in sorted(updates):
            if key not in seen:
                out.append(f"{key}={updates[key]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return sorted(updates)


def read_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def normalize_agent(value: str) -> str:
    normalized = (value or "").strip()
    return AGENT_ALIASES.get(normalized.lower(), normalized or "Unknown")


def normalize_reference_manager(value: str) -> str:
    normalized = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pdf": "pdf_folder",
        "pdfs": "pdf_folder",
        "folder": "pdf_folder",
        "zotero/mendeley/endnote": "zotero",
    }
    return aliases.get(normalized, normalized)


def detect_agent() -> str:
    env = os.environ
    if env.get("CODEX_HOME") or env.get("CODEX_SANDBOX") or env.get("CODEX_CWD"):
        return "Codex"
    if env.get("OPENCLAW_GATEWAY_TOKEN") or env.get("OPENCLAW_HOME"):
        return "OpenClaw"
    if env.get("CLAUDECODE") or env.get("CLAUDE_CODE") or env.get("CLAUDECODE_CWD"):
        return "Claude Code"
    return "Unknown"


def apply_agent(args: argparse.Namespace, env_updates: Dict[str, str], state: Dict[str, Any]) -> None:
    if not args.research_agent:
        if args.detect_agent:
            state["detected_agent"] = detect_agent()
        return

    agent = normalize_agent(args.research_agent)
    scheduler = SCHEDULER_BY_AGENT.get(agent, "custom_agent")

    env_updates["CITEWEAVE_RESEARCH_AGENT"] = agent
    env_updates["CITEWEAVE_QUERY_ORCHESTRATION"] = "openclaw" if agent == "OpenClaw" else "agent_managed"

    if agent == "OpenClaw":
        env_updates.update(
            {
                "CITEWEAVE_LLM_PROVIDER": "openclaw",
                "CITEWEAVE_LLM_MODEL": "openclaw/default",
                "CITEWEAVE_LLM_API_BASE": "http://localhost:18789/v1",
                "CITEWEAVE_LLM_API_KEY": "not-needed-for-openclaw",
            }
        )

    state["research_agent"] = agent
    state["scheduler_adapter"] = scheduler
    if args.detected_agent:
        state["detected_agent"] = normalize_agent(args.detected_agent)
    if args.agent_command:
        state["agent_command"] = args.agent_command


def resolve_pdf_source(source_dir: Optional[str], reference_manager: str) -> Optional[str]:
    if not source_dir:
        return None
    path = Path(source_dir).expanduser().resolve()
    if path.is_dir() and reference_manager == "zotero" and (path / "storage").is_dir():
        return str((path / "storage").resolve())
    return str(path)


def apply_source(args: argparse.Namespace, env_updates: Dict[str, str], state: Dict[str, Any]) -> None:
    reference_manager = normalize_reference_manager(args.reference_manager or "")
    if not reference_manager and not args.source_dir:
        return
    if not reference_manager:
        reference_manager = "pdf_folder"

    source_dir = str(Path(args.source_dir).expanduser().resolve()) if args.source_dir else ""
    resolved_pdf_source = resolve_pdf_source(source_dir, reference_manager)

    env_updates["CITEWEAVE_REFERENCE_MANAGER"] = reference_manager
    if source_dir:
        env_updates["CITEWEAVE_LITERATURE_SOURCE_DIR"] = source_dir
        if reference_manager == "zotero":
            env_updates["CITEWEAVE_ZOTERO_LIBRARY_DIR"] = source_dir

    state["reference_manager"] = reference_manager
    if args.source_location_mode:
        state["source_location_mode"] = args.source_location_mode
    if source_dir:
        state["literature_source_dir"] = source_dir
    if resolved_pdf_source:
        state["resolved_pdf_source"] = resolved_pdf_source
    if reference_manager != "single_pdf_test":
        state.setdefault("install_mode", "library")


def apply_single_pdf(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    if not args.single_pdf:
        return
    single_pdf_path = str(Path(args.single_pdf).expanduser().resolve())
    state["install_mode"] = "single_pdf_test"
    state["single_pdf_path"] = single_pdf_path
    state["sync"] = {
        "enabled": False,
        "reason": "Single PDF Test does not configure recurring sync.",
    }


def previous_embedding(values: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        values.get("CITEWEAVE_EMBEDDING_PROVIDER", ""),
        values.get("CITEWEAVE_EMBEDDING_MODEL", ""),
        values.get("CITEWEAVE_EMBEDDING_DIMENSIONS", ""),
    )


def apply_embedding(
    args: argparse.Namespace,
    env_updates: Dict[str, str],
    state: Dict[str, Any],
    existing_env: Dict[str, str],
    warnings: List[str],
) -> None:
    if not args.embedding_mode and not args.embedding_profile and not args.embedding_model:
        return

    mode = (args.embedding_mode or "local").strip().lower()
    before = previous_embedding(existing_env)

    if mode == "local":
        profile = args.embedding_profile or "bge_large_en"
        if profile in LOCAL_PROFILES:
            env_updates.update(LOCAL_PROFILES[profile])
        elif profile == "other":
            if not args.embedding_model or not args.embedding_dimensions:
                raise SystemExit("--embedding-profile other requires --embedding-model and --embedding-dimensions")
            env_updates.update(
                {
                    "CITEWEAVE_EMBEDDING_PROVIDER": "local",
                    "CITEWEAVE_EMBEDDING_PROFILE": "",
                    "CITEWEAVE_EMBEDDING_MODEL": args.embedding_model,
                    "CITEWEAVE_EMBEDDING_DIMENSIONS": str(args.embedding_dimensions),
                }
            )
            warnings.append("Custom local embedding profiles are experimental and require an embedding smoke test.")
        else:
            raise SystemExit(f"Unsupported local embedding profile: {profile}")

        if args.embedding_device:
            env_updates["CITEWEAVE_EMBEDDING_DEVICE"] = args.embedding_device
        if args.embedding_batch_size:
            env_updates["CITEWEAVE_EMBEDDING_BATCH_SIZE"] = str(args.embedding_batch_size)
        if args.embedding_trust_remote_code:
            env_updates["CITEWEAVE_EMBEDDING_TRUST_REMOTE_CODE"] = "true"

        state["embedding"] = {
            "provider": "local",
            "profile": env_updates.get("CITEWEAVE_EMBEDDING_PROFILE", ""),
            "model": env_updates["CITEWEAVE_EMBEDDING_MODEL"],
            "dimensions": int(env_updates["CITEWEAVE_EMBEDDING_DIMENSIONS"]),
            "device": env_updates.get("CITEWEAVE_EMBEDDING_DEVICE", ""),
            "batch_size": int(env_updates.get("CITEWEAVE_EMBEDDING_BATCH_SIZE", "0") or 0),
        }
    elif mode == "api":
        provider = (args.api_provider or "openai").strip().lower().replace("-", "_")
        model = args.embedding_model or "text-embedding-3-small"
        dimensions = args.embedding_dimensions or OPENAI_MODELS.get(model)
        if not dimensions:
            raise SystemExit("API embeddings require --embedding-dimensions for unknown models")

        env_updates.update(
            {
                "CITEWEAVE_EMBEDDING_PROVIDER": "openai",
                "CITEWEAVE_EMBEDDING_PROFILE": "",
                "CITEWEAVE_EMBEDDING_MODEL": model,
                "CITEWEAVE_EMBEDDING_DIMENSIONS": str(dimensions),
            }
        )
        if args.api_key:
            env_updates["CITEWEAVE_EMBEDDING_API_KEY"] = args.api_key
        if provider == "openai_compatible":
            if not args.api_base_url:
                raise SystemExit("OpenAI-compatible embeddings require --api-base-url")
            env_updates["CITEWEAVE_EMBEDDING_API_BASE"] = args.api_base_url.rstrip("/")
            warnings.append("OpenAI-compatible embedding APIs are experimental and require an embedding smoke test.")

        state["embedding"] = {
            "provider": "openai",
            "api_provider": provider,
            "model": model,
            "dimensions": int(dimensions),
            "api_key_configured": bool(args.api_key or existing_env.get("CITEWEAVE_EMBEDDING_API_KEY")),
        }
        if args.api_base_url:
            state["embedding"]["api_base_url"] = args.api_base_url.rstrip("/")
    else:
        raise SystemExit(f"Unsupported embedding mode: {args.embedding_mode}")

    after = (
        env_updates.get("CITEWEAVE_EMBEDDING_PROVIDER", before[0]),
        env_updates.get("CITEWEAVE_EMBEDDING_MODEL", before[1]),
        env_updates.get("CITEWEAVE_EMBEDDING_DIMENSIONS", before[2]),
    )
    if any(before) and before != after:
        state["embedding_requires_rebuild"] = True
        warnings.append(
            "Embedding provider, model, or dimensions changed. If an index already exists, require full vector rebuild and full re-ingest before resume."
        )


def apply_sync(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    if not args.sync_schedule:
        return
    if state.get("install_mode") == "single_pdf_test":
        state["sync"] = {
            "enabled": False,
            "reason": "Single PDF Test does not configure recurring sync.",
        }
        return

    schedule = args.sync_schedule.strip().lower()
    if schedule == "none":
        state["sync"] = {"enabled": False}
        return
    if schedule not in {"every_5_minutes", "every_30_minutes", "daily", "custom"}:
        raise SystemExit(f"Unsupported sync schedule: {args.sync_schedule}")

    sync: Dict[str, Any] = {
        "enabled": True,
        "schedule": schedule,
        "skip_failed": True if args.skip_failed is None else bool(args.skip_failed),
        "processors": int(args.processors or 10),
        "resume_only": True,
        "scheduler_rule": "Check for active ingestion first. Never start a second ingest. Never use --force-restart or --clear-progress unless the user explicitly confirms a rebuild.",
    }
    if schedule == "daily" and args.daily_time:
        sync["daily_time"] = args.daily_time
    if schedule == "custom" and args.custom_schedule:
        sync["custom_schedule"] = args.custom_schedule
    state["sync"] = sync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply AI installation choices to CiteWeave config.")
    parser.add_argument("--env-path", default=str(DEFAULT_ENV_PATH), help="Path to .env file to update.")
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH), help="Path to local installation state JSON.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")

    parser.add_argument("--detect-agent", action="store_true", help="Detect current agent and record detected_agent.")
    parser.add_argument("--detected-agent", help="Agent detected by the installer.")
    parser.add_argument("--research-agent", help="Research Agent: OpenClaw, Codex, Claude Code, Others.")
    parser.add_argument("--agent-command", help="Command or path for Others.")

    parser.add_argument("--reference-manager", help="zotero, mendeley, endnote, or pdf_folder.")
    parser.add_argument("--source-location-mode", help="default, custom, or pdf_folder.")
    parser.add_argument("--source-dir", help="Reference manager directory or PDF folder.")
    parser.add_argument("--single-pdf", help="Single PDF path for smoke-test installation.")

    parser.add_argument("--embedding-mode", choices=["local", "api"], help="Embedding mode.")
    parser.add_argument("--embedding-profile", help="Local profile: bge_large_en, mini_l6_compat, qwen profile, or other.")
    parser.add_argument("--embedding-model", help="Embedding model name.")
    parser.add_argument("--embedding-dimensions", type=int, help="Embedding vector dimensions.")
    parser.add_argument("--embedding-device", help="Local embedding device, e.g. cpu, auto, cuda.")
    parser.add_argument("--embedding-batch-size", type=int, help="Local embedding batch size.")
    parser.add_argument("--embedding-trust-remote-code", action="store_true", help="Allow Hugging Face remote code.")
    parser.add_argument("--api-provider", choices=["openai", "openai_compatible"], help="API embedding provider.")
    parser.add_argument("--api-key", help="Embedding API key. Written only to .env, never to state JSON.")
    parser.add_argument("--api-base-url", help="OpenAI-compatible embedding base URL.")

    parser.add_argument("--sync-schedule", help="every_5_minutes, every_30_minutes, daily, custom, or none.")
    parser.add_argument("--daily-time", help="Local wall-clock time for daily sync, e.g. 03:00.")
    parser.add_argument("--custom-schedule", help="Custom scheduler expression or natural-language schedule.")
    parser.add_argument("--processors", type=int, default=10, help="Ingestion worker count for sync state.")
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument("--skip-failed", dest="skip_failed", action="store_true", help="Skip failed files on recurring sync.")
    skip_group.add_argument("--no-skip-failed", dest="skip_failed", action="store_false", help="Retry failed files on recurring sync.")
    parser.set_defaults(skip_failed=None)
    return parser


def redact_env_updates(keys: List[str]) -> List[str]:
    return sorted(keys)


def main() -> int:
    args = build_parser().parse_args()
    env_path = Path(args.env_path).expanduser().resolve()
    state_path = Path(args.state_path).expanduser().resolve()

    existing_lines = read_env_lines(env_path)
    existing_env = parse_env_values(existing_lines)
    state = read_state(state_path)
    env_updates: Dict[str, str] = {}
    warnings: List[str] = []

    if args.detect_agent and not args.research_agent:
        state["detected_agent"] = detect_agent()

    apply_agent(args, env_updates, state)
    apply_source(args, env_updates, state)
    apply_single_pdf(args, state)
    apply_embedding(args, env_updates, state, existing_env, warnings)
    apply_sync(args, state)

    state["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    applied_env_keys = write_env(env_path, env_updates) if env_updates else []
    write_state(state_path, state)

    report = {
        "status": "ok",
        "env_path": str(env_path),
        "state_path": str(state_path),
        "applied_env_keys": redact_env_updates(applied_env_keys),
        "updated_state_keys": sorted(state.keys()),
        "warnings": warnings,
    }
    if args.detect_agent:
        report["detected_agent"] = state.get("detected_agent", detect_agent())

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print("CiteWeave installation choices applied.")
        print(f"Updated state: {state_path}")
        if applied_env_keys:
            print("Updated env keys: " + ", ".join(applied_env_keys))
        for warning in warnings:
            print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
