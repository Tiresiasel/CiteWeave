"""
env_config.py
Environment-variable override system for CiteWeave.
Allows deployment without hardcoded values; environment variables take precedence over JSON config.
"""

import os
from typing import Optional

# ---------------------------------------------------------------------------
# LLM Provider Overrides
# ---------------------------------------------------------------------------
# Set CITEWEAVE_LLM_PROVIDER=openclaw to route all LLM calls through the
# local OpenClaw gateway (no separate API key required).
#
# When using openclaw provider:
#   - CITEWEAVE_LLM_API_BASE defaults to http://localhost:18789/v1
#   - CITEWEAVE_LLM_API_KEY can be any placeholder (OpenClaw authenticates via session)
#   - CITEWEAVE_LLM_MODEL defaults to openai-codex/gpt-5.4
#
# When using openai provider:
#   - Set OPENAI_API_KEY (or CITEWEAVE_LLM_API_KEY) to your key
#   - CITEWEAVE_LLM_API_BASE is optional (defaults to https://api.openai.com/v1)
# ---------------------------------------------------------------------------

OPENCLAW_DEFAULT_BASE = "http://localhost:18789/v1"
OPENCLAW_DEFAULT_MODEL = "openai-codex/gpt-5.4"


def get_llm_provider() -> str:
    """Returns the LLM provider: 'openclaw', 'openai', or 'ollama'."""
    return os.environ.get("CITEWEAVE_LLM_PROVIDER", "").lower() or ""


def get_llm_model(agent_name: Optional[str] = None) -> str:
    """
    Returns the model name.
    Priority: CITEWEAVE_LLM_MODEL > config file value.
    """
    env_model = os.environ.get("CITEWEAVE_LLM_MODEL", "").strip()
    return env_model if env_model else ""


def get_llm_api_base() -> str:
    """Returns the API base URL for the LLM provider."""
    if get_llm_provider() == "openclaw":
        return os.environ.get("CITEWEAVE_LLM_API_BASE", OPENCLAW_DEFAULT_BASE).rstrip("/")
    return os.environ.get("CITEWEAVE_LLM_API_BASE", "").strip()


def get_llm_api_key() -> str:
    """
    Returns the API key.
    For openclaw provider this can be a placeholder ('not-needed').
    For openai provider this must be a real key.
    """
    explicit = os.environ.get("CITEWEAVE_LLM_API_KEY", "").strip()
    if explicit:
        return explicit
    # When using openclaw gateway, the session provides auth — no API key needed.
    if os.environ.get("CITEWEAVE_LLM_PROVIDER", "").lower() == "openclaw":
        return "not-needed-for-openclaw"
    return os.environ.get("OPENAI_API_KEY", "not-set").strip()


def is_openclaw_mode() -> bool:
    """True when CiteWeave is configured to route LLM calls through the local OpenClaw gateway."""
    return get_llm_provider() == "openclaw"


# ---------------------------------------------------------------------------
# Neo4j / Database Overrides
# ---------------------------------------------------------------------------

def get_neo4j_password() -> str:
    """
    Returns the Neo4j password.
    When CITEWEAVE_NEO4J_PASSWORD is set it takes precedence.
    Falls back to the value in config/neo4j_config.json (or config/default.yaml).
    """
    return os.environ.get("CITEWEAVE_NEO4J_PASSWORD", "").strip()


# ---------------------------------------------------------------------------
# Convenience: build a ChatOpenAI kwargs dict from env
# ---------------------------------------------------------------------------

def chatopenai_kwargs(agent_name: Optional[str] = None) -> dict:
    """
    Returns a dict of kwargs suitable for ChatOpenAI(...).
    Handles both OpenClaw gateway and native OpenAI API modes.
    """
    model = get_llm_model(agent_name) or OPENCLAW_DEFAULT_MODEL
    api_base = get_llm_api_base()
    api_key = get_llm_api_key()

    kwargs = {"model": model}

    if api_base:
        kwargs["openai_api_base"] = api_base
    if api_key and api_key != "not-set":
        kwargs["openai_api_key"] = api_key

    return kwargs
