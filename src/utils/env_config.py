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
#   - CITEWEAVE_LLM_API_KEY can be any placeholder when gateway auth is none
#   - CITEWEAVE_LLM_MODEL is the OpenClaw agent target, default openclaw/default
#   - CITEWEAVE_OPENCLAW_BACKEND_MODEL optionally overrides the gateway backend
#     provider/model through the x-openclaw-model request header
#
# When using openai provider:
#   - Set OPENAI_API_KEY (or CITEWEAVE_LLM_API_KEY) to your key
#   - CITEWEAVE_LLM_API_BASE is optional (defaults to https://api.openai.com/v1)
# ---------------------------------------------------------------------------

OPENCLAW_DEFAULT_BASE = "http://localhost:18789/v1"
OPENCLAW_DEFAULT_MODEL = "openclaw/default"

LOCAL_EMBEDDING_DEFAULT_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_DEFAULT_MODEL = "text-embedding-3-small"


def get_llm_provider() -> str:
    """Returns the LLM provider: 'openclaw', 'openai', or 'ollama'."""
    return os.environ.get("CITEWEAVE_LLM_PROVIDER", "").lower() or ""


def get_llm_model(agent_name: Optional[str] = None) -> str:
    """
    Returns the chat model name.

    In OpenClaw mode this must be an OpenClaw agent target such as
    ``openclaw/default``. For backwards compatibility, if a raw provider model
    is supplied in CITEWEAVE_LLM_MODEL (for example ``openai-codex/gpt-5.4``),
    it is treated as the backend override and the request still targets
    ``openclaw/default``.
    """
    env_model = os.environ.get("CITEWEAVE_LLM_MODEL", "").strip()
    if get_llm_provider() == "openclaw":
        if not env_model or not env_model.startswith("openclaw"):
            return OPENCLAW_DEFAULT_MODEL
    return env_model if env_model else ""


def get_openclaw_backend_model(agent_name: Optional[str] = None) -> str:
    """
    Returns the optional OpenClaw backend provider/model override.

    OpenClaw's OpenAI-compatible endpoint treats the OpenAI ``model`` field as
    an agent target. Backend provider model selection belongs in the
    ``x-openclaw-model`` header.
    """
    explicit = (
        os.environ.get("CITEWEAVE_OPENCLAW_BACKEND_MODEL", "").strip()
        or os.environ.get("CITEWEAVE_LLM_BACKEND_MODEL", "").strip()
    )
    if explicit:
        return explicit

    legacy_model = os.environ.get("CITEWEAVE_LLM_MODEL", "").strip()
    if legacy_model and not legacy_model.startswith("openclaw"):
        return legacy_model

    return ""


def get_llm_api_base() -> str:
    """Returns the API base URL for the LLM provider."""
    configured_base = os.environ.get("CITEWEAVE_LLM_API_BASE", "").strip()
    if get_llm_provider() == "openclaw":
        return (configured_base or OPENCLAW_DEFAULT_BASE).rstrip("/")
    return configured_base


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
# Embedding Provider Overrides
# ---------------------------------------------------------------------------
# Embeddings are separate from LLM calls. OpenClaw can coordinate deployment
# and query flow, while CiteWeave still owns the local vector index.
#
# Supported providers:
#   - local  : sentence-transformers model, default all-MiniLM-L6-v2 (384 dims)
#   - openai : OpenAI Embeddings API, default text-embedding-3-small (1536 dims)
# ---------------------------------------------------------------------------

def get_embedding_provider() -> str:
    """Returns the embedding provider override: 'local' or 'openai'."""
    return os.environ.get("CITEWEAVE_EMBEDDING_PROVIDER", "").strip().lower()


def get_embedding_model() -> str:
    """Returns the embedding model override, if set."""
    return os.environ.get("CITEWEAVE_EMBEDDING_MODEL", "").strip()


def get_embedding_dimensions() -> Optional[int]:
    """Returns an explicit embedding dimension override, if set."""
    raw = os.environ.get("CITEWEAVE_EMBEDDING_DIMENSIONS", "").strip()
    if not raw:
        return None
    return int(raw)


def get_embedding_profile() -> str:
    """Returns the named embedding profile override, if set."""
    return os.environ.get("CITEWEAVE_EMBEDDING_PROFILE", "").strip()


def get_embedding_device() -> str:
    """Returns the local embedding device override: auto, cpu, cuda, cuda:0, etc."""
    return os.environ.get("CITEWEAVE_EMBEDDING_DEVICE", "").strip()


def get_embedding_batch_size() -> Optional[int]:
    """Returns the embedding encode batch-size override, if set."""
    raw = os.environ.get("CITEWEAVE_EMBEDDING_BATCH_SIZE", "").strip()
    if not raw:
        return None
    return int(raw)


def get_embedding_require_cuda() -> Optional[bool]:
    """Returns whether local embeddings must use CUDA, if explicitly configured."""
    raw = os.environ.get("CITEWEAVE_EMBEDDING_REQUIRE_CUDA", "").strip().lower()
    if not raw:
        return None
    return raw in {"1", "true", "yes", "y", "on"}


def get_embedding_trust_remote_code() -> Optional[bool]:
    """Returns whether local embedding models may use Hugging Face remote code."""
    raw = os.environ.get("CITEWEAVE_EMBEDDING_TRUST_REMOTE_CODE", "").strip().lower()
    if not raw:
        return None
    return raw in {"1", "true", "yes", "y", "on"}


def get_embedding_api_key() -> str:
    """Returns the API key for remote embedding providers."""
    explicit = os.environ.get("CITEWEAVE_EMBEDDING_API_KEY", "").strip()
    if explicit:
        return explicit
    return os.environ.get("OPENAI_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# Neo4j / Database Overrides
# ---------------------------------------------------------------------------

def get_neo4j_uri() -> str:
    """Returns the Neo4j Bolt URI override, if set."""
    return os.environ.get("CITEWEAVE_NEO4J_URI", "").strip()


def get_neo4j_username() -> str:
    """Returns the Neo4j username override, if set."""
    return os.environ.get("CITEWEAVE_NEO4J_USERNAME", "").strip()


def get_neo4j_database() -> str:
    """Returns the Neo4j database override, if set."""
    return os.environ.get("CITEWEAVE_NEO4J_DATABASE", "").strip()


def get_neo4j_password() -> str:
    """Returns the Neo4j password override, if set."""
    return os.environ.get("CITEWEAVE_NEO4J_PASSWORD", "").strip()


def apply_neo4j_env_overrides(config: dict) -> dict:
    """Return a Neo4j config dict with environment variables applied."""
    merged = dict(config)
    if get_neo4j_uri():
        merged["uri"] = get_neo4j_uri()
    if get_neo4j_username():
        merged["username"] = get_neo4j_username()
    if get_neo4j_password():
        merged["password"] = get_neo4j_password()
    if get_neo4j_database():
        merged["database"] = get_neo4j_database()
    return merged


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

    if is_openclaw_mode():
        backend_model = get_openclaw_backend_model(agent_name)
        if backend_model:
            kwargs["default_headers"] = {"x-openclaw-model": backend_model}

    return kwargs
