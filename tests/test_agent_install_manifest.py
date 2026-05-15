from pathlib import Path

import yaml


MANIFEST = Path(__file__).resolve().parents[1] / "docs" / "agent" / "install_manifest.yaml"


def test_agent_install_manifest_is_parseable_and_english():
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["language"] == "en"
    assert payload["state_file"] == "config/install_session.local.json"
    assert "scripts/apply_install_choices.py" == payload["apply_script"]

    manifest_text = MANIFEST.read_text(encoding="utf-8")
    assert "Detected agent: {agent}. Use this as your Research Agent?" in manifest_text
    assert "Choose embedding mode for vector-level literature retrieval." in manifest_text
    assert "Choose how often CiteWeave should update the literature index." in manifest_text


def test_agent_install_manifest_choices_have_config_effects():
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    agent_choices = payload["steps"]["agent_detection"]["fallback_prompt"]["choices"]
    assert agent_choices["codex"]["state_patch"]["scheduler_adapter"] == "codex_heartbeat"
    assert agent_choices["openclaw"]["env_patch"]["CITEWEAVE_LLM_PROVIDER"] == "openclaw"
    assert agent_choices["claude_code"]["state_patch"]["scheduler_adapter"] == "launchd"

    source_choices = payload["steps"]["source"]["choices"]
    assert source_choices["zotero"]["env_patch"]["CITEWEAVE_ZOTERO_LIBRARY_DIR"] == "{source_dir}"
    assert source_choices["pdf_folder"]["env_patch"]["CITEWEAVE_REFERENCE_MANAGER"] == "pdf_folder"
    assert source_choices["single_pdf_test"]["state_patch"]["sync"]["enabled"] is False

    embedding = payload["steps"]["embedding"]["choices"]
    assert embedding["local"]["models"]["bge_large_en"]["env_patch"]["CITEWEAVE_EMBEDDING_DIMENSIONS"] == "1024"
    assert embedding["api"]["providers"]["openai"]["models"]["text-embedding-3-small"]["dimensions"] == 1536
    assert embedding["api"]["providers"]["openai"]["models"]["text-embedding-3-large"]["dimensions"] == 3072
