import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "utils" / "env_config.py"


def load_env_config_module():
    spec = importlib.util.spec_from_file_location("env_config_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EnvConfigTests(unittest.TestCase):
    def test_openclaw_mode_never_forwards_real_openai_key(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "user-key-from-env",
                "CITEWEAVE_LLM_PROVIDER": "openclaw",
                "CITEWEAVE_LLM_MODEL": "openai-codex/gpt-5.4",
                "CITEWEAVE_LLM_API_BASE": "http://localhost:18789/v1",
            },
            clear=False,
        ):
            mod = load_env_config_module()
            kwargs = mod.chatopenai_kwargs()
            self.assertEqual(kwargs["model"], "openclaw/default")
            self.assertEqual(kwargs["openai_api_base"], "http://localhost:18789/v1")
            self.assertEqual(kwargs["openai_api_key"], "not-needed-for-openclaw")
            self.assertEqual(kwargs["default_headers"], {"x-openclaw-model": "openai-codex/gpt-5.4"})
            self.assertNotIn("user-key-from-env", str(kwargs))

    def test_openclaw_mode_uses_default_base_when_env_value_is_blank(self):
        with patch.dict(
            os.environ,
            {
                "CITEWEAVE_LLM_PROVIDER": "openclaw",
                "CITEWEAVE_LLM_API_BASE": "",
            },
            clear=True,
        ):
            mod = load_env_config_module()
            kwargs = mod.chatopenai_kwargs()
            self.assertEqual(kwargs["model"], "openclaw/default")
            self.assertEqual(kwargs["openai_api_base"], "http://localhost:18789/v1")
            self.assertEqual(kwargs["openai_api_key"], "not-needed-for-openclaw")
            self.assertNotIn("default_headers", kwargs)

    def test_openclaw_mode_supports_explicit_backend_model_header(self):
        with patch.dict(
            os.environ,
            {
                "CITEWEAVE_LLM_PROVIDER": "openclaw",
                "CITEWEAVE_LLM_MODEL": "openclaw/research",
                "CITEWEAVE_OPENCLAW_BACKEND_MODEL": "openai/gpt-5.5",
            },
            clear=True,
        ):
            mod = load_env_config_module()
            kwargs = mod.chatopenai_kwargs()
            self.assertEqual(kwargs["model"], "openclaw/research")
            self.assertEqual(kwargs["default_headers"], {"x-openclaw-model": "openai/gpt-5.5"})

    def test_openai_mode_uses_real_key(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "user-key-from-env",
                "CITEWEAVE_LLM_PROVIDER": "openai",
                "CITEWEAVE_LLM_MODEL": "gpt-4o-mini",
            },
            clear=False,
        ):
            mod = load_env_config_module()
            kwargs = mod.chatopenai_kwargs()
            self.assertEqual(kwargs["model"], "gpt-4o-mini")
            self.assertEqual(kwargs["openai_api_key"], "user-key-from-env")

    def test_neo4j_env_overrides_config(self):
        with patch.dict(
            os.environ,
            {
                "CITEWEAVE_NEO4J_URI": "bolt://db.example:7687",
                "CITEWEAVE_NEO4J_USERNAME": "app",
                "CITEWEAVE_NEO4J_PASSWORD": "local-secret",
                "CITEWEAVE_NEO4J_DATABASE": "citeweave",
            },
            clear=False,
        ):
            mod = load_env_config_module()
            config = mod.apply_neo4j_env_overrides(
                {
                    "uri": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "template",
                    "database": "neo4j",
                }
            )
            self.assertEqual(config["uri"], "bolt://db.example:7687")
            self.assertEqual(config["username"], "app")
            self.assertEqual(config["password"], "local-secret")
            self.assertEqual(config["database"], "citeweave")


if __name__ == "__main__":
    unittest.main()
