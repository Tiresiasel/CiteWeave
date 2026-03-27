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
                "OPENAI_API_KEY": "sk-real-user-key",
                "CITEWEAVE_LLM_PROVIDER": "openclaw",
                "CITEWEAVE_LLM_MODEL": "openai-codex/gpt-5.4",
                "CITEWEAVE_LLM_API_BASE": "http://localhost:18789/v1",
            },
            clear=False,
        ):
            mod = load_env_config_module()
            kwargs = mod.chatopenai_kwargs()
            self.assertEqual(kwargs["model"], "openai-codex/gpt-5.4")
            self.assertEqual(kwargs["openai_api_base"], "http://localhost:18789/v1")
            self.assertEqual(kwargs["openai_api_key"], "not-needed-for-openclaw")
            self.assertNotIn("sk-real-user-key", str(kwargs))

    def test_openai_mode_uses_real_key(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-real-user-key",
                "CITEWEAVE_LLM_PROVIDER": "openai",
                "CITEWEAVE_LLM_MODEL": "gpt-4o-mini",
            },
            clear=False,
        ):
            mod = load_env_config_module()
            kwargs = mod.chatopenai_kwargs()
            self.assertEqual(kwargs["model"], "gpt-4o-mini")
            self.assertEqual(kwargs["openai_api_key"], "sk-real-user-key")


if __name__ == "__main__":
    unittest.main()
