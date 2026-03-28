import importlib.util
import io
import json
import sys
import types
import unittest
import uuid
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path


CLI_PATH = Path(__file__).resolve().parents[1] / "src" / "core" / "cli.py"


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_cli_module():
    module_name = f"citeweave_cli_{uuid.uuid4().hex}"

    class DummyDocumentProcessor:
        pass

    class DummyLangGraphResearchSystem:
        pass

    _stub_module("prompt_toolkit", prompt=lambda *args, **kwargs: "")
    _stub_module("src", __path__=[])
    _stub_module("src.processing", __path__=[])
    _stub_module("src.processing.pdf", __path__=[])
    _stub_module("src.processing.pdf.document_processor", DocumentProcessor=DummyDocumentProcessor)
    _stub_module("src.agents", __path__=[])
    _stub_module("src.agents.multi_agent_research_system", LangGraphResearchSystem=DummyLangGraphResearchSystem)
    _stub_module(
        "src.agents.routing",
        active_route_configuration=lambda: {
            "default_route": "vector_search",
            "valid_routes": [],
            "aliases": {},
            "priority_map": {},
            "alias_overrides": {},
            "priority_overrides": {},
            "ignored_alias_overrides": [],
            "ignored_priority_overrides": [],
            "addon_config_paths": [],
            "addon_config_issues": [],
        },
    )

    spec = importlib.util.spec_from_file_location(module_name, CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CliRoutesCommandTests(unittest.TestCase):
    def test_handle_routes_command_supports_json_output(self):
        cli = _load_cli_module()
        expected = {
            "default_route": "graph_analysis",
            "valid_routes": ["graph_analysis", "vector_search"],
            "aliases": {"graph": "graph_analysis"},
            "priority_map": {"graph_database": "graph_analysis"},
            "alias_overrides": {},
            "priority_overrides": {},
            "ignored_alias_overrides": [],
            "ignored_priority_overrides": [],
            "addon_config_paths": ["/tmp/routes.json"],
            "addon_config_issues": [],
        }

        cli.active_route_configuration = lambda: expected

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_routes_command(Namespace(json=True))

        self.assertEqual(json.loads(buf.getvalue()), expected)


if __name__ == "__main__":
    unittest.main()
