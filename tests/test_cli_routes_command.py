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

    class FakeKernel:
        def routes_snapshot(self):
            return {
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
            }

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
    _stub_module("src.kernel", CiteWeaveKernel=FakeKernel, BatchUploadTracker=object)

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

        class ExpectedKernel:
            def routes_snapshot(self):
                return expected

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_routes_command(Namespace(json=True))

        self.assertEqual(json.loads(buf.getvalue()), expected)


class CliHealthAndBootstrapCommandTests(unittest.TestCase):
    def test_handle_health_command_supports_json_output(self):
        cli = _load_cli_module()
        expected = {
            "project_root": "/repo",
            "summary": {
                "overall_status": "degraded",
                "missing_files": ["docker_compose"],
                "down_services": ["openclaw_gateway"],
                "action_items": [
                    "Create or restore required config files: docker_compose",
                    "Start or fix backend services: openclaw_gateway",
                ],
            },
            "env": {
                "llm_provider": "openclaw",
                "llm_model": "gpt-test",
                "gateway_base": "http://localhost:18789/v1",
            },
            "files": {".env": True, "docker_compose": False},
            "services": {
                "qdrant": {"ok": True, "status": 200, "url": "http://localhost:6333/collections"},
                "openclaw_gateway": {"ok": False, "status": 503, "url": "http://localhost:18789/v1/models", "error": "unavailable"},
            },
        }

        class ExpectedKernel:
            def health_snapshot(self):
                return expected

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_health_command(Namespace(json=True))

        self.assertEqual(json.loads(buf.getvalue()), expected)

    def test_handle_health_command_text_output_leads_with_status_and_actions(self):
        cli = _load_cli_module()
        expected = {
            "project_root": "/repo",
            "summary": {
                "overall_status": "degraded",
                "missing_files": ["docker_compose"],
                "down_services": ["openclaw_gateway"],
                "action_items": [
                    "Create or restore required config files: docker_compose",
                    "Start or fix backend services: openclaw_gateway",
                ],
            },
            "env": {
                "llm_provider": "openclaw",
                "llm_model": "gpt-test",
                "gateway_base": "http://localhost:18789/v1",
            },
            "files": {".env": True, "docker_compose": False},
            "services": {
                "qdrant": {"ok": True, "status": 200, "url": "http://localhost:6333/collections"},
                "openclaw_gateway": {"ok": False, "status": 503, "url": "http://localhost:18789/v1/models", "error": "unavailable"},
            },
        }

        class ExpectedKernel:
            def health_snapshot(self):
                return expected

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_health_command(Namespace(json=False))

        output = buf.getvalue()
        self.assertIn("Overall status: degraded", output)
        self.assertIn("Recommended next actions:", output)
        self.assertIn("Create or restore required config files: docker_compose", output)
        self.assertIn("Start or fix backend services: openclaw_gateway", output)

    def test_handle_bootstrap_plan_command_supports_json_output(self):
        cli = _load_cli_module()
        expected = {
            "local_cli": {
                "script": "bash scripts/bootstrap_local.sh",
                "next_steps": [".venv/bin/python -m src.core.cli upload path/to/paper.pdf"],
            },
            "openclaw": {
                "script": "bash scripts/bootstrap_openclaw.sh",
                "next_steps": ["openclaw gateway status"],
            },
        }

        class ExpectedKernel:
            def bootstrap_plan(self):
                return expected

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_bootstrap_plan_command(Namespace(json=True))

        self.assertEqual(json.loads(buf.getvalue()), expected)


if __name__ == "__main__":
    unittest.main()
