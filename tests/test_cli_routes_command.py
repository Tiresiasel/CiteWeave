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

        def query_history_snapshot(self, limit=10, status="all", since_hours=None):
            return {
                "log_file": "data/query_history.jsonl",
                "requested_limit": limit,
                "status_filter": status,
                "since_hours": since_hours,
                "entries_returned": 0,
                "entries_considered": 0,
                "success_count": 0,
                "error_count": 0,
                "corrupt_count": 0,
                "average_duration_ms": None,
                "max_duration_ms": None,
                "latest_status": None,
                "latest_question": None,
                "latest_error": None,
                "entries": [],
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


class CliProgressCommandTests(unittest.TestCase):
    def test_handle_progress_command_supports_json_output(self):
        cli = _load_cli_module()
        expected = {
            "directory": "/papers",
            "cleared": False,
            "total_pdf_files": 3,
            "summary": {
                "total_tracked": 2,
                "completed": 1,
                "failed": 1,
                "success_rate": 50.0,
                "completed_files": ["/papers/ok.pdf"],
                "failed_files": {"/papers/bad.pdf": "parse error"},
                "aggregate_stats": {
                    "total_sentences": 10,
                    "sentences_with_citations": 4,
                    "total_citations": 7,
                    "total_references": 8,
                },
                "total_completed_duration_seconds": 5.0,
                "average_completed_duration_seconds": 5.0,
                "last_completed": {
                    "pdf_path": "/papers/ok.pdf",
                    "paper_id": "paper-1",
                    "processed_at": 123,
                    "duration_seconds": 5.0,
                    "stats": {"total_sentences": 10},
                },
                "failure_reasons": [{"error": "parse error", "count": 1}],
            },
            "pending_count": 2,
            "pending_files": ["/papers/bad.pdf", "/papers/pending.pdf"],
            "not_started_count": 1,
            "not_started_files": ["/papers/pending.pdf"],
            "retryable_failed_count": 1,
            "retryable_failed_files": ["/papers/bad.pdf"],
            "completed_count": 1,
            "completed_files": ["/papers/ok.pdf"],
            "failed_count": 1,
            "failed_files": {"/papers/bad.pdf": "parse error"},
            "average_completed_duration_seconds": 5.0,
            "estimated_remaining_seconds": 10.0,
        }

        class ExpectedKernel:
            def progress_summary(self, directory, clear=False):
                assert directory == "/papers"
                assert clear is False
                return expected

        cli.CiteWeaveKernel = ExpectedKernel
        original_isdir = cli.os.path.isdir
        cli.os.path.isdir = lambda path: path == "/papers"
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.handle_progress_command(Namespace(directory="/papers", clear=False, json=True, show_completed=False))
        finally:
            cli.os.path.isdir = original_isdir

        self.assertEqual(json.loads(buf.getvalue()), expected)

    def test_handle_progress_command_text_output_is_actionable(self):
        cli = _load_cli_module()
        progress = {
            "directory": "/papers",
            "cleared": False,
            "total_pdf_files": 3,
            "summary": {
                "total_tracked": 2,
                "completed": 1,
                "failed": 1,
                "success_rate": 50.0,
                "completed_files": ["/papers/ok.pdf"],
                "failed_files": {"/papers/bad.pdf": "parse error"},
                "aggregate_stats": {
                    "total_sentences": 10,
                    "sentences_with_citations": 4,
                    "total_citations": 7,
                    "total_references": 8,
                },
                "total_completed_duration_seconds": 5.0,
                "average_completed_duration_seconds": 5.0,
                "last_completed": {
                    "pdf_path": "/papers/ok.pdf",
                    "paper_id": "paper-1",
                    "processed_at": 123,
                    "duration_seconds": 5.0,
                    "stats": {"total_sentences": 10},
                },
                "failure_reasons": [{"error": "parse error", "count": 1}],
            },
            "pending_count": 2,
            "pending_files": ["/papers/bad.pdf", "/papers/pending.pdf"],
            "not_started_count": 1,
            "not_started_files": ["/papers/pending.pdf"],
            "retryable_failed_count": 1,
            "retryable_failed_files": ["/papers/bad.pdf"],
            "completed_count": 1,
            "completed_files": ["/papers/ok.pdf"],
            "failed_count": 1,
            "failed_files": {"/papers/bad.pdf": "parse error"},
            "average_completed_duration_seconds": 5.0,
            "estimated_remaining_seconds": 10.0,
        }

        class ExpectedKernel:
            def progress_summary(self, directory, clear=False):
                return progress

        cli.CiteWeaveKernel = ExpectedKernel
        original_isdir = cli.os.path.isdir
        cli.os.path.isdir = lambda path: True
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.handle_progress_command(Namespace(directory="/papers", clear=False, json=False, show_completed=True))
        finally:
            cli.os.path.isdir = original_isdir

        output = buf.getvalue()
        self.assertIn("Pending / resumable: 2", output)
        self.assertIn("Not started yet: 1", output)
        self.assertIn("Retryable failed files: 1", output)
        self.assertIn("Observed average time per completed file: 5.0s", output)
        self.assertIn("Estimated remaining wall time: 10.0s", output)
        self.assertIn("Total sentences processed: 10", output)
        self.assertIn("1 × parse error", output)
        self.assertIn("Tip: run batch-upload --resume", output)
        self.assertIn("Completed Files", output)
        self.assertIn("ok.pdf", output)


class CliQueryHistoryCommandTests(unittest.TestCase):
    def test_handle_query_history_command_supports_json_output(self):
        cli = _load_cli_module()
        expected = {
            "log_file": "data/query_history.jsonl",
            "requested_limit": 5,
            "status_filter": "error",
            "source_filter": "cli.query",
            "entries_returned": 2,
            "entries_considered": 2,
            "success_count": 0,
            "error_count": 2,
            "corrupt_count": 0,
            "average_duration_ms": 225.0,
            "max_duration_ms": 250,
            "latest_status": "error",
            "latest_question": "Why did retrieval fail?",
            "latest_source": "cli.query",
            "latest_error": "retrieval unavailable",
            "source_breakdown": [{"source": "cli.query", "count": 2}],
            "confirmation_breakdown": [{"confirmation": "continue", "count": 2}],
            "entries": [
                {"status": "error", "source": "cli.query", "confirmation": "continue", "question": "Why did retrieval fail?", "duration_ms": 250, "error": "retrieval unavailable"},
                {"status": "error", "source": "cli.query", "confirmation": "continue", "question": "Why did ranking fail?", "duration_ms": 200, "error": "timeout"},
            ],
        }

        class ExpectedKernel:
            def query_history_snapshot(self, limit=10, status="all", source="all"):
                assert limit == 5
                assert status == "error"
                assert source == "cli.query"
                return expected

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_query_history_command(Namespace(limit=5, status="error", source="cli.query", json=True, since_hours=None))

        self.assertEqual(json.loads(buf.getvalue()), expected)

    def test_handle_query_history_command_text_output_is_actionable(self):
        cli = _load_cli_module()
        expected = {
            "log_file": "data/query_history.jsonl",
            "requested_limit": 2,
            "status_filter": "all",
            "source_filter": "all",
            "entries_returned": 2,
            "entries_considered": 2,
            "success_count": 1,
            "error_count": 1,
            "corrupt_count": 0,
            "average_duration_ms": 175.0,
            "max_duration_ms": 250,
            "latest_status": "error",
            "latest_question": "Why did retrieval fail?",
            "latest_source": "cli.query",
            "latest_error": "retrieval unavailable",
            "source_breakdown": [{"source": "cli.query", "count": 2}],
            "confirmation_breakdown": [{"confirmation": "continue", "count": 2}],
            "entries": [
                {"status": "error", "source": "cli.query", "confirmation": "continue", "question": "Why did retrieval fail?", "duration_ms": 250, "error": "retrieval unavailable"},
                {"status": "success", "source": "cli.query", "confirmation": "continue", "question": "Summarize Porter", "duration_ms": 100},
            ],
        }

        class ExpectedKernel:
            def query_history_snapshot(self, limit=10, status="all", source="all"):
                assert limit == 2
                assert status == "all"
                assert source == "all"
                return expected

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_query_history_command(Namespace(limit=2, status="all", source="all", json=False, since_hours=None))

        output = buf.getvalue()
        self.assertIn("Status filter: all", output)
        self.assertIn("Source filter: all", output)
        self.assertIn("Successful queries: 1", output)
        self.assertIn("Failed queries: 1", output)
        self.assertIn("Average duration: 175.0 ms", output)
        self.assertIn("Slowest query: 250 ms", output)
        self.assertIn("Latest query: error [cli.query] - Why did retrieval fail?", output)
        self.assertIn("Latest error: retrieval unavailable", output)
        self.assertIn("Sources:", output)
        self.assertIn("Confirmations:", output)
        self.assertIn("[error] {cli.query} (250 ms) Why did retrieval fail?", output)
        self.assertIn("[success] {cli.query} (100 ms) Summarize Porter", output)


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


class CliQueryHistoryCommandTests(unittest.TestCase):
    def test_handle_query_history_command_passes_since_hours_to_kernel(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def query_history_snapshot(self, limit=10, status="all", source="all", confirmation="all", since_hours=None):
                assert limit == 5
                assert status == "error"
                assert source == "openclaw.facade.query"
                assert confirmation == "expand"
                assert since_hours == 24
                return {
                    "log_file": "data/query_history.jsonl",
                    "requested_limit": limit,
                    "status_filter": status,
                    "since_hours": since_hours,
                    "source_filter": source,
                    "confirmation_filter": confirmation,
                    "entries_returned": 1,
                    "entries_considered": 1,
                    "success_count": 0,
                    "error_count": 1,
                    "corrupt_count": 0,
                    "average_duration_ms": 123.0,
                    "max_duration_ms": 123,
                    "latest_status": "error",
                    "latest_question": "recent failure",
                    "latest_source": "openclaw.facade.query",
                    "latest_error": "timeout",
                    "source_breakdown": [{"source": "openclaw.facade.query", "count": 1}],
                    "confirmation_breakdown": [{"confirmation": "continue", "count": 1}],
                    "entries": [
                        {
                            "status": "error",
                            "source": "openclaw.facade.query",
                            "confirmation": "continue",
                            "duration_ms": 123,
                            "question": "recent failure",
                            "timestamp": 1_800_000_000,
                            "error": "timeout",
                        }
                    ],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_query_history_command(Namespace(limit=5, status="error", source="openclaw.facade.query", confirmation="expand", since_hours=24, json=False))

        output = buf.getvalue()
        self.assertIn("Time window: last 24 hours", output)
        self.assertIn("Source filter: openclaw.facade.query", output)
        self.assertIn("Confirmation filter: expand", output)
        self.assertIn("recent failure", output)
        self.assertIn("timeout", output)


class RepoHygieneTests(unittest.TestCase):
    def test_gitignore_explicitly_ignores_test_files_runtime_artifacts(self):
        gitignore = (CLI_PATH.parents[2] / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("test_files/*.", gitignore)
        self.assertIn("test_files/*", gitignore)
        self.assertIn("!test_files/README.md", gitignore)


if __name__ == "__main__":
    unittest.main()
