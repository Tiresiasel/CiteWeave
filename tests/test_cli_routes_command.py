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

        def query_history_snapshot(self, limit=10, status="all", source="all", confirmation="all", satisfaction="all", since_hours=None, contains="", question_contains="", error_contains="", response_contains="", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None):
            return {
                "log_file": "data/query_history.jsonl",
                "requested_limit": limit,
                "status_filter": status,
                "source_filter": source,
                "confirmation_filter": confirmation,
                "contains_filter": contains,
                "question_contains_filter": question_contains,
                "error_contains_filter": error_contains,
                "response_contains_filter": response_contains,
                "planned_database_filter": planned_database,
                "planned_method_filter": planned_method,
                "planned_route_filter": planned_route,
                "since_hours": since_hours,
                "min_duration_ms_filter": min_duration_ms,
                "max_duration_ms_filter": max_duration_ms,
                "min_response_chars_filter": min_response_chars,
                "max_response_chars_filter": max_response_chars,
                "entries_returned": 0,
                "matching_entries_total": 0,
                "entries_considered": 0,
                "success_count": 0,
                "error_count": 0,
                "corrupt_count": 0,
                "average_duration_ms": None,
                "max_duration_ms": None,
                "latest_status": None,
                "latest_question": None,
                "latest_source": None,
                "latest_error": None,
                "source_breakdown": [],
                "confirmation_breakdown": [],
                "query_plan_database_breakdown": [],
                "query_plan_method_breakdown": [],
                "query_plan_route_breakdown": [],
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




class CliQueryPlanSummaryTests(unittest.TestCase):
    def test_query_plan_summary_labels_routes_inferred_from_database_names(self):
        cli = _load_cli_module()

        self.assertEqual(
            cli._format_query_plan_summary({
                "query_plan_databases": ["graph_db", "vector_db"],
                "query_plan_methods": ["get_papers_citing_paper"],
            }),
            "routes=graph_analysis, vector_search (inferred from db) | db=graph_db, vector_db | methods=get_papers_citing_paper",
        )

    def test_query_plan_summary_marks_mixed_explicit_and_inferred_routes(self):
        cli = _load_cli_module()

        self.assertEqual(
            cli._format_query_plan_summary({
                "query_plan_routes": ["vector_search"],
                "query_plan_databases": ["vector_db", "pdf_db"],
                "query_plan_methods": ["search_relevant_sentences", "get_full_pdf_content"],
            }),
            "routes=vector_search, pdf_analysis (+1 inferred from db) | db=vector_db, pdf_db | methods=search_relevant_sentences, get_full_pdf_content",
        )

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
            cli.handle_routes_command(Namespace(json=True, check=False))

        self.assertEqual(json.loads(buf.getvalue()), expected)

    def test_handle_routes_command_check_reports_success(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def routes_snapshot(self):
                return {
                    "default_route": "vector_search",
                    "valid_routes": ["vector_search"],
                    "aliases": {},
                    "priority_map": {},
                    "alias_overrides": {},
                    "priority_overrides": {},
                    "ignored_alias_overrides": [],
                    "ignored_priority_overrides": [],
                    "addon_config_paths": ["/tmp/routes.json"],
                    "addon_config_issues": [],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_routes_command(Namespace(json=False, check=True))

        output = buf.getvalue()
        self.assertIn("Route configuration check: ok", output)
        self.assertIn("ignored alias overrides: 0", output)
        self.assertIn("addon config sources: 1", output)

    def test_handle_routes_command_check_exits_nonzero_on_invalid_config(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def routes_snapshot(self):
                return {
                    "default_route": "vector_search",
                    "valid_routes": ["vector_search"],
                    "aliases": {},
                    "priority_map": {},
                    "alias_overrides": {},
                    "priority_overrides": {},
                    "ignored_alias_overrides": [{"key": "bad_alias", "route": "unknown", "reason": "unknown_route"}],
                    "ignored_priority_overrides": [],
                    "addon_config_paths": ["/tmp/routes.json"],
                    "addon_config_issues": [{"reason": "missing_file", "path": "/tmp/routes.json"}],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as exc:
            cli.handle_routes_command(Namespace(json=False, check=True))

        self.assertEqual(exc.exception.code, 1)
        output = buf.getvalue()
        self.assertIn("Route configuration check: invalid", output)
        self.assertIn("Tip: run `.venv/bin/python -m src.core.cli routes`", output)

    def test_handle_routes_command_check_supports_json_output(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def routes_snapshot(self):
                return {
                    "default_route": "vector_search",
                    "valid_routes": ["vector_search"],
                    "aliases": {},
                    "priority_map": {},
                    "alias_overrides": {},
                    "priority_overrides": {},
                    "ignored_alias_overrides": [],
                    "ignored_priority_overrides": [{"key": "bad", "route": "unknown", "reason": "unknown_route"}],
                    "addon_config_paths": [],
                    "addon_config_issues": [],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit):
            cli.handle_routes_command(Namespace(json=True, check=True))

        self.assertEqual(
            json.loads(buf.getvalue()),
            {
                "addon_config_issues": 0,
                "addon_config_sources": 0,
                "ignored_alias_overrides": 0,
                "ignored_priority_overrides": 1,
                "ok": False,
            },
        )


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
            "confirmation_filter": "continue",
            "contains_filter": "retrieval",
            "planned_database_filter": "vector_db",
            "planned_method_filter": "search_relevant_sentences",
            "min_duration_ms_filter": 200,
            "min_response_chars_filter": None,
            "max_response_chars_filter": None,
            "entries_returned": 2,
            "matching_entries_total": 2,
            "entries_considered": 2,
            "success_count": 0,
            "error_count": 2,
            "corrupt_count": 0,
            "average_duration_ms": 225.0,
            "max_duration_ms": 250,
            "average_response_chars": 42.5,
            "max_response_chars": 50,
            "latest_status": "error",
            "latest_question": "Why did retrieval fail?",
            "latest_source": "cli.query",
            "latest_error": "retrieval unavailable",
            "latest_response_preview": "Could not retrieve Porter evidence right now.",
            "error_breakdown": [{"error": "retrieval unavailable", "count": 1}, {"error": "timeout", "count": 1}],
            "source_breakdown": [{"source": "cli.query", "count": 2}],
            "confirmation_breakdown": [{"confirmation": "continue", "count": 2}],
            "query_plan_database_breakdown": [{"database": "vector_db", "count": 2}],
            "query_plan_method_breakdown": [{"method": "search_relevant_sentences", "count": 2}],
            "entries": [
                {"status": "error", "source": "cli.query", "confirmation": "continue", "question": "Why did retrieval fail?", "duration_ms": 250, "response_chars": 50, "response_preview": "Could not retrieve Porter evidence right now.", "error": "retrieval unavailable", "query_plan_databases": ["vector_db"], "query_plan_methods": ["search_relevant_sentences"]},
                {"status": "error", "source": "cli.query", "confirmation": "continue", "question": "Why did ranking fail?", "duration_ms": 200, "response_chars": 35, "response_preview": "Ranking failed because the request timed out.", "error": "timeout", "query_plan_databases": ["vector_db"], "query_plan_methods": ["search_relevant_sentences"]},
            ],
        }

        class ExpectedKernel:
            def query_history_snapshot(self, limit=10, status="all", source="all", confirmation="all", satisfaction="all", since_hours=None, contains="", question_contains="", error_contains="", response_contains="", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None):
                assert limit == 5
                assert status == "error"
                assert source == "cli.query"
                assert confirmation == "continue"
                assert since_hours is None
                assert contains == "retrieval"
                assert question_contains == ""
                assert error_contains == ""
                assert response_contains == ""
                assert planned_database == "vector_db"
                assert planned_method == "search_relevant_sentences"
                assert min_duration_ms == 200
                assert max_duration_ms == 260
                assert min_response_chars is None
                assert max_response_chars is None
                return expected

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_query_history_command(Namespace(limit=5, status="error", source="cli.query", confirmation="continue", contains="retrieval", planned_database="vector_db", planned_method="search_relevant_sentences", planned_route="all", min_duration_ms=200, min_response_chars=None, max_response_chars=None, json=True, since_hours=None))

        self.assertEqual(json.loads(buf.getvalue()), expected)

    def test_handle_query_history_command_passes_specific_text_filters_to_kernel(self):
        cli = _load_cli_module()
        expected = {
            "log_file": "data/query_history.jsonl",
            "requested_limit": 3,
            "status_filter": "all",
            "source_filter": "all",
            "confirmation_filter": "all",
            "satisfaction_filter": "all",
            "contains_filter": "",
            "question_contains_filter": "why did",
            "error_contains_filter": "retrieval",
            "response_contains_filter": "porter evidence",
            "planned_database_filter": "all",
            "planned_method_filter": "all",
            "planned_route_filter": "all",
            "min_duration_ms_filter": None,
            "max_duration_ms_filter": None,
            "min_response_chars_filter": None,
            "max_response_chars_filter": None,
            "entries_returned": 1,
            "matching_entries_total": 1,
            "entries_considered": 1,
            "success_count": 0,
            "error_count": 1,
            "corrupt_count": 0,
            "average_duration_ms": 250.0,
            "max_duration_ms": 250,
            "average_response_chars": 47.0,
            "max_response_chars": 47,
            "latest_status": "error",
            "latest_question": "Why did retrieval fail?",
            "latest_source": "cli.query",
            "latest_error": "retrieval unavailable",
            "latest_response_preview": "Could not retrieve Porter evidence right now.",
            "error_breakdown": [{"error": "retrieval unavailable", "count": 1}],
            "source_breakdown": [{"source": "cli.query", "count": 1}],
            "confirmation_breakdown": [{"confirmation": "continue", "count": 1}],
            "query_plan_database_breakdown": [],
            "query_plan_method_breakdown": [],
            "query_plan_route_breakdown": [],
            "entries": [
                {"status": "error", "source": "cli.query", "confirmation": "continue", "question": "Why did retrieval fail?", "duration_ms": 250, "response_chars": 47, "response_preview": "Could not retrieve Porter evidence right now.", "error": "retrieval unavailable"},
            ],
        }

        class ExpectedKernel:
            def query_history_snapshot(self, **kwargs):
                assert kwargs["contains"] == ""
                assert kwargs["question_contains"] == "why did"
                assert kwargs["error_contains"] == "retrieval"
                assert kwargs["response_contains"] == "porter evidence"
                return expected

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_query_history_command(Namespace(limit=3, status="all", source="all", confirmation="all", satisfaction="all", contains="", question_contains="why did", error_contains="retrieval", response_contains="porter evidence", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None, json=False, since_hours=None))

        output = buf.getvalue()
        self.assertIn("Question filter: why did", output)
        self.assertIn("Error filter: retrieval", output)
        self.assertIn("Response filter: porter evidence", output)
        self.assertIn("Why did retrieval fail?", output)


    def test_handle_query_history_command_text_output_is_actionable(self):
        cli = _load_cli_module()
        expected = {
            "log_file": "data/query_history.jsonl",
            "requested_limit": 2,
            "status_filter": "all",
            "source_filter": "all",
            "confirmation_filter": "all",
            "contains_filter": "",
            "planned_database_filter": "all",
            "planned_method_filter": "all",
            "planned_route_filter": "all",
            "min_duration_ms_filter": 200,
            "max_duration_ms_filter": 260,
            "min_response_chars_filter": None,
            "max_response_chars_filter": None,
            "entries_returned": 2,
            "matching_entries_total": 2,
            "entries_considered": 2,
            "success_count": 1,
            "error_count": 1,
            "corrupt_count": 0,
            "average_duration_ms": 175.0,
            "max_duration_ms": 250,
            "average_response_chars": 41.5,
            "max_response_chars": 47,
            "latest_status": "error",
            "latest_question": "Why did retrieval fail?",
            "latest_source": "cli.query",
            "latest_error": "retrieval unavailable",
            "latest_response_preview": "Could not retrieve Porter evidence right now.",
            "error_breakdown": [{"error": "retrieval unavailable", "count": 1}],
            "source_breakdown": [{"source": "cli.query", "count": 2}],
            "confirmation_breakdown": [{"confirmation": "continue", "count": 2}],
            "query_plan_database_breakdown": [{"database": "vector_db", "count": 2}, {"database": "pdf_db", "count": 1}],
            "query_plan_method_breakdown": [{"method": "search_relevant_sentences", "count": 2}, {"method": "get_full_pdf_content", "count": 1}],
            "query_plan_route_breakdown": [{"route": "vector_search", "count": 2}, {"route": "pdf_analysis", "count": 1}],
            "entries": [
                {"status": "error", "source": "cli.query", "confirmation": "continue", "question": "Why did retrieval fail?", "duration_ms": 250, "response_chars": 47, "response_preview": "Could not retrieve Porter evidence right now.", "error": "retrieval unavailable", "query_plan_databases": ["vector_db"], "query_plan_methods": ["search_relevant_sentences"], "query_plan_routes": ["vector_search"]},
                {"status": "success", "source": "cli.query", "confirmation": "continue", "question": "Summarize Porter", "duration_ms": 100, "response_chars": 36, "response_preview": "Porter links firm advantage to activity fit.", "query_plan_databases": ["vector_db", "pdf_db"], "query_plan_methods": ["search_relevant_sentences", "get_full_pdf_content"], "query_plan_routes": ["vector_search", "pdf_analysis"]},
            ],
        }

        class ExpectedKernel:
            def query_history_snapshot(self, limit=10, status="all", source="all", confirmation="all", satisfaction="all", since_hours=None, contains="", question_contains="", error_contains="", response_contains="", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None):
                assert satisfaction == "all"
                assert limit == 2
                assert status == "all"
                assert source == "all"
                assert confirmation == "all"
                assert since_hours is None
                assert contains == ""
                assert question_contains == ""
                assert error_contains == ""
                assert response_contains == ""
                assert planned_database == "all"
                assert planned_method == "all"
                assert planned_route == "all"
                assert min_duration_ms == 200
                assert max_duration_ms == 260
                assert min_response_chars is None
                assert max_response_chars is None
                return expected

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_query_history_command(Namespace(limit=2, status="all", source="all", confirmation="all", contains="", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=200, max_duration_ms=260, min_response_chars=None, max_response_chars=None, json=False, since_hours=None))

        output = buf.getvalue()
        self.assertIn("Status filter: all", output)
        self.assertIn("Source filter: all", output)
        self.assertIn("Matching entries before limit: 2", output)
        self.assertIn("Minimum duration filter: 200 ms", output)
        self.assertIn("Maximum duration filter: 260 ms", output)
        self.assertIn("Successful queries: 1", output)
        self.assertIn("Failed queries: 1", output)
        self.assertIn("Average duration: 175.0 ms", output)
        self.assertIn("Slowest query: 250 ms", output)
        self.assertIn("Average response size: 41.5 chars", output)
        self.assertIn("Longest response: 47 chars", output)
        self.assertIn("Latest query: error [cli.query] - Why did retrieval fail?", output)
        self.assertIn("Latest error: retrieval unavailable", output)
        self.assertIn("Latest response preview: Could not retrieve Porter evidence right now.", output)
        self.assertIn("Sources:", output)
        self.assertIn("Confirmations:", output)
        self.assertIn("Errors:", output)
        self.assertIn("  - retrieval unavailable: 1", output)
        self.assertIn("Planned databases:", output)
        self.assertIn("Planned methods:", output)
        self.assertIn("Planned routes:", output)
        self.assertIn("[error] {cli.query} (250 ms) Why did retrieval fail?", output)
        self.assertIn("error: retrieval unavailable", output)
        self.assertIn("response chars: 47", output)
        self.assertIn("response: Could not retrieve Porter evidence right now.", output)
        self.assertIn("plan: routes=vector_search | db=vector_db | methods=search_relevant_sentences", output)
        self.assertIn("[success] {cli.query} (100 ms) Summarize Porter", output)
        self.assertIn("response: Porter links firm advantage to activity fit.", output)
        self.assertIn("plan: routes=vector_search, pdf_analysis | db=vector_db, pdf_db | methods=search_relevant_sentences, get_full_pdf_content", output)

    def test_handle_query_history_command_labels_matching_window_breakdowns(self):
        cli = _load_cli_module()
        expected = {
            "log_file": "data/query_history.jsonl",
            "requested_limit": 1,
            "status_filter": "all",
            "source_filter": "all",
            "confirmation_filter": "all",
            "satisfaction_filter": "all",
            "contains_filter": "",
            "planned_database_filter": "all",
            "planned_method_filter": "all",
            "planned_route_filter": "all",
            "entries_returned": 1,
            "matching_entries_total": 3,
            "entries_considered": 1,
            "matching_entries_considered": 3,
            "success_count": 1,
            "error_count": 0,
            "matching_success_count": 2,
            "matching_error_count": 1,
            "corrupt_count": 0,
            "source_breakdown": [{"source": "cli.query", "count": 1}],
            "matching_source_breakdown": [
                {"source": "cli.query", "count": 2},
                {"source": "openclaw.facade.query", "count": 1},
            ],
            "confirmation_breakdown": [{"confirmation": "continue", "count": 1}],
            "matching_confirmation_breakdown": [
                {"confirmation": "continue", "count": 2},
                {"confirmation": "expand", "count": 1},
            ],
            "satisfaction_breakdown": [{"satisfaction": "satisfied", "count": 1}],
            "matching_satisfaction_breakdown": [
                {"satisfaction": "satisfied", "count": 2},
                {"satisfaction": "dissatisfied", "count": 1},
            ],
            "query_plan_database_breakdown": [{"database": "vector_db", "count": 1}],
            "matching_query_plan_database_breakdown": [
                {"database": "vector_db", "count": 2},
                {"database": "graph_db", "count": 1},
            ],
            "query_plan_method_breakdown": [{"method": "search_relevant_sentences", "count": 1}],
            "matching_query_plan_method_breakdown": [
                {"method": "search_relevant_sentences", "count": 2},
                {"method": "get_papers_citing_paper", "count": 1},
            ],
            "query_plan_route_breakdown": [{"route": "vector_search", "count": 1}],
            "matching_query_plan_route_breakdown": [
                {"route": "vector_search", "count": 2},
                {"route": "graph_analysis", "count": 1},
            ],
            "entries": [
                {"status": "success", "source": "cli.query", "confirmation": "continue", "question": "Recent vector query"},
            ],
        }

        class ExpectedKernel:
            def query_history_snapshot(self, **kwargs):
                return expected

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_query_history_command(Namespace(limit=1, status="all", source="all", confirmation="all", satisfaction="all", contains="", question_contains="", error_contains="", response_contains="", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None, json=False, since_hours=None))

        output = buf.getvalue()
        self.assertIn("Matching-window sources:", output)
        self.assertIn("  - openclaw.facade.query: 1", output)
        self.assertIn("Matching-window confirmations:", output)
        self.assertIn("  - expand: 1", output)
        self.assertIn("Matching-window satisfaction:", output)
        self.assertIn("  - dissatisfied: 1", output)
        self.assertIn("Matching-window planned databases:", output)
        self.assertIn("  - graph_db: 1", output)
        self.assertIn("Matching-window planned methods:", output)
        self.assertIn("  - get_papers_citing_paper: 1", output)
        self.assertIn("Matching-window planned routes:", output)
        self.assertIn("  - graph_analysis: 1", output)


class CliQueryHistoryCheckCommandTests(unittest.TestCase):
    def test_handle_query_history_command_check_empty_reports_success(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def query_history_snapshot(self, **kwargs):
                return {
                    "log_file": "data/query_history.jsonl",
                    "requested_limit": kwargs.get("limit", 10),
                    "status_filter": kwargs.get("status", "all"),
                    "source_filter": kwargs.get("source", "all"),
                    "confirmation_filter": kwargs.get("confirmation", "all"),
                    "contains_filter": kwargs.get("contains", ""),
                    "question_contains_filter": kwargs.get("question_contains", ""),
                    "error_contains_filter": kwargs.get("error_contains", ""),
                    "response_contains_filter": kwargs.get("response_contains", ""),
                    "planned_database_filter": kwargs.get("planned_database", "all"),
                    "planned_method_filter": kwargs.get("planned_method", "all"),
                    "planned_route_filter": kwargs.get("planned_route", "all"),
                    "min_duration_ms_filter": kwargs.get("min_duration_ms"),
                    "max_duration_ms_filter": kwargs.get("max_duration_ms"),
                    "min_response_chars_filter": kwargs.get("min_response_chars"),
                    "max_response_chars_filter": kwargs.get("max_response_chars"),
                    "since_hours": kwargs.get("since_hours"),
                    "matching_entries_total": 0,
                    "entries_returned": 0,
                    "entries_considered": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "success_rate": None,
                    "error_rate": None,
                    "corrupt_count": 0,
                    "average_duration_ms": None,
                    "max_duration_ms": None,
                    "latest_status": None,
                    "latest_question": None,
                    "latest_source": None,
                    "latest_error": None,
                    "source_breakdown": [],
                    "confirmation_breakdown": [],
                    "query_plan_database_breakdown": [],
                    "query_plan_method_breakdown": [],
                    "entries": [],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_query_history_command(Namespace(limit=5, status="error", source="cli.query", confirmation="all", contains="timeout", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=300, max_duration_ms=900, min_response_chars=20, max_response_chars=120, json=False, since_hours=24, check_empty=True, check_max_errors=None, check_max_error_rate=None))

        output = buf.getvalue()
        self.assertIn("Query history check: ok", output)
        self.assertIn("matching entries: 0", output)
        self.assertIn("error count: 0", output)
        self.assertIn("status filter: error", output)
        self.assertIn("minimum duration: 300 ms", output)
        self.assertIn("maximum duration: 900 ms", output)
        self.assertIn("minimum response size: 20 chars", output)
        self.assertIn("maximum response size: 120 chars", output)
        self.assertIn("time window: last 24 hours", output)

    def test_handle_query_history_command_check_empty_exits_nonzero_when_matches_exist(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def query_history_snapshot(self, **kwargs):
                return {
                    "log_file": "data/query_history.jsonl",
                    "requested_limit": kwargs.get("limit", 10),
                    "status_filter": kwargs.get("status", "all"),
                    "source_filter": kwargs.get("source", "all"),
                    "confirmation_filter": kwargs.get("confirmation", "all"),
                    "contains_filter": kwargs.get("contains", ""),
                    "question_contains_filter": kwargs.get("question_contains", ""),
                    "error_contains_filter": kwargs.get("error_contains", ""),
                    "response_contains_filter": kwargs.get("response_contains", ""),
                    "planned_database_filter": kwargs.get("planned_database", "all"),
                    "planned_method_filter": kwargs.get("planned_method", "all"),
                    "planned_route_filter": kwargs.get("planned_route", "all"),
                    "min_duration_ms_filter": kwargs.get("min_duration_ms"),
                    "max_duration_ms_filter": kwargs.get("max_duration_ms"),
                    "min_response_chars_filter": kwargs.get("min_response_chars"),
                    "max_response_chars_filter": kwargs.get("max_response_chars"),
                    "since_hours": kwargs.get("since_hours"),
                    "matching_entries_total": 2,
                    "entries_returned": 2,
                    "entries_considered": 2,
                    "success_count": 0,
                    "error_count": 2,
                    "success_rate": 0.0,
                    "error_rate": 1.0,
                    "corrupt_count": 0,
                    "average_duration_ms": 123.0,
                    "max_duration_ms": 150,
                    "latest_status": "error",
                    "latest_question": "recent failure",
                    "latest_source": "cli.query",
                    "latest_error": "timeout",
                    "source_breakdown": [],
                    "confirmation_breakdown": [],
                    "query_plan_database_breakdown": [],
                    "query_plan_method_breakdown": [],
                    "entries": [],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as exc:
            cli.handle_query_history_command(Namespace(limit=5, status="error", source="cli.query", confirmation="all", contains="", planned_database="vector_db", planned_method="search_relevant_sentences", planned_route="vector_search", min_duration_ms=250, max_duration_ms=600, min_response_chars=10, max_response_chars=80, json=True, since_hours=12, check_empty=True, check_max_errors=None, check_max_error_rate=None))

        self.assertEqual(exc.exception.code, 1)
        self.assertEqual(
            json.loads(buf.getvalue()),
            {
                "check_empty": True,
                "check_max_empty_query_plans": None,
                "check_max_duration_ms": None,
                "check_max_error_rate": None,
                "check_max_errors": None,
                "check_max_no_planned_routes": None,
                "check_min_response_chars": None,
                "check_min_success_rate": None,
                "confirmation_filter": "all",
                "contains_filter": "",
                "empty_query_plan_count": 0,
                "error_count": 2,
                "error_rate": 1.0,
                "success_rate": 0.0,
                "failure_reasons": ["not empty"],
                "max_duration_ms": 150,
                "question_contains_filter": "",
                "error_contains_filter": "",
                "response_contains_filter": "",
                "matching_entries_total": 2,
                "min_duration_ms_filter": 250,
                "max_duration_ms_filter": 600,
                "min_response_chars_filter": 10,
                "max_response_chars_filter": 80,
                "no_planned_route_count": 0,
                "ok": False,
                "planned_database_filter": "vector_db",
                "planned_method_filter": "search_relevant_sentences",
                "planned_route_filter": "vector_search",
                "satisfaction_filter": "all",
                "shortest_success_response_chars": None,
                "since_hours": 12,
                "source_filter": "cli.query",
                "status_filter": "error",
            },
        )

    def test_handle_query_history_command_check_max_error_rate_exits_when_threshold_is_exceeded(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def query_history_snapshot(self, **kwargs):
                return {
                    "log_file": "data/query_history.jsonl",
                    "requested_limit": kwargs.get("limit", 10),
                    "status_filter": kwargs.get("status", "all"),
                    "source_filter": kwargs.get("source", "all"),
                    "confirmation_filter": kwargs.get("confirmation", "all"),
                    "satisfaction_filter": kwargs.get("satisfaction", "all"),
                    "contains_filter": kwargs.get("contains", ""),
                    "question_contains_filter": kwargs.get("question_contains", ""),
                    "error_contains_filter": kwargs.get("error_contains", ""),
                    "response_contains_filter": kwargs.get("response_contains", ""),
                    "planned_database_filter": kwargs.get("planned_database", "all"),
                    "planned_method_filter": kwargs.get("planned_method", "all"),
                    "planned_route_filter": kwargs.get("planned_route", "all"),
                    "min_duration_ms_filter": kwargs.get("min_duration_ms"),
                    "max_duration_ms_filter": kwargs.get("max_duration_ms"),
                    "min_response_chars_filter": kwargs.get("min_response_chars"),
                    "max_response_chars_filter": kwargs.get("max_response_chars"),
                    "since_hours": kwargs.get("since_hours"),
                    "matching_entries_total": 4,
                    "entries_returned": 4,
                    "entries_considered": 4,
                    "success_count": 1,
                    "error_count": 3,
                    "success_rate": 0.25,
                    "error_rate": 0.75,
                    "corrupt_count": 0,
                    "entries": [],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as exc:
            cli.handle_query_history_command(Namespace(limit=10, status="all", source="all", confirmation="all", satisfaction="all", contains="", question_contains="", error_contains="", response_contains="", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None, json=False, since_hours=6, check_empty=False, check_max_errors=3, check_max_error_rate=0.5))

        self.assertEqual(exc.exception.code, 1)
        output = buf.getvalue()
        self.assertIn("Query history check: error rate too high", output)
        self.assertIn("error count: 3", output)
        self.assertIn("error rate: 0.75", output)
        self.assertIn("max errors check: 3", output)
        self.assertIn("max error rate check: 0.5", output)

    def test_handle_query_history_command_check_min_success_rate_exits_when_threshold_is_missed(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def query_history_snapshot(self, **kwargs):
                return {
                    "log_file": "data/query_history.jsonl",
                    "requested_limit": kwargs.get("limit", 10),
                    "status_filter": kwargs.get("status", "all"),
                    "source_filter": kwargs.get("source", "all"),
                    "confirmation_filter": kwargs.get("confirmation", "all"),
                    "satisfaction_filter": kwargs.get("satisfaction", "all"),
                    "contains_filter": kwargs.get("contains", ""),
                    "question_contains_filter": kwargs.get("question_contains", ""),
                    "error_contains_filter": kwargs.get("error_contains", ""),
                    "response_contains_filter": kwargs.get("response_contains", ""),
                    "planned_database_filter": kwargs.get("planned_database", "all"),
                    "planned_method_filter": kwargs.get("planned_method", "all"),
                    "planned_route_filter": kwargs.get("planned_route", "all"),
                    "min_duration_ms_filter": kwargs.get("min_duration_ms"),
                    "max_duration_ms_filter": kwargs.get("max_duration_ms"),
                    "min_response_chars_filter": kwargs.get("min_response_chars"),
                    "max_response_chars_filter": kwargs.get("max_response_chars"),
                    "since_hours": kwargs.get("since_hours"),
                    "matching_entries_total": 10,
                    "matching_error_count": 3,
                    "matching_error_rate": 0.3,
                    "matching_success_rate": 0.7,
                    "entries": [],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as exc:
            cli.handle_query_history_command(Namespace(limit=10, status="all", source="all", confirmation="all", satisfaction="all", contains="", question_contains="", error_contains="", response_contains="", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None, json=False, since_hours=24, check_empty=False, check_max_errors=None, check_max_error_rate=None, check_min_success_rate=0.9))

        self.assertEqual(exc.exception.code, 1)
        output = buf.getvalue()
        self.assertIn("Query history check: success rate too low", output)
        self.assertIn("success rate: 0.7", output)
        self.assertIn("min success rate check: 0.9", output)

    def test_handle_query_history_command_checks_full_matching_window_not_display_limit(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def query_history_snapshot(self, **kwargs):
                return {
                    "log_file": "data/query_history.jsonl",
                    "requested_limit": kwargs.get("limit", 10),
                    "status_filter": kwargs.get("status", "all"),
                    "source_filter": kwargs.get("source", "all"),
                    "confirmation_filter": kwargs.get("confirmation", "all"),
                    "satisfaction_filter": kwargs.get("satisfaction", "all"),
                    "contains_filter": kwargs.get("contains", ""),
                    "question_contains_filter": kwargs.get("question_contains", ""),
                    "error_contains_filter": kwargs.get("error_contains", ""),
                    "response_contains_filter": kwargs.get("response_contains", ""),
                    "planned_database_filter": kwargs.get("planned_database", "all"),
                    "planned_method_filter": kwargs.get("planned_method", "all"),
                    "planned_route_filter": kwargs.get("planned_route", "all"),
                    "min_duration_ms_filter": kwargs.get("min_duration_ms"),
                    "max_duration_ms_filter": kwargs.get("max_duration_ms"),
                    "min_response_chars_filter": kwargs.get("min_response_chars"),
                    "max_response_chars_filter": kwargs.get("max_response_chars"),
                    "since_hours": kwargs.get("since_hours"),
                    "matching_entries_total": 12,
                    "matching_error_count": 4,
                    "matching_error_rate": 0.3333,
                    "entries_returned": 1,
                    "entries_considered": 1,
                    "success_count": 1,
                    "error_count": 0,
                    "success_rate": 1.0,
                    "error_rate": 0.0,
                    "corrupt_count": 0,
                    "entries": [],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as exc:
            cli.handle_query_history_command(Namespace(limit=1, status="all", source="all", confirmation="all", satisfaction="all", contains="", question_contains="", error_contains="", response_contains="", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None, json=False, since_hours=None, check_empty=False, check_max_errors=3, check_max_error_rate=0.5))

        self.assertEqual(exc.exception.code, 1)
        output = buf.getvalue()
        self.assertIn("Query history check: too many errors", output)
        self.assertIn("matching entries: 12", output)
        self.assertIn("error count: 4", output)
        self.assertIn("error rate: 0.3333", output)

    def test_handle_query_history_command_quality_gates_use_full_matching_window(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def query_history_snapshot(self, **kwargs):
                return {
                    "log_file": "data/query_history.jsonl",
                    "requested_limit": kwargs.get("limit", 10),
                    "status_filter": kwargs.get("status", "all"),
                    "source_filter": kwargs.get("source", "all"),
                    "confirmation_filter": kwargs.get("confirmation", "all"),
                    "satisfaction_filter": kwargs.get("satisfaction", "all"),
                    "contains_filter": kwargs.get("contains", ""),
                    "question_contains_filter": kwargs.get("question_contains", ""),
                    "error_contains_filter": kwargs.get("error_contains", ""),
                    "response_contains_filter": kwargs.get("response_contains", ""),
                    "planned_database_filter": kwargs.get("planned_database", "all"),
                    "planned_method_filter": kwargs.get("planned_method", "all"),
                    "planned_route_filter": kwargs.get("planned_route", "all"),
                    "min_duration_ms_filter": kwargs.get("min_duration_ms"),
                    "max_duration_ms_filter": kwargs.get("max_duration_ms"),
                    "min_response_chars_filter": kwargs.get("min_response_chars"),
                    "max_response_chars_filter": kwargs.get("max_response_chars"),
                    "since_hours": kwargs.get("since_hours"),
                    "matching_entries_total": 7,
                    "matching_error_count": 0,
                    "matching_error_rate": 0.0,
                    "matching_max_duration_ms": 1250,
                    "matching_min_success_response_chars": 18,
                    "entries_returned": 1,
                    "entries_considered": 1,
                    "success_count": 1,
                    "error_count": 0,
                    "success_rate": 1.0,
                    "error_rate": 0.0,
                    "corrupt_count": 0,
                    "entries": [],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as exc:
            cli.handle_query_history_command(Namespace(limit=1, status="all", source="all", confirmation="all", satisfaction="all", contains="", question_contains="", error_contains="", response_contains="", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None, json=False, since_hours=None, check_empty=False, check_max_errors=None, check_max_error_rate=None, check_max_duration_ms=1000, check_min_response_chars=50))

        self.assertEqual(exc.exception.code, 1)
        output = buf.getvalue()
        self.assertIn("Query history check: query too slow, response too short", output)
        self.assertIn("slowest query: 1250 ms", output)
        self.assertIn("shortest successful response: 18 chars", output)
        self.assertIn("max duration check: 1000 ms", output)
        self.assertIn("minimum successful response check: 50 chars", output)

    def test_handle_query_history_command_route_plan_quality_gates_use_full_matching_window(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def query_history_snapshot(self, **kwargs):
                return {
                    "log_file": "data/query_history.jsonl",
                    "requested_limit": kwargs.get("limit", 10),
                    "status_filter": kwargs.get("status", "all"),
                    "source_filter": kwargs.get("source", "all"),
                    "confirmation_filter": kwargs.get("confirmation", "all"),
                    "satisfaction_filter": kwargs.get("satisfaction", "all"),
                    "contains_filter": kwargs.get("contains", ""),
                    "question_contains_filter": kwargs.get("question_contains", ""),
                    "error_contains_filter": kwargs.get("error_contains", ""),
                    "response_contains_filter": kwargs.get("response_contains", ""),
                    "planned_database_filter": kwargs.get("planned_database", "all"),
                    "planned_method_filter": kwargs.get("planned_method", "all"),
                    "planned_route_filter": kwargs.get("planned_route", "all"),
                    "min_duration_ms_filter": kwargs.get("min_duration_ms"),
                    "max_duration_ms_filter": kwargs.get("max_duration_ms"),
                    "min_response_chars_filter": kwargs.get("min_response_chars"),
                    "max_response_chars_filter": kwargs.get("max_response_chars"),
                    "since_hours": kwargs.get("since_hours"),
                    "matching_entries_total": 9,
                    "matching_error_count": 0,
                    "matching_error_rate": 0.0,
                    "matching_success_rate": 1.0,
                    "matching_empty_query_plan_count": 2,
                    "matching_no_planned_route_count": 3,
                    "entries_returned": 1,
                    "entries_considered": 1,
                    "success_count": 1,
                    "error_count": 0,
                    "success_rate": 1.0,
                    "error_rate": 0.0,
                    "corrupt_count": 0,
                    "entries": [],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as exc:
            cli.handle_query_history_command(Namespace(limit=1, status="all", source="all", confirmation="all", satisfaction="all", contains="", question_contains="", error_contains="", response_contains="", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None, json=False, since_hours=None, check_empty=False, check_max_errors=None, check_max_error_rate=None, check_max_empty_query_plans=1, check_max_no_planned_routes=2))

        self.assertEqual(exc.exception.code, 1)
        output = buf.getvalue()
        self.assertIn("Query history check: too many empty query plans, too many entries without planned routes", output)
        self.assertIn("empty query plans: 2", output)
        self.assertIn("entries without planned routes: 3", output)
        self.assertIn("max empty query plans check: 1", output)
        self.assertIn("max entries without planned routes check: 2", output)


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
            def query_history_snapshot(self, limit=10, status="all", source="all", confirmation="all", satisfaction="all", since_hours=None, contains="", question_contains="", error_contains="", response_contains="", planned_database="all", planned_method="all", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None):
                assert satisfaction == "all"
                assert limit == 5
                assert status == "error"
                assert source == "openclaw.facade.query"
                assert confirmation == "expand"
                assert since_hours == 24
                assert contains == ""
                assert question_contains == ""
                assert error_contains == ""
                assert response_contains == ""
                assert planned_database == "pdf_db"
                assert planned_method == "get_full_pdf_content"
                return {
                    "log_file": "data/query_history.jsonl",
                    "requested_limit": limit,
                    "status_filter": status,
                    "since_hours": since_hours,
                    "source_filter": source,
                    "confirmation_filter": confirmation,
                    "planned_database_filter": "pdf_db",
                    "planned_method_filter": "get_full_pdf_content",
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
                    "query_plan_database_breakdown": [{"database": "vector_db", "count": 1}],
                    "query_plan_method_breakdown": [{"method": "search_relevant_sentences", "count": 1}],
                    "entries": [
                        {
                            "status": "error",
                            "source": "openclaw.facade.query",
                            "confirmation": "continue",
                            "duration_ms": 123,
                            "question": "recent failure",
                            "timestamp": 1_800_000_000,
                            "error": "timeout",
                            "query_plan_databases": ["vector_db"],
                            "query_plan_methods": ["search_relevant_sentences"],
                        }
                    ],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_query_history_command(Namespace(limit=5, status="error", source="openclaw.facade.query", confirmation="expand", satisfaction="all", since_hours=24, contains="", planned_database="pdf_db", planned_method="get_full_pdf_content", planned_route="all", min_duration_ms=None, max_duration_ms=None, min_response_chars=None, max_response_chars=None, json=False))

        output = buf.getvalue()
        self.assertIn("Time window: last 24 hours", output)
        self.assertIn("Source filter: openclaw.facade.query", output)
        self.assertIn("Confirmation filter: expand", output)
        self.assertIn("Planned database filter: pdf_db", output)
        self.assertIn("Planned method filter: get_full_pdf_content", output)
        self.assertIn("recent failure", output)
        self.assertIn("timeout", output)


class CliPendingCitationsCommandTests(unittest.TestCase):
    def test_handle_list_pending_citations_command_supports_json_output(self):
        cli = _load_cli_module()
        expected = {
            "requested_limit": 3,
            "total_stub_papers": 2,
            "network_stats": {
                "total_papers": 5,
                "uploaded_papers": 3,
                "stub_papers": 2,
                "total_citation_relations": 7,
            },
            "stub_papers": [
                {
                    "paper_id": "porter_1980",
                    "title": "Competitive Strategy",
                    "authors": ["Michael Porter"],
                    "year": 1980,
                    "cited_by_count": 4,
                }
            ],
        }

        class ExpectedKernel:
            def list_pending_citations_snapshot(self, limit=10):
                assert limit == 3
                return expected

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_list_pending_citations_command(Namespace(limit=3, json=True))

        payload = json.loads(buf.getvalue())
        self.assertEqual(payload, expected)

    def test_handle_list_pending_citations_command_renders_text_output(self):
        cli = _load_cli_module()

        class ExpectedKernel:
            def list_pending_citations_snapshot(self, limit=10):
                assert limit == 2
                return {
                    "requested_limit": 2,
                    "total_stub_papers": 1,
                    "network_stats": {
                        "total_papers": 5,
                        "uploaded_papers": 4,
                        "stub_papers": 1,
                        "total_citation_relations": 8,
                    },
                    "stub_papers": [
                        {
                            "paper_id": "porter_1980",
                            "title": "Competitive Strategy",
                            "authors": ["Michael Porter"],
                            "year": 1980,
                            "cited_by_count": 4,
                        }
                    ],
                }

        cli.CiteWeaveKernel = ExpectedKernel

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.handle_list_pending_citations_command(Namespace(limit=2, json=False))

        output = buf.getvalue()
        self.assertIn("Pending citations available: 1", output)
        self.assertIn("Competitive Strategy (1980)", output)
        self.assertIn("cited by: 4", output)
        self.assertIn("authors: Michael Porter", output)
        self.assertIn("paper_id: porter_1980", output)


class RepoHygieneTests(unittest.TestCase):
    def test_gitignore_explicitly_ignores_test_files_runtime_artifacts(self):
        gitignore = (CLI_PATH.parents[2] / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("test_files/*.", gitignore)
        self.assertIn("test_files/*", gitignore)
        self.assertIn("!test_files/README.md", gitignore)


if __name__ == "__main__":
    unittest.main()
