import importlib.util
import json
import os
import sys
import tempfile
import types
import uuid
from pathlib import Path


QUERY_HISTORY_PATH = Path(__file__).resolve().parents[1] / "src" / "kernel" / "query_history.py"
SERVICE_PATH = Path(__file__).resolve().parents[1] / "src" / "kernel" / "service.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def test_query_history_recorder_appends_jsonl_entries():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        recorder.record({"question": "What cites Porter (1980)?", "status": "success"})
        recorder.record({"question": "Who refutes RBV?", "status": "error"})

        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert rows == [
            {"question": "What cites Porter (1980)?", "status": "success"},
            {"question": "Who refutes RBV?", "status": "error"},
        ]


def test_query_history_summary_reports_recent_metrics_and_corrupt_rows():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_summary_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "old success", "status": "success", "duration_ms": 100}),
                    "{not-json}",
                    json.dumps({"question": "latest error", "status": "error", "duration_ms": 250}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=3)

        assert summary["status_filter"] == "all"
        assert summary["source_filter"] == "all"
        assert summary["entries_returned"] == 3
        assert summary["entries_considered"] == 2
        assert summary["success_count"] == 1
        assert summary["error_count"] == 1
        assert summary["corrupt_count"] == 1
        assert summary["average_duration_ms"] == 175.0
        assert summary["max_duration_ms"] == 250
        assert summary["latest_status"] == "error"
        assert summary["latest_question"] == "latest error"
        assert summary["latest_source"] is None
        assert summary["latest_error"] is None
        assert summary["query_plan_database_breakdown"] == []
        assert summary["query_plan_method_breakdown"] == []
        assert summary["entries"][0]["question"] == "latest error"
        assert summary["entries"][1]["status"] == "corrupt"


def test_query_history_summary_can_filter_to_errors_only():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_filtered_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "ok", "status": "success", "duration_ms": 100, "source": "cli.query"}),
                    json.dumps({"question": "broken", "status": "error", "duration_ms": 250, "error": "timeout", "source": "cli.query"}),
                    json.dumps({"question": "still broken", "status": "error", "duration_ms": 300, "error": "rate limit", "source": "openclaw.facade.query"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=2, status="error")

        assert summary["status_filter"] == "error"
        assert summary["entries_returned"] == 2
        assert summary["entries_considered"] == 2
        assert summary["success_count"] == 0
        assert summary["error_count"] == 2
        assert summary["latest_status"] == "error"
        assert summary["latest_question"] == "still broken"
        assert summary["latest_source"] == "openclaw.facade.query"
        assert summary["latest_error"] == "rate limit"
        assert summary["source_breakdown"] == [
            {"source": "openclaw.facade.query", "count": 1},
            {"source": "cli.query", "count": 1},
        ]
        assert summary["query_plan_database_breakdown"] == []
        assert summary["query_plan_method_breakdown"] == []
        assert [entry["question"] for entry in summary["entries"]] == ["still broken", "broken"]


def test_query_history_summary_can_filter_to_source_and_recent_time_window():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_recent_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        now = 1_800_000_000
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "stale", "status": "success", "duration_ms": 90, "timestamp": now - 5 * 3600, "source": "cli.query"}),
                    json.dumps({"question": "recent ok", "status": "success", "duration_ms": 110, "timestamp": now - 1800, "source": "cli.query"}),
                    json.dumps({"question": "recent error", "status": "error", "duration_ms": 220, "timestamp": now - 600, "error": "timeout", "source": "openclaw.facade.query"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=10, source="cli.query", since_hours=2, now=now)

        assert summary["since_hours"] == 2
        assert summary["source_filter"] == "cli.query"
        assert summary["entries_returned"] == 1
        assert summary["entries_considered"] == 1
        assert summary["success_count"] == 1
        assert summary["error_count"] == 0
        assert [entry["question"] for entry in summary["entries"]] == ["recent ok"]


def test_query_history_summary_can_filter_to_confirmation_mode():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_confirmation_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "base run", "status": "success", "duration_ms": 100, "confirmation": "continue", "source": "cli.query"}),
                    json.dumps({"question": "expanded run", "status": "success", "duration_ms": 150, "confirmation": "expand", "source": "cli.query"}),
                    json.dumps({"question": "refined run", "status": "error", "duration_ms": 200, "confirmation": "refine", "source": "openclaw.facade.query", "error": "timeout"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=10, confirmation="expand")

        assert summary["confirmation_filter"] == "expand"
        assert summary["entries_returned"] == 1
        assert summary["entries_considered"] == 1
        assert summary["success_count"] == 1
        assert summary["error_count"] == 0
        assert summary["latest_question"] == "expanded run"
        assert summary["confirmation_breakdown"] == [{"confirmation": "expand", "count": 1}]
        assert [entry["question"] for entry in summary["entries"]] == ["expanded run"]


def test_query_history_summary_can_filter_by_question_or_error_substring():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_contains_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "Why did retrieval fail?", "status": "error", "duration_ms": 250, "source": "cli.query", "error": "retrieval unavailable"}),
                    json.dumps({"question": "Summarize Porter", "status": "success", "duration_ms": 100, "source": "cli.query"}),
                    json.dumps({"question": "Why did ranking fail?", "status": "error", "duration_ms": 200, "source": "cli.query", "error": "timeout"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=5, contains="retrieval")

        assert summary["contains_filter"] == "retrieval"
        assert summary["matching_entries_total"] == 1
        assert summary["entries_returned"] == 1
        assert summary["entries_considered"] == 1
        assert summary["latest_question"] == "Why did retrieval fail?"
        assert [entry["question"] for entry in summary["entries"]] == ["Why did retrieval fail?"]


def test_query_history_summary_reports_query_plan_breakdowns():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_plan_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({
                        "question": "Find Porter summaries",
                        "status": "success",
                        "duration_ms": 100,
                        "source": "cli.query",
                        "query_plan_databases": ["vector_db", "pdf_db"],
                        "query_plan_methods": ["search_relevant_sentences", "get_full_pdf_content"],
                    }),
                    json.dumps({
                        "question": "Find RBV papers",
                        "status": "success",
                        "duration_ms": 120,
                        "source": "cli.query",
                        "query_plan_databases": ["vector_db"],
                        "query_plan_methods": ["search_relevant_sentences"],
                    }),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=10)

        assert summary["query_plan_database_breakdown"] == [
            {"database": "vector_db", "count": 2},
            {"database": "pdf_db", "count": 1},
        ]
        assert summary["query_plan_method_breakdown"] == [
            {"method": "search_relevant_sentences", "count": 2},
            {"method": "get_full_pdf_content", "count": 1},
        ]


def test_kernel_query_records_success_metrics_to_history_file():
    query_history = _load_module(QUERY_HISTORY_PATH, f"src.kernel.query_history_{uuid.uuid4().hex}")

    class DummyDocumentProcessor:
        pass

    class DummyResearchSystem:
        def research_question(self, question, confirmation):
            assert question == "Summarize Porter's theory"
            assert confirmation == "continue"
            return "Competitive advantage summary"

    _stub_module("src", __path__=[])
    _stub_module("src.processing", __path__=[])
    _stub_module("src.processing.pdf", __path__=[])
    _stub_module("src.processing.pdf.document_processor", DocumentProcessor=DummyDocumentProcessor)
    _stub_module("src.agents", __path__=[])
    _stub_module("src.agents.multi_agent_research_system", LangGraphResearchSystem=DummyResearchSystem)
    _stub_module("src.agents.routing", active_route_configuration=lambda: {"default_route": "vector_search"})
    _stub_module("src.kernel", __path__=[])
    _stub_module("src.kernel.batch_tracker", BatchUploadTracker=object)
    sys.modules["src.kernel.query_history"] = query_history

    service = _load_module(SERVICE_PATH, f"src.kernel.service_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        os.environ["CITEWEAVE_QUERY_HISTORY_FILE"] = str(log_path)
        try:
            kernel = service.CiteWeaveKernel()
            response = kernel.query("Summarize Porter's theory", source="cli.query")
        finally:
            os.environ.pop("CITEWEAVE_QUERY_HISTORY_FILE", None)

        assert response == "Competitive advantage summary"
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        entry = rows[0]
        assert entry["question"] == "Summarize Porter's theory"
        assert entry["confirmation"] == "continue"
        assert entry["status"] == "success"
        assert entry["source"] == "cli.query"
        assert entry["response_chars"] == len("Competitive advantage summary")
        assert entry["response_preview"] == "Competitive advantage summary"
        assert entry["satisfaction"] is None
        assert entry["query_plan_step_count"] == 0
        assert entry["query_plan_databases"] == []
        assert entry["query_plan_methods"] == []
        assert isinstance(entry["duration_ms"], int)
        assert entry["duration_ms"] >= 0


def test_kernel_query_records_query_plan_details_when_available():
    query_history = _load_module(QUERY_HISTORY_PATH, f"src.kernel.query_history_{uuid.uuid4().hex}")

    class DummyDocumentProcessor:
        pass

    class DummyResearchSystem:
        def research_question_details(self, question, confirmation):
            assert question == "Find local first routes"
            assert confirmation == "continue"
            return {
                "final_response": "Planned local-first retrieval",
                "error": None,
                "query_plan": {
                    "query_sequence": [
                        {"database": "vector_db", "method": "search_relevant_sentences"},
                        {"database": "pdf_db", "method": "get_full_pdf_content"},
                        {"database": "vector_db", "method": "search_relevant_sentences"},
                    ]
                },
            }

    _stub_module("src", __path__=[])
    _stub_module("src.processing", __path__=[])
    _stub_module("src.processing.pdf", __path__=[])
    _stub_module("src.processing.pdf.document_processor", DocumentProcessor=DummyDocumentProcessor)
    _stub_module("src.agents", __path__=[])
    _stub_module("src.agents.multi_agent_research_system", LangGraphResearchSystem=DummyResearchSystem)
    _stub_module("src.agents.routing", active_route_configuration=lambda: {"default_route": "vector_search"})
    _stub_module("src.kernel", __path__=[])
    _stub_module("src.kernel.batch_tracker", BatchUploadTracker=object)
    sys.modules["src.kernel.query_history"] = query_history

    service = _load_module(SERVICE_PATH, f"src.kernel.service_plan_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        os.environ["CITEWEAVE_QUERY_HISTORY_FILE"] = str(log_path)
        try:
            kernel = service.CiteWeaveKernel()
            response = kernel.query("Find local first routes", source="cli.query")
        finally:
            os.environ.pop("CITEWEAVE_QUERY_HISTORY_FILE", None)

        assert response == "Planned local-first retrieval"
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        entry = rows[0]
        assert entry["query_plan_step_count"] == 3
        assert entry["query_plan_databases"] == ["vector_db", "pdf_db"]
        assert entry["query_plan_methods"] == ["search_relevant_sentences", "get_full_pdf_content"]


def test_kernel_query_records_failures_before_reraising():
    query_history = _load_module(QUERY_HISTORY_PATH, f"src.kernel.query_history_{uuid.uuid4().hex}")

    class DummyDocumentProcessor:
        pass

    class DummyResearchSystem:
        def research_question(self, question, confirmation):
            raise RuntimeError("llm unavailable")

    _stub_module("src", __path__=[])
    _stub_module("src.processing", __path__=[])
    _stub_module("src.processing.pdf", __path__=[])
    _stub_module("src.processing.pdf.document_processor", DocumentProcessor=DummyDocumentProcessor)
    _stub_module("src.agents", __path__=[])
    _stub_module("src.agents.multi_agent_research_system", LangGraphResearchSystem=DummyResearchSystem)
    _stub_module("src.agents.routing", active_route_configuration=lambda: {"default_route": "vector_search"})
    _stub_module("src.kernel", __path__=[])
    _stub_module("src.kernel.batch_tracker", BatchUploadTracker=object)
    sys.modules["src.kernel.query_history"] = query_history

    service = _load_module(SERVICE_PATH, f"src.kernel.service_failure_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        os.environ["CITEWEAVE_QUERY_HISTORY_FILE"] = str(log_path)
        try:
            kernel = service.CiteWeaveKernel()
            try:
                kernel.query("Why is the model down?", source="openclaw.facade.query")
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert str(exc) == "llm unavailable"
        finally:
            os.environ.pop("CITEWEAVE_QUERY_HISTORY_FILE", None)

        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        entry = rows[0]
        assert entry["question"] == "Why is the model down?"
        assert entry["status"] == "error"
        assert entry["source"] == "openclaw.facade.query"
        assert entry["error"] == "llm unavailable"
        assert entry["response_chars"] == 0
        assert entry["response_preview"] == ""
        assert entry["satisfaction"] is None
        assert entry["query_plan_step_count"] == 0
        assert entry["query_plan_databases"] == []
        assert entry["query_plan_methods"] == []
