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
        assert summary["success_rate"] == 0.5
        assert summary["error_rate"] == 0.5
        assert summary["corrupt_count"] == 1
        assert summary["average_duration_ms"] == 175.0
        assert summary["max_duration_ms"] == 250
        assert summary["average_response_chars"] is None
        assert summary["max_response_chars"] is None
        assert summary["latest_status"] == "error"
        assert summary["latest_question"] == "latest error"
        assert summary["latest_source"] is None
        assert summary["latest_error"] is None
        assert summary["latest_response_preview"] is None
        assert summary["error_breakdown"] == []
        assert summary["query_plan_database_breakdown"] == []
        assert summary["query_plan_method_breakdown"] == []
        assert summary["query_plan_route_breakdown"] == []
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
        assert summary["success_rate"] == 0.0
        assert summary["error_rate"] == 1.0
        assert summary["latest_status"] == "error"
        assert summary["latest_question"] == "still broken"
        assert summary["latest_source"] == "openclaw.facade.query"
        assert summary["latest_error"] == "rate limit"
        assert summary["error_breakdown"] == [
            {"error": "rate limit", "count": 1},
            {"error": "timeout", "count": 1},
        ]
        assert summary["source_breakdown"] == [
            {"source": "openclaw.facade.query", "count": 1},
            {"source": "cli.query", "count": 1},
        ]
        assert summary["query_plan_database_breakdown"] == []
        assert summary["query_plan_method_breakdown"] == []
        assert summary["query_plan_route_breakdown"] == []
        assert [entry["question"] for entry in summary["entries"]] == ["still broken", "broken"]


def test_query_history_summary_reports_matching_window_metrics_independent_of_limit():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_matching_metrics_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "old error", "status": "error", "duration_ms": 100, "response_chars": 50, "error": "timeout"}),
                    json.dumps({"question": "old success", "status": "success", "duration_ms": 120, "response_chars": 100}),
                    json.dumps({"question": "new error", "status": "error", "duration_ms": 140, "response_chars": 150, "error": "llm unavailable"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=1)

        assert summary["entries_returned"] == 1
        assert summary["entries_considered"] == 1
        assert summary["error_count"] == 1
        assert summary["error_rate"] == 1.0
        assert summary["matching_entries_total"] == 3
        assert summary["matching_entries_considered"] == 3
        assert summary["matching_success_count"] == 1
        assert summary["matching_error_count"] == 2
        assert summary["matching_success_rate"] == 0.3333
        assert summary["matching_error_rate"] == 0.6667
        assert summary["matching_corrupt_count"] == 0
        assert summary["average_duration_ms"] == 140.0
        assert summary["max_duration_ms"] == 140
        assert summary["matching_average_duration_ms"] == 120.0
        assert summary["matching_max_duration_ms"] == 140
        assert summary["average_response_chars"] == 150.0
        assert summary["max_response_chars"] == 150
        assert summary["min_success_response_chars"] is None
        assert summary["matching_average_response_chars"] == 100.0
        assert summary["matching_max_response_chars"] == 150
        assert summary["matching_min_success_response_chars"] == 100


def test_query_history_summary_reports_shortest_success_response_for_quality_gates():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_success_response_min_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "terse success", "status": "success", "response_chars": 12}),
                    json.dumps({"question": "empty error", "status": "error", "response_chars": 0, "error": "timeout"}),
                    json.dumps({"question": "rich success", "status": "success", "response_chars": 240}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=1)

        assert summary["min_success_response_chars"] == 240
        assert summary["matching_min_success_response_chars"] == 12



def test_query_history_summary_reports_matching_window_breakdowns_independent_of_limit():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_matching_breakdowns_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({
                        "question": "old graph error",
                        "status": "error",
                        "duration_ms": 100,
                        "error": "neo4j unavailable",
                        "source": "cli.query",
                        "query_plan_databases": ["graph_db"],
                    }),
                    json.dumps({
                        "question": "vector ok",
                        "status": "success",
                        "duration_ms": 120,
                        "source": "openclaw.facade.query",
                        "query_plan_databases": ["vector_db"],
                    }),
                    json.dumps({
                        "question": "new graph error",
                        "status": "error",
                        "duration_ms": 140,
                        "error": "neo4j unavailable",
                        "source": "cli.query",
                        "query_plan_databases": ["graph_db"],
                    }),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=1)

        assert summary["error_breakdown"] == [{"error": "neo4j unavailable", "count": 1}]
        assert summary["matching_error_breakdown"] == [{"error": "neo4j unavailable", "count": 2}]
        assert summary["source_breakdown"] == [{"source": "cli.query", "count": 1}]
        assert summary["matching_source_breakdown"] == [
            {"source": "cli.query", "count": 2},
            {"source": "openclaw.facade.query", "count": 1},
        ]
        assert summary["query_plan_route_breakdown"] == [{"route": "graph_analysis", "count": 1}]
        assert summary["matching_query_plan_route_breakdown"] == [
            {"route": "graph_analysis", "count": 2},
            {"route": "vector_search", "count": 1},
        ]

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


def test_query_history_summary_can_filter_to_satisfaction_and_report_breakdown():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_satisfaction_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "Grounded answer", "status": "success", "duration_ms": 100, "satisfaction": True}),
                    json.dumps({"question": "Mixed answer", "status": "success", "duration_ms": 140, "satisfaction": 3}),
                    json.dumps({"question": "Bad answer", "status": "error", "duration_ms": 200, "satisfaction": "thumbs_down", "error": "not useful"}),
                    json.dumps({"question": "Unrated answer", "status": "success", "duration_ms": 90, "satisfaction": None}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=10, satisfaction="dissatisfied")

        assert summary["satisfaction_filter"] == "dissatisfied"
        assert summary["matching_entries_total"] == 1
        assert summary["entries_returned"] == 1
        assert summary["latest_question"] == "Bad answer"
        assert summary["error_breakdown"] == [{"error": "not useful", "count": 1}]
        assert summary["satisfaction_breakdown"] == [{"satisfaction": "dissatisfied", "count": 1}]
        assert [entry["question"] for entry in summary["entries"]] == ["Bad answer"]


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


def test_query_history_summary_can_filter_by_question_error_or_response_substring():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_contains_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "Why did retrieval fail?", "status": "error", "duration_ms": 250, "source": "cli.query", "error": "retrieval unavailable"}),
                    json.dumps({"question": "Summarize Porter", "status": "success", "duration_ms": 100, "source": "cli.query", "response_preview": "Competitive advantage summary", "response_chars": 29}),
                    json.dumps({"question": "Why did ranking fail?", "status": "error", "duration_ms": 200, "source": "cli.query", "error": "timeout"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=5, contains="competitive advantage")

        assert summary["contains_filter"] == "competitive advantage"
        assert summary["matching_entries_total"] == 1
        assert summary["entries_returned"] == 1
        assert summary["entries_considered"] == 1
        assert summary["latest_question"] == "Summarize Porter"
        assert summary["latest_response_preview"] == "Competitive advantage summary"
        assert [entry["question"] for entry in summary["entries"]] == ["Summarize Porter"]


def test_query_history_summary_can_filter_question_error_and_response_text_independently():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_text_filters_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "Why did retrieval fail?", "status": "error", "duration_ms": 250, "error": "retrieval unavailable", "response_preview": "Could not retrieve Porter evidence right now."}),
                    json.dumps({"question": "Summarize Porter", "status": "success", "duration_ms": 100, "response_preview": "Competitive advantage summary"}),
                    json.dumps({"question": "Why did ranking fail?", "status": "error", "duration_ms": 200, "error": "timeout", "response_preview": "Ranking failed because the request timed out."}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(
            limit=10,
            question_contains="why did",
            error_contains="retrieval",
            response_contains="porter evidence",
        )

        assert summary["question_contains_filter"] == "why did"
        assert summary["error_contains_filter"] == "retrieval"
        assert summary["response_contains_filter"] == "porter evidence"
        assert summary["matching_entries_total"] == 1
        assert summary["entries_returned"] == 1
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
        assert summary["query_plan_route_breakdown"] == [
            {"route": "vector_search", "count": 2},
            {"route": "pdf_analysis", "count": 1},
        ]



def test_query_history_summary_can_filter_by_minimum_duration():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_min_duration_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "Fast query", "status": "success", "duration_ms": 90, "source": "cli.query"}),
                    json.dumps({"question": "Slow success", "status": "success", "duration_ms": 450, "source": "cli.query"}),
                    json.dumps({"question": "Slow error", "status": "error", "duration_ms": 700, "source": "cli.query", "error": "timeout"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=10, min_duration_ms=400)

        assert summary["min_duration_ms_filter"] == 400
        assert summary["matching_entries_total"] == 2
        assert summary["entries_returned"] == 2
        assert summary["entries_considered"] == 2
        assert summary["average_duration_ms"] == 575.0
        assert summary["max_duration_ms"] == 700
        assert summary["average_response_chars"] is None
        assert summary["max_response_chars"] is None
        assert [entry["question"] for entry in summary["entries"]] == ["Slow error", "Slow success"]



def test_query_history_summary_can_filter_by_maximum_duration():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_max_duration_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "Fast answer", "status": "success", "duration_ms": 90}),
                    json.dumps({"question": "Borderline answer", "status": "success", "duration_ms": 150}),
                    json.dumps({"question": "Slow answer", "status": "error", "duration_ms": 450, "error": "timeout"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=10, max_duration_ms=150)

        assert summary["max_duration_ms_filter"] == 150
        assert summary["matching_entries_total"] == 2
        assert summary["entries_returned"] == 2
        assert summary["entries_considered"] == 2
        assert summary["average_duration_ms"] == 120.0
        assert summary["max_duration_ms"] == 150
        assert [entry["question"] for entry in summary["entries"]] == ["Borderline answer", "Fast answer"]


def test_query_history_summary_can_filter_by_response_size():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_response_size_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "Tiny answer", "status": "success", "duration_ms": 90, "source": "cli.query", "response_chars": 18, "response_preview": "Too short to trust"}),
                    json.dumps({"question": "Useful answer", "status": "success", "duration_ms": 110, "source": "cli.query", "response_chars": 140, "response_preview": "A grounded explanation with evidence."}),
                    json.dumps({"question": "Verbose answer", "status": "success", "duration_ms": 130, "source": "cli.query", "response_chars": 420, "response_preview": "A very long answer"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=10, min_response_chars=100, max_response_chars=200)

        assert summary["min_response_chars_filter"] == 100
        assert summary["max_response_chars_filter"] == 200
        assert summary["matching_entries_total"] == 1
        assert summary["entries_returned"] == 1
        assert summary["average_response_chars"] == 140.0
        assert summary["max_response_chars"] == 140
        assert summary["latest_question"] == "Useful answer"
        assert [entry["question"] for entry in summary["entries"]] == ["Useful answer"]


def test_query_history_summary_can_filter_by_planned_database_and_method():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_plan_filters_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({
                        "question": "Vector and PDF query",
                        "status": "success",
                        "duration_ms": 100,
                        "source": "cli.query",
                        "query_plan_databases": ["vector_db", "pdf_db"],
                        "query_plan_methods": ["search_relevant_sentences", "get_full_pdf_content"],
                    }),
                    json.dumps({
                        "question": "Vector only query",
                        "status": "success",
                        "duration_ms": 120,
                        "source": "cli.query",
                        "query_plan_databases": ["vector_db"],
                        "query_plan_methods": ["search_relevant_sentences"],
                    }),
                    json.dumps({
                        "question": "Graph query",
                        "status": "success",
                        "duration_ms": 140,
                        "source": "cli.query",
                        "query_plan_databases": ["graph_db"],
                        "query_plan_methods": ["get_graph_context"],
                    }),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=10, planned_database="pdf_db", planned_method="get_full_pdf_content")

        assert summary["planned_database_filter"] == "pdf_db"
        assert summary["planned_method_filter"] == "get_full_pdf_content"
        assert summary["matching_entries_total"] == 1
        assert summary["entries_returned"] == 1
        assert summary["latest_question"] == "Vector and PDF query"
        assert [entry["question"] for entry in summary["entries"]] == ["Vector and PDF query"]


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


def test_query_history_summary_can_filter_by_planned_route_and_infer_from_databases():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_route_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({
                        "question": "Graph lookup",
                        "status": "success",
                        "duration_ms": 90,
                        "query_plan_databases": ["graph_db"],
                        "query_plan_methods": ["get_papers_citing_paper"],
                    }),
                    json.dumps({
                        "question": "Vector lookup",
                        "status": "success",
                        "duration_ms": 110,
                        "query_plan_databases": ["vector_db"],
                        "query_plan_methods": ["search_relevant_sentences"],
                    }),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=10, planned_route="graph_analysis")

        assert summary["planned_route_filter"] == "graph_analysis"
        assert summary["entries_returned"] == 1
        assert summary["latest_question"] == "Graph lookup"
        assert summary["query_plan_route_breakdown"] == [{"route": "graph_analysis", "count": 1}]
        assert [entry["question"] for entry in summary["entries"]] == ["Graph lookup"]


def test_query_history_summary_counts_entries_without_route_plans():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_no_routes_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({
                        "question": "empty router output",
                        "status": "success",
                        "query_plan_step_count": 0,
                        "query_plan_databases": [],
                    }),
                    json.dumps({
                        "question": "unknown database route",
                        "status": "success",
                        "query_plan_step_count": 1,
                        "query_plan_databases": ["custom_db"],
                    }),
                    json.dumps({
                        "question": "vector route",
                        "status": "success",
                        "query_plan_step_count": 1,
                        "query_plan_databases": ["vector_db"],
                    }),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        summary = recorder.summary(limit=2)

        assert summary["empty_query_plan_count"] == 0
        assert summary["no_planned_route_count"] == 1
        assert summary["matching_empty_query_plan_count"] == 1
        assert summary["matching_no_planned_route_count"] == 2
        assert summary["query_plan_route_breakdown"] == [{"route": "vector_search", "count": 1}]


def test_query_history_summary_can_sort_display_window_before_limit():
    query_history = _load_module(QUERY_HISTORY_PATH, f"query_history_sort_{uuid.uuid4().hex}")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "query_history.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"question": "old slow", "status": "success", "duration_ms": 900, "response_chars": 80}),
                    json.dumps({"question": "middle fast", "status": "success", "duration_ms": 100, "response_chars": 40}),
                    json.dumps({"question": "new medium", "status": "success", "duration_ms": 500, "response_chars": 120}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        recorder = query_history.QueryHistoryRecorder(log_file=str(log_path))
        slowest = recorder.summary(limit=2, sort_order="slowest")
        shortest_response = recorder.summary(limit=2, sort_order="shortest-response")

        assert slowest["sort_order"] == "slowest"
        assert slowest["matching_entries_total"] == 3
        assert [entry["question"] for entry in slowest["entries"]] == ["old slow", "new medium"]
        assert slowest["latest_question"] == "new medium"
        assert shortest_response["sort_order"] == "shortest-response"
        assert [entry["question"] for entry in shortest_response["entries"]] == ["middle fast", "old slow"]
