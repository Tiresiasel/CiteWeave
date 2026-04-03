import importlib.util
import json
import sys
import tempfile
import types
import uuid
from pathlib import Path


BATCH_TRACKER_PATH = Path(__file__).resolve().parents[1] / "src" / "kernel" / "batch_tracker.py"
SERVICE_PATH = Path(__file__).resolve().parents[1] / "src" / "kernel" / "service.py"


def _load_module(path: Path, prefix: str, module_name: str | None = None):
    effective_name = module_name or f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(effective_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[effective_name] = module
    spec.loader.exec_module(module)
    return module


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def test_batch_tracker_summary_includes_completed_and_failed_files():
    batch_tracker = _load_module(BATCH_TRACKER_PATH, "batch_tracker")

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker_file = Path(tmpdir) / "tracker.json"
        tracker = batch_tracker.BatchUploadTracker("/papers", tracker_file=str(tracker_file))

        tracker.mark_file_completed(
            "/papers/ok.pdf",
            {
                "paper_id": "paper-1",
                "processing_time": 1234567890,
                "total_sentences": 20,
                "sentences_with_citations": 5,
                "total_citations": 8,
                "total_references": 10,
            },
        )
        tracker.mark_file_failed("/papers/bad.pdf", "grobid timeout")

        summary = tracker.get_progress_summary()

        assert summary["completed"] == 1
        assert summary["failed"] == 1
        assert summary["completed_files"] == ["/papers/ok.pdf"]
        assert summary["failed_files"] == {"/papers/bad.pdf": "grobid timeout"}
        assert summary["aggregate_stats"] == {
            "total_sentences": 20,
            "sentences_with_citations": 5,
            "total_citations": 8,
            "total_references": 10,
        }
        assert summary["last_completed"]["pdf_path"] == "/papers/ok.pdf"
        assert summary["last_completed"]["paper_id"] == "paper-1"
        assert summary["failure_reasons"] == [{"error": "grobid timeout", "count": 1}]

        persisted = json.loads(tracker_file.read_text(encoding="utf-8"))
        assert persisted["/papers/ok.pdf"]["paper_id"] == "paper-1"
        assert persisted["/papers/bad.pdf"]["error"] == "grobid timeout"


def test_kernel_batch_upload_preserves_tracker_aggregate_stats():
    batch_tracker = _load_module(
        BATCH_TRACKER_PATH,
        "batch_tracker",
        module_name=f"src.kernel.batch_tracker_upload_{uuid.uuid4().hex}",
    )

    class DummyDocumentProcessor:
        def process_document(self, pdf_path, save_results=True):
            return {
                "paper_id": "paper-1",
                "processing_stats": {
                    "total_sentences": 14,
                    "sentences_with_citations": 6,
                    "total_citations": 11,
                    "total_references": 13,
                },
            }

    class DummyResearchSystem:
        pass

    _stub_module("src", __path__=[])
    _stub_module("src.processing", __path__=[])
    _stub_module("src.processing.pdf", __path__=[])
    _stub_module("src.processing.pdf.document_processor", DocumentProcessor=DummyDocumentProcessor)
    _stub_module("src.agents", __path__=[])
    _stub_module("src.agents.multi_agent_research_system", LangGraphResearchSystem=DummyResearchSystem)
    _stub_module("src.agents.routing", active_route_configuration=lambda: {"default_route": "vector_search"})
    _stub_module("src.kernel", __path__=[])
    sys.modules["src.kernel.batch_tracker"] = batch_tracker

    service = _load_module(
        SERVICE_PATH,
        "kernel_service_upload",
        module_name=f"src.kernel.service_upload_{uuid.uuid4().hex}",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_dir = Path(tmpdir) / "papers"
        pdf_dir.mkdir()
        (pdf_dir / "ok.pdf").write_bytes(b"%PDF-1.4 ok")

        tracker_file = Path(tmpdir) / "tracker.json"
        original_tracker_cls = service.BatchUploadTracker

        service.BatchUploadTracker = lambda directory: batch_tracker.BatchUploadTracker(directory, tracker_file=str(tracker_file))
        try:
            kernel = service.CiteWeaveKernel()
            result = kernel.batch_upload(str(pdf_dir), resume=False, force_restart=True)
        finally:
            service.BatchUploadTracker = original_tracker_cls

        summary = result["summary"]
        assert result["processed_count"] == 1
        assert result["failed_count"] == 0
        assert summary["aggregate_stats"] == {
            "total_sentences": 14,
            "sentences_with_citations": 6,
            "total_citations": 11,
            "total_references": 13,
        }
        assert summary["last_completed"]["paper_id"] == "paper-1"


def test_kernel_progress_summary_returns_actionable_breakdown():
    batch_tracker = _load_module(
        BATCH_TRACKER_PATH,
        "batch_tracker",
        module_name=f"src.kernel.batch_tracker_test_{uuid.uuid4().hex}",
    )

    class DummyDocumentProcessor:
        pass

    class DummyResearchSystem:
        pass

    _stub_module("src", __path__=[])
    _stub_module("src.processing", __path__=[])
    _stub_module("src.processing.pdf", __path__=[])
    _stub_module("src.processing.pdf.document_processor", DocumentProcessor=DummyDocumentProcessor)
    _stub_module("src.agents", __path__=[])
    _stub_module("src.agents.multi_agent_research_system", LangGraphResearchSystem=DummyResearchSystem)
    _stub_module("src.agents.routing", active_route_configuration=lambda: {"default_route": "vector_search"})
    _stub_module("src.kernel", __path__=[])
    sys.modules["src.kernel.batch_tracker"] = batch_tracker

    service = _load_module(
        SERVICE_PATH,
        "kernel_service",
        module_name=f"src.kernel.service_test_{uuid.uuid4().hex}",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_dir = Path(tmpdir) / "papers"
        pdf_dir.mkdir()
        (pdf_dir / "ok.pdf").write_bytes(b"%PDF-1.4 ok")
        (pdf_dir / "bad.pdf").write_bytes(b"%PDF-1.4 bad")
        (pdf_dir / "pending.pdf").write_bytes(b"%PDF-1.4 pending")

        tracker_file = Path(tmpdir) / "tracker.json"
        tracker = batch_tracker.BatchUploadTracker(str(pdf_dir), tracker_file=str(tracker_file))
        tracker.mark_file_completed(
            str(pdf_dir / "ok.pdf"),
            {"paper_id": "paper-1", "processing_time": 1, "total_sentences": 10, "total_citations": 2},
        )
        tracker.mark_file_failed(str(pdf_dir / "bad.pdf"), "parse error")

        original_tracker_cls = service.BatchUploadTracker
        tracker_file_path = tracker_file

        def tracker_factory(directory, tracker_file=None):
            return batch_tracker.BatchUploadTracker(directory, tracker_file=str(tracker_file or tracker_file_path))

        service.BatchUploadTracker = tracker_factory
        try:
            kernel = service.CiteWeaveKernel()
            progress = kernel.progress_summary(str(pdf_dir))
        finally:
            service.BatchUploadTracker = original_tracker_cls

        assert progress["total_pdf_files"] == 3
        assert progress["completed_count"] == 1
        assert progress["failed_count"] == 1
        assert progress["pending_count"] == 2
        assert progress["not_started_count"] == 1
        assert progress["retryable_failed_count"] == 1
        assert str(pdf_dir / "ok.pdf") in progress["completed_files"]
        assert progress["failed_files"] == {str(pdf_dir / "bad.pdf"): "parse error"}
        assert progress["retryable_failed_files"] == [str(pdf_dir / "bad.pdf")]
        assert progress["not_started_files"] == [str(pdf_dir / "pending.pdf")]
        assert progress["pending_files"] == sorted([
            str(pdf_dir / "bad.pdf"),
            str(pdf_dir / "pending.pdf"),
        ])
        assert progress["summary"]["aggregate_stats"] == {
            "total_sentences": 10,
            "sentences_with_citations": 0,
            "total_citations": 2,
            "total_references": 0,
        }
        assert progress["summary"]["last_completed"]["pdf_path"] == str(pdf_dir / "ok.pdf")
        assert progress["summary"]["failure_reasons"] == [{"error": "parse error", "count": 1}]
