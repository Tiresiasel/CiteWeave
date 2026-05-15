import importlib.util
import json
import tempfile
import uuid
from pathlib import Path


BATCH_TRACKER_PATH = Path(__file__).resolve().parents[1] / "src" / "kernel" / "batch_tracker.py"
SERVICE_PATH = Path(__file__).resolve().parents[1] / "src" / "kernel" / "service.py"
QUERY_HISTORY_PATH = Path(__file__).resolve().parents[1] / "src" / "kernel" / "query_history.py"


def _load_module(path: Path, prefix: str, module_name: str | None = None):
    effective_name = module_name or f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(effective_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Unique module names are used for standalone modules that do not shadow the
    # package under test.  Tests that need package stubs use ModuleSandbox below.
    import sys
    sys.modules[effective_name] = module
    spec.loader.exec_module(module)
    return module


from module_isolation import ModuleSandbox


def _load_service_with_stubs(batch_tracker, document_processor_cls, research_system_cls, suffix: str):
    with ModuleSandbox() as sandbox:
        sandbox.stub("src", __path__=[])
        sandbox.stub("src.processing", __path__=[])
        sandbox.stub("src.processing.pdf", __path__=[])
        sandbox.stub("src.processing.pdf.document_processor", DocumentProcessor=document_processor_cls)
        sandbox.stub("src.agents", __path__=[])
        sandbox.stub("src.agents.multi_agent_research_system", LangGraphResearchSystem=research_system_cls)
        sandbox.stub("src.agents.routing", active_route_configuration=lambda: {"default_route": "vector_search"})
        sandbox.stub("src.kernel", __path__=[])
        sandbox.set("src.kernel.batch_tracker", batch_tracker)
        query_history = sandbox.load(QUERY_HISTORY_PATH, f"src.kernel.query_history_{suffix}_{uuid.uuid4().hex}")
        sandbox.set("src.kernel.query_history", query_history)
        return sandbox.load(SERVICE_PATH, f"src.kernel.service_{suffix}_{uuid.uuid4().hex}")


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
                "duration_seconds": 3.5,
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
        assert summary["total_completed_duration_seconds"] == 3.5
        assert summary["average_completed_duration_seconds"] == 3.5
        assert summary["last_completed"]["pdf_path"] == "/papers/ok.pdf"
        assert summary["last_completed"]["paper_id"] == "paper-1"
        assert summary["last_completed"]["duration_seconds"] == 3.5
        assert summary["failure_reasons"] == [{"error": "grobid timeout", "count": 1}]

        persisted = json.loads(tracker_file.read_text(encoding="utf-8"))
        assert persisted["/papers/ok.pdf"]["paper_id"] == "paper-1"
        assert persisted["/papers/ok.pdf"]["duration_seconds"] == 3.5
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

    service = _load_service_with_stubs(batch_tracker, DummyDocumentProcessor, DummyResearchSystem, "upload")

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
        assert summary["average_completed_duration_seconds"] is not None
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

    service = _load_service_with_stubs(batch_tracker, DummyDocumentProcessor, DummyResearchSystem, "progress")

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
            {"paper_id": "paper-1", "processing_time": 1, "duration_seconds": 4.0, "total_sentences": 10, "total_citations": 2},
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
        assert progress["completion_percent"] == 33.33
        assert progress["average_completed_duration_seconds"] == 4.0
        assert progress["average_completed_duration_human"] == "4s"
        assert progress["estimated_remaining_seconds"] == 8.0
        assert progress["estimated_remaining_human"] == "8s"
        assert progress["summary"]["aggregate_stats"] == {
            "total_sentences": 10,
            "sentences_with_citations": 0,
            "total_citations": 2,
            "total_references": 0,
        }
        assert progress["summary"]["last_completed"]["pdf_path"] == str(pdf_dir / "ok.pdf")
        assert progress["summary"]["last_completed"]["duration_seconds"] == 4.0
        assert progress["summary"]["failure_reasons"] == [{"error": "parse error", "count": 1}]


def test_content_deduplication_marks_duplicate_paths_without_deleting_files():
    batch_tracker = _load_module(BATCH_TRACKER_PATH, "batch_tracker_dedupe")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_dir = Path(tmpdir) / "papers"
        pdf_dir.mkdir()
        canonical = pdf_dir / "already-done.pdf"
        duplicate = pdf_dir / "same-bytes.pdf"
        unique = pdf_dir / "unique.pdf"
        canonical.write_bytes(b"%PDF same bytes")
        duplicate.write_bytes(b"%PDF same bytes")
        unique.write_bytes(b"%PDF unique bytes")

        tracker_file = Path(tmpdir) / "tracker.json"
        tracker = batch_tracker.BatchUploadTracker(str(pdf_dir), tracker_file=str(tracker_file))
        tracker.mark_file_completed(
            str(canonical),
            {"paper_id": "paper-1", "processing_time": 1, "duration_seconds": 2.0},
        )

        dedupe = tracker.apply_content_deduplication([str(duplicate), str(unique), str(canonical)])
        summary = tracker.get_progress_summary()

        assert dedupe["total_pdf_files"] == 3
        assert dedupe["unique_content_files"] == 2
        assert dedupe["duplicate_files"] == 1
        assert summary["completed"] == 1
        assert summary["duplicate"] == 1
        assert summary["duplicate_files"] == {str(duplicate): str(canonical)}
        assert tracker.get_pending_files([str(duplicate), str(unique), str(canonical)]) == [str(unique)]
        assert tracker.file_hash_for_path(str(canonical)) == tracker.file_hash_for_path(str(duplicate))
        assert duplicate.exists()


def test_get_pending_files_can_skip_previous_failures():
    batch_tracker = _load_module(
        BATCH_TRACKER_PATH,
        "batch_tracker_skip_failed",
        module_name=f"src.kernel.batch_tracker_skip_failed_{uuid.uuid4().hex}",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_dir = Path(tmpdir) / "papers"
        pdf_dir.mkdir()
        failed = pdf_dir / "failed.pdf"
        pending = pdf_dir / "pending.pdf"
        failed.write_bytes(b"%PDF failed")
        pending.write_bytes(b"%PDF pending")

        tracker = batch_tracker.BatchUploadTracker(str(pdf_dir), tracker_file=str(Path(tmpdir) / "tracker.json"))
        tracker.mark_file_failed(str(failed), "timeout")

        all_files = [str(failed), str(pending)]
        assert tracker.get_pending_files(all_files) == all_files
        assert tracker.get_pending_files(all_files, retry_failed=False) == [str(pending)]


def test_kernel_batch_upload_processes_only_unique_pdf_content():
    batch_tracker = _load_module(
        BATCH_TRACKER_PATH,
        "batch_tracker_upload_dedupe",
        module_name=f"src.kernel.batch_tracker_upload_dedupe_{uuid.uuid4().hex}",
    )

    class DummyDocumentProcessor:
        calls = []

        def process_document(self, pdf_path, save_results=True):
            self.calls.append(Path(pdf_path).name)
            return {
                "paper_id": f"paper-{Path(pdf_path).stem}",
                "processing_stats": {
                    "total_sentences": 1,
                    "sentences_with_citations": 0,
                    "total_citations": 0,
                    "total_references": 0,
                },
            }

    class DummyResearchSystem:
        pass

    service = _load_service_with_stubs(batch_tracker, DummyDocumentProcessor, DummyResearchSystem, "upload_dedupe")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_dir = Path(tmpdir) / "papers"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_bytes(b"%PDF duplicate")
        (pdf_dir / "b.pdf").write_bytes(b"%PDF duplicate")
        (pdf_dir / "c.pdf").write_bytes(b"%PDF unique")

        tracker_file = Path(tmpdir) / "tracker.json"
        original_tracker_cls = service.BatchUploadTracker
        service.BatchUploadTracker = lambda directory: batch_tracker.BatchUploadTracker(directory, tracker_file=str(tracker_file))
        try:
            kernel = service.CiteWeaveKernel()
            result = kernel.batch_upload(str(pdf_dir), resume=False, force_restart=True)
            progress = kernel.progress_summary(str(pdf_dir))
        finally:
            service.BatchUploadTracker = original_tracker_cls

        assert result["processed_count"] == 2
        assert result["deduplication"]["duplicate_files"] == 1
        assert sorted(DummyDocumentProcessor.calls) == ["a.pdf", "c.pdf"]
        assert progress["total_pdf_files"] == 3
        assert progress["unique_content_count"] == 2
        assert progress["duplicate_count"] == 1
        assert progress["completed_count"] == 2
        assert progress["pending_count"] == 0
        assert progress["completion_percent"] == 100.0
