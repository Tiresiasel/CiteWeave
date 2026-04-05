import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path


CLI_PATH = Path(__file__).resolve().parents[1] / "src" / "core" / "cli.py"


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class DummyTracker:
    def __init__(self):
        self.completed = []
        self.failed = []

    def mark_file_completed(self, pdf_path, result_data):
        self.completed.append((pdf_path, result_data))

    def mark_file_failed(self, pdf_path, error_msg):
        self.failed.append((pdf_path, str(error_msg)))


class FakeKernel:
    def upload_document(self, pdf_path, save_results=True):
        assert save_results is True
        return {
            "paper_id": f"paper:{Path(pdf_path).stem}",
            "processing_stats": {
                "total_sentences": 12,
                "sentences_with_citations": 4,
                "total_citations": 7,
                "total_references": 9,
            },
            "sentences_with_citations": [
                {
                    "sentence_text": "Smith (2020) says useful things.",
                    "citations": [
                        {
                            "intext": "Smith (2020)",
                            "reference": {"title": "Useful Things", "year": "2020"},
                        }
                    ],
                }
            ],
        }


def _load_cli_module():
    module_name = f"citeweave_cli_batch_{uuid.uuid4().hex}"

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
    _stub_module("src.kernel", CiteWeaveKernel=FakeKernel, BatchUploadTracker=DummyTracker)

    spec = importlib.util.spec_from_file_location(module_name, CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CliBatchUploadSequentialTests(unittest.TestCase):
    def test_process_files_sequentially_records_real_tracker_stats(self):
        cli = _load_cli_module()
        tracker = DummyTracker()

        cli.process_files_sequentially(["/tmp/example.pdf"], tracker)

        self.assertEqual(tracker.failed, [])
        self.assertEqual(len(tracker.completed), 1)

        pdf_path, result_data = tracker.completed[0]
        self.assertEqual(pdf_path, "/tmp/example.pdf")
        self.assertEqual(result_data["paper_id"], "paper:example")
        self.assertEqual(result_data["total_sentences"], 12)
        self.assertEqual(result_data["sentences_with_citations"], 4)
        self.assertEqual(result_data["total_citations"], 7)
        self.assertEqual(result_data["total_references"], 9)
        self.assertIn("processing_time", result_data)
        self.assertIn("processed_at", result_data)
        self.assertIn("duration_seconds", result_data)


if __name__ == "__main__":
    unittest.main()
