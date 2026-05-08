import unittest
import uuid
from pathlib import Path


CLI_PATH = Path(__file__).resolve().parents[1] / "src" / "core" / "cli.py"


from module_isolation import ModuleSandbox


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

    with ModuleSandbox() as sandbox:
        sandbox.stub("prompt_toolkit", prompt=lambda *args, **kwargs: "")
        sandbox.stub("src", __path__=[])
        sandbox.stub("src.processing", __path__=[])
        sandbox.stub("src.processing.pdf", __path__=[])
        sandbox.stub("src.processing.pdf.document_processor", DocumentProcessor=DummyDocumentProcessor)
        sandbox.stub("src.agents", __path__=[])
        sandbox.stub("src.agents.multi_agent_research_system", LangGraphResearchSystem=DummyLangGraphResearchSystem)
        sandbox.stub(
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
        sandbox.stub("src.kernel", CiteWeaveKernel=FakeKernel, BatchUploadTracker=DummyTracker)
        return sandbox.load(CLI_PATH, module_name)


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
