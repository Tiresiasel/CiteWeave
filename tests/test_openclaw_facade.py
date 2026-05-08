import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "adapters" / "openclaw_facade.py"


def load_facade_module():
    for name in [
        "src",
        "src.processing",
        "src.processing.pdf",
        "src.processing.pdf.document_processor",
        "src.agents",
        "src.agents.multi_agent_research_system",
        "src.agents.routing",
        "src.kernel",
        "src.kernel.service",
        "src.kernel.batch_tracker",
        "src.kernel.query_history",
    ]:
        if name in sys.modules and getattr(sys.modules[name], "__file__", None) is None:
            sys.modules.pop(name, None)

    spec = importlib.util.spec_from_file_location("openclaw_facade_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeKernel:
    def upload_document(self, pdf_path, save_results=True):
        return {"paper_id": "paper-1", "processing_stats": {"total_sentences": 10}, "sentences_with_citations": []}

    def diagnose_document(self, pdf_path):
        return {"ok": True, "pdf_path": pdf_path}

    def query(self, question, confirmation="continue", source="kernel.query"):
        return f"answer:{question}:{confirmation}:{source}"

    def routes_snapshot(self):
        return {"default_route": "vector_search"}

    def progress_summary(self, directory, clear=False):
        return {"directory": directory, "cleared": clear}

    def chat_turn(self, user_input, history=None, menu_choice=None, collected_data=None):
        return {"text": f"chat:{user_input}", "needs_user_choice": False, "collected_data": collected_data or {}}

    def batch_upload(self, directory, resume=True, force_restart=False, clear_progress=False):
        return {"directory": directory, "processed_count": 3, "failed_count": 0}

    def health_snapshot(self):
        return {"services": {"openclaw_gateway": {"ok": True}}}

    def bootstrap_plan(self):
        return {"local_cli": {"script": "bash scripts/bootstrap_local.sh"}}


class OpenClawFacadeTests(unittest.TestCase):
    def test_facade_delegates_query_and_chat(self):
        mod = load_facade_module()
        facade = mod.OpenClawCiteWeaveFacade(kernel=FakeKernel())
        query_result = facade.query("hello", confirmation="continue")
        self.assertEqual(query_result["answer"], "answer:hello:continue:openclaw.facade.query")
        chat_result = facade.chat_turn("hi")
        self.assertEqual(chat_result["text"], "chat:hi")

    def test_facade_exposes_health_and_bootstrap(self):
        mod = load_facade_module()
        facade = mod.OpenClawCiteWeaveFacade(kernel=FakeKernel())
        self.assertTrue(facade.health()["services"]["openclaw_gateway"]["ok"])
        self.assertEqual(facade.bootstrap_plan()["local_cli"]["script"], "bash scripts/bootstrap_local.sh")


if __name__ == "__main__":
    unittest.main()
