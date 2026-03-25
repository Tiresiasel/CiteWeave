import io
import sys
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _install_cli_stubs():
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None
    sys.modules.setdefault("dotenv", dotenv_module)

    prompt_toolkit_module = types.ModuleType("prompt_toolkit")
    prompt_toolkit_module.prompt = lambda *args, **kwargs: ""
    sys.modules.setdefault("prompt_toolkit", prompt_toolkit_module)

    doc_processor_module = types.ModuleType("src.processing.pdf.document_processor")
    doc_processor_module.DocumentProcessor = object
    sys.modules.setdefault("src.processing.pdf.document_processor", doc_processor_module)

    research_module = types.ModuleType("src.agents.multi_agent_research_system")
    research_module.LangGraphResearchSystem = object
    sys.modules.setdefault("src.agents.multi_agent_research_system", research_module)


_install_cli_stubs()

from src.core import cli  # noqa: E402


class QueryCommandTests(unittest.TestCase):
    def test_handle_query_command_runs_research_workflow(self):
        args = Namespace(question="What does CiteWeave know about platform ecosystems?", confirmation="expand")

        fake_system = unittest.mock.Mock()
        fake_system.research_question.return_value = "Structured answer"

        stdout = io.StringIO()
        with patch.object(cli, "LangGraphResearchSystem", return_value=fake_system):
            with redirect_stdout(stdout):
                cli.handle_query_command(args)

        fake_system.research_question.assert_called_once_with(
            "What does CiteWeave know about platform ecosystems?",
            "expand",
        )
        output = stdout.getvalue()
        self.assertIn("Querying: What does CiteWeave know about platform ecosystems?", output)
        self.assertIn("Structured answer", output)

    def test_handle_query_command_defaults_confirmation_to_continue(self):
        args = Namespace(question="Summarize ambidexterity research")

        fake_system = unittest.mock.Mock()
        fake_system.research_question.return_value = "Summary"

        with patch.object(cli, "LangGraphResearchSystem", return_value=fake_system):
            with redirect_stdout(io.StringIO()):
                cli.handle_query_command(args)

        fake_system.research_question.assert_called_once_with(
            "Summarize ambidexterity research",
            "continue",
        )

    def test_handle_query_command_exits_on_failure(self):
        args = Namespace(question="broken query", confirmation="continue")

        stdout = io.StringIO()
        with patch.object(cli, "LangGraphResearchSystem", side_effect=RuntimeError("model unavailable")):
            with self.assertRaises(SystemExit) as exc:
                with redirect_stdout(stdout):
                    cli.handle_query_command(args)

        self.assertEqual(exc.exception.code, 1)
        self.assertIn("Error querying argument graph: model unavailable", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
