"""OpenClaw-facing adapter facade for CiteWeave.

This module is intentionally thin: OpenClaw should treat CiteWeave as a kernel
and call explicit, serializable methods here (or a future HTTP layer), instead
of scraping terminal output.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.kernel import CiteWeaveKernel


class OpenClawCiteWeaveFacade:
    """Stable facade intended for a future OpenClaw Skill adapter.

    Return values are structured dictionaries so OpenClaw skills or tools can
    consume them without scraping terminal output.
    """

    def __init__(self, kernel: CiteWeaveKernel | None = None):
        self.kernel = kernel or CiteWeaveKernel()

    def upload_pdf(self, pdf_path: str) -> Dict[str, Any]:
        result = self.kernel.upload_document(pdf_path)
        stats = result.get("processing_stats", {})
        return {
            "paper_id": result.get("paper_id"),
            "stats": stats,
            "sentences_with_citations": result.get("sentences_with_citations", []),
        }

    def diagnose_pdf(self, pdf_path: str) -> Dict[str, Any]:
        return self.kernel.diagnose_document(pdf_path)

    def query(self, question: str, confirmation: str = "continue") -> Dict[str, Any]:
        answer = self.kernel.query(question, confirmation, source="openclaw.facade.query")
        return {
            "question": question,
            "confirmation": confirmation,
            "answer": answer,
        }

    def routes(self) -> Dict[str, Any]:
        return self.kernel.routes_snapshot()

    def progress(self, directory: str, clear: bool = False) -> Dict[str, Any]:
        return self.kernel.progress_summary(directory, clear=clear)


    def chat_turn(
        self,
        user_input: str,
        history: Optional[list] = None,
        menu_choice: Optional[str] = None,
        collected_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = self.kernel.chat_turn(
            user_input,
            history=history,
            menu_choice=menu_choice,
            collected_data=collected_data,
        )
        return {
            "input": user_input,
            "menu_choice": menu_choice,
            **result,
        }

    def batch_upload(
        self,
        directory: str,
        resume: bool = True,
        force_restart: bool = False,
        clear_progress: bool = False,
    ) -> Dict[str, Any]:
        return self.kernel.batch_upload(
            directory,
            resume=resume,
            force_restart=force_restart,
            clear_progress=clear_progress,
        )

    def health(self) -> Dict[str, Any]:
        return self.kernel.health_snapshot()

    def bootstrap_plan(self) -> Dict[str, Any]:
        return self.kernel.bootstrap_plan()
