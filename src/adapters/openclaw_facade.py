"""OpenClaw-facing adapter facade for CiteWeave.

This module is intentionally thin: OpenClaw should treat CiteWeave as a kernel
and call explicit, serializable methods here (or a future HTTP layer), instead
of scraping terminal output.
"""

from __future__ import annotations

from typing import Any, Dict

from src.kernel import CiteWeaveKernel


class OpenClawCiteWeaveFacade:
    """Stable facade intended for a future OpenClaw Skill adapter."""

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
        answer = self.kernel.query(question, confirmation)
        return {
            "question": question,
            "confirmation": confirmation,
            "answer": answer,
        }

    def routes(self) -> Dict[str, Any]:
        return self.kernel.routes_snapshot()

    def progress(self, directory: str, clear: bool = False) -> Dict[str, Any]:
        return self.kernel.progress_summary(directory, clear=clear)
