"""CiteWeave kernel service.

This is the adapter-neutral application layer for the project.
Entry points such as the CLI, OpenClaw integration, and future HTTP APIs
should call this service instead of directly composing processing and agent
objects on their own.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict

from src.processing.pdf.document_processor import DocumentProcessor
from src.agents.multi_agent_research_system import LangGraphResearchSystem
from src.agents.routing import active_route_configuration
from .batch_tracker import BatchUploadTracker


class CiteWeaveKernel:
    """Stable application service boundary for CiteWeave."""

    def __init__(self):
        self._document_processor = None
        self._research_system = None

    @property
    def document_processor(self) -> DocumentProcessor:
        if self._document_processor is None:
            self._document_processor = DocumentProcessor()
        return self._document_processor

    @property
    def research_system(self) -> LangGraphResearchSystem:
        if self._research_system is None:
            self._research_system = LangGraphResearchSystem()
        return self._research_system

    def upload_document(self, pdf_path: str, save_results: bool = True) -> Dict[str, Any]:
        return self.document_processor.process_document(pdf_path, save_results=save_results)

    def diagnose_document(self, pdf_path: str) -> Dict[str, Any]:
        return self.document_processor.diagnose_document_processing(pdf_path)

    def query(self, question: str, confirmation: str = "continue") -> str:
        return self.research_system.research_question(question, confirmation)

    def start_chat_system(self) -> LangGraphResearchSystem:
        return self.research_system

    def routes_snapshot(self) -> Dict[str, Any]:
        return active_route_configuration()

    def progress_summary(self, directory: str, clear: bool = False) -> Dict[str, Any]:
        tracker = BatchUploadTracker(directory)
        summary_before = tracker.get_progress_summary()
        if clear:
            tracker.clear_progress(directory)
        pending_files = tracker.get_pending_files(
            glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True),
            force_restart=False,
        )
        return {
            "directory": directory,
            "summary": tracker.get_progress_summary() if clear else summary_before,
            "pending_files": pending_files,
            "cleared": clear,
        }
