"""CiteWeave kernel service.

This is the adapter-neutral application layer for the project.
Entry points such as the CLI, OpenClaw integration, and future HTTP APIs
should call this service instead of directly composing processing and agent
objects on their own.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request, error

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

    def chat_turn(self, user_input: str, history: Optional[list] = None, menu_choice: Optional[str] = None, collected_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.research_system.interactive_research_chat(
            user_input,
            history=history,
            menu_choice=menu_choice,
            collected_data=collected_data,
        )

    def batch_upload(self, directory: str, resume: bool = True, force_restart: bool = False, clear_progress: bool = False) -> Dict[str, Any]:
        tracker = BatchUploadTracker(directory)
        if clear_progress:
            tracker.clear_progress(directory)

        all_files = glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True)
        pending_files = tracker.get_pending_files(all_files, force_restart=force_restart or not resume)

        processed = []
        failed = []
        for pdf_path in pending_files:
            try:
                result = self.upload_document(pdf_path, save_results=True)
                stats = result.get("processing_stats", {})
                compact = {
                    "pdf_path": pdf_path,
                    "paper_id": result.get("paper_id"),
                    "processing_time": time.time(),
                    "total_sentences": stats.get("total_sentences", 0),
                    "sentences_with_citations": stats.get("sentences_with_citations", 0),
                    "total_citations": stats.get("total_citations", 0),
                    "total_references": stats.get("total_references", 0),
                }
                tracker.mark_file_completed(pdf_path, compact)
                processed.append(compact)
            except Exception as e:
                tracker.mark_file_failed(pdf_path, str(e))
                failed.append({"pdf_path": pdf_path, "error": str(e)})

        return {
            "directory": directory,
            "total_files": len(all_files),
            "processed_count": len(processed),
            "failed_count": len(failed),
            "processed": processed,
            "failed": failed,
            "summary": tracker.get_progress_summary(),
        }

    def health_snapshot(self) -> Dict[str, Any]:
        def probe(url: str) -> Dict[str, Any]:
            try:
                with request.urlopen(url, timeout=5) as resp:
                    return {"ok": True, "status": resp.getcode(), "url": url}
            except error.HTTPError as e:
                return {"ok": False, "status": e.code, "url": url, "error": str(e)}
            except Exception as e:
                return {"ok": False, "status": None, "url": url, "error": str(e)}

        env_mode = os.environ.get("CITEWEAVE_LLM_PROVIDER", "openai")
        gateway_url = os.environ.get("CITEWEAVE_LLM_API_BASE", "http://localhost:18789/v1").rstrip("/")

        return {
            "project_root": str(Path.cwd()),
            "env": {
                "llm_provider": env_mode,
                "llm_model": os.environ.get("CITEWEAVE_LLM_MODEL", ""),
                "gateway_base": gateway_url if env_mode == "openclaw" else None,
            },
            "files": {
                ".env": Path('.env').exists(),
                "docker_compose": Path('docker-compose.yml').exists(),
                "model_config": Path('config/model_config.json').exists(),
                "neo4j_config": Path('config/neo4j_config.json').exists(),
            },
            "services": {
                "qdrant": probe("http://localhost:6333/collections"),
                "grobid": probe("http://localhost:8070/api/isalive"),
                "neo4j_http": probe("http://localhost:7474"),
                "openclaw_gateway": probe(gateway_url + "/models") if env_mode == "openclaw" else None,
            },
        }

    def bootstrap_plan(self) -> Dict[str, Any]:
        return {
            "local_cli": {
                "script": "bash scripts/bootstrap_local.sh",
                "next_steps": [
                    ".venv/bin/python -m src.core.cli upload path/to/paper.pdf",
                    '.venv/bin/python -m src.core.cli query "<question>"',
                    ".venv/bin/python -m src.core.cli chat",
                ],
            },
            "openclaw": {
                "script": "bash scripts/bootstrap_openclaw.sh",
                "next_steps": [
                    "openclaw gateway status",
                    ".venv/bin/python -m src.core.cli routes",
                    "bash scripts/deployment_check.sh",
                ],
            },
        }

    def progress_summary(self, directory: str, clear: bool = False) -> Dict[str, Any]:
        tracker = BatchUploadTracker(directory)
        if clear:
            tracker.clear_progress(directory)

        all_files = glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True)
        summary = tracker.get_progress_summary()
        pending_files = tracker.get_pending_files(all_files, force_restart=False)

        return {
            "directory": directory,
            "cleared": clear,
            "total_pdf_files": len(all_files),
            "summary": summary,
            "pending_count": len(pending_files),
            "pending_files": sorted(pending_files),
            "completed_count": summary["completed"],
            "completed_files": summary["completed_files"],
            "failed_count": summary["failed"],
            "failed_files": summary["failed_files"],
        }
