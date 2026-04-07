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
from .query_history import QueryHistoryRecorder


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
        started_at = time.time()
        recorder = QueryHistoryRecorder()

        try:
            response = self.research_system.research_question(question, confirmation)
        except Exception as exc:
            recorder.record(
                {
                    "timestamp": started_at,
                    "question": question,
                    "confirmation": confirmation,
                    "status": "error",
                    "duration_ms": int((time.time() - started_at) * 1000),
                    "response_chars": 0,
                    "response_preview": "",
                    "error": str(exc),
                    "satisfaction": None,
                }
            )
            raise

        recorder.record(
            {
                "timestamp": started_at,
                "question": question,
                "confirmation": confirmation,
                "status": "success",
                "duration_ms": int((time.time() - started_at) * 1000),
                "response_chars": len(response),
                "response_preview": response[:500],
                "satisfaction": None,
            }
        )
        return response

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
            started_at = time.time()
            try:
                result = self.upload_document(pdf_path, save_results=True)
                finished_at = time.time()
                stats = result.get("processing_stats", {})
                compact = {
                    "pdf_path": pdf_path,
                    "paper_id": result.get("paper_id"),
                    "processed_at": finished_at,
                    "processing_time": finished_at,
                    "duration_seconds": round(finished_at - started_at, 3),
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

        files = {
            ".env": Path('.env').exists(),
            "docker_compose": Path('docker-compose.yml').exists(),
            "model_config": Path('config/model_config.json').exists(),
            "neo4j_config": Path('config/neo4j_config.json').exists(),
        }
        services = {
            "qdrant": probe("http://localhost:6333/collections"),
            "grobid": probe("http://localhost:8070/api/isalive"),
            "neo4j_http": probe("http://localhost:7474"),
            "openclaw_gateway": probe(gateway_url + "/models") if env_mode == "openclaw" else None,
        }

        missing_files = [name for name, exists in files.items() if not exists]
        down_services = [name for name, result in services.items() if result is not None and not result.get("ok")]

        action_items = []
        if missing_files:
            action_items.append(f"Create or restore required config files: {', '.join(missing_files)}")
        if down_services:
            action_items.append(f"Start or fix backend services: {', '.join(down_services)}")
        if not action_items:
            action_items.append("System looks healthy. Continue with upload/query/chat commands.")

        if missing_files or down_services:
            overall_status = "degraded"
        else:
            overall_status = "ok"

        return {
            "project_root": str(Path.cwd()),
            "summary": {
                "overall_status": overall_status,
                "missing_files": missing_files,
                "down_services": down_services,
                "action_items": action_items,
            },
            "env": {
                "llm_provider": env_mode,
                "llm_model": os.environ.get("CITEWEAVE_LLM_MODEL", ""),
                "gateway_base": gateway_url if env_mode == "openclaw" else None,
            },
            "files": files,
            "services": services,
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
        failed_files = summary["failed_files"]
        failed_paths = set(failed_files.keys())
        not_started_files = sorted(pdf_path for pdf_path in all_files if pdf_path not in tracker.progress_data)
        retryable_failed_files = sorted(pdf_path for pdf_path in all_files if pdf_path in failed_paths)
        average_completed_duration_seconds = summary.get("average_completed_duration_seconds")
        estimated_remaining_seconds = None
        if average_completed_duration_seconds is not None and pending_files:
            estimated_remaining_seconds = round(float(average_completed_duration_seconds) * len(pending_files), 3)

        return {
            "directory": directory,
            "cleared": clear,
            "total_pdf_files": len(all_files),
            "summary": summary,
            "pending_count": len(pending_files),
            "pending_files": sorted(pending_files),
            "not_started_count": len(not_started_files),
            "not_started_files": not_started_files,
            "retryable_failed_count": len(retryable_failed_files),
            "retryable_failed_files": retryable_failed_files,
            "completed_count": summary["completed"],
            "completed_files": summary["completed_files"],
            "failed_count": summary["failed"],
            "failed_files": failed_files,
            "average_completed_duration_seconds": average_completed_duration_seconds,
            "estimated_remaining_seconds": estimated_remaining_seconds,
        }

    def query_history_snapshot(self, limit: int = 10, status: str = "all") -> Dict[str, Any]:
        recorder = QueryHistoryRecorder()
        return recorder.summary(limit=limit, status=status)
