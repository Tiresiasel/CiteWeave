"""
cli.py
Command-line interface for CiteWeave.
"""

import argparse
import sys
import logging
import os
import glob
import threading
import time
import multiprocessing
import json
import signal
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # If python-dotenv is not installed, skip

from prompt_toolkit import prompt
import warnings
warnings.filterwarnings("ignore", message=".*found in sys.modules after import of package.*", category=RuntimeWarning)


def find_project_root() -> str:
    """Return the repository root without depending on the caller's cwd."""
    return str(Path(__file__).resolve().parents[2])


def ensure_project_root() -> str:
    """Switch CLI execution to the repository root.

    This is intentionally called from main(), not at import time, so importing
    the CLI module in tests or tools does not mutate global process state.
    """
    project_root = find_project_root()
    if os.getcwd() != project_root:
        os.chdir(project_root)
        print(f"[INFO] Changed working directory to project root: {project_root}")
    return project_root

# Set up logging based on environment variable (before importing other modules)
env = os.environ.get("CITEWEAVE_ENV", "production").lower()
if env in ("test", "development", "dev"):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.WARNING)
    # Silence common noisy loggers in production
    for noisy_logger in [
        "CiteWeave", "httpx", "sentence_transformers", "root", "ModelConfigManager", "LangGraphResearchSystem"
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from src.processing.pdf.document_processor import DocumentProcessor
from src.kernel import CiteWeaveKernel, BatchUploadTracker

def process_single_pdf_worker(pdf_path, diagnose=False, force=False):
    """
    Worker function for multiprocessing that processes a single PDF file.
    This function must be defined at module level for multiprocessing to work.
    """
    try:
        logging.info(f"START: Processing PDF file {pdf_path}")
        started_at = time.time()

        # Initialize the document processor in the worker process
        doc_processor = DocumentProcessor()
        
        # Process the document
        results = doc_processor.process_document(pdf_path, save_results=True)
        
        # Return success with basic stats
        stats = results.get('processing_stats', {})
        finished_at = time.time()
        result_data = {
            'status': 'success',
            'pdf_path': pdf_path,
            'paper_id': results.get('paper_id', 'unknown'),
            'total_sentences': stats.get('total_sentences', 0),
            'sentences_with_citations': stats.get('sentences_with_citations', 0),
            'total_citations': stats.get('total_citations', 0),
            'total_references': stats.get('total_references', 0),
            'processed_at': finished_at,
            'processing_time': finished_at,
            'duration_seconds': round(finished_at - started_at, 3),
        }
        
        logging.info(f"FINISH: Successfully processed {pdf_path} - Paper ID: {result_data['paper_id']}, Sentences: {result_data['total_sentences']}, Citations: {result_data['total_citations']}")
        return result_data
        
    except Exception as e:
        logging.error(f"FINISH: Failed to process {pdf_path} - Error: {str(e)}")
        return {
            'status': 'error',
            'pdf_path': pdf_path,
            'error': str(e),
            'processing_time': time.time()
        }

def main():
    """Main entry point for the CLI."""
    ensure_project_root()
    parser = argparse.ArgumentParser(description="CiteWeave CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload and process a PDF document with sentence-level citation analysis.")
    upload_parser.add_argument("pdf_path", type=str, help="Path to the PDF file.")
    upload_parser.add_argument("--diagnose", action="store_true", help="Run quality diagnosis before processing.")
    upload_parser.add_argument("--force", action="store_true", help="Force reprocessing even if cached results exist.")

    # Query command  
    query_parser = subparsers.add_parser("query", help="Query the citation knowledge graph.")
    query_parser.add_argument("question", type=str, help="Question to ask.")
    query_parser.add_argument(
        "--confirmation",
        default="continue",
        help="User confirmation mode to pass into the research workflow (default: continue)."
    )

    # Chat command
    subparsers.add_parser("chat", help="Start an interactive chat with the multi-agent research system.")

    # Diagnose command
    diagnose_parser = subparsers.add_parser("diagnose", help="Diagnose PDF processing quality.")
    diagnose_parser.add_argument("pdf_path", type=str, help="Path to the PDF file.")

    # Batch upload command
    batch_upload_parser = subparsers.add_parser("batch-upload", help="Upload and process all PDF files in a directory.")
    batch_upload_parser.add_argument("directory", type=str, help="Path to the directory containing PDF files.")
    batch_upload_parser.add_argument("--processors", type=int, default=4, 
                                   help="Number of processors to use for parallel processing (default: 4)")
    batch_upload_parser.add_argument("--sequential", action="store_true", 
                                   help="Force sequential processing (disable multiprocessing)")
    batch_upload_parser.add_argument("--resume", action="store_true", 
                                   help="Resume from previous batch upload (skip already processed files)")
    batch_upload_parser.add_argument("--force-restart", action="store_true", 
                                   help="Force restart and reprocess all files (ignore previous progress)")
    batch_upload_parser.add_argument("--clear-progress", action="store_true", 
                                   help="Clear progress tracking for this directory before starting")
    batch_upload_parser.add_argument("--preserve-order", action="store_true",
                                   help="Process pending PDFs in filesystem discovery order instead of resumable small-first order")
    batch_upload_parser.add_argument("--skip-failed", action="store_true",
                                   help="Resume without retrying files already marked failed")
    batch_upload_parser.add_argument("--file-timeout-seconds", type=int, default=0,
                                   help="Sequential mode only: fail and continue if one PDF exceeds this many seconds")

    # Progress status command
    progress_parser = subparsers.add_parser("progress", help="View batch upload progress status.")
    progress_parser.add_argument("directory", type=str, help="Path to the directory to check progress for.")
    progress_parser.add_argument("--clear", action="store_true", help="Clear progress for this directory.")
    progress_parser.add_argument("--json", action="store_true", help="Print machine-readable progress information as JSON.")
    progress_parser.add_argument("--show-completed", action="store_true", help="Also list completed files in text output.")

    # Papers command
    papers_parser = subparsers.add_parser("papers", help="Search or list the local paper index without exposing local PDF paths.")
    papers_parser.add_argument("--search", default="", help="Filter papers by title, author, journal, or year.")
    papers_parser.add_argument("--author", default="", help="Filter papers by author name using the author index.")
    papers_parser.add_argument("--title", default="", help="Filter papers by title substring after author/search filters.")
    papers_parser.add_argument("--limit", type=int, default=20, help="Maximum papers to display (default: 20; use 0 for all matches).")
    papers_parser.add_argument("--json", action="store_true", help="Print machine-readable paper index results as JSON.")
    papers_parser.add_argument(
        "--pdf-status",
        choices=["all", "available", "missing"],
        default="all",
        help="Filter by whether an indexed paper has an associated PDF path (default: all).",
    )

    # Routes command
    routes_parser = subparsers.add_parser("routes", help="Show active route configuration.")
    routes_parser.add_argument("--json", action="store_true", help="Print machine-readable route configuration as JSON.")
    routes_parser.add_argument("--check", action="store_true", help="Exit non-zero when route overrides or addon config files contain ignored/invalid entries.")

    # Health command
    health_parser = subparsers.add_parser("health", help="Show machine-readable service and environment health.")
    health_parser.add_argument("--json", action="store_true", help="Print machine-readable health information as JSON.")

    # Bootstrap plan command
    bootstrap_parser = subparsers.add_parser("bootstrap-plan", help="Show the recommended local and OpenClaw bootstrap steps.")
    bootstrap_parser.add_argument("--json", action="store_true", help="Print machine-readable bootstrap plan as JSON.")

    # Query history command
    query_history_parser = subparsers.add_parser("query-history", help="Inspect recent query telemetry from the local JSONL history log.")
    query_history_parser.add_argument("--limit", type=int, default=10, help="How many recent query records to include (default: 10).")
    query_history_parser.add_argument("--status", choices=["all", "success", "error", "corrupt"], default="all", help="Filter to a specific query status (default: all).")
    query_history_parser.add_argument("--source", default="all", help="Filter to a specific query source, such as cli.query or openclaw.facade.query.")
    query_history_parser.add_argument("--confirmation", default="all", help="Filter to a specific confirmation mode, such as continue or expand.")
    query_history_parser.add_argument("--satisfaction", default="all", help="Filter to a specific satisfaction bucket, such as satisfied, neutral, dissatisfied, or unrated.")
    query_history_parser.add_argument("--since-hours", type=float, default=None, help="Only include query records from the last N hours.")
    query_history_parser.add_argument("--contains", default="", help="Only include query records whose question, error, response preview, or raw corrupt row contains this substring.")
    query_history_parser.add_argument("--question-contains", default="", help="Only include query records whose question text contains this substring.")
    query_history_parser.add_argument("--error-contains", default="", help="Only include query records whose error text contains this substring.")
    query_history_parser.add_argument("--response-contains", default="", help="Only include query records whose response preview contains this substring.")
    query_history_parser.add_argument("--planned-database", default="all", help="Only include query records whose planned query path used this database, such as vector_db or pdf_db.")
    query_history_parser.add_argument("--planned-method", default="all", help="Only include query records whose planned query path used this method, such as search_relevant_sentences.")
    query_history_parser.add_argument("--planned-route", default="all", help="Only include query records whose planned query path used this route, such as vector_search or graph_analysis.")
    query_history_parser.add_argument("--min-duration-ms", type=int, default=None, help="Only include query records whose duration was at least this many milliseconds.")
    query_history_parser.add_argument("--max-duration-ms", type=int, default=None, help="Only include query records whose duration was at most this many milliseconds.")
    query_history_parser.add_argument("--min-response-chars", type=int, default=None, help="Only include query records whose response size was at least this many characters.")
    query_history_parser.add_argument("--max-response-chars", type=int, default=None, help="Only include query records whose response size was at most this many characters.")
    query_history_parser.add_argument("--sort", choices=["recent", "oldest", "slowest", "fastest", "longest-response", "shortest-response"], default="recent", help="Order displayed query records before applying --limit (default: recent).")
    query_history_parser.add_argument("--check-empty", action="store_true", help="Exit non-zero when the filtered query history is not empty. Useful for automation that should fail on recent matching errors.")
    query_history_parser.add_argument("--check-max-errors", type=int, default=None, help="Exit non-zero when the filtered window contains more than this many error rows.")
    query_history_parser.add_argument("--check-max-error-rate", type=float, default=None, help="Exit non-zero when the filtered window exceeds this error-rate threshold between 0 and 1.")
    query_history_parser.add_argument("--check-max-duration-ms", type=int, default=None, help="Exit non-zero when the full matching window contains a query slower than this many milliseconds.")
    query_history_parser.add_argument("--check-min-response-chars", type=int, default=None, help="Exit non-zero when the full matching window contains a successful query response shorter than this many characters.")
    query_history_parser.add_argument("--check-min-success-rate", type=float, default=None, help="Exit non-zero when the filtered window falls below this success-rate threshold between 0 and 1.")
    query_history_parser.add_argument("--check-max-empty-query-plans", type=int, default=None, help="Exit non-zero when the full matching window contains more than this many entries with empty query plans.")
    query_history_parser.add_argument("--check-max-no-planned-routes", type=int, default=None, help="Exit non-zero when the full matching window contains more than this many entries without resolved planned routes.")
    query_history_parser.add_argument("--json", action="store_true", help="Print machine-readable query history as JSON.")

    pending_citations_parser = subparsers.add_parser("list_pending_citations", help="List unresolved stub papers that still need uploaded source documents.")
    pending_citations_parser.add_argument("--limit", type=int, default=10, help="How many unresolved stub papers to display (default: 10).")
    pending_citations_parser.add_argument("--json", action="store_true", help="Print machine-readable pending-citation diagnostics as JSON.")

    args = parser.parse_args()

    if args.command == "upload":
        handle_upload_command(args)
    elif args.command == "query":
        handle_query_command(args)
    elif args.command == "diagnose":
        handle_diagnose_command(args)
    elif args.command == "chat":
        handle_chat_command(args)
    elif args.command == "batch-upload":
        handle_batch_upload_command(args)
    elif args.command == "progress":
        handle_progress_command(args)
    elif args.command == "papers":
        handle_papers_command(args)
    elif args.command == "routes":
        handle_routes_command(args)
    elif args.command == "health":
        handle_health_command(args)
    elif args.command == "bootstrap-plan":
        handle_bootstrap_plan_command(args)
    elif args.command == "query-history":
        handle_query_history_command(args)
    elif args.command == "list_pending_citations":
        handle_list_pending_citations_command(args)
    else:
        parser.print_help()

def handle_upload_command(args):
    """Handle the upload command through the kernel service."""
    try:
        kernel = CiteWeaveKernel()

        if args.diagnose:
            print("Running quality diagnosis...")
            diagnosis = kernel.diagnose_document(args.pdf_path)

            print(f"Quality Level: {diagnosis['overall_assessment']['quality_level']}")
            print(f"Is Processable: {diagnosis['overall_assessment']['is_processable']}")

            if diagnosis['overall_assessment']['recommendations']:
                print("Recommendations:")
                for rec in diagnosis['overall_assessment']['recommendations']:
                    print(f"  - {rec}")

            if not diagnosis['overall_assessment']['is_processable']:
                print("Warning: Document may not process well. Continue anyway? (y/n)")
                response = input().strip().lower()
                if response != 'y':
                    sys.exit(1)

        print(f"Processing document: {args.pdf_path}")
        results = kernel.upload_document(args.pdf_path, save_results=True)
        
        # Display results
        stats = results['processing_stats']
        print("\nProcessing completed successfully!")
        print(f"Paper ID: {results['paper_id']}")
        print(f"Total sentences: {stats['total_sentences']}")
        print(f"Sentences with citations: {stats['sentences_with_citations']}")
        print(f"Total citations found: {stats['total_citations']}")
        print(f"Total references: {stats['total_references']}")
        
        # Show some example citations
        citation_sentences = results.get('sentences_with_citations') or results.get('sentences', [])
        sentences_with_cites = [s for s in citation_sentences if s.get('citations')]
        if not citation_sentences and stats.get('sentences_with_citations', 0) == 0:
            print("Warning: No citation-bearing sentences found in results. This document may not contain any extracted citation sentences.")
        if sentences_with_cites:
            print("\nExample sentences with citations:")
            for i, sentence in enumerate(sentences_with_cites[:3]):  # Show first 3
                print(f"\n{i+1}. {sentence.get('sentence_text', '')[:100]}...")
                for cite in sentence.get('citations', []):
                    ref = cite.get('reference', {})
                    print(f"   → {cite.get('intext', '')} → {ref.get('title', 'Unknown')[:50]}... ({ref.get('year', 'Unknown')})")
        
    except Exception as e:
        print(f"Error processing document: {e}")
        logging.exception("Upload command failed")
        sys.exit(1)

def handle_query_command(args):
    """Handle the query command via the kernel service."""
    confirmation = getattr(args, "confirmation", "continue") or "continue"

    try:
        kernel = CiteWeaveKernel()
        print(f"Querying: {args.question}")
        response = kernel.query(args.question, confirmation, source="cli.query")
        print()
        print(response)
    except Exception as e:
        print(f"Error querying CiteWeave: {e}")
        logging.exception("Query command failed")
        sys.exit(1)

def handle_diagnose_command(args):
    """Handle the diagnose command."""
    try:
        kernel = CiteWeaveKernel()
        diagnosis = kernel.diagnose_document(args.pdf_path)
        
        print("=== Document Processing Diagnosis ===")
        print(f"File: {args.pdf_path}")
        print(f"Quality Level: {diagnosis['overall_assessment']['quality_level']}")
        print(f"Is Processable: {diagnosis['overall_assessment']['is_processable']}")
        
        # PDF diagnosis
        pdf_diag = diagnosis.get('pdf_diagnosis', {})
        if pdf_diag:
            print("\n--- PDF Processing ---")
            print(f"Best Quality Score: {pdf_diag.get('best_quality_score', 'Unknown')}")
            print(f"Recommended Engine: {pdf_diag.get('recommended_engine', 'Unknown')}")
            
        # Citation diagnosis  
        cite_diag = diagnosis.get('citation_diagnosis', {})
        if cite_diag:
            print("\n--- Citation Processing ---")
            print(f"References Count: {cite_diag.get('references_count', 0)}")
            print(f"References Extraction Success: {cite_diag.get('references_extraction_success', False)}")
            print(f"Has DOI: {cite_diag.get('has_doi', False)}")
        
        # Recommendations
        if diagnosis['overall_assessment']['recommendations']:
            print("\n--- Recommendations ---")
            for rec in diagnosis['overall_assessment']['recommendations']:
                print(f"  - {rec}")
                
    except Exception as e:
        print(f"Error diagnosing document: {e}")
        logging.exception("Diagnose command failed")
        sys.exit(1)

def handle_chat_command(args):
    """Handle the chat command for interactive multi-turn conversation (stateless AI version)."""
    try:
        kernel = CiteWeaveKernel()
        system = kernel.start_chat_system()
        print("🤖 CiteWeave Multi-Agent Research System (Chat Mode)")
        print("=" * 60)
        print("Type 'exit' or 'quit' to end the chat.")
        print("=" * 60)
        history = []
        collected_data = None  # <-- Initialize collected_data
        expecting_menu = False
        expecting_info_input = False
        last_question = None
        user_input = prompt("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Exiting chat.")
            return
        while True:
            if not user_input:
                user_input = prompt("You: ").strip()
                continue
            spinner_running = True
            def spinner():
                symbols = ['|', '/', '-', '\\']
                idx = 0
                print("AI: ", end="", flush=True)
                while spinner_running:
                    print(f"\b{symbols[idx % 4]}", end="", flush=True)
                    idx += 1
                    time.sleep(0.1)
                print("\b", end="", flush=True)
            spinner_thread = threading.Thread(target=spinner)
            spinner_thread.start()
            try:
                if expecting_menu:
                    response = system.interactive_research_chat(last_question, history, menu_choice=user_input, collected_data=collected_data)
                elif expecting_info_input:
                    response = system.interactive_research_chat(user_input, history, collected_data=collected_data)
                else:
                    response = system.interactive_research_chat(user_input, history)
            finally:
                spinner_running = False
                spinner_thread.join()
                print()
            print(response["text"])
            # Persist the collected data for the next turn
            collected_data = response.get("collected_data")
            if not expecting_menu and not expecting_info_input:
                last_question = user_input
            history.append({"user": user_input, "ai": response["text"]})
            # Handle next state
            if response.get("needs_user_choice"):
                for idx, option in enumerate(response["menu"], 1):
                    print(f"{idx}. {option}")
                user_input = prompt("Enter your choice: ").strip()
                # Exit immediately if the user selects the explicit Exit option
                if user_input == "4":
                    print("Exiting chat.")
                    break
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting chat.")
                    break
                expecting_menu = True
                expecting_info_input = False
            elif response.get("needs_user_input"):
                user_input = prompt("Your input: ").strip()
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting chat.")
                    break
                expecting_menu = False
                expecting_info_input = True
            else:
                # If a final answer is returned, reset collected_data for the new question
                collected_data = None
                user_input = prompt("You: ").strip()
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting chat.")
                    break
                expecting_menu = False
                expecting_info_input = False
    except Exception as e:
        print(f"Error during chat: {e}")
        logging.exception("Chat command failed")
        sys.exit(1)

def handle_batch_upload_command(args):
    """Handle the batch-upload command to process all PDFs in a directory with multiprocessing support."""
    logging.info("START: Batch upload command initiated")
    
    directory = args.directory
    num_processors = args.processors
    use_sequential = args.sequential
    resume_mode = args.resume
    force_restart = args.force_restart
    clear_progress = args.clear_progress
    preserve_order = getattr(args, "preserve_order", False)
    retry_failed = not getattr(args, "skip_failed", False)
    file_timeout_seconds = max(0, int(getattr(args, "file_timeout_seconds", 0) or 0))
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
    
    # Initialize progress tracker
    tracker = BatchUploadTracker(directory)
    
    # Clear progress if requested
    if clear_progress:
        print("Clearing previous progress for this directory...")
        tracker.clear_progress(directory)
        logging.info(f"Cleared progress for directory: {directory}")
    
    # Find all PDF files (recursively)
    print(f"Searching for PDF files in {directory}...")
    logging.info(f"START: Searching for PDF files in {directory}")
    pdf_files = [
        path
        for path in glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True)
        if os.path.isfile(path)
    ]
    logging.info(f"FINISH: Found {len(pdf_files)} PDF files in {directory}")
    
    if not pdf_files:
        print(f"No PDF files found in {directory}.")
        sys.exit(0)
    
    print(f"Found {len(pdf_files)} PDF files in {directory}.")

    print("Hashing PDF content and marking byte-identical duplicates...")
    logging.info("START: Content-hash deduplication for batch upload")
    dedupe_summary = tracker.apply_content_deduplication(pdf_files)
    logging.info("FINISH: Content-hash deduplication summary: %s", dedupe_summary)
    print(
        "   Unique PDF contents: "
        f"{dedupe_summary['unique_content_files']} / {dedupe_summary['total_pdf_files']}"
    )
    print(f"   Duplicate PDF paths skipped: {dedupe_summary['duplicate_files']}")
    if dedupe_summary.get("duplicate_completed_paths"):
        print(f"   Previously completed duplicate paths reclassified: {dedupe_summary['duplicate_completed_paths']}")
    if dedupe_summary.get("hash_error_count"):
        print(f"   Hashing errors: {dedupe_summary['hash_error_count']} files; they will be handled as normal pending files if reachable.")

    # Get pending files based on resume mode.  Pending is content-aware: duplicate
    # paths are aliases of their canonical PDF and are never sent into parsing.
    if resume_mode or not force_restart:
        pending_files = tracker.get_pending_files(pdf_files, force_restart=force_restart, retry_failed=retry_failed)
        pending_files = order_pending_files_for_resumable_batch(pending_files, preserve_order=preserve_order)
        summary = tracker.get_progress_summary()

        if summary["completed"] > 0 or summary["failed"] > 0 or summary.get("duplicate", 0) > 0:
            print("📊 Progress Summary:")
            print(f"   Previously completed unique PDFs: {summary['completed']}")
            print(f"   Duplicate paths skipped: {summary.get('duplicate', 0)}")
            print(f"   Previously failed: {summary['failed']}")
            print(f"   Success rate: {summary['success_rate']:.1f}%")
            print(f"   Unique PDFs to process: {len(pending_files)}")

            if len(pending_files) == 0:
                print("✅ All unique PDF contents have been processed successfully!")
                return
        else:
            print("🆕 No previous progress found. Starting fresh deduplicated batch upload.")
    else:
        pending_files = tracker.get_pending_files(pdf_files, force_restart=True)
        pending_files = order_pending_files_for_resumable_batch(pending_files, preserve_order=preserve_order)
        print("🔄 Force restart mode: Reprocessing canonical unique PDF contents only.")
    
    # Determine processing mode
    if use_sequential:
        print("Using sequential processing (multiprocessing disabled)")
        logging.info("START: Sequential processing mode")
        process_files_sequentially(pending_files, tracker, file_timeout_seconds=file_timeout_seconds)
        logging.info("FINISH: Sequential processing completed")
    else:
        # Validate processor count
        max_processors = multiprocessing.cpu_count()
        if num_processors > max_processors:
            print(f"Warning: Requested {num_processors} processors but only {max_processors} available. Using {max_processors}.")
            num_processors = max_processors
        elif num_processors < 1:
            print(f"Warning: Invalid processor count {num_processors}. Using 1.")
            num_processors = 1
        
        print(f"Using multiprocessing with {num_processors} processors")
        logging.info(f"START: Parallel processing with {num_processors} processors")
        process_files_parallel(pending_files, num_processors, tracker)
        logging.info("FINISH: Parallel processing completed")
    
    # Final summary
    final_summary = tracker.get_progress_summary()
    print("\n📊 Final Summary:")
    print(f"   Total files processed: {final_summary['total_tracked']}")
    print(f"   Successfully completed unique PDFs: {final_summary['completed']}")
    print(f"   Duplicate paths skipped: {final_summary.get('duplicate', 0)}")
    print(f"   Failed: {final_summary['failed']}")
    print(f"   Overall success rate: {final_summary['success_rate']:.1f}%")
    
    logging.info("FINISH: Batch upload command completed")

def order_pending_files_for_resumable_batch(pdf_files, preserve_order=False):
    """Return a stable processing order that avoids large-PDF head-of-line blocking.

    Resume state is recorded per completed file, not per ordinal position. When a
    host or user service restarts mid-file, putting a very large/scanned PDF at
    the front can pin the whole batch forever: every restart retries the same
    long file before any later file gets a chance. Small-first ordering preserves
    the guarantee that every pending PDF remains in the queue while allowing the
    tracker to make durable forward progress between interruptions.
    """
    if preserve_order:
        return list(pdf_files)

    def sort_key(pdf_path):
        try:
            size = os.path.getsize(pdf_path)
        except OSError:
            size = 0
        return (size, str(pdf_path))

    return sorted(pdf_files, key=sort_key)


class _PerFileTimeout(Exception):
    pass


def _raise_per_file_timeout(signum, frame):
    raise _PerFileTimeout("PDF processing exceeded configured per-file timeout")


def process_files_sequentially(pdf_files, tracker, file_timeout_seconds=0):
    """Process files sequentially while preserving real tracker statistics."""
    print("Starting sequential batch upload...")
    success_count = 0
    fail_count = 0
    kernel = CiteWeaveKernel()

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_path}", flush=True)
        started_at = time.time()
        try:
            print(f"Processing document: {pdf_path}", flush=True)
            previous_handler = None
            if file_timeout_seconds:
                previous_handler = signal.signal(signal.SIGALRM, _raise_per_file_timeout)
                signal.alarm(file_timeout_seconds)
            results = kernel.upload_document(pdf_path, save_results=True)
            if file_timeout_seconds:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
            stats = results.get('processing_stats', {})

            print("\nProcessing completed successfully!")
            print(f"Paper ID: {results['paper_id']}")
            print(f"Total sentences: {stats.get('total_sentences', 0)}")
            print(f"Sentences with citations: {stats.get('sentences_with_citations', 0)}")
            print(f"Total citations found: {stats.get('total_citations', 0)}")
            print(f"Total references: {stats.get('total_references', 0)}")

            citation_sentences = results.get('sentences_with_citations') or results.get('sentences', [])
            sentences_with_cites = [
                s for s in citation_sentences if s.get('citations')
            ]
            if not citation_sentences and stats.get('sentences_with_citations', 0) == 0:
                print("Warning: No citation-bearing sentences found in results. This document may not contain any extracted citation sentences.")
            if sentences_with_cites:
                print("\nExample sentences with citations:")
                for i, sentence in enumerate(sentences_with_cites[:3]):
                    print(f"\n{i+1}. {sentence.get('sentence_text', '')[:100]}...")
                    for cite in sentence.get('citations', []):
                        ref = cite.get('reference', {})
                        print(f"   → {cite.get('intext', '')} → {ref.get('title', 'Unknown')[:50]}... ({ref.get('year', 'Unknown')})")

            finished_at = time.time()
            tracker.mark_file_completed(
                pdf_path,
                {
                    'paper_id': results.get('paper_id'),
                    'processed_at': finished_at,
                    'processing_time': finished_at,
                    'duration_seconds': round(finished_at - started_at, 3),
                    'total_sentences': stats.get('total_sentences', 0),
                    'sentences_with_citations': stats.get('sentences_with_citations', 0),
                    'total_citations': stats.get('total_citations', 0),
                    'total_references': stats.get('total_references', 0),
                    'file_hash': tracker.file_hash_for_path(pdf_path),
                },
            )
            success_count += 1

        except Exception as e:
            if file_timeout_seconds:
                signal.alarm(0)
                if previous_handler is not None:
                    signal.signal(signal.SIGALRM, previous_handler)
            print(f"Failed to process {pdf_path}: {e}")
            fail_count += 1
            tracker.mark_file_failed(pdf_path, e)

    print(f"\nBatch upload complete. Success: {success_count}, Failed: {fail_count}")

def process_files_parallel(pdf_files, num_processors, tracker):
    """Process files using multiprocessing with progress tracking."""
    print("Starting parallel batch upload...")
    
    # Set up multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    success_count = 0
    fail_count = 0
    completed_count = 0
    total_files = len(pdf_files)
    
    # Create a partial function with fixed arguments
    worker_func = partial(process_single_pdf_worker, diagnose=False, force=False)
    
    print(f"Processing {total_files} files with {num_processors} processors...")
    print("=" * 60)
    
    with ProcessPoolExecutor(max_workers=num_processors) as executor:
        # Submit all tasks
        future_to_pdf = {executor.submit(worker_func, pdf_path): pdf_path for pdf_path in pdf_files}
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            completed_count += 1
            
            try:
                result = future.result()
                
                if result['status'] == 'success':
                    success_count += 1
                    print(f"[{completed_count}/{total_files}] ✅ {os.path.basename(pdf_path)}")
                    print(f"    Paper ID: {result['paper_id']}")
                    print(f"    Sentences: {result['total_sentences']}, Citations: {result['total_citations']}")
                    result['file_hash'] = tracker.file_hash_for_path(pdf_path)
                    tracker.mark_file_completed(pdf_path, result)
                else:
                    fail_count += 1
                    print(f"[{completed_count}/{total_files}] ❌ {os.path.basename(pdf_path)}")
                    print(f"    Error: {result['error']}")
                    tracker.mark_file_failed(pdf_path, result['error'])
                    
            except Exception as e:
                fail_count += 1
                print(f"[{completed_count}/{total_files}] ❌ {os.path.basename(pdf_path)}")
                print(f"    Exception: {str(e)}")
                tracker.mark_file_failed(pdf_path, e)
    
    print("=" * 60)
    print("Batch upload complete!")
    print(f"Success: {success_count}, Failed: {fail_count}")
    
    if success_count > 0:
        print(f"Success rate: {(success_count/total_files)*100:.1f}%")
    
    if fail_count > 0:
        print(f"Failed files: {fail_count}/{total_files}")
        print("Consider running with --sequential flag for more detailed error messages.")

def handle_papers_command(args):
    """Search/list locally indexed papers without printing absolute PDF paths."""
    kernel = CiteWeaveKernel()
    snapshot = kernel.paper_index_snapshot(
        search=getattr(args, "search", "") or "",
        author=getattr(args, "author", "") or "",
        title=getattr(args, "title", "") or "",
        limit=max(0, getattr(args, "limit", 20)),
        pdf_status=getattr(args, "pdf_status", "all") or "all",
    )

    if getattr(args, "json", False):
        print(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True))
        return

    print("\n=== Local Paper Index ===")
    if snapshot["search_filter"]:
        print(f"Search filter: {snapshot['search_filter']}")
    if snapshot["author_filter"]:
        print(f"Author filter: {snapshot['author_filter']}")
    if snapshot.get("title_filter"):
        print(f"Title filter: {snapshot['title_filter']}")
    if snapshot.get("pdf_status_filter", "all") != "all":
        print(f"PDF status filter: {snapshot['pdf_status_filter']}")
    print(f"Matches: {snapshot['total_matches']} | Displayed: {snapshot['entries_returned']}")

    if not snapshot["papers"]:
        print("No indexed papers matched. Try a broader --search term or rebuild the author-paper index after uploads.")
        return

    for idx, paper in enumerate(snapshot["papers"], 1):
        pdf_marker = "PDF" if paper["pdf_available"] else "no PDF path"
        year = paper["year"] if paper["year"] else "?"
        print(f"\n{idx}. {paper['title']}")
        print(f"   Year: {year} | {pdf_marker}")
        print(f"   Authors: {paper['authors']}")
        print(f"   Paper ID: {paper['paper_id']}")
        if paper.get("journal"):
            print(f"   Journal: {paper['journal']}")

    if snapshot["requested_limit"] and snapshot["total_matches"] > snapshot["entries_returned"]:
        print("\nTip: use --limit 0 to show all matching papers.")


def handle_progress_command(args):
    """Handle the progress command to view batch upload progress status."""
    directory = args.directory

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    kernel = CiteWeaveKernel()
    progress = kernel.progress_summary(directory, clear=args.clear)

    if getattr(args, "json", False):
        print(json.dumps(progress, indent=2, ensure_ascii=False, sort_keys=True))
        return

    print(f"\n=== Batch Upload Progress for {directory} ===")
    if progress["cleared"]:
        print("Progress cleared before reporting.")

    summary = progress["summary"]
    total_pdf_files = progress["total_pdf_files"]
    unique_content_count = progress.get("unique_content_count", total_pdf_files)
    duplicate_count = progress.get("duplicate_count", max(0, total_pdf_files - unique_content_count))
    completion_percent = progress.get("completion_percent")
    if completion_percent is None:
        completion_percent = round((progress["completed_count"] / unique_content_count * 100), 2) if unique_content_count else 0.0

    print(f"Total PDF files discovered: {total_pdf_files}")
    print(f"Unique PDF contents: {unique_content_count}")
    print(f"Duplicate PDF paths skipped: {duplicate_count}")
    print(f"Total files tracked: {summary['total_tracked']}")
    print(f"Completed unique PDFs: {progress['completed_count']}")
    print(f"Failed: {progress['failed_count']}")
    print(f"Pending / resumable: {progress['pending_count']} unique PDFs")
    print(f"  • Not started yet: {progress['not_started_count']}")
    print(f"  • Retryable failed files: {progress['retryable_failed_count']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")

    average_completed_duration_seconds = progress.get("average_completed_duration_seconds")
    estimated_remaining_seconds = progress.get("estimated_remaining_seconds")
    average_completed_duration_human = progress.get("average_completed_duration_human")
    estimated_remaining_human = progress.get("estimated_remaining_human")
    print(f"Completion: {completion_percent:.2f}%")
    if average_completed_duration_seconds is not None:
        display_average = average_completed_duration_human or f"{average_completed_duration_seconds:.1f}s"
        print(f"Observed average time per completed file: {display_average}")
    if estimated_remaining_seconds is not None:
        display_remaining = estimated_remaining_human or f"{estimated_remaining_seconds:.1f}s"
        print(f"Estimated remaining wall time: {display_remaining}")

    aggregate_stats = summary.get("aggregate_stats", {})
    if aggregate_stats:
        print("\n--- Completed Workload ---")
        print(f"Total sentences processed: {aggregate_stats.get('total_sentences', 0)}")
        print(f"Sentences with citations: {aggregate_stats.get('sentences_with_citations', 0)}")
        print(f"Total citations found: {aggregate_stats.get('total_citations', 0)}")
        print(f"Total references found: {aggregate_stats.get('total_references', 0)}")

    last_completed = summary.get("last_completed")
    if last_completed:
        print("\n--- Last Completed File ---")
        print(f"File: {os.path.basename(last_completed['pdf_path'])}")
        if last_completed.get("paper_id"):
            print(f"Paper ID: {last_completed['paper_id']}")

    if summary.get("failure_reasons"):
        print("\n--- Failure Reasons ---")
        for item in summary["failure_reasons"]:
            print(f"- {item['count']} × {item['error']}")

    if progress["failed_files"]:
        print("\n--- Failed Files ---")
        for idx, (pdf_path, error_msg) in enumerate(progress["failed_files"].items(), 1):
            print(f"{idx}. {os.path.basename(pdf_path)}")
            if error_msg:
                print(f"   Error: {error_msg}")

    print("\n--- Retryable Failed Files ---")
    if progress["retryable_failed_files"]:
        for i, pdf_path in enumerate(progress["retryable_failed_files"], 1):
            print(f"{i}. {os.path.basename(pdf_path)}")
    else:
        print("No failed files need a retry.")

    print("\n--- Not Started Yet ---")
    if progress["not_started_files"]:
        for i, pdf_path in enumerate(progress["not_started_files"], 1):
            print(f"{i}. {os.path.basename(pdf_path)}")
    else:
        print("No untouched files remain.")

    print("\n--- Pending Files ---")
    if progress["pending_files"]:
        for i, pdf_path in enumerate(progress["pending_files"], 1):
            print(f"{i}. {os.path.basename(pdf_path)}")
    else:
        print("No files pending processing.")

    if progress.get("duplicate_files"):
        print("\n--- Duplicate Files Skipped ---")
        for i, (pdf_path, canonical_path) in enumerate(list(progress["duplicate_files"].items())[:50], 1):
            print(f"{i}. {os.path.basename(pdf_path)} → {os.path.basename(canonical_path)}")
        if len(progress["duplicate_files"]) > 50:
            print(f"... {len(progress['duplicate_files']) - 50} more duplicate paths omitted")

    if progress["pending_count"] > 0:
        print("\nTip: run batch-upload --resume to continue remaining unique files, including retries for previous failures.")

    if getattr(args, "show_completed", False) and progress["completed_files"]:
        print("\n--- Completed Files ---")
        for i, pdf_path in enumerate(progress["completed_files"], 1):
            print(f"{i}. {os.path.basename(pdf_path)}")


def handle_routes_command(args):
    """Display the active route configuration for diagnostics."""
    kernel = CiteWeaveKernel()
    config = kernel.routes_snapshot()
    validation = {
        "ok": not config.get("ignored_alias_overrides") and not config.get("ignored_priority_overrides") and not config.get("addon_config_issues"),
        "ignored_alias_overrides": len(config.get("ignored_alias_overrides", [])),
        "ignored_priority_overrides": len(config.get("ignored_priority_overrides", [])),
        "addon_config_issues": len(config.get("addon_config_issues", [])),
        "addon_config_sources": len(config.get("addon_config_paths", [])),
    }

    if getattr(args, "check", False):
        if getattr(args, "json", False):
            print(json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True))
        else:
            status_text = "ok" if validation["ok"] else "invalid"
            print(f"Route configuration check: {status_text}")
            print(f"  ignored alias overrides: {validation['ignored_alias_overrides']}")
            print(f"  ignored priority overrides: {validation['ignored_priority_overrides']}")
            print(f"  addon config issues: {validation['addon_config_issues']}")
            print(f"  addon config sources: {validation['addon_config_sources']}")
            if not validation["ok"]:
                print("  Tip: run `.venv/bin/citeweave routes` for the full diagnostic report.")
        if not validation["ok"]:
            raise SystemExit(1)
        return

    if getattr(args, "json", False):
        print(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True))
        return

    print("\n=== CiteWeave Route Configuration ===\n")
    print(f"Default route: {config['default_route']}")

    print("\nValid routes:")
    for route in sorted(config["valid_routes"]):
        marker = " (default)" if route == config["default_route"] else ""
        print(f"  {route}{marker}")

    if config["aliases"]:
        print(f"\nRoute aliases ({len(config['aliases'])} active):")
        for alias, canonical in sorted(config["aliases"].items()):
            if alias != canonical:
                print(f"  {alias} → {canonical}")

    if config["priority_map"]:
        print("\nPriority → Route mapping:")
        for priority, route in sorted(config["priority_map"].items()):
            print(f"  {priority} → {route}")

    if config["alias_overrides"]:
        print(f"\nAlias overrides ({len(config['alias_overrides'])} active):")
        for alias, canonical in sorted(config["alias_overrides"].items()):
            source = "addon" if alias in config.get("addon_alias_overrides", {}) else "env"
            print(f"  {alias} → {canonical} [{source}]")

    if config["priority_overrides"]:
        print(f"\nPriority overrides ({len(config['priority_overrides'])} active):")
        for priority, route in sorted(config["priority_overrides"].items()):
            source = "addon" if priority in config.get("addon_priority_overrides", {}) else "env"
            print(f"  {priority} → {route} [{source}]")

    if config["ignored_alias_overrides"]:
        print(f"\nIgnored alias overrides ({len(config['ignored_alias_overrides'])}):")
        for entry in config["ignored_alias_overrides"]:
            route = entry.get("route", "<missing>")
            key = entry.get("key", "<missing>")
            print(f"  {key} → {route}  [{entry['reason']}]")

    if config["ignored_priority_overrides"]:
        print(f"\nIgnored priority overrides ({len(config['ignored_priority_overrides'])}):")
        for entry in config["ignored_priority_overrides"]:
            route = entry.get("route", "<missing>")
            key = entry.get("key", "<missing>")
            print(f"  {key} → {route}  [{entry['reason']}]")

    if config["addon_config_paths"]:
        print(f"\nAddon config sources ({len(config['addon_config_paths'])}):")
        for path in config["addon_config_paths"]:
            print(f"  {path}")

    if config["addon_config_issues"]:
        print(f"\nAddon config issues ({len(config['addon_config_issues'])}):")
        for issue in config["addon_config_issues"]:
            loc = f" ({issue.get('path', '')})" if issue.get("path") else ""
            detail = f": {issue.get('detail')}" if issue.get("detail") else ""
            print(f"  {issue['reason']}{loc}{detail}")

    print()


def handle_health_command(args):
    """Display machine-readable CiteWeave service and environment health."""
    kernel = CiteWeaveKernel()
    snapshot = kernel.health_snapshot()

    if getattr(args, "json", False):
        print(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True))
        return

    summary = snapshot.get("summary", {})
    env = snapshot.get("env", {})
    files = snapshot.get("files", {})
    services = snapshot.get("services", {})

    print("\n=== CiteWeave Health Snapshot ===\n")
    print(f"Overall status: {summary.get('overall_status', 'unknown')}")
    print(f"Project root: {snapshot.get('project_root', '')}")
    print(f"LLM provider: {env.get('llm_provider') or 'unknown'}")
    if env.get("llm_model"):
        print(f"LLM model: {env['llm_model']}")
    if env.get("gateway_base"):
        print(f"Gateway base: {env['gateway_base']}")

    action_items = summary.get("action_items", [])
    if action_items:
        print("\nRecommended next actions:")
        for item in action_items:
            print(f"  - {item}")

    if files:
        print("\nFiles:")
        for name, exists in files.items():
            status = "present" if exists else "missing"
            print(f"  {name}: {status}")

    if services:
        print("\nServices:")
        for name, result in services.items():
            if result is None:
                continue
            state = "ok" if result.get("ok") else "down"
            status = result.get("status")
            status_text = f"status={status}" if status is not None else "status=unavailable"
            detail = f" error={result['error']}" if result.get("error") else ""
            print(f"  {name}: {state} ({status_text}){detail}")

    print()


def handle_bootstrap_plan_command(args):
    """Display recommended local and OpenClaw bootstrap steps."""
    kernel = CiteWeaveKernel()
    plan = kernel.bootstrap_plan()

    if getattr(args, "json", False):
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        return

    print("\n=== CiteWeave Bootstrap Plan ===")
    for section_name, section in plan.items():
        print(f"\n[{section_name}]")
        print(f"script: {section.get('script', '')}")
        next_steps = section.get("next_steps", [])
        if next_steps:
            print("next steps:")
            for step in next_steps:
                print(f"  - {step}")
    print()


def handle_list_pending_citations_command(args):
    """Display unresolved stub papers that still need uploaded source documents."""
    kernel = CiteWeaveKernel()
    snapshot = kernel.list_pending_citations_snapshot(limit=max(0, getattr(args, "limit", 10)))

    if getattr(args, "json", False):
        print(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True))
        return

    print("\n=== CiteWeave Pending Citations ===\n")
    if snapshot.get("error"):
        print(f"Error: {snapshot['error']}")
        raise SystemExit(1)

    stats = snapshot.get("network_stats", {})
    print(f"Requested limit: {snapshot.get('requested_limit', 0)}")
    if stats:
        print(f"Total papers: {stats.get('total_papers', 0)}")
        print(f"Uploaded papers: {stats.get('uploaded_papers', 0)}")
        print(f"Stub papers: {stats.get('stub_papers', 0)}")
        print(f"Citation relations: {stats.get('total_citation_relations', 0)}")
    print(f"Pending citations available: {snapshot.get('total_stub_papers', 0)}")

    stub_papers = snapshot.get("stub_papers", []) or []
    if not stub_papers:
        print("No pending citations found.")
        print()
        return

    print("\nTop unresolved cited papers:")
    for index, stub in enumerate(stub_papers, start=1):
        title = stub.get("title") or "Untitled"
        year = stub.get("year") or "unknown year"
        cited_by = stub.get("cited_by_count", 0)
        print(f"  {index}. {title} ({year})")
        print(f"     cited by: {cited_by}")
        if stub.get("authors"):
            authors = stub["authors"] if isinstance(stub["authors"], str) else ", ".join(stub["authors"])
            print(f"     authors: {authors}")
        if stub.get("paper_id"):
            print(f"     paper_id: {stub['paper_id']}")
    print()



def _format_query_history_timestamp(timestamp):
    if not isinstance(timestamp, (int, float)):
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")



def _format_relative_age(timestamp):
    if not isinstance(timestamp, (int, float)):
        return ""

    delta_seconds = max(0, int(time.time() - timestamp))
    if delta_seconds < 60:
        return "just now"
    if delta_seconds < 3600:
        minutes = delta_seconds // 60
        return f"{minutes}m ago"
    if delta_seconds < 86400:
        hours = delta_seconds // 3600
        minutes = (delta_seconds % 3600) // 60
        return f"{hours}h {minutes}m ago" if minutes else f"{hours}h ago"
    days = delta_seconds // 86400
    hours = (delta_seconds % 86400) // 3600
    return f"{days}d {hours}h ago" if hours else f"{days}d ago"



def _format_query_plan_summary(entry):
    database_route_map = {
        "graph_db": "graph_analysis",
        "vector_db": "vector_search",
        "pdf_db": "pdf_analysis",
    }
    databases = [value for value in (entry.get("query_plan_databases") or []) if isinstance(value, str) and value]
    methods = [value for value in (entry.get("query_plan_methods") or []) if isinstance(value, str) and value]
    explicit_routes = [value for value in (entry.get("query_plan_routes") or []) if isinstance(value, str) and value]
    routes = []
    for route in explicit_routes:
        if route not in routes:
            routes.append(route)
    inferred_routes = []
    for database in databases:
        route = database_route_map.get(database)
        if route and route not in routes and route not in inferred_routes:
            inferred_routes.append(route)
    routes.extend(inferred_routes)
    parts = []
    if routes:
        route_label = "routes=" + ", ".join(routes)
        if inferred_routes and not explicit_routes:
            route_label += " (inferred from db)"
        elif inferred_routes:
            route_label += f" (+{len(inferred_routes)} inferred from db)"
        parts.append(route_label)
    if databases:
        parts.append("db=" + ", ".join(databases))
    if methods:
        parts.append("methods=" + ", ".join(methods[:3]))
        if len(methods) > 3:
            parts[-1] += f" (+{len(methods) - 3} more)"
    return " | ".join(parts)



def handle_query_history_command(args):
    """Display recent query telemetry from the local query history log."""
    kernel = CiteWeaveKernel()
    since_hours = getattr(args, "since_hours", None)
    snapshot = kernel.query_history_snapshot(
        limit=max(0, getattr(args, "limit", 10)),
        status=getattr(args, "status", "all") or "all",
        source=getattr(args, "source", "all") or "all",
        confirmation=getattr(args, "confirmation", "all") or "all",
        satisfaction=getattr(args, "satisfaction", "all") or "all",
        since_hours=since_hours,
        contains=getattr(args, "contains", "") or "",
        question_contains=getattr(args, "question_contains", "") or "",
        error_contains=getattr(args, "error_contains", "") or "",
        response_contains=getattr(args, "response_contains", "") or "",
        planned_database=getattr(args, "planned_database", "all") or "all",
        planned_method=getattr(args, "planned_method", "all") or "all",
        planned_route=getattr(args, "planned_route", "all") or "all",
        min_duration_ms=getattr(args, "min_duration_ms", None),
        max_duration_ms=getattr(args, "max_duration_ms", None),
        min_response_chars=getattr(args, "min_response_chars", None),
        max_response_chars=getattr(args, "max_response_chars", None),
        sort_order=getattr(args, "sort", "recent"),
    )

    check_empty = getattr(args, "check_empty", False)
    check_max_errors = getattr(args, "check_max_errors", None)
    check_max_error_rate = getattr(args, "check_max_error_rate", None)
    check_max_duration_ms = getattr(args, "check_max_duration_ms", None)
    check_min_response_chars = getattr(args, "check_min_response_chars", None)
    check_min_success_rate = getattr(args, "check_min_success_rate", None)
    check_max_empty_query_plans = getattr(args, "check_max_empty_query_plans", None)
    check_max_no_planned_routes = getattr(args, "check_max_no_planned_routes", None)
    if (
        check_empty
        or check_max_errors is not None
        or check_max_error_rate is not None
        or check_max_duration_ms is not None
        or check_min_response_chars is not None
        or check_min_success_rate is not None
        or check_max_empty_query_plans is not None
        or check_max_no_planned_routes is not None
    ):
        matching_entries_total = snapshot.get("matching_entries_total", 0)
        error_count = snapshot.get("matching_error_count", snapshot.get("error_count", 0))
        error_rate = snapshot.get("matching_error_rate", snapshot.get("error_rate"))
        success_rate = snapshot.get("matching_success_rate", snapshot.get("success_rate"))
        max_duration_ms = snapshot.get("matching_max_duration_ms", snapshot.get("max_duration_ms"))
        shortest_success_response_chars = snapshot.get("matching_min_success_response_chars", snapshot.get("min_success_response_chars"))
        empty_query_plan_count = snapshot.get("matching_empty_query_plan_count", snapshot.get("empty_query_plan_count", 0))
        no_planned_route_count = snapshot.get("matching_no_planned_route_count", snapshot.get("no_planned_route_count", 0))
        failure_reasons = []
        if check_empty and matching_entries_total != 0:
            failure_reasons.append("not empty")
        if check_max_errors is not None and error_count > check_max_errors:
            failure_reasons.append("too many errors")
        if check_max_error_rate is not None:
            normalized_threshold = max(0.0, min(1.0, check_max_error_rate))
            if error_rate is not None and error_rate > normalized_threshold:
                failure_reasons.append("error rate too high")
        if check_max_duration_ms is not None and max_duration_ms is not None and max_duration_ms > check_max_duration_ms:
            failure_reasons.append("query too slow")
        if (
            check_min_response_chars is not None
            and shortest_success_response_chars is not None
            and shortest_success_response_chars < check_min_response_chars
        ):
            failure_reasons.append("response too short")
        if check_min_success_rate is not None:
            normalized_success_threshold = max(0.0, min(1.0, check_min_success_rate))
            if success_rate is not None and success_rate < normalized_success_threshold:
                failure_reasons.append("success rate too low")
        if check_max_empty_query_plans is not None and empty_query_plan_count > check_max_empty_query_plans:
            failure_reasons.append("too many empty query plans")
        if check_max_no_planned_routes is not None and no_planned_route_count > check_max_no_planned_routes:
            failure_reasons.append("too many entries without planned routes")
        validation = {
            "ok": not failure_reasons,
            "failure_reasons": failure_reasons,
            "matching_entries_total": matching_entries_total,
            "error_count": error_count,
            "error_rate": error_rate,
            "success_rate": success_rate,
            "max_duration_ms": max_duration_ms,
            "shortest_success_response_chars": shortest_success_response_chars,
            "empty_query_plan_count": empty_query_plan_count,
            "no_planned_route_count": no_planned_route_count,
            "check_empty": check_empty,
            "check_max_errors": check_max_errors,
            "check_max_error_rate": max(0.0, min(1.0, check_max_error_rate)) if check_max_error_rate is not None else None,
            "check_max_duration_ms": check_max_duration_ms,
            "check_min_response_chars": check_min_response_chars,
            "check_min_success_rate": max(0.0, min(1.0, check_min_success_rate)) if check_min_success_rate is not None else None,
            "check_max_empty_query_plans": check_max_empty_query_plans,
            "check_max_no_planned_routes": check_max_no_planned_routes,
            "status_filter": snapshot.get("status_filter", "all"),
            "source_filter": snapshot.get("source_filter", "all"),
            "confirmation_filter": snapshot.get("confirmation_filter", "all"),
            "satisfaction_filter": snapshot.get("satisfaction_filter", "all"),
            "contains_filter": snapshot.get("contains_filter", ""),
            "question_contains_filter": snapshot.get("question_contains_filter", ""),
            "error_contains_filter": snapshot.get("error_contains_filter", ""),
            "response_contains_filter": snapshot.get("response_contains_filter", ""),
            "planned_database_filter": snapshot.get("planned_database_filter", "all"),
            "planned_method_filter": snapshot.get("planned_method_filter", "all"),
            "planned_route_filter": snapshot.get("planned_route_filter", "all"),
            "min_duration_ms_filter": snapshot.get("min_duration_ms_filter"),
            "max_duration_ms_filter": snapshot.get("max_duration_ms_filter"),
            "min_response_chars_filter": snapshot.get("min_response_chars_filter"),
            "max_response_chars_filter": snapshot.get("max_response_chars_filter"),
            "sort_order": snapshot.get("sort_order", "recent"),
            "since_hours": snapshot.get("since_hours"),
        }
        if getattr(args, "json", False):
            print(json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True))
        else:
            status_text = "ok" if validation["ok"] else ", ".join(validation["failure_reasons"])
            print(f"Query history check: {status_text}")
            print(f"  matching entries: {validation['matching_entries_total']}")
            print(f"  error count: {validation['error_count']}")
            if validation["error_rate"] is not None:
                print(f"  error rate: {validation['error_rate']}")
            if validation["success_rate"] is not None:
                print(f"  success rate: {validation['success_rate']}")
            if validation["max_duration_ms"] is not None:
                print(f"  slowest query: {validation['max_duration_ms']} ms")
            if validation["shortest_success_response_chars"] is not None:
                print(f"  shortest successful response: {validation['shortest_success_response_chars']} chars")
            print(f"  empty query plans: {validation['empty_query_plan_count']}")
            print(f"  entries without planned routes: {validation['no_planned_route_count']}")
            print(f"  status filter: {validation['status_filter']}")
            print(f"  source filter: {validation['source_filter']}")
            print(f"  confirmation filter: {validation['confirmation_filter']}")
            if validation["sort_order"] != "recent":
                print(f"  sort order: {validation['sort_order']}")
            if validation["check_empty"]:
                print("  empty check: enabled")
            if validation["check_max_errors"] is not None:
                print(f"  max errors check: {validation['check_max_errors']}")
            if validation["check_max_error_rate"] is not None:
                print(f"  max error rate check: {validation['check_max_error_rate']}")
            if validation["check_max_duration_ms"] is not None:
                print(f"  max duration check: {validation['check_max_duration_ms']} ms")
            if validation["check_min_response_chars"] is not None:
                print(f"  minimum successful response check: {validation['check_min_response_chars']} chars")
            if validation["check_min_success_rate"] is not None:
                print(f"  min success rate check: {validation['check_min_success_rate']}")
            if validation["check_max_empty_query_plans"] is not None:
                print(f"  max empty query plans check: {validation['check_max_empty_query_plans']}")
            if validation["check_max_no_planned_routes"] is not None:
                print(f"  max entries without planned routes check: {validation['check_max_no_planned_routes']}")
            if validation["satisfaction_filter"] != "all":
                print(f"  satisfaction filter: {validation['satisfaction_filter']}")
            if validation["contains_filter"]:
                print(f"  contains filter: {validation['contains_filter']}")
            if validation["question_contains_filter"]:
                print(f"  question filter: {validation['question_contains_filter']}")
            if validation["error_contains_filter"]:
                print(f"  error filter: {validation['error_contains_filter']}")
            if validation["response_contains_filter"]:
                print(f"  response filter: {validation['response_contains_filter']}")
            if validation["planned_database_filter"] != "all":
                print(f"  planned database filter: {validation['planned_database_filter']}")
            if validation["planned_method_filter"] != "all":
                print(f"  planned method filter: {validation['planned_method_filter']}")
            if validation["planned_route_filter"] != "all":
                print(f"  planned route filter: {validation['planned_route_filter']}")
            if validation["min_duration_ms_filter"] is not None:
                print(f"  minimum duration: {validation['min_duration_ms_filter']} ms")
            if validation.get("max_duration_ms_filter") is not None:
                print(f"  maximum duration: {validation['max_duration_ms_filter']} ms")
            if validation.get("min_response_chars_filter") is not None:
                print(f"  minimum response size: {validation['min_response_chars_filter']} chars")
            if validation.get("max_response_chars_filter") is not None:
                print(f"  maximum response size: {validation['max_response_chars_filter']} chars")
            if validation["since_hours"] is not None:
                print(f"  time window: last {validation['since_hours']} hours")
        if not validation["ok"]:
            raise SystemExit(1)
        return

    if getattr(args, "json", False):
        print(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True))
        return

    print("\n=== CiteWeave Query History ===\n")
    print(f"Log file: {snapshot.get('log_file', '')}")
    print(f"Requested limit: {snapshot.get('requested_limit', 0)}")
    print(f"Status filter: {snapshot.get('status_filter', 'all')}")
    print(f"Source filter: {snapshot.get('source_filter', 'all')}")
    print(f"Confirmation filter: {snapshot.get('confirmation_filter', 'all')}")
    satisfaction_filter = snapshot.get("satisfaction_filter", "all")
    if satisfaction_filter != "all":
        print(f"Satisfaction filter: {satisfaction_filter}")
    contains_filter = snapshot.get("contains_filter", "")
    if contains_filter:
        print(f"Contains filter: {contains_filter}")
    question_contains_filter = snapshot.get("question_contains_filter", "")
    if question_contains_filter:
        print(f"Question filter: {question_contains_filter}")
    error_contains_filter = snapshot.get("error_contains_filter", "")
    if error_contains_filter:
        print(f"Error filter: {error_contains_filter}")
    response_contains_filter = snapshot.get("response_contains_filter", "")
    if response_contains_filter:
        print(f"Response filter: {response_contains_filter}")
    if snapshot.get("since_hours") is not None:
        print(f"Time window: last {snapshot.get('since_hours')} hours")
    sort_order = snapshot.get("sort_order", "recent")
    if sort_order != "recent":
        print(f"Sort order: {sort_order}")
    planned_database_filter = snapshot.get("planned_database_filter", "all")
    planned_method_filter = snapshot.get("planned_method_filter", "all")
    planned_route_filter = snapshot.get("planned_route_filter", "all")
    if planned_database_filter != "all":
        print(f"Planned database filter: {planned_database_filter}")
    if planned_method_filter != "all":
        print(f"Planned method filter: {planned_method_filter}")
    if planned_route_filter != "all":
        print(f"Planned route filter: {planned_route_filter}")
    min_duration_filter = snapshot.get("min_duration_ms_filter")
    if min_duration_filter is not None:
        print(f"Minimum duration filter: {min_duration_filter} ms")
    max_duration_filter = snapshot.get("max_duration_ms_filter")
    if max_duration_filter is not None:
        print(f"Maximum duration filter: {max_duration_filter} ms")
    min_response_filter = snapshot.get("min_response_chars_filter")
    if min_response_filter is not None:
        print(f"Minimum response size filter: {min_response_filter} chars")
    max_response_filter = snapshot.get("max_response_chars_filter")
    if max_response_filter is not None:
        print(f"Maximum response size filter: {max_response_filter} chars")
    print(f"Entries returned: {snapshot.get('entries_returned', 0)}")
    if snapshot.get("matching_entries_total") is not None:
        print(f"Matching entries before limit: {snapshot.get('matching_entries_total', 0)}")
    print(f"Successful queries: {snapshot.get('success_count', 0)}")
    print(f"Failed queries: {snapshot.get('error_count', 0)}")
    if snapshot.get("success_rate") is not None:
        print(f"Success rate: {snapshot.get('success_rate')}")
    if snapshot.get("error_rate") is not None:
        print(f"Error rate: {snapshot.get('error_rate')}")
    if snapshot.get("matching_entries_total") != snapshot.get("entries_returned"):
        print(f"Matching successful queries: {snapshot.get('matching_success_count', 0)}")
        print(f"Matching failed queries: {snapshot.get('matching_error_count', 0)}")
        if snapshot.get("matching_success_rate") is not None:
            print(f"Matching success rate: {snapshot.get('matching_success_rate')}")
        if snapshot.get("matching_error_rate") is not None:
            print(f"Matching error rate: {snapshot.get('matching_error_rate')}")
    print(f"Corrupt rows skipped into diagnostics: {snapshot.get('corrupt_count', 0)}")

    average_duration_ms = snapshot.get("average_duration_ms")
    max_duration_ms = snapshot.get("max_duration_ms")
    matching_average_duration_ms = snapshot.get("matching_average_duration_ms")
    matching_max_duration_ms = snapshot.get("matching_max_duration_ms")
    average_response_chars = snapshot.get("average_response_chars")
    max_response_chars = snapshot.get("max_response_chars")
    matching_average_response_chars = snapshot.get("matching_average_response_chars")
    matching_max_response_chars = snapshot.get("matching_max_response_chars")
    if average_duration_ms is not None:
        print(f"Average duration: {average_duration_ms} ms")
    if max_duration_ms is not None:
        print(f"Slowest query: {max_duration_ms} ms")
    if average_response_chars is not None:
        print(f"Average response size: {average_response_chars} chars")
    if max_response_chars is not None:
        print(f"Longest response: {max_response_chars} chars")
    if snapshot.get("matching_entries_total") != snapshot.get("entries_returned"):
        if matching_average_duration_ms is not None:
            print(f"Matching average duration: {matching_average_duration_ms} ms")
        if matching_max_duration_ms is not None:
            print(f"Matching slowest query: {matching_max_duration_ms} ms")
        if matching_average_response_chars is not None:
            print(f"Matching average response size: {matching_average_response_chars} chars")
        if matching_max_response_chars is not None:
            print(f"Matching longest response: {matching_max_response_chars} chars")

    latest_status = snapshot.get("latest_status")
    latest_question = snapshot.get("latest_question")
    latest_source = snapshot.get("latest_source")
    if latest_status or latest_question:
        source_suffix = f" [{latest_source}]" if latest_source else ""
        print(f"Latest query: {latest_status or 'unknown'}{source_suffix} - {latest_question or ''}")

    latest_error = snapshot.get("latest_error")
    if latest_error:
        print(f"Latest error: {latest_error}")
    latest_response_preview = snapshot.get("latest_response_preview")
    if latest_response_preview:
        print(f"Latest response preview: {latest_response_preview}")

    source_breakdown = snapshot.get("source_breakdown") or []
    if source_breakdown:
        print("Sources:")
        for item in source_breakdown:
            print(f"  - {item['source']}: {item['count']}")

    matching_source_breakdown = snapshot.get("matching_source_breakdown") or []
    if matching_source_breakdown and matching_source_breakdown != source_breakdown:
        print("Matching-window sources:")
        for item in matching_source_breakdown:
            print(f"  - {item['source']}: {item['count']}")

    confirmation_breakdown = snapshot.get("confirmation_breakdown") or []
    if confirmation_breakdown:
        print("Confirmations:")
        for item in confirmation_breakdown:
            print(f"  - {item['confirmation']}: {item['count']}")

    matching_confirmation_breakdown = snapshot.get("matching_confirmation_breakdown") or []
    if matching_confirmation_breakdown and matching_confirmation_breakdown != confirmation_breakdown:
        print("Matching-window confirmations:")
        for item in matching_confirmation_breakdown:
            print(f"  - {item['confirmation']}: {item['count']}")

    satisfaction_breakdown = snapshot.get("satisfaction_breakdown") or []
    if satisfaction_breakdown:
        print("Satisfaction:")
        for item in satisfaction_breakdown:
            print(f"  - {item['satisfaction']}: {item['count']}")

    matching_satisfaction_breakdown = snapshot.get("matching_satisfaction_breakdown") or []
    if matching_satisfaction_breakdown and matching_satisfaction_breakdown != satisfaction_breakdown:
        print("Matching-window satisfaction:")
        for item in matching_satisfaction_breakdown:
            print(f"  - {item['satisfaction']}: {item['count']}")

    error_breakdown = snapshot.get("error_breakdown") or []
    if error_breakdown:
        print("Errors:")
        for item in error_breakdown[:8]:
            print(f"  - {item['error']}: {item['count']}")

    matching_error_breakdown = snapshot.get("matching_error_breakdown") or []
    if matching_error_breakdown and matching_error_breakdown != error_breakdown:
        print("Matching-window errors:")
        for item in matching_error_breakdown[:8]:
            print(f"  - {item['error']}: {item['count']}")

    query_plan_database_breakdown = snapshot.get("query_plan_database_breakdown") or []
    if query_plan_database_breakdown:
        print("Planned databases:")
        for item in query_plan_database_breakdown:
            print(f"  - {item['database']}: {item['count']}")

    matching_database_breakdown = snapshot.get("matching_query_plan_database_breakdown") or []
    if matching_database_breakdown and matching_database_breakdown != query_plan_database_breakdown:
        print("Matching-window planned databases:")
        for item in matching_database_breakdown:
            print(f"  - {item['database']}: {item['count']}")

    query_plan_method_breakdown = snapshot.get("query_plan_method_breakdown") or []
    if query_plan_method_breakdown:
        print("Planned methods:")
        for item in query_plan_method_breakdown[:8]:
            print(f"  - {item['method']}: {item['count']}")

    matching_method_breakdown = snapshot.get("matching_query_plan_method_breakdown") or []
    if matching_method_breakdown and matching_method_breakdown != query_plan_method_breakdown:
        print("Matching-window planned methods:")
        for item in matching_method_breakdown[:8]:
            print(f"  - {item['method']}: {item['count']}")

    query_plan_route_breakdown = snapshot.get("query_plan_route_breakdown") or []
    if query_plan_route_breakdown:
        print("Planned routes:")
        for item in query_plan_route_breakdown:
            print(f"  - {item['route']}: {item['count']}")

    no_planned_route_count = snapshot.get("no_planned_route_count", 0)
    empty_query_plan_count = snapshot.get("empty_query_plan_count", 0)
    if no_planned_route_count:
        print(f"Entries without planned routes: {no_planned_route_count}")
    if empty_query_plan_count:
        print(f"Entries with empty query plans: {empty_query_plan_count}")

    matching_route_breakdown = snapshot.get("matching_query_plan_route_breakdown") or []
    if matching_route_breakdown and matching_route_breakdown != query_plan_route_breakdown:
        print("Matching-window planned routes:")
        for item in matching_route_breakdown:
            print(f"  - {item['route']}: {item['count']}")

    matching_no_planned_route_count = snapshot.get("matching_no_planned_route_count", 0)
    matching_empty_query_plan_count = snapshot.get("matching_empty_query_plan_count", 0)
    if matching_no_planned_route_count and matching_no_planned_route_count != no_planned_route_count:
        print(f"Matching-window entries without planned routes: {matching_no_planned_route_count}")
    if matching_empty_query_plan_count and matching_empty_query_plan_count != empty_query_plan_count:
        print(f"Matching-window entries with empty query plans: {matching_empty_query_plan_count}")

    entries = snapshot.get("entries", [])
    if not entries:
        print("\nNo query history recorded yet.")
        print()
        return

    print("\nRecent entries:")
    for idx, entry in enumerate(entries, 1):
        status = entry.get("status", "unknown")
        duration = entry.get("duration_ms")
        question = entry.get("question") or entry.get("raw_line") or ""
        source = entry.get("source")
        duration_text = f" ({duration} ms)" if isinstance(duration, int) else ""
        source_text = f" {{{source}}}" if source else ""
        timestamp_text = _format_query_history_timestamp(entry.get("timestamp"))
        relative_age = _format_relative_age(entry.get("timestamp"))
        when_text = f" at {timestamp_text}" if timestamp_text else ""
        age_text = f" [{relative_age}]" if relative_age else ""
        print(f"{idx}. [{status}]{source_text}{duration_text}{when_text}{age_text} {question}")
        error_text = entry.get("error")
        if error_text:
            print(f"    error: {error_text}")
        response_chars = entry.get("response_chars")
        response_preview = entry.get("response_preview")
        if isinstance(response_chars, int):
            print(f"    response chars: {response_chars}")
        if response_preview:
            print(f"    response: {response_preview}")
        plan_summary = _format_query_plan_summary(entry)
        if plan_summary:
            print(f"    plan: {plan_summary}")

    print()


if __name__ == "__main__":
    main() 
