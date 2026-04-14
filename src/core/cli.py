"""
cli.py
Command-line interface for the argument graph project.
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

import os
import sys
from prompt_toolkit import prompt
import warnings
warnings.filterwarnings("ignore", message=".*found in sys.modules after import of package.*", category=RuntimeWarning)


def find_project_root():
    cur = os.path.abspath(os.getcwd())
    while cur != "/" and not os.path.exists(os.path.join(cur, "README.md")):
        cur = os.path.dirname(cur)
    return cur

project_root = find_project_root()
if os.getcwd() != project_root:
    os.chdir(project_root)
    print(f"[INFO] Changed working directory to project root: {project_root}")

# Set up logging based on environment variable (before importing other modules)
env = os.environ.get("CITEWEAVE_ENV", "production").lower()
import logging
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
    parser = argparse.ArgumentParser(description="Argument Graph CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload and process a PDF document with sentence-level citation analysis.")
    upload_parser.add_argument("pdf_path", type=str, help="Path to the PDF file.")
    upload_parser.add_argument("--diagnose", action="store_true", help="Run quality diagnosis before processing.")
    upload_parser.add_argument("--force", action="store_true", help="Force reprocessing even if cached results exist.")

    # Query command  
    query_parser = subparsers.add_parser("query", help="Query the argument graph.")
    query_parser.add_argument("question", type=str, help="Question to ask.")
    query_parser.add_argument(
        "--confirmation",
        default="continue",
        help="User confirmation mode to pass into the research workflow (default: continue)."
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat with the multi-agent research system.")

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

    # Progress status command
    progress_parser = subparsers.add_parser("progress", help="View batch upload progress status.")
    progress_parser.add_argument("directory", type=str, help="Path to the directory to check progress for.")
    progress_parser.add_argument("--clear", action="store_true", help="Clear progress for this directory.")
    progress_parser.add_argument("--json", action="store_true", help="Print machine-readable progress information as JSON.")
    progress_parser.add_argument("--show-completed", action="store_true", help="Also list completed files in text output.")

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
    query_history_parser.add_argument("--since-hours", type=float, default=None, help="Only include query records from the last N hours.")
    query_history_parser.add_argument("--contains", default="", help="Only include query records whose question or error text contains this substring.")
    query_history_parser.add_argument("--planned-database", default="all", help="Only include query records whose planned query path used this database, such as vector_db or pdf_db.")
    query_history_parser.add_argument("--planned-method", default="all", help="Only include query records whose planned query path used this method, such as search_relevant_sentences.")
    query_history_parser.add_argument("--json", action="store_true", help="Print machine-readable query history as JSON.")

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
    elif args.command == "routes":
        handle_routes_command(args)
    elif args.command == "health":
        handle_health_command(args)
    elif args.command == "bootstrap-plan":
        handle_bootstrap_plan_command(args)
    elif args.command == "query-history":
        handle_query_history_command(args)
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
        print(f"\nProcessing completed successfully!")
        print(f"Paper ID: {results['paper_id']}")
        print(f"Total sentences: {stats['total_sentences']}")
        print(f"Sentences with citations: {stats['sentences_with_citations']}")
        print(f"Total citations found: {stats['total_citations']}")
        print(f"Total references: {stats['total_references']}")
        
        # Show some example citations
        sentences_with_cites = [s for s in results.get('sentences_with_citations', []) if s.get('citations')]
        if not results.get('sentences_with_citations'):
            print("Warning: No 'sentences_with_citations' found in results. This document may not contain any extracted citation sentences.")
        if sentences_with_cites:
            print(f"\nExample sentences with citations:")
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
        print(f"Error querying argument graph: {e}")
        logging.exception("Query command failed")
        sys.exit(1)

def handle_diagnose_command(args):
    """Handle the diagnose command."""
    try:
        kernel = CiteWeaveKernel()
        diagnosis = kernel.diagnose_document(args.pdf_path)
        
        print(f"=== Document Processing Diagnosis ===")
        print(f"File: {args.pdf_path}")
        print(f"Quality Level: {diagnosis['overall_assessment']['quality_level']}")
        print(f"Is Processable: {diagnosis['overall_assessment']['is_processable']}")
        
        # PDF diagnosis
        pdf_diag = diagnosis.get('pdf_diagnosis', {})
        if pdf_diag:
            print(f"\n--- PDF Processing ---")
            print(f"Best Quality Score: {pdf_diag.get('best_quality_score', 'Unknown')}")
            print(f"Recommended Engine: {pdf_diag.get('recommended_engine', 'Unknown')}")
            
        # Citation diagnosis  
        cite_diag = diagnosis.get('citation_diagnosis', {})
        if cite_diag:
            print(f"\n--- Citation Processing ---")
            print(f"References Count: {cite_diag.get('references_count', 0)}")
            print(f"References Extraction Success: {cite_diag.get('references_extraction_success', False)}")
            print(f"Has DOI: {cite_diag.get('has_doi', False)}")
        
        # Recommendations
        if diagnosis['overall_assessment']['recommendations']:
            print(f"\n--- Recommendations ---")
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
    pdf_files = glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True)
    logging.info(f"FINISH: Found {len(pdf_files)} PDF files in {directory}")
    
    if not pdf_files:
        print(f"No PDF files found in {directory}.")
        sys.exit(0)
    
    print(f"Found {len(pdf_files)} PDF files in {directory}.")
    
    # Get pending files based on resume mode
    if resume_mode or not force_restart:
        pending_files = tracker.get_pending_files(pdf_files, force_restart=force_restart)
        completed_count = len(pdf_files) - len(pending_files)
        
        if completed_count > 0:
            print(f"📊 Progress Summary:")
            summary = tracker.get_progress_summary()
            print(f"   Previously completed: {completed_count}")
            print(f"   Previously failed: {summary['failed']}")
            print(f"   Success rate: {summary['success_rate']:.1f}%")
            print(f"   Files to process: {len(pending_files)}")
            
            if len(pending_files) == 0:
                print("✅ All files have been processed successfully!")
                return
        else:
            print("🆕 No previous progress found. Starting fresh batch upload.")
    else:
        pending_files = pdf_files
        print("🔄 Force restart mode: Processing all files.")
    
    # Determine processing mode
    if use_sequential:
        print("Using sequential processing (multiprocessing disabled)")
        logging.info("START: Sequential processing mode")
        process_files_sequentially(pending_files, tracker)
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
    print(f"\n📊 Final Summary:")
    print(f"   Total files processed: {final_summary['total_tracked']}")
    print(f"   Successfully completed: {final_summary['completed']}")
    print(f"   Failed: {final_summary['failed']}")
    print(f"   Overall success rate: {final_summary['success_rate']:.1f}%")
    
    logging.info("FINISH: Batch upload command completed")

def process_files_sequentially(pdf_files, tracker):
    """Process files sequentially while preserving real tracker statistics."""
    print("Starting sequential batch upload...")
    success_count = 0
    fail_count = 0
    kernel = CiteWeaveKernel()

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_path}")
        started_at = time.time()
        try:
            print(f"Processing document: {pdf_path}")
            results = kernel.upload_document(pdf_path, save_results=True)
            stats = results.get('processing_stats', {})

            print(f"\nProcessing completed successfully!")
            print(f"Paper ID: {results['paper_id']}")
            print(f"Total sentences: {stats.get('total_sentences', 0)}")
            print(f"Sentences with citations: {stats.get('sentences_with_citations', 0)}")
            print(f"Total citations found: {stats.get('total_citations', 0)}")
            print(f"Total references: {stats.get('total_references', 0)}")

            sentences_with_cites = [
                s for s in results.get('sentences_with_citations', []) if s.get('citations')
            ]
            if not results.get('sentences_with_citations'):
                print("Warning: No 'sentences_with_citations' found in results. This document may not contain any extracted citation sentences.")
            if sentences_with_cites:
                print(f"\nExample sentences with citations:")
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
                },
            )
            success_count += 1

        except Exception as e:
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
    print(f"Batch upload complete!")
    print(f"Success: {success_count}, Failed: {fail_count}")
    
    if success_count > 0:
        print(f"Success rate: {(success_count/total_files)*100:.1f}%")
    
    if fail_count > 0:
        print(f"Failed files: {fail_count}/{total_files}")
        print("Consider running with --sequential flag for more detailed error messages.")

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
    print(f"Total PDF files discovered: {progress['total_pdf_files']}")
    print(f"Total files tracked: {summary['total_tracked']}")
    print(f"Completed: {progress['completed_count']}")
    print(f"Failed: {progress['failed_count']}")
    print(f"Pending / resumable: {progress['pending_count']}")
    print(f"  • Not started yet: {progress['not_started_count']}")
    print(f"  • Retryable failed files: {progress['retryable_failed_count']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")

    average_completed_duration_seconds = progress.get("average_completed_duration_seconds")
    estimated_remaining_seconds = progress.get("estimated_remaining_seconds")
    if average_completed_duration_seconds is not None:
        print(f"Observed average time per completed file: {average_completed_duration_seconds:.1f}s")
    if estimated_remaining_seconds is not None:
        print(f"Estimated remaining wall time: {estimated_remaining_seconds:.1f}s")

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

    if progress["pending_count"] > 0:
        print("\nTip: run batch-upload --resume to continue remaining files, including retries for previous failures.")

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
                print("  Tip: run `.venv/bin/python -m src.core.cli routes` for the full diagnostic report.")
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
    databases = [value for value in (entry.get("query_plan_databases") or []) if isinstance(value, str) and value]
    methods = [value for value in (entry.get("query_plan_methods") or []) if isinstance(value, str) and value]
    parts = []
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
        since_hours=since_hours,
        contains=getattr(args, "contains", "") or "",
        planned_database=getattr(args, "planned_database", "all") or "all",
        planned_method=getattr(args, "planned_method", "all") or "all",
    )

    if getattr(args, "json", False):
        print(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True))
        return

    print("\n=== CiteWeave Query History ===\n")
    print(f"Log file: {snapshot.get('log_file', '')}")
    print(f"Requested limit: {snapshot.get('requested_limit', 0)}")
    print(f"Status filter: {snapshot.get('status_filter', 'all')}")
    print(f"Source filter: {snapshot.get('source_filter', 'all')}")
    print(f"Confirmation filter: {snapshot.get('confirmation_filter', 'all')}")
    contains_filter = snapshot.get("contains_filter", "")
    if contains_filter:
        print(f"Contains filter: {contains_filter}")
    if snapshot.get("since_hours") is not None:
        print(f"Time window: last {snapshot.get('since_hours')} hours")
    planned_database_filter = snapshot.get("planned_database_filter", "all")
    planned_method_filter = snapshot.get("planned_method_filter", "all")
    if planned_database_filter != "all":
        print(f"Planned database filter: {planned_database_filter}")
    if planned_method_filter != "all":
        print(f"Planned method filter: {planned_method_filter}")
    print(f"Entries returned: {snapshot.get('entries_returned', 0)}")
    if snapshot.get("matching_entries_total") is not None:
        print(f"Matching entries before limit: {snapshot.get('matching_entries_total', 0)}")
    print(f"Successful queries: {snapshot.get('success_count', 0)}")
    print(f"Failed queries: {snapshot.get('error_count', 0)}")
    print(f"Corrupt rows skipped into diagnostics: {snapshot.get('corrupt_count', 0)}")

    average_duration_ms = snapshot.get("average_duration_ms")
    max_duration_ms = snapshot.get("max_duration_ms")
    if average_duration_ms is not None:
        print(f"Average duration: {average_duration_ms} ms")
    if max_duration_ms is not None:
        print(f"Slowest query: {max_duration_ms} ms")

    latest_status = snapshot.get("latest_status")
    latest_question = snapshot.get("latest_question")
    latest_source = snapshot.get("latest_source")
    if latest_status or latest_question:
        source_suffix = f" [{latest_source}]" if latest_source else ""
        print(f"Latest query: {latest_status or 'unknown'}{source_suffix} - {latest_question or ''}")

    latest_error = snapshot.get("latest_error")
    if latest_error:
        print(f"Latest error: {latest_error}")

    source_breakdown = snapshot.get("source_breakdown") or []
    if source_breakdown:
        print("Sources:")
        for item in source_breakdown:
            print(f"  - {item['source']}: {item['count']}")

    confirmation_breakdown = snapshot.get("confirmation_breakdown") or []
    if confirmation_breakdown:
        print("Confirmations:")
        for item in confirmation_breakdown:
            print(f"  - {item['confirmation']}: {item['count']}")

    query_plan_database_breakdown = snapshot.get("query_plan_database_breakdown") or []
    if query_plan_database_breakdown:
        print("Planned databases:")
        for item in query_plan_database_breakdown:
            print(f"  - {item['database']}: {item['count']}")

    query_plan_method_breakdown = snapshot.get("query_plan_method_breakdown") or []
    if query_plan_method_breakdown:
        print("Planned methods:")
        for item in query_plan_method_breakdown[:8]:
            print(f"  - {item['method']}: {item['count']}")

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
        plan_summary = _format_query_plan_summary(entry)
        if plan_summary:
            print(f"    plan: {plan_summary}")

    print()


if __name__ == "__main__":
    main() 
