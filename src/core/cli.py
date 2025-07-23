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
from src.agents.multi_agent_research_system import LangGraphResearchSystem

class BatchUploadTracker:
    """Tracks batch upload progress to enable resuming interrupted uploads."""
    
    def __init__(self, directory, tracker_file=None):
        self.directory = directory
        if tracker_file is None:
            # Create tracker file in data folder
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            self.tracker_file = data_dir / "batch_upload_tracker.json"
        else:
            self.tracker_file = Path(tracker_file)
        
        self.progress_data = self._load_progress()
    
    def _load_progress(self):
        """Load existing progress from tracker file."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Filter to only include entries for current directory
                    return {k: v for k, v in data.items() if v.get('directory') == self.directory}
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not load progress tracker: {e}")
                return {}
        return {}
    
    def _save_progress(self):
        """Save progress to tracker file."""
        try:
            # Load existing data to preserve other directories
            existing_data = {}
            if self.tracker_file.exists():
                try:
                    with open(self.tracker_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    existing_data = {}
            
            # Update with current directory data
            existing_data.update(self.progress_data)
            
            # Save back to file
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
        except IOError as e:
            logging.error(f"Could not save progress tracker: {e}")
    
    def mark_file_completed(self, pdf_path, result):
        """Mark a file as successfully processed."""
        self.progress_data[pdf_path] = {
            'status': 'completed',
            'directory': self.directory,
            'paper_id': result.get('paper_id', 'unknown'),
            'total_sentences': result.get('total_sentences', 0),
            'total_citations': result.get('total_citations', 0),
            'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'error': None
        }
        self._save_progress()
    
    def mark_file_failed(self, pdf_path, error):
        """Mark a file as failed."""
        self.progress_data[pdf_path] = {
            'status': 'failed',
            'directory': self.directory,
            'paper_id': None,
            'total_sentences': 0,
            'total_citations': 0,
            'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(error)
        }
        self._save_progress()
    
    def is_file_completed(self, pdf_path):
        """Check if a file has been successfully processed."""
        return pdf_path in self.progress_data and self.progress_data[pdf_path]['status'] == 'completed'
    
    def is_file_failed(self, pdf_path):
        """Check if a file has failed processing."""
        return pdf_path in self.progress_data and self.progress_data[pdf_path]['status'] == 'failed'
    
    def get_pending_files(self, all_files, force_restart=False):
        """Get list of files that need processing."""
        if force_restart:
            return all_files
        
        pending = []
        for pdf_path in all_files:
            if not self.is_file_completed(pdf_path):
                pending.append(pdf_path)
        
        return pending
    
    def get_progress_summary(self):
        """Get summary of current progress."""
        total = len(self.progress_data)
        completed = sum(1 for v in self.progress_data.values() if v['status'] == 'completed')
        failed = sum(1 for v in self.progress_data.values() if v['status'] == 'failed')
        
        return {
            'total_tracked': total,
            'completed': completed,
            'failed': failed,
            'success_rate': (completed / total * 100) if total > 0 else 0
        }
    
    def clear_progress(self, directory=None):
        """Clear progress for a specific directory or all progress."""
        if directory:
            # Remove entries for specific directory
            self.progress_data = {k: v for k, v in self.progress_data.items() 
                                if v.get('directory') != directory}
        else:
            # Clear all progress
            self.progress_data = {}
        
        self._save_progress()

def process_single_pdf_worker(pdf_path, diagnose=False, force=False):
    """
    Worker function for multiprocessing that processes a single PDF file.
    This function must be defined at module level for multiprocessing to work.
    """
    try:
        logging.info(f"START: Processing PDF file {pdf_path}")
        
        # Initialize the document processor in the worker process
        doc_processor = DocumentProcessor()
        
        # Process the document
        results = doc_processor.process_document(pdf_path, save_results=True)
        
        # Return success with basic stats
        stats = results.get('processing_stats', {})
        result_data = {
            'status': 'success',
            'pdf_path': pdf_path,
            'paper_id': results.get('paper_id', 'unknown'),
            'total_sentences': stats.get('total_sentences', 0),
            'sentences_with_citations': stats.get('sentences_with_citations', 0),
            'total_citations': stats.get('total_citations', 0),
            'total_references': stats.get('total_references', 0),
            'processing_time': time.time()  # Add timestamp for tracking
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
    else:
        parser.print_help()

def handle_upload_command(args):
    """Handle the upload command with integrated PDF and citation processing."""
    try:
        # Initialize the unified document processor
        doc_processor = DocumentProcessor()
        
        # Run diagnosis if requested
        if args.diagnose:
            print("Running quality diagnosis...")
            diagnosis = doc_processor.diagnose_document_processing(args.pdf_path)
            
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
        
        # Process the document
        print(f"Processing document: {args.pdf_path}")
        results = doc_processor.process_document(args.pdf_path, save_results=True)
        
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
    """Handle the query command."""
    print(f"Querying: {args.question}")
    # TODO: Implement query functionality
    print("Query functionality not yet implemented.")

def handle_diagnose_command(args):
    """Handle the diagnose command."""
    try:
        doc_processor = DocumentProcessor()
        diagnosis = doc_processor.diagnose_document_processing(args.pdf_path)
        
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
        system = LangGraphResearchSystem()
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
    """Process files sequentially (original behavior)."""
    print("Starting sequential batch upload...")
    success_count = 0
    fail_count = 0
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_path}")
        try:
            class Args:
                pass
            file_args = Args()
            file_args.pdf_path = pdf_path
            file_args.diagnose = False
            file_args.force = False
            handle_upload_command(file_args)
            success_count += 1
            
            # Mark as completed in tracker
            result = {
                'paper_id': 'unknown',  # We don't have detailed results from handle_upload_command
                'total_sentences': 0,
                'total_citations': 0
            }
            tracker.mark_file_completed(pdf_path, result)
            
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
    clear_progress = args.clear

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    tracker = BatchUploadTracker(directory)

    print(f"\n=== Batch Upload Progress for {directory} ===")
    summary = tracker.get_progress_summary()
    print(f"Total files tracked: {summary['total_tracked']}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")

    if clear_progress:
        print("\nClearing progress for this directory...")
        tracker.clear_progress(directory)
        print("Progress cleared.")
        summary = tracker.get_progress_summary()
        print(f"Total files tracked after clearing: {summary['total_tracked']}")
        print(f"Completed after clearing: {summary['completed']}")
        print(f"Failed after clearing: {summary['failed']}")
        print(f"Success rate after clearing: {summary['success_rate']:.1f}%")

    print("\n--- Pending Files ---")
    pending_files = tracker.get_pending_files(glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True), force_restart=False)
    if pending_files:
        for i, pdf_path in enumerate(pending_files, 1):
            print(f"{i}. {os.path.basename(pdf_path)}")
    else:
        print("No files pending processing.")

if __name__ == "__main__":
    main() 