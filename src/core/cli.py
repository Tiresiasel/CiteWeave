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
# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # If python-dotenv is not installed, skip

import os
import sys

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
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)
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
    """Handle the chat command for interactive multi-turn conversation."""
    try:
        system = LangGraphResearchSystem()
        print("🤖 CiteWeave Multi-Agent Research System (Chat Mode)")
        print("=" * 60)
        print("Type 'exit' or 'quit' to end the chat.")
        print("=" * 60)
        while True:
            question = input("You: ").strip()
            if question.lower() in ("exit", "quit"):
                print("Exiting chat.")
                break
            if not question:
                continue
            # --- Spinner logic ---
            spinner_running = True
            def spinner():
                symbols = ['|', '/', '-', '\\']
                idx = 0
                print("AI: ", end="", flush=True)
                while spinner_running:
                    print(f"\b{symbols[idx % 4]}", end="", flush=True)
                    idx += 1
                    time.sleep(0.1)
                print("\b", end="", flush=True)  # Clean up spinner
            spinner_thread = threading.Thread(target=spinner)
            spinner_thread.start()
            # --- Run AI ---
            try:
                result = system.interactive_research_chat(question)
            finally:
                spinner_running = False
                spinner_thread.join()
            print(f"{result}\n")
    except Exception as e:
        print(f"Error during chat: {e}")
        logging.exception("Chat command failed")
        sys.exit(1)

def handle_batch_upload_command(args):
    """Handle the batch-upload command to process all PDFs in a directory."""
    directory = args.directory
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
    # Find all PDF files (recursively)
    pdf_files = glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True)
    if not pdf_files:
        print(f"No PDF files found in {directory}.")
        sys.exit(0)
    print(f"Found {len(pdf_files)} PDF files in {directory}. Starting batch upload...")
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
        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")
            fail_count += 1
    print(f"\nBatch upload complete. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main() 