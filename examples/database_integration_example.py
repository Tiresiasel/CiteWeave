#!/usr/bin/env python3
"""
Database Integration Example for CiteWeave

This example demonstrates how to use the DatabaseIntegrator to import
processed documents into Neo4j and vector databases.

The example showcases the separation of concerns between document processing
and database integration, allowing for flexible and modular operation.
"""

import sys
import os
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from document_processor import DocumentProcessor
from database_integrator import DatabaseIntegrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_and_store_document(pdf_path: str, config_path: str = "config"):
    """
    Complete workflow: Process a PDF document and store in databases.
    
    This demonstrates the clean separation between processing and storage:
    1. DocumentProcessor handles PDF processing and file storage
    2. DatabaseIntegrator handles database import from files
    
    Args:
        pdf_path: Path to PDF file to process
        config_path: Path to configuration directory
    """
    
    print("=" * 60)
    print("CiteWeave: Complete Document Processing and Storage")
    print("=" * 60)
    
    # Step 1: Process document with DocumentProcessor
    print("\n📄 Step 1: Processing document...")
    
    processor = DocumentProcessor(
        storage_root="data/papers",
        enable_argument_classification=True
    )
    
    try:
        # Process the document (saves to files automatically)
        results = processor.process_document(pdf_path, save_results=True)
        paper_id = results["paper_id"]
        
        print(f"✅ Document processed successfully!")
        print(f"   Paper ID: {paper_id}")
        print(f"   Citations found: {results['processing_stats']['total_citations']}")
        print(f"   Argument relations: {results['processing_stats']['total_argument_relations']}")
        
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        return False
    
    # Step 2: Import to databases with DatabaseIntegrator
    print("\n🗄️  Step 2: Importing to databases...")
    
    integrator = DatabaseIntegrator(
        config_path=config_path,
        storage_root="data/papers"
    )
    
    try:
        # Initialize database connections
        if not integrator.initialize_connections():
            print("❌ Failed to initialize database connections")
            return False
        
        # Import the processed document
        success = integrator.import_document(paper_id, force_reimport=True)
        
        if success:
            stats = integrator.get_import_status()
            print(f"✅ Database import successful!")
            print(f"   Sentences indexed: {stats['stats']['sentences_indexed']}")
            print(f"   Citations stored: {stats['stats']['citations_stored']}")
            print(f"   Argument relations: {stats['stats']['argument_relations_stored']}")
        else:
            print("❌ Database import failed")
            return False
    
    except Exception as e:
        print(f"❌ Database import failed: {e}")
        return False
    
    finally:
        integrator.close_connections()
    
    print("\n🎉 Complete workflow finished successfully!")
    return True

def batch_import_existing_documents(config_path: str = "config"):
    """
    Import all existing processed documents to databases.
    
    This demonstrates how you can retroactively import documents
    that were processed earlier, showcasing the flexibility of
    the modular architecture.
    """
    
    print("=" * 60)
    print("CiteWeave: Batch Import of Existing Documents")
    print("=" * 60)
    
    integrator = DatabaseIntegrator(
        config_path=config_path,
        storage_root="data/papers"
    )
    
    try:
        # Initialize database connections
        if not integrator.initialize_connections():
            print("❌ Failed to initialize database connections")
            return False
        
        print("\n📁 Scanning for processed documents...")
        
        # Import all existing processed documents
        stats = integrator.import_all_documents(force_reimport=False)
        
        print(f"\n✅ Batch import completed!")
        print(f"   Papers processed: {stats['papers_processed']}")
        print(f"   Sentences indexed: {stats['sentences_indexed']}")
        print(f"   Citations stored: {stats['citations_stored']}")
        print(f"   Argument relations: {stats['argument_relations_stored']}")
        
        if stats['errors']:
            print(f"   Errors encountered: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"     - {error}")
        
        return True
    
    except Exception as e:
        print(f"❌ Batch import failed: {e}")
        return False
    
    finally:
        integrator.close_connections()

def selective_import_by_criteria():
    """
    Demonstrate selective import based on criteria.
    
    This shows how the modular architecture allows for
    sophisticated import logic based on document metadata.
    """
    
    print("=" * 60)
    print("CiteWeave: Selective Import Example")
    print("=" * 60)
    
    integrator = DatabaseIntegrator(
        config_path="config",
        storage_root="data/papers"
    )
    
    if not integrator.initialize_connections():
        print("❌ Failed to initialize database connections")
        return False
    
    try:
        # Find all paper directories
        paper_dirs = [d for d in os.listdir("data/papers") 
                     if os.path.isdir(os.path.join("data/papers", d))]
        
        imported_count = 0
        
        for paper_id in paper_dirs:
            # Load metadata to check criteria
            metadata_file = os.path.join("data/papers", paper_id, "metadata.json")
            if not os.path.exists(metadata_file):
                continue
            
            try:
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Example criteria: Only import papers from 2020 onwards
                year = metadata.get('year', '')
                if year and int(year) >= 2020:
                    print(f"📑 Importing {metadata.get('title', paper_id)[:50]}...")
                    
                    success = integrator.import_document(paper_id, force_reimport=False)
                    if success:
                        imported_count += 1
                    
            except Exception as e:
                print(f"⚠️  Failed to process {paper_id}: {e}")
                continue
        
        print(f"\n✅ Selective import completed!")
        print(f"   Papers imported: {imported_count}")
        
        # Show final statistics
        status = integrator.get_import_status()
        print(f"   Total statistics: {status['stats']}")
        
        return True
    
    finally:
        integrator.close_connections()

def main():
    """
    Main example runner with different scenarios.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Integration Examples")
    parser.add_argument("--mode", choices=["single", "batch", "selective"], 
                       default="batch", help="Example mode to run")
    parser.add_argument("--pdf", help="PDF file path (for single mode)")
    parser.add_argument("--config", default="config", help="Configuration path")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.pdf:
            print("Please provide --pdf argument for single mode")
            return
        
        process_and_store_document(args.pdf, args.config)
    
    elif args.mode == "batch":
        batch_import_existing_documents(args.config)
    
    elif args.mode == "selective":
        selective_import_by_criteria()

if __name__ == "__main__":
    main() 