#!/usr/bin/env python3
"""
Test script for DatabaseIntegrator

This script tests the DatabaseIntegrator functionality by importing
existing processed documents and verifying the database operations.
"""

import sys
import os
import logging
import json
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.database_integrator import DatabaseIntegrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_processed_documents():
    """
    Check what processed documents are available for testing.
    """
    print("🔍 Checking available processed documents...")
    
    storage_root = "data/papers"
    if not os.path.exists(storage_root):
        print(f"❌ Storage root not found: {storage_root}")
        return []
    
    paper_dirs = []
    for item in os.listdir(storage_root):
        paper_dir = os.path.join(storage_root, item)
        if os.path.isdir(paper_dir):
            processed_file = os.path.join(paper_dir, "processed_document.json")
            if os.path.exists(processed_file):
                paper_dirs.append(item)
                print(f"  ✅ Found: {item}")
    
    print(f"\n📊 Total processed documents: {len(paper_dirs)}")
    return paper_dirs

def analyze_document_content(paper_id: str) -> Dict:
    """
    Analyze the content of a processed document to understand what we'll be importing.
    """
    print(f"\n🔬 Analyzing document: {paper_id}")
    
    processed_file = os.path.join("data/papers", paper_id, "processed_document.json")
    
    try:
        with open(processed_file, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        
        # Extract statistics
        metadata = doc.get("metadata", {})
        sentences = doc.get("sentences_with_citations", [])
        
        stats = {
            "title": metadata.get("title", "Unknown"),
            "year": metadata.get("year", "Unknown"),
            "authors": metadata.get("authors", []),
            "total_sentences": len(sentences),
            "sentences_with_citations": sum(1 for s in sentences if s.get("has_citations", False)),
            "total_citations": sum(len(s.get("citations", [])) for s in sentences),
            "sentences_with_arguments": sum(1 for s in sentences if s.get("argument_analysis", {}).get("has_argument_relations", False)),
            "total_argument_relations": 0,
            "relation_types": set()
        }
        
        # Count argument relations and types
        for sentence in sentences:
            for citation in sentence.get("citations", []):
                arg_analysis = citation.get("argument_analysis", {})
                if arg_analysis.get("has_argument_relations", False):
                    entities = arg_analysis.get("entities", [])
                    stats["total_argument_relations"] += len(entities)
                    for entity in entities:
                        stats["relation_types"].add(entity.get("relation_type", "UNKNOWN"))
        
        stats["relation_types"] = list(stats["relation_types"])
        
        print(f"  📄 Title: {stats['title']}")
        print(f"  📅 Year: {stats['year']}")
        print(f"  👥 Authors: {len(stats['authors'])} authors")
        print(f"  📝 Sentences: {stats['total_sentences']} total, {stats['sentences_with_citations']} with citations")
        print(f"  📚 Citations: {stats['total_citations']} total")
        print(f"  🧠 Arguments: {stats['sentences_with_arguments']} sentences with arguments, {stats['total_argument_relations']} relations")
        print(f"  🔗 Relation types: {', '.join(stats['relation_types'])}")
        
        return stats
        
    except Exception as e:
        print(f"  ❌ Failed to analyze document: {e}")
        return {}

def test_database_connections():
    """
    Test database connection initialization.
    """
    print("\n🔗 Testing database connections...")
    
    integrator = DatabaseIntegrator(
        config_path="config",
        storage_root="data/papers"
    )
    
    try:
        success = integrator.initialize_connections()
        if success:
            print("  ✅ Database connections initialized successfully")
            
            # Test Neo4j connection
            if integrator.graph_db:
                print("  ✅ Neo4j connection ready")
            else:
                print("  ⚠️  Neo4j connection not available")
            
            # Test vector database connection
            if integrator.vector_indexer:
                print("  ✅ Vector database connection ready")
            else:
                print("  ⚠️  Vector database connection not available")
            
            return integrator
        else:
            print("  ❌ Failed to initialize database connections")
            return None
            
    except Exception as e:
        print(f"  ❌ Database connection test failed: {e}")
        return None

def test_single_document_import(integrator: DatabaseIntegrator, paper_id: str):
    """
    Test importing a single document.
    """
    print(f"\n📥 Testing single document import: {paper_id}")
    
    try:
        # Get initial stats
        initial_stats = integrator.get_import_status()
        initial_counts = initial_stats['stats'].copy()
        print(f"  📊 Initial stats: {initial_counts}")
        
        # Import the document
        success = integrator.import_document(paper_id, force_reimport=True)
        
        if success:
            # Get final stats
            final_stats = integrator.get_import_status()
            final_counts = final_stats['stats'].copy()
            print(f"  ✅ Import successful!")
            print(f"  📊 Final stats: {final_counts}")
            
            # Calculate differences (this document's contribution)
            diff = {
                "sentences_indexed": final_counts['sentences_indexed'] - initial_counts['sentences_indexed'],
                "citations_stored": final_counts['citations_stored'] - initial_counts['citations_stored'],
                "argument_relations_stored": final_counts['argument_relations_stored'] - initial_counts['argument_relations_stored']
            }
            print(f"  📈 Import diff: {diff}")
            
            return True
        else:
            print(f"  ❌ Import failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        return False

def test_batch_import(integrator: DatabaseIntegrator):
    """
    Test batch import functionality.
    """
    print(f"\n📦 Testing batch import...")
    
    try:
        # Reset stats for clean test
        integrator.reset_stats()
        
        # Run batch import
        stats = integrator.import_all_documents(force_reimport=True)
        
        print(f"  ✅ Batch import completed!")
        print(f"  📊 Final stats: {stats}")
        
        if stats['errors']:
            print(f"  ⚠️  Errors encountered:")
            for error in stats['errors']:
                print(f"    - {error}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Batch import test failed: {e}")
        return False

def test_database_queries(integrator: DatabaseIntegrator, paper_id: str):
    """
    Test basic database queries to verify data was imported correctly.
    """
    print(f"\n🔍 Testing database queries...")
    
    if not integrator.graph_db:
        print("  ⚠️  No Neo4j connection available for query testing")
        return
    
    try:
        # Test basic paper query
        query = "MATCH (p:Paper) RETURN count(p) as paper_count"
        with integrator.graph_db.driver.session() as session:
            result = session.run(query)
            record = result.single()
            paper_count = record["paper_count"] if record else 0
            print(f"  📄 Papers in database: {paper_count}")
        
        # Test argument query
        query = "MATCH (a:Argument) RETURN count(a) as arg_count"
        with integrator.graph_db.driver.session() as session:
            result = session.run(query)
            record = result.single()
            arg_count = record["arg_count"] if record else 0
            print(f"  🧠 Arguments in database: {arg_count}")
        
        # Test relationship query
        query = "MATCH ()-[r:RELATES]->() RETURN count(r) as rel_count"
        with integrator.graph_db.driver.session() as session:
            result = session.run(query)
            record = result.single()
            rel_count = record["rel_count"] if record else 0
            print(f"  🔗 Relationships in database: {rel_count}")
        
        # Test specific paper query
        query = "MATCH (p:Paper {id: $paper_id}) RETURN p.title as title"
        with integrator.graph_db.driver.session() as session:
            result = session.run(query, paper_id=paper_id)
            record = result.single()
            title = record["title"] if record else "Not found"
            print(f"  📚 Test paper title: {title}")
        
        print(f"  ✅ Database queries completed successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Database query test failed: {e}")
        return False

def main():
    """
    Main test runner.
    """
    print("=" * 60)
    print("🧪 DatabaseIntegrator Test Suite")
    print("=" * 60)
    
    # Step 1: Check available documents
    paper_ids = check_processed_documents()
    if not paper_ids:
        print("❌ No processed documents found for testing")
        return
    
    # Step 2: Analyze a sample document
    test_paper_id = paper_ids[0]
    doc_stats = analyze_document_content(test_paper_id)
    if not doc_stats:
        print("❌ Failed to analyze sample document")
        return
    
    # Step 3: Test database connections
    integrator = test_database_connections()
    if not integrator:
        print("❌ Database connection failed - skipping import tests")
        print("\n💡 Tip: Make sure Neo4j is running and config/neo4j_config.json is correct")
        return
    
    try:
        # Step 4: Test single document import
        single_success = test_single_document_import(integrator, test_paper_id)
        
        # Step 5: Test database queries
        if single_success:
            test_database_queries(integrator, test_paper_id)
        
        # Step 6: Test batch import (if we have multiple documents)
        if len(paper_ids) > 1:
            test_batch_import(integrator)
        
        print("\n🎉 Test suite completed!")
        
    finally:
        # Clean up
        integrator.close_connections()

if __name__ == "__main__":
    main() 