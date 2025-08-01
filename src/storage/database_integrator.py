"""
Database Integration Module for CiteWeave

This module handles the integration of processed document data into databases.
It reads the standardized JSON/JSONL files generated by DocumentProcessor
and stores the data in Neo4j graph database and vector database.

Design principles:
- Separation of concerns: Pure database integration logic
- Flexibility: Support batch processing and selective imports
- Robustness: Comprehensive error handling and transaction management
- Performance: Optimized bulk operations
"""

import os
import json
import logging
from typing import List, Dict, Optional, Set
from datetime import datetime
import hashlib

from src.storage.graph_builder import GraphDB
from src.storage.vector_indexer import VectorIndexer as MultiLevelVectorIndexer
from src.utils.config_manager import ConfigManager
from src.utils.paper_id_utils import PaperIDGenerator

class DatabaseIntegrator:
    """
    Handles integration of processed document data into databases.
    
    This class reads the standardized output files from DocumentProcessor
    and imports the data into Neo4j graph database and vector database.
    """
    
    def __init__(self, config_path: str = "config", 
                 storage_root: str = "data/papers"):
        """
        Initialize database connections and configuration.
        
        Args:
            config_path: Path to configuration files
            storage_root: Root directory where processed documents are stored
        """
        self.storage_root = storage_root
        self.config_manager = ConfigManager(config_path)
        self.paper_id_generator = PaperIDGenerator()
        
        # Initialize database connections
        self.graph_db = None
        self.vector_indexer = None
        
        # Track processed documents to avoid duplicates
        self._processed_papers: Set[str] = set()
        
        # Statistics
        self.stats = {
            "papers_processed": 0,
            "sentences_indexed": 0,
            "paragraphs_indexed": 0,
            "sections_indexed": 0,
            "citations_stored": 0,
            "argument_relations_stored": 0,
            "paper_citations_created": 0,
            "stub_papers_created": 0,
            "errors": []
        }
        
        logging.info("DatabaseIntegrator initialized")
    
    def initialize_connections(self) -> bool:
        """
        Initialize connections to databases.
        
        Returns:
            bool: True if all connections successful, False otherwise
        """
        try:
            # Initialize Neo4j connection
            neo4j_config = self.config_manager.neo4j_config
            self.graph_db = GraphDB(
                uri=neo4j_config["uri"],
                user=neo4j_config["username"],
                password=neo4j_config["password"]
            )
            logging.info("Neo4j connection initialized")
            
            # Initialize vector database connection
            vector_config = self.config_manager.get_qdrant_config()
            self.vector_indexer = MultiLevelVectorIndexer(
                paper_root=self.storage_root,
                index_path="./data/vector_index"
            )
            logging.info("Vector database connection initialized")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize database connections: {e}")
            return False
    
    def close_connections(self):
        """Close all database connections."""
        if self.graph_db:
            self.graph_db.close()
            logging.info("Neo4j connection closed")
        
        # Vector indexer doesn't need explicit closing
        logging.info("Database connections closed")
    
    def import_document(self, paper_id: str, force_reimport: bool = False) -> bool:
        """
        Import a single processed document into databases.
        
        Args:
            paper_id: Unique paper identifier
            force_reimport: Whether to reimport if already processed
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not force_reimport and paper_id in self._processed_papers:
            logging.info(f"Paper {paper_id} already processed, skipping")
            return True
        
        paper_dir = os.path.join(self.storage_root, paper_id)
        if not os.path.exists(paper_dir):
            logging.error(f"Paper directory not found: {paper_dir}")
            return False
        
        try:
            # Load processed document data
            processed_doc = self._load_processed_document(paper_dir)
            if not processed_doc:
                return False
            
            # Import to Neo4j graph database
            if self.graph_db:
                self._import_to_graph_db(paper_id, processed_doc)
            
            # Import to vector database
            if self.vector_indexer:
                self._import_to_vector_db(paper_id, processed_doc)
            
            # Mark as processed
            self._processed_papers.add(paper_id)
            self.stats["papers_processed"] += 1
            
            logging.info(f"Successfully imported paper {paper_id} to databases")
            return True
            
        except Exception as e:
            error_msg = f"Failed to import paper {paper_id}: {e}"
            logging.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False
    
    def import_all_documents(self, force_reimport: bool = False) -> Dict:
        """
        Import all processed documents from storage directory.
        
        Args:
            force_reimport: Whether to reimport already processed papers
            
        Returns:
            Dict: Import statistics
        """
        if not os.path.exists(self.storage_root):
            logging.error(f"Storage root does not exist: {self.storage_root}")
            return self.stats
        
        paper_dirs = [d for d in os.listdir(self.storage_root) 
                     if os.path.isdir(os.path.join(self.storage_root, d))]
        
        logging.info(f"Found {len(paper_dirs)} paper directories to process")
        
        for paper_id in paper_dirs:
            try:
                self.import_document(paper_id, force_reimport=force_reimport)
            except Exception as e:
                error_msg = f"Failed to process paper directory {paper_id}: {e}"
                logging.error(error_msg)
                self.stats["errors"].append(error_msg)
        
        logging.info(f"Completed import of {self.stats['papers_processed']} papers")
        return self.stats
    
    def _load_processed_document(self, paper_dir: str) -> Optional[Dict]:
        """
        Load processed document data from directory.
        
        Args:
            paper_dir: Path to paper directory
            
        Returns:
            Dict: Processed document data or None if failed
        """
        processed_file = os.path.join(paper_dir, "processed_document.json")
        
        if not os.path.exists(processed_file):
            logging.error(f"Processed document file not found: {processed_file}")
            return None
        
        try:
            with open(processed_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load processed document: {e}")
            return None
    
    def _import_to_graph_db(self, paper_id: str, processed_doc: Dict):
        """
        Import document data to Neo4j graph database.
        
        Args:
            paper_id: Unique paper identifier
            processed_doc: Processed document data
        """
        metadata = processed_doc["metadata"]
        sentences = processed_doc["sentences_with_citations"]
        
        # Create or update main paper node (not a stub since we have the actual paper)
        self.graph_db.ensure_paper_exists(
            paper_id=paper_id,
            title=metadata.get("title", ""),
            authors=metadata.get("authors", []),
            year=int(metadata.get("year", 0)) if metadata.get("year") else 0,
            stub=False,  # This is an actual uploaded paper
            doi=metadata.get("doi"),
            journal=metadata.get("journal"),
            abstract=metadata.get("abstract"),
            publisher=metadata.get("publisher"),
            volume=metadata.get("volume"),
            issue=metadata.get("issue"),
            pages=metadata.get("pages"),
            issn=metadata.get("issn"),
            url=metadata.get("url"),
            type=metadata.get("type")
        )
        
        # Track citation relationships to create paper-to-paper citation edges
        cited_papers = {}  # paper_id -> citation_count
        
        # Process sentences with citations and argument relations
        for sentence_data in sentences:
            if not sentence_data.get("has_citations", False):
                continue
            
            sentence_index = sentence_data["sentence_index"]
            sentence_text = sentence_data["sentence_text"]
            
            # Create argument node for the sentence (if it has citations, it's likely an argument)
            arg_id = f"{paper_id}_sent_{sentence_index}"
            
            # Determine claim type based on argument analysis
            claim_type = "NEUTRAL"
            if sentence_data.get("argument_analysis", {}).get("has_argument_relations", False):
                claim_type = "CLAIM_MAIN"  # Could be enhanced with more specific types
            
            self.graph_db.create_argument(
                arg_id=arg_id,
                paper_id=paper_id,
                text=sentence_text,
                claim_type=claim_type,
                section=None,  # Could be extracted if available
                version="v2.0"
            )
            
            # Process each citation in the sentence
            for citation in sentence_data.get("citations", []):
                cited_paper_id = citation.get("paper_id")
                if not cited_paper_id:
                    continue
                
                # Track citation count for paper-to-paper relationships
                if cited_paper_id not in cited_papers:
                    cited_papers[cited_paper_id] = 0
                cited_papers[cited_paper_id] += 1
                
                # Get citation reference information
                reference = citation.get("reference", {})
                
                # Ensure cited paper exists (create stub if necessary)
                self.graph_db.ensure_paper_exists(
                    paper_id=cited_paper_id,
                    title=reference.get("title", "Unknown Title"),
                    authors=reference.get("authors", ["Unknown Author"]),
                    year=int(reference.get("year", 0)) if reference.get("year") else 0,
                    stub=True  # This is a referenced paper, potentially a stub
                )
                
                # Create argument-to-paper relation with detailed argument analysis
                argument_analysis = citation.get("argument_analysis", {})
                if argument_analysis.get("has_argument_relations", False):
                    entities = argument_analysis.get("entities", [])
                    for entity in entities:
                        relation_type = entity.get("relation_type", "CITES")
                        confidence = entity.get("confidence")
                        
                        self.graph_db.create_relation(
                            from_arg=arg_id,
                            to_arg_or_paper=cited_paper_id,
                            relation_type=relation_type,
                            confidence=confidence,
                            version="v2.0"
                        )
                        
                        self.stats["argument_relations_stored"] += 1
                else:
                    # Create basic citation relation
                    self.graph_db.create_relation(
                        from_arg=arg_id,
                        to_arg_or_paper=cited_paper_id,
                        relation_type="CITES",
                        confidence=None,
                        version="v2.0"
                    )
                    
                    self.stats["argument_relations_stored"] += 1
                
                self.stats["citations_stored"] += 1
        
        # Create paper-to-paper citation relationships
        for cited_paper_id, citation_count in cited_papers.items():
            self.graph_db.create_paper_citation(
                citing_paper_id=paper_id,
                cited_paper_id=cited_paper_id,
                citation_count=citation_count
            )
            self.stats["paper_citations_created"] += 1
        
        logging.info(f"Created {len(cited_papers)} paper citation relationships for {paper_id}")
    
    def _import_to_vector_db(self, paper_id: str, processed_doc: Dict):
        """
        Import document data to vector database with multi-level indexing.
        
        Args:
            paper_id: Unique paper identifier
            processed_doc: Processed document data
        """
        metadata = processed_doc["metadata"]
        sentences = processed_doc["sentences_with_citations"]
        
        # Extract sentences with citations for sentence-level indexing
        sentence_texts = []
        claim_types = []
        
        for sentence_data in sentences:
            if sentence_data.get("has_citations", False):
                sentence_texts.append(sentence_data["sentence_text"])
                
                # Determine claim type based on argument analysis
                if sentence_data.get("argument_analysis", {}).get("has_argument_relations", False):
                    claim_types.append("CLAIM_MAIN")  # Could be enhanced
                else:
                    claim_types.append("NEUTRAL")
        
        if sentence_texts:
            # Index sentences in vector database
            self.vector_indexer.index_sentences(
                paper_id=paper_id,
                sentences=sentence_texts,
                metadata=metadata,
                claim_types=claim_types
            )
            
            self.stats["sentences_indexed"] += len(sentence_texts)
        
        # Multi-level indexing: Create enhanced document structure and index all levels
        try:
            # If the document already has structured paragraphs/sections, use them
            if "paragraphs" in processed_doc or "sections" in processed_doc:
                enhanced_doc = processed_doc
            else:
                # Generate paragraphs and sections from sentences
                enhanced_doc = self.vector_indexer._enhance_document_structure(processed_doc)
            
            # Index paragraphs and sections
            if "paragraphs" in enhanced_doc:
                self.vector_indexer.index_paragraphs(paper_id, enhanced_doc["paragraphs"], metadata)
                self.stats["paragraphs_indexed"] += len(enhanced_doc["paragraphs"])
                logging.info(f"Indexed {len(enhanced_doc['paragraphs'])} paragraphs for paper {paper_id}")
            
            if "sections" in enhanced_doc:
                self.vector_indexer.index_sections(paper_id, enhanced_doc["sections"], metadata)
                self.stats["sections_indexed"] += len(enhanced_doc["sections"])
                logging.info(f"Indexed {len(enhanced_doc['sections'])} sections for paper {paper_id}")
                
        except Exception as e:
            logging.warning(f"Failed to create multi-level indexes for paper {paper_id}: {e}")
            # Don't fail the entire import if multi-level indexing fails
    
    def get_import_status(self) -> Dict:
        """
        Get current import statistics.
        
        Returns:
            Dict: Current statistics
        """
        return {
            "processed_papers": list(self._processed_papers),
            "stats": self.stats,
            "timestamp": datetime.now().isoformat()
        }
    
    def reset_stats(self):
        """Reset import statistics and processed papers cache."""
        self.stats = {
            "papers_processed": 0,
            "sentences_indexed": 0,
            "paragraphs_indexed": 0,
            "sections_indexed": 0,
            "citations_stored": 0,
            "argument_relations_stored": 0,
            "paper_citations_created": 0,
            "stub_papers_created": 0,
            "errors": []
        }
        self._processed_papers.clear()
        logging.info("Import statistics reset")
    
    def get_citation_network_overview(self) -> Dict:
        """
        Get overview of the citation network.
        
        Returns:
            Dict: Citation network statistics
        """
        if not self.graph_db:
            return {"error": "Neo4j connection not initialized"}
        
        try:
            network_stats = self.graph_db.get_citation_network_stats()
            stub_papers = self.graph_db.list_stub_papers()
            
            return {
                "network_stats": network_stats,
                "stub_papers": stub_papers[:10],  # Top 10 most cited stub papers
                "total_stub_papers": len(stub_papers)
            }
        except Exception as e:
            logging.error(f"Failed to get citation network overview: {e}")
            return {"error": str(e)}
    
    def resolve_stub_paper(self, paper_id: str, title: str, authors: List[str], year: int) -> bool:
        """
        Resolve a stub paper with actual paper information.
        
        This method is called when a paper that was previously referenced
        is actually uploaded to the system.
        
        Args:
            paper_id: Paper ID to resolve
            title: Actual paper title
            authors: Actual author list
            year: Actual publication year
            
        Returns:
            bool: True if successful
        """
        if not self.graph_db:
            logging.error("Neo4j connection not initialized")
            return False
        
        try:
            self.graph_db.update_paper_from_stub(paper_id, title, authors, year)
            logging.info(f"Resolved stub paper {paper_id} with actual data")
            return True
        except Exception as e:
            logging.error(f"Failed to resolve stub paper {paper_id}: {e}")
            return False


def main():
    """
    Command-line interface for database integration.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Import processed documents to databases")
    parser.add_argument("--paper-id", help="Import specific paper by ID")
    parser.add_argument("--all", action="store_true", help="Import all processed papers")
    parser.add_argument("--force", action="store_true", help="Force reimport of existing papers")
    parser.add_argument("--config", default="config", help="Configuration directory path")
    parser.add_argument("--storage", default="data/papers", help="Storage root directory")
    parser.add_argument("--network-overview", action="store_true", help="Show citation network overview")
    
    args = parser.parse_args()
    
    # Initialize integrator
    integrator = DatabaseIntegrator(config_path=args.config, storage_root=args.storage)
    
    if not integrator.initialize_connections():
        logging.error("Failed to initialize database connections")
        return
    
    try:
        if args.network_overview:
            # Show citation network overview
            overview = integrator.get_citation_network_overview()
            print("\n📊 Citation Network Overview")
            print("=" * 50)
            
            if "error" in overview:
                print(f"❌ Error: {overview['error']}")
                return
            
            stats = overview["network_stats"]
            print(f"📄 Total Papers: {stats['total_papers']}")
            print(f"✅ Uploaded Papers: {stats['uploaded_papers']}")
            print(f"🔗 Stub Papers: {stats['stub_papers']}")
            print(f"➡️ Citation Relations: {stats['total_citation_relations']}")
            print(f"📊 Citation Instances: {stats['total_citation_instances']}")
            
            if overview["stub_papers"]:
                print(f"\n🔗 Top Cited Stub Papers (showing {min(5, len(overview['stub_papers']))}):")
                for stub in overview["stub_papers"][:5]:
                    print(f"   • {stub['title']} ({stub['year']}) - Cited {stub['cited_by_count']} times")
            
        elif args.all:
            # Import all documents
            stats = integrator.import_all_documents(force_reimport=args.force)
            print(f"Import completed: {stats}")
            
        elif args.paper_id:
            # Import specific document
            success = integrator.import_document(args.paper_id, force_reimport=args.force)
            if success:
                print(f"Successfully imported paper {args.paper_id}")
            else:
                print(f"Failed to import paper {args.paper_id}")
        else:
            print("Please specify --all, --paper-id, or --network-overview")
    
    finally:
        integrator.close_connections()


if __name__ == "__main__":
    main() 