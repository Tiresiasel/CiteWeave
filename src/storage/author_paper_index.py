"""
author_paper_index.py
Author-Paper Index for CiteWeave

This module provides functionality to:
1. Index papers by author for quick lookup
2. Find all papers by a specific author
3. Get original PDF paths for document-level AI queries
4. Support batch queries for multiple authors
"""

import os
import json
import logging
from typing import List, Dict, Optional, Set
from collections import defaultdict
import sqlite3
from pathlib import Path

class AuthorPaperIndex:
    """
    Manages an index of papers by author for efficient lookups.
    Supports finding all papers by an author and getting PDF paths.
    """
    
    def __init__(self, storage_root: str = "data/papers", index_db_path: str = "data/author_paper_index.db"):
        """
        Initialize the author-paper index.
        
        Args:
            storage_root: Root directory where processed papers are stored
            index_db_path: Path to SQLite database for author index
        """
        self.storage_root = storage_root
        self.index_db_path = index_db_path
        
        # Create index database directory if it doesn't exist
        os.makedirs(os.path.dirname(index_db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logging.info(f"AuthorPaperIndex initialized with storage_root: {storage_root}")
    
    def _init_database(self):
        """Initialize SQLite database for author-paper index."""
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.cursor()
            
            # Create authors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS authors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    UNIQUE(normalized_name)
                )
            """)
            
            # Create papers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    year INTEGER,
                    journal TEXT,
                    pdf_path TEXT,
                    processed_date TEXT
                )
            """)
            
            # Create author_papers relationship table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS author_papers (
                    author_id INTEGER,
                    paper_id TEXT,
                    author_position INTEGER,
                    FOREIGN KEY (author_id) REFERENCES authors (id),
                    FOREIGN KEY (paper_id) REFERENCES papers (paper_id),
                    PRIMARY KEY (author_id, paper_id)
                )
            """)
            
            # Create indexes for efficient queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_authors_normalized ON authors(normalized_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_author_papers_author ON author_papers(author_id)")
            
            conn.commit()
    
    def _normalize_author_name(self, name: str) -> str:
        """
        Normalize author name for consistent matching.
        
        Args:
            name: Author name
            
        Returns:
            Normalized name (lowercase, spaces trimmed)
        """
        return name.strip().lower()
    
    def rebuild_index(self):
        """
        Rebuild the entire author-paper index from stored documents.
        This scans all papers in storage_root and rebuilds the index.
        """
        logging.info("Rebuilding author-paper index...")
        
        # Clear existing index
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM author_papers")
            cursor.execute("DELETE FROM papers")
            cursor.execute("DELETE FROM authors")
            conn.commit()
        
        if not os.path.exists(self.storage_root):
            logging.warning(f"Storage root does not exist: {self.storage_root}")
            return
        
        processed_count = 0
        error_count = 0
        
        for paper_id in os.listdir(self.storage_root):
            paper_dir = os.path.join(self.storage_root, paper_id)
            if not os.path.isdir(paper_dir):
                continue
            
            try:
                self._index_paper(paper_id, paper_dir)
                processed_count += 1
            except Exception as e:
                logging.error(f"Failed to index paper {paper_id}: {e}")
                error_count += 1
        
        logging.info(f"Index rebuild completed: {processed_count} papers indexed, {error_count} errors")
    
    def _index_paper(self, paper_id: str, paper_dir: str):
        """
        Index a single paper and its authors.
        
        Args:
            paper_id: Unique paper identifier
            paper_dir: Path to paper directory
        """
        metadata_file = os.path.join(paper_dir, "metadata.json")
        if not os.path.exists(metadata_file):
            return
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Find original PDF path (try common patterns)
        pdf_path = self._find_original_pdf(paper_id, metadata)
        
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.cursor()
            
            # Insert paper
            cursor.execute("""
                INSERT OR REPLACE INTO papers 
                (paper_id, title, year, journal, pdf_path, processed_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                paper_id,
                metadata.get("title", ""),
                metadata.get("year"),
                metadata.get("journal"),
                pdf_path,
                metadata.get("processed_date")
            ))
            
            # Process authors
            authors = metadata.get("authors", [])
            for position, author_name in enumerate(authors):
                if not author_name:
                    continue
                
                normalized_name = self._normalize_author_name(author_name)
                
                # Insert or get author
                cursor.execute("""
                    INSERT OR IGNORE INTO authors (name, normalized_name)
                    VALUES (?, ?)
                """, (author_name, normalized_name))
                
                cursor.execute("""
                    SELECT id FROM authors WHERE normalized_name = ?
                """, (normalized_name,))
                author_id = cursor.fetchone()[0]
                
                # Link author to paper
                cursor.execute("""
                    INSERT OR REPLACE INTO author_papers 
                    (author_id, paper_id, author_position)
                    VALUES (?, ?, ?)
                """, (author_id, paper_id, position))
            
            conn.commit()
    
    def _find_original_pdf(self, paper_id: str, metadata: Dict) -> Optional[str]:
        """
        Try to find the original PDF file for a paper.
        
        Args:
            paper_id: Paper identifier
            metadata: Paper metadata
            
        Returns:
            Path to original PDF file if found, None otherwise
        """
        # Try common PDF storage locations
        possible_paths = [
            f"test_files/{metadata.get('title', '')}.pdf",
            f"test_files/{paper_id}.pdf", 
            f"data/pdfs/{paper_id}.pdf",
            f"pdfs/{paper_id}.pdf"
        ]
        
        for pdf_path in possible_paths:
            if os.path.exists(pdf_path):
                return os.path.abspath(pdf_path)
        
        # Try finding PDFs with similar names in test_files
        test_files_dir = "test_files"
        if os.path.exists(test_files_dir):
            title = metadata.get("title", "").lower()
            for filename in os.listdir(test_files_dir):
                if filename.endswith(".pdf"):
                    # Simple fuzzy matching
                    if any(word in filename.lower() for word in title.split() if len(word) > 3):
                        return os.path.abspath(os.path.join(test_files_dir, filename))
        
        return None
    
    def find_papers_by_author(self, author_name: str, exact_match: bool = False) -> List[Dict]:
        """
        Find all papers by a specific author.
        
        Args:
            author_name: Author name to search for
            exact_match: If True, use exact matching; if False, use fuzzy matching
            
        Returns:
            List of paper dictionaries with metadata
        """
        normalized_search = self._normalize_author_name(author_name)
        
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.cursor()
            
            if exact_match:
                query = """
                    SELECT p.paper_id, p.title, p.year, p.journal, p.pdf_path, a.name
                    FROM papers p
                    JOIN author_papers ap ON p.paper_id = ap.paper_id
                    JOIN authors a ON ap.author_id = a.id
                    WHERE a.normalized_name = ?
                    ORDER BY p.year DESC, ap.author_position
                """
                cursor.execute(query, (normalized_search,))
            else:
                # Fuzzy matching using LIKE
                query = """
                    SELECT p.paper_id, p.title, p.year, p.journal, p.pdf_path, a.name
                    FROM papers p
                    JOIN author_papers ap ON p.paper_id = ap.paper_id
                    JOIN authors a ON ap.author_id = a.id
                    WHERE a.normalized_name LIKE ?
                    ORDER BY p.year DESC, ap.author_position
                """
                cursor.execute(query, (f"%{normalized_search}%",))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "paper_id": row[0],
                    "title": row[1],
                    "year": row[2],
                    "journal": row[3],
                    "pdf_path": row[4],
                    "author_name": row[5]
                })
            
            return results
    
    def get_paper_pdf_path(self, paper_id: str) -> Optional[str]:
        """
        Get the original PDF path for a specific paper.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            Path to PDF file if available, None otherwise
        """
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pdf_path FROM papers WHERE paper_id = ?", (paper_id,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_papers_pdf_paths(self, paper_ids: List[str]) -> Dict[str, Optional[str]]:
        """
        Get PDF paths for multiple papers.
        
        Args:
            paper_ids: List of paper identifiers
            
        Returns:
            Dictionary mapping paper_id to PDF path (or None if not available)
        """
        if not paper_ids:
            return {}
        
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(paper_ids))
            cursor.execute(f"SELECT paper_id, pdf_path FROM papers WHERE paper_id IN ({placeholders})", paper_ids)
            
            result = {}
            for row in cursor.fetchall():
                result[row[0]] = row[1]
            
            # Add None for missing papers
            for paper_id in paper_ids:
                if paper_id not in result:
                    result[paper_id] = None
            
            return result
    
    def get_all_authors(self) -> List[Dict]:
        """
        Get list of all authors in the index.
        
        Returns:
            List of author dictionaries with paper counts
        """
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.name, a.normalized_name, COUNT(ap.paper_id) as paper_count
                FROM authors a
                LEFT JOIN author_papers ap ON a.id = ap.author_id
                GROUP BY a.id, a.name, a.normalized_name
                ORDER BY paper_count DESC, a.name
            """)
            
            return [
                {
                    "name": row[0],
                    "normalized_name": row[1], 
                    "paper_count": row[2]
                }
                for row in cursor.fetchall()
            ]
    
    def get_statistics(self) -> Dict:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        with sqlite3.connect(self.index_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM authors")
            total_authors = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM papers WHERE pdf_path IS NOT NULL")
            papers_with_pdf = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT a.id) FROM authors a JOIN author_papers ap ON a.id = ap.author_id")
            authors_with_papers = cursor.fetchone()[0]
            
            return {
                "total_papers": total_papers,
                "total_authors": total_authors,
                "papers_with_pdf": papers_with_pdf,
                "authors_with_papers": authors_with_papers,
                "pdf_availability_rate": papers_with_pdf / total_papers if total_papers > 0 else 0
            }


def main():
    """CLI interface for author-paper index management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Author-Paper Index Management")
    parser.add_argument("--storage", default="data/papers", help="Storage root directory")
    parser.add_argument("--index-db", default="data/author_paper_index.db", help="Index database path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Rebuild index
    subparsers.add_parser("rebuild", help="Rebuild the entire index")
    
    # Search by author
    search_parser = subparsers.add_parser("search", help="Search papers by author")
    search_parser.add_argument("author", help="Author name to search")
    search_parser.add_argument("--exact", action="store_true", help="Use exact matching")
    
    # List authors
    subparsers.add_parser("authors", help="List all authors")
    
    # Statistics
    subparsers.add_parser("stats", help="Show index statistics")
    
    # Get PDF path
    pdf_parser = subparsers.add_parser("pdf", help="Get PDF path for paper")
    pdf_parser.add_argument("paper_id", help="Paper ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize index
    index = AuthorPaperIndex(storage_root=args.storage, index_db_path=args.index_db)
    
    if args.command == "rebuild":
        logging.info("🔄 Rebuilding author-paper index...")
        index.rebuild_index()
        logging.info("✅ Index rebuild completed")
        
    elif args.command == "search":
        logging.info(f"🔍 Searching for papers by '{args.author}'...")
        papers = index.find_papers_by_author(args.author, exact_match=args.exact)
        
        if not papers:
            logging.warning(f"No papers found for author '{args.author}'")
        else:
            logging.info(f"Found {len(papers)} papers:")
            for paper in papers:
                pdf_status = "📄 PDF Available" if paper["pdf_path"] else "❌ PDF Missing"
                logging.info(f"\n• {paper['title'][:60]}...")
                logging.info(f"  Year: {paper['year']} | {pdf_status}")
                logging.info(f"  Paper ID: {paper['paper_id']}")
                if paper["pdf_path"]:
                    logging.info(f"  PDF: {paper['pdf_path']}")
                    
    elif args.command == "authors":
        logging.info("👥 All authors in index:")
        authors = index.get_all_authors()
        for author in authors[:20]:  # Show top 20
            logging.info(f"• {author['name']} ({author['paper_count']} papers)")
        if len(authors) > 20:
            logging.info(f"... and {len(authors) - 20} more authors")
            
    elif args.command == "stats":
        logging.info("📊 Index Statistics:")
        stats = index.get_statistics()
        logging.info(f"Total papers: {stats['total_papers']}")
        logging.info(f"Total authors: {stats['total_authors']}")
        logging.info(f"Papers with PDF: {stats['papers_with_pdf']}")
        logging.info(f"Authors with papers: {stats['authors_with_papers']}")
        logging.info(f"PDF availability: {stats['pdf_availability_rate']:.1%}")
        
    elif args.command == "pdf":
        pdf_path = index.get_paper_pdf_path(args.paper_id)
        if pdf_path:
            logging.info(f"📄 PDF path for {args.paper_id}:")
            logging.info(f"  {pdf_path}")
            logging.info(f"  Exists: {'✅' if os.path.exists(pdf_path) else '❌'}")
        else:
            logging.error(f"❌ No PDF path found for paper {args.paper_id}")


if __name__ == "__main__":
    main() 