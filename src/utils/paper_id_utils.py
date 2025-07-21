"""
Paper ID Utilities for CiteWeave

This module provides unified paper ID generation functionality that ensures
consistent paper identification across the entire system, whether papers
are uploaded directly or referenced in citations.

Key Features:
- Consistent hashing algorithm for paper IDs
- Support for generating IDs from minimal citation information
- Normalization of title and year information
- Fallback mechanisms for incomplete data
"""

import hashlib
import re
from typing import Dict, List, Optional, Union


class PaperIDGenerator:
    """
    Unified paper ID generation utility.
    
    This class ensures consistent paper identification across the system,
    whether papers are directly uploaded or referenced in citations.
    Paper IDs are generated based on normalized title and year only.
    """
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """
        Normalize paper title for consistent hashing.
        
        Args:
            title: Raw title string
            
        Returns:
            str: Normalized title
        """
        if not title or title.lower() in ["unknown title", "unknown", "", "none"]:
            return "unknown_title"
        
        # Remove punctuation and normalize whitespace
        normalized = re.sub(r"[^\w\s]", "", title)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        return normalized if normalized else "unknown_title"
    
    @staticmethod
    def normalize_year(year: Union[str, int]) -> str:
        """
        Normalize publication year for consistent hashing.
        
        Args:
            year: Raw year (string or integer)
            
        Returns:
            str: Normalized year
        """
        if not year:
            return "unknown_year"
        
        # Convert to string and extract 4-digit year
        year_str = str(year).strip()
        year_match = re.search(r'\b(19|20)\d{2}\b', year_str)
        
        if year_match:
            return year_match.group(0)
        
        return "unknown_year"
    
    @staticmethod
    def normalize_authors(authors: Union[List[str], str]) -> List[str]:
        """
        Normalize author list for consistent processing.
        
        Args:
            authors: Raw author information (list or string)
            
        Returns:
            List[str]: Normalized author list
        """
        if not authors:
            return ["unknown_author"]
        
        if isinstance(authors, str):
            # Handle string input - split by common delimiters
            authors = re.split(r'[,;]|and\s+', authors)
        
        normalized_authors = []
        for author in authors:
            if isinstance(author, str):
                # Clean author name
                clean_author = re.sub(r'[^\w\s.-]', '', author.strip())
                clean_author = re.sub(r'\s+', ' ', clean_author).strip()
                
                if clean_author and clean_author.lower() not in ["unknown", "unknown author", ""]:
                    normalized_authors.append(clean_author.lower())
        
        return normalized_authors if normalized_authors else ["unknown_author"]
    
    @classmethod
    def generate_paper_id(cls, title: str, year: Union[str, int], 
                         authors: Optional[Union[List[str], str]] = None) -> str:
        """
        Generate a unique paper ID based on title and year.
        
        This method ensures consistent paper identification across the system,
        whether papers are uploaded directly or referenced in citations.
        
        Args:
            title: Paper title
            year: Publication year
            authors: Author list (ignored for ID generation)
            
        Returns:
            str: SHA256 hash of normalized paper information
        """
        # Normalize inputs
        norm_title = cls.normalize_title(title)
        norm_year = cls.normalize_year(year)
        
        # Create string for hashing (title + year only)
        combined_str = f"{norm_title}_{norm_year}"
        
        # Generate SHA256 hash
        hash_sha256 = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()
        return hash_sha256
    
    @classmethod
    def generate_from_citation(cls, citation_info: Dict) -> str:
        """
        Generate paper ID from citation information.
        
        This method handles the common case where we have citation information
        but haven't uploaded the actual paper yet.
        
        Args:
            citation_info: Dictionary containing citation information
                Expected keys: 'title', 'year'
                
        Returns:
            str: Generated paper ID
        """
        title = citation_info.get("title", "")
        year = citation_info.get("year", "")
        
        return cls.generate_paper_id(title, year)
    
    @classmethod
    def generate_from_reference(cls, reference: Dict) -> str:
        """
        Generate paper ID from reference entry.
        
        This method handles reference entries from bibliography sections.
        
        Args:
            reference: Dictionary containing reference information
                Expected keys: 'title', 'year'
                
        Returns:
            str: Generated paper ID
        """
        return cls.generate_from_citation(reference)
    
    @classmethod
    def are_same_paper(cls, paper_id_1: str, paper_id_2: str) -> bool:
        """
        Check if two paper IDs refer to the same paper.
        
        Args:
            paper_id_1: First paper ID
            paper_id_2: Second paper ID
            
        Returns:
            bool: True if they refer to the same paper
        """
        return paper_id_1 == paper_id_2
    
    @classmethod
    def validate_paper_id(cls, paper_id: str) -> bool:
        """
        Validate if a string is a valid paper ID format.
        
        Args:
            paper_id: Paper ID to validate
            
        Returns:
            bool: True if valid SHA256 hash format
        """
        if not paper_id or not isinstance(paper_id, str):
            return False
        
        # Check if it's a 64-character hexadecimal string (SHA256)
        return len(paper_id) == 64 and all(c in '0123456789abcdef' for c in paper_id.lower())


def generate_paper_id(title: str, year: Union[str, int]) -> str:
    """
    Convenience function for generating paper IDs.
    
    Args:
        title: Paper title
        year: Publication year
        
    Returns:
        str: Generated paper ID
    """
    return PaperIDGenerator.generate_paper_id(title, year)


# Backward compatibility - maintain the original function signature
def _generate_paper_id(title: str, year: str) -> str:
    """Legacy function for backward compatibility."""
    return PaperIDGenerator.generate_paper_id(title, year)


if __name__ == "__main__":
    # Example usage and testing
    generator = PaperIDGenerator()
    
    # Test cases
    test_cases = [
        {
            "title": "Imitation of Complex Strategies",
            "year": "2000",
            "authors": ["Jan W. Rivkin"]
        },
        {
            "title": "Competitive Strategy",
            "year": "1980", 
            "authors": ["Michael E. Porter"]
        },
        {
            "title": "An Evolutionary Theory of Economic Change",
            "year": "1982",
            "authors": ["Richard Nelson", "Sidney Winter"]
        }
    ]
    
    print("Testing Paper ID Generation:")
    print("=" * 50)
    
    for case in test_cases:
        paper_id = generator.generate_paper_id(
            case["title"], 
            case["year"]
        )
        print(f"Title: {case['title']}")
        print(f"Year: {case['year']}")
        print(f"Authors: {case['authors']} (ignored for ID generation)")
        print(f"Paper ID: {paper_id}")
        print("-" * 30) 