# This file makes 'src' a Python package for Poetry and Python imports.

# Main modules for the argument graph project
from src.processing.pdf.document_processor import DocumentProcessor
from src.processing.pdf.pdf_processor import PDFProcessor  
from src.processing.citation_parser import CitationParser

__all__ = ['DocumentProcessor', 'PDFProcessor', 'CitationParser'] 