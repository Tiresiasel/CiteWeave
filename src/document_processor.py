"""
document_processor.py
Unified document processing pipeline that coordinates PDF processing and citation analysis.
Performs sentence-level citation analysis as the primary output.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from pdf_processor import PDFProcessor
from citation_parser import CitationParser

logging.basicConfig(level=logging.INFO)

class DocumentProcessor:
    """
    Unified document processor that coordinates PDF processing and citation analysis.
    Main responsibility: Extract sentences and analyze citations for each sentence.
    """
    
    def __init__(self, storage_root: str = "./data/papers/", preferred_pdf_engine: str = "auto"):
        """
        Initialize the document processor with both PDF and citation processing capabilities.
        
        Args:
            storage_root: Root directory for storing processed documents
            preferred_pdf_engine: Preferred PDF processing engine ("auto", "pymupdf", etc.)
        """
        self.storage_root = storage_root
        self.pdf_processor = PDFProcessor(storage_root=storage_root, preferred_engine=preferred_pdf_engine)
        self.citation_parser = None  # Will be initialized per document
        
        # Cache for avoiding duplicate processing
        self._metadata_cache = {}
        self._references_cache = {}
        
        logging.info("DocumentProcessor initialized with unified PDF and citation processing")
    
    def process_document(self, pdf_path: str, save_results: bool = True) -> Dict:
        """
        Main processing method: extract sentences and analyze citations for each sentence.
        
        Args:
            pdf_path: Path to the PDF file
            save_results: Whether to save results to disk
            
        Returns:
            Dict containing:
            - metadata: Document metadata
            - sentences_with_citations: List of sentences with their citation analysis
            - processing_stats: Statistics about the processing
        """
        logging.info(f"Starting document processing for: {pdf_path}")
        
        # Step 1: Extract metadata (shared between PDF and citation processing)
        metadata = self._get_or_extract_metadata(pdf_path)
        paper_id = self.pdf_processor._generate_paper_id(metadata["title"], metadata["year"])
        
        # Step 2: Extract sentences using PDFProcessor
        logging.info("Extracting sentences from PDF...")
        sentences = self.pdf_processor.parse_sentences(pdf_path)
        
        # Step 3: Initialize CitationParser with extracted metadata and references
        logging.info("Initializing citation analysis...")
        references = self._get_or_extract_references(pdf_path)
        
        # Get full document text for CitationParser (if it needs it)
        full_text, _ = self.pdf_processor.extract_text_with_best_engine(pdf_path)
        
        # Initialize CitationParser with shared data
        self.citation_parser = CitationParser(
            pdf_path=pdf_path,
            full_doc_text=full_text,
            references=references
        )
        
        # Step 4: Analyze citations for each sentence
        logging.info(f"Analyzing citations for {len(sentences)} sentences...")
        sentences_with_citations = self._analyze_sentences_citations(sentences)
        
        # Step 5: Compile results
        results = {
            "metadata": metadata,
            "paper_id": paper_id,
            "sentences_with_citations": sentences_with_citations,
            "processing_stats": {
                "total_sentences": len(sentences),
                "sentences_with_citations": len([s for s in sentences_with_citations if s["citations"]]),
                "total_citations": sum(len(s["citations"]) for s in sentences_with_citations),
                "total_references": len(references),
                "processing_timestamp": datetime.now().isoformat()
            }
        }
        
        # Step 6: Save results if requested
        if save_results:
            self._save_processed_document(paper_id, results)
        
        logging.info(f"Document processing completed. Found {results['processing_stats']['total_citations']} citations in {results['processing_stats']['sentences_with_citations']} sentences")
        
        return results
    
    def _analyze_sentences_citations(self, sentences: List[str]) -> List[Dict]:
        """
        Analyze citations for each sentence using the CitationParser.
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            List of dictionaries with sentence text and citation analysis
        """
        sentences_with_citations = []
        
        for idx, sentence in enumerate(sentences):
            try:
                # Use CitationParser to extract and match citations for this sentence
                citation_mappings = self.citation_parser.parse_sentence(sentence)
                
                # Clean the sentence text for final output
                cleaned_sentence = self.pdf_processor._clean_sentence_text(sentence)
                
                sentence_data = {
                    "sentence_index": idx,
                    "sentence_text": cleaned_sentence,
                    "citations": citation_mappings,
                    "word_count": len(cleaned_sentence.split()),
                    "char_count": len(cleaned_sentence)
                }
                
                sentences_with_citations.append(sentence_data)
                
                if citation_mappings:
                    logging.debug(f"Sentence {idx}: Found {len(citation_mappings)} citations")
                
            except Exception as e:
                logging.warning(f"Failed to analyze citations for sentence {idx}: {e}")
                
                # Clean the sentence text even for failed analysis
                cleaned_sentence = self.pdf_processor._clean_sentence_text(sentence)
                
                sentences_with_citations.append({
                    "sentence_index": idx,
                    "sentence_text": cleaned_sentence,
                    "citations": [],
                    "error": str(e),
                    "word_count": len(cleaned_sentence.split()),
                    "char_count": len(cleaned_sentence)
                })
        
        return sentences_with_citations
    
    def _get_or_extract_metadata(self, pdf_path: str) -> Dict:
        """
        Get metadata from cache or extract it using PDFProcessor.
        Avoids duplicate metadata extraction.
        """
        if pdf_path not in self._metadata_cache:
            logging.info("Extracting document metadata...")
            self._metadata_cache[pdf_path] = self.pdf_processor.extract_pdf_metadata(pdf_path)
        return self._metadata_cache[pdf_path]
    
    def _get_or_extract_references(self, pdf_path: str) -> List[Dict]:
        """
        Get references from cache or extract them.
        Avoids duplicate reference extraction.
        """
        if pdf_path not in self._references_cache:
            logging.info("Extracting document references...")
            try:
                # Create a temporary CitationParser just for reference extraction
                temp_parser = CitationParser(pdf_path)
                extracted_refs = temp_parser.references
                
                if not extracted_refs:
                    logging.info(f"No references found in {pdf_path} - document may not have a reference section")
                else:
                    logging.info(f"Successfully extracted {len(extracted_refs)} references")
                
                self._references_cache[pdf_path] = extracted_refs
                
            except Exception as e:
                logging.warning(f"Failed to extract references from {pdf_path}: {e}")
                logging.info("Continuing processing without references - citation analysis will be limited")
                self._references_cache[pdf_path] = []
                
        return self._references_cache[pdf_path]
    
    def _save_processed_document(self, paper_id: str, results: Dict):
        """
        Save the complete processed document results to disk.
        
        Args:
            paper_id: Unique paper identifier
            results: Complete processing results
        """
        # Create paper directory
        paper_dir = os.path.join(self.storage_root, paper_id)
        os.makedirs(paper_dir, exist_ok=True)
        
        # Save complete results
        results_path = os.path.join(paper_dir, "processed_document.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save sentences with citations in JSONL format for easy querying
        sentences_path = os.path.join(paper_dir, "sentences_with_citations.jsonl")
        with open(sentences_path, "w", encoding="utf-8") as f:
            for sentence_data in results["sentences_with_citations"]:
                f.write(json.dumps(sentence_data, ensure_ascii=False) + "\n")
        
        # Save metadata separately for compatibility
        metadata_path = os.path.join(paper_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(results["metadata"], f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved processed document to {paper_dir}")
    
    def load_processed_document(self, pdf_path: str) -> Optional[Dict]:
        """
        Load previously processed document results.
        
        Args:
            pdf_path: Path to the original PDF file
            
        Returns:
            Processed document results or None if not found
        """
        try:
            metadata = self._get_or_extract_metadata(pdf_path)
            paper_id = self.pdf_processor._generate_paper_id(metadata["title"], metadata["year"])
            
            results_path = os.path.join(self.storage_root, paper_id, "processed_document.json")
            
            if os.path.exists(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            
        except Exception as e:
            logging.warning(f"Failed to load processed document: {e}")
        
        return None
    
    def get_sentences_with_citations(self, pdf_path: str, force_reprocess: bool = False) -> List[Dict]:
        """
        Get sentences with citation analysis, either from cache or by processing.
        
        Args:
            pdf_path: Path to the PDF file
            force_reprocess: If True, reprocess even if cached results exist
            
        Returns:
            List of sentences with their citation analysis
        """
        if not force_reprocess:
            cached_results = self.load_processed_document(pdf_path)
            if cached_results:
                logging.info("Using cached processed document")
                return cached_results["sentences_with_citations"]
        
        # Process the document
        results = self.process_document(pdf_path)
        return results["sentences_with_citations"]
    
    def diagnose_document_processing(self, pdf_path: str) -> Dict:
        """
        Comprehensive diagnosis of document processing quality.
        Combines PDF processing and citation analysis diagnostics.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Comprehensive diagnosis report
        """
        diagnosis = {
            "pdf_path": pdf_path,
            "timestamp": datetime.now().isoformat(),
            "pdf_diagnosis": {},
            "citation_diagnosis": {},
            "overall_assessment": {}
        }
        
        try:
            # PDF processing diagnosis
            diagnosis["pdf_diagnosis"] = self.pdf_processor.diagnose_pdf_quality(pdf_path)
            
            # Citation processing diagnosis
            metadata = self._get_or_extract_metadata(pdf_path)
            references = self._get_or_extract_references(pdf_path)
            
            diagnosis["citation_diagnosis"] = {
                "references_count": len(references),
                "references_extraction_success": len(references) > 0,
                "metadata_quality": len(metadata.get("title", "")) > 5,
                "has_doi": metadata.get("doi", "Unknown DOI") != "Unknown DOI"
            }
            
            # Overall assessment
            pdf_quality = diagnosis["pdf_diagnosis"].get("best_quality_score", 0)
            citation_quality = len(references) > 0
            
            diagnosis["overall_assessment"] = {
                "is_processable": pdf_quality > 2.0 and citation_quality,
                "quality_level": self._assess_quality_level(pdf_quality, citation_quality),
                "recommendations": self._generate_processing_recommendations(diagnosis)
            }
            
        except Exception as e:
            diagnosis["error"] = str(e)
            diagnosis["overall_assessment"] = {
                "is_processable": False,
                "quality_level": "failed",
                "recommendations": ["Document processing failed - manual inspection required"]
            }
        
        return diagnosis
    
    def _assess_quality_level(self, pdf_quality: float, citation_quality: bool) -> str:
        """Assess overall document processing quality level."""
        if pdf_quality >= 7.0 and citation_quality:
            return "excellent"
        elif pdf_quality >= 5.0 and citation_quality:
            return "good"
        elif pdf_quality >= 3.0:
            return "fair"
        else:
            return "poor"
    
    def _generate_processing_recommendations(self, diagnosis: Dict) -> List[str]:
        """Generate recommendations based on diagnosis."""
        recommendations = []
        
        pdf_quality = diagnosis["pdf_diagnosis"].get("best_quality_score", 0)
        citation_quality = diagnosis["citation_diagnosis"].get("references_extraction_success", False)
        
        if pdf_quality < 3.0:
            recommendations.append("PDF text extraction quality is low - consider manual preprocessing")
        
        if not citation_quality:
            recommendations.append("No references extracted - check GROBID service and PDF format")
        
        if diagnosis["citation_diagnosis"].get("references_count", 0) < 5:
            recommendations.append("Few references found - document may not be a research paper")
        
        if not diagnosis["citation_diagnosis"].get("has_doi", False):
            recommendations.append("No DOI found - metadata quality may be limited")
        
        if not recommendations:
            recommendations.append("Document appears suitable for citation analysis")
        
        return recommendations


# Example usage and integration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the unified document processor
    doc_processor = DocumentProcessor()
    
    # Example: Process a document and get sentence-level citation analysis
    pdf_path = "test_files/Rivkin - 2000 - Imitation of Complex Strategies.pdf"
    
    # Get sentences with citations
    sentences_with_citations = doc_processor.get_sentences_with_citations(pdf_path)
    
    # Print summary
    total_citations = sum(len(s["citations"]) for s in sentences_with_citations)
    sentences_with_citations_count = len([s for s in sentences_with_citations if s["citations"]])
    
    print(f"Processed {len(sentences_with_citations)} sentences")
    print(f"Found {total_citations} citations in {sentences_with_citations_count} sentences")
    
    # Example: Diagnose processing quality
    diagnosis = doc_processor.diagnose_document_processing(pdf_path)
    print(f"Processing quality: {diagnosis['overall_assessment']['quality_level']}") 