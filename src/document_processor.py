"""
document_processor.py
Unified document processing pipeline that coordinates PDF processing and citation analysis.
Performs sentence-level citation analysis as the primary output.
Enhanced with sentence+paragraph dual-layer citation network architecture.
"""

import os
import json
import logging
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from src.pdf_processor import PDFProcessor
from src.citation_parser import CitationParser
from src.graph_builder import GraphDB
from src.config_manager import ConfigManager
from src.paper_id_utils import PaperIDGenerator

logging.basicConfig(level=logging.INFO)

@dataclass
class SentenceData:
    """句子数据结构"""
    id: str
    text: str
    index: int
    has_citations: bool
    citations: List[Dict]
    word_count: int
    char_count: int

@dataclass  
class ParagraphData:
    """段落数据结构"""
    id: str
    text: str
    index: int
    section: str
    sentences: List[SentenceData]
    citation_count: int

class DocumentProcessor:
    """
    Unified document processor that coordinates PDF processing and citation analysis.
    Main responsibility: Extract sentences and analyze citations for each sentence.
    Enhanced with sentence+paragraph dual-layer citation network architecture.
    """
    
    def __init__(self, storage_root: str = "./data/papers/", preferred_pdf_engine: str = "auto", 
                 config_path: str = "config", enable_graph_db: bool = True):
        """
        Initialize the document processor with both PDF and citation processing capabilities.
        
        Args:
            storage_root: Root directory for storing processed documents
            preferred_pdf_engine: Preferred PDF processing engine ("auto", "pymupdf", etc.)
            config_path: Path to configuration files
            enable_graph_db: Whether to enable graph database integration
        """
        self.storage_root = storage_root
        self.pdf_processor = PDFProcessor(storage_root=storage_root, preferred_engine=preferred_pdf_engine)
        self.citation_parser = None  # Will be initialized per document
        
        # Initialize Paper ID Generator
        self.paper_id_generator = PaperIDGenerator()
        
        # Graph database integration
        self.enable_graph_db = enable_graph_db
        self.graph_db = None
        if enable_graph_db:
            try:
                config_manager = ConfigManager(config_path)
                neo4j_config = config_manager.neo4j_config
                self.graph_db = GraphDB(
                    uri=neo4j_config["uri"],
                    user=neo4j_config["username"], 
                    password=neo4j_config["password"]
                )
                logging.info("Graph database integration enabled")
            except Exception as e:
                logging.warning(f"Failed to initialize graph database: {e}")
                self.enable_graph_db = False
        
        # Cache for avoiding duplicate processing
        self._metadata_cache = {}
        self._references_cache = {}
        
        logging.info("DocumentProcessor initialized with unified PDF and citation processing")
    
    def process_document(self, pdf_path: str, save_results: bool = True, create_graph: bool = True) -> Dict:
        """
        Main processing method: extract sentences and analyze citations for each sentence.
        
        Args:
            pdf_path: Path to the PDF file
            save_results: Whether to save results to disk
            create_graph: Whether to create graph database entries
            
        Returns:
            Dict containing:
            - metadata: Document metadata
            - sentences_with_citations: List of sentences with their citation analysis
            - processing_stats: Statistics about the processing
        """
        logging.info(f"Starting document processing for: {pdf_path}")
        
        # Step 1: Extract metadata (shared between PDF and citation processing)
        metadata = self._get_or_extract_metadata(pdf_path)
        
        # Step 1.5: Generate consistent paper ID using PaperIDGenerator
        paper_id = self.paper_id_generator.generate_paper_id(
            title=metadata["title"], 
            year=metadata["year"],
            authors=metadata.get("authors", [])
        )
        logging.info(f"Generated paper ID: {paper_id}")
        
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
        
        # Step 5: Create graph database entries if enabled
        graph_stats = {}
        if create_graph and self.enable_graph_db and self.graph_db:
            logging.info("Creating graph database entries...")
            graph_stats = self._create_graph_entries(paper_id, metadata, sentences_with_citations)
        
        # Step 6: Compile results
        results = {
            "metadata": metadata,
            "paper_id": paper_id,
            "sentences_with_citations": sentences_with_citations,
            "processing_stats": {
                "total_sentences": len(sentences),
                "sentences_with_citations": len([s for s in sentences_with_citations if s["citations"]]),
                "total_citations": sum(len(s["citations"]) for s in sentences_with_citations),
                "total_references": len(references),
                "processing_timestamp": datetime.now().isoformat(),
                "graph_db_stats": graph_stats
            }
        }
        
        # Step 7: Save results if requested
        if save_results:
            self._save_processed_document(paper_id, results)
        
        logging.info(f"Document processing completed. Found {results['processing_stats']['total_citations']} citations in {results['processing_stats']['sentences_with_citations']} sentences")
        
        return results

    def _create_graph_entries(self, paper_id: str, metadata: Dict, sentences_with_citations: List[Dict]) -> Dict:
        """
        Create graph database entries using the new sentence+paragraph architecture.
        
        Args:
            paper_id: Unique paper identifier
            metadata: Document metadata
            sentences_with_citations: Processed sentences with citation analysis
            
        Returns:
            Statistics about graph creation
        """
        try:
            # Create main paper node
            self.graph_db.create_paper(
                paper_id=paper_id,
                title=metadata.get("title", "Unknown"),
                authors=metadata.get("authors", ["Unknown"]),
                year=int(metadata.get("year", 0)) if metadata.get("year") else 0,
                doi=metadata.get("doi"),
                journal=metadata.get("journal"),
                publisher=metadata.get("publisher")
            )
            
            # Group sentences into paragraphs
            paragraphs = self._group_sentences_into_paragraphs(sentences_with_citations, paper_id)
            
            stats = {
                "paragraphs_created": 0,
                "sentences_created": 0,
                "citation_relations_created": 0,
                "cited_papers_created": 0
            }
            
            # Process each paragraph
            for paragraph_data in paragraphs:
                # Create paragraph node
                self.graph_db.create_paragraph(
                    paragraph_id=paragraph_data.id,
                    paper_id=paper_id,
                    text=paragraph_data.text,
                    paragraph_index=paragraph_data.index,
                    section=paragraph_data.section,
                    citation_count=paragraph_data.citation_count,
                    sentence_count=len(paragraph_data.sentences),
                    has_citations=paragraph_data.citation_count > 0
                )
                stats["paragraphs_created"] += 1
                
                # Process sentences in paragraph
                for sentence in paragraph_data.sentences:
                    # Create sentence node
                    self.graph_db.create_sentence(
                        sentence_id=sentence.id,
                        paper_id=paper_id,
                        paragraph_id=paragraph_data.id,
                        text=sentence.text,
                        sentence_index=sentence.index,
                        has_citations=sentence.has_citations,
                        word_count=sentence.word_count,
                        char_count=sentence.char_count
                    )
                    stats["sentences_created"] += 1
                    
                    # Create sentence-level citation relationships
                    for citation in sentence.citations:
                        cited_paper_id = self._get_or_create_cited_paper(citation)
                        
                        self.graph_db.create_sentence_citation(
                            sentence_id=sentence.id,
                            cited_paper_id=cited_paper_id,
                            citation_text=citation.get("intext", ""),
                            citation_context=self._extract_citation_context(sentence.text, citation),
                            confidence=citation.get("confidence", 1.0)
                        )
                        stats["citation_relations_created"] += 1
                
                # Create paragraph-level citation relationships (aggregated)
                paragraph_citations = self._aggregate_paragraph_citations(paragraph_data)
                for cited_paper_id, count in paragraph_citations.items():
                    citation_density = count / len(paragraph_data.sentences) if paragraph_data.sentences else 0
                    
                    self.graph_db.create_paragraph_citation(
                        paragraph_id=paragraph_data.id,
                        cited_paper_id=cited_paper_id,
                        citation_count=count,
                        citation_density=citation_density
                    )
            
            logging.info(f"Graph database entries created: {stats}")
            return stats
            
        except Exception as e:
            logging.error(f"Failed to create graph database entries: {e}")
            return {"error": str(e)}

    def _group_sentences_into_paragraphs(self, sentences_with_citations: List[Dict], paper_id: str) -> List[ParagraphData]:
        """
        Group sentences into paragraphs for the new architecture.
        Simple implementation - can be enhanced with more sophisticated paragraph detection.
        """
        paragraphs = []
        current_paragraph_sentences = []
        paragraph_index = 0
        
        for i, sentence_data in enumerate(sentences_with_citations):
            sentence_obj = SentenceData(
                id=f"{paper_id}_sent_{i}",
                text=sentence_data["sentence_text"],
                index=sentence_data["sentence_index"],
                has_citations=len(sentence_data["citations"]) > 0,
                citations=sentence_data["citations"],
                word_count=sentence_data["word_count"],
                char_count=sentence_data["char_count"]
            )
            
            current_paragraph_sentences.append(sentence_obj)
            
            # Simple paragraph break detection (every 5 sentences or at natural breaks)
            if (i + 1) % 5 == 0 or i == len(sentences_with_citations) - 1:
                if current_paragraph_sentences:
                    paragraph_text = " ".join([s.text for s in current_paragraph_sentences])
                    total_citations = sum(len(s.citations) for s in current_paragraph_sentences)
                    
                    paragraph = ParagraphData(
                        id=f"{paper_id}_para_{paragraph_index}",
                        text=paragraph_text,
                        index=paragraph_index,
                        section=self._determine_section(paragraph_text, paragraph_index),
                        sentences=current_paragraph_sentences,
                        citation_count=total_citations
                    )
                    
                    paragraphs.append(paragraph)
                    current_paragraph_sentences = []
                    paragraph_index += 1
        
        return paragraphs

    def _determine_section(self, paragraph_text: str, paragraph_index: int) -> str:
        """Determine paragraph section based on content and position."""
        text_lower = paragraph_text.lower()
        
        if paragraph_index < 3:
            return "Introduction"
        elif "method" in text_lower or "approach" in text_lower:
            return "Methodology"
        elif "result" in text_lower or "finding" in text_lower:
            return "Results"
        elif "conclusion" in text_lower or "summary" in text_lower:
            return "Conclusion"
        elif "literature" in text_lower or "prior" in text_lower:
            return "Literature Review"
        else:
            return "Main Content"

    def _aggregate_paragraph_citations(self, paragraph_data: ParagraphData) -> Dict[str, int]:
        """Aggregate citation counts at paragraph level."""
        cited_papers = {}
        
        for sentence in paragraph_data.sentences:
            for citation in sentence.citations:
                cited_paper_id = self._get_or_create_cited_paper(citation)
                cited_papers[cited_paper_id] = cited_papers.get(cited_paper_id, 0) + 1
        
        return cited_papers

    def _get_or_create_cited_paper(self, citation: Dict) -> str:
        """Get or create a cited paper node, returning its ID using PaperIDGenerator."""
        reference = citation.get("reference", {})
        
        # Extract reference information
        title = reference.get("title", "Unknown Title")
        year = reference.get("year", "Unknown")
        authors = reference.get("authors", ["Unknown"])
        
        # Generate consistent paper ID using PaperIDGenerator
        cited_paper_id = self.paper_id_generator.generate_paper_id(
            title=title,
            year=year,
            authors=authors
        )
        
        if self.graph_db:
            # Create stub paper node with generated ID
            self.graph_db.create_paper(
                paper_id=cited_paper_id,
                title=title,
                authors=authors,
                year=int(year) if year != "Unknown" and year.isdigit() else 0,
                stub=True,
                doi=reference.get("doi"),
                journal=reference.get("journal")
            )
        
        return cited_paper_id

    def _extract_citation_context(self, sentence_text: str, citation: Dict) -> str:
        """Extract citation context from sentence."""
        intext = citation.get("intext", "")
        if intext in sentence_text:
            start = sentence_text.find(intext)
            context_start = max(0, start - 50)
            context_end = min(len(sentence_text), start + len(intext) + 50)
            return sentence_text[context_start:context_end]
        return sentence_text

    def get_citation_analysis_context(self, cited_paper_id: str) -> Dict:
        """
        Get citation context for AI analysis using the new architecture.
        This is the core advantage: directly retrieve all relevant information.
        """
        if not self.graph_db:
            return {"error": "Graph database not available"}
        
        return self.graph_db.get_citation_context_for_ai_analysis(cited_paper_id)

    # ==================== 原有方法保持不变 ====================
    
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
            paper_id = self.paper_id_generator.generate_paper_id(
                title=metadata["title"], 
                year=metadata["year"],
                authors=metadata.get("authors", [])
            )
            
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

    def close(self):
        """Close database connections."""
        if self.graph_db:
            self.graph_db.close()


# Example usage and integration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the unified document processor with graph database
    doc_processor = DocumentProcessor(enable_graph_db=True)
    
    # Example: Process a document and get sentence-level citation analysis
    pdf_path = "test_files/Rivkin - 2000 - Imitation of Complex Strategies.pdf"
    
    # Process document with graph creation
    results = doc_processor.process_document(pdf_path, create_graph=True)
    
    # Print summary
    total_citations = results["processing_stats"]["total_citations"]
    sentences_with_citations_count = results["processing_stats"]["sentences_with_citations"]
    
    print(f"Processed {results['processing_stats']['total_sentences']} sentences")
    print(f"Found {total_citations} citations in {sentences_with_citations_count} sentences")
    print(f"Graph DB stats: {results['processing_stats']['graph_db_stats']}")
    
    # Example: Get citation context for AI analysis
    context = doc_processor.get_citation_analysis_context("porter_1980")
    print(f"Citation context: {context}")
    
    # Example: Diagnose processing quality
    diagnosis = doc_processor.diagnose_document_processing(pdf_path)
    print(f"Processing quality: {diagnosis['overall_assessment']['quality_level']}")
    
    doc_processor.close() 