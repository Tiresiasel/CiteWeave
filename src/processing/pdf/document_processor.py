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
import shutil  # Add this at the top if not already present

from src.processing.pdf.pdf_processor import PDFProcessor
from src.processing.citation_parser import CitationParser
from src.storage.graph_builder import GraphDB
from src.utils.config_manager import ConfigManager
from src.utils.paper_id_utils import PaperIDGenerator
from src.storage.vector_indexer import VectorIndexer

logging.basicConfig(level=logging.INFO)

# Dataclasses removed - using direct dictionary structures for better compatibility

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
        # Initialize Paper ID Generator
        self.paper_id_generator = PaperIDGenerator()
        
        # Initialize PDF processor based on configuration
        config_manager = ConfigManager()
        pdf_config = config_manager.model_config.get('pdf_processing', {})
        enable_mineru = pdf_config.get('enable_mineru', False)
        
        if enable_mineru:
            try:
                from src.processing.pdf.pdf_processor_mineru import MinerUPDFProcessor
                self.pdf_processor = MinerUPDFProcessor(
                    storage_root=storage_root, 
                    preferred_engine=preferred_pdf_engine,
                    mineru_enabled=True,
                    mineru_fallback=pdf_config.get('mineru_fallback', True)
                )
                logging.info("Initialized with MinerU-enhanced PDF processor (enabled via config)")
            except ImportError:
                logging.warning("MinerU enabled in config but not available, using traditional processor")
                from src.processing.pdf.pdf_processor import PDFProcessor
                self.pdf_processor = PDFProcessor(storage_root=storage_root, preferred_engine=preferred_pdf_engine)
            except Exception as e:
                logging.warning(f"Failed to initialize MinerU processor, using traditional: {e}")
                from src.processing.pdf.pdf_processor import PDFProcessor
                self.pdf_processor = PDFProcessor(storage_root=storage_root, preferred_engine=preferred_pdf_engine)
        else:
            # 使用传统PDF处理器
            from src.processing.pdf.pdf_processor import PDFProcessor
            self.pdf_processor = PDFProcessor(storage_root=storage_root, preferred_engine=preferred_pdf_engine)
            logging.info("Initialized with traditional PDF processor (MinerU disabled in config)")
        
        self.citation_parser = None  # Will be initialized per document
        
        # Initialize Vector Indexer for multi-level embedding
        self.vector_indexer = None
        try:
            from src.storage.vector_indexer import VectorIndexer
            self.vector_indexer = VectorIndexer()
            logging.info("Vector indexer initialized for multi-level embedding")
        except Exception as e:
            logging.warning(f"Failed to initialize vector indexer: {e}")
        
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
    
    def process_document(self, pdf_path: str, save_results: bool = True, create_graph: bool = True, create_embeddings: bool = True) -> Dict:
        """
        Main processing method: extract sentences and analyze citations for each sentence.
        
        Args:
            pdf_path: Path to the PDF file
            save_results: Whether to save results to disk
            create_graph: Whether to create graph database entries
            create_embeddings: Whether to create vector embeddings
            
        Returns:
            Dict containing:
            - metadata: Document metadata
            - sections: List of document sections
            - paragraphs: List of document paragraphs  
            - sentences_with_citations: List of sentences with their citation analysis
            - processing_stats: Statistics about the processing
        """
        logging.info(f"Starting document processing for: {pdf_path}")
        
        # Step 1: Extract metadata (shared between PDF and citation processing)
        metadata = self._get_or_extract_metadata(pdf_path)
        
        # Step 1.5: Generate consistent paper ID using PaperIDGenerator
        paper_id = self.paper_id_generator.generate_paper_id(
            title=metadata["title"], 
            year=metadata["year"]
        )
        logging.info(f"Generated paper ID: {paper_id}")

        # Copy the original PDF into the paper's folder
        paper_dir = os.path.join(self.storage_root, paper_id)
        os.makedirs(paper_dir, exist_ok=True)
        pdf_dest = os.path.join(paper_dir, "original.pdf")
        if not os.path.exists(pdf_dest):
            shutil.copy2(pdf_path, pdf_dest)
            logging.info(f"Copied original PDF to {pdf_dest}")
        else:
            logging.info(f"Original PDF already exists at {pdf_dest}")
        
        # Step 2: Extract document structure (sections, paragraphs, sentences)
        logging.info("Extracting document structure from PDF...")
        structure = self.pdf_processor.extract_document_structure(pdf_path)
        sections = structure["sections"]
        structured_paragraphs = structure["paragraphs"]
        
        # Step 3: Extract sentences for citation analysis using the same text source
        logging.info("Extracting sentences for citation analysis...")
        # 使用相同的文本源确保一致性
        full_text, _ = self.pdf_processor.extract_text_with_best_engine(pdf_path)
        main_content, _ = self.pdf_processor._separate_main_content_and_references(full_text)
        sentences = self.pdf_processor._split_sentences_academic_aware(main_content)
        sentences = self.pdf_processor._filter_invalid_sentences(sentences)
        sentences = [self.pdf_processor._clean_sentence_text(sent) for sent in sentences if sent.strip()]
        
        # Step 4: Initialize CitationParser with extracted metadata and references
        logging.info("Initializing citation analysis...")
        references = self._get_or_extract_references(pdf_path)
        
        
        # Initialize CitationParser with shared data
        self.citation_parser = CitationParser(
            pdf_path=pdf_path,
            full_doc_text=main_content,  # 使用相同的main_content
            references=references
        )
        
        # Step 5: Analyze citations for each sentence
        logging.info(f"Analyzing citations for {len(sentences)} sentences...")
        sentences_with_citations = self._analyze_sentences_citations(sentences)
        
        # Step 5.5: Map citations to paragraphs (只调用一次，确保一致性)
        logging.info("Mapping citations to paragraphs...")
        paragraph_citation_map = self._map_paragraphs_to_citations(structured_paragraphs, sentences_with_citations)
        
        # Step 6: Create graph database entries if enabled
        graph_stats = {}
        if create_graph and self.enable_graph_db and self.graph_db:
            logging.info("Creating graph database entries...")
            graph_stats = self._create_graph_entries_structured(paper_id, metadata, sections, structured_paragraphs, sentences_with_citations, paragraph_citation_map)
        
        # Step 7: Create vector embeddings if enabled
        embedding_stats = {}
        if create_embeddings and self.vector_indexer:
            logging.info("Creating vector embeddings...")
            embedding_stats = self._create_vector_embeddings(paper_id, metadata, sections, structured_paragraphs, sentences_with_citations, paragraph_citation_map)
        
        # Step 8: Convert to parallel structure and unify citation format
        logging.info("Converting to parallel structure with unified citation format...")
        
        # Convert sections to sentence-like format
        sections_with_citations = self._convert_sections_to_citation_format(sections, paragraph_citation_map)
        
        # Convert paragraphs to sentence-like format  
        paragraphs_with_citations = self._convert_paragraphs_to_citation_format(structured_paragraphs, paragraph_citation_map)
        
        # Step 8: Compile results with parallel structure
        results = {
            "metadata": metadata,
            "paper_id": paper_id,
            "sections": sections_with_citations,
            "paragraphs": paragraphs_with_citations,
            "sentences": sentences_with_citations,
            "processing_stats": {
                "total_sections": len(sections_with_citations),
                "total_paragraphs": len(paragraphs_with_citations),
                "total_sentences": len(sentences_with_citations),
                "sentences_with_citations": len([s for s in sentences_with_citations if s["citations"]]),
                "sections_with_citations": len([s for s in sections_with_citations if s["citations"]]),
                "paragraphs_with_citations": len([p for p in paragraphs_with_citations if p["citations"]]),
                "total_citations": sum(len(s["citations"]) for s in sentences_with_citations),
                "total_references": len(references),
                "processing_timestamp": datetime.now().isoformat(),
                "graph_db_stats": graph_stats,
                "embedding_stats": embedding_stats
            }
        }
        
        # Step 9: Save results if requested
        if save_results:
            self._save_processed_document(paper_id, results)
        
        logging.info(f"Document processing completed. Found {results['processing_stats']['total_sections']} sections, {results['processing_stats']['total_paragraphs']} paragraphs, {results['processing_stats']['total_citations']} citations in {results['processing_stats']['sentences_with_citations']} sentences")
        
        return results

# Method removed - replaced by _create_graph_entries_structured
    
    def _create_graph_entries_structured(self, paper_id: str, metadata: Dict, sections: List[Dict], 
                                       paragraphs: List[Dict], sentences_with_citations: List[Dict],
                                       paragraph_citation_map: Dict[str, List[Dict]]) -> Dict:
        """
        使用真实PDF结构创建图数据库条目
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
            
            stats = {
                "sections_created": 0,
                "paragraphs_created": 0,
                "sentences_created": 0,
                "citation_relations_created": 0
            }
            
            # Process structured paragraphs and link to citations
            # paragraph_citation_map is already populated by _map_paragraphs_to_citations
            
            for paragraph in paragraphs:
                # Create paragraph node with structure information
                para_citations = paragraph_citation_map.get(paragraph["id"], [])
                
                self.graph_db.create_paragraph(
                    paragraph_id=paragraph["id"],
                    paper_id=paper_id,
                    text=paragraph["text"],
                    paragraph_index=paragraph["index"],
                    section=paragraph.get("section", "Unknown"),
                    citation_count=len(para_citations),
                    sentence_count=paragraph.get("sentence_count", 0),
                    has_citations=len(para_citations) > 0
                )
                stats["paragraphs_created"] += 1
                
                # Create paragraph-level citation relationships
                for citation in para_citations:
                    cited_paper_id = self._get_or_create_cited_paper(citation)
                    
                    self.graph_db.create_paragraph_citation(
                        paragraph_id=paragraph["id"],
                        cited_paper_id=cited_paper_id,
                        citation_count=1,
                        citation_density=1.0 / paragraph.get("sentence_count", 1)
                    )
                    stats["citation_relations_created"] += 1
            
            # Process sentence-level data for detailed citation analysis
            for sentence_data in sentences_with_citations:
                sentence_id = f"{paper_id}_sent_{sentence_data['sentence_index']}"
                
                # Find paragraph for sentence by text matching
                paragraph_id = "unknown_paragraph"
                sentence_text = sentence_data["sentence_text"]
                for para in paragraphs:
                    if sentence_text in para["text"]:
                        paragraph_id = para["id"]
                        break
                
                self.graph_db.create_sentence(
                    sentence_id=sentence_id,
                    paper_id=paper_id,
                    paragraph_id=paragraph_id,
                    text=sentence_data["sentence_text"],
                    sentence_index=sentence_data["sentence_index"],
                    has_citations=len(sentence_data["citations"]) > 0,
                    word_count=sentence_data["word_count"],
                    char_count=sentence_data["char_count"]
                )
                stats["sentences_created"] += 1
                
                # Create sentence-level citation relationships
                for citation in sentence_data["citations"]:
                    cited_paper_id = self._get_or_create_cited_paper(citation)
                    
                    self.graph_db.create_sentence_citation(
                        sentence_id=sentence_id,
                        cited_paper_id=cited_paper_id,
                        citation_text=citation.get("intext", ""),
                        citation_context=self._extract_citation_context(sentence_data["sentence_text"], citation),
                        confidence=citation.get("confidence", 1.0)
                    )
                    stats["citation_relations_created"] += 1
            
            stats["sections_created"] = len(sections)
            logging.info(f"Structured graph database entries created: {stats}")
            return stats
            
        except Exception as e:
            logging.error(f"Failed to create structured graph database entries: {e}")
            return {"error": str(e)}
    
    def _create_vector_embeddings(self, paper_id: str, metadata: Dict, sections: List[Dict], 
                                 paragraphs: List[Dict], sentences_with_citations: List[Dict],
                                 paragraph_citation_map: Dict[str, List[Dict]]) -> Dict:
        """
        创建多层次向量嵌入
        """
        try:
            stats = {
                "sentences_indexed": 0,
                "paragraphs_indexed": 0,
                "sections_indexed": 0,
                "citations_indexed": 0
            }
            
            # 首先更新段落的引用计数（确保数据一致性）
            # paragraph_citation_map is already populated by _map_paragraphs_to_citations
            
            # Index sentences
            if sentences_with_citations:
                sentence_texts = [s["sentence_text"] for s in sentences_with_citations]
                sentence_types = [s.get("argument_type", "unspecified") for s in sentences_with_citations]
                
                self.vector_indexer.index_sentences(
                    paper_id=paper_id,
                    sentences=sentence_texts,
                    metadata=metadata,
                    claim_types=sentence_types
                )
                stats["sentences_indexed"] = len(sentence_texts)
            
            # Index paragraphs (使用更新后的引用计数)
            if paragraphs:
                # Convert paragraph format for vector indexer
                paragraph_data = []
                for para in paragraphs:
                    # 使用更新后的citation_count
                    citation_count = para.get("citation_count", 0)
                    
                    paragraph_data.append({
                        "text": para["text"],
                        "section": para.get("section", ""),
                        "citation_count": citation_count,
                        "sentence_count": para.get("sentence_count", 0),
                        "has_citations": citation_count > 0
                    })
                
                self.vector_indexer.index_paragraphs(
                    paper_id=paper_id,
                    paragraphs=paragraph_data,
                    metadata=metadata
                )
                stats["paragraphs_indexed"] = len(paragraph_data)
            
            # Index sections
            if sections:
                section_data = []
                for section in sections:
                    section_data.append({
                        "text": section["text"],
                        "title": section["title"],
                        "type": section["section_type"],
                        "paragraph_count": section["paragraph_count"]
                    })
                
                self.vector_indexer.index_sections(
                    paper_id=paper_id,
                    sections=section_data,
                    metadata=metadata
                )
                stats["sections_indexed"] = len(section_data)
            
            # Index citations
            all_citations = []
            for sentence_data in sentences_with_citations:
                for citation in sentence_data["citations"]:
                    citation_data = {
                        "text": citation.get("intext", ""),
                        "cited_paper_id": self._get_or_create_cited_paper(citation),
                        "context": self._extract_citation_context(sentence_data["sentence_text"], citation),
                        "confidence": citation.get("confidence", 1.0)
                    }
                    all_citations.append(citation_data)
            
            if all_citations:
                self.vector_indexer.index_citations(
                    paper_id=paper_id,
                    citations=all_citations,
                    metadata=metadata
                )
                stats["citations_indexed"] = len(all_citations)
            
            logging.info(f"Vector embeddings created: {stats}")
            return stats
            
        except Exception as e:
            logging.error(f"Failed to create vector embeddings: {e}")
            return {"error": str(e)}
    
    def _map_paragraphs_to_citations(self, paragraphs: List[Dict], sentences_with_citations: List[Dict]) -> Dict:
        """
        将段落映射到其包含的引用 - 改进版本
        """
        paragraph_citation_map = {}
        
        def normalize_text(text: str) -> str:
            """标准化文本用于比较"""
            import re
            # 移除多余空白、换行符和特殊字符
            normalized = re.sub(r'\s+', ' ', text.strip())
            normalized = re.sub(r'[^\w\s\(\)\[\],.]', ' ', normalized)
            normalized = re.sub(r'\s+', ' ', normalized)
            return normalized.lower()
        
        for paragraph in paragraphs:
            para_citations = []
            para_text = paragraph["text"]
            para_normalized = normalize_text(para_text)
            
            # 改进的文本匹配策略
            for sentence_data in sentences_with_citations:
                sentence_text = sentence_data["sentence_text"].strip()
                sentence_normalized = normalize_text(sentence_text)
                
                # 多种匹配策略
                match_found = False
                
                # 1. 标准化文本直接包含检查
                if sentence_normalized in para_normalized:
                    match_found = True
                
                # 2. 反向检查 - 段落文本片段在句子中
                elif len(sentence_normalized) > 100:
                    # 对于长句子，检查段落的关键片段是否包含
                    para_words = para_normalized.split()
                    if len(para_words) > 10:
                        # 取段落的中间部分进行匹配
                        middle_start = len(para_words) // 4
                        middle_end = 3 * len(para_words) // 4
                        middle_text = ' '.join(para_words[middle_start:middle_end])
                        if len(middle_text) > 30 and middle_text in sentence_normalized:
                            match_found = True
                
                # 3. 词汇重叠度检查
                elif len(sentence_normalized) > 30:
                    sentence_words = set(sentence_normalized.split())
                    para_words = set(para_normalized.split())
                    
                    # 计算交集比例
                    if len(sentence_words) > 5:
                        overlap = len(sentence_words & para_words) / len(sentence_words)
                        if overlap > 0.7:  # 70%的词汇重叠
                            match_found = True
                
                # 4. 引用文本匹配：对于包含引用的句子，检查引用是否在段落中
                if not match_found and sentence_data["citations"]:
                    for citation in sentence_data["citations"]:
                        intext = citation.get("intext", "").strip()
                        if intext and len(intext) > 3:
                            # 标准化引用文本
                            intext_normalized = normalize_text(intext)
                            if intext_normalized in para_normalized:
                                match_found = True
                                break
                
                # 5. 关键短语匹配
                if not match_found and len(sentence_normalized) > 50:
                    # 提取句子中的关键短语（连续3-5个词）
                    sentence_words = sentence_normalized.split()
                    for i in range(len(sentence_words) - 4):
                        phrase = ' '.join(sentence_words[i:i+5])
                        if len(phrase) > 20 and phrase in para_normalized:
                            match_found = True
                            break
                
                if match_found:
                    para_citations.extend(sentence_data["citations"])
            
            paragraph_citation_map[paragraph["id"]] = para_citations
            
            # 更新段落的引用计数
            paragraph["citation_count"] = len(para_citations)
        
        return paragraph_citation_map
    
# Method removed - paragraph mapping now handled by _map_paragraphs_to_citations

# Method removed - now using PDF structure-based paragraphs instead of sentence grouping

# Method removed - section information now comes from PDF structure

# Method removed - citation aggregation now handled in _map_paragraphs_to_citations

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
            year=year
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
            for sentence_data in results["sentences"]:
                f.write(json.dumps(sentence_data, ensure_ascii=False) + "\n")
        
        # Save metadata separately for compatibility
        metadata_path = os.path.join(paper_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(results["metadata"], f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved processed document to {paper_dir}")

    def _convert_sections_to_citation_format(self, sections: List[Dict], paragraph_citation_map: Dict[str, List[Dict]]) -> List[Dict]:
        """
        将章节转换为与句子相同的引文格式
        """
        sections_with_citations = []
        
        for section in sections:
            # 收集这个章节中所有段落的引文
            section_citations = []
            seen_citations = set()  # 避免重复引文
            
            for paragraph in section.get("paragraphs", []):
                para_citations = paragraph_citation_map.get(paragraph["id"], [])
                for citation in para_citations:
                    # 使用intext作为去重键
                    citation_key = citation.get("intext", "")
                    if citation_key and citation_key not in seen_citations:
                        seen_citations.add(citation_key)
                        section_citations.append(citation)
            
            # 构建章节的引文格式数据
            section_data = {
                "section_index": section["index"],
                "section_title": section["title"],
                "section_text": section["text"],
                "section_type": section["section_type"],
                "citations": section_citations,
                "word_count": len(section["text"].split()) if section["text"] else 0,
                "char_count": len(section["text"]) if section["text"] else 0,
                "paragraph_count": section.get("paragraph_count", 0),
                "page_start": section.get("page_start", 0)
            }
            
            sections_with_citations.append(section_data)
        
        return sections_with_citations
    
    def _convert_paragraphs_to_citation_format(self, paragraphs: List[Dict], paragraph_citation_map: Dict[str, List[Dict]]) -> List[Dict]:
        """
        将段落转换为与句子相同的引文格式
        """
        paragraphs_with_citations = []
        
        for paragraph in paragraphs:
            # 获取这个段落的引文
            para_citations = paragraph_citation_map.get(paragraph["id"], [])
            
            # 构建段落的引文格式数据
            paragraph_data = {
                "paragraph_index": paragraph["index"],
                "paragraph_text": paragraph["text"],
                "section": paragraph.get("section", "Unknown"),
                "citations": para_citations,
                "word_count": paragraph.get("word_count", 0),
                "char_count": paragraph.get("char_count", 0),
                "sentence_count": paragraph.get("sentence_count", 0),
                "citation_count": len(para_citations),
                "has_citations": len(para_citations) > 0,
                "page": paragraph.get("page", 0)
            }
            
            paragraphs_with_citations.append(paragraph_data)
        
        return paragraphs_with_citations
    
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
                year=metadata["year"]
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
    
    # Initialize the unified document processor with both graph and vector databases
    doc_processor = DocumentProcessor(enable_graph_db=True)
    
    # Test files to process
    test_files = [
        "test_files/Rivkin - 2000 - Imitation of Complex Strategies.pdf",
        "test_files/Porter - Competitive Strategy.pdf",
        "test_files/Business Model Innovation Research 2016.pdf"
    ]
    
    print("=== Starting full database import test ===\n")
    
    for pdf_path in test_files:
        try:
            print(f"📄 Processing document: {pdf_path}")
            print("-" * 50)
            
            # Complete processing: structure parsing + graph DB + vector DB
            results = doc_processor.process_document(
                pdf_path=pdf_path, 
                create_graph=True,      # Create graph DB entries
                create_embeddings=True, # Create vector embeddings
                save_results=True       # Save processing results
            )
            
            # Print processing stats
            stats = results["processing_stats"]
            print(f"✅ Document structure:")
            print(f"   📚 Number of sections: {stats['total_sections']}")
            print(f"   📝 Number of paragraphs: {stats['total_paragraphs']}")
            print(f"   📄 Number of sentences: {stats['total_sentences']}")
            print(f"   🔗 Number of citations: {stats['total_citations']}")
            print(f"   📖 Number of references: {stats['total_references']}")
            
            # Graph DB stats
            if 'graph_db_stats' in stats and stats['graph_db_stats']:
                graph_stats = stats['graph_db_stats']
                print(f"\n✅ Graph DB creation:")
                print(f"   📊 Section nodes: {graph_stats.get('sections_created', 0)}")
                print(f"   📝 Paragraph nodes: {graph_stats.get('paragraphs_created', 0)}")
                print(f"   📄 Sentence nodes: {graph_stats.get('sentences_created', 0)}")
                print(f"   🔗 Citation relations: {graph_stats.get('citation_relations_created', 0)}")
            
            # Vector DB stats
            if 'embedding_stats' in stats and stats['embedding_stats']:
                embedding_stats = stats['embedding_stats']
                print(f"\n✅ Vector DB indexing:")
                print(f"   📄 Sentence vectors: {embedding_stats.get('sentences_indexed', 0)}")
                print(f"   📝 Paragraph vectors: {embedding_stats.get('paragraphs_indexed', 0)}")
                print(f"   📚 Section vectors: {embedding_stats.get('sections_indexed', 0)}")
                print(f"   🔗 Citation vectors: {embedding_stats.get('citations_indexed', 0)}")
            
            print(f"\n📋 Paper ID: {results['paper_id']}")
            print(f"📋 Paper title: {results['metadata']['title']}")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
        
        print("\n" + "="*70 + "\n")
    
    print("🏆 Testing vector DB search functionality:")
    print("-" * 40)
    
    # Test vector search
    if doc_processor.vector_indexer:
        try:
            # Cross-collection search
            search_results = doc_processor.vector_indexer.search_all_collections(
                "strategic competitive advantage", 
                limit_per_collection=2
            )
            
            for collection, results in search_results.items():
                print(f"\n📚 {collection.upper()} search results:")
                if results:
                    for result in results:
                        print(f"   Similarity: {result['score']:.3f}")
                        print(f"   Text: {result['text'][:100]}...")
                        print(f"   Paper: {result.get('title', 'Unknown')}")
                        print("   ---")
                else:
                    print("   No results")
        except Exception as e:
            print(f"❌ Vector search test failed: {e}")
    
    print("\n🏆 Testing graph DB query functionality:")
    print("-" * 40)
    
    # Test graph DB query
    if doc_processor.graph_db:
        try:
            # Test citation network query
            citation_context = doc_processor.get_citation_analysis_context("competitive strategy")
            if citation_context:
                print(f"📖 Found citation context: {len(citation_context.get('citing_sentences', []))} sentences")
                for sentence in citation_context.get('citing_sentences', [])[:3]:
                    print(f"   - {sentence['text'][:100]}...")
            else:
                print("📖 No citation context data available")
        except Exception as e:
            print(f"❌ Graph DB query test failed: {e}")
    
    print("\n🏁 Full database import test completed!")
    print("All test documents have been imported into the graph DB and vector DB") 