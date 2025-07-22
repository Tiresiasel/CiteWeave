"""
query_db_agent.py
Module for querying graph and vector databases with structured functions.
"""

import logging
import re
from typing import List, Dict, Optional, Union
from difflib import SequenceMatcher
from src.storage.graph_builder import GraphDB
from src.storage.vector_indexer import VectorIndexer
from src.utils.config_manager import ConfigManager
import os
import json
from typing import Any
import warnings
import atexit
import sys

# Suppress Neo4j driver warnings and errors
warnings.filterwarnings("ignore", category=DeprecationWarning, module="neo4j")
warnings.filterwarnings("ignore", category=UserWarning, module="neo4j")

logging.basicConfig(level=logging.INFO)

# Global driver reference for cleanup
_neo4j_driver = None

def _cleanup_neo4j_driver():
    """Cleanup function to properly close Neo4j driver on exit"""
    global _neo4j_driver
    if _neo4j_driver:
        try:
            _neo4j_driver.close()
        except Exception:
            pass  # Ignore any cleanup errors
        finally:
            _neo4j_driver = None

# Register cleanup function
atexit.register(_cleanup_neo4j_driver)

class QueryDBAgent:
    """Agent for querying Neo4j graph database, Qdrant vector database, and PDF documents"""
    
    def __init__(self, config_path: str = "config"):
        """
        Initialize the Query DB Agent with graph and vector database connections.
        
        Args:
            config_path: Path to configuration files
        """
        self.config_manager = ConfigManager(config_path)
        
        # Initialize Graph Database
        try:
            neo4j_config = self.config_manager.neo4j_config
            self.graph_db = GraphDB(
                uri=neo4j_config["uri"],
                user=neo4j_config["username"],
                password=neo4j_config["password"]
            )
            logging.info("Graph database connection established")
        except Exception as e:
            logging.error(f"Failed to connect to graph database: {e}")
            self.graph_db = None
        
        # Initialize Vector Database
        try:
            self.vector_indexer = VectorIndexer()
            logging.info("Vector database connection established")
        except Exception as e:
            logging.error(f"Failed to connect to vector database: {e}")
            self.vector_indexer = None
            
        # Add papers directory path
        def find_project_root():
            cur = os.path.abspath(os.getcwd())
            while cur != "/" and not os.path.exists(os.path.join(cur, "README.md")):
                cur = os.path.dirname(cur)
            return cur

        project_root = find_project_root()
        self.papers_dir = os.path.join(project_root, "data", "papers")
        if not os.path.exists(self.papers_dir):
            raise FileNotFoundError(f"Papers directory not found: {self.papers_dir}")
            
    def get_metadata_by_paper_id(self, paper_id: str) -> Dict:
        """
        获取指定paper_id的论文元数据
        
        Args:
            paper_id: 论文的唯一标识符
            
        Returns:
            Dict: 论文的完整元数据信息
        """
        if not self.graph_db:
            logging.error("Graph database not available")
            return {}
        
        try:
            query = """
            MATCH (p:Paper {id: $paper_id})
            RETURN p.id as paper_id,
                   p.title as title,
                   p.authors as authors,
                   p.year as year,
                   p.doi as doi,
                   p.journal as journal,
                   p.publisher as publisher,
                   p.stub as is_stub,
                   p.volume as volume,
                   p.issue as issue,
                   p.pages as pages,
                   p.issn as issn,
                   p.url as url,
                   p.type as type
            """
            
            with self.graph_db.driver.session() as session:
                result = session.run(query, paper_id=paper_id)
                record = result.single()
                
                if record:
                    paper_metadata = {
                        "paper_id": record["paper_id"],
                        "title": record["title"],
                        "authors": record["authors"],
                        "year": record["year"],
                        "doi": record["doi"],
                        "journal": record["journal"],
                        "publisher": record["publisher"],
                        "is_stub": record["is_stub"],
                        "volume": record["volume"],
                        "issue": record["issue"],
                        "pages": record["pages"],
                        "issn": record["issn"],
                        "url": record["url"],
                        "type": record["type"]
                    }
                    
                    logging.info(f"Found metadata for paper {paper_id}")
                    return paper_metadata
                else:
                    logging.warning(f"No paper found with ID {paper_id}")
                    return {}
                    
        except Exception as e:
            logging.error(f"Error getting metadata for paper {paper_id}: {e}")
            return {}
    
    def get_papers_citing_paper(self, target_paper_id: str) -> List[Dict]:
        """
        获取所有引用了指定论文的论文列表
        
        Args:
            target_paper_id: 被引用论文的ID
            
        Returns:
            List[Dict]: 引用该论文的论文列表，包含paper信息和引用统计
        """
        if not self.graph_db:
            logging.error("Graph database not available")
            return []
        
        try:
            query = """
            MATCH (cited_paper:Paper {id: $target_paper_id})
            MATCH (citing_paper:Paper)-[:BELONGS_TO]-(sentence:Sentence)-[:CITES]->(cited_paper)
            WITH citing_paper, COUNT(DISTINCT sentence) as sentence_citation_count
            MATCH (citing_paper)-[:BELONGS_TO]-(paragraph:Paragraph)-[:CITES]->(cited_paper)
            WITH citing_paper, sentence_citation_count, COUNT(DISTINCT paragraph) as paragraph_citation_count
            RETURN citing_paper.id as paper_id,
                   citing_paper.title as title,
                   citing_paper.authors as authors,
                   citing_paper.year as year,
                   citing_paper.journal as journal,
                   sentence_citation_count,
                   paragraph_citation_count,
                   (sentence_citation_count + paragraph_citation_count) as total_citations
            ORDER BY total_citations DESC
            """
            
            with self.graph_db.driver.session() as session:
                result = session.run(query, target_paper_id=target_paper_id)
                
                citing_papers = []
                for record in result:
                    citing_papers.append({
                        "paper_id": record["paper_id"],
                        "title": record["title"],
                        "authors": record["authors"],
                        "year": record["year"],
                        "journal": record["journal"],
                        "sentence_citations": record["sentence_citation_count"],
                        "paragraph_citations": record["paragraph_citation_count"],
                        "total_citations": record["total_citations"]
                    })
                
                logging.info(f"Found {len(citing_papers)} papers citing {target_paper_id}")
                return citing_papers
                
        except Exception as e:
            logging.error(f"Error getting papers citing {target_paper_id}: {e}")
            return []
    
    def get_papers_cited_by_paper(self, source_paper_id: str) -> List[Dict]:
        """
        获取指定论文引用的所有论文列表
        
        Args:
            source_paper_id: 源论文的ID
            
        Returns:
            List[Dict]: 被该论文引用的论文列表，包含被引用论文信息和引用统计
        """
        if not self.graph_db:
            logging.error("Graph database not available")
            return []
        
        try:
            query = """
            MATCH (source_paper:Paper {id: $source_paper_id})
            MATCH (source_paper)-[:BELONGS_TO]-(sentence:Sentence)-[:CITES]->(cited_paper:Paper)
            WITH cited_paper, COUNT(DISTINCT sentence) as sentence_citation_count
            OPTIONAL MATCH (source_paper)-[:BELONGS_TO]-(paragraph:Paragraph)-[:CITES]->(cited_paper)
            WITH cited_paper, sentence_citation_count, COUNT(DISTINCT paragraph) as paragraph_citation_count
            RETURN cited_paper.id as paper_id,
                   cited_paper.title as title,
                   cited_paper.authors as authors,
                   cited_paper.year as year,
                   cited_paper.journal as journal,
                   cited_paper.stub as is_stub,
                   sentence_citation_count,
                   COALESCE(paragraph_citation_count, 0) as paragraph_citation_count,
                   (sentence_citation_count + COALESCE(paragraph_citation_count, 0)) as total_citations
            ORDER BY total_citations DESC
            """
            
            with self.graph_db.driver.session() as session:
                result = session.run(query, source_paper_id=source_paper_id)
                
                cited_papers = []
                for record in result:
                    cited_papers.append({
                        "paper_id": record["paper_id"],
                        "title": record["title"],
                        "authors": record["authors"],
                        "year": record["year"],
                        "journal": record["journal"],
                        "is_stub": record["is_stub"],  # 标记是否为引用存根
                        "sentence_citations": record["sentence_citation_count"],
                        "paragraph_citations": record["paragraph_citation_count"],
                        "total_citations": record["total_citations"]
                    })
                
                logging.info(f"Found {len(cited_papers)} papers cited by {source_paper_id}")
                return cited_papers
                
        except Exception as e:
            logging.error(f"Error getting papers cited by {source_paper_id}: {e}")
            return []
    
    def get_paragraphs_citing_paper(self, target_paper_id: str, count: int = -1) -> List[Dict]:
        """
        获取引用了指定论文的段落列表
        
        Args:
            target_paper_id: 被引用论文的ID
            count: 返回段落数量，-1表示全部
            
        Returns:
            List[Dict]: 引用该论文的段落列表，包含段落内容和上下文信息
        """
        if not self.graph_db:
            logging.error("Graph database not available")
            return []
        
        try:
            query = """
            MATCH (cited_paper:Paper {id: $target_paper_id})
            MATCH (citing_paper:Paper)-[:BELONGS_TO]-(paragraph:Paragraph)-[:CITES]->(cited_paper)
            WITH paragraph, citing_paper, cited_paper
            MATCH (paragraph)-[:CITES]->(cited_paper)
            RETURN paragraph.id as paragraph_id,
                   paragraph.text as text,
                   paragraph.section as section,
                   paragraph.paragraph_index as paragraph_index,
                   paragraph.citation_count as citation_count,
                   paragraph.citation_density as citation_density,
                   citing_paper.id as citing_paper_id,
                   citing_paper.title as citing_paper_title,
                   citing_paper.authors as citing_paper_authors,
                   citing_paper.year as citing_paper_year
            ORDER BY citing_paper.year DESC, paragraph.paragraph_index ASC
            """ + (f"LIMIT {count}" if count > 0 else "")
            
            with self.graph_db.driver.session() as session:
                result = session.run(query, target_paper_id=target_paper_id)
                
                citing_paragraphs = []
                for record in result:
                    citing_paragraphs.append({
                        "paragraph_id": record["paragraph_id"],
                        "text": record["text"],
                        "section": record["section"],
                        "paragraph_index": record["paragraph_index"],
                        "citation_count": record["citation_count"],
                        "citation_density": record["citation_density"],
                        "citing_paper": {
                            "paper_id": record["citing_paper_id"],
                            "title": record["citing_paper_title"],
                            "authors": record["citing_paper_authors"],
                            "year": record["citing_paper_year"]
                        }
                    })
                
                logging.info(f"Found {len(citing_paragraphs)} paragraphs citing {target_paper_id}")
                return citing_paragraphs
                
        except Exception as e:
            logging.error(f"Error getting paragraphs citing {target_paper_id}: {e}")
            return []
    
    def get_sentences_citing_paper(self, target_paper_id: str, count: int = -1) -> List[Dict]:
        """
        获取引用了指定论文的句子列表
        
        Args:
            target_paper_id: 被引用论文的ID
            count: 返回句子数量，-1表示全部
            
        Returns:
            List[Dict]: 引用该论文的句子列表，包含句子内容和引用上下文
        """
        if not self.graph_db:
            logging.error("Graph database not available")
            return []
        
        try:
            query = """
            MATCH (cited_paper:Paper {id: $target_paper_id})
            MATCH (citing_paper:Paper)-[:BELONGS_TO]-(sentence:Sentence)-[:CITES]->(cited_paper)
            OPTIONAL MATCH (sentence)-[:BELONGS_TO]->(paragraph:Paragraph)
            WITH sentence, paragraph, citing_paper, cited_paper
            MATCH (sentence)-[citation:CITES]->(cited_paper)
            RETURN sentence.id as sentence_id,
                   sentence.text as text,
                   sentence.sentence_index as sentence_index,
                   sentence.word_count as word_count,
                   sentence.char_count as char_count,
                   citation.citation_text as citation_text,
                   citation.citation_context as citation_context,
                   citation.confidence as confidence,
                   paragraph.id as paragraph_id,
                   paragraph.section as section,
                   citing_paper.id as citing_paper_id,
                   citing_paper.title as citing_paper_title,
                   citing_paper.authors as citing_paper_authors,
                   citing_paper.year as citing_paper_year
            ORDER BY citing_paper.year DESC, sentence.sentence_index ASC
            """ + (f"LIMIT {count}" if count > 0 else "")
            
            with self.graph_db.driver.session() as session:
                result = session.run(query, target_paper_id=target_paper_id)
                
                citing_sentences = []
                for record in result:
                    citing_sentences.append({
                        "sentence_id": record["sentence_id"],
                        "text": record["text"],
                        "sentence_index": record["sentence_index"],
                        "word_count": record["word_count"],
                        "char_count": record["char_count"],
                        "citation_text": record["citation_text"],
                        "citation_context": record["citation_context"],
                        "confidence": record["confidence"],
                        "paragraph_id": record["paragraph_id"],
                        "section": record["section"],
                        "citing_paper": {
                            "paper_id": record["citing_paper_id"],
                            "title": record["citing_paper_title"],
                            "authors": record["citing_paper_authors"],
                            "year": record["citing_paper_year"]
                        }
                    })
                
                logging.info(f"Found {len(citing_sentences)} sentences citing {target_paper_id}")
                return citing_sentences
                
        except Exception as e:
            logging.error(f"Error getting sentences citing {target_paper_id}: {e}")
            return []
    
    def get_papers_id_by_title(self, title: str, year: Optional[str] = None, 
                              authors: Optional[List[str]] = None, 
                              journal: Optional[str] = None,
                              similarity_threshold: float = 0.6) -> Dict:
        """
        通过论文标题模糊搜索Paper ID，支持多种搜索场景
        
        Args:
            title: 论文标题（必需）
            year: 发表年份（可选，用于筛选）
            authors: 作者列表（可选，用于筛选）
            journal: 期刊名称（可选，用于筛选）
            similarity_threshold: 相似度阈值（0.0-1.0，默认0.6）
            
        Returns:
            Dict: 搜索结果，包含以下情况：
            - 'status': 'single_match' | 'multiple_matches' | 'no_match'
            - 'paper_id': str (仅在single_match时)
            - 'candidates': List[Dict] (在multiple_matches时包含所有候选)
            - 'message': str (描述性信息)
        """
        if not self.graph_db:
            return {
                "status": "error",
                "message": "Graph database not available"
            }
        
        try:
            # 构建查询条件
            query_conditions = []
            query_params = {}
            
            # 基础查询：获取所有论文的元数据（包含stub）
            base_query = """
            MATCH (p:Paper)
            """
            
            # 添加年份筛选
            if year:
                query_conditions.append("p.year = $year")
                query_params["year"] = int(year)
            
            # 添加期刊筛选
            if journal:
                query_conditions.append("toLower(p.journal) CONTAINS toLower($journal)")
                query_params["journal"] = journal
            
            # 构建完整查询
            if query_conditions:
                full_query = base_query + " AND " + " AND ".join(query_conditions)
            else:
                full_query = base_query
            
            full_query += """
            RETURN p.id as paper_id,
                   p.title as title,
                   p.authors as authors,
                   p.year as year,
                   p.journal as journal,
                   p.doi as doi
            ORDER BY p.year DESC
            """
            
            with self.graph_db.driver.session() as session:
                result = session.run(full_query, **query_params)
                all_papers = [dict(record) for record in result]
            
            if not all_papers:
                return {
                    "status": "no_match",
                    "message": f"No papers found with the given criteria"
                }
            
            # 计算标题相似度和子串匹配
            candidates = []
            for paper in all_papers:
                paper_title = paper.get("title", "")
                input_title = title
                
                # 检查子串匹配（高优先级）
                is_substring_match = self._is_title_substring_match(input_title, paper_title)
                
                # 计算传统相似度
                similarity = SequenceMatcher(None, input_title.lower().strip(), paper_title.lower().strip()).ratio()
                
                # 如果满足任一条件，添加到候选列表
                if is_substring_match or similarity >= similarity_threshold:
                    candidate = {
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "authors": paper["authors"],
                        "year": paper["year"],
                        "journal": paper["journal"],
                        "doi": paper["doi"],
                        "similarity_score": round(similarity, 3),
                        "is_substring_match": is_substring_match
                    }
                    
                    # 如果是子串匹配，计算子串匹配置信度
                    if is_substring_match:
                        substring_confidence = self._calculate_substring_match_confidence(input_title, paper_title)
                        candidate["substring_confidence"] = round(substring_confidence, 3)
                        # 子串匹配时，使用更高的综合分数
                        candidate["combined_score"] = max(similarity, substring_confidence)
                    else:
                        candidate["combined_score"] = similarity
                    
                    # 如果有作者筛选条件，计算作者匹配度
                    if authors:
                        author_match_score = self._calculate_author_similarity(authors, paper.get("authors", []))
                        candidate["author_match_score"] = round(author_match_score, 3)
                    
                    candidates.append(candidate)
            
            # 按综合分数排序（优先考虑子串匹配）
            candidates.sort(key=lambda x: (x["is_substring_match"], x["combined_score"]), reverse=True)
            
            # 判断搜索结果
            if not candidates:
                return {
                    "status": "no_match",
                    "message": f"No papers found with title similarity >= {similarity_threshold} or substring match"
                }
            elif len(candidates) == 1:
                match_type = "substring" if candidates[0]["is_substring_match"] else "similarity"
                score = candidates[0].get("substring_confidence", candidates[0]["similarity_score"])
                return {
                    "status": "single_match",
                    "paper_id": candidates[0]["paper_id"],
                    "paper_info": candidates[0],
                    "message": f"Found {match_type} match with score {score}"
                }
            else:
                # 检查是否有明显的最佳匹配
                best_candidate = candidates[0]
                second_best_candidate = candidates[1] if len(candidates) > 1 else None
                
                # 子串匹配优先
                if best_candidate["is_substring_match"]:
                    # 如果最佳匹配是子串匹配，且没有其他子串匹配，认为是确定匹配
                    other_substring_matches = [c for c in candidates[1:] if c["is_substring_match"]]
                    if not other_substring_matches:
                        return {
                            "status": "single_match",
                            "paper_id": best_candidate["paper_id"],
                            "paper_info": best_candidate,
                            "message": f"Found confident substring match with confidence {best_candidate.get('substring_confidence', 0.0)}"
                        }
                
                # 传统相似度判断
                best_score = best_candidate["similarity_score"]
                second_best_score = second_best_candidate["similarity_score"] if second_best_candidate else 0
                
                # 如果最佳匹配的相似度比第二名高出0.2以上，认为是确定匹配
                if best_score - second_best_score >= 0.2 and best_score >= 0.9:
                    return {
                        "status": "single_match",
                        "paper_id": best_candidate["paper_id"],
                        "paper_info": best_candidate,
                        "message": f"Found confident similarity match with score {best_score}"
                    }
                else:
                    return {
                        "status": "multiple_matches",
                        "candidates": candidates,
                        "count": len(candidates),
                        "message": f"Found {len(candidates)} potential matches. Please specify additional criteria or select manually."
                    }
                    
        except Exception as e:
            logging.error(f"Error in fuzzy title search: {e}")
            return {
                "status": "error",
                "message": f"Search failed: {str(e)}"
            }
    
    def get_papers_id_by_author(self, author_name: str, 
                               title_hint: Optional[str] = None,
                               year: Optional[str] = None,
                               case_sensitive: bool = False) -> Dict:
        """
        通过作者姓名Token子集匹配搜索Paper ID
        
        使用Token-based子集匹配：输入的作者名tokens必须是候选作者tokens的子集
        例如: "porter" 能匹配 "Michael E. Porter", "David Porter", 等
        
        Args:
            author_name: 作者姓名或姓名片段
            title_hint: 论文标题提示（可选，用于进一步筛选）
            year: 发表年份（可选）
            case_sensitive: 是否区分大小写（默认false）
            
        Returns:
            Dict: 搜索结果，格式同get_papers_id_by_title
        """
        if not self.graph_db:
            return {
                "status": "error",
                "message": "Graph database not available"
            }
        
        try:
            # 构建查询（包含所有论文，无论是否为stub）
            query_conditions = []
            query_params = {}
            
            if year:
                query_conditions.append("p.year = $year")
                query_params["year"] = int(year)
            
            # 构建完整查询
            if query_conditions:
                query = f"""
                MATCH (p:Paper)
                WHERE {" AND ".join(query_conditions)}
                RETURN p.id as paper_id,
                       p.title as title,
                       p.authors as authors,
                       p.year as year,
                       p.journal as journal,
                       p.doi as doi
                ORDER BY p.year DESC
                """
            else:
                query = """
                MATCH (p:Paper)
                RETURN p.id as paper_id,
                       p.title as title,
                       p.authors as authors,
                       p.year as year,
                       p.journal as journal,
                       p.doi as doi
                ORDER BY p.year DESC
                """
            
            with self.graph_db.driver.session() as session:
                result = session.run(query, **query_params)
                all_papers = [dict(record) for record in result]
            
            if not all_papers:
                return {
                    "status": "no_match",
                    "message": "No papers found with the given criteria"
                }
            
            # 预处理输入的作者名：分解为tokens
            input_tokens = self._tokenize_author_name(author_name, case_sensitive)
            if not input_tokens:
                return {
                    "status": "no_match",
                    "message": "Invalid author name provided"
                }
            
            # 搜索匹配的作者
            candidates = []
            
            for paper in all_papers:
                paper_authors = paper.get("authors", [])
                if not paper_authors:
                    continue
                
                # 检查每个作者是否匹配
                matched_authors = []
                for author in paper_authors:
                    if isinstance(author, str) and author.strip():
                        author_tokens = self._tokenize_author_name(author, case_sensitive)
                        
                        # 检查输入tokens是否为作者tokens的子集
                        if self._is_token_subset(input_tokens, author_tokens):
                            matched_authors.append({
                                "name": author,
                                "match_ratio": len(input_tokens) / len(author_tokens) if author_tokens else 0
                            })
                
                if matched_authors:
                    # 选择匹配度最高的作者
                    best_match = max(matched_authors, key=lambda x: x["match_ratio"])
                    
                    candidate = {
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "authors": paper["authors"],
                        "year": paper["year"],
                        "journal": paper["journal"],
                        "doi": paper["doi"],
                        "matched_author": best_match["name"],
                        "match_ratio": round(best_match["match_ratio"], 3),
                        "all_matched_authors": [m["name"] for m in matched_authors]
                    }
                    
                    # 如果有标题提示，计算标题相似度
                    if title_hint:
                        paper_title = paper.get("title", "").lower().strip()
                        title_similarity = SequenceMatcher(None, title_hint.lower().strip(), paper_title).ratio()
                        candidate["title_similarity_score"] = round(title_similarity, 3)
                    
                    candidates.append(candidate)
            
            # 排序：优先按匹配比例，然后按标题相似度（如果有）
            if title_hint:
                candidates.sort(key=lambda x: (x["match_ratio"], x.get("title_similarity_score", 0)), reverse=True)
            else:
                candidates.sort(key=lambda x: x["match_ratio"], reverse=True)
            
            # 判断搜索结果
            if not candidates:
                return {
                    "status": "no_match",
                    "message": f"No authors found containing tokens: {input_tokens}"
                }
            elif len(candidates) == 1:
                return {
                    "status": "single_match",
                    "paper_id": candidates[0]["paper_id"],
                    "paper_info": candidates[0],
                    "message": f"Found unique match: '{candidates[0]['matched_author']}'"
                }
            else:
                return {
                    "status": "multiple_matches",
                    "candidates": candidates,
                    "count": len(candidates),
                    "message": f"Found {len(candidates)} papers with authors containing '{author_name}'"
                }
                
        except Exception as e:
            logging.error(f"Error in author token search: {e}")
            return {
                "status": "error",
                "message": f"Search failed: {str(e)}"
            }
    
    def _tokenize_author_name(self, author_name: str, case_sensitive: bool = False) -> List[str]:
        """
        将作者姓名分解为tokens
        
        Args:
            author_name: 作者姓名
            case_sensitive: 是否区分大小写
            
        Returns:
            List[str]: 作者姓名的token列表
        """
        if not author_name or not isinstance(author_name, str):
            return []
        
        # 清理字符串：移除多余的标点符号和空格
        cleaned = re.sub(r'[^\w\s.-]', ' ', author_name)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if not case_sensitive:
            cleaned = cleaned.lower()
        
        # 分解为tokens，过滤掉短token (如单字母缩写可能保留)
        tokens = []
        for token in cleaned.split():
            token = token.strip('.-')  # 移除首尾的点和破折号
            if token and len(token) >= 1:  # 保留所有非空token，包括单字母缩写
                tokens.append(token)
        
        return tokens
    
    def _is_token_subset(self, input_tokens: List[str], candidate_tokens: List[str]) -> bool:
        """
        检查输入tokens是否为候选tokens的子集
        
        Args:
            input_tokens: 输入的token列表
            candidate_tokens: 候选的token列表
            
        Returns:
            bool: 如果输入tokens是候选tokens的子集，返回True
        """
        if not input_tokens:
            return False
        
        if not candidate_tokens:
            return False
        
        # 将所有tokens转换为小写进行比较
        input_set = {token.lower() for token in input_tokens}
        candidate_set = {token.lower() for token in candidate_tokens}
        
        # 检查是否为子集
        return input_set.issubset(candidate_set)
    
    def _calculate_author_similarity(self, input_authors: List[str], paper_authors: List[str]) -> float:
        """
        计算输入作者列表与论文作者列表的相似度
        
        Args:
            input_authors: 输入的作者列表
            paper_authors: 论文的作者列表
            
        Returns:
            float: 相似度分数（0.0-1.0）
        """
        if not input_authors or not paper_authors:
            return 0.0
        
        total_similarity = 0
        matched_count = 0
        
        for input_author in input_authors:
            input_clean = input_author.lower().strip()
            max_similarity = 0
            
            for paper_author in paper_authors:
                if isinstance(paper_author, str):
                    paper_clean = paper_author.lower().strip()
                    similarity = SequenceMatcher(None, input_clean, paper_clean).ratio()
                    max_similarity = max(max_similarity, similarity)
            
            if max_similarity > 0.7:  # 只计算相似度较高的匹配
                total_similarity += max_similarity
                matched_count += 1
        
        if matched_count == 0:
            return 0.0
        
        # 返回平均相似度，但对匹配数量进行权重调整
        average_similarity = total_similarity / matched_count
        coverage_bonus = min(matched_count / len(input_authors), 1.0) * 0.2
        
        return min(average_similarity + coverage_bonus, 1.0)

    def _is_title_substring_match(self, input_title: str, paper_title: str, min_length: int = 5) -> bool:
        """
        检查输入标题是否为论文标题的子串匹配
        
        Args:
            input_title: 输入的标题
            paper_title: 论文的完整标题
            min_length: 最小匹配长度阈值
            
        Returns:
            bool: 如果输入标题是论文标题的子串且长度满足要求，返回True
        """
        if not input_title or not paper_title:
            return False
        
        # 标准化处理：转换为小写并清理
        input_clean = input_title.lower().strip()
        paper_clean = paper_title.lower().strip()
        
        # 检查长度阈值
        if len(input_clean) < min_length:
            return False
        
        # 检查是否为子串
        return input_clean in paper_clean

    def _calculate_substring_match_confidence(self, input_title: str, paper_title: str) -> float:
        """
        计算子串匹配的置信度分数
        
        Args:
            input_title: 输入的标题
            paper_title: 论文的完整标题
            
        Returns:
            float: 置信度分数（0.0-1.0）
        """
        if not input_title or not paper_title:
            return 0.0
        
        input_clean = input_title.lower().strip()
        paper_clean = paper_title.lower().strip()
        
        # 基础分数：子串长度占完整标题的比例
        length_ratio = len(input_clean) / len(paper_clean)
        
        # 位置奖励：如果子串在标题开头，给予额外奖励
        position_bonus = 0.0
        if paper_clean.startswith(input_clean):
            position_bonus = 0.1
        
        # 长度奖励：较长的子串获得更高分数
        length_bonus = min(length_ratio * 0.2, 0.2)
        
        # 计算最终置信度
        confidence = length_ratio + position_bonus + length_bonus
        
        return min(confidence, 1.0)

    def search_relevant_sentences(self, text: str, top_n: int = 50, 
                                 paper_ids: Optional[List[str]] = None,
                                 min_score: float = 0.0,
                                 years: Optional[List[int]] = None,
                                 journals: Optional[List[str]] = None,
                                 sentence_types: Optional[List[str]] = None) -> Dict:
        """
        在向量数据库中搜索与给定文本最相关的句子
        
        Args:
            text: 查询文本
            top_n: 返回结果数量 (默认50)
            paper_ids: 限制搜索的论文ID列表 (可选)
            min_score: 最小相似度分数阈值 (0.0-1.0)
            years: 限制搜索的年份列表 (可选)
            journals: 限制搜索的期刊列表 (可选)
            sentence_types: 限制搜索的句子类型 (可选)
            
        Returns:
            Dict: 搜索结果包含匹配的句子及其元数据
        """
        if not self.vector_indexer:
            return {
                "status": "error",
                "message": "Vector database not available"
            }
        
        try:
            # 构建过滤条件
            filter_conditions = {}
            
            if paper_ids:
                filter_conditions["paper_id"] = {"$in": paper_ids}
            
            if years:
                filter_conditions["year"] = {"$in": years}
            
            if journals:
                # 使用模糊匹配处理期刊名称
                filter_conditions["journal"] = {"$in": journals}
            
            if sentence_types:
                filter_conditions["sentence_type"] = {"$in": sentence_types}
            
            # 执行向量搜索
            search_results = self.vector_indexer.search(
                query=text,
                collection_name="sentences",
                limit=min(top_n * 2, 200)  # 搜索更多结果以便过滤
            )
            
            if not search_results:
                return {
                    "status": "no_match",
                    "message": f"No relevant sentences found for query: '{text[:50]}...'",
                    "query": text,
                    "total_results": 0
                }
            
            # 过滤结果 (如果有过滤条件)
            filtered_results = search_results
            
            if filter_conditions:
                filtered_results = []
                for result in search_results:
                    # 应用各种过滤条件
                    if paper_ids and result.get("paper_id") not in paper_ids:
                        continue
                    if years and result.get("year") not in [str(y) for y in years] and result.get("year") not in years:
                        continue
                    if journals and not any(j.lower() in str(result.get("journal", "")).lower() for j in journals):
                        continue
                    if sentence_types and result.get("sentence_type") not in sentence_types:
                        continue
                    if result.get("score", 0.0) < min_score:
                        continue
                    filtered_results.append(result)
            
            # 处理搜索结果
            processed_results = []
            for result in filtered_results[:top_n]:
                processed_result = {
                    "id": result.get("paper_id", "") + f"_sent_{result.get('sentence_index', 0)}",
                    "text": result.get("text", ""),
                    "score": round(result.get("score", 0.0), 4),
                    "paper_info": {
                        "paper_id": result.get("paper_id", ""),
                        "title": result.get("title", "Unknown"),
                        "authors": result.get("authors", []),
                        "year": result.get("year", "Unknown"),
                        "journal": result.get("journal", "Unknown"),
                        "doi": result.get("doi", "Unknown")
                    },
                    "sentence_metadata": {
                        "sentence_type": result.get("sentence_type", ""),
                        "collection": result.get("collection", "sentences")
                    }
                }
                processed_results.append(processed_result)
            
            return {
                "status": "success",
                "message": f"Found {len(processed_results)} relevant sentences",
                "query": text,
                "total_results": len(processed_results),
                "max_score": processed_results[0]["score"] if processed_results else 0.0,
                "min_score": processed_results[-1]["score"] if processed_results else 0.0,
                "results": processed_results
            }
            
        except Exception as e:
            logging.error(f"Error in sentence search: {e}")
            return {
                "status": "error",
                "message": f"Search failed: {str(e)}",
                "query": text
            }
    
    def search_relevant_paragraphs(self, text: str, top_n: int = 50,
                                  paper_ids: Optional[List[str]] = None,
                                  min_score: float = 0.0,
                                  years: Optional[List[int]] = None,
                                  journals: Optional[List[str]] = None,
                                  sections: Optional[List[str]] = None,
                                  has_citations: Optional[bool] = None) -> Dict:
        """
        在向量数据库中搜索与给定文本最相关的段落
        
        Args:
            text: 查询文本
            top_n: 返回结果数量 (默认50)
            paper_ids: 限制搜索的论文ID列表 (可选)
            min_score: 最小相似度分数阈值 (0.0-1.0)
            years: 限制搜索的年份列表 (可选)
            journals: 限制搜索的期刊列表 (可选)
            sections: 限制搜索的章节列表 (可选，如"introduction", "methodology")
            has_citations: 是否包含引用 (可选)
            
        Returns:
            Dict: 搜索结果包含匹配的段落及其元数据
        """
        if not self.vector_indexer:
            return {
                "status": "error", 
                "message": "Vector database not available"
            }
        
        try:
            # 构建过滤条件
            filter_conditions = {}
            
            if paper_ids:
                filter_conditions["paper_id"] = {"$in": paper_ids}
            
            if years:
                filter_conditions["year"] = {"$in": years}
            
            if journals:
                filter_conditions["journal"] = {"$in": journals}
            
            if sections:
                filter_conditions["section"] = {"$in": sections}
            
            if has_citations is not None:
                filter_conditions["has_citations"] = has_citations
            
            # 执行向量搜索
            search_results = self.vector_indexer.search(
                query=text,
                collection_name="paragraphs",
                limit=min(top_n * 2, 200)
            )
            
            if not search_results:
                return {
                    "status": "no_match",
                    "message": f"No relevant paragraphs found for query: '{text[:50]}...'",
                    "query": text,
                    "total_results": 0
                }
            
            # 过滤结果 (如果有过滤条件)
            filtered_results = search_results
            
            if filter_conditions:
                filtered_results = []
                for result in search_results:
                    # 应用各种过滤条件
                    if paper_ids and result.get("paper_id") not in paper_ids:
                        continue
                    if years and result.get("year") not in [str(y) for y in years] and result.get("year") not in years:
                        continue
                    if journals and not any(j.lower() in str(result.get("journal", "")).lower() for j in journals):
                        continue
                    if sections and result.get("section") not in sections:
                        continue
                    if has_citations is not None and result.get("has_citations") != has_citations:
                        continue
                    if result.get("score", 0.0) < min_score:
                        continue
                    filtered_results.append(result)
            
            # 处理搜索结果
            processed_results = []
            for result in filtered_results[:top_n]:
                text_content = result.get("text", "")
                processed_result = {
                    "id": result.get("paper_id", "") + f"_para_{result.get('paragraph_index', 0)}",
                    "text": text_content[:500] + "..." if len(text_content) > 500 else text_content,
                    "full_text": text_content,
                    "score": round(result.get("score", 0.0), 4),
                    "paper_info": {
                        "paper_id": result.get("paper_id", ""),
                        "title": result.get("title", "Unknown"),
                        "authors": result.get("authors", []),
                        "year": result.get("year", "Unknown"),
                        "journal": result.get("journal", "Unknown"),
                        "doi": result.get("doi", "Unknown")
                    },
                    "paragraph_metadata": {
                        "section": result.get("section", ""),
                        "paragraph_index": result.get("paragraph_index", 0),
                        "citation_count": result.get("citation_count", 0),
                        "sentence_count": result.get("sentence_count", 0),
                        "has_citations": result.get("has_citations", False),
                        "collection": result.get("collection", "paragraphs")
                    }
                }
                processed_results.append(processed_result)
            
            return {
                "status": "success",
                "message": f"Found {len(processed_results)} relevant paragraphs",
                "query": text,
                "total_results": len(processed_results),
                "max_score": processed_results[0]["score"] if processed_results else 0.0,
                "min_score": processed_results[-1]["score"] if processed_results else 0.0,
                "results": processed_results
            }
            
        except Exception as e:
            logging.error(f"Error in paragraph search: {e}")
            return {
                "status": "error",
                "message": f"Search failed: {str(e)}",
                "query": text
            }
    
    def search_relevant_sections(self, text: str, top_n: int = 50,
                                paper_ids: Optional[List[str]] = None,
                                min_score: float = 0.0,
                                years: Optional[List[int]] = None,
                                journals: Optional[List[str]] = None,
                                section_types: Optional[List[str]] = None) -> Dict:
        """
        在向量数据库中搜索与给定文本最相关的章节
        
        Args:
            text: 查询文本
            top_n: 返回结果数量 (默认50)
            paper_ids: 限制搜索的论文ID列表 (可选)
            min_score: 最小相似度分数阈值 (0.0-1.0)
            years: 限制搜索的年份列表 (可选)
            journals: 限制搜索的期刊列表 (可选)
            section_types: 限制搜索的章节类型 (可选，如"introduction", "methodology", "conclusion")
            
        Returns:
            Dict: 搜索结果包含匹配的章节及其元数据
        """
        if not self.vector_indexer:
            return {
                "status": "error",
                "message": "Vector database not available"
            }
        
        try:
            # 构建过滤条件
            filter_conditions = {}
            
            if paper_ids:
                filter_conditions["paper_id"] = {"$in": paper_ids}
            
            if years:
                filter_conditions["year"] = {"$in": years}
            
            if journals:
                filter_conditions["journal"] = {"$in": journals}
            
            if section_types:
                filter_conditions["type"] = {"$in": section_types}
            
            # 执行向量搜索
            search_results = self.vector_indexer.search(
                query=text,
                collection_name="sections",
                limit=min(top_n * 2, 200)
            )
            
            if not search_results:
                return {
                    "status": "no_match",
                    "message": f"No relevant sections found for query: '{text[:50]}...'",
                    "query": text,
                    "total_results": 0
                }
            
            # 过滤结果 (如果有过滤条件)
            filtered_results = search_results
            
            if filter_conditions:
                filtered_results = []
                for result in search_results:
                    # 应用各种过滤条件
                    if paper_ids and result.get("paper_id") not in paper_ids:
                        continue
                    if years and result.get("year") not in [str(y) for y in years] and result.get("year") not in years:
                        continue
                    if journals and not any(j.lower() in str(result.get("journal", "")).lower() for j in journals):
                        continue
                    if section_types and result.get("section_type") not in section_types:
                        continue
                    if result.get("score", 0.0) < min_score:
                        continue
                    filtered_results.append(result)
            
            # 处理搜索结果
            processed_results = []
            for result in filtered_results[:top_n]:
                text_content = result.get("text", "")
                processed_result = {
                    "id": result.get("paper_id", "") + f"_sect_{result.get('section_index', 0)}",
                    "title": result.get("section_title", ""),
                    "text": text_content[:800] + "..." if len(text_content) > 800 else text_content,
                    "full_text": text_content,
                    "score": round(result.get("score", 0.0), 4),
                    "paper_info": {
                        "paper_id": result.get("paper_id", ""),
                        "title": result.get("title", "Unknown"),
                        "authors": result.get("authors", []),
                        "year": result.get("year", "Unknown"),
                        "journal": result.get("journal", "Unknown"),
                        "doi": result.get("doi", "Unknown")
                    },
                    "section_metadata": {
                        "section_type": result.get("section_type", ""),
                        "paragraph_count": result.get("paragraph_count", 0),
                        "collection": result.get("collection", "sections")
                    }
                }
                processed_results.append(processed_result)
            
            return {
                "status": "success",
                "message": f"Found {len(processed_results)} relevant sections",
                "query": text,
                "total_results": len(processed_results),
                "max_score": processed_results[0]["score"] if processed_results else 0.0,
                "min_score": processed_results[-1]["score"] if processed_results else 0.0,
                "results": processed_results
            }
            
        except Exception as e:
            logging.error(f"Error in section search: {e}")
            return {
                "status": "error",
                "message": f"Search failed: {str(e)}",
                "query": text
            }
    
    def search_all_content_types(self, text: str, top_n_per_type: int = 20,
                                min_score: float = 0.0,
                                paper_ids: Optional[List[str]] = None,
                                years: Optional[List[int]] = None) -> Dict:
        """
        跨所有内容类型（句子、段落、章节）进行综合搜索
        
        Args:
            text: 查询文本
            top_n_per_type: 每种类型返回的结果数量
            min_score: 最小相似度分数阈值
            paper_ids: 限制搜索的论文ID列表 (可选)
            years: 限制搜索的年份列表 (可选)
            
        Returns:
            Dict: 包含所有类型搜索结果的综合报告
        """
        try:
            # 并行搜索所有类型
            sentences_result = self.search_relevant_sentences(
                text, top_n_per_type, paper_ids, min_score, years
            )
            
            paragraphs_result = self.search_relevant_paragraphs(
                text, top_n_per_type, paper_ids, min_score, years
            )
            
            sections_result = self.search_relevant_sections(
                text, top_n_per_type, paper_ids, min_score, years
            )
            
            # 统计总结果
            total_results = (
                sentences_result.get("total_results", 0) +
                paragraphs_result.get("total_results", 0) +
                sections_result.get("total_results", 0)
            )
            
            # 找出最高分数
            max_scores = [
                sentences_result.get("max_score", 0.0),
                paragraphs_result.get("max_score", 0.0),
                sections_result.get("max_score", 0.0)
            ]
            overall_max_score = max(max_scores) if max_scores else 0.0
            
            return {
                "status": "success",
                "message": f"Comprehensive search completed: {total_results} total results",
                "query": text,
                "overall_stats": {
                    "total_results": total_results,
                    "max_score": overall_max_score,
                    "sentences_count": sentences_result.get("total_results", 0),
                    "paragraphs_count": paragraphs_result.get("total_results", 0),
                    "sections_count": sections_result.get("total_results", 0)
                },
                "sentences": sentences_result,
                "paragraphs": paragraphs_result,
                "sections": sections_result
            }
            
        except Exception as e:
            logging.error(f"Error in comprehensive search: {e}")
            return {
                "status": "error",
                "message": f"Comprehensive search failed: {str(e)}",
                "query": text
            }

    def query_pdf_content(self, paper_id: str, query: str, context_window: int = 500) -> Dict[str, Any]:
        """Query PDF content directly using the stored processed document
        
        Args:
            paper_id: The paper ID to query
            query: The question or search query
            context_window: Number of characters around matching text to include
            
        Returns:
            Dict containing relevant content from the PDF
        """
        try:
            paper_path = os.path.join(self.papers_dir, paper_id)
            processed_doc_path = os.path.join(paper_path, "processed_document.json")
            
            if not os.path.exists(processed_doc_path):
                return {
                    "found": False,
                    "error": f"Processed document not found for paper {paper_id}",
                    "data": []
                }
            
            # Load the processed document
            with open(processed_doc_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Search through sections
            relevant_content = []
            query_lower = query.lower()
            
            for section in doc_data.get("sections", []):
                section_text = section.get("section_text", "")
                section_title = section.get("section_title", "")
                
                # Simple keyword matching (can be enhanced with semantic search)
                if query_lower in section_text.lower() or query_lower in section_title.lower():
                    # Find specific matches within the section
                    text_lower = section_text.lower()
                    matches = []
                    start_pos = 0
                    
                    while True:
                        pos = text_lower.find(query_lower, start_pos)
                        if pos == -1:
                            break
                        
                        # Extract context around the match
                        context_start = max(0, pos - context_window//2)
                        context_end = min(len(section_text), pos + len(query) + context_window//2)
                        context = section_text[context_start:context_end]
                        
                        matches.append({
                            "position": pos,
                            "context": context.strip(),
                            "highlight_start": pos - context_start,
                            "highlight_end": pos - context_start + len(query)
                        })
                        
                        start_pos = pos + 1
                    
                    if matches:
                        relevant_content.append({
                            "section_index": section.get("section_index"),
                            "section_title": section_title,
                            "section_type": section.get("section_type"),
                            "matches": matches,
                            "word_count": section.get("word_count", 0),
                            "citations": section.get("citations", [])
                        })
            
            return {
                "found": len(relevant_content) > 0,
                "paper_id": paper_id,
                "metadata": doc_data.get("metadata", {}),
                "query": query,
                "total_matches": sum(len(section["matches"]) for section in relevant_content),
                "data": relevant_content
            }
            
        except Exception as e:
            return {
                "found": False,
                "error": f"Error querying PDF content: {str(e)}",
                "data": []
            }
    
    def query_pdf_by_title_and_content(self, title_query: str, content_query: str) -> Dict[str, Any]:
        """Find papers by title and then query their content
        
        Args:
            title_query: Query to find papers by title
            content_query: Query to search within the found papers
            
        Returns:
            Dict containing papers and their relevant content
        """
        try:
            # First find papers by title
            title_results = self.get_papers_id_by_title(title_query)
            
            if not title_results.get("found") or not title_results.get("data"):
                return {
                    "found": False,
                    "error": "No papers found matching the title query",
                    "title_query": title_query,
                    "content_query": content_query,
                    "data": []
                }
            
            # Query content in each found paper
            content_results = []
            for paper in title_results["data"]:
                paper_id = paper.get("paper_id")
                if paper_id:
                    content_result = self.query_pdf_content(paper_id, content_query)
                    if content_result.get("found"):
                        content_results.append({
                            "paper_metadata": paper,
                            "content_matches": content_result
                        })
            
            return {
                "found": len(content_results) > 0,
                "title_query": title_query,
                "content_query": content_query,
                "papers_found": len(title_results["data"]),
                "papers_with_content": len(content_results),
                "data": content_results
            }
            
        except Exception as e:
            return {
                "found": False,
                "error": f"Error in title and content query: {str(e)}",
                "data": []
            }
    
    def query_pdf_by_author_and_content(self, author_name: str, content_query: str) -> Dict[str, Any]:
        """Find papers by author and then query their content
        
        Args:
            author_name: Author name to search for
            content_query: Query to search within the found papers
            
        Returns:
            Dict containing papers and their relevant content
        """
        try:
            # First find papers by author
            author_results = self.get_papers_id_by_author(author_name)
            
            if not author_results.get("found") or not author_results.get("data"):
                return {
                    "found": False,
                    "error": f"No papers found for author: {author_name}",
                    "author_name": author_name,
                    "content_query": content_query,
                    "data": []
                }
            
            # Query content in each found paper
            content_results = []
            for paper in author_results["data"]:
                paper_id = paper.get("paper_id")
                if paper_id:
                    content_result = self.query_pdf_content(paper_id, content_query)
                    if content_result.get("found"):
                        content_results.append({
                            "paper_metadata": paper,
                            "content_matches": content_result
                        })
            
            return {
                "found": len(content_results) > 0,
                "author_name": author_name,
                "content_query": content_query,
                "papers_found": len(author_results["data"]),
                "papers_with_content": len(content_results),
                "data": content_results
            }
            
        except Exception as e:
            return {
                "found": False,
                "error": f"Error in author and content query: {str(e)}",
                "data": []
            }
    
    def get_full_pdf_content(self, paper_id: str) -> Dict[str, Any]:
        """Get the complete content of a PDF paper
        
        Args:
            paper_id: The paper ID
            
        Returns:
            Dict containing the full paper content
        """
        try:
            paper_path = os.path.join(self.papers_dir, paper_id)
            processed_doc_path = os.path.join(paper_path, "processed_document.json")
            
            if not os.path.exists(processed_doc_path):
                return {
                    "found": False,
                    "error": f"Processed document not found for paper {paper_id}",
                    "data": {}
                }
            
            # Load the complete processed document
            with open(processed_doc_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Extract full text from all sections
            full_text_parts = []
            section_summaries = []
            
            for section in doc_data.get("sections", []):
                section_text = section.get("section_text", "")
                section_title = section.get("section_title", "")
                
                if section_text:
                    full_text_parts.append(f"## {section_title}\n\n{section_text}")
                    
                    section_summaries.append({
                        "section_index": section.get("section_index"),
                        "section_title": section_title,
                        "section_type": section.get("section_type"),
                        "word_count": section.get("word_count", 0),
                        "citations_count": len(section.get("citations", [])),
                        "preview": section_text[:200] + "..." if len(section_text) > 200 else section_text
                    })
            
            full_text = "\n\n".join(full_text_parts)
            
            return {
                "found": True,
                "paper_id": paper_id,
                "metadata": doc_data.get("metadata", {}),
                "sections_count": len(doc_data.get("sections", [])),
                "total_word_count": sum(section.get("word_count", 0) for section in doc_data.get("sections", [])),
                "section_summaries": section_summaries,
                "full_text": full_text,
                "data": doc_data
            }
            
        except Exception as e:
            return {
                "found": False,
                "error": f"Error retrieving full PDF content: {str(e)}",
                "data": {}
            }
    
    def semantic_search_pdf_content(self, paper_id: str, query: str, similarity_threshold: float = 0.5) -> Dict[str, Any]:
        """Perform semantic search within a specific PDF using sentence transformers
        
        Args:
            paper_id: The paper ID to search within
            query: The search query
            similarity_threshold: Minimum similarity score for results
            
        Returns:
            Dict containing semantically similar content
        """
        try:
            # Get the full PDF content first
            pdf_result = self.get_full_pdf_content(paper_id)
            
            if not pdf_result.get("found"):
                return pdf_result
            
            # Try to use sentence transformers for semantic search
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Initialize the model (use a lightweight model)
                model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Split content into chunks for semantic search
                sections = pdf_result["data"].get("sections", [])
                chunks = []
                chunk_metadata = []
                
                for section in sections:
                    section_text = section.get("section_text", "")
                    section_title = section.get("section_title", "")
                    
                    # Split long sections into smaller chunks
                    if len(section_text) > 1000:
                        # Split by paragraphs or sentences
                        paragraphs = section_text.split('\n\n')
                        for i, paragraph in enumerate(paragraphs):
                            if len(paragraph.strip()) > 50:  # Skip very short paragraphs
                                chunks.append(paragraph.strip())
                                chunk_metadata.append({
                                    "section_index": section.get("section_index"),
                                    "section_title": section_title,
                                    "paragraph_index": i,
                                    "chunk_type": "paragraph"
                                })
                    else:
                        chunks.append(section_text)
                        chunk_metadata.append({
                            "section_index": section.get("section_index"),
                            "section_title": section_title,
                            "paragraph_index": 0,
                            "chunk_type": "full_section"
                        })
                
                if not chunks:
                    return {
                        "found": False,
                        "error": "No content chunks found for semantic search",
                        "data": []
                    }
                
                # Encode query and chunks
                query_embedding = model.encode([query])
                chunk_embeddings = model.encode(chunks)
                
                # Calculate similarities
                similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
                
                # Filter by threshold and sort by similarity
                relevant_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
                relevant_indices.sort(key=lambda i: similarities[i], reverse=True)
                
                # Prepare results
                semantic_results = []
                for idx in relevant_indices[:10]:  # Top 10 results
                    semantic_results.append({
                        "content": chunks[idx],
                        "similarity_score": float(similarities[idx]),
                        "metadata": chunk_metadata[idx]
                    })
                
                return {
                    "found": len(semantic_results) > 0,
                    "paper_id": paper_id,
                    "query": query,
                    "similarity_threshold": similarity_threshold,
                    "total_chunks_searched": len(chunks),
                    "relevant_chunks_found": len(semantic_results),
                    "data": semantic_results
                }
                
            except ImportError:
                # Fallback to keyword search if sentence transformers not available
                return self.query_pdf_content(paper_id, query)
                
        except Exception as e:
            return {
                "found": False,
                "error": f"Error in semantic search: {str(e)}",
                "data": []
            }

    def close(self):
        """Safely close Neo4j driver"""
        if self.graph_db:
            try:
                self.graph_db.close()
            except Exception as e:
                logging.debug(f"Error closing Neo4j driver: {e}")
            finally:
                self.graph_db = None


# 测试用例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 初始化Query Agent
    agent = QueryDBAgent()
    
    print("=== Query DB Agent 完整测试 ===\n")
    
    # ==================== 第一部分：Fuzzy Search 测试 ====================
    print("📚 第一部分：Fuzzy Search 模糊搜索测试")
    print("=" * 60)
    
    # 测试1: 按标题搜索（完全匹配）
    print("\n1. 测试标题精确搜索:")
    title_result = agent.get_papers_id_by_title("Technology search strategies and competition due to import penetration")
    print(f"   搜索结果: {title_result['status']}")
    print(f"   消息: {title_result['message']}")
    if title_result['status'] == 'single_match':
        print(f"   论文ID: {title_result['paper_id']}")
        print(f"   论文信息: {title_result['paper_info']['title']} ({title_result['paper_info']['year']})")
    elif title_result['status'] == 'multiple_matches':
        print(f"   找到 {title_result['count']} 个候选:")
        for i, candidate in enumerate(title_result['candidates'][:3]):
            print(f"     {i+1}. {candidate['title']} ({candidate['year']}) - 相似度: {candidate['similarity_score']}")
    
    # # 测试2: 按标题搜索（部分匹配）
    # print("\n2. 测试标题模糊搜索:")
    # fuzzy_title_result = agent.get_papers_id_by_title("imitation complex")
    # print(f"   搜索结果: {fuzzy_title_result['status']}")
    # print(f"   消息: {fuzzy_title_result['message']}")
    # if fuzzy_title_result['status'] == 'multiple_matches':
    #     print(f"   找到 {fuzzy_title_result['count']} 个候选:")
    #     for i, candidate in enumerate(fuzzy_title_result['candidates'][:3]):
    #         print(f"     {i+1}. {candidate['title']} ({candidate['year']}) - 相似度: {candidate['similarity_score']}")
    # else:
    #     print(f"   搜索结果: {fuzzy_title_result['status']}")
    #     print(f"   消息: {fuzzy_title_result['message']}")
    
    # # 测试3: 按作者搜索 (Token子集匹配)
    # print("\n3. 测试作者Token搜索:")
    # author_result = agent.get_papers_id_by_author("Porter")
    # print(f"   搜索结果: {author_result['status']}")
    # print(f"   消息: {author_result['message']}")
    # if author_result['status'] == 'multiple_matches':
    #     print(f"   找到 {author_result['count']} 个候选:")
    #     for i, candidate in enumerate(author_result['candidates'][:3]):
    #         print(f"     {i+1}. {candidate['title']} - 匹配作者: {candidate['matched_author']} - 匹配率: {candidate['match_ratio']}")
    #         if len(candidate['all_matched_authors']) > 1:
    #             print(f"         所有匹配作者: {candidate['all_matched_authors']}")
    # else:
    #     print(f"   搜索结果: {author_result['status']}")
    #     print(f"   消息: {author_result['message']}")
    
    # # 测试4: 部分名字搜索
    # print("\n4. 测试部分作者名搜索:")
    # partial_result = agent.get_papers_id_by_author("michael")
    # print(f"   搜索结果: {partial_result['status']}")
    # print(f"   消息: {partial_result['message']}")
    # if partial_result['status'] == 'multiple_matches':
    #     print(f"   找到 {partial_result['count']} 个候选:")
    #     for i, candidate in enumerate(partial_result['candidates'][:3]):
    #         print(f"     {i+1}. {candidate['title']} - 匹配作者: {candidate['matched_author']} - 匹配率: {candidate['match_ratio']}")
    # else:
    #     print(f"   搜索结果: {partial_result['status']}")
    #     print(f"   消息: {partial_result['message']}")
    
    # # 测试5: 组合搜索（作者+标题提示）
    # print("\n5. 测试组合搜索 (作者+标题提示):")
    # combo_result = agent.get_papers_id_by_author("Porter", title_hint="competitive", year="1980")
    # print(f"   搜索结果: {combo_result['status']}")
    # print(f"   消息: {combo_result['message']}")
    # if combo_result['status'] == 'single_match':
    #     print(f"   论文ID: {combo_result['paper_id']}")
    #     info = combo_result['paper_info']
    #     print(f"   论文: {info['title']} ({info['year']})")
    #     print(f"   匹配作者: {info['matched_author']} (匹配率: {info['match_ratio']})")
    #     if info.get('title_similarity_score'):
    #         print(f"   标题相似度: {info['title_similarity_score']}")
    # else:
    #     print(f"   搜索结果: {combo_result['status']}")
    #     print(f"   消息: {combo_result['message']}")
    
    # # 测试6: 测试Token化功能
    # print("\n6. 测试Token化功能:")
    # test_names = ["Michael E. Porter", "J.R.R. Tolkien", "Mary O'Connor", "Van Der Berg"]
    # for name in test_names:
    #     tokens = agent._tokenize_author_name(name)
    #     print(f"   '{name}' -> tokens: {tokens}")
    
    # # 测试7: 测试子集匹配
    # print("\n7. 测试子集匹配:")
    # test_cases = [
    #     (["porter"], ["michael", "e", "porter"]),
    #     (["michael", "porter"], ["michael", "e", "porter"]),
    #     (["tolkien"], ["j", "r", "r", "tolkien"]),
    #     (["john"], ["michael", "e", "porter"])
    # ]
    # for input_tokens, candidate_tokens in test_cases:
    #     is_subset = agent._is_token_subset(input_tokens, candidate_tokens)
    #     print(f"   {input_tokens} ⊆ {candidate_tokens} = {is_subset}")
    
    # # ==================== 第二部分：使用找到的Paper ID进行详细查询 ====================
    # print("\n\n📖 第二部分：使用找到的Paper ID进行详细查询")
    # print("=" * 60)
    
    # # 获取一个实际的paper ID用于测试
    # test_paper_id = "babcd89569ffe6cb373ed21a762c1799ace907d68f5cffa189e2d6be77af0504"
    # # if title_result['status'] == 'single_match':
    # #     test_paper_id = title_result['paper_id']
    # # elif combo_result['status'] == 'single_match':
    # #     test_paper_id = combo_result['paper_id']
    # # elif title_result['status'] == 'multiple_matches' and title_result['candidates']:
    # #     test_paper_id = title_result['candidates'][0]['paper_id']
    
    # if test_paper_id:
    #     print(f"\n使用Paper ID进行测试: {test_paper_id}")
        
    #     # 测试5: 获取论文元数据
    #     print("\n5. 测试获取论文元数据:")
    #     metadata = agent.get_metadata_by_paper_id(test_paper_id)
    #     if metadata:
    #         print(f"   标题: {metadata.get('title', 'N/A')}")
    #         print(f"   作者: {metadata.get('authors', 'N/A')}")
    #         print(f"   年份: {metadata.get('year', 'N/A')}")
    #         print(f"   期刊: {metadata.get('journal', 'N/A')}")
    #         print(f"   是否为存根: {metadata.get('is_stub', 'N/A')}")
        
    #     # 测试6: 获取引用该论文的所有论文
    #     print("\n6. 测试获取引用该论文的论文:")
    #     citing_papers = agent.get_papers_citing_paper(test_paper_id)
    #     print(f"   找到 {len(citing_papers)} 篇论文引用了该论文")
    #     for paper in citing_papers[:2]:  # 显示前2个
    #         print(f"   - {paper['title']} ({paper['year']}): {paper['total_citations']} 引用")
        
    #     # 测试7: 获取该论文引用的所有论文
    #     print("\n7. 测试获取该论文引用的论文:")
    #     cited_papers = agent.get_papers_cited_by_paper(test_paper_id)
    #     print(f"   该论文引用了 {len(cited_papers)} 篇论文")
    #     for paper in cited_papers[:2]:  # 显示前2个
    #         print(f"   - {paper['title']} ({paper['year']}): {paper['total_citations']} 次引用")
        
    #     # 测试8: 获取引用该论文的句子（限制5个）
    #     print("\n8. 测试获取引用句子:")
    #     citing_sentences = agent.get_sentences_citing_paper(test_paper_id, count=5)
    #     print(f"   找到 {len(citing_sentences)} 个引用句子")
    #     for sent in citing_sentences[:2]:  # 显示前2个
    #         print(f"   - 来自 {sent['citing_paper']['title']}: {sent['text'][:60]}...")
    #         if sent['citation_text']:
    #             print(f"     引用文本: {sent['citation_text']}")
    # else:
    #     print("\n⚠️  没有找到可用的Paper ID进行详细查询测试")
    
    # print("\n🏁 完整测试完成!")
    
    # # ==================== 第三部分：向量搜索测试 ====================
    # print("\n\n🔍 第三部分：向量数据库语义搜索测试")
    # print("=" * 60)
    
    # # 测试9: 句子级语义搜索
    # print("\n9. 测试句子语义搜索:")
    # sentence_search = agent.search_relevant_sentences(
    #     "competitive strategy and market positioning", 
    #     top_n=5, 
    #     min_score=0.3
    # )
    # print(f"   搜索结果: {sentence_search['status']}")
    # print(f"   消息: {sentence_search['message']}")
    # if sentence_search['status'] == 'success':
    #     print(f"   最高分数: {sentence_search['max_score']}")
    #     print(f"   结果预览:")
    #     for i, result in enumerate(sentence_search['results'][:2]):
    #         print(f"     {i+1}. 分数: {result['score']}")
    #         print(f"        文本: {result['text'][:80]}...")
    #         print(f"        来源: {result['paper_info']['title']}")
    
    # # 测试10: 段落级语义搜索 
    # print("\n10. 测试段落语义搜索:")
    # paragraph_search = agent.search_relevant_paragraphs(
    #     "innovation and organizational change",
    #     top_n=3,
    #     has_citations=True,  # 只搜索包含引用的段落
    #     min_score=0.2
    # )
    # print(f"   搜索结果: {paragraph_search['status']}")
    # print(f"   消息: {paragraph_search['message']}")
    # if paragraph_search['status'] == 'success':
    #     print(f"   结果预览:")
    #     for i, result in enumerate(paragraph_search['results'][:1]):
    #         print(f"     {i+1}. 分数: {result['score']}")
    #         print(f"        段落: {result['text']}")
    #         print(f"        章节: {result['paragraph_metadata']['section']}")
    #         print(f"        引用数: {result['paragraph_metadata']['citation_count']}")
    
    # # 测试11: 章节级语义搜索
    # print("\n11. 测试章节语义搜索:")
    # section_search = agent.search_relevant_sections(
    #     "methodology research design",
    #     top_n=3,
    #     section_types=["methodology", "methods"],  # 只搜索方法论章节
    #     min_score=0.25
    # )
    # print(f"   搜索结果: {section_search['status']}")
    # print(f"   消息: {section_search['message']}")
    # if section_search['status'] == 'success':
    #     print(f"   结果预览:")
    #     for i, result in enumerate(section_search['results'][:1]):
    #         print(f"     {i+1}. 分数: {result['score']}")
    #         print(f"        章节标题: {result['title']}")
    #         print(f"        章节类型: {result['section_metadata']['section_type']}")
    #         print(f"        内容预览: {result['text'][:100]}...")
    
    # # 测试12: 综合搜索
    # print("\n12. 测试综合语义搜索:")
    # comprehensive_search = agent.search_all_content_types(
    #     "strategic management competitive advantage",
    #     top_n_per_type=3,
    #     min_score=0.2
    # )
    # print(f"   搜索结果: {comprehensive_search['status']}")
    # print(f"   消息: {comprehensive_search['message']}")
    # if comprehensive_search['status'] == 'success':
    #     stats = comprehensive_search['overall_stats']
    #     print(f"   总体统计:")
    #     print(f"     总结果数: {stats['total_results']}")
    #     print(f"     最高分数: {stats['max_score']}")
    #     print(f"     句子: {stats['sentences_count']} 个")
    #     print(f"     段落: {stats['paragraphs_count']} 个") 
    #     print(f"     章节: {stats['sections_count']} 个")
        
    #     # 显示每种类型的最佳结果
    #     for content_type in ['sentences', 'paragraphs', 'sections']:
    #         type_results = comprehensive_search[content_type]
    #         if type_results['status'] == 'success' and type_results['results']:
    #             best = type_results['results'][0]
    #             print(f"     最佳{content_type[:-1]}: {best['score']} - {best['text'][:50]}...")
    
    # # 测试13: 带过滤条件的搜索
    # if test_paper_id:
    #     print("\n13. 测试带过滤条件的搜索:")
    #     filtered_search = agent.search_relevant_sentences(
    #         "strategy",
    #         top_n=10,
    #         paper_ids=[test_paper_id],  # 只在特定论文中搜索
    #         min_score=0.1
    #     )
    #     print(f"   搜索结果: {filtered_search['status']}")
    #     print(f"   消息: {filtered_search['message']}")
    #     if filtered_search['status'] == 'success':
    #         print(f"   在特定论文中找到 {filtered_search['total_results']} 个相关句子")
    
    # print("\n🎉 向量搜索测试完成!")
    # agent.close() 