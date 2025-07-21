# neo4j_graph_operations.py

from neo4j import GraphDatabase
from typing import List, Dict, Optional
import logging

class GraphDB:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # ==================== 原有的论文和论证节点操作 ====================

    def create_paper(self, paper_id: str, title: str, authors: List[str], year: int, stub: bool = False, **metadata):
        """创建或更新论文节点"""
        query = """
        MERGE (p:Paper {id: $paper_id})
        SET p.title = $title, 
            p.authors = $authors, 
            p.year = $year, 
            p.stub = $stub,
            p.doi = $doi,
            p.journal = $journal,
            p.publisher = $publisher
        """
        with self.driver.session() as session:
            session.run(query, 
                paper_id=paper_id, title=title, authors=authors, 
                year=year, stub=stub,
                doi=metadata.get('doi'),
                journal=metadata.get('journal'), 
                publisher=metadata.get('publisher')
            )

    def create_argument(self, arg_id: str, paper_id: str, text: str, claim_type: str,
                        section: Optional[str] = None, version: str = "v1.0",
                        confidence: Optional[float] = None, custom_tags: Optional[List[str]] = None):
        query = """
        MERGE (a:Argument {id: $arg_id})
        SET a.text = $text, a.claim_type = $claim_type, a.section = $section,
            a.version = $version, a.confidence = $confidence, a.custom_tags = $custom_tags
        WITH a
        MATCH (p:Paper {id: $paper_id})
        MERGE (a)-[:BELONGS_TO]->(p)
        """
        with self.driver.session() as session:
            session.run(query, arg_id=arg_id, paper_id=paper_id, text=text, claim_type=claim_type,
                        section=section, version=version, confidence=confidence, custom_tags=custom_tags)

    def create_relation(self, from_arg: str, to_arg_or_paper: str, relation_type: str,
                        confidence: Optional[float] = None, version: str = "v1.0"):
        query = """
        MATCH (a1:Argument {id: $from_arg})
        OPTIONAL MATCH (a2:Argument {id: $to_arg})
        OPTIONAL MATCH (p:Paper {id: $to_arg})
        WITH a1, coalesce(a2, p) as target
        WHERE target IS NOT NULL
        MERGE (a1)-[r:RELATES {relation_type: $relation_type}]->(target)
        SET r.confidence = $confidence, r.version = $version
        """
        with self.driver.session() as session:
            session.run(query, from_arg=from_arg, to_arg_or_paper=to_arg_or_paper,
                        relation_type=relation_type, confidence=confidence, version=version)

    def create_stub_relation(self, from_arg: str, to_paper_id: str, relation_type: str,
                             confidence: Optional[float] = None, version: str = "v1.0"):
        """
        If the target paper is not in the graph yet, create a stub Paper node and link to it.
        """
        query = """
        MATCH (a1:Argument {id: $from_arg})
        MERGE (p:Paper {id: $to_paper_id})
        ON CREATE SET p.stub = true
        MERGE (a1)-[r:RELATES {relation_type: $relation_type}]->(p)
        SET r.confidence = $confidence, r.version = $version
        """
        with self.driver.session() as session:
            session.run(query, from_arg=from_arg, to_paper_id=to_paper_id,
                        relation_type=relation_type, confidence=confidence, version=version)

    # ==================== 新增：句子节点操作 ====================
    
    def create_sentence(self, sentence_id: str, paper_id: str, paragraph_id: str,
                       text: str, sentence_index: int, has_citations: bool = False,
                       word_count: int = 0, char_count: int = 0):
        """创建句子节点及其关系"""
        query = """
        // 创建句子节点
        MERGE (s:Sentence {id: $sentence_id})
        SET s.text = $text,
            s.sentence_index = $sentence_index,
            s.has_citations = $has_citations,
            s.word_count = $word_count,
            s.char_count = $char_count
        
        // 连接到论文
        WITH s
        MATCH (p:Paper {id: $paper_id})
        MERGE (s)-[:BELONGS_TO]->(p)
        
        // 连接到段落
        WITH s  
        MATCH (para:Paragraph {id: $paragraph_id})
        MERGE (s)-[:BELONGS_TO]->(para)
        """
        with self.driver.session() as session:
            session.run(query,
                sentence_id=sentence_id, paper_id=paper_id, 
                paragraph_id=paragraph_id, text=text,
                sentence_index=sentence_index, has_citations=has_citations,
                word_count=word_count, char_count=char_count
            )
    
    def create_sentence_citation(self, sentence_id: str, cited_paper_id: str,
                                citation_text: str, citation_context: str = "",
                                confidence: float = 1.0):
        """创建句子引用关系"""
        query = """
        MATCH (s:Sentence {id: $sentence_id})
        MATCH (p:Paper {id: $cited_paper_id})
        MERGE (s)-[c:CITES]->(p)
        SET c.citation_text = $citation_text,
            c.citation_context = $citation_context,
            c.confidence = $confidence,
            c.created_at = datetime()
        """
        with self.driver.session() as session:
            session.run(query,
                sentence_id=sentence_id, cited_paper_id=cited_paper_id,
                citation_text=citation_text, citation_context=citation_context,
                confidence=confidence
            )
    
    # ==================== 新增：段落节点操作 ====================
    
    def create_paragraph(self, paragraph_id: str, paper_id: str, text: str,
                        paragraph_index: int, section: str = "",
                        citation_count: int = 0, sentence_count: int = 0,
                        has_citations: bool = False):
        """创建段落节点及其关系，支持has_citations属性"""
        query = """
        // 创建段落节点
        MERGE (para:Paragraph {id: $paragraph_id})
        SET para.text = $text,
            para.paragraph_index = $paragraph_index,
            para.section = $section,
            para.citation_count = $citation_count,
            para.sentence_count = $sentence_count,
            para.has_citations = $has_citations
        // 连接到论文
        WITH para
        MATCH (p:Paper {id: $paper_id})
        MERGE (para)-[:BELONGS_TO]->(p)
        """
        with self.driver.session() as session:
            session.run(query,
                paragraph_id=paragraph_id, paper_id=paper_id, text=text,
                paragraph_index=paragraph_index, section=section,
                citation_count=citation_count, sentence_count=sentence_count,
                has_citations=has_citations
            )
    
    def create_paragraph_citation(self, paragraph_id: str, cited_paper_id: str,
                                 citation_count: int, citation_density: float = 0.0):
        """创建段落引用关系（聚合级别）"""
        query = """
        MATCH (para:Paragraph {id: $paragraph_id})
        MATCH (p:Paper {id: $cited_paper_id})
        MERGE (para)-[c:CITES]->(p)
        SET c.citation_count = $citation_count,
            c.citation_density = $citation_density,
            c.created_at = datetime()
        """
        with self.driver.session() as session:
            session.run(query,
                paragraph_id=paragraph_id, cited_paper_id=cited_paper_id,
                citation_count=citation_count, citation_density=citation_density
            )
    

    def get_sentences_citing_paper(self, cited_paper_id: str, limit: int = 100) -> List[Dict]:
        """
        Retrieve all sentences that cite a given paper.
        """
        query = """
        MATCH (s:Sentence)-[c:CITES]->(p:Paper {id: $cited_paper_id})
        MATCH (s)-[:BELONGS_TO]->(source_paper:Paper)
        RETURN s.id as sentence_id,
               s.text as sentence_text,
               s.sentence_index as sentence_index,
               c.citation_text as citation_text,
               c.confidence as confidence,
               source_paper.title as source_paper_title,
               source_paper.authors as source_paper_authors,
               source_paper.year as source_paper_year
        ORDER BY source_paper.year DESC, s.sentence_index ASC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, cited_paper_id=cited_paper_id, limit=limit)
            return [dict(record) for record in result]
    
    def get_paragraphs_citing_paper(self, cited_paper_id: str, limit: int = 50) -> List[Dict]:
        """
        Retrieve all paragraphs that cite a given paper.
        """
        query = """
        MATCH (para:Paragraph)-[c:CITES]->(p:Paper {id: $cited_paper_id})
        MATCH (para)-[:BELONGS_TO]->(source_paper:Paper)
        RETURN para.id as paragraph_id,
               para.text as paragraph_text,
               para.section as section,
               c.citation_count as citation_count,
               c.citation_density as citation_density,
               source_paper.title as source_paper_title,
               source_paper.year as source_paper_year
        ORDER BY c.citation_count DESC, source_paper.year DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, cited_paper_id=cited_paper_id, limit=limit)
            return [dict(record) for record in result]
    
    def get_citation_context_for_ai_analysis(self, cited_paper_id: str) -> Dict:
        """
        获取引用某篇论文的完整上下文，用于AI分析
        返回句子级和段落级的完整信息
        """
        sentences = self.get_sentences_citing_paper(cited_paper_id, limit=200)
        paragraphs = self.get_paragraphs_citing_paper(cited_paper_id, limit=100)
        
        return {
            "cited_paper_id": cited_paper_id,
            "sentence_level_citations": sentences,
            "paragraph_level_citations": paragraphs,
            "total_citing_sentences": len(sentences),
            "total_citing_paragraphs": len(paragraphs)
        }
    
    def get_paper_citation_network(self, paper_id: str) -> Dict:
        """获取论文的完整引用网络"""
        query = """
        // 获取该论文引用的所有论文（出度）
        MATCH (source_paper:Paper {id: $paper_id})<-[:BELONGS_TO]-(s:Sentence)-[:CITES]->(cited:Paper)
        WITH cited, count(s) as citation_frequency
        
        // 获取引用该论文的所有论文（入度）
        OPTIONAL MATCH (citing_paper:Paper)<-[:BELONGS_TO]-(citing_s:Sentence)-[:CITES]->(:Paper {id: $paper_id})
        
        RETURN {
            outgoing_citations: collect(DISTINCT {
                paper_id: cited.id,
                title: cited.title,
                frequency: citation_frequency
            }),
            incoming_citations: collect(DISTINCT {
                paper_id: citing_paper.id,
                title: citing_paper.title,
                citing_sentences: count(citing_s)
            })
        } as citation_network
        """
        with self.driver.session() as session:
            result = session.run(query, paper_id=paper_id)
            return dict(result.single())['citation_network']

    # ==================== 原有的Schema节点操作 ====================

    def create_claim_schema_node(self, claim_id: str, label: str, description: str,
                                 is_deprecated: bool = False, is_default: bool = True):
        query = """
        MERGE (c:ClaimTypeSchema {id: $claim_id})
        SET c.label = $label, c.description = $description,
            c.is_deprecated = $is_deprecated, c.is_default = $is_default
        """
        with self.driver.session() as session:
            session.run(query, claim_id=claim_id, label=label,
                        description=description, is_deprecated=is_deprecated, is_default=is_default)

    def create_relation_schema_node(self, rel_id: str, label: str, description: str,
                                    is_deprecated: bool = False, is_default: bool = True):
        query = """
        MERGE (r:RelationTypeSchema {id: $rel_id})
        SET r.label = $label, r.description = $description,
            r.is_deprecated = $is_deprecated, r.is_default = $is_default
        """
        with self.driver.session() as session:
            session.run(query, rel_id=rel_id, label=label,
                        description=description, is_deprecated=is_deprecated, is_default=is_default)


if __name__ == "__main__":
    db = GraphDB(uri="bolt://localhost:7687", user="neo4j", password="12345678")
    
    # 原有测试
    # db.create_paper(paper_id="1", title="Paper 1", authors=["Author 1"], year=2021)
    # db.create_argument(arg_id="1", paper_id="1", text="Argument 1", claim_type="CLAIM_MAIN")
    # db.create_relation(from_arg="1", to_arg_or_paper="2", relation_type="CITES")
    # db.create_stub_relation(from_arg="1", to_paper_id="2", relation_type="CITES")
    
    # 新架构测试
    db.create_paper("paper_a", "Strategic Positioning", ["Author A"], 2020)
    db.create_paper("porter_1980", "Competitive Strategy", ["Michael Porter"], 1980, stub=True)
    
    # 创建段落和句子
    db.create_paragraph("paper_a_para_1", "paper_a", "This paragraph discusses strategy...", 1, "Introduction")
    db.create_sentence("paper_a_sent_1", "paper_a", "paper_a_para_1", 
                      "Porter (1980) argues that competitive advantage stems from positioning.", 1, True)
    
    # 创建引用关系
    db.create_sentence_citation("paper_a_sent_1", "porter_1980", "Porter (1980)")
    db.create_paragraph_citation("paper_a_para_1", "porter_1980", 1, 0.1)
    
    # 查询测试
    context = db.get_citation_context_for_ai_analysis("porter_1980")
    print("Citation context for AI analysis:", context)
    
    db.close()