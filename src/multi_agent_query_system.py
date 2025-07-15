"""
Multi-Agent Query System for CiteWeave
Based on LangGraph for robust academic paper querying with disambiguation
"""

import re
import json
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from .graph_builder import GraphDB
from .vector_indexer import VectorIndexer
from .llm_interface import LLMInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries the system can handle"""
    PAPER_SEARCH = "paper_search"
    CITATION_ANALYSIS = "citation_analysis"
    AUTHOR_PAPERS = "author_papers"
    ARGUMENT_RELATIONS = "argument_relations"
    UNKNOWN = "unknown"

@dataclass
class QueryState:
    """State shared between agents in the query processing pipeline"""
    # Input
    user_query: str = ""
    
    # Query Analysis
    query_type: QueryType = QueryType.UNKNOWN
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    
    # Paper Disambiguation
    candidate_papers: List[Dict] = field(default_factory=list)
    selected_papers: List[Dict] = field(default_factory=list)
    disambiguation_needed: bool = False
    
    # Results
    search_results: List[Dict] = field(default_factory=list)
    citation_relationships: Dict = field(default_factory=dict)
    vector_results: List[Dict] = field(default_factory=list)
    
    # Response Generation
    final_response: str = ""
    confidence_score: float = 0.0
    
    # Error Handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Agent Communication
    messages: List[str] = field(default_factory=list)

class MultiAgentQuerySystem:
    """
    Multi-agent system for academic paper querying with robust error handling
    """
    
    def __init__(self, 
                 graph_db: GraphDB,
                 vector_indexer: VectorIndexer,
                 llm_interface: LLMInterface):
        self.graph_db = graph_db
        self.vector_indexer = vector_indexer
        self.llm_interface = llm_interface
        
        # Build the agent workflow graph
        self.workflow = self._build_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.info("MultiAgentQuerySystem initialized")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with all agents"""
        workflow = StateGraph(QueryState)
        
        # Add nodes (agents)
        workflow.add_node("query_analyzer", self._query_analyzer_agent)
        workflow.add_node("paper_disambiguator", self._paper_disambiguator_agent)
        workflow.add_node("citation_analyzer", self._citation_analyzer_agent)
        workflow.add_node("vector_searcher", self._vector_searcher_agent)
        workflow.add_node("response_generator", self._response_generator_agent)
        
        # Define the workflow edges
        workflow.add_edge(START, "query_analyzer")
        
        # Conditional routing based on query type
        workflow.add_conditional_edges(
            "query_analyzer",
            self._route_after_analysis,
            {
                "disambiguate": "paper_disambiguator",
                "vector_search": "vector_searcher",
                "direct_citation": "citation_analyzer",
                "end": "response_generator"
            }
        )
        
        workflow.add_conditional_edges(
            "paper_disambiguator", 
            self._route_after_disambiguation,
            {
                "citation_analysis": "citation_analyzer",
                "vector_search": "vector_searcher", 
                "end": "response_generator"
            }
        )
        
        workflow.add_edge("citation_analyzer", "response_generator")
        workflow.add_edge("vector_searcher", "response_generator")
        workflow.add_edge("response_generator", END)
        
        return workflow

    def _query_analyzer_agent(self, state: QueryState) -> QueryState:
        """
        Agent 1: Analyze user query and extract entities
        """
        logger.info(f"Analyzing query: {state.user_query}")
        
        try:
            # Extract entities using regex and LLM
            entities = self._extract_entities(state.user_query)
            state.extracted_entities = entities
            
            # Determine query type
            state.query_type = self._classify_query_type(state.user_query, entities)
            
            state.messages.append(f"Query analyzed: type={state.query_type.value}, entities={entities}")
            
        except Exception as e:
            error_msg = f"Query analysis failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
            
        return state

    def _paper_disambiguator_agent(self, state: QueryState) -> QueryState:
        """
        Agent 2: Handle paper disambiguation when multiple papers match
        """
        logger.info("Running paper disambiguation")
        
        try:
            entities = state.extracted_entities
            
            # Search for papers by author/year
            if "author" in entities and "year" in entities:
                papers = self.graph_db.find_papers_by_author_year(
                    entities["author"], 
                    entities.get("year"), 
                    fuzzy=True
                )
            elif "author" in entities:
                papers = self.graph_db.find_papers_by_author_year(
                    entities["author"], 
                    fuzzy=True
                )
            else:
                papers = []
            
            state.candidate_papers = papers
            
            if len(papers) == 0:
                state.warnings.append(f"No papers found for author '{entities.get('author', 'unknown')}'")
            elif len(papers) == 1:
                state.selected_papers = papers
                state.messages.append(f"Single paper found: {papers[0]['title']}")
            else:
                # Multiple papers found - need disambiguation
                state.disambiguation_needed = True
                state.selected_papers = papers  # Return all for user to choose
                state.messages.append(f"Found {len(papers)} papers by {entities.get('author')}")
                
        except Exception as e:
            error_msg = f"Paper disambiguation failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
            
        return state

    def _citation_analyzer_agent(self, state: QueryState) -> QueryState:
        """
        Agent 3: Analyze citation relationships
        """
        logger.info("Running citation analysis")
        
        try:
            if not state.selected_papers:
                state.warnings.append("No papers selected for citation analysis")
                return state
            
            all_relationships = {}
            
            for paper in state.selected_papers:
                paper_id = paper["paper_id"]
                
                # Get citation relationships
                relationships = self.graph_db.get_citation_relationships(paper_id, "both")
                
                if "error" in relationships:
                    state.warnings.append(relationships["error"])
                else:
                    all_relationships[paper_id] = relationships
                    
            state.citation_relationships = all_relationships
            state.messages.append(f"Citation analysis completed for {len(all_relationships)} papers")
            
        except Exception as e:
            error_msg = f"Citation analysis failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
            
        return state

    def _vector_searcher_agent(self, state: QueryState) -> QueryState:
        """
        Agent 4: Perform semantic vector search
        """
        logger.info("Running vector search")
        
        try:
            # Use the original query for semantic search
            results = self.vector_indexer.search(state.user_query, limit=10)
            state.vector_results = results
            state.messages.append(f"Vector search returned {len(results)} results")
            
        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
            
        return state

    def _response_generator_agent(self, state: QueryState) -> QueryState:
        """
        Agent 5: Generate final response based on all collected information
        """
        logger.info("Generating final response")
        
        try:
            response_parts = []
            
            # Handle different response scenarios
            if state.errors:
                response_parts.append("⚠️ Some issues occurred:")
                for error in state.errors:
                    response_parts.append(f"- {error}")
                response_parts.append("")
            
            if state.warnings:
                response_parts.append("⚡ Note:")
                for warning in state.warnings:
                    response_parts.append(f"- {warning}")
                response_parts.append("")
            
            # Generate main response content
            if state.query_type == QueryType.PAPER_SEARCH or state.disambiguation_needed:
                response_parts.extend(self._generate_paper_search_response(state))
            elif state.query_type == QueryType.CITATION_ANALYSIS:
                response_parts.extend(self._generate_citation_response(state))
            elif state.citation_relationships:
                response_parts.extend(self._generate_citation_response(state))
            elif state.vector_results:
                response_parts.extend(self._generate_semantic_search_response(state))
            else:
                response_parts.append("😅 Sorry, I couldn't find relevant information. Please try:")
                response_parts.append("- Check the spelling of the author's name")
                response_parts.append("- Confirm the publication year of the paper")
                response_parts.append("- Use more specific keywords")
            
            state.final_response = "\n".join(response_parts)
            state.confidence_score = self._calculate_confidence_score(state)
            
        except Exception as e:
            error_msg = f"Response generation failed: {str(e)}"
            state.errors.append(error_msg)
            state.final_response = f"❌ System error: {error_msg}"
            logger.error(error_msg)
            
        return state

    def _generate_paper_search_response(self, state: QueryState) -> List[str]:
        """Generate response for paper search queries"""
        response_parts = []
        
        if state.disambiguation_needed and len(state.candidate_papers) > 1:
            author = state.extracted_entities.get("author", "the author")
            year = state.extracted_entities.get("year")
            
            if year:
                response_parts.append(f"📚 Found {len(state.candidate_papers)} papers published by {author} in {year}:")
            else:
                response_parts.append(f"📚 Found {len(state.candidate_papers)} papers published by {author}:")
            response_parts.append("")
            
            for i, paper in enumerate(state.candidate_papers[:10], 1):  # Limit to 10
                title = paper.get("title", "Unknown Title")
                year = paper.get("year", "Unknown Year")
                journal = paper.get("journal", "Unknown Journal")
                is_stub = paper.get("is_stub", False)
                
                status = " [Citation Placeholder]" if is_stub else ""
                response_parts.append(f"{i}. **{title}** ({year}){status}")
                if journal and journal != "Unknown Journal":
                    response_parts.append(f"   📖 Published in: {journal}")
                response_parts.append("")
                
        elif state.selected_papers:
            paper = state.selected_papers[0]
            title = paper.get("title", "Unknown Title")
            year = paper.get("year", "Unknown Year") 
            authors = paper.get("authors", [])
            journal = paper.get("journal")
            
            response_parts.append(f"📄 **{title}** ({year})")
            if authors:
                response_parts.append(f"👥 Authors: {', '.join(authors)}")
            if journal:
                response_parts.append(f"📖 Journal: {journal}")
            response_parts.append("")
            
        return response_parts

    def _generate_citation_response(self, state: QueryState) -> List[str]:
        """Generate response for citation analysis"""
        response_parts = []
        
        for paper_id, relationships in state.citation_relationships.items():
            paper_info = relationships.get("paper_info", {})
            title = paper_info.get("title", "Unknown Paper")
            
            response_parts.append(f"📄 **{title}** 的引用关系：")
            response_parts.append("")
            
            # Incoming citations (who cites this paper)
            incoming = relationships.get("incoming", [])
            if incoming:
                response_parts.append(f"📈 **被以下{len(incoming)}篇论文引用：**")
                for citation in incoming[:5]:  # Limit to 5
                    citing_title = citation.get("citing_title", "Unknown Title")
                    citing_year = citation.get("citing_year", "Unknown Year")
                    relation_type = citation.get("citation_relation_type", "引用")
                    is_stub = citation.get("citing_is_stub", False)
                    
                    status = " [占位]" if is_stub else ""
                    response_parts.append(f"  • {citing_title} ({citing_year}) - {relation_type}{status}")
                
                if len(incoming) > 5:
                    response_parts.append(f"  ... 还有{len(incoming) - 5}篇论文")
                response_parts.append("")
            
            # Outgoing citations (what this paper cites)
            outgoing = relationships.get("outgoing", [])
            if outgoing:
                response_parts.append(f"📚 **引用了以下{len(outgoing)}篇论文：**")
                for citation in outgoing[:5]:  # Limit to 5
                    cited_title = citation.get("cited_title", "Unknown Title")
                    cited_year = citation.get("cited_year", "Unknown Year")
                    relation_type = citation.get("citation_relation_type", "引用")
                    is_stub = citation.get("cited_is_stub", False)
                    
                    status = " [占位]" if is_stub else ""
                    response_parts.append(f"  • {cited_title} ({cited_year}) - {relation_type}{status}")
                
                if len(outgoing) > 5:
                    response_parts.append(f"  ... 还有{len(outgoing) - 5}篇论文")
                response_parts.append("")
            
            if not incoming and not outgoing:
                response_parts.append("🔍 暂无引用关系数据")
                response_parts.append("")
        
        return response_parts

    def _generate_semantic_search_response(self, state: QueryState) -> List[str]:
        """Generate response for semantic search results"""
        response_parts = []
        
        response_parts.append(f"🔍 **语义搜索结果** (找到{len(state.vector_results)}条相关内容)：")
        response_parts.append("")
        
        for i, result in enumerate(state.vector_results[:5], 1):  # Top 5
            text = result.get("text", "")[:200] + "..." if len(result.get("text", "")) > 200 else result.get("text", "")
            title = result.get("title", "Unknown Paper")
            year = result.get("year", "Unknown Year")
            score = result.get("score", 0)
            claim_type = result.get("claim_type", "未分类")
            
            response_parts.append(f"**{i}. {title}** ({year}) - Similarity: {score}")
            response_parts.append(f"📝 Content Type: {claim_type}")
            response_parts.append(f"💭 Content: {text}")
            response_parts.append("")
        
        return response_parts

    # Routing functions for conditional edges
    def _route_after_analysis(self, state: QueryState) -> str:
        """Route after query analysis"""
        entities = state.extracted_entities
        
        if "author" in entities:
            return "disambiguate"
        elif state.query_type == QueryType.CITATION_ANALYSIS:
            return "direct_citation"
        else:
            return "vector_search"

    def _route_after_disambiguation(self, state: QueryState) -> str:
        """Route after paper disambiguation"""
        if state.query_type == QueryType.CITATION_ANALYSIS and state.selected_papers:
            return "citation_analysis"
        elif state.selected_papers:
            return "citation_analysis"
        else:
            return "vector_search"

    # Helper functions
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from user query"""
        entities = {}
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        if year_match:
            entities["year"] = int(year_match.group())
        
        # Extract author names (simple heuristic)
        # Look for capitalized words that might be names
        potential_authors = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Filter out common words that aren't names
        common_words = {'The', 'In', 'Of', 'And', 'For', 'With', 'By', 'To', 'From'}
        authors = [name for name in potential_authors if name not in common_words]
        
        if authors:
            entities["author"] = " ".join(authors)
        
        # Extract keywords for semantic search
        keywords = re.findall(r'\b\w+\b', query.lower())
        entities["keywords"] = [kw for kw in keywords if len(kw) > 2]
        
        return entities

    def _classify_query_type(self, query: str, entities: Dict) -> QueryType:
        """Classify the type of query"""
        query_lower = query.lower()
        
        # Citation-related keywords
        citation_keywords = ['引用', '被引用', '引用了', '支持', '反驳', '批评', '扩展', '详述']
        if any(keyword in query_lower for keyword in citation_keywords):
            return QueryType.CITATION_ANALYSIS
        
        # Author search keywords  
        if "author" in entities:
            return QueryType.AUTHOR_PAPERS
        
        # Default to semantic search
        return QueryType.PAPER_SEARCH

    def _calculate_confidence_score(self, state: QueryState) -> float:
        """Calculate confidence score for the response"""
        score = 1.0
        
        # Deduct for errors and warnings
        score -= len(state.errors) * 0.3
        score -= len(state.warnings) * 0.1
        
        # Boost for successful operations
        if state.selected_papers:
            score += 0.2
        if state.citation_relationships:
            score += 0.2
        if state.vector_results:
            score += 0.1
            
        return max(0.0, min(1.0, score))

    def query(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Main entry point for querying the system
        
        Args:
            user_query: User's natural language question
            thread_id: Thread ID for conversation memory
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Initialize state
            initial_state = QueryState(user_query=user_query)
            
            # Run the workflow
            config = {"configurable": {"thread_id": thread_id}}
            result = self.app.invoke(initial_state, config=config)
            
            return {
                "response": result.final_response,
                "confidence": result.confidence_score,
                "query_type": result.query_type.value,
                "candidate_papers": result.candidate_papers,
                "errors": result.errors,
                "warnings": result.warnings,
                "debug_messages": result.messages
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "response": f"❌ System error: {str(e)}",
                "confidence": 0.0,
                "query_type": "error",
                "errors": [str(e)],
                "warnings": [],
                "debug_messages": []
            }

# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the system
    pass 