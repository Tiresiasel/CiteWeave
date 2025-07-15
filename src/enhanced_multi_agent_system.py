"""
Enhanced Multi-Agent Query System for CiteWeave
Includes language processing, clarification questions, memory management, and robust error handling
"""

import re
import json
import logging
import asyncio
import os
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph_builder import GraphDB
from vector_indexer import MultiLevelVectorIndexer
from author_paper_index import AuthorPaperIndex
from enhanced_llm_manager import EnhancedLLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries the system can handle"""
    # Graph database queries
    CITATION_CONCEPT_ANALYSIS = "citation_concept_analysis"  # When paper A is cited, what concepts are most mentioned
    AUTHOR_CITATION_REASON = "author_citation_reason"       # Why is a person cited
    
    # Vector database queries  
    CONCEPT_DEFINITION = "concept_definition"               # What does a concept mean
    
    # PDF + AI queries
    DOCUMENT_CONTENT = "document_content"                   # What does this paper discuss
    AUTHOR_COLLECTION = "author_collection"                # What do all of Porter's papers discuss
    
    # Legacy types for backward compatibility
    PAPER_SEARCH = "paper_search"
    CITATION_ANALYSIS = "citation_analysis" 
    AUTHOR_PAPERS = "author_papers"
    ARGUMENT_RELATIONS = "argument_relations"
    CLARIFICATION_NEEDED = "clarification_needed"
    UNKNOWN = "unknown"

class ActionType(Enum):
    """Types of actions agents can take"""
    CONTINUE = "continue"
    ASK_CLARIFICATION = "ask_clarification"
    END_WITH_RESULTS = "end_with_results"
    END_WITH_ERROR = "end_with_error"

@dataclass
class QueryState:
    """Enhanced state shared between agents"""
    # Input and language processing
    original_query: str = ""
    user_language: str = "en"
    processed_query: str = ""  # English version for internal processing
    
    # Query analysis
    query_type: QueryType = QueryType.UNKNOWN
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    confidence_threshold: float = 0.3
    
    # Paper disambiguation
    candidate_papers: List[Dict] = field(default_factory=list)
    selected_papers: List[Dict] = field(default_factory=list)
    disambiguation_needed: bool = False
    clarification_question: str = ""
    
    # Results from different data sources
    search_results: List[Dict] = field(default_factory=list)
    citation_relationships: Dict = field(default_factory=dict)
    vector_results: List[Dict] = field(default_factory=list)
    
    # New: PDF and author collection results
    pdf_paths: List[str] = field(default_factory=list)
    pdf_content_analysis: str = ""
    author_papers: List[Dict] = field(default_factory=list)
    
    # Translation tracking
    translation_used: bool = False
    original_language_detected: str = ""
    
    # Multi-route processing
    required_routes: List[str] = field(default_factory=list)  # List of required routes
    completed_routes: List[str] = field(default_factory=list)  # List of completed routes
    route_results: Dict[str, Any] = field(default_factory=dict)  # Results for each route
    
    # Response
    final_response: str = ""
    response_language: str = "en"
    confidence_score: float = 0.0
    next_action: ActionType = ActionType.CONTINUE
    
    # Context and memory
    conversation_context: str = ""
    thread_id: str = "default"
    user_id: str = "default"
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    debug_messages: List[str] = field(default_factory=list)

class EnhancedMultiAgentSystem:
    """
    Enhanced multi-agent system with:
    - Multi-language support and translation
    - Intelligent routing to different data sources
    - Graph database for citation analysis
    - Vector database for concept definitions  
    - PDF analysis for document content
    - Author index for author collections
    - Memory management and context
    - Clarification questions
    - Robust error handling
    """
    
    def __init__(self, 
                 graph_db: GraphDB,
                 vector_indexer: MultiLevelVectorIndexer,
                 author_index: AuthorPaperIndex,
                 config_path: str = "config/model_config.json"):
        
        self.graph_db = graph_db
        self.vector_indexer = vector_indexer
        self.author_index = author_index
        self.llm_manager = EnhancedLLMManager(config_path)
        
        # Build the workflow
        self.workflow = self._build_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.info("Enhanced Multi-Agent System initialized with multi-language support")

    def _build_workflow(self) -> StateGraph:
        """Build the enhanced LangGraph workflow with multi-route parallel processing"""
        workflow = StateGraph(QueryState)
        
        # Add agent nodes
        workflow.add_node("language_processor", self._language_processor_agent)
        workflow.add_node("smart_router", self._smart_router_agent)
        workflow.add_node("paper_disambiguator", self._paper_disambiguator_agent)
        workflow.add_node("route_coordinator", self._route_coordinator_agent)
        
        # Specialized data source agents (can run in parallel)
        workflow.add_node("graph_citation_analyzer", self._graph_citation_analyzer_agent)
        workflow.add_node("vector_concept_searcher", self._vector_concept_searcher_agent)
        workflow.add_node("pdf_content_analyzer", self._pdf_content_analyzer_agent)
        workflow.add_node("author_collection_handler", self._author_collection_handler_agent)
        
        workflow.add_node("response_generator", self._response_generator_agent)
        workflow.add_node("clarification_handler", self._clarification_handler_agent)
        
        # Define workflow edges
        workflow.add_edge(START, "language_processor")
        workflow.add_edge("language_processor", "smart_router")
        
        # Smart routing can go to multiple destinations
        workflow.add_conditional_edges(
            "smart_router",
            self._intelligent_multi_routing,
            {
                "coordinate": "route_coordinator",
                "disambiguate": "paper_disambiguator",
                "clarification": "clarification_handler"
            }
        )
        
        # Route coordinator manages parallel execution
        workflow.add_conditional_edges(
            "route_coordinator",
            self._coordinate_routes,
            {
                "graph_analysis": "graph_citation_analyzer",
                "vector_search": "vector_concept_searcher",
                "pdf_analysis": "pdf_content_analyzer",
                "author_collection": "author_collection_handler",
                "generate_response": "response_generator"
            }
        )
        
        # After disambiguation, go to coordinator
        workflow.add_edge("paper_disambiguator", "route_coordinator")
        
        # All data source agents return to coordinator for next route or completion
        workflow.add_edge("graph_citation_analyzer", "route_coordinator")
        workflow.add_edge("vector_concept_searcher", "route_coordinator")
        workflow.add_edge("pdf_content_analyzer", "route_coordinator")
        workflow.add_edge("author_collection_handler", "route_coordinator")
        
        # Final response generation and clarification
        workflow.add_edge("clarification_handler", "response_generator")
        workflow.add_edge("response_generator", END)
        
        return workflow

    async def _language_processor_agent(self, state: QueryState) -> QueryState:
        """Agent 1: Multi-language processing and translation to English for internal processing"""
        logger.info("Multi-language processing agent started")
        
        try:
            # Get conversation context
            state.conversation_context = self.llm_manager.get_conversation_context(
                state.thread_id, state.user_id
            )
            
            # Detect user language
            detected_lang = await self._detect_language(state.original_query)
            state.user_language = detected_lang
            state.response_language = detected_lang
            state.original_language_detected = detected_lang
            
            # Translate to English for internal processing if needed
            if detected_lang != "en":
                english_query = await self._translate_to_english(state.original_query, detected_lang)
                state.processed_query = english_query
                state.translation_used = True
                
                state.debug_messages.append(
                    f"Language detected: {detected_lang}, Original: '{state.original_query}', Translated: '{english_query}'"
                )
            else:
                state.processed_query = state.original_query
                state.translation_used = False
                
                state.debug_messages.append(
                    f"Language detected: English, No translation needed"
                )
            
        except Exception as e:
            error_msg = f"Language processing failed: {str(e)}"
            state.errors.append(error_msg)
            state.processed_query = state.original_query  # Fallback
            state.user_language = "en"  # Default fallback
            logger.error(error_msg)
        
        return state

    async def _smart_router_agent(self, state: QueryState) -> QueryState:
        """Agent 2: Intelligent multi-route analysis using powerful AI to determine which data sources needed"""
        logger.info("Smart router agent started - using powerful AI for routing decisions")
        
        try:
            # Use the most powerful model for routing decisions
            routing_model = self.llm_manager.get_agent_model("query_analyzer")
            
            # Enhanced entity extraction
            entities = await self._enhanced_entity_extraction(state.processed_query)
            state.extracted_entities = entities
            
            # Use AI to determine required routes (can be multiple)
            required_routes = await self._ai_route_analysis(state.processed_query, entities, routing_model)
            state.required_routes = required_routes
            
            # Check if disambiguation is needed
            if self._needs_disambiguation(entities, required_routes):
                state.next_action = ActionType.ASK_CLARIFICATION
                state.debug_messages.append("Paper disambiguation needed before routing")
            else:
                state.next_action = ActionType.CONTINUE
            
            # Check overall confidence
            confidence = await self._calculate_routing_confidence(state.processed_query, entities, required_routes, routing_model)
            
            if confidence < state.confidence_threshold:
                state.next_action = ActionType.ASK_CLARIFICATION
                state.debug_messages.append(f"Low routing confidence ({confidence:.2f}), clarification needed")
            
            state.debug_messages.append(
                f"Required routes: {required_routes}, Entities: {entities}, Confidence: {confidence:.2f}"
            )
            
        except Exception as e:
            error_msg = f"Smart routing failed: {str(e)}"
            state.errors.append(error_msg)
            # Fallback to basic routing
            state.required_routes = ["vector_search"]
            logger.error(error_msg)
        
        return state

    async def _route_coordinator_agent(self, state: QueryState) -> QueryState:
        """Route Coordinator: Manages parallel execution of multiple data sources"""
        logger.info("Route coordinator agent started")
        
        try:
            # Check which routes still need to be processed
            remaining_routes = [route for route in state.required_routes if route not in state.completed_routes]
            
            if not remaining_routes:
                # All routes completed, ready for response generation
                state.next_action = ActionType.END_WITH_RESULTS
                state.debug_messages.append("All routes completed, generating response")
            else:
                # Process next route (the workflow will handle parallel execution)
                next_route = remaining_routes[0]
                state.debug_messages.append(f"Processing route: {next_route}, Remaining: {remaining_routes}")
            
        except Exception as e:
            error_msg = f"Route coordination failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
        
        return state

    async def _query_analyzer_agent(self, state: QueryState) -> QueryState:
        """Agent 2: Analyze query intent and extract entities"""
        logger.info("Query analyzer agent started")
        
        try:
            model = self.llm_manager.get_agent_model("query_analyzer")
            
            # Enhanced entity extraction with LLM assistance
            entities = await self._enhanced_entity_extraction(state.processed_query)
            state.extracted_entities = entities
            
            # Classify query type
            state.query_type = self._classify_query_type(state.processed_query, entities)
            
            # Check confidence and determine if clarification is needed
            confidence = self._calculate_query_confidence(state.processed_query, entities)
            
            if confidence < state.confidence_threshold:
                state.next_action = ActionType.ASK_CLARIFICATION
                state.debug_messages.append(f"Low confidence ({confidence:.2f}), clarification needed")
            
            state.debug_messages.append(
                f"Query type: {state.query_type.value}, Entities: {entities}, Confidence: {confidence:.2f}"
            )
            
        except Exception as e:
            error_msg = f"Query analysis failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
        
        return state

    async def _paper_disambiguator_agent(self, state: QueryState) -> QueryState:
        """Agent 3: Handle paper disambiguation and search"""
        logger.info("Paper disambiguator agent started")
        
        try:
            entities = state.extracted_entities
            
            # Search for papers
            papers = []
            
            if "author" in entities:
                if "year" in entities:
                    papers = self.graph_db.find_papers_by_author_year(
                        entities["author"], entities["year"], fuzzy=True
                    )
                else:
                    papers = self.graph_db.find_papers_by_author_year(
                        entities["author"], fuzzy=True
                    )
            
            state.candidate_papers = papers
            
            # Determine next action based on results
            if len(papers) == 0:
                # No papers found - need clarification
                state.next_action = ActionType.ASK_CLARIFICATION
                state.warnings.append(f"No papers found for author '{entities.get('author', 'unknown')}'")
                
            elif len(papers) == 1:
                # Single paper found - proceed
                state.selected_papers = papers
                state.debug_messages.append(f"Single paper found: {papers[0]['title']}")
                
            elif len(papers) <= 5:
                # Multiple papers but manageable - show options
                state.disambiguation_needed = True
                state.selected_papers = papers
                state.debug_messages.append(f"Found {len(papers)} papers for disambiguation")
                
            else:
                # Too many papers - need more specific query
                state.next_action = ActionType.ASK_CLARIFICATION
                state.warnings.append(f"Found {len(papers)} papers - too many to display")
                
        except Exception as e:
            error_msg = f"Paper disambiguation failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
        
        return state

    async def _graph_citation_analyzer_agent(self, state: QueryState) -> QueryState:
        """Agent: Graph-based citation analysis for citation concepts and author citation reasons"""
        logger.info("Graph citation analyzer agent started")
        
        try:
            if not state.selected_papers:
                state.warnings.append("No papers selected for citation analysis")
                return state
            
            all_relationships = {}
            
            # Use papers from disambiguation or search in entities
            papers_to_analyze = state.selected_papers or self._find_papers_from_entities(state.extracted_entities)
            
            for paper in papers_to_analyze:
                paper_id = paper.get("paper_id", paper.get("id"))
                if not paper_id:
                    continue
                    
                # Determine citation direction based on query
                direction = self._determine_citation_direction(state.processed_query)
                
                relationships = self.graph_db.get_citation_relationships(paper_id, direction)
                
                if "error" in relationships:
                    state.warnings.append(relationships["error"])
                else:
                    all_relationships[paper_id] = relationships
            
            state.citation_relationships = all_relationships
            state.route_results["graph_analysis"] = all_relationships
            state.completed_routes.append("graph_analysis")
            state.debug_messages.append(f"Graph citation analysis completed for {len(all_relationships)} papers")
            
        except Exception as e:
            error_msg = f"Citation analysis failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
        
        return state

    async def _vector_concept_searcher_agent(self, state: QueryState) -> QueryState:
        """Agent: Vector-based concept definition search using smart search with query routing"""
        logger.info("Vector concept searcher agent started")
        
        try:
            # Use smart search which automatically determines granularity level
            results = self.vector_indexer.smart_search(state.processed_query, limit=10)
            state.vector_results = results
            state.route_results["vector_search"] = results
            state.completed_routes.append("vector_search")
            state.debug_messages.append(f"Vector concept search returned {len(results)} results at appropriate granularity")
            
        except Exception as e:
            error_msg = f"Vector concept search failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
        
        return state

    async def _pdf_content_analyzer_agent(self, state: QueryState) -> QueryState:
        """Agent: PDF content analysis for document-level questions"""
        logger.info("PDF content analyzer agent started")
        
        try:
            # Find PDF paths for papers
            pdf_paths = []
            papers_to_analyze = state.selected_papers or self._find_papers_from_entities(state.extracted_entities)
            
            for paper in papers_to_analyze:
                paper_id = paper.get("paper_id", paper.get("id"))
                if paper_id:
                    pdf_path = self.author_index.get_paper_pdf_path(paper_id)
                    if pdf_path and os.path.exists(pdf_path):
                        pdf_paths.append(pdf_path)
            
            if not pdf_paths:
                state.warnings.append("No accessible PDF files found for the specified papers")
                state.route_results["pdf_analysis"] = "No PDFs available"
            else:
                # Use large context model for PDF analysis
                large_model = self.llm_manager.get_agent_model("response_generator")
                
                pdf_analysis = await self._analyze_pdfs_with_ai(pdf_paths, state.processed_query, large_model)
                state.pdf_content_analysis = pdf_analysis
                state.route_results["pdf_analysis"] = pdf_analysis
                state.debug_messages.append(f"PDF analysis completed for {len(pdf_paths)} documents")
            
            state.completed_routes.append("pdf_analysis")
            
        except Exception as e:
            error_msg = f"PDF content analysis failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
        
        return state

    async def _author_collection_handler_agent(self, state: QueryState) -> QueryState:
        """Agent: Author collection analysis for comprehensive author work analysis"""
        logger.info("Author collection handler agent started")
        
        try:
            entities = state.extracted_entities
            author_name = entities.get("author")
            
            if not author_name:
                state.warnings.append("No author specified for collection analysis")
                state.route_results["author_collection"] = "No author specified"
            else:
                # Find all papers by the author
                author_papers = self.author_index.find_papers_by_author(author_name, exact_match=False)
                state.author_papers = author_papers
                
                if not author_papers:
                    state.warnings.append(f"No papers found for author '{author_name}'")
                    state.route_results["author_collection"] = f"No papers found for {author_name}"
                else:
                    # Get PDF paths for all papers
                    paper_ids = [p["paper_id"] for p in author_papers]
                    pdf_paths_dict = self.author_index.get_papers_pdf_paths(paper_ids)
                    available_pdfs = [path for path in pdf_paths_dict.values() if path and os.path.exists(path)]
                    
                    if available_pdfs:
                        # Use large context model for comprehensive analysis
                        large_model = self.llm_manager.get_agent_model("response_generator")
                        
                        collection_analysis = await self._analyze_author_collection(
                            author_name, author_papers, available_pdfs, state.processed_query, large_model
                        )
                        state.route_results["author_collection"] = collection_analysis
                        state.debug_messages.append(f"Author collection analysis completed for {author_name}: {len(author_papers)} papers, {len(available_pdfs)} PDFs")
                    else:
                        # Fallback to metadata-only analysis
                        metadata_analysis = await self._analyze_author_metadata_only(author_name, author_papers, state.processed_query)
                        state.route_results["author_collection"] = metadata_analysis
                        state.debug_messages.append(f"Author metadata analysis completed for {author_name}: {len(author_papers)} papers")
            
            state.completed_routes.append("author_collection")
            
        except Exception as e:
            error_msg = f"Author collection analysis failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
        
        return state

    async def _clarification_handler_agent(self, state: QueryState) -> QueryState:
        """Agent 6: Generate clarification questions"""
        logger.info("Clarification handler agent started")
        
        try:
            # Determine what clarification is needed
            issue = self._determine_clarification_issue(state)
            
            # Generate clarification question
            clarification = await self.llm_manager.generate_clarification_question(
                issue, state.conversation_context, state.user_language
            )
            
            state.clarification_question = clarification
            state.next_action = ActionType.ASK_CLARIFICATION
            state.debug_messages.append(f"Generated clarification question: {clarification[:100]}...")
            
        except Exception as e:
            error_msg = f"Clarification generation failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)
        
        return state

    async def _response_generator_agent(self, state: QueryState) -> QueryState:
        """Agent 7: Generate final response"""
        logger.info("Response generator agent started")
        
        try:
            model = self.llm_manager.get_agent_model("response_generator")
            
            # Generate response based on action type
            if state.next_action == ActionType.ASK_CLARIFICATION:
                response = await self._generate_clarification_response(state)
            else:
                response = await self._generate_content_response(state)
            
            # Translate back to user's language if needed
            if state.response_language != "en":
                response = await self._translate_response(response, state.response_language)
            
            state.final_response = response
            state.confidence_score = self._calculate_final_confidence(state)
            
            # Add to conversation memory
            self.llm_manager.add_conversation_turn(
                thread_id=state.thread_id,
                user_message=state.original_query,
                user_language=state.user_language,
                agent_response=state.final_response,
                response_language=state.response_language,
                query_type=state.query_type.value,
                confidence=state.confidence_score,
                metadata={
                    "action": state.next_action.value,
                    "papers_found": len(state.candidate_papers),
                    "citations_analyzed": len(state.citation_relationships),
                    "vector_results": len(state.vector_results)
                },
                user_id=state.user_id
            )
            
        except Exception as e:
            error_msg = f"Response generation failed: {str(e)}"
            state.errors.append(error_msg)
            state.final_response = f"I encountered an error: {error_msg}"
            logger.error(error_msg)
        
        return state

    # AI-Powered Routing Methods
    async def _ai_route_analysis(self, query: str, entities: Dict[str, Any], model) -> List[str]:
        """Use powerful AI to analyze query and determine required data sources"""
        try:
            # Prepare context about available data sources
            data_sources_context = """
Available Data Sources:
1. graph_analysis: For citation relationships, author citation reasons, concept mentions in citations
   - Use when: asking about who cites what, why authors are cited, what concepts are mentioned when papers are cited
   - Examples: "What concepts are most mentioned when Porter 1980 is cited", "Why is an author cited"

2. vector_search: For concept definitions, semantic content search
   - Use when: asking what concepts mean, finding papers with similar content
   - Examples: "What is competitive advantage", "competitive strategy concept", "Definition of Five Forces Model"

3. pdf_analysis: For understanding specific document content
   - Use when: asking what a specific paper discusses, document summary
   - Examples: "What does this paper discuss", "Content of Porter 1980's paper", "summarize this paper"

4. author_collection: For analyzing all works by an author
   - Use when: asking about an author's body of work, comprehensive author analysis
   - Examples: "All papers by Porter", "Research topics of this author", "author's contribution overview"

Important: A single query may require MULTIPLE data sources. For example:
- "What is Porter's competitive strategy theory and how is it cited" needs both vector_search AND graph_analysis
- "Analyze the citation impact of all Porter's papers" needs both author_collection AND graph_analysis
"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an expert routing agent for academic paper queries. Analyze the query and determine which data sources are needed.

{data_sources_context}

Return a JSON list of required data sources. Be comprehensive - if a query could benefit from multiple sources, include them all.

Response format: ["source1", "source2", ...]
Available sources: graph_analysis, vector_search, pdf_analysis, author_collection"""),
                ("user", "Query: {query}\nEntities: {entities}")
            ])
            
            chain = prompt | model | StrOutputParser()
            result = await chain.ainvoke({"query": query, "entities": str(entities)})
            
            # Parse the response
            try:
                import json
                routes = json.loads(result.strip())
                if isinstance(routes, list):
                    # Validate routes
                    valid_routes = ["graph_analysis", "vector_search", "pdf_analysis", "author_collection"]
                    filtered_routes = [r for r in routes if r in valid_routes]
                    return filtered_routes if filtered_routes else ["vector_search"]  # fallback
                else:
                    return ["vector_search"]  # fallback
            except:
                # Fallback to pattern matching if JSON parsing fails
                return self._fallback_route_analysis(query, entities)
                
        except Exception as e:
            logger.warning(f"AI route analysis failed: {e}, using fallback")
            return self._fallback_route_analysis(query, entities)

    def _fallback_route_analysis(self, query: str, entities: Dict[str, Any]) -> List[str]:
        """Fallback routing analysis using pattern matching"""
        routes = []
        query_lower = query.lower()
        
        # Check for citation-related queries
        citation_keywords = ['cite', 'citation', 'referenced', 'reference', 'mention']
        if any(kw in query_lower for kw in citation_keywords):
            routes.append("graph_analysis")
        
        # Check for concept definition queries  
        definition_keywords = ['what is', 'definition', 'concept', 'theory']
        if any(kw in query_lower for kw in definition_keywords):
            routes.append("vector_search")
        
        # Check for document content queries
        content_keywords = ['what does it discuss', 'content', 'discussion', 'paper']
        if any(kw in query_lower for kw in content_keywords):
            if entities.get("author") and "all" in query_lower:
                routes.append("author_collection")
            else:
                routes.append("pdf_analysis")
        
        # Author-related queries
        if entities.get("author"):
            if "all" in query_lower:
                routes.append("author_collection")
            elif not routes:  # If no other specific routes, search for author's papers
                routes.append("graph_analysis")
        
        return routes if routes else ["vector_search"]

    async def _calculate_routing_confidence(self, query: str, entities: Dict[str, Any], routes: List[str], model) -> float:
        """Calculate confidence in routing decisions"""
        base_confidence = 0.5
        
        # Boost confidence for clear patterns
        if entities.get("author"):
            base_confidence += 0.2
        if entities.get("year"):
            base_confidence += 0.1
        if len(routes) > 1:
            base_confidence += 0.1  # Multiple routes often indicate comprehensive analysis
        
        # AI confidence assessment
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Rate the clarity and specificity of this academic query on a scale of 0.0 to 1.0. Consider entity clarity, query intent, and information sufficiency. Return only a number."),
                ("user", "Query: {query}\nEntities: {entities}\nSelected routes: {routes}")
            ])
            
            chain = prompt | model | StrOutputParser()
            result = await chain.ainvoke({"query": query, "entities": str(entities), "routes": str(routes)})
            
            try:
                ai_confidence = float(result.strip())
                return min(1.0, (base_confidence + ai_confidence) / 2)
            except:
                return base_confidence
        except:
            return base_confidence

    def _needs_disambiguation(self, entities: Dict[str, Any], routes: List[str]) -> bool:
        """Check if paper disambiguation is needed before routing"""
        # Need disambiguation if we have author but need specific papers for PDF/graph analysis
        if entities.get("author") and ("pdf_analysis" in routes or "graph_analysis" in routes):
            return not entities.get("year") and not entities.get("title")
        return False

    # Workflow Routing Methods
    def _intelligent_multi_routing(self, state: QueryState) -> str:
        """Determine initial routing decision"""
        if state.next_action == ActionType.ASK_CLARIFICATION:
            return "clarification"
        elif self._needs_disambiguation(state.extracted_entities, state.required_routes):
            return "disambiguate"
        else:
            return "coordinate"

    def _coordinate_routes(self, state: QueryState) -> str:
        """Coordinate execution of required routes"""
        remaining_routes = [route for route in state.required_routes if route not in state.completed_routes]
        
        if not remaining_routes:
            return "generate_response"
        
        # Return the next route to process
        next_route = remaining_routes[0]
        if next_route == "graph_analysis":
            return "graph_analysis"
        elif next_route == "vector_search":
            return "vector_search"
        elif next_route == "pdf_analysis":
            return "pdf_analysis"
        elif next_route == "author_collection":
            return "author_collection"
        else:
            return "generate_response"

    # Language Processing Methods
    async def _detect_language(self, query: str) -> str:
        """Detect the language of the input query"""
        try:
            # Simple heuristic-based language detection
            chinese_chars = len([c for c in query if '\u4e00' <= c <= '\u9fff'])
            total_chars = len([c for c in query if c.isalpha() or '\u4e00' <= c <= '\u9fff'])
            
            if total_chars > 0 and chinese_chars / total_chars > 0.3:
                return "zh"
            else:
                return "en"
        except:
            return "en"  # Default fallback

    async def _translate_to_english(self, query: str, source_lang: str) -> str:
        """Translate query to English for internal processing"""
        try:
            model = self.llm_manager.get_agent_model("language_processor")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"Translate the following {source_lang} academic query to English. Preserve academic terminology and concepts exactly. Return only the translation."),
                ("user", "{query}")
            ])
            
            chain = prompt | model | StrOutputParser()
            result = await chain.ainvoke({"query": query})
            return result.strip()
            
        except Exception as e:
            logger.warning(f"Translation failed: {e}, using original query")
            return query

    async def _translate_response(self, response: str, target_lang: str) -> str:
        """Translate response back to user's language"""
        if target_lang == "en":
            return response
            
        try:
            model = self.llm_manager.get_agent_model("language_processor")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"Translate the following English academic response to {target_lang}. Preserve academic terminology and maintain the structure. Return only the translation."),
                ("user", "{response}")
            ])
            
            chain = prompt | model | StrOutputParser()
            result = await chain.ainvoke({"response": response})
            return result.strip()
            
        except Exception as e:
            logger.warning(f"Response translation failed: {e}, returning English response")
            return response

    # Helper methods for entity extraction and analysis
    async def _enhanced_entity_extraction(self, query: str) -> Dict[str, Any]:
        """Enhanced entity extraction using LLM"""
        try:
            model = self.llm_manager.get_agent_model("query_analyzer")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract entities from academic queries. Return a JSON object with these fields:
- author: Author name if mentioned (normalize to "First Last" format)
- year: Publication year if mentioned (as integer)
- title: Paper title if mentioned
- keywords: List of relevant academic keywords
- query_intent: One of ["find_papers", "citation_analysis", "author_search", "content_search"]

Example:
Query: "Michael Porter 1980 competitive strategy"
Response: {{"author": "Michael Porter", "year": 1980, "keywords": ["competitive", "strategy"], "query_intent": "find_papers"}}"""),
                ("user", "Query: {query}")
            ])
            
            chain = prompt | model | StrOutputParser()
            result = await chain.ainvoke({"query": query})
            
            # Parse JSON response
            try:
                entities = json.loads(result)
                # Add basic regex extraction as fallback
                basic_entities = self._basic_entity_extraction(query)
                
                # Merge results, preferring LLM results
                for key, value in basic_entities.items():
                    if key not in entities or not entities[key]:
                        entities[key] = value
                
                return entities
            except json.JSONDecodeError:
                # Fallback to basic extraction
                return self._basic_entity_extraction(query)
                
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}, falling back to basic extraction")
            return self._basic_entity_extraction(query)

    def _basic_entity_extraction(self, query: str) -> Dict[str, Any]:
        """Basic regex-based entity extraction as fallback"""
        entities = {}
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        if year_match:
            entities["year"] = int(year_match.group())
        
        # Extract potential author names
        potential_authors = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        common_words = {'The', 'In', 'Of', 'And', 'For', 'With', 'By', 'To', 'From', 'What', 'Who'}
        authors = [name for name in potential_authors if name not in common_words]
        
        if authors:
            entities["author"] = " ".join(authors)
        
        # Extract keywords
        keywords = re.findall(r'\b\w+\b', query.lower())
        entities["keywords"] = [kw for kw in keywords if len(kw) > 2]
        
        return entities

    # Routing functions
    def _route_after_analysis(self, state: QueryState) -> str:
        """Route after query analysis"""
        if state.next_action == ActionType.ASK_CLARIFICATION:
            return "clarification"
        
        entities = state.extracted_entities
        
        if "author" in entities:
            return "disambiguate"
        elif state.query_type == QueryType.CITATION_ANALYSIS:
            return "direct_citation"
        else:
            return "vector_search"

    def _route_after_disambiguation(self, state: QueryState) -> str:
        """Route after paper disambiguation"""
        if state.next_action == ActionType.ASK_CLARIFICATION:
            return "clarification"
        elif state.query_type == QueryType.CITATION_ANALYSIS and state.selected_papers:
            return "citation_analysis"
        elif state.selected_papers:
            return "citation_analysis"
        else:
            return "vector_search"

    # Utility methods
    def _classify_query_type(self, query: str, entities: Dict) -> QueryType:
        """Classify query type based on content and entities"""
        query_lower = query.lower()
        
        citation_keywords = ['cit', 'refer', 'mention', 'quote', 'support', 'refute', 'extend']
        if any(keyword in query_lower for keyword in citation_keywords):
            return QueryType.CITATION_ANALYSIS
        
        if "author" in entities:
            return QueryType.AUTHOR_PAPERS
        
        return QueryType.PAPER_SEARCH

    def _calculate_query_confidence(self, query: str, entities: Dict) -> float:
        """Calculate confidence in query understanding"""
        score = 0.5  # Base score
        
        if entities.get("author"):
            score += 0.2
        if entities.get("year"):
            score += 0.2
        if entities.get("title"):
            score += 0.2
        if len(entities.get("keywords", [])) > 2:
            score += 0.1
        
        return min(1.0, score)

    def _determine_citation_direction(self, query: str) -> str:
        """Determine citation analysis direction"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cited by', 'references to', 'who cites']):
            return "incoming"
        elif any(word in query_lower for word in ['cites', 'references', 'mentions']):
            return "outgoing"
        else:
            return "both"

    def _determine_clarification_issue(self, state: QueryState) -> str:
        """Determine what clarification is needed"""
        if not state.candidate_papers and "author" in state.extracted_entities:
            return f"No papers found for author '{state.extracted_entities['author']}'"
        elif len(state.candidate_papers) > 5:
            return f"Too many papers ({len(state.candidate_papers)}) found, need more specific criteria"
        elif not state.extracted_entities.get("author") and not state.extracted_entities.get("title"):
            return "Need more specific information about the paper or author"
        else:
            return "Query unclear or ambiguous"

    async def _generate_clarification_response(self, state: QueryState) -> str:
        """Generate response for clarification questions"""
        if state.clarification_question:
            return state.clarification_question
        else:
            return "Could you provide more specific information about your query?"

    async def _generate_content_response(self, state: QueryState) -> str:
        """Generate main content response"""
        model = self.llm_manager.get_agent_model("response_generator")
        
        # Prepare context for response generation from all routes
        context = {
            "query": state.processed_query,
            "original_language": state.user_language,
            "translation_used": state.translation_used,
            "route_results": state.route_results,
            "completed_routes": state.completed_routes,
            "papers": state.candidate_papers,
            "citations": state.citation_relationships,
            "vector_results": state.vector_results,
            "author_papers": state.author_papers,
            "pdf_analysis": state.pdf_content_analysis,
            "errors": state.errors,
            "warnings": state.warnings
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a comprehensive response for an academic paper query using multiple data sources.

Data Sources Available:
- Graph Analysis: Citation relationships, author citation reasons
- Vector Search: Concept definitions, semantic content
- PDF Analysis: Document content analysis
- Author Collection: Comprehensive author work analysis

Guidelines:
- Synthesize information from all available data sources
- Be specific and informative
- Include paper titles, authors, and years when available
- Explain relationships and concepts clearly
- Structure response logically (definitions first, then analysis, then relationships)
- If there are issues or limitations, mention them
- Use bullet points and sections for organization
- Maintain academic rigor and precision"""),
            ("user", "Generate comprehensive response for query: {query}\n\nMulti-source context: {context}")
        ])
        
        chain = prompt | model | StrOutputParser()
        response = await chain.ainvoke({
            "query": state.processed_query,
            "context": json.dumps(context, indent=2)
        })
        
        return response.strip()

    # PDF and Author Analysis Methods
    def _find_papers_from_entities(self, entities: Dict[str, Any]) -> List[Dict]:
        """Find papers based on extracted entities"""
        papers = []
        
        if entities.get("author"):
            author_papers = self.author_index.find_papers_by_author(entities["author"], exact_match=False)
            
            # Filter by year if specified
            if entities.get("year"):
                author_papers = [p for p in author_papers if p.get("year") == entities["year"]]
            
            papers.extend(author_papers)
        
        return papers

    async def _analyze_pdfs_with_ai(self, pdf_paths: List[str], query: str, model) -> str:
        """Analyze PDF content using AI with large context"""
        try:
            # For now, return a placeholder - PDF text extraction would be implemented here
            # This would involve:
            # 1. Extract text from PDFs using PyMuPDF or similar
            # 2. Send to large context model for analysis
            # 3. Return comprehensive analysis
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert academic document analyzer. Analyze the provided documents and answer the user's question comprehensively.

Guidelines:
- Provide detailed, structured analysis
- Focus on key concepts, methodologies, and findings
- Compare documents if multiple are provided
- Cite specific sections when relevant
- Maintain academic rigor"""),
                ("user", f"Query: {query}\n\nAnalyze the documents at these paths: {pdf_paths}\n\nNote: Full text extraction would be implemented in production.")
            ])
            
            chain = prompt | model | StrOutputParser()
            result = await chain.ainvoke({"query": query, "paths": str(pdf_paths)})
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            return f"PDF analysis failed: {str(e)}"

    async def _analyze_author_collection(self, author_name: str, papers: List[Dict], pdf_paths: List[str], query: str, model) -> str:
        """Analyze an author's complete body of work"""
        try:
            # Prepare papers metadata
            papers_info = []
            for paper in papers:
                papers_info.append({
                    "title": paper.get("title", "Unknown"),
                    "year": paper.get("year", "Unknown"),
                    "journal": paper.get("journal", "Unknown")
                })
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an expert academic researcher analyzing the complete body of work for {author_name}.

Provide a comprehensive analysis covering:
1. Overall research themes and evolution
2. Key contributions and methodologies
3. Impact and influence in the field
4. Chronological development of ideas
5. Answer to the specific user query

Available papers: {len(papers)} total
Available full texts: {len(pdf_paths)} PDFs

Be thorough and scholarly in your analysis."""),
                ("user", "Query: {query}\n\nAuthor: {author_name}\n\nPapers metadata: {papers_info}\n\nPDF paths: {pdf_paths}")
            ])
            
            chain = prompt | model | StrOutputParser()
            result = await chain.ainvoke({
                "query": query, 
                "author_name": author_name, 
                "papers_info": json.dumps(papers_info, indent=2),
                "pdf_paths": str(pdf_paths)
            })
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Author collection analysis failed: {e}")
            return f"Author collection analysis failed: {str(e)}"

    async def _analyze_author_metadata_only(self, author_name: str, papers: List[Dict], query: str) -> str:
        """Analyze author based on metadata only when PDFs not available"""
        try:
            model = self.llm_manager.get_agent_model("query_analyzer")
            
            papers_info = []
            for paper in papers:
                papers_info.append({
                    "title": paper.get("title", "Unknown"),
                    "year": paper.get("year", "Unknown"), 
                    "journal": paper.get("journal", "Unknown")
                })
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""Analyze {author_name}'s research profile based on available metadata.

Provide insights on:
1. Research areas and themes (inferred from titles)
2. Publication timeline and productivity
3. Journal preferences and academic scope
4. Answer to the user's specific query

Note: Analysis based on metadata only - full text not available."""),
                ("user", "Query: {query}\n\nAuthor: {author_name}\n\nPapers: {papers_info}")
            ])
            
            chain = prompt | model | StrOutputParser()
            result = await chain.ainvoke({
                "query": query,
                "author_name": author_name,
                "papers_info": json.dumps(papers_info, indent=2)
            })
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Author metadata analysis failed: {e}")
            return f"Author metadata analysis failed: {str(e)}"

    def _calculate_final_confidence(self, state: QueryState) -> float:
        """Calculate final confidence score"""
        score = 0.5
        
        # Boost for successful operations
        if state.selected_papers:
            score += 0.2
        if state.citation_relationships:
            score += 0.2
        if state.vector_results:
            score += 0.1
        
        # Penalize for errors and warnings
        score -= len(state.errors) * 0.2
        score -= len(state.warnings) * 0.1
        
        return max(0.0, min(1.0, score))

    # Main query interface
    async def query(self, 
                   user_query: str, 
                   thread_id: str = "default",
                   user_id: str = "default") -> Dict[str, Any]:
        """
        Main query interface
        
        Args:
            user_query: User's natural language question in any supported language
            thread_id: Thread ID for conversation memory
            user_id: User ID for personalization
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Initialize state
            initial_state = QueryState(
                original_query=user_query,
                thread_id=thread_id,
                user_id=user_id
            )
            
            # Run the workflow
            config = {"configurable": {"thread_id": thread_id}}
            result = await self.app.ainvoke(initial_state, config=config)
            
            # Handle both QueryState object and dict responses
            if hasattr(result, 'final_response'):
                # QueryState object
                return {
                    "response": result.final_response,
                    "confidence": result.confidence_score,
                    "query_type": result.query_type.value if hasattr(result.query_type, 'value') else str(result.query_type),
                    "action": result.next_action.value if hasattr(result.next_action, 'value') else str(result.next_action),
                    "user_language": result.user_language,
                    "response_language": result.response_language,
                    "translation_used": result.translation_used,
                    "required_routes": result.required_routes,
                    "completed_routes": result.completed_routes,
                    "route_results": result.route_results,
                    "candidate_papers": result.candidate_papers,
                    "disambiguation_needed": result.disambiguation_needed,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "debug_messages": result.debug_messages
                }
            else:
                # Dict response (fallback)
                return {
                    "response": result.get("final_response", "No response generated"),
                    "confidence": result.get("confidence_score", 0.0),
                    "query_type": result.get("query_type", "unknown"),
                    "action": result.get("next_action", "unknown"),
                    "user_language": result.get("user_language", "unknown"),
                    "response_language": result.get("response_language", "unknown"),
                    "candidate_papers": result.get("candidate_papers", []),
                    "disambiguation_needed": result.get("disambiguation_needed", False),
                    "errors": result.get("errors", []),
                    "warnings": result.get("warnings", []),
                    "debug_messages": result.get("debug_messages", [])
                }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "response": f"I encountered a system error: {str(e)}",
                "confidence": 0.0,
                "query_type": "error",
                "action": "end_with_error",
                "errors": [str(e)],
                "warnings": [],
                "debug_messages": []
            }

    # Synchronous wrapper for backwards compatibility
    def query_sync(self, user_query: str, thread_id: str = "default", user_id: str = "default") -> Dict[str, Any]:
        """Synchronous wrapper for the async query method"""
        return asyncio.run(self.query(user_query, thread_id, user_id))

# Example usage
if __name__ == "__main__":
    # This would be used for testing
    pass 