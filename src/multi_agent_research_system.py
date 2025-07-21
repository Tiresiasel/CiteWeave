"""
LangGraph-based Multi-Agent Research System
Modern implementation using LangGraph for agent orchestration
"""

import json
import logging
import uuid
import os
from typing import Dict, List, Optional, Tuple, Any, Union, TypedDict, Annotated
from dataclasses import dataclass, asdict
from enum import Enum
import time
import operator
import re

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.prebuilt import ToolNode
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.tools import tool
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LangGraph/LangChain not available: {e}")
    LANGGRAPH_AVAILABLE = False
    # Mock classes for when LangGraph is not available
    class StateGraph:
        def __init__(self, *args, **kwargs): pass
        def add_node(self, *args, **kwargs): pass
        def add_edge(self, *args, **kwargs): pass
        def compile(self): return lambda x: x
    class ToolNode:
        def __init__(self, *args, **kwargs): pass
    def tool(func): return func
    START = END = None

from query_db_agent import QueryDBAgent
from vector_indexer import VectorIndexer
from config_manager import ConfigManager

# --- Sophisticated Structured Logging Setup ---
logger = logging.getLogger("CiteWeave")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def log_event(agent, event_type, data, level=logging.INFO, request_id=None):
    log_entry = {
        "agent": agent,
        "event": event_type,
        "data": data,
        "request_id": request_id or "N/A"
    }
    logger.log(level, json.dumps(log_entry, ensure_ascii=False, default=str))

class ModelConfigManager:
    """Centralized model configuration management"""
    
    def __init__(self, config_path: str = "config/model_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        log_event("ModelConfigManager", "config_loaded", {"config_path": config_path, "config": self.config}, level=logging.DEBUG)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration as fallback"""
        return {
            "llm": {
                "agents": {
                    "query_analyzer": {
                        "provider": "openai",
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.1,
                        "max_tokens": 1000
                    },
                    "response_generator": {
                        "provider": "openai", 
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.2,
                        "max_tokens": 2000
                    },
                    "paper_disambiguator": {
                        "provider": "openai",
                        "model": "gpt-3.5-turbo", 
                        "temperature": 0.1,
                        "max_tokens": 1000
                    },
                    "language_processor": {
                        "provider": "openai",
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.1,
                        "max_tokens": 1000
                    }
                },
                "default": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.1,
                    "max_tokens": 1500
                }
            },
            "conversation": {
                "summarization_threshold": 10
            }
        }
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent"""
        agents_config = self.config.get("llm", {}).get("agents", {})
        agent_config = agents_config.get(agent_name)
        
        if not agent_config:
            logger.warning(f"No config found for agent '{agent_name}', using default")
            agent_config = self.config.get("llm", {}).get("default", {})
        
        log_event("ModelConfigManager", "agent_config_retrieved", {"agent_name": agent_name, "config": agent_config}, level=logging.DEBUG)
        return agent_config
    
    def get_model_instance(self, agent_name: str) -> Any:
        """Create a model instance for the specified agent"""
        config = self.get_agent_config(agent_name)
        
        try:
            if config.get("provider") == "openai" and LANGGRAPH_AVAILABLE:
                model_instance = ChatOpenAI(
                    model=config.get("model", "gpt-3.5-turbo"),
                    temperature=config.get("temperature", 0.1),
                    max_tokens=config.get("max_tokens", 1000)
                )
                log_event("ModelConfigManager", "model_instance_created", {"agent_name": agent_name, "model": config.get("model"), "provider": config.get("provider")}, level=logging.DEBUG)
                return model_instance
            else:
                logger.warning(f"Unsupported provider or LangChain not available: {config.get('provider')}")
                return None
        except Exception as e:
            logger.error(f"Failed to create model instance for {agent_name}: {e}")
            return None
    
    def get_conversation_config(self) -> Dict[str, Any]:
        """Get conversation-related configuration"""
        return self.config.get("conversation", {"summarization_threshold": 10})
    
    def get_pdf_processing_config(self) -> Dict[str, Any]:
        """Get PDF processing configuration"""
        return self.config.get("pdf_processing", {})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return self.config.get("embedding", {})

class IntelligentEntityExtractor:
    """LLM-based entity extraction for academic queries"""
    
    def __init__(self, model_config_manager: ModelConfigManager):
        self.model_config_manager = model_config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_entities(self, question: str, request_id=None) -> Dict[str, Any]:
        """Use LLM to intelligently extract entities from the question"""
        log_event("IntelligentEntityExtractor", "extract_start", {"question": question}, level=logging.DEBUG, request_id=request_id)
        
        # Get configured model for entity extraction
        entity_model = self.model_config_manager.get_model_instance("language_processor")
        
        if not entity_model:
            log_event("IntelligentEntityExtractor", "fallback_used", {"reason": "no_model_available"}, level=logging.WARNING, request_id=request_id)
            return self._fallback_entity_extraction(question)
        
        system_prompt = """You are an expert entity extractor for academic research queries. Your job is to identify and extract key entities from research questions.

ENTITY TYPES TO EXTRACT:
- author_names: Names of researchers, scholars, authors (e.g., "Porter", "Rivkin", "Smith")
- paper_titles: Titles or partial titles of academic papers
- concepts: Research concepts, theories, frameworks (e.g., "competitive strategy", "innovation")
- institutions: Universities, organizations, companies
- years: Publication years or time periods
- journals: Journal names or conference names

EXTRACTION RULES:
1. For author names: Extract surnames and full names mentioned in the context of academic work
2. For papers: Extract any quoted titles or descriptive phrases about specific papers
3. For concepts: Extract academic terms, theories, or research topics
4. Be conservative: Only extract entities that are clearly relevant to academic research
5. Handle multiple entities of the same type
6. Distinguish between citing and cited entities

EXAMPLES:

Question: "The paper cite Rivkin, what is the citation context?"
Extract: {
  "author_names": ["Rivkin"],
  "paper_titles": [],
  "concepts": ["citation context"],
  "primary_entity": "Rivkin",
  "primary_entity_type": "author",
  "query_focus": "reverse_citation"
}

Question: "What papers cite Porter's competitive strategy framework?"
Extract: {
  "author_names": ["Porter"],
  "paper_titles": [],
  "concepts": ["competitive strategy framework"],
  "primary_entity": "Porter",
  "primary_entity_type": "author",
  "query_focus": "reverse_citation"
}

Question: "Find the Innovation paper by Johnson from 2020"
Extract: {
  "author_names": ["Johnson"],
  "paper_titles": ["Innovation paper"],
  "concepts": ["Innovation"],
  "years": ["2020"],
  "primary_entity": "Innovation paper by Johnson",
  "primary_entity_type": "paper",
  "query_focus": "paper_search"
}

Return ONLY a JSON object with the extracted entities."""

        user_prompt = f"Extract entities from this research question: {question}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = entity_model.invoke(messages)
            
            # Parse the LLM's JSON response
            try:
                response_content = response.content.strip()
                
                # Handle JSON wrapped in markdown
                if "```json" in response_content:
                    json_start = response_content.find("```json") + 7
                    json_end = response_content.find("```", json_start)
                    json_text = response_content[json_start:json_end].strip()
                elif "{" in response_content and "}" in response_content:
                    json_start = response_content.find("{")
                    json_end = response_content.rfind("}") + 1
                    json_text = response_content[json_start:json_end]
                else:
                    raise ValueError("No JSON found in response")
                
                entities = json.loads(json_text)
                
                # Validate and clean the entities
                cleaned_entities = self._validate_entities(entities)
                
                log_event("IntelligentEntityExtractor", "extract_success", {"entities": cleaned_entities}, level=logging.INFO, request_id=request_id)
                return cleaned_entities
                
            except (json.JSONDecodeError, ValueError, KeyError) as parse_error:
                self.logger.warning(f"Failed to parse LLM entity extraction: {parse_error}")
                log_event("IntelligentEntityExtractor", "parse_error", {"error": str(parse_error), "response": response.content[:200]}, level=logging.WARNING, request_id=request_id)
                return self._fallback_entity_extraction(question)
                
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            log_event("IntelligentEntityExtractor", "extract_error", {"error": str(e)}, level=logging.ERROR, request_id=request_id)
            return self._fallback_entity_extraction(question)
    
    def _validate_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted entities"""
        validated = {
            "author_names": entities.get("author_names", []),
            "paper_titles": entities.get("paper_titles", []),
            "concepts": entities.get("concepts", []),
            "institutions": entities.get("institutions", []),
            "years": entities.get("years", []),
            "journals": entities.get("journals", []),
            "primary_entity": entities.get("primary_entity", ""),
            "primary_entity_type": entities.get("primary_entity_type", "concept"),
            "query_focus": entities.get("query_focus", "concept_search")
        }
        
        # Ensure lists are actually lists
        for key in ["author_names", "paper_titles", "concepts", "institutions", "years", "journals"]:
            if not isinstance(validated[key], list):
                validated[key] = [validated[key]] if validated[key] else []
        
        # Clean string values
        for key in ["primary_entity", "primary_entity_type", "query_focus"]:
            if not isinstance(validated[key], str):
                validated[key] = str(validated[key])
        
        return validated
    
    def _fallback_entity_extraction(self, question: str) -> Dict[str, Any]:
        """Simple fallback entity extraction when LLM is not available"""
        question_lower = question.lower()
        
        # Basic patterns for common entities
        author_patterns = [
            r'\b([A-Z][a-z]+)\b',  # Capitalized words
            r'cite\s+([A-Za-z]+)',  # After "cite"
            r'by\s+([A-Z][a-z]+)',  # After "by"
        ]
        
        concept_patterns = [
            r'(citation context)',
            r'(competitive strategy)',
            r'(innovation)',
            r'(framework)',
            r'(theory)',
            r'(model)'
        ]
        
        # Extract potential authors
        author_names = []
        for pattern in author_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            author_names.extend(matches)
        
        # Filter common words
        common_words = {'The', 'What', 'Who', 'When', 'Where', 'How', 'Why', 'In', 'Of', 'And', 'For', 'With', 'By', 'To', 'From', 'Paper', 'Papers', 'Article', 'Articles'}
        author_names = [name for name in author_names if name not in common_words]
        
        # Extract concepts
        concepts = []
        for pattern in concept_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            concepts.extend(matches)
        
        # Determine primary entity
        primary_entity = ""
        primary_entity_type = "concept"
        
        if author_names:
            primary_entity = author_names[0]
            primary_entity_type = "author"
        elif concepts:
            primary_entity = concepts[0]
            primary_entity_type = "concept"
        else:
            primary_entity = question
            primary_entity_type = "concept"
        
        # Determine query focus
        query_focus = "concept_search"
        if "cite" in question_lower:
            query_focus = "reverse_citation"
        elif "paper" in question_lower or "article" in question_lower:
            query_focus = "paper_search"
        elif author_names:
            query_focus = "author_search"
        
        return {
            "author_names": list(set(author_names)),
            "paper_titles": [],
            "concepts": list(set(concepts)),
            "institutions": [],
            "years": [],
            "journals": [],
            "primary_entity": primary_entity,
            "primary_entity_type": primary_entity_type,
            "query_focus": query_focus
        }

class QueryType(Enum):
    CITATION_ANALYSIS = "citation_analysis"
    REVERSE_CITATION_ANALYSIS = "reverse_citation_analysis"
    PAPER_SEARCH = "paper_search"
    AUTHOR_SEARCH = "author_search"
    CONCEPT_SEARCH = "concept_search"

class EntityType(Enum):
    AUTHOR = "author"
    PAPER = "paper"
    CONCEPT = "concept"

# LangGraph State Definition
class ResearchState(TypedDict):
    """State for the research workflow"""
    question: str
    query_intent: Optional[Dict[str, Any]]
    target_entity: Optional[Dict[str, Any]]
    clarification_needed: bool
    query_plan: Optional[Dict[str, Any]]
    collected_data: Optional[Dict[str, Any]]
    reflection_result: Optional[Dict[str, Any]]
    final_response: Optional[str]
    messages: Annotated[List[Any], operator.add]
    error: Optional[str]
    request_id: Optional[str]  # <-- for traceability

@dataclass
class QueryIntent:
    """结构化的查询意图"""
    query_type: QueryType
    target_entity: str
    entity_type: EntityType
    required_info: List[str]
    complexity: str
    original_question: str
 
@dataclass
class ClarificationRequest:
    """用户澄清请求"""
    clarification_type: str
    message: str
    options: List[Dict[str, Any]]
    user_selection: Optional[str] = None

@dataclass
class ReflectionResult:
    """反思结果"""
    sufficient: bool
    missing_aspects: List[str]
    next_queries: List[str]
    confidence: float
    collected_info: Dict[str, Any]

class QuestionAnalysisAgent:
    """问题分析智能体"""
    
    def __init__(self, model_config_manager: ModelConfigManager = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_config_manager = model_config_manager or ModelConfigManager()
    
    def analyze_question(self, question: str, request_id=None) -> QueryIntent:
        """分析用户问题，提取查询意图"""
        log_event("QuestionAnalysisAgent", "input", {"question": question}, level=logging.DEBUG, request_id=request_id)
        self.logger.info(f"Analyzing question: {question}")
        
        question_lower = question.lower()
        
        # 检测查询类型
        if "引用" in question and ("观点" in question or "内容" in question):
            # 如果是"引用X的文章"，这是反向引用分析
            query_type = QueryType.REVERSE_CITATION_ANALYSIS
            required_info = ["citing_papers", "citation_contexts", "cited_viewpoints"]
        elif "引用" in question and ("的文章" in question or "论文" in question):
            # 如果只是"引用X的文章"，也是反向引用分析
            query_type = QueryType.REVERSE_CITATION_ANALYSIS
            required_info = ["citing_papers", "citation_contexts"]
        elif "文章" in question or "论文" in question:
            query_type = QueryType.PAPER_SEARCH
            required_info = ["paper_metadata", "paper_content"]
        elif "作者" in question or "author" in question_lower:
            query_type = QueryType.AUTHOR_SEARCH
            required_info = ["author_papers", "author_info"]
        else:
            query_type = QueryType.CONCEPT_SEARCH
            required_info = ["relevant_content"]
        
        # 提取目标实体
        target_entity = self._extract_target_entity(question)
        
        # 确定实体类型
        if query_type in [QueryType.CITATION_ANALYSIS, QueryType.REVERSE_CITATION_ANALYSIS]:
            entity_type = EntityType.AUTHOR
        elif query_type == QueryType.PAPER_SEARCH:
            entity_type = EntityType.PAPER
        else:
            entity_type = EntityType.CONCEPT
        
        # 评估复杂度
        complexity = "high" if len(required_info) > 2 else "medium"
        
        intent = QueryIntent(
            query_type=query_type,
            target_entity=target_entity,
            entity_type=entity_type,
            required_info=required_info,
            complexity=complexity,
            original_question=question
        )
        
        log_event("QuestionAnalysisAgent", "output", asdict(intent), level=logging.INFO, request_id=request_id)
        return intent
    
    def _extract_target_entity(self, question: str) -> str:
        """提取目标实体名称"""
        # 简单的实体提取逻辑，可以后续优化
        question_lower = question.lower()
        
        # 查找常见模式
        if "波特" in question or "porter" in question_lower:
            return "porter"
        elif "引用" in question:
            # 尝试提取引用后的名字
            import re
            # 匹配中文和英文名字模式
            patterns = [
                r'引用([^\s的]{2,10})',
                r'([A-Za-z]+(?:\s+[A-Za-z]+)*)',
            ]
            for pattern in patterns:
                match = re.search(pattern, question)
                if match:
                    return match.group(1).strip()
        
        # 默认返回问题中的关键词
        words = question.split()
        for word in words:
            if len(word) > 2 and word not in ["的", "文章", "论文", "引用", "观点", "什么"]:
                return word
        
        return "unknown"

class FuzzyMatchingAgent:
    """模糊匹配智能体"""
    
    def __init__(self, query_agent: QueryDBAgent, model_config_manager: ModelConfigManager = None):
        self.query_agent = query_agent
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_config_manager = model_config_manager or ModelConfigManager()
    
    def find_matching_entities(self, entity_name: str, entity_type: EntityType, request_id=None) -> Tuple[List[Dict], float]:
        """查找匹配的实体"""
        log_event("FuzzyMatchingAgent", "input", {"entity_name": entity_name, "entity_type": entity_type.value}, level=logging.DEBUG, request_id=request_id)
        self.logger.info(f"Finding matches for {entity_type.value}: {entity_name}")
        
        if entity_type == EntityType.AUTHOR:
            return self._find_matching_authors(entity_name)
        elif entity_type == EntityType.PAPER:
            return self._find_matching_papers(entity_name)
        else:
            return [], 0.0
    
    def _find_matching_authors(self, author_name: str) -> Tuple[List[Dict], float]:
        """查找匹配的作者"""
        try:
            # 首先尝试图数据库查询
            results = self.query_agent.get_papers_id_by_author(author_name)
            
            if results.get("found"):
                matches = results.get("results", [])
                if len(matches) == 1:
                    confidence = 0.9
                elif len(matches) <= 3:
                    confidence = 0.7
                else:
                    confidence = 0.5
                
                # 格式化匹配结果
                formatted_matches = []
                for match in matches:
                    formatted_matches.append({
                        "id": match.get("paper_id"),
                        "name": ", ".join(match.get("authors", [])),
                        "title": match.get("title", ""),
                        "year": match.get("year", ""),
                        "match_score": match.get("match_score", 0.0)
                    })
                
                log_event("FuzzyMatchingAgent", "candidates", {"matches": matches, "confidence": confidence}, level=logging.INFO, request_id=request_id)
                return formatted_matches, confidence
            
            # 如果图数据库不可用，使用向量搜索回退
            self.logger.info("Graph DB not available, trying vector search for author")
            return self._find_authors_via_vector_search(author_name)
            
        except Exception as e:
            self.logger.error(f"Error finding authors: {e}")
            return self._find_authors_via_vector_search(author_name)
    
    def _find_authors_via_vector_search(self, author_name: str) -> Tuple[List[Dict], float]:
        """通过向量搜索查找作者"""
        try:
            from vector_indexer import VectorIndexer
            vector_indexer = VectorIndexer()
            search_results = vector_indexer.search_all_collections(
                query=author_name, 
                limit_per_collection=5
            )
            
            # 从搜索结果中提取作者相关的论文
            author_papers = {}
            for collection, results in search_results.items():
                for result in results:
                    authors = result.get("authors", [])
                    paper_id = result.get("paper_id")
                    title = result.get("title", "")
                    year = result.get("year", "")
                    
                    # 检查是否匹配作者名
                    if any(author_name.lower() in author.lower() for author in authors):
                        if paper_id not in author_papers:
                            author_papers[paper_id] = {
                                "id": paper_id,
                                "paper_id": paper_id,
                                "name": ", ".join(authors),
                                "authors": authors,
                                "title": title,
                                "year": year,
                                "match_score": result.get("score", 0.0)
                            }
            
            matches = list(author_papers.values())
            if len(matches) == 1:
                confidence = 0.8
            elif len(matches) <= 3:
                confidence = 0.6
            else:
                confidence = 0.4
            
            self.logger.info(f"Found {len(matches)} papers via vector search for {author_name}")
            return matches, confidence
            
        except Exception as e:
            self.logger.error(f"Error in vector search for authors: {e}")
            return [], 0.0
    
    def _find_matching_papers(self, paper_title: str) -> Tuple[List[Dict], float]:
        """查找匹配的论文"""
        try:
            results = self.query_agent.get_papers_id_by_title(paper_title)
            
            if not results.get("found"):
                return [], 0.0
            
            matches = results.get("results", [])
            if len(matches) == 1:
                confidence = 0.9
            elif len(matches) <= 3:
                confidence = 0.7
            else:
                confidence = 0.5
            
            return matches, confidence
            
        except Exception as e:
            self.logger.error(f"Error finding papers: {e}")
            return [], 0.0

class UserClarificationAgent:
    """用户澄清智能体"""
    
    def __init__(self, ambiguity_threshold: float = 0.8, model_config_manager: ModelConfigManager = None):
        self.ambiguity_threshold = ambiguity_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_config_manager = model_config_manager or ModelConfigManager()
    
    def needs_clarification(self, matches: List[Dict], confidence: float) -> bool:
        """判断是否需要用户澄清"""
        return confidence < self.ambiguity_threshold and len(matches) > 1
    
    def create_clarification_request(self, matches: List[Dict], entity_type: EntityType, request_id=None) -> ClarificationRequest:
        """创建澄清请求"""
        log_event("UserClarificationAgent", "clarification_needed", {"matches": matches, "entity_type": entity_type.value}, level=logging.INFO, request_id=request_id)
        if entity_type == EntityType.AUTHOR:
            clarification_type = "multiple_authors"
            message = f"Found {len(matches)} authors matching your query. Please select one:"
        else:
            clarification_type = "multiple_papers"
            message = f"Found {len(matches)} papers matching your query. Please select one:"
        
        # 格式化选项
        options = []
        for i, match in enumerate(matches):
            if entity_type == EntityType.AUTHOR:
                option = {
                    "id": str(i),
                    "name": match.get("name", "Unknown"),
                    "description": f"Paper: {match.get('title', '')} ({match.get('year', '')})",
                    "match_data": match
                }
            else:
                option = {
                    "id": str(i),
                    "title": match.get("title", "Unknown"),
                    "authors": match.get("authors", []),
                    "year": match.get("year", ""),
                    "match_data": match
                }
            options.append(option)
        
        log_event("UserClarificationAgent", "clarification_message", {"message": message, "options": options}, level=logging.INFO, request_id=request_id)
        return ClarificationRequest(
            clarification_type=clarification_type,
            message=message,
            options=options
        )
    
    def get_user_selection(self, clarification: ClarificationRequest) -> Optional[Dict]:
        """获取用户选择（在实际实现中，这会是一个交互式界面）"""
        print(f"\n{clarification.message}")
        
        for i, option in enumerate(clarification.options):
            if clarification.clarification_type == "multiple_authors":
                print(f"{i}: {option['name']} - {option['description']}")
            else:
                print(f"{i}: {option['title']} by {', '.join(option['authors'])} ({option['year']})")
        
        try:
            selection = input("\nPlease enter your choice (number): ").strip()
            selected_index = int(selection)
            
            if 0 <= selected_index < len(clarification.options):
                return clarification.options[selected_index]["match_data"]
            else:
                print("Invalid selection.")
                return None
                
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled.")
            return None

class QueryPlanningAgent:
    """基于Function Calling的智能查询规划智能体"""
    
    def __init__(self, query_agent: QueryDBAgent, vector_indexer: VectorIndexer, model_config_manager: ModelConfigManager = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.query_agent = query_agent
        self.vector_indexer = vector_indexer
        self.model_config_manager = model_config_manager or ModelConfigManager()
        
        # 定义可用的函数工具
        self.available_functions = self._define_function_tools()
    
    def _define_function_tools(self) -> List[Dict]:
        """Define available query function tools for LLM selection"""
        return [
            {
                "name": "get_papers_citing_paper",
                "description": "Find all papers that cite a specific paper. Returns paper IDs and metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string", "description": "Target paper ID"}
                    },
                    "required": ["paper_id"]
                }
            },
            {
                "name": "get_papers_cited_by_paper", 
                "description": "Find all papers cited by a specific paper. Returns cited paper IDs and metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string", "description": "Source paper ID"}
                    },
                    "required": ["paper_id"]
                }
            },
            {
                "name": "get_sentences_citing_paper",
                "description": "Get exact citation sentences with context. Returns specific quotes and viewpoints.",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "paper_id": {"type": "string", "description": "Cited paper ID"}
                    },
                    "required": ["paper_id"]
                }
            },
            {
                "name": "get_paragraphs_citing_paper",
                "description": "Get citation paragraphs with broader context. Returns detailed citation discussions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string", "description": "Cited paper ID"}
                    },
                    "required": ["paper_id"]
                }
            },
            {
                "name": "get_papers_id_by_author",
                "description": "Find papers by author name with fuzzy matching. Returns paper IDs and titles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "author_name": {"type": "string", "description": "Author name (partial match supported)"}
                    },
                    "required": ["author_name"]
                }
            },
            {
                "name": "get_papers_id_by_title",
                "description": "Find papers by title with fuzzy matching. Returns exact paper matches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Paper title (partial match supported)"}
                    },
                    "required": ["title"]
                }
            },
            {
                "name": "search_all_collections",
                "description": "Semantic search across all vector collections. Best for concept queries and content discovery.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit_per_collection": {"type": "integer", "description": "Max results per collection", "default": 10}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_relevant_sentences",
                "description": "Search sentences by semantic similarity. Best for finding specific claims or viewpoints.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_n": {"type": "integer", "description": "Max results", "default": 10}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_relevant_paragraphs",
                "description": "Search paragraphs by semantic similarity. Best for finding detailed discussions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_n": {"type": "integer", "description": "Max results", "default": 10}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_relevant_sections",
                "description": "Search sections by semantic similarity. Best for finding topical chapters or sections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_n": {"type": "integer", "description": "Max results", "default": 10}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def create_query_plan(self, intent: QueryIntent, target_entity: Dict, request_id=None) -> Dict[str, Any]:
        """使用Function Calling智能生成查询计划"""
        log_event("QueryPlanningAgent", "input", {"intent": asdict(intent), "target_entity": target_entity}, level=logging.DEBUG, request_id=request_id)
        self.logger.info(f"Using Function Calling to analyze: {intent.original_question}")
        
        # 构建system prompt，让LLM理解可用函数并选择合适的查询策略
        system_prompt = self._build_system_prompt()
        
        # 构建用户查询prompt
        user_prompt = self._build_user_prompt(intent, target_entity)
        
        # Always use Function Calling - no rule-based fallback
        try:
            response = self._call_llm_with_functions(system_prompt, user_prompt)
            plan = self._parse_llm_response_to_plan(response, intent, target_entity)
            
            # If no valid plan generated, return error plan
            if not plan.get("query_sequence"):
                plan = self._create_error_plan("No valid function calls generated by LLM")
                
        except Exception as e:
            self.logger.error(f"LLM Function Calling failed: {e}")
            plan = self._create_error_plan(f"Function calling failed: {str(e)}")
        
        log_event("QueryPlanningAgent", "plan", plan, level=logging.INFO, request_id=request_id)
        return plan
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM to understand available functions"""
        return """You are an expert academic research query analyzer. Your job is to understand the user's research intent and extract structured information.

QUERY TYPES:
- reverse_citation_analysis: "Who cites X?", "Papers citing X", "Citation context of X"
- citation_analysis: "What does X cite?", "References in X paper"  
- author_search: "Papers by author X", "X's publications"
- paper_search: "Find paper about Y", "Paper titled Z"
- concept_search: "What is X?", "Explain concept Y"

ENTITY EXTRACTION:
- For citation queries: Extract the AUTHOR NAME being cited (e.g., "Rivkin", "Porter", "Smith")
- For paper queries: Extract the PAPER TITLE or keywords
- For author queries: Extract the AUTHOR NAME
- For concept queries: Extract the CONCEPT term

EXAMPLES:

Query: "The paper cite Rivkin, what is the citation context?"
Analysis: {
  "query_type": "reverse_citation_analysis",
  "target_entity": "Rivkin", 
  "entity_type": "author",
  "reasoning": "User wants citation context for Rivkin - this is reverse citation analysis"
}

Query: "What papers cite Porter's competitive strategy?"
Analysis: {
  "query_type": "reverse_citation_analysis",
  "target_entity": "Porter",
  "entity_type": "author", 
  "reasoning": "Looking for papers that cite Porter's work"
}

Query: "What does the Innovation paper by Smith cite?"
Analysis: {
  "query_type": "citation_analysis",
  "target_entity": "Innovation paper by Smith",
  "entity_type": "paper",
  "reasoning": "User wants to know what this specific paper references"
}

Query: "Papers by Johnson on strategy"
Analysis: {
  "query_type": "author_search", 
  "target_entity": "Johnson",
  "entity_type": "author",
  "reasoning": "Looking for papers authored by Johnson"
}

INSTRUCTIONS:
1. Focus on the MAIN ACTION: citing, cited by, authored by, about topic
2. Extract the KEY ENTITY: author name, paper identifier, or concept
3. Choose the most specific query type that matches the user's intent
4. Always extract clean entity names (e.g., "Rivkin" not "Rivkin, what is the citation")

Return ONLY a JSON object with: query_type, target_entity, entity_type, reasoning"""

    def _build_user_prompt(self, intent: QueryIntent, target_entity: Dict) -> str:
        """Build user query prompt for LLM"""
        prompt_parts = []
        prompt_parts.append(f"User Question: {intent.original_question}")
        prompt_parts.append(f"Query Type: {intent.query_type.value}")
        prompt_parts.append(f"Target Entity: {intent.target_entity}")
        
        if target_entity:
            prompt_parts.append(f"Entity Info: {target_entity}")
        
        prompt_parts.append("\nSelect appropriate functions to answer this question. Choose 1-3 function calls in logical execution order.")
        
        return "\n".join(prompt_parts)

    def _call_llm_with_functions(self, system_prompt: str, user_prompt: str) -> Any:
        """Call LLM with function calling capability"""
        try:
            from enhanced_llm_manager import EnhancedLLMManager
            
            # Get the configured model for query planning
            model_config_manager = ModelConfigManager()
            planning_config = model_config_manager.get_agent_config("query_analyzer")
            
            # Use EnhancedLLMManager with configuration
            llm_manager = EnhancedLLMManager(config_path="config/model_config.json")
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Perform Function Calling with configured parameters
            response = llm_manager.call_with_tools(
                messages=messages,
                tools=self.available_functions,
                max_tokens=planning_config.get("max_tokens", 1000)
            )
            
            return response
                
        except Exception as e:
            self.logger.error(f"LLM Function Calling failed: {e}")
            return None

    def _parse_llm_response_to_plan(self, response: Any, intent: QueryIntent, target_entity: Dict) -> Dict[str, Any]:
        """Parse LLM response and convert to query plan"""
        plan = {
            "query_sequence": [],
            "fallback_strategies": [],
            "success_criteria": {"minimum_results": 1},
            "reasoning": "Function Calling based plan"
        }
        
        if not response:
            return self._create_error_plan("No response from LLM")
        
        try:
            # 解析function calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                step_counter = 1
                for tool_call in response.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # 确定数据库类型
                    if function_name.startswith('get_') or function_name.startswith('search_'):
                        if function_name in ['get_papers_citing_paper', 'get_papers_cited_by_paper', 
                                           'get_sentences_citing_paper', 'get_paragraphs_citing_paper',
                                           'get_papers_id_by_author', 'get_papers_id_by_title']:
                            database = "graph_db"
                        else:
                            database = "vector_db"
                        
                        plan["query_sequence"].append({
                            "step": step_counter,
                            "database": database,
                            "method": function_name,
                            "params": function_args,
                            "expected_result": self._get_expected_result_name(function_name),
                            "required": True,
                            "reasoning": f"LLM selected function: {function_name}"
                        })
                        step_counter += 1
                        
            # If no function calls found, return error
            if not plan["query_sequence"]:
                return self._create_error_plan("No function calls found in LLM response")
                
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return self._create_error_plan(f"Failed to parse LLM response: {str(e)}")
        
        return plan

    def _create_error_plan(self, error_message: str) -> Dict[str, Any]:
        """Create an error plan when function calling fails"""
        self.logger.error(f"Creating error plan: {error_message}")
        
        return {
            "query_sequence": [],
            "fallback_strategies": [],
            "success_criteria": {"minimum_results": 0},
            "reasoning": f"Error plan - {error_message}",
            "error": True,
            "error_message": error_message
        }

    def _analyze_information_requirements(self, intent: QueryIntent) -> Dict[str, Any]:
        """分析问题需要什么类型的信息"""
        question = intent.original_question.lower()
        requirements = {
            "needs_citing_papers": False,
            "needs_cited_papers": False, 
            "needs_citation_contexts": False,
            "needs_author_papers": False,
            "needs_content_search": False,
            "needs_viewpoints": False,
            "primary_focus": "unknown"
        }
        
        # 分析关键词模式
        if "引用" in question and ("的" in question and ("文章" in question or "论文" in question)):
            if "观点" in question or "内容" in question or "分别是什么" in question:
                requirements.update({
                    "needs_citing_papers": True,
                    "needs_citation_contexts": True,
                    "needs_viewpoints": True,
                    "primary_focus": "reverse_citation_with_content"
                })
            else:
                requirements.update({
                    "needs_citing_papers": True,
                    "primary_focus": "reverse_citation"
                })
        
        elif "被引用" in question or "引用了" in question:
            requirements.update({
                "needs_cited_papers": True,
                "primary_focus": "forward_citation"
            })
        
        elif ("作者" in question and ("文章" in question or "论文" in question)) or \
             (("的论文" in question or "的文章" in question) and not "引用" in question):
            requirements.update({
                "needs_author_papers": True,
                "primary_focus": "author_analysis"
            })
        
        elif "研究" in question or "内容" in question or "概念" in question:
            requirements.update({
                "needs_content_search": True,
                "primary_focus": "content_analysis"
            })
        
        self.logger.info(f"Information requirements analysis: {requirements}")
        return requirements
    
    def _analyze_target_entity(self, target_entity: Dict, intent: QueryIntent) -> Dict[str, Any]:
        """分析目标实体的可用信息"""
        analysis = {
            "has_paper_id": bool(target_entity.get("id") or target_entity.get("paper_id")),
            "has_authors": bool(target_entity.get("authors")),
            "has_title": bool(target_entity.get("title")),
            "entity_confidence": 0.0,
            "search_terms": []
        }
        
        # 计算实体信息的置信度
        if analysis["has_paper_id"]:
            analysis["entity_confidence"] += 0.5
        if analysis["has_authors"]:
            analysis["entity_confidence"] += 0.3
            analysis["search_terms"].extend(target_entity.get("authors", []))
        if analysis["has_title"]:
            analysis["entity_confidence"] += 0.2
            analysis["search_terms"].append(target_entity.get("title", ""))
        
        # 添加原始查询中的实体名称
        if intent.target_entity:
            analysis["search_terms"].append(intent.target_entity)
        
        self.logger.info(f"Target entity analysis: {analysis}")
        return analysis
    
    def _generate_query_strategy(self, requirements: Dict, entity_analysis: Dict, intent: QueryIntent) -> Dict[str, Any]:
        """生成查询策略"""
        strategy = {
            "primary_database": None,
            "primary_method": None,
            "secondary_queries": [],
            "fallback_strategy": None,
            "reasoning": ""
        }
        
        primary_focus = requirements["primary_focus"]
        
        if primary_focus == "reverse_citation_with_content":
            if entity_analysis["has_paper_id"]:
                strategy.update({
                    "primary_database": "graph_db",
                    "primary_method": "get_papers_citing_paper",
                    "secondary_queries": ["get_sentences_citing_paper"],
                    "fallback_strategy": "vector_search_by_author",
                    "reasoning": "目标有明确的paper_id，使用图数据库进行精确的反向引用查询，并获取引用上下文"
                })
            else:
                strategy.update({
                    "primary_database": "vector_db", 
                    "primary_method": "search_all_collections",
                    "secondary_queries": [],
                    "reasoning": "目标实体不明确，使用向量数据库进行语义搜索找到相关引用"
                })
        
        elif primary_focus == "reverse_citation":
            if entity_analysis["has_paper_id"]:
                strategy.update({
                    "primary_database": "graph_db",
                    "primary_method": "get_papers_citing_paper", 
                    "reasoning": "精确的paper_id可以直接查询引用关系"
                })
            else:
                strategy.update({
                    "primary_database": "vector_db",
                    "primary_method": "search_all_collections",
                    "reasoning": "实体不明确，需要先通过语义搜索找到相关论文"
                })
        
        elif primary_focus == "forward_citation":
            strategy.update({
                "primary_database": "graph_db",
                "primary_method": "get_papers_cited_by_paper",
                "reasoning": "查询论文引用的其他论文，图数据库最适合"
            })
        
        elif primary_focus == "author_analysis":
            strategy.update({
                "primary_database": "graph_db", 
                "primary_method": "get_papers_id_by_author",
                "fallback_strategy": "vector_search_by_author",
                "reasoning": "作者查询优先使用图数据库的精确匹配"
            })
        
        else:  # content_analysis or unknown
            strategy.update({
                "primary_database": "vector_db",
                "primary_method": "search_all_collections",
                "reasoning": "内容分析和未知类型查询使用向量数据库的语义搜索"
            })
        
        self.logger.info(f"Query strategy: {strategy}")
        return strategy
    
    def _build_execution_plan(self, strategy: Dict, intent: QueryIntent, target_entity: Dict) -> Dict[str, Any]:
        """构建具体的执行计划"""
        plan = {
            "query_sequence": [],
            "fallback_strategies": [],
            "success_criteria": {},
            "reasoning": strategy["reasoning"]
        }
        
        step_counter = 1
        
        # 主要查询
        if strategy["primary_database"] and strategy["primary_method"]:
            primary_params = self._build_method_params(
                strategy["primary_method"], 
                target_entity, 
                intent
            )
            
            plan["query_sequence"].append({
                "step": step_counter,
                "database": strategy["primary_database"],
                "method": strategy["primary_method"],
                "params": primary_params,
                "expected_result": self._get_expected_result_name(strategy["primary_method"]),
                "required": True,
                "reasoning": f"主要查询: {strategy['reasoning']}"
            })
            step_counter += 1
        
        # 次要查询
        for secondary_method in strategy.get("secondary_queries", []):
            secondary_params = self._build_method_params(
                secondary_method,
                target_entity, 
                intent
            )
            
            plan["query_sequence"].append({
                "step": step_counter,
                "database": strategy["primary_database"],  # 通常与主查询使用同一数据库
                "method": secondary_method,
                "params": secondary_params,
                "expected_result": self._get_expected_result_name(secondary_method),
                "required": False,
                "reasoning": f"补充查询: 获取更多上下文信息"
            })
            step_counter += 1
        
        # 回退策略
        if strategy.get("fallback_strategy"):
            plan["fallback_strategies"].append({
                "condition": "primary_query_failed",
                "database": "vector_db",
                "method": "search_all_collections",
                "params": {"query": intent.target_entity, "limit_per_collection": 10}
            })
        
        # 成功标准
        plan["success_criteria"] = {
            "minimum_results": 1,
            "required_fields": ["results"]
        }
        
        return plan
    
    def _build_method_params(self, method: str, target_entity: Dict, intent: QueryIntent) -> Dict:
        """根据方法构建参数"""
        if method in ["get_papers_citing_paper", "get_papers_cited_by_paper", "get_sentences_citing_paper", "get_paragraphs_citing_paper"]:
            return {"paper_id": target_entity.get("paper_id") or target_entity.get("id")}
        elif method == "get_papers_id_by_author":
            return {"author_name": intent.target_entity}
        elif method == "get_papers_id_by_title":
            return {"title": intent.target_entity}
        elif method == "search_all_collections":
            return {"query": intent.original_question, "limit_per_collection": 10}
        else:
            return {}
    
    def _get_expected_result_name(self, method: str) -> str:
        """根据方法获取期望结果名称"""
        result_mapping = {
            "get_papers_citing_paper": "citing_papers",
            "get_papers_cited_by_paper": "cited_papers",
            "get_sentences_citing_paper": "citation_contexts",
            "get_paragraphs_citing_paper": "citation_paragraphs",
            "get_papers_id_by_author": "author_papers",
            "get_papers_id_by_title": "paper_matches",
            "search_all_collections": "relevant_content",
            "search_relevant_sentences": "relevant_sentences",
            "search_relevant_paragraphs": "relevant_paragraphs",
            "search_relevant_sections": "relevant_sections"
        }
        return result_mapping.get(method, "query_results")

class DataRetrievalCoordinator:
    """数据检索协调器 - 根据查询计划精确执行数据检索"""
    
    def __init__(self, query_agent: QueryDBAgent, vector_indexer: VectorIndexer, model_config_manager: ModelConfigManager = None):
        self.query_agent = query_agent
        self.vector_indexer = vector_indexer
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_config_manager = model_config_manager or ModelConfigManager()
    
    def execute_query_plan(self, plan: Dict[str, Any], request_id=None) -> Dict[str, Any]:
        """Execute data retrieval based on query plan"""
        log_event("DataRetrievalCoordinator", "plan_start", plan, level=logging.INFO, request_id=request_id)
        # Check if this is an error plan
        if plan.get("error", False):
            self.logger.error(f"Cannot execute error plan: {plan.get('error_message', 'Unknown error')}")
            return {
                "plan_executed": plan,
                "results": {},
                "execution_log": [f"Error: {plan.get('error_message', 'Query planning failed')}"],
                "error": True
            }
        
        self.logger.info(f"Executing query plan with {len(plan.get('query_sequence', []))} steps")
        
        collected_data = {"plan_executed": plan, "results": {}, "execution_log": []}
        
        for query_step in plan["query_sequence"]:
            log_event("DataRetrievalCoordinator", "query_start", query_step, level=logging.DEBUG, request_id=request_id)
            step_num = query_step["step"]
            database = query_step["database"]
            method = query_step["method"]
            params = query_step["params"]
            expected_result = query_step["expected_result"]
            
            self.logger.info(f"Executing step {step_num}: {database}.{method}")
            
            try:
                if database == "graph_db":
                    result = self._execute_graph_query(method, params)
                elif database == "vector_db":
                    result = self._execute_vector_query(method, params)
                else:
                    result = {"error": f"Unknown database: {database}"}
                
                collected_data["results"][expected_result] = result
                collected_data["execution_log"].append({
                    "step": step_num,
                    "method": f"{database}.{method}",
                    "success": True,
                    "result_count": len(result) if isinstance(result, list) else 1
                })
                
                log_event("DataRetrievalCoordinator", "query_result", {"step": query_step["step"], "result": result}, level=logging.INFO, request_id=request_id)
                
                self.logger.info(f"Step {step_num} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Step {step_num} failed: {e}")
                collected_data["execution_log"].append({
                    "step": step_num,
                    "method": f"{database}.{method}",
                    "success": False,
                    "error": str(e)
                })
                
                log_event("DataRetrievalCoordinator", "query_error", {"step": query_step["step"], "error": str(e)}, level=logging.ERROR, request_id=request_id)
                
                # 尝试fallback策略
                if query_step.get("required", False):
                    fallback_result = self._try_fallback_strategies(plan["fallback_strategies"], expected_result)
                    if fallback_result:
                        collected_data["results"][expected_result] = fallback_result
                        self.logger.info(f"Fallback successful for step {step_num}")
        
        return collected_data
    
    def _execute_graph_query(self, method: str, params: Dict) -> Any:
        """执行图数据库查询"""
        if method == "get_papers_citing_paper":
            return self.query_agent.get_papers_citing_paper(params["paper_id"])
        elif method == "get_papers_cited_by_paper":
            return self.query_agent.get_papers_cited_by_paper(params["paper_id"])
        elif method == "get_sentences_citing_paper":
            return self.query_agent.get_sentences_citing_paper(params["paper_id"])
        elif method == "get_paragraphs_citing_paper":
            return self.query_agent.get_paragraphs_citing_paper(params["paper_id"])
        elif method == "get_papers_id_by_author":
            return self.query_agent.get_papers_id_by_author(params["author_name"])
        elif method == "get_papers_id_by_title":
            return self.query_agent.get_papers_id_by_title(params["title"])
        else:
            raise ValueError(f"Unknown graph query method: {method}")
    
    def _execute_vector_query(self, method: str, params: Dict) -> Any:
        """执行向量数据库查询"""
        if method == "search_all_collections":
            if self.vector_indexer:
                return self.vector_indexer.search_all_collections(
                    query=params["query"],
                    limit_per_collection=params.get("limit_per_collection", 5)
                )
            else:
                raise Exception("Vector indexer not available")
        elif method == "search_relevant_sentences":
            if self.vector_indexer:
                return self.vector_indexer.search_relevant_sentences(
                    query=params["query"],
                    top_n=params.get("top_n", 10)
                )
            else:
                raise Exception("Vector indexer not available")
        elif method == "search_relevant_paragraphs":
            if self.vector_indexer:
                return self.vector_indexer.search_relevant_paragraphs(
                    query=params["query"],
                    top_n=params.get("top_n", 10)
                )
            else:
                raise Exception("Vector indexer not available")
        elif method == "search_relevant_sections":
            if self.vector_indexer:
                return self.vector_indexer.search_relevant_sections(
                    query=params["query"],
                    top_n=params.get("top_n", 10)
                )
            else:
                raise Exception("Vector indexer not available")
        else:
            raise ValueError(f"Unknown vector query method: {method}")
    
    def _try_fallback_strategies(self, fallback_strategies: List[Dict], expected_result: str) -> Any:
        """尝试fallback策略"""
        for strategy in fallback_strategies:
            try:
                database = strategy["database"]
                method = strategy["method"]
                params = strategy["params"]
                
                if database == "vector_db":
                    return self._execute_vector_query(method, params)
                elif database == "graph_db":
                    return self._execute_graph_query(method, params)
                    
            except Exception as e:
                self.logger.warning(f"Fallback strategy failed: {e}")
                continue
        
        return None
    
    def _execute_reverse_citation_analysis(self, target_entity: Dict) -> Dict[str, Any]:
        """执行反向引用分析"""
        results = {}
        
        try:
            # 获取目标论文ID
            paper_id = target_entity.get("id") or target_entity.get("paper_id")
            
            if paper_id:
                # 首先尝试图数据库查询
                try:
                    citing_papers = self.query_agent.get_papers_citing_paper(paper_id)
                    citing_paragraphs = self.query_agent.get_paragraphs_citing_paper(paper_id)
                    citing_sentences = self.query_agent.get_sentences_citing_paper(paper_id)
                    
                    results["citing_papers"] = citing_papers
                    results["citing_paragraphs"] = citing_paragraphs
                    results["citing_sentences"] = citing_sentences
                    
                    self.logger.info(f"Found {len(citing_papers)} citing papers, {len(citing_paragraphs)} citing paragraphs, {len(citing_sentences)} citing sentences")
                    
                except Exception as graph_error:
                    self.logger.warning(f"Graph DB query failed: {graph_error}, trying vector search")
                    # 使用向量搜索作为回退
                    vector_results = self._find_citations_via_vector_search(target_entity)
                    results.update(vector_results)
            else:
                # 如果没有paper_id，直接使用向量搜索
                vector_results = self._find_citations_via_vector_search(target_entity)
                results.update(vector_results)
            
        except Exception as e:
            self.logger.error(f"Error in reverse citation analysis: {e}")
            results["error"] = str(e)
        
        return results
    
    def _find_citations_via_vector_search(self, target_entity: Dict) -> Dict[str, Any]:
        """通过向量搜索查找引用关系"""
        results = {"citing_papers": [], "citing_paragraphs": [], "citing_sentences": []}
        
        try:
            if not self.vector_indexer:
                from vector_indexer import VectorIndexer
                vector_indexer = VectorIndexer()
            else:
                vector_indexer = self.vector_indexer
            
            # 获取目标作者信息
            target_authors = target_entity.get("authors", [])
            target_name = target_entity.get("name", "")
            
            # 构建搜索查询
            search_queries = []
            for author in target_authors:
                if " " in author:
                    # 提取姓氏
                    last_name = author.split()[-1]
                    search_queries.append(last_name)
                search_queries.append(author)
            
            if target_name and target_name not in search_queries:
                search_queries.append(target_name)
            
            # 搜索引用
            all_citations = []
            citing_papers_dict = {}
            
            for query in search_queries:
                search_results = vector_indexer.search_all_collections(
                    query=query, 
                    limit_per_collection=10
                )
                
                # 分析citations集合中的结果
                for citation_result in search_results.get("citations", []):
                    citation_text = citation_result.get("text", "")
                    citation_context = citation_result.get("citation_context", "")
                    citing_paper_id = citation_result.get("paper_id")
                    citing_title = citation_result.get("title", "")
                    citing_authors = citation_result.get("authors", [])
                    citing_year = citation_result.get("year", "")
                    
                    # 检查是否是对目标作者的引用
                    if self._is_citation_to_target(citation_text, target_authors, target_name):
                        all_citations.append({
                            "sentence_text": citation_context,
                            "citation_text": citation_text,
                            "citing_paper_id": citing_paper_id
                        })
                        
                        # 记录引用论文
                        if citing_paper_id not in citing_papers_dict:
                            citing_papers_dict[citing_paper_id] = {
                                "paper_id": citing_paper_id,
                                "title": citing_title,
                                "authors": citing_authors,
                                "year": citing_year
                            }
            
            results["citing_papers"] = list(citing_papers_dict.values())
            results["citing_sentences"] = all_citations
            results["citing_paragraphs"] = []  # 可以后续从sentences推导
            
            self.logger.info(f"Vector search found {len(results['citing_papers'])} citing papers with {len(results['citing_sentences'])} citation instances")
            
        except Exception as e:
            self.logger.error(f"Error in vector search for citations: {e}")
        
        return results
    
    def _is_citation_to_target(self, citation_text: str, target_authors: List[str], target_name: str) -> bool:
        """判断引用是否指向目标作者"""
        citation_lower = citation_text.lower()
        
        # 检查目标作者名字
        for author in target_authors:
            if " " in author:
                last_name = author.split()[-1].lower()
                if last_name in citation_lower:
                    return True
            if author.lower() in citation_lower:
                return True
        
        # 检查目标名称
        if target_name and target_name.lower() in citation_lower:
            return True
        
        return False
    
    def _execute_citation_analysis(self, target_entity: Dict) -> Dict[str, Any]:
        """执行引用分析"""
        results = {}
        
        try:
            paper_id = target_entity.get("id") or target_entity.get("paper_id")
            
            if paper_id:
                # 获取该论文引用的所有文章
                cited_papers = self.query_agent.get_papers_cited_by_paper(paper_id)
                results["cited_papers"] = cited_papers
                
                self.logger.info(f"Found {len(cited_papers)} cited papers")
            
        except Exception as e:
            self.logger.error(f"Error in citation analysis: {e}")
            results["error"] = str(e)
        
        return results
    
    def _execute_basic_search(self, intent: QueryIntent, target_entity: Dict) -> Dict[str, Any]:
        """执行基础搜索"""
        results = {}
        
        try:
            # 使用向量搜索获取相关内容
            if self.vector_indexer:
                search_results = self.vector_indexer.search_all_collections(
                    query=intent.original_question,
                    limit_per_collection=10
                )
                results["vector_search_results"] = search_results
            else:
                self.logger.warning("Vector indexer not available for basic search")
                results["vector_search_results"] = {}
            
        except Exception as e:
            self.logger.error(f"Error in basic search: {e}")
            results["error"] = str(e)
        
        return results

class ReflectionAgent:
    """AI-driven reflection agent for information sufficiency evaluation"""
    
    def __init__(self, model_config_manager: ModelConfigManager = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_config_manager = model_config_manager or ModelConfigManager()
    
    def evaluate_information_sufficiency(self, intent: QueryIntent, collected_data: Dict[str, Any], request_id=None) -> ReflectionResult:
        """Use AI to evaluate if collected information is sufficient to answer the user's question"""
        log_event("ReflectionAgent", "reflection_input", {"intent": asdict(intent), "collected_data": collected_data}, level=logging.DEBUG, request_id=request_id)
        self.logger.info("Using AI to evaluate information sufficiency")
        
        try:
            # Use AI to evaluate information sufficiency
            evaluation_result = self._ai_evaluate_sufficiency(intent, collected_data)
            
            result = ReflectionResult(
                sufficient=evaluation_result.get("sufficient", False),
                missing_aspects=evaluation_result.get("missing_aspects", []),
                next_queries=evaluation_result.get("next_queries", []),
                confidence=evaluation_result.get("confidence", 0.0),
                collected_info=collected_data
            )
            
            log_event("ReflectionAgent", "reflection_result", asdict(result), level=logging.INFO, request_id=request_id)
            self.logger.info(f"AI reflection result: sufficient={result.sufficient}, confidence={result.confidence:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"AI evaluation failed: {e}")
            # Return a default result that allows proceeding with response generation
            return ReflectionResult(
                sufficient=True,  # Default to sufficient to allow response generation
                missing_aspects=[],
                next_queries=[],
                confidence=0.5,
                collected_info=collected_data
            )
    
    def _ai_evaluate_sufficiency(self, intent: QueryIntent, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to evaluate information sufficiency"""
        try:
            from enhanced_llm_manager import EnhancedLLMManager
            
            # Use configured model for reflection
            llm_manager = EnhancedLLMManager(config_path="config/model_config.json")
            reflection_config = self.model_config_manager.get_agent_config("query_analyzer")  # Use query_analyzer config for reflection
            
            # Build evaluation prompt
            system_prompt = self._build_evaluation_system_prompt()
            user_prompt = self._build_evaluation_user_prompt(intent, collected_data)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Define evaluation function for AI to call
            evaluation_function = {
                "name": "evaluate_information_sufficiency",
                "description": "Evaluate if the collected information is sufficient to answer the user's question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sufficient": {
                            "type": "boolean",
                            "description": "Whether the information is sufficient to answer the question"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score from 0.0 to 1.0"
                        },
                        "missing_aspects": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of missing information aspects if not sufficient"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of the evaluation decision"
                        }
                    },
                    "required": ["sufficient", "confidence", "reasoning"]
                }
            }
            
            response = llm_manager.call_with_tools(
                messages=messages,
                tools=[evaluation_function],
                max_tokens=reflection_config.get("max_tokens", 800)
            )
            
            # Parse the evaluation result
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                if tool_call.function.name == "evaluate_information_sufficiency":
                    evaluation_args = json.loads(tool_call.function.arguments)
                    
                    self.logger.info(f"AI evaluation reasoning: {evaluation_args.get('reasoning', 'No reasoning provided')}")
                    
                    return {
                        "sufficient": evaluation_args.get("sufficient", False),
                        "confidence": evaluation_args.get("confidence", 0.0),
                        "missing_aspects": evaluation_args.get("missing_aspects", []),
                        "next_queries": [],  # AI can suggest additional queries if needed
                        "reasoning": evaluation_args.get("reasoning", "")
                    }
            
            # Fallback if no proper function call
            return {"sufficient": True, "confidence": 0.5, "missing_aspects": [], "next_queries": []}
            
        except Exception as e:
            self.logger.error(f"AI evaluation error: {e}")
            raise e
    
    def _build_evaluation_system_prompt(self) -> str:
        """Build system prompt for AI evaluation"""
        return """You are an information sufficiency evaluator for an academic research system. Your task is to determine if the collected information is sufficient to answer the user's research question.

EVALUATION CRITERIA:
- For citation analysis: Check if we have citing papers and their contexts/viewpoints
- For author searches: Check if we have the author's papers and relevant metadata  
- For concept queries: Check if we have relevant content explaining the concept
- For specific paper queries: Check if we have the target paper and related information

CONFIDENCE SCORING:
- 1.0: Complete information, can fully answer the question
- 0.8: Good information, minor gaps but answerable
- 0.6: Adequate information, some important aspects missing
- 0.4: Limited information, significant gaps
- 0.2: Very little relevant information
- 0.0: No relevant information found

Call the evaluate_information_sufficiency function with your assessment."""

    def _build_evaluation_user_prompt(self, intent: QueryIntent, collected_data: Dict[str, Any]) -> str:
        """Build user prompt for AI evaluation"""
        prompt_parts = []
        prompt_parts.append(f"USER QUESTION: {intent.original_question}")
        prompt_parts.append(f"QUERY TYPE: {intent.query_type.value}")
        prompt_parts.append(f"TARGET ENTITY: {intent.target_entity}")
        
        # Summarize collected data
        results = collected_data.get("results", {})
        execution_log = collected_data.get("execution_log", [])
        
        prompt_parts.append(f"\nCOLLECTED DATA SUMMARY:")
        
        if results:
            for key, value in results.items():
                if isinstance(value, list):
                    prompt_parts.append(f"- {key}: {len(value)} items")
                    if len(value) > 0 and isinstance(value[0], dict):
                        # Show sample data structure
                        sample_keys = list(value[0].keys())[:3]
                        prompt_parts.append(f"  Sample fields: {sample_keys}")
                elif isinstance(value, dict):
                    prompt_parts.append(f"- {key}: {len(value)} entries")
                else:
                    prompt_parts.append(f"- {key}: {str(value)[:100]}...")
        else:
            prompt_parts.append("- No results collected")
        
        if execution_log:
            prompt_parts.append(f"\nEXECUTION LOG:")
            for log_entry in execution_log[-3:]:  # Show last 3 log entries
                prompt_parts.append(f"- {log_entry}")
        
        prompt_parts.append(f"\nPlease evaluate if this information is sufficient to answer the user's question.")
        
        return "\n".join(prompt_parts)

class ResponseGenerationAgent:
    """AI-driven response generation agent"""
    
    def __init__(self, model_config_manager: ModelConfigManager = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_config_manager = model_config_manager or ModelConfigManager()
    
    def generate_response(self, intent: QueryIntent, reflection_result: ReflectionResult, request_id=None) -> str:
        """Use AI to generate comprehensive final response"""
        log_event("ResponseGenerationAgent", "response_input", {"intent": asdict(intent), "reflection_result": asdict(reflection_result)}, level=logging.DEBUG, request_id=request_id)
        self.logger.info("Using AI to generate final response")
        
        data = reflection_result.collected_info
        
        # Check if there was an error in query planning or execution
        if data.get("error", False):
            error_message = data.get("execution_log", ["Unknown error"])[0]
            return f"❌ Sorry, I couldn't process your query due to an error: {error_message}\n\nPlease try rephrasing your question or check if the required data is available."
        
        try:
            # Use AI to generate response based on collected data
            response = self._ai_generate_response(intent, data, reflection_result)
            log_event("ResponseGenerationAgent", "response_generated", {"response": response}, level=logging.INFO, request_id=request_id)
            return response
            
        except Exception as e:
            self.logger.error(f"AI response generation failed: {e}")
            # Fallback to a simple summary
            return self._generate_fallback_response(intent, data)
    
    def _ai_generate_response(self, intent: QueryIntent, data: Dict[str, Any], reflection_result: ReflectionResult) -> str:
        """Use AI to generate comprehensive response"""
        try:
            from enhanced_llm_manager import EnhancedLLMManager
            
            # Use configured model for response generation
            llm_manager = EnhancedLLMManager(config_path="config/model_config.json")
            response_config = self.model_config_manager.get_agent_config("response_generator")
            
            # Build response generation prompt
            system_prompt = self._build_response_system_prompt()
            user_prompt = self._build_response_user_prompt(intent, data, reflection_result)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = llm_manager.generate_response(
                messages=messages,
                max_tokens=response_config.get("max_tokens", 2000),
                temperature=response_config.get("temperature", 0.2)
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"AI response generation error: {e}")
            raise e
    
    def _build_response_system_prompt(self) -> str:
        """Build system prompt for AI response generation"""
        return """You are an expert academic research assistant. Generate a comprehensive, well-structured response to the user's research question based on the collected data.

RESPONSE GUIDELINES:
- Use clear headings and bullet points for organization
- Include specific citations, paper titles, authors, and years when available
- For citation analysis: Present citing papers and their specific viewpoints/contexts
- For author searches: List papers with metadata and brief descriptions
- For concept queries: Provide definitions and related academic discussions
- Use academic tone but remain accessible
- Include relevant quotes or excerpts when available
- Organize information logically (e.g., chronological, by relevance)

FORMAT:
- Start with a brief summary answering the main question
- Use markdown formatting for structure
- Include numbered or bulleted lists for multiple items
- End with a conclusion if appropriate

Be thorough but concise. Focus on directly answering the user's question."""

    def _build_response_user_prompt(self, intent: QueryIntent, data: Dict[str, Any], reflection_result: ReflectionResult) -> str:
        """Build user prompt for AI response generation"""
        prompt_parts = []
        prompt_parts.append(f"USER QUESTION: {intent.original_question}")
        prompt_parts.append(f"QUERY TYPE: {intent.query_type.value}")
        
        # Include reflection assessment
        prompt_parts.append(f"\nINFORMATION ASSESSMENT:")
        prompt_parts.append(f"- Sufficiency: {'Sufficient' if reflection_result.sufficient else 'Insufficient'}")
        prompt_parts.append(f"- Confidence: {reflection_result.confidence:.2f}")
        if reflection_result.missing_aspects:
            prompt_parts.append(f"- Missing aspects: {', '.join(reflection_result.missing_aspects)}")
        
        # Include collected data
        results = data.get("results", {})
        prompt_parts.append(f"\nCOLLECTED DATA:")
        
        for key, value in results.items():
            prompt_parts.append(f"\n### {key.upper()}:")
            
            if isinstance(value, list) and len(value) > 0:
                prompt_parts.append(f"Found {len(value)} items:")
                
                # Show detailed data for first few items
                for i, item in enumerate(value[:5]):
                    if isinstance(item, dict):
                        prompt_parts.append(f"\nItem {i+1}:")
                        for item_key, item_value in item.items():
                            if isinstance(item_value, str) and len(item_value) > 200:
                                prompt_parts.append(f"  {item_key}: {item_value[:200]}...")
                            else:
                                prompt_parts.append(f"  {item_key}: {item_value}")
                    else:
                        prompt_parts.append(f"  - {str(item)[:100]}...")
                
                if len(value) > 5:
                    prompt_parts.append(f"\n... and {len(value) - 5} more items")
                    
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    prompt_parts.append(f"  {sub_key}: {str(sub_value)[:100]}...")
            else:
                prompt_parts.append(f"  {str(value)[:200]}...")
        
        prompt_parts.append(f"\nPlease generate a comprehensive response that directly answers the user's question using this collected data.")
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_response(self, intent: QueryIntent, data: Dict[str, Any]) -> str:
        """Generate simple fallback response when AI fails"""
        response_parts = []
        response_parts.append(f"## Query Results\n")
        response_parts.append(f"**Question**: {intent.original_question}\n")
        
        results = data.get("results", {})
        if results:
            response_parts.append("**Found Information**:")
            for key, value in results.items():
                if isinstance(value, list):
                    response_parts.append(f"- {key}: {len(value)} items")
                elif isinstance(value, dict):
                    response_parts.append(f"- {key}: {len(value)} entries")
                else:
                    response_parts.append(f"- {key}: Available")
        else:
            response_parts.append("No specific results found.")
        
        response_parts.append(f"\nFor detailed analysis, please try a more specific query.")
        
        return "\n".join(response_parts)



class LangGraphResearchSystem:
    """LangGraph-based Multi-Agent Research System"""
    
    def __init__(self, config_path: str = "config"):
        # Initialize configuration manager
        self.model_config_manager = ModelConfigManager(f"{config_path}/model_config.json")
        self.config = self._load_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize intelligent entity extractor
        self.entity_extractor = IntelligentEntityExtractor(self.model_config_manager)
        
        # Initialize data access components
        self.query_agent = QueryDBAgent()
        
        try:
            self.vector_indexer = VectorIndexer()
        except Exception as e:
            self.logger.warning(f"Vector indexer not available: {e}")
            self.vector_indexer = None
        
        # Initialize LLM with configuration (with graceful fallback)
        self.llm = None
        if LANGGRAPH_AVAILABLE:
            try:
                # Use query_analyzer as the main LLM model
                self.llm = self.model_config_manager.get_model_instance("query_analyzer")
                if not self.llm:
                    # Fallback to default if agent-specific model fails
                    default_config = self.model_config_manager.get_agent_config("default")
                    self.llm = ChatOpenAI(
                        model=default_config.get("model", "gpt-3.5-turbo"),
                        temperature=default_config.get("temperature", 0.1),
                        max_tokens=default_config.get("max_tokens", 2000)
                    )
                log_event("LangGraphResearchSystem", "llm_initialized", {"model": self.llm.model_name if self.llm else "None"}, level=logging.INFO)
            except Exception as e:
                self.logger.warning(f"Could not initialize LLM: {e}")
                log_event("LangGraphResearchSystem", "llm_init_failed", {"error": str(e)}, level=logging.WARNING)
        else:
            self.logger.warning("LangGraph not available, using fallback mode")
        
        # Create LangGraph tools
        self.tools = self._create_tools()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        log_event("LangGraphResearchSystem", "system_initialized", {"tools_count": len(self.tools), "llm_available": self.llm is not None, "entity_extractor_available": self.entity_extractor is not None}, level=logging.INFO)
        self.logger.info("LangGraph Research System initialized")
    
    def _get_agent_model(self, agent_name: str):
        """Get a model instance for a specific agent"""
        model = self.model_config_manager.get_model_instance(agent_name)
        if not model and self.llm:
            # Fallback to main LLM if agent-specific model not available
            log_event("LangGraphResearchSystem", "agent_model_fallback", {"agent_name": agent_name, "fallback_to": "main_llm"}, level=logging.DEBUG)
            return self.llm
        return model

    def _create_tools(self) -> List:
        """Create LangGraph tools from available functions"""
        
        @tool
        def get_papers_citing_paper(paper_id: str) -> Dict[str, Any]:
            """Find all papers that cite a specific paper. Returns paper IDs and metadata."""
            try:
                return self.query_agent.get_papers_citing_paper(paper_id)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool 
        def get_papers_cited_by_paper(paper_id: str) -> Dict[str, Any]:
            """Find all papers cited by a specific paper. Returns cited paper IDs and metadata."""
            try:
                return self.query_agent.get_papers_cited_by_paper(paper_id)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def get_sentences_citing_paper(paper_id: str) -> Dict[str, Any]:
            """Get exact citation sentences with context. Returns specific quotes and viewpoints."""
            try:
                return self.query_agent.get_sentences_citing_paper(paper_id)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def get_paragraphs_citing_paper(paper_id: str) -> Dict[str, Any]:
            """Get citation paragraphs with broader context. Returns detailed citation discussions."""
            try:
                return self.query_agent.get_paragraphs_citing_paper(paper_id)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def get_papers_id_by_author(author_name: str) -> Dict[str, Any]:
            """Find papers by author name with fuzzy matching. Returns paper IDs and titles."""
            try:
                return self.query_agent.get_papers_id_by_author(author_name)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def get_papers_id_by_title(title: str) -> Dict[str, Any]:
            """Find papers by title with fuzzy matching. Returns exact paper matches."""
            try:
                return self.query_agent.get_papers_id_by_title(title)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def search_all_collections(query: str, limit_per_collection: int = 10) -> Dict[str, Any]:
            """Semantic search across all vector collections. Best for concept queries and content discovery."""
            if not self.vector_indexer:
                return {"error": "Vector indexer not available", "found": False}
            try:
                return self.vector_indexer.search_all_collections(query, limit_per_collection)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def search_relevant_sentences(query: str, top_n: int = 10) -> Dict[str, Any]:
            """Search sentences by semantic similarity. Best for finding specific claims or viewpoints."""
            if not self.vector_indexer:
                return {"error": "Vector indexer not available", "found": False}
            try:
                return self.vector_indexer.search_relevant_sentences(query, top_n)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def search_relevant_paragraphs(query: str, top_n: int = 10) -> Dict[str, Any]:
            """Search paragraphs by semantic similarity. Best for finding detailed discussions.""" 
            if not self.vector_indexer:
                return {"error": "Vector indexer not available", "found": False}
            try:
                return self.vector_indexer.search_relevant_paragraphs(query, top_n)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def search_relevant_sections(query: str, top_n: int = 10) -> Dict[str, Any]:
            """Search sections by semantic similarity. Best for finding topical chapters or sections."""
            if not self.vector_indexer:
                return {"error": "Vector indexer not available", "found": False}
            try:
                return self.vector_indexer.search_relevant_sections(query, top_n)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        # New PDF-based query tools
        @tool
        def query_pdf_content(paper_id: str, query: str, context_window: int = 500) -> Dict[str, Any]:
            """Query content directly from a PDF paper using keyword search. Best for finding specific mentions."""
            try:
                return self.query_agent.query_pdf_content(paper_id, query, context_window)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def get_full_pdf_content(paper_id: str) -> Dict[str, Any]:
            """Get the complete content of a PDF paper. Use for comprehensive paper analysis."""
            try:
                return self.query_agent.get_full_pdf_content(paper_id)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def query_pdf_by_author_and_content(author_name: str, content_query: str) -> Dict[str, Any]:
            """Find papers by author and search their content. Best for author-specific content analysis."""
            try:
                return self.query_agent.query_pdf_by_author_and_content(author_name, content_query)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def query_pdf_by_title_and_content(title_query: str, content_query: str) -> Dict[str, Any]:
            """Find papers by title and search their content. Best for specific paper content analysis."""
            try:
                return self.query_agent.query_pdf_by_title_and_content(title_query, content_query)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        @tool
        def semantic_search_pdf_content(paper_id: str, query: str, similarity_threshold: float = 0.5) -> Dict[str, Any]:
            """Perform semantic search within a specific PDF. Best for conceptual content discovery."""
            try:
                return self.query_agent.semantic_search_pdf_content(paper_id, query, similarity_threshold)
            except Exception as e:
                return {"error": str(e), "found": False}
        
        return [
            get_papers_citing_paper,
            get_papers_cited_by_paper, 
            get_sentences_citing_paper,
            get_paragraphs_citing_paper,
            get_papers_id_by_author,
            get_papers_id_by_title,
            search_all_collections,
            search_relevant_sentences,
            search_relevant_paragraphs,
            search_relevant_sections,
            # PDF-based tools
            query_pdf_content,
            get_full_pdf_content,
            query_pdf_by_author_and_content,
            query_pdf_by_title_and_content,
            semantic_search_pdf_content
        ]
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def analyze_question(state: ResearchState) -> ResearchState:
            """Analyze the user's question and determine intent"""
            question = state["question"]
            request_id = state.get("request_id")
            
            log_event("WorkflowAgent", "step_start", {"step": "analyze_question", "question": question}, level=logging.INFO, request_id=request_id)
            
            # Get the configured model for query analysis
            query_analyzer_model = self._get_agent_model("query_analyzer")
            
            if not query_analyzer_model:
                log_event("WorkflowAgent", "step_start", {"step": "entity_extraction", "question": question}, level=logging.INFO, request_id=request_id)
                entities = self.entity_extractor.extract_entities(question, request_id)
                log_event("WorkflowAgent", "step_finish", {"step": "entity_extraction", "entities": entities}, level=logging.INFO, request_id=request_id)
                
                query_intent = {
                    "query_type": entities.get("query_focus", "concept_search"),
                    "target_entity": entities.get("primary_entity", question),
                    "entity_type": entities.get("primary_entity_type", "concept"),
                    "original_question": question,
                    "extracted_entities": entities
                }
                log_event("WorkflowAgent", "step_finish", {"step": "analyze_question", "query_intent": query_intent}, level=logging.INFO, request_id=request_id)
                return {**state, "query_intent": query_intent, "messages": []}
            
            # First extract entities using the intelligent extractor
            log_event("WorkflowAgent", "step_start", {"step": "entity_extraction", "question": question}, level=logging.INFO, request_id=request_id)
            entities = self.entity_extractor.extract_entities(question, request_id)
            log_event("WorkflowAgent", "step_finish", {"step": "entity_extraction", "entities": entities}, level=logging.INFO, request_id=request_id)
            
            system_prompt = """You are an expert academic research query analyzer. Your job is to understand the user's research intent and extract structured information.

QUERY TYPES:
- reverse_citation_analysis: "Who cites X?", "Papers citing X", "Citation context of X"
- citation_analysis: "What does X cite?", "References in X paper"  
- author_search: "Papers by author X", "X's publications"
- paper_search: "Find paper about Y", "Paper titled Z"
- concept_search: "What is X?", "Explain concept Y"

ENTITY EXTRACTION:
- For citation queries: Extract the AUTHOR NAME being cited (e.g., "Rivkin", "Porter", "Smith")
- For paper queries: Extract the PAPER TITLE or keywords
- For author queries: Extract the AUTHOR NAME
- For concept queries: Extract the CONCEPT term

EXAMPLES:

Query: "The paper cite Rivkin, what is the citation context?"
Analysis: {
  "query_type": "reverse_citation_analysis",
  "target_entity": "Rivkin", 
  "entity_type": "author",
  "reasoning": "User wants citation context for Rivkin - this is reverse citation analysis"
}

Query: "What papers cite Porter's competitive strategy?"
Analysis: {
  "query_type": "reverse_citation_analysis",
  "target_entity": "Porter",
  "entity_type": "author", 
  "reasoning": "Looking for papers that cite Porter's work"
}

Query: "What does the Innovation paper by Smith cite?"
Analysis: {
  "query_type": "citation_analysis",
  "target_entity": "Innovation paper by Smith",
  "entity_type": "paper",
  "reasoning": "User wants to know what this specific paper references"
}

Query: "Papers by Johnson on strategy"
Analysis: {
  "query_type": "author_search", 
  "target_entity": "Johnson",
  "entity_type": "author",
  "reasoning": "Looking for papers authored by Johnson"
}

INSTRUCTIONS:
1. Focus on the MAIN ACTION: citing, cited by, authored by, about topic
2. Extract the KEY ENTITY: author name, paper identifier, or concept
3. Choose the most specific query type that matches the user's intent
4. Always extract clean entity names (e.g., "Rivkin" not "Rivkin, what is the citation")

Return ONLY a JSON object with: query_type, target_entity, entity_type, reasoning"""
            
            entities_context = f"\nExtracted entities: {json.dumps(entities, ensure_ascii=False)}"
            user_prompt = f"Analyze this research question: {question}{entities_context}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            log_event("WorkflowAgent", "step_start", {"step": "llm_intent_analysis", "question": question}, level=logging.INFO, request_id=request_id)
            try:
                response = query_analyzer_model.invoke(messages)
                log_event("WorkflowAgent", "step_finish", {"step": "llm_intent_analysis", "llm_response": response.content[:200]}, level=logging.INFO, request_id=request_id)
                
                try:
                    response_content = response.content
                    if "```json" in response_content:
                        json_start = response_content.find("```json") + 7
                        json_end = response_content.find("```", json_start)
                        json_text = response_content[json_start:json_end].strip()
                    elif "{" in response_content and "}" in response_content:
                        json_start = response_content.find("{")
                        json_end = response_content.rfind("}") + 1
                        json_text = response_content[json_start:json_end]
                    else:
                        raise ValueError("No JSON found in response")
                    analysis = json.loads(json_text)
                    query_intent = {
                        "query_type": analysis.get("query_type", entities.get("query_focus", "concept_search")),
                        "target_entity": analysis.get("target_entity", entities.get("primary_entity", question)),
                        "entity_type": analysis.get("entity_type", entities.get("primary_entity_type", "concept")),
                        "original_question": question,
                        "reasoning": analysis.get("reasoning", ""),
                        "extracted_entities": entities
                    }
                    log_event("WorkflowAgent", "step_finish", {"step": "analyze_question", "query_intent": query_intent}, level=logging.INFO, request_id=request_id)
                except (json.JSONDecodeError, ValueError, KeyError) as parse_error:
                    self.logger.warning(f"Failed to parse LLM analysis: {parse_error}, using entity-based fallback")
                    log_event("WorkflowAgent", "step_finish", {"step": "analyze_question", "fallback": "entity_extraction", "error": str(parse_error)}, level=logging.WARNING, request_id=request_id)
                    query_intent = {
                        "query_type": entities.get("query_focus", "concept_search"),
                        "target_entity": entities.get("primary_entity", question),
                        "entity_type": entities.get("primary_entity_type", "concept"),
                        "original_question": question,
                        "reasoning": "Fallback analysis using intelligent entity extraction",
                        "extracted_entities": entities
                    }
                return {**state, "query_intent": query_intent, "messages": [response]}
            except Exception as e:
                log_event("WorkflowAgent", "step_finish", {"step": "analyze_question", "error": str(e)}, level=logging.ERROR, request_id=request_id)
                query_intent = {
                    "query_type": entities.get("query_focus", "concept_search"),
                    "target_entity": entities.get("primary_entity", question),
                    "entity_type": entities.get("primary_entity_type", "concept"),
                    "original_question": question,
                    "reasoning": f"Error fallback using entity extraction: {str(e)}",
                    "extracted_entities": entities
                }
                return {**state, "query_intent": query_intent, "messages": []}
        
        def plan_and_execute(state: ResearchState) -> ResearchState:
            """Plan queries and execute them using available tools"""
            query_intent = state.get("query_intent", {})
            question = state["question"]
            request_id = state.get("request_id")
            
            log_event("WorkflowAgent", "plan_and_execute_start", {"query_intent": query_intent, "question": question}, level=logging.DEBUG, request_id=request_id)
            
            # Get the configured model for planning and execution
            planner_model = self._get_agent_model("query_analyzer")
            
            if not planner_model:
                # Fallback to direct tool execution based on query intent
                log_event("WorkflowAgent", "plan_and_execute_fallback", {"reason": "no_planner_model"}, level=logging.INFO, request_id=request_id)
                return self._execute_tools_based_on_intent(state, query_intent)
            
            # Create tool-enabled agent
            tool_node = ToolNode(self.tools)
            llm_with_tools = planner_model.bind_tools(self.tools)
            
            system_prompt = """You are a research assistant with access to academic database tools and direct PDF content access. 
            Based on the user's question, select and use the most appropriate tools to gather information.
            
            Available tools:
            
            DATABASE TOOLS (Graph & Vector):
            - get_papers_citing_paper: Find papers that cite a specific paper
            - get_papers_cited_by_paper: Find papers cited by a specific paper  
            - get_sentences_citing_paper: Get citation sentences with context
            - get_paragraphs_citing_paper: Get citation paragraphs
            - get_papers_id_by_author: Find papers by author name
            - get_papers_id_by_title: Find papers by title
            - search_all_collections: Semantic search across all content
            - search_relevant_sentences: Find specific claims/viewpoints
            - search_relevant_paragraphs: Find detailed discussions
            - search_relevant_sections: Find topical sections
            
            PDF CONTENT TOOLS (Direct paper access):
            - query_pdf_content: Search specific content within a paper by paper_id
            - get_full_pdf_content: Get complete content of a paper by paper_id
            - query_pdf_by_author_and_content: Find author papers and search their content
            - query_pdf_by_title_and_content: Find papers by title and search their content
            - semantic_search_pdf_content: Semantic search within a specific paper
            
            STRATEGY:
            - For questions about specific papers: Use PDF tools for direct content access
            - For questions about citations and relationships: Use database tools
            - For concept questions: Use both database and PDF tools for comprehensive answers
            - For author-specific content queries: Use query_pdf_by_author_and_content
            - For detailed paper analysis: Use get_full_pdf_content then query_pdf_content
            
            The PDF tools provide direct access to the complete paper content, which is often more comprehensive than database extracts."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Research question: {question}\nQuery intent: {query_intent}")
            ]
            
            try:
                # Get tool calls from LLM
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                
                collected_data = {"results": {}, "tool_calls": []}
                
                # Execute tool calls if any
                if response.tool_calls:
                    log_event("WorkflowAgent", "tool_calls_received", {"tool_calls": len(response.tool_calls)}, level=logging.DEBUG, request_id=request_id)
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        
                        log_event("WorkflowAgent", "tool_call_start", {"tool": tool_name, "args": tool_args}, level=logging.DEBUG, request_id=request_id)
                        
                        # Find and execute the tool
                        for tool in self.tools:
                            if tool.name == tool_name:
                                try:
                                    result = tool.invoke(tool_args)
                                    collected_data["results"][tool_name] = result
                                    collected_data["tool_calls"].append({
                                        "tool": tool_name,
                                        "args": tool_args,
                                        "result": result
                                    })
                                    log_event("WorkflowAgent", "tool_call_success", {"tool": tool_name, "result_type": type(result).__name__}, level=logging.INFO, request_id=request_id)
                                except Exception as e:
                                    collected_data["results"][tool_name] = {"error": str(e)}
                                    log_event("WorkflowAgent", "tool_call_error", {"tool": tool_name, "error": str(e)}, level=logging.ERROR, request_id=request_id)
                                break
                
                log_event("WorkflowAgent", "plan_and_execute_success", {"collected_data_keys": list(collected_data["results"].keys())}, level=logging.INFO, request_id=request_id)
                return {**state, "collected_data": collected_data, "messages": messages}
                
            except Exception as e:
                log_event("WorkflowAgent", "plan_and_execute_error", {"error": str(e)}, level=logging.ERROR, request_id=request_id)
                return {**state, "error": f"Planning and execution failed: {str(e)}", "messages": []}
        
        def generate_response(state: ResearchState) -> ResearchState:
            """Generate the final response based on collected data"""
            question = state["question"]
            collected_data = state.get("collected_data", {})
            request_id = state.get("request_id")
            
            log_event("WorkflowAgent", "generate_response_start", {"question": question, "collected_data_keys": list(collected_data.get("results", {}).keys())}, level=logging.DEBUG, request_id=request_id)
            
            # Get the configured model for response generation
            response_generator_model = self._get_agent_model("response_generator")
            
            if not response_generator_model:
                # Fallback to simple structured response
                log_event("WorkflowAgent", "generate_response_fallback", {"reason": "no_response_model"}, level=logging.INFO, request_id=request_id)
                return self._generate_simple_response(state, collected_data)
            
            system_prompt = """You are an expert academic research assistant. Generate a comprehensive response 
            to the user's research question based on the collected data. Use clear structure with headings, 
            include specific citations and details, and organize the information logically."""
            
            data_summary = ""
            if collected_data and "results" in collected_data:
                data_summary = "Collected data:\n"
                for tool_name, result in collected_data["results"].items():
                    if isinstance(result, dict) and result.get("found", True):
                        data_summary += f"- {tool_name}: {len(result.get('data', result))} items found\n"
                    else:
                        data_summary += f"- {tool_name}: No results or error\n"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Question: {question}\n\n{data_summary}\n\nDetailed data: {json.dumps(collected_data, indent=2, ensure_ascii=False)}")
            ]
            
            try:
                response = response_generator_model.invoke(messages)
                final_response = response.content
                
                log_event("WorkflowAgent", "generate_response_success", {"response_length": len(final_response)}, level=logging.INFO, request_id=request_id)
                return {**state, "final_response": final_response, "messages": state.get("messages", []) + [response]}
            except Exception as e:
                log_event("WorkflowAgent", "generate_response_error", {"error": str(e)}, level=logging.ERROR, request_id=request_id)
                return {**state, "final_response": f"Failed to generate response: {str(e)}", "messages": state.get("messages", [])}
        
        # Build the workflow graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("analyze", analyze_question)
        workflow.add_node("research", plan_and_execute)
        workflow.add_node("respond", generate_response)
        
        # Add edges
        workflow.add_edge(START, "analyze")
        workflow.add_edge("analyze", "research")
        workflow.add_edge("research", "respond")
        workflow.add_edge("respond", END)
        
        return workflow.compile()
    
    def _execute_tools_based_on_intent(self, state: ResearchState, query_intent: Dict) -> ResearchState:
        """Execute tools based on query intent when LLM is not available"""
        question = state["question"]
        request_id = state.get("request_id")
        collected_data = {"results": {}, "tool_calls": []}
        
        log_event("WorkflowAgent", "execute_tools_start", {"query_intent": query_intent}, level=logging.DEBUG, request_id=request_id)
        
        try:
            query_type = query_intent.get("query_type", "concept_search")
            target_entity = query_intent.get("target_entity", "")
            entity_type = query_intent.get("entity_type", "concept")
            
            # Get extracted entities for more intelligent processing
            extracted_entities = query_intent.get("extracted_entities", {})
            author_names = extracted_entities.get("author_names", [])
            paper_titles = extracted_entities.get("paper_titles", [])
            concepts = extracted_entities.get("concepts", [])
            
            # Use the most relevant entity based on query type
            if query_type == "reverse_citation_analysis" and entity_type == "author":
                # Use the first author name from extracted entities, fallback to target_entity
                search_entity = author_names[0] if author_names else target_entity
                
                log_event("WorkflowAgent", "reverse_citation_search", {"search_entity": search_entity, "author_names": author_names}, level=logging.DEBUG, request_id=request_id)
                
                # Execute author search first for any author
                author_result = self.tools[4].invoke({"author_name": search_entity})  # get_papers_id_by_author
                collected_data["results"]["get_papers_id_by_author"] = author_result
                
                # Check for multiple author matches and handle ambiguity
                if author_result.get("found") and author_result.get("data"):
                    matches = author_result.get("data", [])
                    
                    # If multiple matches found, create clarification response
                    if len(matches) > 1:
                        # Group by unique authors to avoid duplicate papers by same author
                        unique_authors = {}
                        for match in matches:
                            authors_str = ", ".join(match.get("authors", []))
                            if authors_str not in unique_authors:
                                unique_authors[authors_str] = match
                        
                        unique_matches = list(unique_authors.values())
                        
                        if len(unique_matches) > 1:
                            # Create clarification message
                            clarification_parts = [
                                f"I found {len(unique_matches)} different authors named {search_entity.title()}. Please specify which one you meant:",
                                ""
                            ]
                            
                            for i, match in enumerate(unique_matches, 1):
                                authors = ", ".join(match.get("authors", ["Unknown"]))
                                title = match.get("title", "Unknown Title")
                                year = match.get("year", "Unknown Year")
                                clarification_parts.append(f"{i}. {authors} - \"{title}\" ({year})")
                            
                            clarification_parts.extend([
                                "",
                                "Please respond with the number of your choice, and I'll analyze the citations for that specific author."
                            ])
                            
                            log_event("WorkflowAgent", "disambiguation_required", {"entity": search_entity, "matches": len(unique_matches)}, level=logging.INFO, request_id=request_id)
                            
                            # Return clarification request instead of proceeding
                            return {**state, "collected_data": {
                                "results": {"clarification_needed": True},
                                "clarification_message": "\n".join(clarification_parts),
                                "clarification_options": unique_matches
                            }}
                    
                    # Single match or user already clarified - proceed normally
                    paper_id = matches[0].get("paper_id")
                    if paper_id:
                        log_event("WorkflowAgent", "citation_analysis_start", {"paper_id": paper_id, "search_entity": search_entity}, level=logging.INFO, request_id=request_id)
                        
                        # Get citation relationships
                        citing_result = self.tools[0].invoke({"paper_id": paper_id})  # get_papers_citing_paper
                        collected_data["results"]["get_papers_citing_paper"] = citing_result
                        
                        # Get citation contexts - THIS IS WHAT USER ASKED FOR!
                        context_result = self.tools[2].invoke({"paper_id": paper_id})  # get_sentences_citing_paper
                        collected_data["results"]["get_sentences_citing_paper"] = context_result
                        
                        # Get paragraphs for more context
                        paragraph_result = self.tools[3].invoke({"paper_id": paper_id})  # get_paragraphs_citing_paper
                        collected_data["results"]["get_paragraphs_citing_paper"] = paragraph_result
                        
                        # Get full PDF content for comprehensive analysis
                        pdf_content_result = self.tools[11].invoke({"paper_id": paper_id})  # get_full_pdf_content
                        collected_data["results"]["get_full_pdf_content"] = pdf_content_result
                        
                        # Search for specific content in author's papers using extracted entities
                        content_query = " ".join(concepts) if concepts else "citation context"
                        author_content_result = self.tools[12].invoke({"author_name": search_entity, "content_query": content_query})  # query_pdf_by_author_and_content
                        collected_data["results"]["query_pdf_by_author_and_content"] = author_content_result
            
            elif query_type == "paper_search" or entity_type == "paper":
                # Use paper titles from extracted entities
                search_title = paper_titles[0] if paper_titles else target_entity
                
                log_event("WorkflowAgent", "paper_search", {"search_title": search_title, "paper_titles": paper_titles}, level=logging.DEBUG, request_id=request_id)
                
                # Handle paper title search with disambiguation
                title_result = self.tools[5].invoke({"title": search_title})  # get_papers_id_by_title
                collected_data["results"]["get_papers_id_by_title"] = title_result
                
                # Check for multiple paper matches and handle ambiguity
                if title_result.get("found") and title_result.get("data"):
                    matches = title_result.get("data", [])
                    
                    # If multiple matches found, create clarification response
                    if len(matches) > 1:
                        # Create clarification message for papers
                        clarification_parts = [
                            f"I found {len(matches)} papers with similar titles. Please specify which one you meant:",
                            ""
                        ]
                        
                        for i, match in enumerate(matches, 1):
                            title = match.get("title", "Unknown Title")
                            authors = ", ".join(match.get("authors", ["Unknown Author"]))
                            year = match.get("year", "Unknown Year")
                            journal = match.get("journal", "")
                            journal_info = f" ({journal})" if journal else ""
                            clarification_parts.append(f"{i}. \"{title}\" by {authors} ({year}){journal_info}")
                        
                        clarification_parts.extend([
                            "",
                            "Please respond with the number of your choice, and I'll analyze that specific paper."
                        ])
                        
                        log_event("WorkflowAgent", "paper_disambiguation_required", {"search_title": search_title, "matches": len(matches)}, level=logging.INFO, request_id=request_id)
                        
                        # Return clarification request instead of proceeding
                        return {**state, "collected_data": {
                            "results": {"clarification_needed": True},
                            "clarification_message": "\n".join(clarification_parts),
                            "clarification_options": matches
                        }}
                    
                    # Single match or user already clarified - proceed with paper analysis
                    paper_id = matches[0].get("paper_id")
                    if paper_id:
                        log_event("WorkflowAgent", "paper_analysis_start", {"paper_id": paper_id, "search_title": search_title}, level=logging.INFO, request_id=request_id)
                        
                        # Get papers citing this paper
                        citing_result = self.tools[0].invoke({"paper_id": paper_id})  # get_papers_citing_paper
                        collected_data["results"]["get_papers_citing_paper"] = citing_result
                        
                        # Get papers cited by this paper
                        cited_result = self.tools[1].invoke({"paper_id": paper_id})  # get_papers_cited_by_paper
                        collected_data["results"]["get_papers_cited_by_paper"] = cited_result
                        
                        # Get citation contexts
                        context_result = self.tools[2].invoke({"paper_id": paper_id})  # get_sentences_citing_paper
                        collected_data["results"]["get_sentences_citing_paper"] = context_result
                        
                        # Get full PDF content
                        pdf_content_result = self.tools[11].invoke({"paper_id": paper_id})  # get_full_pdf_content
                        collected_data["results"]["get_full_pdf_content"] = pdf_content_result
            
            elif query_type == "author_search" and entity_type == "author":
                # Use author names from extracted entities
                search_author = author_names[0] if author_names else target_entity
                
                log_event("WorkflowAgent", "author_search", {"search_author": search_author, "author_names": author_names}, level=logging.DEBUG, request_id=request_id)
                
                # Handle author search (papers by author)
                author_result = self.tools[4].invoke({"author_name": search_author})  # get_papers_id_by_author
                collected_data["results"]["get_papers_id_by_author"] = author_result
                
                # Search for specific content in author's papers using concepts
                content_query = " ".join(concepts) if concepts else question
                author_content_result = self.tools[12].invoke({"author_name": search_author, "content_query": content_query})  # query_pdf_by_author_and_content
                collected_data["results"]["query_pdf_by_author_and_content"] = author_content_result
            
            else:
                # For other query types, use semantic search with intelligent query construction
                search_query = " ".join(concepts) if concepts else question
                
                log_event("WorkflowAgent", "concept_search", {"search_query": search_query, "concepts": concepts}, level=logging.DEBUG, request_id=request_id)
                
                search_result = self.tools[6].invoke({"query": search_query, "limit_per_collection": 10})  # search_all_collections
                collected_data["results"]["search_all_collections"] = search_result
                
                # Try additional searches based on extracted entities
                if author_names:
                    # Search for author's papers and their content
                    for author in author_names:
                        author_content_result = self.tools[12].invoke({"author_name": author, "content_query": search_query})  # query_pdf_by_author_and_content
                        collected_data["results"][f"query_pdf_by_author_and_content_{author}"] = author_content_result
                
                if paper_titles:
                    # Search for specific papers by title
                    for title in paper_titles:
                        title_content_result = self.tools[13].invoke({"title_query": title, "content_query": search_query})  # query_pdf_by_title_and_content
                        collected_data["results"][f"query_pdf_by_title_and_content_{title[:20]}"] = title_content_result
        
        except Exception as e:
            log_event("WorkflowAgent", "execute_tools_error", {"error": str(e)}, level=logging.ERROR, request_id=request_id)
            collected_data["results"]["error"] = {"error": str(e)}
        
        log_event("WorkflowAgent", "execute_tools_complete", {"results_count": len(collected_data["results"])}, level=logging.INFO, request_id=request_id)
        return {**state, "collected_data": collected_data}
    
    def _generate_simple_response(self, state: ResearchState, collected_data: Dict) -> ResearchState:
        """Generate a simple structured response when LLM is not available"""
        question = state["question"]
        response_parts = []
        
        # Check if clarification is needed
        if collected_data.get("clarification_message"):
            # Return the clarification message directly
            final_response = collected_data["clarification_message"]
            return {**state, "final_response": final_response}
        
        # Check if this is a clarification response with options available
        results = collected_data.get("results", {})
        if results.get("clarification_needed"):
            clarification_msg = collected_data.get("clarification_message", "Please provide more specific information.")
            return {**state, "final_response": clarification_msg}
        
        # Normal content response generation
        response_parts.append(f"# Research Results\n")
        response_parts.append(f"**Question**: {question}\n")
        
        if results:
            response_parts.append("## Findings:\n")
            
            for tool_name, result in results.items():
                if isinstance(result, dict) and result.get("found", False):
                    data = result.get("data", [])
                    if data:
                        response_parts.append(f"### {tool_name.replace('_', ' ').title()}")
                        response_parts.append(f"Found {len(data)} items:\n")
                        
                        for i, item in enumerate(data[:5], 1):  # Show first 5 items
                            if isinstance(item, dict):
                                title = item.get("title", item.get("text", "Unknown"))
                                response_parts.append(f"{i}. {title[:100]}...")
                        
                        if len(data) > 5:
                            response_parts.append(f"... and {len(data) - 5} more items\n")
                    else:
                        response_parts.append(f"### {tool_name.replace('_', ' ').title()}")
                        response_parts.append("No specific data found\n")
                else:
                    response_parts.append(f"### {tool_name.replace('_', ' ').title()}")
                    response_parts.append("Search completed but no results found\n")
        else:
            response_parts.append("No data was collected for this query.\n")
        
        response_parts.append("\n*Note: This is a simplified response generated without LLM processing.*")
        
        final_response = "\n".join(response_parts)
        return {**state, "final_response": final_response}
    
    def research_question(self, question: str) -> str:
        """Main method to research a question using the LangGraph workflow"""
        self.logger.info(f"Starting LangGraph research for: {question}")
        request_id = str(uuid.uuid4())
        initial_state = ResearchState(
            question=question,
            query_intent=None,
            target_entity=None,
            clarification_needed=False,
            query_plan=None,
            collected_data=None,
            reflection_result=None,
            final_response=None,
            messages=[],
            error=None,
            request_id=request_id
        )
        log_event("System", "user_question", {"question": question}, level=logging.INFO, request_id=request_id)
        try:
            final_state = self.workflow.invoke(initial_state)
            if final_state.get("error"):
                log_event("System", "error", {"error": final_state["error"]}, level=logging.ERROR, request_id=request_id)
                return f"❌ Error: {final_state['error']}"
            log_event("System", "final_response", {"response": final_state.get("final_response", "No response generated")}, level=logging.INFO, request_id=request_id)
            return final_state.get("final_response", "No response generated")
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            log_event("System", "exception", {"error": str(e)}, level=logging.ERROR, request_id=request_id)
            return f"❌ Workflow error: {str(e)}"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        return {
            "ambiguity_threshold": 0.8,
            "max_reflection_cycles": 3,
            "user_timeout": 30,
            "max_results_per_query": 50,
            "confidence_threshold": 0.7
        }


if __name__ == "__main__":
    """Test function"""
    logging.basicConfig(level=logging.DEBUG)
    system = LangGraphResearchSystem()
    
    # Test question
    test_question = "The paper cite Rivkin, what is the citation context?"
    
    print("🤖 CiteWeave Multi-Agent Research System")
    print("=" * 50)
    print(f"Question: {test_question}")
    print("=" * 50)
    
    result = system.research_question(test_question)
    print(result)() 