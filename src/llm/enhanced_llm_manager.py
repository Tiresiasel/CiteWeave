"""
Enhanced LLM Manager for Multi-Agent System
Handles model configuration, language processing, and memory management
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """A single turn in a conversation"""
    timestamp: datetime
    user_message: str
    user_language: str
    agent_response: str
    response_language: str
    query_type: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationMemory:
    """Manages conversation history and context"""
    thread_id: str
    user_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    summary: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    context_metadata: Dict[str, Any] = field(default_factory=dict)

class LanguageDetector:
    """Simple language detection using common patterns"""
    
    LANGUAGE_PATTERNS = {
        'zh': ['的', '是', '在', '了', '和', '有', '我', '你', '他', '她', '论文', '引用', '作者',"这","那"],
        'fr': ['le', 'la', 'les', 'un', 'une', 'de', 'du', 'des', 'et', 'ou', 'qui', 'que'],
        'de': ['der', 'die', 'das', 'und', 'oder', 'ist', 'sind', 'ein', 'eine', 'mit', 'zu'],
        'es': ['el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'y', 'o', 'que', 'es'],
        'it': ['il', 'la', 'lo', 'gli', 'le', 'un', 'una', 'di', 'del', 'e', 'o', 'che'],
        'ja': ['の', 'は', 'が', 'を', 'に', 'で', 'と', 'から', 'まで', '論文', '引用'],
        'ko': ['이', '가', '을', '를', '에', '의', '는', '은', '와', '과', '논문', '인용'],
        'pt': ['o', 'a', 'os', 'as', 'um', 'uma', 'de', 'do', 'da', 'e', 'ou', 'que'],
        'ru': ['и', 'в', 'на', 'с', 'по', 'за', 'от', 'до', 'для', 'из', 'при']
    }
    
    @classmethod
    def detect_language(cls, text: str) -> str:
        """Detect language of input text"""
        text_lower = text.lower()
        scores = {}
        
        for lang, patterns in cls.LANGUAGE_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                scores[lang] = score
        
        if scores:
            return max(scores, key=scores.get)
        return 'en'  # Default to English

class EnhancedLLMManager:
    """
    Enhanced LLM Manager that handles:
    - Multiple agents with different model configurations
    - Language processing and translation
    - Conversation memory management
    - Context summarization
    """
    
    def __init__(self, config_path: str = "config/model_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.models: Dict[str, Any] = {}
        self.memory_store: Dict[str, ConversationMemory] = {}
        
        # Initialize models for each agent
        self._initialize_agent_models()
        
        logger.info("EnhancedLLMManager initialized with agents: " + 
                   ", ".join(self.config["llm"]["agents"].keys()))

    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise

    def _initialize_agent_models(self):
        """Initialize LLM models for each agent"""
        for agent_name, agent_config in self.config["llm"]["agents"].items():
            try:
                if agent_config["provider"] == "openai":
                    model = ChatOpenAI(
                        model=agent_config["model"],
                        temperature=agent_config.get("temperature", 0.1),
                        max_tokens=agent_config.get("max_tokens", 1000)
                    )
                elif agent_config["provider"] == "ollama":
                    from langchain_ollama import Ollama
                    model = Ollama(
                        model=agent_config["model"],
                        temperature=agent_config.get("temperature", 0.1)
                    )
                else:
                    raise ValueError(f"Unsupported provider: {agent_config['provider']}")
                
                self.models[agent_name] = model
                logger.info(f"Initialized {agent_name} with {agent_config['model']}")
                
            except Exception as e:
                logger.error(f"Failed to initialize model for {agent_name}: {e}")
                raise

    def get_agent_model(self, agent_name: str):
        """Get the LLM model for a specific agent"""
        if agent_name not in self.models:
            logger.warning(f"Agent {agent_name} not found, using default")
            agent_name = "default"
        
        return self.models.get(agent_name, self.models.get("default"))

    async def process_language(self, text: str, target_language: str = "en") -> Tuple[str, str]:
        """
        Process language: detect source language and translate to target language
        
        Returns:
            Tuple of (translated_text, detected_language)
        """
        try:
            # Detect source language
            detected_lang = LanguageDetector.detect_language(text)
            
            # If already in target language, return as-is
            if detected_lang == target_language:
                return text, detected_lang
            
            # Translate to target language
            model = self.get_agent_model("language_processor")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a professional translator. Translate the following text from {detected_lang} to {target_language}.                                
                Instructions:
                - Preserve technical terms and proper names (especially author names, paper titles)
                - Maintain the original meaning and tone
                - Do not add explanations or comments
                - Return ONLY the translated text
                - For all key academic terms, author names, paper titles, and technical keywords, always provide the original English term in parentheses or slashes after the translated/localized term, regardless of the output language. This helps the reader match the original source."""),
                ("user", "{text}")
            ])
            
            chain = prompt | model | StrOutputParser()
            translated_text = await chain.ainvoke({"text": text})
            
            return translated_text.strip(), detected_lang
            
        except Exception as e:
            logger.error(f"Language processing failed: {e}")
            return text, "en"  # Fallback to original text

    def get_memory(self, thread_id: str, user_id: str = "default") -> ConversationMemory:
        """Get or create conversation memory for a thread"""
        memory_key = f"{user_id}:{thread_id}"
        
        if memory_key not in self.memory_store:
            self.memory_store[memory_key] = ConversationMemory(
                thread_id=thread_id,
                user_id=user_id
            )
        
        return self.memory_store[memory_key]

    def add_conversation_turn(self, 
                            thread_id: str,
                            user_message: str,
                            user_language: str,
                            agent_response: str,
                            response_language: str,
                            query_type: str,
                            confidence: float,
                            metadata: Optional[Dict] = None,
                            user_id: str = "default"):
        """Add a new conversation turn to memory"""
        memory = self.get_memory(thread_id, user_id)
        
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            user_language=user_language,
            agent_response=agent_response,
            response_language=response_language,
            query_type=query_type,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        memory.turns.append(turn)
        memory.last_updated = datetime.now()
        
        # Check if summarization is needed
        # Use a default threshold if config is missing
        threshold = self.config.get("conversation", {}).get("summarization_threshold", 10)
        if len(memory.turns) >= threshold:
            self._summarize_conversation(memory)

    def _summarize_conversation(self, memory: ConversationMemory):
        """Summarize conversation when it gets too long"""
        try:
            model = self.get_agent_model("memory_manager")
            
            # Prepare conversation history for summarization
            conversation_text = []
            for turn in memory.turns[:-3]:  # Keep last 3 turns, summarize the rest
                conversation_text.append(f"User: {turn.user_message}")
                conversation_text.append(f"Assistant: {turn.agent_response}")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Summarize this academic paper discussion conversation. Include:
                1. Key papers and authors mentioned
                2. Main topics explored
                3. Types of queries asked (citation analysis, paper search, etc.)
                4. Important findings or patterns discovered

                Keep the summary concise but informative for future context.
                For all key academic terms, author names, paper titles, and technical keywords, always provide the original English term in parentheses or slashes after the translated/localized term, regardless of the output language. This helps the reader match the original source."""),
                ("user", "Conversation to summarize:\n\n{{conversation}}")
            ])
            
            chain = prompt | model | StrOutputParser()
            summary = None
            try:
                summary = chain.invoke({"conversation": "\n".join(conversation_text)})
                logger.info(f"Conversation summary generated: {summary}")
            except Exception as e:
                logger.error(f"Summary generation failed: {e}")
                summary = ""
            # Update memory
            if summary and isinstance(summary, str):
                summary = summary.replace('{', '').replace('}', '')
            memory.summary = summary if summary is not None else ""
            memory.turns = memory.turns[-3:]  # Keep only last 3 turns
            
            logger.info(f"Summarized conversation for thread {memory.thread_id}")
            
        except Exception as e:
            logger.error(f"Conversation summarization failed: {e}")

    def get_conversation_context(self, thread_id: str, user_id: str = "default") -> str:
        """Get conversation context for agents"""
        memory = self.get_memory(thread_id, user_id)
        
        context_parts = []
        
        # Add summary if available
        if memory.summary:
            if not isinstance(memory.summary, str):
                logger.warning(f"Memory summary is not a string: {memory.summary}")
                memory.summary = str(memory.summary)
            context_parts.append(f"Previous conversation summary: {memory.summary}")
        
        # Add recent turns
        if memory.turns:
            context_parts.append("Recent conversation:")
            for turn in memory.turns[-3:]:  # Last 3 turns
                context_parts.append(f"User ({turn.user_language}): {turn.user_message}")
                context_parts.append(f"Assistant: {turn.agent_response[:200]}...")
        
        return "\n".join(context_parts) if context_parts else ""

    async def generate_clarification_question(self, 
                                            issue: str,
                                            context: str,
                                            user_language: str = "en") -> str:
        """Generate a clarification question when information is ambiguous or missing"""
        try:
            model = self.get_agent_model("paper_disambiguator")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are helping users with academic paper queries. When information is unclear or missing, 
                you need to ask clarifying questions in {user_language}.

                Generate a helpful clarification question that:
                1. Explains what information is unclear or missing
                2. Suggests specific details the user could provide
                3. Offers options when possible
                4. Is polite and encouraging

                For all key academic terms, author names, paper titles, and technical keywords, always provide the original English term in parentheses or slashes after the translated/localized term, regardless of the output language. This helps the reader match the original source.

                Issue: {issue}
                Context: {context}"""),
                ("user", "Generate a clarification question for this situation.")
            ])
            
            chain = prompt | model | StrOutputParser()
            question = await chain.ainvoke({})
            
            return question.strip()
            
        except Exception as e:
            logger.error(f"Clarification question generation failed: {e}")
            return f"Could you provide more specific information about your query?"

    def cleanup_old_conversations(self, days_threshold: int = 30):
        """Clean up old conversation memories"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        to_remove = []
        for key, memory in self.memory_store.items():
            if memory.last_updated < cutoff_date:
                to_remove.append(key)
        
        for key in to_remove:
            del self.memory_store[key]
        
        logger.info(f"Cleaned up {len(to_remove)} old conversations")

    def get_agent_config(self, agent_name: str) -> Dict:
        """Get configuration for a specific agent"""
        return self.config["llm"]["agents"].get(agent_name, self.config["llm"]["default"])

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.config.get("supported_languages", ["en"])

    def call_with_tools(self, messages: List[Dict], tools: List[Dict], max_tokens: int = 1000) -> Any:
        """Call LLM with function calling capability"""
        try:
            # Get the query_analyzer model for function calling
            model = self.get_agent_model("query_analyzer")
            
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Convert tools to LangChain format
            from langchain.tools import tool
            from langchain_core.tools import BaseTool
            
            langchain_tools = []
            for tool_def in tools:
                # Create a simple tool function
                def create_tool_function(tool_name, tool_description, tool_parameters):
                    @tool(name=tool_name, description=tool_description)
                    def tool_function(**kwargs):
                        # This is a placeholder - the actual execution happens elsewhere
                        return {"tool_name": tool_name, "parameters": kwargs, "status": "called"}
                    return tool_function
                
                tool_func = create_tool_function(
                    tool_def["name"],
                    tool_def["description"],
                    tool_def["parameters"]
                )
                langchain_tools.append(tool_func)
            
            # Bind tools to the model
            model_with_tools = model.bind_tools(langchain_tools)
            
            # Call the model
            response = model_with_tools.invoke(langchain_messages)
            
            return response
            
        except Exception as e:
            logger.error(f"Function calling failed: {e}")
            return None

    def generate_response(self, messages: List[Dict], max_tokens: int = 2000, temperature: float = 0.2) -> str:
        """Generate a simple text response without function calling"""
        try:
            # Get the response_generator model
            model = self.get_agent_model("response_generator")
            
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Call the model
            response = model.invoke(langchain_messages)
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I encountered an error while generating the response."

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_language_processing():
        manager = EnhancedLLMManager()
        
        # Test language detection and translation
        test_queries = [
            "迈克尔波特1980年的论文讲什么",
            "Qu'est-ce que Michael Porter a écrit en 1980?",
            "Was hat Michael Porter 1980 geschrieben?",
        ]
        
        for query in test_queries:
            translated, detected = await manager.process_language(query, "en")
            logging.info(f"Original ({detected}): {query}")
            logging.info(f"Translated (en): {translated}")
            logging.info("---")
    
    # asyncio.run(test_language_processing()) 