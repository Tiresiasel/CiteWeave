"""
Enhanced Multi-Agent System Demo for CiteWeave
Demonstrates language processing, clarification questions, memory management, and robust error handling
"""

import asyncio
import sys
import os
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.archive.multi_agent_system import EnhancedMultiAgentSystem
from src.graph_builder import GraphDB
from src.vector_indexer import VectorIndexer
from src.config_manager import ConfigManager

async def main():
    """Demo the enhanced multi-agent system"""
    
    print("🚀 Enhanced Multi-Agent CiteWeave Demo")
    print("=" * 50)
    
    # Initialize configuration
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    config_manager = ConfigManager(config_dir)
    neo4j_config = config_manager.neo4j_config
    
    # Initialize database connections
    print("📊 Initializing database connections...")
    
    graph_db = GraphDB(
        uri=neo4j_config["uri"],
        user=neo4j_config["username"],
        password=neo4j_config["password"]
    )
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    vector_indexer = VectorIndexer(
        paper_root=os.path.join(project_root, "data", "papers"),
        index_path=os.path.join(project_root, "data", "vector_index")
    )
    
    # Initialize the enhanced multi-agent system
    print("🤖 Initializing Enhanced Multi-Agent System...")
    
    agent_system = EnhancedMultiAgentSystem(
        graph_db=graph_db,
        vector_indexer=vector_indexer,
        config_path=os.path.join(project_root, "config", "model_config.json")
    )
    
    print("✅ System initialized successfully!")
    print()
    
    # Demo scenarios in different languages
    demo_queries = [
        {
            "query": "What does Michael Porter's 1980 paper discuss?",
            "language": "Chinese",
            "description": "Chinese query about Michael Porter's 1980 papers"
        },
        {
            "query": "Qu'est-ce que Michael Porter a écrit en 1980?",
            "language": "French",
            "description": "French query about Michael Porter's 1980 papers"
        },
        {
            "query": "Who cites Porter's competitive strategy work?",
            "language": "English",
            "description": "English citation analysis query"
        },
        {
            "query": "Strategic management theory",
            "language": "Chinese",
            "description": "Chinese semantic search query"
        },
        {
            "query": "John Smith 2000",  # Likely ambiguous/no results
            "language": "English", 
            "description": "Ambiguous query that should trigger clarification"
        }
    ]
    
    # Run demo queries
    for i, demo in enumerate(demo_queries, 1):
        print(f"🔍 Demo {i}: {demo['description']}")
        print(f"   Language: {demo['language']}")
        print(f"   Query: \"{demo['query']}\"")
        print()
        
        try:
            # Query the system
            result = await agent_system.query(
                user_query=demo['query'],
                thread_id=f"demo_thread_{i}",
                user_id="demo_user"
            )
            
            # Display results
            print("📋 Results:")
            print(f"   Response Language: {result.get('response_language', 'unknown')}")
            print(f"   Query Type: {result.get('query_type', 'unknown')}")
            print(f"   Action: {result.get('action', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print()
            
            print("💬 Response:")
            print(f"   {result.get('response', 'No response')}")
            print()
            
            # Show additional info if available
            if result.get('candidate_papers'):
                print(f"📄 Found {len(result['candidate_papers'])} candidate papers")
            
            if result.get('errors'):
                print(f"❌ Errors: {result['errors']}")
            
            if result.get('warnings'):
                print(f"⚠️  Warnings: {result['warnings']}")
            
            if result.get('debug_messages'):
                print("🔧 Debug Info:")
                for msg in result['debug_messages'][:3]:  # Show first 3
                    print(f"   - {msg}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("-" * 50)
        print()
    
    # Demo conversation memory
    print("🧠 Testing Conversation Memory...")
    
    # Series of related queries in the same thread
    memory_queries = [
        "Michael Porter competitive strategy",
        "Who cites this paper?",  # Should reference previous context
        "What about his five forces model?"  # Should build on conversation
    ]
    
    thread_id = "memory_demo_thread"
    
    for i, query in enumerate(memory_queries, 1):
        print(f"Memory Query {i}: \"{query}\"")
        
        try:
            result = await agent_system.query(
                user_query=query,
                thread_id=thread_id,
                user_id="memory_demo_user"
            )
            
            print(f"Response: {result.get('response', 'No response')[:200]}...")
            print()
            
        except Exception as e:
            print(f"Error: {e}")
        
    print("=" * 50)
    print("✅ Demo completed!")
    
    # Cleanup
    graph_db.close()

def demo_configuration():
    """Demo the configuration and model setup"""
    
    print("⚙️  Configuration Demo")
    print("=" * 30)
    
    try:
        from src.llm.enhanced_llm_manager import EnhancedLLMManager
        
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "model_config.json")
        llm_manager = EnhancedLLMManager(config_path)
        
        print(f"Supported Languages: {llm_manager.get_supported_languages()}")
        print()
        
        print("Agent Configurations:")
        for agent_name in ["language_processor", "query_analyzer", "paper_disambiguator"]:
            config = llm_manager.get_agent_config(agent_name)
            print(f"  {agent_name}:")
            print(f"    Model: {config.get('model', 'unknown')}")
            print(f"    Provider: {config.get('provider', 'unknown')}")
            print(f"    Temperature: {config.get('temperature', 'unknown')}")
            print()
        
    except Exception as e:
        print(f"Configuration demo error: {e}")

async def test_language_processing():
    """Test language processing capabilities"""
    
    print("🌍 Language Processing Test")
    print("=" * 30)
    
    try:
        from src.llm.enhanced_llm_manager import EnhancedLLMManager
        
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "model_config.json")
        llm_manager = EnhancedLLMManager(config_path)
        
        test_queries = [
            "Michael Porter's 1980 paper",
            "Qu'est-ce que Michael Porter a écrit?",
            "Was hat Michael Porter geschrieben?",
            "Michael Porter's competitive strategy"
        ]
        
        for query in test_queries:
            try:
                translated, detected = await llm_manager.process_language(query, "en")
                print(f"Original ({detected}): {query}")
                print(f"Translated (en): {translated}")
                print()
            except Exception as e:
                print(f"Translation error for '{query}': {e}")
                
    except Exception as e:
        print(f"Language processing test error: {e}")

if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Full Multi-Agent Demo (requires database)")
    print("2. Configuration Demo")
    print("3. Language Processing Test")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(main())
    elif choice == "2":
        demo_configuration()
    elif choice == "3":
        asyncio.run(test_language_processing())
    else:
        print("Invalid choice. Running configuration demo...")
        demo_configuration() 