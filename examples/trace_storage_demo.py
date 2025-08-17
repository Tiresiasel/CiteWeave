#!/usr/bin/env python3
"""
Demo script for the new Query Trace Storage system
Shows how to use the persistent storage features
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.multi_agent_system import LangGraphResearchSystem
from storage.graph_builder import GraphDB
from storage.vector_indexer import VectorIndexer
from storage.author_paper_index import AuthorPaperIndex

async def demo_trace_storage():
    """Demonstrate the trace storage functionality"""
    
    print("🚀 Starting Query Trace Storage Demo...")
    
    # Initialize the system (you'll need to provide actual config paths)
    try:
        # Initialize components (these would normally come from your config)
        graph_db = GraphDB("bolt://localhost:7687", "neo4j", "password")
        vector_indexer = VectorIndexer("http://localhost:6333")
        author_index = AuthorPaperIndex("data/author_paper_index.db")
        
        # Initialize the multi-agent system
        system = LangGraphResearchSystem(
            graph_db=graph_db,
            vector_indexer=vector_indexer,
            author_index=author_index
        )
        
        print("✅ System initialized successfully")
        
        # Example 1: Get database statistics
        print("\n📊 Database Statistics:")
        stats = system.get_database_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Example 2: Process a query and trace it
        print("\n🔍 Processing a sample query...")
        query = "Which papers cite Porter's competitive strategy theory?"
        
        result = await system.query(
            user_query=query,
            thread_id="demo_session_001",
            user_id="demo_user"
        )
        
        print(f"✅ Query completed with response: {result['response'][:100]}...")
        
        # Example 3: Get session queries
        print("\n📋 Session Queries:")
        session_queries = system.get_session_queries("demo_session_001", limit=10)
        for query_info in session_queries:
            print(f"  Query: {query_info['query_text'][:50]}...")
            print(f"    Status: {query_info['execution_status']}")
            print(f"    Steps: {query_info['total_steps']}")
            print(f"    Created: {query_info['created_at']}")
            print()
        
        # Example 4: Get agent performance statistics
        print("\n📈 Agent Performance Statistics:")
        agent_stats = system.get_agent_performance_stats("_language_processor_agent", days=7)
        for key, value in agent_stats.items():
            print(f"  {key}: {value}")
        
        # Example 5: Cleanup old traces (optional)
        print("\n🧹 Cleaning up old traces (older than 30 days)...")
        system.cleanup_old_traces(days=30)
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure your Neo4j and Qdrant services are running")

def demo_storage_only():
    """Demo using just the storage system without full agent system"""
    
    print("🔧 Storage System Only Demo...")
    
    try:
        from storage.query_trace_storage import QueryTraceStorage
        
        # Initialize storage
        storage = QueryTraceStorage("logs/demo_traces.db")
        
        # Create a sample query trace
        query_id = storage.start_query_trace(
            query_text="Sample query for demo",
            thread_id="demo_thread",
            user_id="demo_user",
            query_type="demo",
            language="en"
        )
        
        print(f"✅ Created query trace: {query_id}")
        
        # Add some sample steps
        storage.add_query_step(
            query_id=query_id,
            agent_name="demo_agent",
            step_number=1,
            input_data={"input": "sample input"},
            output_data={"output": "sample output"},
            execution_time=0.5,
            memory_usage=1024
        )
        
        storage.add_query_step(
            query_id=query_id,
            agent_name="demo_agent_2",
            step_number=2,
            input_data={"input": "step 2 input"},
            output_data={"output": "step 2 output"},
            execution_time=0.3,
            memory_usage=512
        )
        
        # Complete the trace
        storage.complete_query_trace(
            query_id=query_id,
            final_response="Demo response",
            confidence_score=0.9,
            total_execution_time=0.8,
            total_memory_usage=1536,
            data_sources_used=["demo_source"],
            errors=[],
            warnings=[]
        )
        
        print("✅ Added sample steps and completed trace")
        
        # Retrieve the trace
        trace = storage.get_query_trace(query_id)
        print(f"\n📋 Retrieved trace:")
        print(f"  Query: {trace['query_info']['query_text']}")
        print(f"  Steps: {len(trace['steps'])}")
        print(f"  Status: {trace['query_info']['execution_status']}")
        
        # Get database stats
        stats = storage.get_database_stats()
        print(f"\n📊 Database stats: {stats}")
        
        print("\n🎉 Storage demo completed!")
        
    except Exception as e:
        print(f"❌ Error in storage demo: {e}")

if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Full system demo (requires running services)")
    print("2. Storage system only demo")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(demo_trace_storage())
    elif choice == "2":
        demo_storage_only()
    else:
        print("Invalid choice. Running storage demo only...")
        demo_storage_only()
