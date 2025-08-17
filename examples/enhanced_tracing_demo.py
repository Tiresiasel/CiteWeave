#!/usr/bin/env python3
"""
Enhanced Tracing Demo for CiteWeave
Demonstrates the new multi-agent workflow tracing capabilities
"""

import os
import sys
import time
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def demo_enhanced_tracing():
    """Demonstrate enhanced tracing capabilities"""
    
    print("🚀 Enhanced CiteWeave Tracing Demo")
    print("=" * 60)
    print()
    
    try:
        # Import the enhanced tracing system
        from src.storage.query_trace_storage import QueryTraceStorage
        
        print("✅ Successfully imported enhanced tracing system")
        print()
        
        # Initialize storage
        storage = QueryTraceStorage()
        print("✅ QueryTraceStorage initialized")
        print()
        
        # Show available methods
        print("🔍 Available Enhanced Tracing Methods:")
        print("  - start_query_trace() - Start tracking a new query")
        print("  - add_workflow_trace() - Record workflow steps")
        print("  - add_agent_interaction() - Record agent communications")
        print("  - add_query_step() - Record execution steps")
        print("  - get_workflow_trace() - Get complete workflow analysis")
        print("  - get_agent_interactions() - Get agent interaction details")
        print("  - get_complete_workflow_analysis() - Get full analysis")
        print()
        
        # Create a sample query trace
        print("📝 Creating Sample Query Trace...")
        query_id = storage.start_query_trace(
            query_text="Sample query for demonstration",
            thread_id="demo_thread",
            user_id="demo_user",
            query_type="demo",
            language="en"
        )
        print(f"✅ Created query trace: {query_id}")
        print()
        
        # Add workflow trace steps
        print("🔄 Adding Workflow Trace Steps...")
        
        # Step 1: Query Start
        storage.add_workflow_trace(
            query_id=query_id,
            workflow_step="query_start",
            agent_name="system",
            routing_decision="start_demo_query",
            input_state={"query": "Sample query", "user": "demo_user"},
            output_state={"status": "started"},
            decision_reasoning="Demo query initiated",
            next_agent="query_analyzer",
            execution_order=1
        )
        print("  ✅ Added workflow step: query_start")
        
        # Step 2: Query Analysis
        storage.add_workflow_trace(
            query_id=query_id,
            workflow_step="query_analysis",
            agent_name="query_analyzer",
            routing_decision="analyze_query_intent",
            input_state={"query": "Sample query"},
            output_state={"intent": "information_retrieval", "complexity": "medium"},
            decision_reasoning="Query requires information retrieval with medium complexity",
            next_agent="research_planner",
            execution_order=2
        )
        print("  ✅ Added workflow step: query_analysis")
        
        # Step 3: Research Planning
        storage.add_workflow_trace(
            query_id=query_id,
            workflow_step="research_planning",
            agent_name="research_planner",
            routing_decision="plan_research_strategy",
            input_state={"intent": "information_retrieval", "complexity": "medium"},
            output_state={"strategy": "multi_source_retrieval", "sources": ["vector_db", "graph_db"]},
            decision_reasoning="Multi-source approach needed for medium complexity query",
            next_agent="data_retriever",
            execution_order=3
        )
        print("  ✅ Added workflow step: research_planning")
        
        # Step 4: Data Retrieval
        storage.add_workflow_trace(
            query_id=query_id,
            workflow_step="data_retrieval",
            agent_name="data_retriever",
            routing_decision="retrieve_from_multiple_sources",
            input_state={"strategy": "multi_source_retrieval", "sources": ["vector_db", "graph_db"]},
            output_state={"results": {"vector_db": 5, "graph_db": 3}, "total_items": 8},
            decision_reasoning="Successfully retrieved data from both sources",
            next_agent="information_synthesizer",
            execution_order=4
        )
        print("  ✅ Added workflow step: data_retrieval")
        
        # Step 5: Information Synthesis
        storage.add_workflow_trace(
            query_id=query_id,
            workflow_step="information_synthesis",
            agent_name="information_synthesizer",
            routing_decision="synthesize_final_answer",
            input_state={"results": {"vector_db": 5, "graph_db": 3}, "total_items": 8},
            output_state={"synthesized_answer": "Comprehensive response generated", "confidence": "high"},
            decision_reasoning="Sufficient data collected, high confidence synthesis possible",
            next_agent="response_generator",
            execution_order=5
        )
        print("  ✅ Added workflow step: information_synthesis")
        
        # Step 6: Response Generation
        storage.add_workflow_trace(
            query_id=query_id,
            workflow_step="response_generation",
            agent_name="response_generator",
            routing_decision="generate_final_response",
            input_state={"synthesized_answer": "Comprehensive response generated", "confidence": "high"},
            output_state={"final_response": "This is a comprehensive answer based on 8 data sources"},
            decision_reasoning="High confidence synthesis allows direct response generation",
            next_agent="completion",
            execution_order=6
        )
        print("  ✅ Added workflow step: response_generation")
        print()
        
        # Add agent interactions
        print("🤝 Adding Agent Interactions...")
        
        storage.add_agent_interaction(
            query_id=query_id,
            from_agent="query_analyzer",
            to_agent="research_planner",
            interaction_type="data_transfer",
            data_passed={"intent": "information_retrieval", "complexity": "medium"},
            routing_condition="intent_analysis_complete"
        )
        print("  ✅ Added interaction: query_analyzer -> research_planner")
        
        storage.add_agent_interaction(
            query_id=query_id,
            from_agent="research_planner",
            to_agent="data_retriever",
            interaction_type="strategy_transfer",
            data_passed={"strategy": "multi_source_retrieval", "sources": ["vector_db", "graph_db"]},
            routing_condition="research_plan_complete"
        )
        print("  ✅ Added interaction: research_planner -> data_retriever")
        
        storage.add_agent_interaction(
            query_id=query_id,
            from_agent="data_retriever",
            to_agent="information_synthesizer",
            interaction_type="results_transfer",
            data_passed={"results": {"vector_db": 5, "graph_db": 3}, "total_items": 8},
            routing_condition="data_retrieval_complete"
        )
        print("  ✅ Added interaction: data_retriever -> information_synthesizer")
        print()
        
        # Add execution steps
        print("⚡ Adding Execution Steps...")
        
        storage.add_query_step(
            query_id=query_id,
            agent_name="query_analyzer",
            step_number=1,
            input_data={"query": "Sample query"},
            output_data={"intent": "information_retrieval", "complexity": "medium"},
            execution_time=0.15,
            memory_usage=1024
        )
        print("  ✅ Added execution step: query_analyzer")
        
        storage.add_query_step(
            query_id=query_id,
            agent_name="research_planner",
            step_number=2,
            input_data={"intent": "information_retrieval", "complexity": "medium"},
            output_data={"strategy": "multi_source_retrieval", "sources": ["vector_db", "graph_db"]},
            execution_time=0.23,
            memory_usage=1536
        )
        print("  ✅ Added execution step: research_planner")
        
        storage.add_query_step(
            query_id=query_id,
            agent_name="data_retriever",
            step_number=3,
            input_data={"strategy": "multi_source_retrieval", "sources": ["vector_db", "graph_db"]},
            output_data={"results": {"vector_db": 5, "graph_db": 3}, "total_items": 8},
            execution_time=0.45,
            memory_usage=2048
        )
        print("  ✅ Added execution step: data_retriever")
        print()
        
        # Complete the query trace
        print("✅ Completing Query Trace...")
        storage.complete_query_trace(
            query_id=query_id,
            final_response="This is a comprehensive answer based on 8 data sources",
            confidence_score=0.9,
            total_execution_time=0.83,
            total_memory_usage=4608,
            data_sources_used=["vector_db", "graph_db"],
            errors=[],
            warnings=[]
        )
        print("✅ Query trace completed")
        print()
        
        # Now demonstrate the enhanced analysis capabilities
        print("🔍 Demonstrating Enhanced Analysis Capabilities...")
        print()
        
        # Get workflow trace
        workflow_trace = storage.get_workflow_trace(query_id)
        print(f"📊 Workflow Trace: {len(workflow_trace)} steps")
        for step in workflow_trace:
            print(f"  {step['execution_order']}. {step['workflow_step']} -> {step['agent_name']}")
            print(f"     Decision: {step['routing_decision']}")
            print(f"     Next: {step['next_agent']}")
            print()
        
        # Get agent interactions
        interactions = storage.get_agent_interactions(query_id)
        print(f"🤝 Agent Interactions: {len(interactions)} interactions")
        for interaction in interactions:
            print(f"  {interaction['from_agent']} -> {interaction['to_agent']} ({interaction['interaction_type']})")
            if interaction['routing_condition']:
                print(f"     Condition: {interaction['routing_condition']}")
            print()
        
        # Get complete workflow analysis
        complete_analysis = storage.get_complete_workflow_analysis(query_id)
        print("📈 Complete Workflow Analysis:")
        print(f"  Total Workflow Steps: {complete_analysis['total_workflow_steps']}")
        print(f"  Total Interactions: {complete_analysis['total_interactions']}")
        print(f"  Total Execution Steps: {complete_analysis['total_execution_steps']}")
        print()
        
        print("🎉 Enhanced Tracing Demo Completed Successfully!")
        print()
        print("💡 What You Can Now Track:")
        print("  🔄 Complete workflow execution path")
        print("  🤝 Agent-to-agent communication")
        print("  ⚡ Performance metrics for each step")
        print("  🧠 Decision reasoning and routing logic")
        print("  📊 Input/output state for each agent")
        print()
        print("💡 Next Steps:")
        print("  1. Run actual queries to see real workflow tracing")
        print("  2. Analyze agent performance and routing patterns")
        print("  3. Optimize your multi-agent system based on traces")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_enhanced_tracing()
