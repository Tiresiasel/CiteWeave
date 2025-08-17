#!/usr/bin/env python3
"""
Optimized Tracing Demo for CiteWeave
Demonstrates the new optimized multi-agent workflow tracing system
"""

import os
import sys
import time
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def demo_optimized_tracing():
    """Demonstrate optimized tracing capabilities"""
    
    print("🚀 Optimized CiteWeave Tracing Demo")
    print("=" * 60)
    print()
    
    try:
        # Import the optimized tracing system
        from src.storage.query_trace_storage import QueryTraceStorage
        
        print("✅ Successfully imported optimized tracing system")
        print()
        
        # Initialize storage
        storage = QueryTraceStorage()
        print("✅ QueryTraceStorage initialized")
        print()
        
        # Show available methods
        print("🔍 Available Optimized Tracing Methods:")
        print("  - start_query_trace() - Start tracking with config")
        print("  - add_execution_step() - Record unified execution step")
        print("  - add_agent_interaction() - Record agent interactions")
        print("  - get_execution_steps() - Get unified execution data")
        print("  - get_agent_interactions() - Get interaction details")
        print("  - get_complete_workflow_analysis() - Get full analysis")
        print()
        
        # Create a sample query trace with enhanced configuration
        print("📝 Creating Sample Query Trace with Enhanced Config...")
        
        workflow_config = {
            "workflow_type": "research_pipeline",
            "version": "2.0",
            "agents": ["query_analyzer", "research_planner", "data_retriever", "synthesizer"],
            "routing_rules": "confidence_based"
        }
        
        agent_config = {
            "llm_model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "retry_attempts": 3
        }
        
        query_id = storage.start_query_trace(
            query_text="Sample optimized query for demonstration",
            thread_id="demo_thread",
            user_id="demo_user",
            query_type="optimized_demo",
            language="en",
            workflow_config=workflow_config,
            agent_config=agent_config
        )
        print(f"✅ Created query trace: {query_id}")
        print()
        
        # Add execution steps with enhanced data
        print("🔄 Adding Optimized Execution Steps...")
        
        # Step 1: Query Analysis
        step1_id = storage.add_execution_step(
            query_id=query_id,
            step_order=1,
            agent_name="query_analyzer",
            workflow_step="query_analysis",
            routing_decision="analyze_intent",
            input_state={"query": "Sample query", "user_context": "demo_user"},
            output_state={"intent": "information_retrieval", "complexity": "medium", "confidence": 0.85},
            decision_reasoning="Query requires information retrieval with medium complexity",
            next_agent="research_planner",
            execution_time=0.12,
            memory_usage=1024,
            agent_state={"status": "completed", "confidence": 0.85, "analysis_depth": "medium"}
        )
        print("  ✅ Added execution step: query_analysis")
        
        # Step 2: Research Planning
        step2_id = storage.add_execution_step(
            query_id=query_id,
            step_order=2,
            agent_name="research_planner",
            workflow_step="research_planning",
            routing_decision="plan_multi_source_strategy",
            input_state={"intent": "information_retrieval", "complexity": "medium"},
            output_state={"strategy": "multi_source_retrieval", "sources": ["vector_db", "graph_db"], "priority": "high"},
            decision_reasoning="Multi-source approach needed for medium complexity query",
            next_agent="data_retriever",
            execution_time=0.18,
            memory_usage=1536,
            agent_state={"status": "completed", "strategy_confidence": 0.9, "estimated_time": "2-3s"}
        )
        print("  ✅ Added execution step: research_planning")
        
        # Step 3: Data Retrieval
        step3_id = storage.add_execution_step(
            query_id=query_id,
            step_order=3,
            agent_name="data_retriever",
            workflow_step="data_retrieval",
            routing_decision="retrieve_from_multiple_sources",
            input_state={"strategy": "multi_source_retrieval", "sources": ["vector_db", "graph_db"]},
            output_state={"results": {"vector_db": 5, "graph_db": 3}, "total_items": 8, "quality_score": 0.87},
            decision_reasoning="Successfully retrieved data from both sources with high quality",
            next_agent="information_synthesizer",
            execution_time=0.45,
            memory_usage=2048,
            agent_state={"status": "completed", "retrieval_success": True, "quality_threshold_met": True}
        )
        print("  ✅ Added execution step: data_retrieval")
        
        # Step 4: Information Synthesis
        step4_id = storage.add_execution_step(
            query_id=query_id,
            step_order=4,
            agent_name="information_synthesizer",
            workflow_step="information_synthesis",
            routing_decision="synthesize_comprehensive_answer",
            input_state={"results": {"vector_db": 5, "graph_db": 3}, "total_items": 8, "quality_score": 0.87},
            output_state={"synthesized_answer": "Comprehensive response generated", "confidence": "high", "coverage": "complete"},
            decision_reasoning="High quality data allows comprehensive synthesis",
            next_agent="response_generator",
            execution_time=0.32,
            memory_usage=1792,
            agent_state={"status": "completed", "synthesis_quality": "high", "coverage_score": 0.95}
        )
        print("  ✅ Added execution step: information_synthesis")
        
        # Step 5: Response Generation
        step5_id = storage.add_execution_step(
            query_id=query_id,
            step_order=5,
            agent_name="response_generator",
            workflow_step="response_generation",
            routing_decision="generate_final_response",
            input_state={"synthesized_answer": "Comprehensive response generated", "confidence": "high"},
            output_state={"final_response": "This is a comprehensive answer based on 8 high-quality data sources"},
            decision_reasoning="High confidence synthesis allows direct response generation",
            next_agent="completion",
            execution_time=0.28,
            memory_usage=1280,
            agent_state={"status": "completed", "response_quality": "excellent", "readability_score": 0.92}
        )
        print("  ✅ Added execution step: response_generation")
        print()
        
        # Add agent interactions with step references
        print("🤝 Adding Agent Interactions with Step References...")
        
        storage.add_agent_interaction(
            query_id=query_id,
            from_step_id=step1_id,
            to_step_id=step2_id,
            interaction_type="data_transfer",
            data_passed={"intent": "information_retrieval", "complexity": "medium", "confidence": 0.85},
            routing_condition="intent_analysis_complete",
            interaction_metadata={"transfer_method": "direct", "data_size": "small", "priority": "high"}
        )
        print("  ✅ Added interaction: query_analyzer -> research_planner")
        
        storage.add_agent_interaction(
            query_id=query_id,
            from_step_id=step2_id,
            to_step_id=step3_id,
            interaction_type="strategy_transfer",
            data_passed={"strategy": "multi_source_retrieval", "sources": ["vector_db", "graph_db"], "priority": "high"},
            routing_condition="research_plan_complete",
            interaction_metadata={"transfer_method": "strategy_package", "data_size": "medium", "priority": "high"}
        )
        print("  ✅ Added interaction: research_planner -> data_retriever")
        
        storage.add_agent_interaction(
            query_id=query_id,
            from_step_id=step3_id,
            to_step_id=step4_id,
            interaction_type="results_transfer",
            data_passed={"results": {"vector_db": 5, "graph_db": 3}, "total_items": 8, "quality_score": 0.87},
            routing_condition="data_retrieval_complete",
            interaction_metadata={"transfer_method": "bulk_data", "data_size": "large", "priority": "critical"}
        )
        print("  ✅ Added interaction: data_retriever -> information_synthesizer")
        
        storage.add_agent_interaction(
            query_id=query_id,
            from_step_id=step4_id,
            to_step_id=step5_id,
            interaction_type="synthesis_transfer",
            data_passed={"synthesized_answer": "Comprehensive response generated", "confidence": "high", "coverage": "complete"},
            routing_condition="synthesis_complete",
            interaction_metadata={"transfer_method": "processed_data", "data_size": "medium", "priority": "high"}
        )
        print("  ✅ Added interaction: information_synthesizer -> response_generator")
        print()
        
        # Complete the query trace
        print("✅ Completing Query Trace...")
        storage.complete_query_trace(
            query_id=query_id,
            final_response="This is a comprehensive answer based on 8 high-quality data sources",
            confidence_score=0.92,
            total_execution_time=1.35,
            total_memory_usage=7680,
            data_sources_used=["vector_db", "graph_db"],
            errors=[],
            warnings=[]
        )
        print("✅ Query trace completed")
        print()
        
        # Demonstrate the enhanced analysis capabilities
        print("🔍 Demonstrating Enhanced Analysis Capabilities...")
        print()
        
        # Get execution steps
        execution_steps = storage.get_execution_steps(query_id)
        print(f"📊 Execution Steps: {len(execution_steps)} steps")
        for step in execution_steps:
            print(f"  {step['step_order']}. {step['workflow_step']} -> {step['agent_name']}")
            print(f"     Decision: {step['routing_decision']}")
            print(f"     Time: {step['execution_time']:.3f}s, Memory: {step['memory_usage']} bytes")
            print(f"     Next: {step['next_agent']}")
            if step['agent_state']:
                print(f"     State: {step['agent_state']}")
            print()
        
        # Get agent interactions
        interactions = storage.get_agent_interactions(query_id)
        print(f"🤝 Agent Interactions: {len(interactions)} interactions")
        for interaction in interactions:
            print(f"  {interaction['from_agent']} -> {interaction['to_agent']} ({interaction['interaction_type']})")
            if interaction['routing_condition']:
                print(f"     Condition: {interaction['routing_condition']}")
            if interaction['interaction_metadata']:
                print(f"     Metadata: {interaction['interaction_metadata']}")
            print()
        
        # Get complete workflow analysis
        complete_analysis = storage.get_complete_workflow_analysis(query_id)
        print("📈 Complete Workflow Analysis:")
        print(f"  Total Execution Steps: {complete_analysis['total_execution_steps']}")
        print(f"  Total Interactions: {complete_analysis['total_interactions']}")
        print(f"  Performance Metrics:")
        metrics = complete_analysis['performance_metrics']
        print(f"    Total Time: {metrics['total_execution_time']:.3f}s")
        print(f"    Total Memory: {metrics['total_memory_usage']} bytes")
        print(f"    Error Count: {metrics['error_count']}")
        print(f"    Avg Time: {metrics['avg_execution_time']:.3f}s")
        print(f"    Avg Memory: {metrics['avg_memory_usage']} bytes")
        print()
        
        print("🎉 Optimized Tracing Demo Completed Successfully!")
        print()
        print("💡 Key Improvements in the New Design:")
        print("  🔄 Unified execution tracking (no more duplicate tables)")
        print("  🤝 Precise agent interaction mapping via step IDs")
        print("  📊 Enhanced performance metrics and state tracking")
        print("  🧠 Better decision reasoning and routing logic")
        print("  📈 Comprehensive workflow analysis capabilities")
        print()
        print("💡 Benefits of the New Structure:")
        print("  1. Reduced data redundancy and storage overhead")
        print("  2. Improved query performance with better indexing")
        print("  3. More accurate agent interaction tracking")
        print("  4. Enhanced performance analysis and optimization")
        print("  5. Cleaner, more maintainable code structure")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_optimized_tracing()
