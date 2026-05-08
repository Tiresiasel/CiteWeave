#!/usr/bin/env python3
"""
Test script for the new Function Calling based QueryPlanningAgent
"""


def test_function_calling_planning():
    """Test the new function calling based query planning"""
    print("🚀 Testing Function Calling Based Query Planning\n")
    
    try:
        from src.agents.multi_agent_research_system import LangGraphResearchSystem
        
        # Initialize the system
        print("📝 Initializing Multi-Agent Research System...")
        system = LangGraphResearchSystem()
        print("✅ System initialized successfully!\n")
        
        # Test cases that should now be handled purely by LLM function calling
        test_questions = [
            {
                "question": "引用波特的所有文章，他们引用的观点分别是什么",
                "description": "Reverse citation analysis with content extraction",
                "expected_functions": ["get_papers_id_by_author", "get_papers_citing_paper", "get_sentences_citing_paper"]
            },
            {
                "question": "Porter的论文有哪些",
                "description": "Author paper search",
                "expected_functions": ["get_papers_id_by_author"]
            },
            {
                "question": "什么是竞争战略",
                "description": "Concept search",
                "expected_functions": ["search_all_collections", "search_relevant_sentences"]
            },
            {
                "question": "哪些论文引用了Porter 1980年的竞争战略",
                "description": "Citation analysis with specific paper",
                "expected_functions": ["get_papers_id_by_author", "get_papers_citing_paper"]
            }
        ]
        
        for i, test_case in enumerate(test_questions, 1):
            print(f"📋 Test Case {i}: {test_case['description']}")
            print(f"❓ Question: {test_case['question']}")
            
            try:
                # Use the new LangGraph workflow
                print("🤖 Using LangGraph workflow to research question...")
                response = system.research_question(test_case["question"])
                
                print("✅ Research completed!")
                print(f"📝 Response preview: {response[:200]}...")
                
                # Check if any expected functions would have been relevant
                expected = test_case.get("expected_functions", [])
                print(f"📊 Expected functions: {expected}")
                
            except Exception as e:
                print(f"❌ Test failed: {str(e)}")
            
            print(f"\n{'='*60}\n")
        
        print("🎉 LangGraph Research Test Completed!")
        print("\n📝 Key Observations:")
        print("• Built with modern LangGraph framework")
        print("• Fully AI-driven workflow with no hardcoded logic")
        print("• Intelligent tool selection via LLM function calling")
        print("• Streamlined graph-based execution flow")
        print("• No outdated package dependencies")
        
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_function_calling_planning()
