#!/usr/bin/env python3
"""
测试智能查询规划智能体
"""
import sys
import os
sys.path.append('/Users/tiresias/Documents/projects/CiteWeave/src')

from multi_agent_research_system import MultiAgentResearchSystem

def test_intelligent_planning():
    """测试智能查询规划功能"""
    print("🧠 Testing Intelligent Query Planning Agent")
    print("=" * 60)
    
    # 初始化系统
    try:
        system = MultiAgentResearchSystem()
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # 测试不同类型的问题
    test_questions = [
        {
            "question": "引用波特的所有文章，他们引用的观点分别是什么",
            "expected_focus": "reverse_citation_with_content",
            "expected_db": "graph_db or vector_db"
        },
        {
            "question": "Porter的论文有哪些", 
            "expected_focus": "author_analysis",
            "expected_db": "graph_db"
        },
        {
            "question": "什么是竞争战略",
            "expected_focus": "content_analysis", 
            "expected_db": "vector_db"
        }
    ]
    
    for i, test_case in enumerate(test_questions, 1):
        print(f"\n🔍 Test {i}: {test_case['question']}")
        print("-" * 60)
        
        try:
            # Step 1: 问题分析
            intent = system.question_analysis_agent.analyze_question(test_case["question"])
            print(f"📋 Query Intent: {intent.query_type.value}")
            print(f"🎯 Target Entity: {intent.target_entity}")
            
            # Step 2: 模糊匹配
            matches, confidence = system.fuzzy_matching_agent.find_matching_entities(
                intent.target_entity, intent.entity_type
            )
            print(f"🔍 Found {len(matches)} matches with confidence {confidence:.2f}")
            
            # Step 3: 选择目标实体
            target_entity = matches[0] if matches else {"name": intent.target_entity}
            print(f"🎯 Selected Entity: {target_entity}")
            
            # Step 4: 智能查询规划
            query_plan = system.query_planning_agent.create_query_plan(intent, target_entity)
            print(f"\n📋 Generated Query Plan:")
            print(f"🧠 Reasoning: {query_plan.get('reasoning', 'No reasoning')}")
            print(f"📊 Query Steps: {len(query_plan['query_sequence'])}")
            
            for step in query_plan["query_sequence"]:
                print(f"  Step {step['step']}: {step['database']}.{step['method']}")
                print(f"    💡 Reasoning: {step.get('reasoning', 'No reasoning')}")
                print(f"    ⚡ Required: {step.get('required', False)}")
            
            # Step 5: 执行查询计划（只显示计划，不实际执行以避免数据库错误）
            print(f"\n✅ Query plan generated successfully!")
            print(f"Expected focus: {test_case['expected_focus']}")
            print(f"Expected database: {test_case['expected_db']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    test_intelligent_planning() 