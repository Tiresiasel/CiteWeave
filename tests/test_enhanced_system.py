#!/usr/bin/env python3
"""
测试改进的多智能体系统
Test the enhanced multi-agent system with multi-language support and intelligent routing
"""

import asyncio
import os
import sys
sys.path.append('src')

from src.enhanced_multi_agent_system import EnhancedMultiAgentSystem
from src.graph_builder import GraphDB
from src.vector_indexer import MultiLevelVectorIndexer
from src.author_paper_index import AuthorPaperIndex

async def test_enhanced_system():
    """测试改进的多智能体系统"""
    
    print("🚀 测试改进的多智能体系统")
    print("=" * 50)
    
    try:
        # 初始化组件
        print("📊 正在初始化系统组件...")
        
        # 初始化图数据库
        graph_db = GraphDB(
            uri="bolt://localhost:7687",
            user="neo4j", 
            password="password"
        )
        
        # 初始化多层向量索引器
        vector_indexer = MultiLevelVectorIndexer(
            paper_root="data/papers",
            index_path="data/vector_index"
        )
        
        # 初始化作者索引
        author_index = AuthorPaperIndex(
            storage_root="data/papers",
            index_db_path="data/author_paper_index.db"
        )
        
        # 初始化多智能体系统
        agent_system = EnhancedMultiAgentSystem(
            graph_db=graph_db,
            vector_indexer=vector_indexer,
            author_index=author_index,
            config_path="config/model_config.json"
        )
        
        print("✅ 系统组件初始化完成")
        print()
        
        # 测试用例
        test_queries = [
            # 中文概念定义查询 (应该路由到 vector_search)
            {
                "query": "什么是竞争优势？",
                "expected_routes": ["vector_search"],
                "description": "中文概念定义查询"
            },
            
            # 英文引用分析查询 (应该路由到 graph_analysis)
            {
                "query": "How is Porter cited in academic papers?",
                "expected_routes": ["graph_analysis"],
                "description": "英文引用分析查询"
            },
            
            # 中文文档内容查询 (应该路由到 pdf_analysis 或 author_collection)
            {
                "query": "Porter的文章讲了什么？",
                "expected_routes": ["pdf_analysis", "author_collection"],
                "description": "中文文档内容查询"
            },
            
            # 复杂查询 (应该路由到多个数据源)
            {
                "query": "Porter的竞争战略理论是什么以及如何被引用？",
                "expected_routes": ["vector_search", "graph_analysis"],
                "description": "复杂多数据源查询"
            }
        ]
        
        # 执行测试
        for i, test_case in enumerate(test_queries, 1):
            print(f"🧪 测试案例 {i}: {test_case['description']}")
            print(f"📝 查询: {test_case['query']}")
            
            try:
                # 执行查询
                result = await agent_system.query(
                    test_case["query"],
                    thread_id=f"test_{i}",
                    user_id="test_user"
                )
                
                # 显示结果
                print(f"🌐 检测语言: {result.get('user_language', 'unknown')}")
                print(f"🔄 使用翻译: {result.get('translation_used', False)}")
                print(f"🎯 识别路由: {result.get('required_routes', [])}")
                print(f"✅ 完成路由: {result.get('completed_routes', [])}")
                print(f"💯 置信度: {result.get('confidence', 0):.2f}")
                
                if result.get('errors'):
                    print(f"❌ 错误: {result['errors']}")
                
                if result.get('warnings'):
                    print(f"⚠️  警告: {result['warnings']}")
                
                # 显示响应片段
                response = result.get('response', 'No response')
                if len(response) > 200:
                    response = response[:200] + "..."
                print(f"💬 响应: {response}")
                
                print(f"🔍 调试信息:")
                for debug_msg in result.get('debug_messages', [])[:3]:  # 只显示前3条
                    print(f"   - {debug_msg}")
                
            except Exception as e:
                print(f"❌ 查询失败: {str(e)}")
            
            print()
            print("-" * 50)
            print()
        
        print("🎉 测试完成！")
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            if 'graph_db' in locals():
                graph_db.close()
        except:
            pass

def test_basic_components():
    """测试基础组件功能"""
    print("🔧 测试基础组件功能")
    print("=" * 30)
    
    try:
        # 测试作者索引
        print("📚 测试作者索引...")
        author_index = AuthorPaperIndex()
        stats = author_index.get_statistics()
        print(f"   论文总数: {stats['total_papers']}")
        print(f"   作者总数: {stats['total_authors']}")
        print(f"   PDF可用率: {stats['pdf_availability_rate']:.1%}")
        
        # 测试向量索引器
        print("🔍 测试向量索引器...")
        vector_indexer = MultiLevelVectorIndexer()
        # 简单测试搜索功能
        results = vector_indexer.smart_search("competitive advantage", limit=3)
        print(f"   搜索结果数量: {len(results)}")
        
        print("✅ 基础组件测试完成")
        
    except Exception as e:
        print(f"❌ 基础组件测试失败: {str(e)}")

if __name__ == "__main__":
    print("🚀 CiteWeave 改进多智能体系统测试")
    print("=" * 60)
    
    # 首先测试基础组件
    test_basic_components()
    print()
    
    # 然后测试完整系统 (需要LLM配置)
    if os.path.exists("config/model_config.json"):
        print("🤖 发现模型配置，开始测试完整系统...")
        asyncio.run(test_enhanced_system())
    else:
        print("⚠️  未发现模型配置文件，跳过LLM测试")
        print("   要测试完整功能，请配置 config/model_config.json") 