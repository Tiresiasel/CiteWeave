#!/usr/bin/env python3
"""
CiteWeave Multi-Agent System Query Test Runner
自动化测试运行器 - 运行各种类型的学术论文查询测试用例
"""

import json
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from enhanced_multi_agent_system import EnhancedMultiAgentSystem
from graph_builder import GraphDB
from llm_evaluator import LLMEvaluator, EvaluationResult
from vector_indexer import VectorIndexer
from author_paper_index import AuthorPaperIndex
from config_manager import ConfigManager

@dataclass
class TestResult:
    """测试结果数据类"""
    test_id: str
    category: str
    query_cn: str
    query_en: str
    expected_result_type: str
    complexity: str
    evaluation_criteria: List[str]
    
    # 运行结果
    success: bool = False
    response: str = ""
    confidence: float = 0.0
    query_type: str = ""
    action: str = ""
    response_language: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # 评估结果
    evaluation_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    execution_time: float = 0.0
    notes: str = ""
    
    # LLM评估结果
    llm_evaluation: Optional[EvaluationResult] = None
    enhanced_score: float = 0.0  # 包含LLM评估的增强评分

class QueryTestRunner:
    """查询测试运行器"""
    
    def __init__(self, config_dir: str = "../../config", enable_llm_evaluation: bool = True):
        """初始化测试运行器"""
        self.config_dir = config_dir
        self.agent_system = None
        self.test_cases = {}
        self.results = []
        self.enable_llm_evaluation = enable_llm_evaluation
        self.llm_evaluator = None
        
        # 初始化LLM评估器
        if self.enable_llm_evaluation:
            try:
                model_config_path = os.path.join(self.config_dir, "model_config.json")
                self.llm_evaluator = LLMEvaluator(model_config_path)
                print("✅ LLM评估器初始化成功")
            except Exception as e:
                print(f"⚠️ LLM评估器初始化失败，将使用基础评估: {e}")
                self.enable_llm_evaluation = False
        
    async def initialize_system(self):
        """初始化多智能体系统"""
        try:
            print("🔧 初始化测试环境...")
            
            # 加载配置
            config_manager = ConfigManager(self.config_dir)
            neo4j_config = config_manager.get_neo4j_config()
            
            # 初始化数据库连接
            graph_db = GraphDB(
                uri=neo4j_config["uri"],
                user=neo4j_config["user"],
                password=neo4j_config["password"]
            )
            
            # 获取项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            
            vector_indexer = VectorIndexer(
                paper_root=os.path.join(project_root, "data", "papers"),
                index_path=os.path.join(project_root, "data", "vector_index")
            )
            
            # 初始化作者索引
            author_index = AuthorPaperIndex(
                storage_root=os.path.join(project_root, "data", "papers"),
                index_db_path=os.path.join(project_root, "data", "author_paper_index.db")
            )
            
            # 初始化智能体系统
            self.agent_system = EnhancedMultiAgentSystem(
                graph_db=graph_db,
                vector_indexer=vector_indexer,
                author_index=author_index,
                config_path=os.path.join(self.config_dir, "model_config.json")
            )
            
            print("✅ 系统初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            return False
    
    def load_test_cases(self, test_file: str = "simplified_test_cases.json"):
        """加载测试用例"""
        try:
            test_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", test_file)
            with open(test_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.test_cases = data["test_cases_by_category"]
            
            print(f"📁 加载了 {len(self.test_cases)} 个测试类别")
            return True
            
        except Exception as e:
            print(f"❌ 加载测试用例失败: {e}")
            return False
    
    async def run_single_test(self, category: str, test_case: Dict[str, Any], language: str = "cn") -> TestResult:
        """运行单个测试用例"""
        start_time = datetime.now()
        
        # 创建测试结果对象
        result = TestResult(
            test_id=test_case["id"],
            category=category,
            query_cn=test_case["query_cn"],
            query_en=test_case["query_en"],
            expected_result_type=test_case["expected_result_type"],
            complexity=test_case["complexity"],
            evaluation_criteria=test_case["evaluation_criteria"]
        )
        
        try:
            # 选择查询语言
            query = test_case["query_cn"] if language == "cn" else test_case["query_en"]
            
            # 运行查询
            response = await self.agent_system.query(
                user_query=query,
                thread_id=f"test_{test_case['id']}",
                user_id="test_runner"
            )
            
            # 记录结果
            result.success = True
            result.response = response.get("response", "")
            result.confidence = response.get("confidence", 0.0)
            result.query_type = response.get("query_type", "")
            result.action = response.get("action", "")
            result.response_language = response.get("response_language", "")
            result.errors = response.get("errors", [])
            result.warnings = response.get("warnings", [])
            
            # 计算执行时间
            end_time = datetime.now()
            result.execution_time = (end_time - start_time).total_seconds()
            
            # 基础评估
            result.overall_score = self._evaluate_response(result)
            
            # LLM评估 (如果启用)
            if result.success and result.response:
                await self._perform_llm_evaluation(result)
                # 使用增强评分作为最终评分
                final_score = result.enhanced_score if result.enhanced_score > 0 else result.overall_score
                print(f"✅ {test_case['id']}: 基础评分={result.overall_score:.1f}/10, 最终评分={final_score:.1f}/10")
            else:
                print(f"✅ {test_case['id']}: {result.overall_score:.2f}/10.0")
            
        except Exception as e:
            result.success = False
            result.errors = [str(e)]
            result.overall_score = 0.0
            end_time = datetime.now()
            result.execution_time = (end_time - start_time).total_seconds()
            
            print(f"❌ {test_case['id']}: {str(e)}")
        
        return result
    
    def _evaluate_response(self, result: TestResult) -> float:
        """简单的响应评估函数"""
        score = 0.0
        
        # 基础分数 - 是否成功返回响应
        if result.success and result.response:
            score += 3.0
        
        # 置信度分数
        score += result.confidence * 2.0
        
        # 响应长度合理性 (100-2000字符为合理范围)
        if 100 <= len(result.response) <= 2000:
            score += 2.0
        elif 50 <= len(result.response) < 100:
            score += 1.0
        
        # 错误惩罚
        if result.errors:
            score -= len(result.errors) * 0.5
        
        # 警告惩罚
        if result.warnings:
            score -= len(result.warnings) * 0.2
        
        # 复杂度调整
        if result.complexity == "high" and score >= 7.0:
            score += 1.0  # 高复杂度问题的奖励
        
        return min(max(score, 0.0), 10.0)  # 限制在0-10范围内
    
    async def _enhanced_evaluate_response(self, result: TestResult) -> float:
        """增强的响应评估函数 - 结合基础评估和LLM评估"""
        # 基础评分 (0-10分)
        basic_score = self._evaluate_response(result)
        
        # 如果没有启用LLM评估或LLM评估失败，返回基础评分
        if not self.enable_llm_evaluation or not result.llm_evaluation:
            return basic_score
        
        # LLM评估权重配置
        llm_weight = 0.7  # LLM评估权重70%
        basic_weight = 0.3  # 基础评估权重30%
        
        # 计算LLM综合评分 (取overall_score)
        llm_score = result.llm_evaluation.overall_score
        
        # 计算加权最终评分
        enhanced_score = (llm_score * llm_weight) + (basic_score * basic_weight)
        
        return min(max(enhanced_score, 0.0), 10.0)  # 限制在0-10范围内
    
    async def _perform_llm_evaluation(self, result: TestResult):
        """为单个测试结果执行LLM评估"""
        if not self.enable_llm_evaluation or not self.llm_evaluator:
            return
        
        try:
            # 选择查询语言
            query = result.query_cn if result.response_language == "zh" else result.query_en
            
            # 执行LLM评估
            llm_eval = await self.llm_evaluator.evaluate_response(
                query=query,
                response=result.response,
                query_type=result.category,
                expected_criteria=result.evaluation_criteria
            )
            
            # 保存评估结果
            result.llm_evaluation = llm_eval
            
            # 计算增强评分
            result.enhanced_score = await self._enhanced_evaluate_response(result)
            
            print(f"🤖 LLM评估完成 - 总体评分: {llm_eval.overall_score:.1f}/10")
            
        except Exception as e:
            print(f"⚠️ LLM评估失败: {e}")
            result.enhanced_score = result.overall_score
    
    async def run_category_tests(self, category: str, language: str = "cn") -> List[TestResult]:
        """运行特定类别的所有测试"""
        if category not in self.test_cases:
            print(f"❌ 未找到测试类别: {category}")
            return []
        
        category_info = self.test_cases[category]
        test_cases = category_info["test_cases"]
        
        print(f"\n🔍 运行 '{category}' 类别测试 ({len(test_cases)} 个用例)")
        print(f"📋 检索策略: {category_info['retrieval_strategy']}")
        print("-" * 60)
        
        results = []
        for test_case in test_cases:
            result = await self.run_single_test(category, test_case, language)
            results.append(result)
            self.results.append(result)
            
            # 短暂延迟避免API限制
            await asyncio.sleep(1)
        
        return results
    
    async def run_all_tests(self, language: str = "cn") -> List[TestResult]:
        """运行所有测试用例"""
        print(f"\n🚀 开始运行所有测试用例 (语言: {language})")
        print("=" * 70)
        
        all_results = []
        for category in self.test_cases.keys():
            category_results = await self.run_category_tests(category, language)
            all_results.extend(category_results)
        
        return all_results
    
    def generate_report(self, output_file: str = None) -> str:
        """生成测试报告"""
        if not self.results:
            return "没有测试结果可以生成报告"
        
        # 计算统计信息
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        avg_score = sum(r.overall_score for r in self.results) / total_tests
        avg_time = sum(r.execution_time for r in self.results) / total_tests
        
        # LLM评估统计
        llm_evaluated_tests = sum(1 for r in self.results if r.llm_evaluation is not None)
        avg_enhanced_score = 0.0
        if llm_evaluated_tests > 0:
            enhanced_scores = [r.enhanced_score for r in self.results if r.enhanced_score > 0]
            avg_enhanced_score = sum(enhanced_scores) / len(enhanced_scores) if enhanced_scores else 0.0
        
        # 按类别统计
        category_stats = {}
        for result in self.results:
            if result.category not in category_stats:
                category_stats[result.category] = {
                    'total': 0, 'success': 0, 'scores': [], 'times': [], 
                    'llm_evaluated': 0, 'enhanced_scores': []
                }
            
            stats = category_stats[result.category]
            stats['total'] += 1
            if result.success:
                stats['success'] += 1
            stats['scores'].append(result.overall_score)
            stats['times'].append(result.execution_time)
            
            # LLM评估统计
            if result.llm_evaluation is not None:
                stats['llm_evaluated'] += 1
            if result.enhanced_score > 0:
                stats['enhanced_scores'].append(result.enhanced_score)
        
        # 生成报告
        evaluation_mode = "增强评估 (基础评估 + LLM评估)" if self.enable_llm_evaluation else "基础评估"
        
        report = f"""
# CiteWeave 多智能体系统测试报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
评估模式: {evaluation_mode}

## 📊 测试总览
- **总测试数**: {total_tests}
- **成功数**: {successful_tests} ({successful_tests/total_tests*100:.1f}%)
- **基础平均得分**: {avg_score:.2f}/10.0
- **平均执行时间**: {avg_time:.2f}秒"""

        # 添加LLM评估信息
        if self.enable_llm_evaluation:
            report += f"""
- **LLM评估覆盖**: {llm_evaluated_tests}/{total_tests} ({llm_evaluated_tests/total_tests*100:.1f}%)
- **增强平均得分**: {avg_enhanced_score:.2f}/10.0"""

        report += "\n\n## 📋 分类统计\n"
        
        for category, stats in category_stats.items():
            avg_cat_score = sum(stats['scores']) / len(stats['scores'])
            avg_cat_time = sum(stats['times']) / len(stats['times'])
            success_rate = stats['success'] / stats['total'] * 100
            
            report += f"""
### {category}
- 测试数: {stats['total']}
- 成功率: {success_rate:.1f}%
- 基础平均得分: {avg_cat_score:.2f}/10.0  
- 平均时间: {avg_cat_time:.2f}秒"""

            # 添加LLM评估信息
            if self.enable_llm_evaluation and stats['enhanced_scores']:
                avg_enhanced_cat_score = sum(stats['enhanced_scores']) / len(stats['enhanced_scores'])
                llm_coverage = stats['llm_evaluated'] / stats['total'] * 100
                report += f"""
- LLM评估覆盖: {stats['llm_evaluated']}/{stats['total']} ({llm_coverage:.1f}%)
- 增强平均得分: {avg_enhanced_cat_score:.2f}/10.0"""
            
            report += "\n"
        
        report += "\n## 🔍 详细结果\n"
        
        for result in self.results:
            status = "✅" if result.success else "❌"
            report += f"""
### {result.test_id} - {result.category} {status}
- **查询**: {result.query_cn}
- **基础得分**: {result.overall_score:.2f}/10.0"""

            # 添加增强评分信息
            if result.enhanced_score > 0:
                report += f"""
- **增强得分**: {result.enhanced_score:.2f}/10.0"""
            
            report += f"""
- **执行时间**: {result.execution_time:.2f}秒
- **置信度**: {result.confidence:.2f}
"""
            
            # 添加LLM评估详情
            if result.llm_evaluation:
                llm_eval = result.llm_evaluation
                report += f"""
#### 🤖 LLM评估详情
- **准确性**: {llm_eval.accuracy_score:.1f}/10
- **完整性**: {llm_eval.completeness_score:.1f}/10
- **相关性**: {llm_eval.relevance_score:.1f}/10
- **清晰度**: {llm_eval.clarity_score:.1f}/10
- **LLM总体评分**: {llm_eval.overall_score:.1f}/10
- **评估理由**: {llm_eval.reasoning}
- **优点**: {llm_eval.strengths}
- **不足**: {llm_eval.weaknesses}
- **改进建议**: {llm_eval.suggestions}
"""
            
            if result.errors:
                report += f"- **错误**: {'; '.join(result.errors)}\n"
            if result.warnings:
                report += f"- **警告**: {'; '.join(result.warnings)}\n"
        
        # 保存报告
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 报告已保存到: {output_file}")
        
        return report
    
    def export_results_csv(self, output_file: str):
        """导出结果为CSV格式"""
        if not self.results:
            print("没有结果可以导出")
            return
        
        # 准备数据
        data = []
        for result in self.results:
            row = {
                'test_id': result.test_id,
                'category': result.category,
                'query_cn': result.query_cn,
                'query_en': result.query_en,
                'complexity': result.complexity,
                'success': result.success,
                'overall_score': result.overall_score,
                'confidence': result.confidence,
                'execution_time': result.execution_time,
                'response_length': len(result.response),
                'error_count': len(result.errors),
                'warning_count': len(result.warnings),
                'query_type': result.query_type,
                'action': result.action
            }
            
            # 添加LLM评估字段
            if result.llm_evaluation:
                row.update({
                    'llm_evaluated': True,
                    'enhanced_score': result.enhanced_score,
                    'llm_accuracy_score': result.llm_evaluation.accuracy_score,
                    'llm_completeness_score': result.llm_evaluation.completeness_score,
                    'llm_relevance_score': result.llm_evaluation.relevance_score,
                    'llm_clarity_score': result.llm_evaluation.clarity_score,
                    'llm_overall_score': result.llm_evaluation.overall_score,
                    'llm_reasoning': result.llm_evaluation.reasoning,
                    'llm_strengths': result.llm_evaluation.strengths,
                    'llm_weaknesses': result.llm_evaluation.weaknesses,
                    'llm_suggestions': result.llm_evaluation.suggestions
                })
            else:
                row.update({
                    'llm_evaluated': False,
                    'enhanced_score': result.overall_score,
                    'llm_accuracy_score': None,
                    'llm_completeness_score': None,
                    'llm_relevance_score': None,
                    'llm_clarity_score': None,
                    'llm_overall_score': None,
                    'llm_reasoning': None,
                    'llm_strengths': None,
                    'llm_weaknesses': None,
                    'llm_suggestions': None
                })
            
            data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"📊 结果已导出到: {output_file}")

async def main():
    """主函数"""
    print("🎯 CiteWeave 多智能体查询测试系统")
    print("=" * 50)
    
    # 询问是否启用LLM评估
    print("\n🤖 评估模式选择:")
    print("1. 基础评估 (快速，仅基于长度、置信度等)")
    print("2. 增强评估 (使用GPT-4o-mini进行内容质量评估)")
    
    eval_choice = input("请选择评估模式 (1-2): ").strip()
    enable_llm_evaluation = eval_choice == "2"
    
    evaluation_mode = "增强评估模式" if enable_llm_evaluation else "基础评估模式"
    print(f"✅ 已选择: {evaluation_mode}")
    
    # 初始化测试运行器
    runner = QueryTestRunner(enable_llm_evaluation=enable_llm_evaluation)
    
    # 初始化系统
    if not await runner.initialize_system():
        print("系统初始化失败，退出测试")
        return
    
    # 加载测试用例
    if not runner.load_test_cases():
        print("测试用例加载失败，退出测试")
        return
    
    # 交互式选择
    print("\n请选择测试模式:")
    print("1. 运行所有测试 (中文)")
    print("2. 运行所有测试 (英文)")
    print("3. 运行特定类别测试")
    print("4. 显示测试类别列表")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if choice == "1":
        # 运行所有测试 (中文)
        await runner.run_all_tests("cn")
        report = runner.generate_report(f"test_reports/full_test_report_cn_{timestamp}.md")
        runner.export_results_csv(f"test_reports/full_test_results_cn_{timestamp}.csv")
        
    elif choice == "2":
        # 运行所有测试 (英文)
        await runner.run_all_tests("en")
        report = runner.generate_report(f"test_reports/full_test_report_en_{timestamp}.md")
        runner.export_results_csv(f"test_reports/full_test_results_en_{timestamp}.csv")
        
    elif choice == "3":
        # 运行特定类别
        print("\n可用的测试类别:")
        for i, category in enumerate(runner.test_cases.keys(), 1):
            print(f"{i}. {category}")
        
        cat_choice = input("请选择类别编号: ").strip()
        try:
            category_list = list(runner.test_cases.keys())
            selected_category = category_list[int(cat_choice) - 1]
            
            language = input("选择语言 (cn/en): ").strip() or "cn"
            
            await runner.run_category_tests(selected_category, language)
            report = runner.generate_report(f"test_reports/{selected_category}_test_report_{timestamp}.md")
            runner.export_results_csv(f"test_reports/{selected_category}_test_results_{timestamp}.csv")
            
        except (ValueError, IndexError):
            print("无效选择")
            return
            
    elif choice == "4":
        # 显示类别列表
        print("\n📋 测试类别详情:")
        for category, info in runner.test_cases.items():
            test_count = len(info["test_cases"])
            print(f"\n🔍 {category}")
            print(f"   描述: {info['description']}")
            print(f"   检索策略: {info['retrieval_strategy']}")
            print(f"   测试用例数: {test_count}")
            
            for test_case in info["test_cases"]:
                print(f"   - {test_case['id']}: {test_case['query_cn']}")
        
        return
    
    else:
        print("无效选择")
        return
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    asyncio.run(main()) 