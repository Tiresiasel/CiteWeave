#!/usr/bin/env python3
"""
CiteWeave Multi-Agent System Query Test Runner
Automated test runner - runs various types of academic paper query test cases
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

from src.agents.multi_agent_system import EnhancedMultiAgentSystem
from src.graph_builder import GraphDB
from tests.query_evaluation.llm_evaluator import LLMEvaluator, EvaluationResult
from src.vector_indexer import VectorIndexer
from src.author_paper_index import AuthorPaperIndex
from src.config_manager import ConfigManager

project_root = os.getcwd()
config_dir = os.path.join(project_root, "config")
config_manager = ConfigManager(config_dir)

@dataclass
class TestResult:
    """Test result data class"""
    test_id: str
    category: str
    query_cn: str
    query_en: str
    expected_result_type: str
    complexity: str
    evaluation_criteria: List[str]
    
    # Execution results
    success: bool = False
    response: str = ""
    confidence: float = 0.0
    query_type: str = ""
    action: str = ""
    response_language: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Evaluation results
    evaluation_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    execution_time: float = 0.0
    notes: str = ""
    
    # LLM evaluation results
    llm_evaluation: Optional[EvaluationResult] = None
    enhanced_score: float = 0.0  # Enhanced score including LLM evaluation

class QueryTestRunner:
    """Query test runner"""
    
    def __init__(self, config_dir: str = None, enable_llm_evaluation: bool = True):
        """Initialize the test runner"""
        # If config_dir is not provided, use the global config_dir
        self.config_dir = config_dir or os.path.join(os.getcwd(), "config")
        self.agent_system = None
        self.test_cases = {}
        self.results = []
        self.enable_llm_evaluation = enable_llm_evaluation
        self.llm_evaluator = None
        
        # Initialize LLM evaluator
        if self.enable_llm_evaluation:
            try:
                model_config_path = os.path.join(self.config_dir, "model_config.json")
                self.llm_evaluator = LLMEvaluator(model_config_path)
                print("✅ LLM evaluator initialized successfully")
            except Exception as e:
                print(f"⚠️ LLM evaluator initialization failed, will use basic evaluation: {e}")
                self.enable_llm_evaluation = False
        
    async def initialize_system(self):
        """Initialize the multi-agent system"""
        try:
            print("🔧 Initializing test environment...")
            
            # Load configuration
            # config_manager = ConfigManager(self.config_dir) # This line is now redundant as config_manager is global
            neo4j_config = config_manager.neo4j_config
            
            # Initialize database connection
            graph_db = GraphDB(
                uri=neo4j_config["uri"],
                user=neo4j_config["username"],
                password=neo4j_config["password"]
            )
            
            # Get project root directory
            # project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # This line is now redundant
            
            vector_indexer = VectorIndexer(
                paper_root=os.path.join(project_root, "data", "papers"),
                index_path=os.path.join(project_root, "data", "vector_index")
            )
            
            # Initialize author index
            author_index = AuthorPaperIndex(
                storage_root=os.path.join(project_root, "data", "papers"),
                index_db_path=os.path.join(project_root, "data", "author_paper_index.db")
            )
            
            # Initialize agent system
            self.agent_system = EnhancedMultiAgentSystem(
                graph_db=graph_db,
                vector_indexer=vector_indexer,
                author_index=author_index,
                config_path=os.path.join(self.config_dir, "model_config.json")
            )
            
            print("✅ System initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
    
    def load_test_cases(self, test_file: str = "simplified_test_cases.json"):
        """Load test cases"""
        try:
            test_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", test_file)
            with open(test_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.test_cases = data["test_cases_by_category"]
            
            print(f"📁 Loaded {len(self.test_cases)} test categories")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load test cases: {e}")
            return False
    
    async def run_single_test(self, category: str, test_case: Dict[str, Any], language: str = "cn") -> TestResult:
        """Run a single test case"""
        start_time = datetime.now()
        
        # Create test result object
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
            # Select query language
            query = test_case["query_cn"] if language == "cn" else test_case["query_en"]
            
            # Run query
            response = await self.agent_system.query(
                user_query=query,
                thread_id=f"test_{test_case['id']}",
                user_id="test_runner"
            )
            
            # Record results
            result.success = True
            result.response = response.get("response", "")
            result.confidence = response.get("confidence", 0.0)
            result.query_type = response.get("query_type", "")
            result.action = response.get("action", "")
            result.response_language = response.get("response_language", "")
            result.errors = response.get("errors", [])
            result.warnings = response.get("warnings", [])
            
            # Save agent trace log for this test case
            trace_path = os.path.join("tests/query_evaluation/test_reports", f"agent_trace_{test_case['id']}.jsonl")
            self.agent_system.export_trace_log(trace_path)
            
            # Calculate execution time
            end_time = datetime.now()
            result.execution_time = (end_time - start_time).total_seconds()
            
            # Basic evaluation
            result.overall_score = self._evaluate_response(result)
            
            # LLM evaluation (if enabled)
            if result.success and result.response:
                await self._perform_llm_evaluation(result)
                # Use enhanced score as final score
                final_score = result.enhanced_score if result.enhanced_score > 0 else result.overall_score
                print(f"✅ {test_case['id']}: Basic score={result.overall_score:.1f}/10, Final score={final_score:.1f}/10")
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
        """Simple response evaluation function"""
        score = 0.0
        
        # Basic score - whether a response is returned successfully
        if result.success and result.response:
            score += 3.0
        
        # Confidence score
        score += result.confidence * 2.0
        
        # Response length reasonability (100-2000 characters is a reasonable range)
        if 100 <= len(result.response) <= 2000:
            score += 2.0
        elif 50 <= len(result.response) < 100:
            score += 1.0
        
        # Error penalty
        if result.errors:
            score -= len(result.errors) * 0.5
        
        # Warning penalty
        if result.warnings:
            score -= len(result.warnings) * 0.2
        
        # Complexity adjustment
        if result.complexity == "high" and score >= 7.0:
            score += 1.0  # Reward for high complexity questions
        
        return min(max(score, 0.0), 10.0)  # Limit between 0-10
    
    async def _enhanced_evaluate_response(self, result: TestResult) -> float:
        """Enhanced response evaluation function - combines basic evaluation and LLM evaluation"""
        # Basic score (0-10 points)
        basic_score = self._evaluate_response(result)
        
        # If LLM evaluation is not enabled or failed, return basic score
        if not self.enable_llm_evaluation or not result.llm_evaluation:
            return basic_score
        
        # LLM evaluation weight configuration
        llm_weight = 0.7  # LLM evaluation weight 70%
        basic_weight = 0.3  # Basic evaluation weight 30%
        
        # Calculate LLM comprehensive score (overall_score)
        llm_score = result.llm_evaluation.overall_score
        
        # Calculate weighted final score
        enhanced_score = (llm_score * llm_weight) + (basic_score * basic_weight)
        
        return min(max(enhanced_score, 0.0), 10.0)  # Limit between 0-10
    
    async def _perform_llm_evaluation(self, result: TestResult):
        """Perform LLM evaluation for a single test result"""
        if not self.enable_llm_evaluation or not self.llm_evaluator:
            return
        
        try:
            # Select query language
            query = result.query_cn if result.response_language == "zh" else result.query_en
            
            # Execute LLM evaluation
            llm_eval = await self.llm_evaluator.evaluate_response(
                query=query,
                response=result.response,
                query_type=result.category,
                expected_criteria=result.evaluation_criteria
            )
            
            # Save evaluation results
            result.llm_evaluation = llm_eval
            
            # Calculate enhanced score
            result.enhanced_score = await self._enhanced_evaluate_response(result)
            
            print(f"🤖 LLM evaluation completed - Overall score: {llm_eval.overall_score:.1f}/10")
            
        except Exception as e:
            print(f"⚠️ LLM evaluation failed: {e}")
            result.enhanced_score = result.overall_score
    
    async def run_category_tests(self, category: str, language: str = "cn") -> List[TestResult]:
        """Run all tests for a specific category"""
        if category not in self.test_cases:
            print(f"❌ Test category not found: {category}")
            return []
        
        category_info = self.test_cases[category]
        test_cases = category_info["test_cases"]
        
        print(f"\n�� Running '{category}' category tests ({len(test_cases)} cases)")
        print(f"📋 Retrieval strategy: {category_info['retrieval_strategy']}")
        print("-" * 60)
        
        results = []
        for test_case in test_cases:
            result = await self.run_single_test(category, test_case, language)
            results.append(result)
            self.results.append(result)
            
            # Short delay to avoid API limits
            await asyncio.sleep(1)
        
        return results
    
    async def run_all_tests(self, language: str = "cn") -> List[TestResult]:
        """Run all test cases"""
        print(f"\n🚀 Starting to run all test cases (language: {language})")
        print("=" * 70)
        
        all_results = []
        for category in self.test_cases.keys():
            category_results = await self.run_category_tests(category, language)
            all_results.extend(category_results)
        
        return all_results
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate test report"""
        if not self.results:
            return "No test results to generate report"
        
        # Calculate statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        avg_score = sum(r.overall_score for r in self.results) / total_tests
        avg_time = sum(r.execution_time for r in self.results) / total_tests
        
        # LLM evaluation statistics
        llm_evaluated_tests = sum(1 for r in self.results if r.llm_evaluation is not None)
        avg_enhanced_score = 0.0
        if llm_evaluated_tests > 0:
            enhanced_scores = [r.enhanced_score for r in self.results if r.enhanced_score > 0]
            avg_enhanced_score = sum(enhanced_scores) / len(enhanced_scores) if enhanced_scores else 0.0
        
        # Category statistics
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
            
            # LLM evaluation statistics
            if result.llm_evaluation is not None:
                stats['llm_evaluated'] += 1
            if result.enhanced_score > 0:
                stats['enhanced_scores'].append(result.enhanced_score)
        
        # Generate report
        evaluation_mode = "Enhanced evaluation (Basic evaluation + LLM evaluation)" if self.enable_llm_evaluation else "Basic evaluation"
        
        report = f"""
# CiteWeave Multi-Agent System Test Report
Generated time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Evaluation mode: {evaluation_mode}

## 📊 Test Overview
- **Total Tests**: {total_tests}
- **Successful Tests**: {successful_tests} ({successful_tests/total_tests*100:.1f}%)
- **Average Basic Score**: {avg_score:.2f}/10.0
- **Average Execution Time**: {avg_time:.2f} seconds"""

        # Add LLM evaluation information
        if self.enable_llm_evaluation:
            report += f"""
- **LLM Evaluation Coverage**: {llm_evaluated_tests}/{total_tests} ({llm_evaluated_tests/total_tests*100:.1f}%)
- **Average Enhanced Score**: {avg_enhanced_score:.2f}/10.0"""

        report += "\n\n## 📋 Category Statistics\n"
        
        for category, stats in category_stats.items():
            avg_cat_score = sum(stats['scores']) / len(stats['scores'])
            avg_cat_time = sum(stats['times']) / len(stats['times'])
            success_rate = stats['success'] / stats['total'] * 100
            
            report += f"""
### {category}
- Total Tests: {stats['total']}
- Success Rate: {success_rate:.1f}%
- Average Basic Score: {avg_cat_score:.2f}/10.0  
- Average Time: {avg_cat_time:.2f} seconds"""

            # Add LLM evaluation information
            if self.enable_llm_evaluation and stats['enhanced_scores']:
                avg_enhanced_cat_score = sum(stats['enhanced_scores']) / len(stats['enhanced_scores'])
                llm_coverage = stats['llm_evaluated'] / stats['total'] * 100
                report += f"""
- LLM Evaluation Coverage: {stats['llm_evaluated']}/{stats['total']} ({llm_coverage:.1f}%)
- Average Enhanced Score: {avg_enhanced_cat_score:.2f}/10.0"""
            
            report += "\n"
        
        report += "\n## 🔍 Detailed Results\n"
        
        for result in self.results:
            status = "✅" if result.success else "❌"
            report += f"""
### {result.test_id} - {result.category} {status}
- **Query**: {result.query_cn}
- **Basic Score**: {result.overall_score:.2f}/10.0"""

            # Add enhanced score information
            if result.enhanced_score > 0:
                report += f"""
- **Enhanced Score**: {result.enhanced_score:.2f}/10.0"""
            
            report += f"""
- **Execution Time**: {result.execution_time:.2f} seconds
- **Confidence**: {result.confidence:.2f}
"""
            
            # Add LLM evaluation details
            if result.llm_evaluation:
                llm_eval = result.llm_evaluation
                report += f"""
#### 🤖 LLM Evaluation Details
- **Accuracy**: {llm_eval.accuracy_score:.1f}/10
- **Completeness**: {llm_eval.completeness_score:.1f}/10
- **Relevance**: {llm_eval.relevance_score:.1f}/10
- **Clarity**: {llm_eval.clarity_score:.1f}/10
- **LLM Overall Score**: {llm_eval.overall_score:.1f}/10
- **Evaluation Reasoning**: {llm_eval.reasoning}
- **Strengths**: {llm_eval.strengths}
- **Weaknesses**: {llm_eval.weaknesses}
- **Suggestions**: {llm_eval.suggestions}
"""
            
            if result.errors:
                report += f"- **Errors**: {'; '.join(result.errors)}\n"
            if result.warnings:
                report += f"- **Warnings**: {'; '.join(result.warnings)}\n"
        
        # Save report
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Report saved to: {output_file}")
        
        return report
    
    def export_results_csv(self, output_file: str):
        """Export results to CSV format"""
        if not self.results:
            print("No results to export")
            return
        
        # Prepare data
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
            
            # Add LLM evaluation fields
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
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"📊 Results exported to: {output_file}")

async def main():
    """Main function"""
    print("🎯 CiteWeave Multi-Agent Query Test System")
    print("=" * 50)
    
    # Ask if LLM evaluation is enabled
    print("\n🤖 Evaluation Mode Selection:")
    print("1. Basic Evaluation (Fast, based on length, confidence, etc.)")
    print("2. Enhanced Evaluation (using GPT-4o-mini for content quality assessment)")
    
    eval_choice = input("Please select evaluation mode (1-2): ").strip()
    enable_llm_evaluation = eval_choice == "2"
    
    evaluation_mode = "Enhanced Evaluation Mode" if enable_llm_evaluation else "Basic Evaluation Mode"
    print(f"✅ Selected: {evaluation_mode}")
    
    # Initialize test runner
    runner = QueryTestRunner(config_dir=config_dir, enable_llm_evaluation=enable_llm_evaluation)
    
    # Initialize system
    if not await runner.initialize_system():
        print("System initialization failed, exiting test")
        return
    
    # Load test cases
    if not runner.load_test_cases():
        print("Test cases loading failed, exiting test")
        return
    
    # Interactive selection
    print("\nPlease select test mode:")
    print("1. Run all tests (Chinese)")
    print("2. Run all tests (English)")
    print("3. Run specific category tests")
    print("4. Display test category list")
    
    choice = input("Please enter selection (1-4): ").strip()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if choice == "1":
        # Run all tests (Chinese)
        await runner.run_all_tests("cn")
        report = runner.generate_report(f"tests/query_evaluation/test_reports/full_test_report_cn_{timestamp}.md")
        runner.export_results_csv(f"tests/query_evaluation/test_reports/full_test_results_cn_{timestamp}.csv")
        
    elif choice == "2":
        # Run all tests (English)
        await runner.run_all_tests("en")
        report = runner.generate_report(f"tests/query_evaluation/test_reports/full_test_report_en_{timestamp}.md")
        runner.export_results_csv(f"tests/query_evaluation/test_reports/full_test_results_en_{timestamp}.csv")
        
    elif choice == "3":
        # Run specific category
        print("\nAvailable test categories:")
        for i, category in enumerate(runner.test_cases.keys(), 1):
            print(f"{i}. {category}")
        
        cat_choice = input("Please select category number: ").strip()
        try:
            category_list = list(runner.test_cases.keys())
            selected_category = category_list[int(cat_choice) - 1]
            
            language = input("Select language (cn/en): ").strip() or "cn"
            
            await runner.run_category_tests(selected_category, language)
            report = runner.generate_report(f"tests/query_evaluation/test_reports/{selected_category}_test_report_{timestamp}.md")
            runner.export_results_csv(f"tests/query_evaluation/test_reports/{selected_category}_test_results_{timestamp}.csv")
            
        except (ValueError, IndexError):
            print("Invalid selection")
            return
            
    elif choice == "4":
        # Display category list
        print("\n📋 Test Category Details:")
        for category, info in runner.test_cases.items():
            test_count = len(info["test_cases"])
            print(f"\n🔍 {category}")
            print(f"    Description: {info['description']}")
            print(f"    Retrieval Strategy: {info['retrieval_strategy']}")
            print(f"    Test Cases: {test_count}")
            
            for test_case in info["test_cases"]:
                print(f"   - {test_case['id']}: {test_case['query_cn']}")
        
        return
    
    else:
        print("Invalid selection")
        return
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    asyncio.run(main()) 