#!/usr/bin/env python3
"""
快速测试脚本 - 用于快速运行单个类别的测试
Usage: python quick_test.py [category] [language]
"""

import sys
import asyncio
import argparse
from automated_test_runner import QueryTestRunner

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CiteWeave 快速测试工具")
    
    parser.add_argument(
        "category", 
        nargs="?",
        help="测试类别 (如: basic_information, citation_relationships, etc.)"
    )
    
    parser.add_argument(
        "-l", "--language",
        choices=["cn", "en"],
        default="cn",
        help="查询语言 (默认: cn)"
    )
    
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="显示所有可用的测试类别"
    )
    
    parser.add_argument(
        "--config-dir",
        default="../config",
        help="配置文件目录 (默认: ../config)"
    )
    
    return parser.parse_args()

async def run_quick_test(category, language="cn", config_dir="../config"):
    """运行快速测试"""
    print(f"🚀 快速测试: {category} ({language})")
    print("-" * 50)
    
    # 初始化测试运行器
    runner = QueryTestRunner(config_dir)
    
    # 初始化系统
    if not await runner.initialize_system():
        print("❌ 系统初始化失败")
        return False
    
    # 加载测试用例
    if not runner.load_test_cases():
        print("❌ 测试用例加载失败")
        return False
    
    # 检查类别是否存在
    if category not in runner.test_cases:
        print(f"❌ 未找到测试类别: {category}")
        print("\n可用类别:")
        for cat in runner.test_cases.keys():
            print(f"  - {cat}")
        return False
    
    # 运行测试
    results = await runner.run_category_tests(category, language)
    
    # 生成简要报告
    print("\n" + "="*60)
    print("📊 测试结果摘要")
    print("="*60)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.success)
    avg_score = sum(r.overall_score for r in results) / total_tests if total_tests > 0 else 0
    avg_time = sum(r.execution_time for r in results) / total_tests if total_tests > 0 else 0
    
    print(f"测试类别: {category}")
    print(f"测试数量: {total_tests}")
    print(f"成功率: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    print(f"平均得分: {avg_score:.2f}/10.0")
    print(f"平均时间: {avg_time:.2f}秒")
    
    print("\n🔍 详细结果:")
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"{status} {result.test_id}: {result.overall_score:.1f}/10.0 - {result.query_cn[:50]}...")
        
        if result.errors:
            print(f"   ⚠️  错误: {'; '.join(result.errors[:2])}...")
    
    # 保存详细报告
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_reports/quick_{category}_{language}_{timestamp}.md"
    csv_file = f"test_reports/quick_{category}_{language}_{timestamp}.csv"
    
    runner.generate_report(report_file)
    runner.export_results_csv(csv_file)
    
    print(f"\n📄 详细报告已保存到: {report_file}")
    print(f"📊 数据导出已保存到: {csv_file}")
    
    return True

def list_categories(config_dir="../config"):
    """列出所有测试类别"""
    runner = QueryTestRunner(config_dir)
    
    if not runner.load_test_cases():
        print("❌ 无法加载测试用例")
        return
    
    print("📋 可用的测试类别:")
    print("="*50)
    
    for category, info in runner.test_cases.items():
        test_count = len(info["test_cases"])
        print(f"\n🔍 {category}")
        print(f"   描述: {info['description']}")
        print(f"   策略: {info['retrieval_strategy']}")
        print(f"   用例数: {test_count}")
        
        # 显示前3个测试用例作为示例
        for i, test_case in enumerate(info["test_cases"][:3]):
            print(f"   {i+1}. {test_case['query_cn']}")
        
        if test_count > 3:
            print(f"   ... 还有 {test_count-3} 个用例")

async def main():
    """主函数"""
    args = parse_arguments()
    
    print("⚡ CiteWeave 快速测试工具")
    print("=" * 30)
    
    # 如果要求列出类别
    if args.list_categories:
        list_categories(args.config_dir)
        return
    
    # 如果没有提供类别，显示帮助
    if not args.category:
        print("请指定要测试的类别，或使用 --list-categories 查看可用类别")
        print("\n使用示例:")
        print("  python quick_test.py basic_information")
        print("  python quick_test.py citation_relationships -l en")
        print("  python quick_test.py --list-categories")
        return
    
    # 运行快速测试
    success = await run_quick_test(args.category, args.language, args.config_dir)
    
    if success:
        print("\n🎉 测试完成!")
    else:
        print("\n❌ 测试失败!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 