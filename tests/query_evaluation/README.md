# CiteWeave 查询评估测试模块

## 📋 模块概述

这个模块专门用于测试和评估CiteWeave多智能体系统的查询回答质量。它提供了全面的测试框架，包括基础评估和LLM增强评估两种模式。

## 🏗️ 模块结构

```
tests/query_evaluation/
├── README.md                    # 本文档
├── automated_test_runner.py     # 自动化测试运行器
├── llm_evaluator.py            # LLM评估器
├── quick_test.py               # 快速测试工具
└── test_reports/               # 测试报告目录
    └── .gitkeep
```

## 🎯 功能特性

### 双重评估体系
- **基础评估**: 基于成功率、响应时间、置信度等指标的快速评估
- **LLM增强评估**: 使用GPT-4o-mini进行内容质量的深度评估

### 多维度测试
覆盖10大类学术查询场景：
- 基础信息查询
- 引用关系查询  
- 主题内容查询
- 学者研究查询
- 时间演化查询
- 对比分析查询
- 影响力评估查询
- 文献综述支持查询
- 跨学科查询
- 实践应用查询

### 多语言支持
- 中文查询测试
- 英文查询测试
- 自动语言检测和处理

## 🚀 快速开始

### 环境要求
```bash
# 确保已安装必要依赖
pip install openai pandas asyncio

# 设置OpenAI API密钥 (用于LLM评估)
export OPENAI_API_KEY="your-openai-api-key"

# 确保Neo4j数据库运行中
# 确保CiteWeave系统已初始化
```

### 运行测试

#### 方式一：交互式完整测试
```bash
cd tests/query_evaluation
python automated_test_runner.py
```

选择评估模式：
- `1` - 基础评估 (快速)
- `2` - LLM增强评估 (推荐，质量更高)

选择测试模式：
- `1` - 运行所有测试 (中文)
- `2` - 运行所有测试 (英文)  
- `3` - 运行特定类别测试
- `4` - 显示测试类别列表

#### 方式二：快速单类别测试
```bash
cd tests/query_evaluation
python quick_test.py basic_information cn
```

#### 方式三：编程调用
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from automated_test_runner import QueryTestRunner

async def run_custom_test():
    # 启用LLM评估的测试运行器
    runner = QueryTestRunner(enable_llm_evaluation=True)
    await runner.initialize_system()
    runner.load_test_cases()
    
    # 运行特定类别
    results = await runner.run_category_tests("basic_information", "cn")
    
    # 生成报告
    report = runner.generate_report("test_reports/my_test_report.md")
    runner.export_results_csv("test_reports/my_test_results.csv")
```

## 📊 评估机制详解

### 基础评估指标 (0-10分)
- **基础分 (3分)**: 成功返回响应
- **置信度分 (2分)**: 系统置信度 × 2  
- **质量分 (2分)**: 响应长度合理性
- **稳定性分 (2分)**: 错误警告惩罚
- **复杂度奖励 (1分)**: 高难度问题额外分

### LLM增强评估 (推荐)
使用GPT-4o-mini对回答内容进行5个维度的深度评估：

#### 评估维度 (各0-10分)
- **准确性 (Accuracy)**: 信息是否准确、事实是否正确
- **完整性 (Completeness)**: 是否涵盖查询的所有关键方面
- **相关性 (Relevance)**: 是否直接相关并满足用户需求
- **清晰度 (Clarity)**: 表达是否清晰、逻辑性强
- **总体质量 (Overall)**: 综合考虑的整体评价

#### 最终评分
**增强评分** = LLM评估总分 × 70% + 基础评分 × 30%

### 详细反馈
LLM评估提供：
- 各维度评分及详细评估理由
- 回答的具体优点和不足  
- 针对性的改进建议

## 📈 测试报告

### 报告内容
- **测试总览**: 成功率、平均得分、执行时间
- **分类统计**: 各类别的详细表现
- **详细结果**: 每个测试用例的完整评估
- **LLM评估详情**: 包含所有维度的评分和反馈

### 报告格式
- **Markdown报告**: 人类可读的详细报告
- **CSV数据**: 用于进一步分析的结构化数据

## ⚙️ 配置说明

### 模型配置
在 `config/model_config.json` 中配置评估器：
```json
{
  "llm": {
    "agents": {
      "evaluator": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 2000
      }
    }
  }
}
```

### 测试数据
测试用例存放在 `test_data/test_cases_by_category.json`，包含：
- 30个精心设计的测试用例
- 涵盖10大查询类别
- 中英双语版本
- 明确的评估标准

## 🔧 扩展开发

### 添加新测试类别
1. 在 `test_data/test_cases_by_category.json` 中添加新类别
2. 定义测试用例和评估标准
3. 可选：在 `automated_test_runner.py` 中添加特殊处理逻辑

### 自定义评估器
继承 `LLMEvaluator` 类并重写评估方法：
```python
class CustomEvaluator(LLMEvaluator):
    def _create_evaluation_prompt(self, query, response, query_type, criteria):
        # 自定义评估提示词
        return custom_prompt
```

### 新增评估维度
修改 `EvaluationResult` 数据类添加新字段，并更新评估逻辑。

## 🚨 故障排除

### 常见问题
1. **API密钥错误**: 确保 `OPENAI_API_KEY` 正确设置
2. **网络连接**: 检查能否访问OpenAI API
3. **数据库连接**: 确保Neo4j正常运行
4. **路径问题**: 确保从正确目录运行测试

### 性能优化
- **API限制**: 系统自动处理速率限制
- **并发控制**: 使用适当的延迟避免API过载
- **缓存机制**: 可考虑添加评估结果缓存

## 📚 相关文档

- [项目主README](../../README.md)
- [数据结构文档](../../docs/data_structures/)
- [测试数据说明](../../test_data/query_test_cases.md)
- [API接口文档](../../docs/data_structures/api_interfaces.md)

## 🤝 贡献指南

1. 遵循现有的代码风格
2. 添加新功能时更新相应文档
3. 确保测试用例能够正确运行
4. 提交前检查所有路径引用正确

---

**注意**: 这个模块是CiteWeave系统质量保证的核心组件，建议在系统开发和优化过程中定期运行测试，特别是使用LLM增强评估模式来获得更准确的质量反馈。 