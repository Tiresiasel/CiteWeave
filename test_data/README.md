> **This project is licensed under the Apache License 2.0. See the LICENSE file for details.**

# CiteWeave 测试数据目录

## 📋 目录说明

这个目录专门存放测试所需的**数据文件**，不包含测试代码。所有测试代码位于 `tests/` 目录下。

## 📁 目录结构

```
test_data/
├── README.md                     # 本文档
├── test_cases_by_category.json   # 查询测试用例数据
├── query_test_cases.md          # 查询类型分类说明
└── papers/                      # 样本论文数据
    └── ...
```

## 📄 文件说明

### test_cases_by_category.json
- **用途**: 查询评估测试用例的定义
- **内容**: 30个测试用例，涵盖10大查询类别
- **格式**: JSON结构化数据
- **语言**: 中英双语版本

### query_test_cases.md  
- **用途**: 查询类型分类体系的详细说明
- **内容**: 10大类查询场景的分析和检索策略
- **格式**: Markdown文档

### papers/
- **用途**: 样本论文数据文件
- **内容**: 用于测试的论文元数据、处理结果等
- **格式**: JSON文件

## 🔗 相关测试代码

测试代码已移动到专门的测试目录：
- **查询评估测试**: `tests/query_evaluation/`
- **其他测试**: `tests/`

## 📝 数据使用

这些测试数据被以下测试模块使用：
- `tests/query_evaluation/automated_test_runner.py`
- `tests/query_evaluation/quick_test.py`
- 其他相关测试脚本

## 🤝 贡献指南

1. 只在此目录添加测试**数据**文件
2. 测试**代码**应放在 `tests/` 目录下
3. 新增数据文件时请更新本README
4. 保持数据文件的结构化和文档化

---

**注意**: 这个目录遵循测试数据与测试代码分离的原则，确保项目结构清晰易维护。 