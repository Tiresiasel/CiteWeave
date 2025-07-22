> **This project is licensed under the Apache License 2.0. See the LICENSE file for details.**

# CiteWeave Test Data Directory

## 📋 Directory Description

This directory is dedicated to storing **data files** required for testing, and does not contain test code. All test code is located in the `tests/` directory.

## 📁 Directory Structure

```
test_data/
├── README.md                     # This document
├── test_cases_by_category.json   # Query test case data
├── query_test_cases.md           # Query type classification description
└── papers/                       # Sample paper data
    └── ...
```

## 📄 File Descriptions

### test_cases_by_category.json
- **Purpose**: Definition of query evaluation test cases
- **Content**: 30 test cases covering 10 major query categories
- **Format**: JSON structured data
- **Language**: Bilingual (Chinese and English)

### query_test_cases.md  
- **Purpose**: Detailed description of query type classification system
- **Content**: Analysis and retrieval strategies for 10 major query scenarios
- **Format**: Markdown document

### papers/
- **Purpose**: Sample paper data files
- **Content**: Metadata and processing results for test papers
- **Format**: JSON files

## 🔗 Related Test Code

Test code has been moved to a dedicated test directory:
- **Query evaluation tests**: `tests/query_evaluation/`
- **Other tests**: `tests/`

## 📝 Data Usage

These test data are used by the following test modules:
- `tests/query_evaluation/automated_test_runner.py`
- `tests/query_evaluation/quick_test.py`
- Other related test scripts

## 🤝 Contribution Guidelines

1. Only add test **data** files in this directory
2. Test **code** should be placed in the `tests/` directory
3. Please update this README when adding new data files
4. Keep data files structured and well-documented

---

**Note**: This directory follows the principle of separating test data from test code to ensure a clear and maintainable project structure. 