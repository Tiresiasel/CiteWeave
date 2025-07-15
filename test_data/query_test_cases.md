# 简化学术论文查询测试用例

基于现有三篇论文的实际内容设计测试用例：
- Porter: Competitive Strategy (竞争战略)
- Rivkin (2000): Imitation of Complex Strategies (复杂策略的模仿)
- Foss & Saebi (2017): Fifteen Years of Research on Business Model Innovation (商业模式创新十五年研究)

## 📋 简化测试用例分类

### 1. 基础信息查询 (Basic Information Queries)
**检索策略**: 元数据查询 + 作者信息

#### 中文测试用例:
- Porter是谁？他的主要贡献是什么？
- Rivkin这篇文章是什么时候发表的？
- 商业模式创新这篇综述分析了多少篇论文？
- 这些论文都发表在哪些期刊上？

#### 英文测试用例:
- Who is Michael Porter and what is he known for?
- When was Rivkin's paper on strategy imitation published?
- How many papers did Foss and Saebi review in their BMI literature review?
- Which journals published these papers?

### 2. 主题内容查询 (Content Queries)
**检索策略**: 向量搜索 + 语义匹配

#### 中文测试用例:
- 什么是竞争战略？
- 复杂策略为什么难以模仿？
- 商业模式创新的主要问题是什么？
- 这些研究用了什么理论框架？

#### 英文测试用例:
- What is competitive strategy according to Porter?
- Why are complex strategies difficult to imitate?
- What are the main problems in business model innovation research?
- What theoretical frameworks are used in these studies?

### 3. 引用关系查询 (Citation Queries)
**检索策略**: 图数据库查询

#### 中文测试用例:
- Rivkin的文章引用了哪些重要文献？
- 有哪些论文引用了Porter的竞争战略理论？
- Foss和Saebi综述了哪些关键作者的工作？
- 这些作者之间有什么引用关系？

#### 英文测试用例:
- What key papers does Rivkin cite in his strategy imitation paper?
- Which papers cite Porter's competitive strategy theory?
- What key authors did Foss and Saebi review in their BMI survey?
- What citation relationships exist between these authors?

### 4. 比较分析查询 (Comparative Queries)
**检索策略**: 多文档对比

#### 中文测试用例:
- Porter和Rivkin对战略的观点有什么不同？
- 2000年和2017年的战略研究有什么变化？
- 竞争战略和商业模式创新有什么关系？
- 这三位作者的研究方法有什么异同？

#### 英文测试用例:
- How do Porter's and Rivkin's views on strategy differ?
- What changes occurred in strategy research between 2000 and 2017?
- What is the relationship between competitive strategy and business model innovation?
- How do the research methods of these three authors compare?

### 5. 应用实践查询 (Application Queries)
**检索策略**: 实践案例匹配

#### 中文测试用例:
- 如何应用Porter的竞争战略框架？
- 企业如何防止策略被模仿？
- 商业模式创新的最佳实践是什么？
- 这些理论对管理实践有什么启示？

#### 英文测试用例:
- How can Porter's competitive strategy framework be applied?
- How can firms prevent their strategies from being imitated?
- What are the best practices for business model innovation?
- What practical implications do these theories offer for management?

## 🎯 测试目标

### 简化目标:
1. **基础功能测试**: 能否正确识别论文基本信息
2. **内容理解测试**: 能否理解论文核心概念和理论
3. **关系分析测试**: 能否分析论文间的引用和概念关系
4. **多语言测试**: 中英文查询的处理能力
5. **综合应用测试**: 能否提供实用的分析和建议

### 期望结果:
- 准确回答基于实际论文内容的问题
- 支持中英文查询
- 提供清晰的论文间关系分析
- 给出实用的应用建议

## 📊 评估指标

### 基础指标:
- 响应成功率 (>=90%)
- 平均响应时间 (<5秒)
- 内容准确性 (>=85%)
- 语言处理准确性 (>=90%)

### 高级指标:
- 答案完整性评分 (1-5分)
- 引用关系准确性 (1-5分)
- 实用性评分 (1-5分)
- 多语言一致性 (1-5分) 