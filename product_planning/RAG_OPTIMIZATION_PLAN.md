# CiteWeave RAG优化实施计划

## 📋 现状分析 vs ChatGPT建议对比

### ✅ 当前已实现（v2.0）
- [x] 句子级引用分析
- [x] 引用级论证关系嵌入
- [x] 10种论证关系类型
- [x] 高质量的实体-引用匹配算法
- [x] JSON/JSONL标准化输出

### 🎯 ChatGPT建议的核心改进点
1. **多粒度上下文**：句子、段落、章节级别
2. **网络结构增强**：二模网络、观点级引用图
3. **语义丰富化**：论证目的、证据类型、上下文依赖
4. **时间演化追踪**：理论发展轨迹
5. **领域知识分类**：研究域、方法论聚类

## 🚀 三阶段实施方案

### Phase 1: 上下文粒度增强 (2-3周)

#### 1.1 数据结构扩展
```python
# 扩展当前sentence_data结构
{
  "sentence_index": 42,
  "sentence_text": "Porter (1980) argues...",
  "context_metadata": {
    "section": "Literature Review",
    "subsection": "Competitive Strategy",
    "paragraph_index": 15,
    "paragraph_theme": "theoretical_foundation",
    "discourse_role": "CLAIM_MAIN",  # 基于现有claim_type
    "semantic_position": "premise|argument|conclusion"
  },
  "citations": [...],
  "argument_analysis": {...}
}
```

#### 1.2 实施任务
- [ ] 扩展PDFProcessor增加段落级解析
- [ ] 添加章节识别功能（Introduction, Methods, Results等）
- [ ] 实现话语角色分类器（基于现有ArgumentClassifier扩展）
- [ ] 更新DocumentProcessor集成新的上下文信息

### Phase 2: 语义关系深化 (3-4周)

#### 2.1 增强论证关系元数据
```python
# 扩展argument_analysis结构
{
  "relation_type": "SUPPORTS",
  "confidence": 0.856,
  "entity_text": "Porter (1980)",
  "semantic_metadata": {
    "argumentative_purpose": "theoretical_foundation",
    "cited_aspect": "competitive_advantage_theory",
    "citing_stance": "acceptance|criticism|neutral",
    "elaboration_type": "conceptual_extension",
    "evidence_type": "theoretical|empirical|methodological"
  }
}
```

#### 2.2 作者-概念网络构建
```python
{
  "author_network": {
    "michael_porter": {
      "key_concepts": ["five_forces", "generic_strategies"],
      "theoretical_contributions": ["competitive_positioning"],
      "citation_patterns": {
        "total_citations": 1500,
        "supporting_citations": 1200,
        "critical_citations": 200,
        "extending_citations": 100
      }
    }
  }
}
```

#### 2.3 实施任务
- [ ] 扩展relation_types.yaml增加语义元数据
- [ ] 实现概念提取器（基于LLM）
- [ ] 构建作者-概念映射数据库
- [ ] 添加引用目的分类器

### Phase 3: 知识图谱与查询优化 (4-5周)

#### 3.1 Neo4j图谱Schema升级
```cypher
// 增强的节点类型
(:Paper {id, title, authors, year, domain, key_concepts, theoretical_framework})
(:Claim {id, text, type, domain, evidence_type, source_paper})
(:Author {name, affiliation, research_domains})
(:Concept {name, domain, definition, related_theories})
(:Theory {name, domain, foundational_papers, evolution_status})

// 增强的关系类型
(:Paper)-[:CITES {
  relation_type, 
  strength, 
  aspect, 
  purpose, 
  stance,
  context_section
}]->(:Paper)

(:Claim)-[:SUPPORTS|REFUTES|EXTENDS {
  strength, 
  evidence_type,
  mechanism
}]->(:Claim)
```

#### 3.2 向量数据库优化
- **多层次嵌入**：句子级、声称级、概念级
- **语义聚类**：研究域、方法论、理论框架
- **时间向量**：捕捉理论演化

#### 3.3 查询接口设计
```python
class EnhancedRAGQuery:
    def query_citations_by_stance(self, author, work, stance="supports"):
        """查询特定立场的引用: 谁支持/反驳了某作者的观点"""
        
    def query_theory_evolution(self, theory_name, time_range):
        """理论演化查询: RBV理论如何从1990年发展到现在"""
        
    def query_methodology_usage(self, method, domain):
        """方法论应用查询: 谁在战略管理中使用了博弈论"""
        
    def query_concept_network(self, concept, relation_types):
        """概念网络查询: 竞争优势理论的支持者和批评者网络"""
```

## 🔧 技术实施细节

### 关键模块改造

#### 1. DocumentProcessor升级
```python
class EnhancedDocumentProcessor(DocumentProcessor):
    def __init__(self):
        super().__init__()
        self.section_classifier = SectionClassifier()
        self.discourse_analyzer = DiscourseAnalyzer()
        self.concept_extractor = ConceptExtractor()
        
    def process_document_enhanced(self, pdf_path):
        # 现有处理逻辑
        result = super().process_document(pdf_path)
        
        # 增强处理
        result = self._add_context_metadata(result)
        result = self._extract_concepts(result)
        result = self._analyze_discourse_roles(result)
        
        return result
```

#### 2. 新增ConceptExtractor
```python
class ConceptExtractor:
    def extract_key_concepts(self, text, domain=None):
        """从文本中提取关键概念"""
        
    def identify_theoretical_frameworks(self, citations):
        """识别引用的理论框架"""
        
    def map_author_contributions(self, author, papers):
        """映射作者的理论贡献"""
```

#### 3. 新增QueryEngine
```python
class AcademicQueryEngine:
    def __init__(self, neo4j_client, vector_db, argument_classifier):
        self.graph = neo4j_client
        self.vectors = vector_db
        self.classifier = argument_classifier
        
    def complex_academic_query(self, query_text):
        """处理复杂学术查询"""
        # 1. 查询意图分析
        intent = self._analyze_query_intent(query_text)
        
        # 2. 多维度检索
        results = self._multi_dimensional_search(intent)
        
        # 3. 结果融合与排序
        return self._fuse_and_rank_results(results)
```

## 📊 预期效果

### 查询能力提升
- **现在**: "找到引用Porter (1980)的句子"
- **升级后**: "找到所有在理论构建部分引用Porter竞争定位理论并表示支持的管理学论文，按引用强度排序"

### 数据结构优势
1. **多维度检索**: 时间、领域、关系、目的
2. **语义理解**: 不仅是关键词匹配，而是理解论证逻辑
3. **网络分析**: 发现隐藏的理论关联和演化路径
4. **个性化查询**: 适应不同类型的学术问题

## 🎯 里程碑计划

| 阶段 | 时间 | 关键交付物 | 验证标准 |
|------|------|-----------|----------|
| Phase 1 | Week 1-3 | 上下文增强的DocumentProcessor | 能识别段落主题和话语角色 |
| Phase 2 | Week 4-7 | 语义丰富的关系分析器 | 能识别引用目的和立场 |
| Phase 3 | Week 8-12 | 完整的学术查询引擎 | 支持复杂自然语言查询 |

这个计划将你现有的v2.0架构升级为真正的学术级RAG系统，能够支持ChatGPT建议的高自由度查询需求。关键是在保持现有架构优势的基础上，逐步增加语义深度和查询灵活性。 