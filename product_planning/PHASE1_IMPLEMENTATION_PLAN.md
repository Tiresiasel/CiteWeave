# Phase 1 实施计划：上下文粒度增强 (v0.7)

## 🎯 Phase 1 目标

在现有v2.0架构基础上，增加多粒度上下文信息，提升引用分析的语义深度，为学术级RAG系统奠定基础。

## 📋 具体实施任务

### Task 1: PDFProcessor段落级解析增强 (Week 1)

#### 1.1 扩展PDF文本提取
```python
class EnhancedPDFProcessor(PDFProcessor):
    def extract_text_with_paragraphs(self, pdf_path):
        """提取文本并保持段落结构"""
        # 基于现有的extract_text_with_best_engine扩展
        # 添加段落边界检测
        
    def identify_paragraph_boundaries(self, text):
        """识别段落边界"""
        # 基于缩进、空行、句子长度等特征
        
    def extract_paragraph_themes(self, paragraph_text):
        """提取段落主题 - 使用简单的关键词提取"""
        # Phase 1使用基于TF-IDF的简单方法
        # Phase 2可升级为LLM-based方法
```

#### 1.2 数据结构扩展
```python
# 扩展现有的sentence处理结果
{
    "sentence_index": 42,
    "sentence_text": "Porter (1980) argues...",
    "paragraph_context": {
        "paragraph_index": 15,
        "paragraph_text": "Previous research has...",
        "paragraph_theme_keywords": ["competitive", "strategy", "theory"],
        "sentences_in_paragraph": [40, 41, 42, 43, 44]
    }
}
```

### Task 2: 章节识别功能 (Week 1-2)

#### 2.1 章节分类器
```python
class SectionClassifier:
    def __init__(self):
        # 使用规则基础的方法 + 简单的ML分类
        self.section_patterns = {
            "Introduction": ["introduction", "background"],
            "Literature Review": ["literature", "review", "related work"],
            "Methods": ["method", "approach", "procedure"],
            "Results": ["result", "finding", "analysis"],
            "Discussion": ["discussion", "implication"],
            "Conclusion": ["conclusion", "summary"]
        }
    
    def classify_section(self, text_block, position_in_doc):
        """基于文本内容和位置识别章节"""
        # Phase 1: 规则基础方法
        # 检查标题关键词、文档位置等
        
    def identify_subsections(self, section_text):
        """识别子章节"""
        # 基于标题层级（字体大小、格式等）
```

#### 2.2 集成到DocumentProcessor
```python
# 更新DocumentProcessor._analyze_sentences_citations
def _analyze_sentences_citations(self, sentences):
    # 现有逻辑...
    
    # 新增：章节分析
    section_info = self.section_classifier.analyze_document_structure(sentences)
    
    for idx, sentence in enumerate(sentences):
        # 现有处理...
        
        # 添加章节信息
        sentence_data["section_metadata"] = {
            "section": section_info[idx]["section"],
            "subsection": section_info[idx].get("subsection"),
            "section_purpose": self._infer_section_purpose(section_info[idx]["section"])
        }
```

### Task 3: 话语角色分类器 (Week 2)

#### 3.1 基于现有ArgumentClassifier扩展
```python
class DiscourseRoleAnalyzer:
    def __init__(self, argument_classifier):
        self.argument_classifier = argument_classifier
        self.discourse_patterns = {
            "CLAIM_MAIN": ["argue", "propose", "suggest", "claim"],
            "EVIDENCE": ["show", "demonstrate", "prove", "indicate"],
            "CONTRAST": ["however", "although", "while", "but"],
            "ELABORATION": ["furthermore", "moreover", "additionally"]
        }
    
    def analyze_discourse_role(self, sentence, context):
        """分析句子的话语角色"""
        # 结合ArgumentClassifier的结果
        # 添加基于语言模式的分析
        
    def identify_semantic_position(self, sentence, paragraph_context):
        """识别语义位置：premise, argument, conclusion"""
        # 基于句子在段落中的位置和语言特征
```

#### 3.2 简化的实现策略
```python
# Phase 1: 使用规则基础的方法
class SimpleDiscourseAnalyzer:
    def analyze_sentence(self, sentence, paragraph_sentences, sentence_idx):
        """简化的话语分析"""
        discourse_role = "NEUTRAL"  # 默认值
        semantic_position = "argument"  # 默认值
        
        # 基于关键词和位置的简单规则
        if self._contains_claim_indicators(sentence):
            discourse_role = "CLAIM_MAIN"
        elif self._contains_evidence_indicators(sentence):
            discourse_role = "EVIDENCE"
        elif self._contains_contrast_indicators(sentence):
            discourse_role = "CONTRAST"
            
        # 基于段落位置推断语义位置
        if sentence_idx == 0:  # 段落首句
            semantic_position = "premise"
        elif sentence_idx == len(paragraph_sentences) - 1:  # 段落末句
            semantic_position = "conclusion"
            
        return discourse_role, semantic_position
```

### Task 4: DocumentProcessor集成 (Week 2-3)

#### 4.1 更新核心数据结构
```python
# 完整的sentence_data结构（Phase 1版本）
{
    "sentence_index": 42,
    "sentence_text": "Porter (1980) argues that competitive advantage stems from strategic positioning.",
    "has_citations": true,
    "citations": [...],  # 现有结构保持不变
    "argument_analysis": {...},  # 现有结构保持不变
    
    # 新增的上下文元数据
    "context_metadata": {
        "paragraph_index": 15,
        "paragraph_theme_keywords": ["competitive", "strategy", "positioning"],
        "section": "Literature Review",
        "subsection": "Competitive Strategy Theory",
        "section_purpose": "theory_building",
        "discourse_role": "CLAIM_MAIN",
        "semantic_position": "premise"
    },
    
    "word_count": 12,
    "char_count": 89
}
```

#### 4.2 处理流程更新
```python
class ContextEnhancedDocumentProcessor(DocumentProcessor):
    def __init__(self):
        super().__init__()
        self.section_classifier = SectionClassifier()
        self.discourse_analyzer = SimpleDiscourseAnalyzer()
        
    def process_document(self, pdf_path, save_results=True):
        """增强的文档处理流程"""
        # 1. 现有的PDF处理和引用分析
        base_result = super().process_document(pdf_path, save_results=False)
        
        # 2. 上下文增强处理
        enhanced_result = self._add_context_metadata(base_result)
        
        # 3. 保存结果
        if save_results:
            self._save_enhanced_results(enhanced_result, pdf_path)
            
        return enhanced_result
        
    def _add_context_metadata(self, document_result):
        """添加上下文元数据"""
        sentences = document_result["sentences_with_citations"]
        
        # 段落分析
        paragraph_info = self._analyze_paragraphs(sentences)
        
        # 章节分析
        section_info = self.section_classifier.analyze_document_structure(sentences)
        
        # 话语角色分析
        for idx, sentence_data in enumerate(sentences):
            # 添加段落信息
            sentence_data["context_metadata"] = {
                "paragraph_index": paragraph_info[idx]["paragraph_index"],
                "paragraph_theme_keywords": paragraph_info[idx]["theme_keywords"],
                "section": section_info[idx]["section"],
                "subsection": section_info[idx].get("subsection"),
                "section_purpose": self._infer_section_purpose(section_info[idx]["section"])
            }
            
            # 话语角色分析
            if sentence_data["has_citations"]:
                discourse_role, semantic_position = self.discourse_analyzer.analyze_sentence(
                    sentence_data["sentence_text"],
                    paragraph_info[idx]["paragraph_sentences"],
                    paragraph_info[idx]["sentence_idx_in_paragraph"]
                )
                sentence_data["context_metadata"]["discourse_role"] = discourse_role
                sentence_data["context_metadata"]["semantic_position"] = semantic_position
        
        return document_result
```

## 📊 Phase 1 验收标准

### 功能验收
- [ ] 能够识别段落边界并提取段落主题关键词
- [ ] 能够识别至少6种基本章节类型（Introduction, Literature Review, Methods, Results, Discussion, Conclusion）
- [ ] 能够为有引用的句子分配话语角色（至少4种：CLAIM_MAIN, EVIDENCE, CONTRAST, ELABORATION）
- [ ] 能够识别句子的语义位置（premise, argument, conclusion）

### 数据质量验收
- [ ] 段落识别准确率 > 80%
- [ ] 章节识别准确率 > 70%
- [ ] 话语角色分类合理性 > 70%（人工评估）

### 性能验收
- [ ] 处理时间增加 < 30%（相比v0.6.1）
- [ ] 输出文件大小增加 < 50%

### 兼容性验收
- [ ] 完全向后兼容现有的v2.0数据结构
- [ ] 现有的测试用例全部通过
- [ ] CLI接口保持不变

## 🛠️ 实施时间表

| 周次 | 主要任务 | 交付物 |
|------|----------|--------|
| Week 1 | PDFProcessor段落解析 + 章节识别基础 | 段落边界检测功能 |
| Week 2 | 章节分类器 + 话语角色分析器 | 基础的章节和角色识别 |
| Week 3 | DocumentProcessor集成 + 测试 | 完整的v0.7版本 |

## 🔍 测试策略

### 单元测试
- 段落边界检测测试
- 章节分类器测试
- 话语角色分析器测试

### 集成测试
- 完整的文档处理流程测试
- 数据结构兼容性测试
- 性能回归测试

### 用户验收测试
- 使用真实的学术论文测试
- 上下文信息质量人工评估
- 查询能力改进验证

Phase 1完成后，我们将有一个具备上下文感知能力的基础版本，为后续的Phase 2（语义关系深化）和Phase 3（知识图谱构建）奠定坚实基础。 