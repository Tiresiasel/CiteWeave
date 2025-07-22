# CiteWeave RAG Optimization Implementation Plan

## 📋 Current Status vs ChatGPT Recommendations

### ✅ Currently Implemented (v2.0)
- [x] Sentence-level citation analysis
- [x] Argument-level citation relationship embedding
- [x] 10 types of argument relationships
- [x] High-quality entity-citation matching algorithm
- [x] JSON/JSONL standardized output

### 🎯 Core Improvements Recommended by ChatGPT
1. **Multi-granularity context**: sentence, paragraph, section level
2. **Network structure enhancement**: bimodal network, viewpoint-level citation graph
3. **Semantic enrichment**: argumentative purpose, evidence type, context dependency
4. **Temporal evolution tracking**: theory development trajectory
5. **Domain knowledge classification**: research domain, methodology clustering

## 🚀 Three-Phase Implementation Plan

### Phase 1: Context Granularity Enhancement (2-3 weeks)

#### 1.1 Data Structure Expansion
```python
# Expand current sentence_data structure
{
  "sentence_index": 42,
  "sentence_text": "Porter (1980) argues...",
  "context_metadata": {
    "section": "Literature Review",
    "subsection": "Competitive Strategy",
    "paragraph_index": 15,
    "paragraph_theme": "theoretical_foundation",
    "discourse_role": "CLAIM_MAIN",  # Based on existing claim_type
    "semantic_position": "premise|argument|conclusion"
  },
  "citations": [...],
  "argument_analysis": {...}
}
```

#### 1.2 Implementation Tasks
- [ ] Expand PDFProcessor to add paragraph-level parsing
- [ ] Add section recognition (Introduction, Methods, Results, etc.)
- [ ] Implement discourse role classifier (extend existing ArgumentClassifier)
- [ ] Update DocumentProcessor to integrate new context information

### Phase 2: Semantic Relationship Deepening (3-4 weeks)

#### 2.1 Enhanced Argument Relationship Metadata
```python
# Expand argument_analysis structure
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

#### 2.2 Author-Concept Network Construction
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

#### 2.3 Implementation Tasks
- [ ] Expand relation_types.yaml to add semantic metadata
- [ ] Implement concept extractor (based on LLM)
- [ ] Build author-concept mapping database
- [ ] Add citation purpose classifier

### Phase 3: Knowledge Graph and Query Optimization (4-5 weeks)

#### 3.1 Neo4j Graph Schema Upgrade
```cypher
// Enhanced node types
(:Paper {id, title, authors, year, domain, key_concepts, theoretical_framework})
(:Claim {id, text, type, domain, evidence_type, source_paper})
(:Author {name, affiliation, research_domains})
(:Concept {name, domain, definition, related_theories})
(:Theory {name, domain, foundational_papers, evolution_status})

// Enhanced relationship types
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

#### 3.2 Vector Database Optimization
- **Multi-level embedding**: sentence-level, claim-level, concept-level
- **Semantic clustering**: research domain, methodology, theoretical framework
- **Time vectors**: capture theoretical evolution

#### 3.3 Query Interface Design
```python
class EnhancedRAGQuery:
    def query_citations_by_stance(self, author, work, stance="supports"):
        """Query specific citations by stance: who supports/criticizes an author's viewpoint"""
        
    def query_theory_evolution(self, theory_name, time_range):
        """Query theory evolution: how RBV theory evolved from 1990 to now"""
        
    def query_methodology_usage(self, method, domain):
        """Query methodology usage: who used game theory in strategic management"""
        
    def query_concept_network(self, concept, relation_types):
        """Query concept network: network of supporters and critics of competitive advantage theory"""
```

## 🔧 Technical Implementation Details

### Key Module Modifications

#### 1. DocumentProcessor Upgrade
```python
class EnhancedDocumentProcessor(DocumentProcessor):
    def __init__(self):
        super().__init__()
        self.section_classifier = SectionClassifier()
        self.discourse_analyzer = DiscourseAnalyzer()
        self.concept_extractor = ConceptExtractor()
        
    def process_document_enhanced(self, pdf_path):
        # Existing processing logic
        result = super().process_document(pdf_path)
        
        # Enhanced processing
        result = self._add_context_metadata(result)
        result = self._extract_concepts(result)
        result = self._analyze_discourse_roles(result)
        
        return result
```

#### 2. New ConceptExtractor
```python
class ConceptExtractor:
    def extract_key_concepts(self, text, domain=None):
        """Extract key concepts from text"""
        
    def identify_theoretical_frameworks(self, citations):
        """Identify cited theoretical frameworks"""
        
    def map_author_contributions(self, author, papers):
        """Map author's theoretical contributions"""
```

#### 3. New QueryEngine
```python
class AcademicQueryEngine:
    def __init__(self, neo4j_client, vector_db, argument_classifier):
        self.graph = neo4j_client
        self.vectors = vector_db
        self.classifier = argument_classifier
        
    def complex_academic_query(self, query_text):
        """Process complex academic queries"""
        # 1. Query intent analysis
        intent = self._analyze_query_intent(query_text)
        
        # 2. Multi-dimensional retrieval
        results = self._multi_dimensional_search(intent)
        
        # 3. Result fusion and ranking
        return self._fuse_and_rank_results(results)
```

## 📊 Expected Effects

### Query Capability Enhancement
- **Now**: "Find sentences citing Porter (1980)"
- **After upgrade**: "Find all management papers that cite Porter's competitive positioning theory and express support, ordered by citation strength"

### Data Structure Advantages
1. **Multi-dimensional retrieval**: time, domain, relation, purpose
2. **Semantic understanding**: not just keyword matching, but understanding argument logic
3. **Network analysis**: discover hidden theoretical connections and evolution paths
4. **Personalized queries**: adapt to different types of academic questions

## 🎯 Milestone Plan

| Phase | Time | Key Deliverables | Validation Criteria |
|------|------|-----------|----------|
| Phase 1 | Week 1-3 | Context-enhanced DocumentProcessor | Can identify paragraph themes and discourse roles |
| Phase 2 | Week 4-7 | Semantic-rich relationship analyzer | Can identify citation purpose and stance |
| Phase 3 | Week 8-12 | Complete academic query engine | Supports complex natural language queries |

This plan will upgrade your existing v2.0 architecture to a true academic-level RAG system, capable of supporting ChatGPT's high-freedom query needs. The key is to gradually increase semantic depth and query flexibility while maintaining the advantages of the existing architecture. 