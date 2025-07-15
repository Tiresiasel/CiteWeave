# Enhanced RAG Database Structure for Academic Query System

## 🎯 Design Goals

Support high-flexibility academic queries such as:
- "谁引用了Porter (1980)并且支持他的竞争优势理论？"
- "有哪些文章反驳了Nelson and Winter (1982)的演化理论？"
- "在战略管理领域，哪些研究扩展了Resource-Based View？"
- "有哪些文献使用了Milgrom and Roberts的方法论？"

## 🏗️ Enhanced Data Architecture

### 1. Multi-Granularity Context Structure

```json
{
  "document_context": {
    "sentence_level": {
      "sentence_index": 42,
      "sentence_text": "Porter (1980) argues that competitive advantage stems from strategic positioning.",
      "section": "Literature Review",
      "subsection": "Competitive Strategy Theory",
      "paragraph_index": 15,
      "discourse_role": "CLAIM_MAIN"  // Based on your existing claim types
    },
    "paragraph_level": {
      "paragraph_text": "Previous research has established several foundational theories...",
      "paragraph_theme": "theoretical_foundation",
      "argument_flow": "introduction -> elaboration -> synthesis"
    },
    "section_level": {
      "section": "Literature Review",
      "section_purpose": "theory_building",
      "research_domain": "strategic_management"
    }
  }
}
```

### 2. Enhanced Citation-Argument Network

```json
{
  "citation_node": {
    "id": "porter_1980_competitive_strategy",
    "metadata": {
      "title": "Competitive Strategy",
      "authors": ["Michael E. Porter"],
      "year": "1980",
      "research_domain": "strategic_management",
      "key_concepts": ["competitive_advantage", "industry_analysis", "generic_strategies"],
      "theoretical_framework": "industrial_organization"
    },
    "citation_instances": [
      {
        "citing_paper_id": "current_paper_id",
        "citation_context": {
          "sentence_text": "Porter (1980) argues that...",
          "argument_relation": {
            "relation_type": "SUPPORTS",
            "relation_strength": 0.856,
            "argumentative_purpose": "theoretical_foundation",
            "cited_aspect": "competitive_advantage_theory",
            "citing_stance": "acceptance",
            "elaboration_type": "conceptual_extension"
          },
          "semantic_context": {
            "preceding_context": "Previous theories of competition have focused on...",
            "following_context": "This framework enables us to understand...",
            "logical_flow": "premise -> argument -> conclusion"
          }
        }
      }
    ]
  }
}
```

### 3. Argument-Claim Knowledge Graph

```json
{
  "claim_node": {
    "id": "competitive_advantage_stems_from_positioning",
    "claim_text": "competitive advantage stems from strategic positioning",
    "source_paper": "porter_1980",
    "claim_type": "CLAIM_MAIN",
    "theoretical_domain": "strategic_management",
    "evidence_type": "theoretical",
    "citations_supporting": ["paper_a", "paper_b"],
    "citations_refuting": ["paper_c"],
    "citations_extending": ["paper_d", "paper_e"],
    "related_claims": [
      {
        "claim_id": "rbv_resources_create_advantage",
        "relation": "ALTERNATIVE_EXPLANATION",
        "papers_comparing": ["paper_f", "paper_g"]
      }
    ]
  }
}
```

### 4. Research Domain Taxonomy

```json
{
  "domain_structure": {
    "strategic_management": {
      "subdomain": "competitive_strategy",
      "key_theories": [
        {
          "theory_name": "Porter's Five Forces",
          "foundational_papers": ["porter_1980"],
          "supporting_papers": ["paper_list"],
          "challenging_papers": ["paper_list"],
          "extending_papers": ["paper_list"]
        }
      ],
      "methodology_clusters": [
        {
          "method_name": "game_theoretic_modeling",
          "foundational_papers": ["milgrom_roberts_1990"],
          "papers_using_method": ["paper_list"]
        }
      ]
    }
  }
}
```

### 5. Temporal Evolution Tracking

```json
{
  "theory_evolution": {
    "theory_id": "resource_based_view",
    "timeline": [
      {
        "year": "1984",
        "paper_id": "wernerfelt_1984",
        "contribution": "foundational_concept",
        "relation_type": "INTRODUCES"
      },
      {
        "year": "1991", 
        "paper_id": "barney_1991",
        "contribution": "theoretical_development",
        "relation_type": "ELABORATES"
      },
      {
        "year": "1995",
        "paper_id": "peteraf_1993",
        "contribution": "framework_extension",
        "relation_type": "EXTENDS"
      }
    ],
    "current_status": "established_theory",
    "recent_challenges": ["paper_x", "paper_y"],
    "emerging_extensions": ["paper_z"]
  }
}
```

## 🔍 Query-Optimized Indexes

### 1. Relation-Based Indexes

```json
{
  "supports_index": {
    "porter_1980": {
      "supporting_papers": [
        {
          "paper_id": "paper_a",
          "strength": 0.9,
          "aspect_supported": "competitive_positioning",
          "section": "theory_section"
        }
      ]
    }
  },
  "refutes_index": {
    "nelson_winter_1982": {
      "refuting_papers": [
        {
          "paper_id": "paper_b", 
          "strength": 0.8,
          "aspect_refuted": "evolutionary_path_dependence",
          "alternative_proposed": "deliberate_strategy_formation"
        }
      ]
    }
  }
}
```

### 2. Author-Concept Network

```json
{
  "author_contributions": {
    "michael_porter": {
      "key_concepts": ["five_forces", "generic_strategies", "value_chain"],
      "papers_citing_positively": ["list"],
      "papers_citing_critically": ["list"],
      "concept_evolution": {
        "five_forces": {
          "extensions": ["paper_list"],
          "criticisms": ["paper_list"],
          "applications": ["paper_list"]
        }
      }
    }
  }
}
```

### 3. Methodology Tracking

```json
{
  "methodology_network": {
    "game_theory": {
      "foundational_papers": ["milgrom_roberts_1990"],
      "papers_using_method": [
        {
          "paper_id": "paper_x",
          "application_context": "strategic_complementarities",
          "innovation": "extended_to_dynamic_setting"
        }
      ],
      "methodological_variations": ["refinements", "adaptations"]
    }
  }
}
```

## 🚀 Implementation Strategy

### Phase 1: Enhance Current Structure
1. **Add context granularity** (sentence/paragraph/section)
2. **Expand relation semantics** (strength, purpose, aspect)
3. **Implement domain classification**

### Phase 2: Build Knowledge Graph Layer
1. **Create claim-level nodes**
2. **Build theory evolution tracking**
3. **Implement cross-paper concept linking**

### Phase 3: Query Optimization
1. **Build specialized indexes**
2. **Implement semantic search optimization**
3. **Add temporal query support**

## 🔧 Database Schema Design

### Neo4j Graph Structure

```cypher
// Core nodes
(:Paper {id, title, authors, year, domain, key_concepts})
(:Claim {id, text, type, domain, evidence_type})
(:Author {name, affiliations, research_domains})
(:Concept {name, domain, definition})
(:Theory {name, domain, status, foundational_papers})

// Enhanced relationships
(:Paper)-[:CITES {relation_type, strength, aspect, purpose}]->(:Paper)
(:Paper)-[:CONTAINS]->(:Claim)
(:Claim)-[:SUPPORTS|REFUTES|EXTENDS {strength, evidence}]->(:Claim)
(:Author)-[:PROPOSES]->(:Theory)
(:Paper)-[:APPLIES_METHOD {innovation, context}]->(:Paper)
```

### Vector Database Enhancement

```json
{
  "embedding_strategy": {
    "sentence_embeddings": "for precise citation context",
    "claim_embeddings": "for argument-level retrieval", 
    "concept_embeddings": "for semantic concept matching",
    "methodology_embeddings": "for method-based queries"
  },
  "metadata_enrichment": {
    "temporal_info": "year, period, evolution_stage",
    "domain_info": "field, subfield, interdisciplinary_tags",
    "relation_info": "relation_type, strength, purpose",
    "context_info": "section, discourse_role, argument_position"
  }
}
```

This enhanced structure enables complex queries like:
- **Temporal queries**: "How has the RBV theory evolved since 1990?"
- **Methodological queries**: "Which papers use game theory for strategic analysis?"
- **Argumentative queries**: "Find all papers that challenge Porter's five forces framework"
- **Domain-specific queries**: "Show me strategic management papers that extend organizational ecology theories" 