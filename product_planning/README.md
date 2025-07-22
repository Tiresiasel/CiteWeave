> **This project is licensed under the Apache License 2.0. See the LICENSE file for details.**

# CiteWeave Product Planning Document Center

## 📁 Document Structure

This folder contains all product design, development plans, and progress tracking documents for the CiteWeave project.

### 📋 Core Documents

| Document | Description | Status |
|------|------|------|
| [`PRODUCT_SPEC.md`](./PRODUCT_SPEC.md) | **Product Requirements Document (PRD)** - Overall project design, feature requirements, technical architecture | ✅ Latest |
| [`Development_Logs.md`](./Development_Logs.md) | **Development Logs** - Module progress, test results, milestone records | 🔄 Ongoing |

### 🚀 RAG System Evolution Plan

| Document | Description | Timeframe |
|------|------|----------|
| [`RAG_OPTIMIZATION_PLAN.md`](./RAG_OPTIMIZATION_PLAN.md) | **RAG Optimization Plan** - Three-phase roadmap, technical architecture | Long-term |
| [`PHASE1_IMPLEMENTATION_PLAN.md`](./PHASE1_IMPLEMENTATION_PLAN.md) | **Phase 1 Implementation Plan** - Context granularity enhancement (v0.7) | 3 weeks |

## 🎯 Project Version Planning

### Current Version: v0.6.1
- ✅ Unified DocumentProcessor architecture
- ✅ Sentence-level citation analysis (v2.0)
- ✅ Argument-level citation relationship embedding
- ✅ JSON/JSONL standardized output

### Planned Version Roadmap

#### 🔥 v0.7 (Phase 1) - Context Granularity Enhancement
**Estimated Time**: 3 weeks  
**Core Features**:
- Paragraph-level context recognition
- Automatic section classification
- Discourse role analysis
- Semantic position annotation

#### 🚀 v0.8 (Phase 2) - Semantic Relationship Deepening
**Estimated Time**: 3-4 weeks  
**Core Features**:
- Author-concept network construction
- Citation stance classification
- Concept extractor
- Semantically rich relationship metadata

#### 🌟 v0.9 (Phase 3) - Knowledge Graph & Intelligent Query
**Estimated Time**: 4-5 weeks  
**Core Features**:
- Complete Neo4j graph architecture
- Natural language query engine
- Temporal evolution tracking
- Multi-level vector retrieval

## 📊 Development Progress Overview

### Completed Modules
- [x] **PDFProcessor** - PDF text extraction, metadata processing
- [x] **CitationParser** - Citation detection and parsing (100% test coverage)
- [x] **DocumentProcessor** - Unified document processing coordinator
- [x] **GraphBuilder** - Citation network construction
- [x] **VectorIndexer** - Semantic search and embeddings

### Modules in Development
- [ ] **ArgumentClassifier** - Argument relationship classification (partially complete)
- [ ] **QueryAgent** - Intelligent query engine (planned)
- [ ] **StubResolver** - Citation completion mechanism (planned)
- [ ] **CLIInterface** - Command-line interface optimization (planned)

## 🎯 Query Capability Evolution Goals

| Version | Query Capability Level | Example Query |
|------|----------------------|--------------|
| **v0.6** (Current) | Basic citation retrieval | "Find sentences citing Porter (1980)" |
| **v0.7** (Phase 1) | Context-aware retrieval | "Find sentences citing Porter in the literature review section" |
| **v0.8** (Phase 2) | Semantic stance query | "Management papers supporting Porter's competitive positioning theory" |
| **v0.9** (Phase 3) | Complex academic query | "Trace the evolution and main criticisms of RBV theory from 1990 to present" |

## 📝 Documentation Maintenance Notes

### Update Frequency
- **PRODUCT_SPEC.md**: Updated on major releases
- **Development_Logs.md**: Updated when each module is completed
- **Phase plan documents**: Finalized before each phase, adjusted during implementation

### Version Control
- All documents use Git version control
- Important changes are recorded in each document's version history
- Cross-references are kept in sync

---

**Last updated**: 2025-01-15  
**Maintainer**: Project Team  
**Next review**: Before Phase 1 starts 