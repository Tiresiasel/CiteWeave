# Development Logs

## **2026-05-14**
- **paper index PDF-status filtering**: added `papers --pdf-status {all,available,missing}` so local corpus audits can isolate entries that are query-ready versus metadata-only without exposing absolute PDF paths.
- **test coverage**: added regression coverage for privacy-safe PDF-status filtering in the paper index snapshot and CLI JSON output.

## **2026-05-11**
- **OCR availability diagnostics**: PDF processing now distinguishes missing Python OCR packages from a missing `tesseract` binary, so scanned-PDF failures point to the actual setup gap instead of silently advertising OCR as available.
- **test coverage**: added focused regression coverage for OCR availability reason messages.

## **2026-05-10**
- **local paper index CLI**: added `papers` to search/list the local author-paper index by title, author, journal, or year while exposing only PDF availability instead of absolute local paths.
- **privacy-aware diagnostics**: paper index snapshots now omit raw `pdf_path` values, making local corpus checks safer to share in automation output.
- **test coverage**: added CLI regression coverage for JSON output and empty-result guidance.

## **2026-05-13**
- **resumable Zotero ingestion hardening**: added sequential per-file timeout support plus `--skip-failed` resume controls so one pathological PDF cannot trap a long batch in a restart loop.
- **test coverage**: added regression coverage for failed-file skipping and Zotero sync forwarding of resume safety flags.

## **2026-05-12**
- **citation-analysis paper resolution**: added deterministic author+year fallback for queries like “Toh and Ahuja 2022” when title lookup fails, avoiding incorrect latest-upload fallback and resolving the cited/source paper from intersected author matches.
- **test coverage**: added regression coverage for author-year citation extraction routing.

## **2026-05-07**
- **query-history triage sorting**: added `query-history --sort` for ordering the displayed matching window by recency, age, duration, or response size before `--limit`, so daily audits can inspect slowest, fastest, longest, or shortest responses without post-processing JSONL.
- **test coverage**: added regression coverage that sorted displays keep full matching-window metrics intact while preserving chronological latest-query diagnostics.

This document tracks the development progress of all major modules in the project.

## Module Checklist (TodoDSS)

- [x] PDFProcessor (Enhanced with MinerU integration)
- [x] GraphBuilder
- [ ] ArgumentClassifier (Aborted)
- [x] CitationParser (Enhanced with page number support)
- [x] VectorIndexer (Multi-level indexing)
- [x] QueryAgent (Basic functions implemented)
- [ ] StubResolver
- [x] CLIInterface (Basic functionality)
- [x] DocumentProcessor (Parallel structure redesign)

> Check off each module as it is completed or reaches a stable milestone.

## Recent Development Progress

## **2026-05-06**
- **query-history route-plan quality gates**: added automation checks for empty query plans and entries without resolved planned routes, using the full matching telemetry window rather than only displayed rows.
- **CLI diagnostics**: query-history check output now includes route-plan gap counts and configured thresholds so cron failures explain whether routing coverage or route inference regressed.
- **test coverage**: added regression coverage for route-plan quality gates in CLI query-history checks.

## **2026-05-05**
- **query-history route-plan gaps**: summaries now count recent and full matching-window entries with empty query plans or no resolved planned routes, making router telemetry regressions visible without hand-scanning JSONL rows.
- **CLI diagnostics**: query-history text output surfaces route-less and empty-plan counts, including matching-window totals when `--limit` hides older affected records.
- **test coverage**: added regression coverage for recent vs. matching-window route-plan gap counts.

## **2026-05-04**
- **query-history route labels**: recent-entry plan summaries now infer and label planned routes from database names when telemetry lacks explicit route labels, keeping per-entry diagnostics consistent with matching-window route breakdowns.
- **test coverage**: added regression coverage for inferred and mixed explicit/inferred route labels in CLI query-history plan summaries.

## **2026-05-03**
- **query-history success-rate gate**: added `--check-min-success-rate` so daily automation can fail on degraded successful-query ratios directly, without manually inverting the error-rate threshold.
- **CLI validation output**: query-history checks now report matching-window success rate alongside error rate when available.
- **test coverage**: added regression coverage for minimum success-rate validation failures.

## **2026-05-02**
- **query-history quality gates**: added full-window automation checks for slowest query duration and shortest successful response size so daily audits can fail on latency spikes or suspiciously terse answers without hand-reading JSONL telemetry.
- **test coverage**: added regression coverage for shortest-success response metrics and CLI validation gates that evaluate the full matching window rather than only displayed rows.

## **2026-05-01**
- **query-history full-window CLI labels**: expanded text diagnostics to label matching-window source, confirmation, satisfaction, database, method, and route breakdowns separately from the recent rows shown by `--limit`.
- **test coverage**: added regression coverage for the matching-window labels so limited query-history output cannot silently hide older matching telemetry.

## **2026-04-30**
- **query-history full-window latency/size metrics**: added matching-window average/max duration and response-size diagnostics so limited displays still show full filtered-window performance characteristics.
- **CLI diagnostics**: text output now surfaces matching-window duration and response-size metrics when `--limit` hides older telemetry.
- **test coverage**: added regression coverage for limit-independent matching-window duration and response-size summaries.

## **2026-04-29**
- **query-history full-window breakdowns**: added matching-window source/error/route/plan breakdowns so limited displays still expose repeated failure modes across the complete filtered window.
- **CLI diagnostics**: text output now distinguishes recent-entry breakdowns from full matching-window breakdowns when `--limit` hides older matching telemetry.
- **test coverage**: added regression coverage for limit-independent matching-window breakdowns.

## **2026-04-28**
- **query-history validation window metrics**: added full filtered-window success/error counts and rates so automation thresholds evaluate every matching query, not only the displayed `--limit` rows.
- **CLI validation hardening**: `query-history --check-max-errors` and `--check-max-error-rate` now use full-window metrics while preserving limited display output for humans.
- **test coverage**: added regression coverage for limit-independent summary metrics and CLI threshold checks.

## **2026-04-27**
- **query-history automation thresholds**: added success/error rate metrics plus CLI validation gates for `--check-max-errors` and `--check-max-error-rate`, so cron and CI can tolerate small blips while still failing on sustained query regressions.
- **test coverage**: added regression coverage for summary-level rate metrics and threshold-based CLI checks.

## **2026-04-26**
- **query-history recurring error diagnostics**: added aggregated error breakdowns to `query-history` summaries/text output so daily iteration can spot repeated failure modes without scanning every entry manually.
- **test coverage**: added regression coverage for summary-level error aggregation and CLI rendering.

## **2026-04-24**
- **pending citation diagnostics**: added first-class graph accessors for citation-network stats, unresolved stub-paper listing, and stub resolution updates so delayed citation workflows no longer depend on missing GraphDB methods.
- **CLI coverage**: added `list_pending_citations` to surface the most-cited unresolved stub papers directly from the main CLI, with text/JSON output for local audits and automation.
- **test coverage**: added regression coverage for the new pending-citation CLI output plus GraphDB stub-support helpers.

## **2026-04-23**
- **query-history satisfaction diagnostics**: added satisfaction normalization/bucketing plus `query-history --satisfaction` filtering and satisfaction breakdowns so daily iteration can isolate dissatisfied, neutral, satisfied, or unrated runs instead of leaving the telemetry field unused.
- **test coverage**: added regression coverage for satisfaction filtering and breakdown reporting in query history summaries.

## **2026-04-21**
- **query-history terse-answer filters**: added `--min-response-chars` / `--max-response-chars` so autonomous iteration and local audits can isolate suspiciously short answers or intentionally concise responses without grepping raw JSONL.
- **test coverage**: added regression coverage for response-size filtering in recorder summaries and CLI output/check flows.

## **2026-04-20**
- **query-history UX telemetry**: extended `query-history` matching and text diagnostics to include `response_preview` / `response_chars`, so daily iteration can audit answer quality from the local JSONL log instead of only question/error metadata.
- **response-size summaries**: added average/max response length metrics plus latest response preview in query-history summaries, making it easier to spot terse failures versus useful answers during autonomous iteration.
- **test coverage**: added regression coverage for response-preview filtering and the enriched CLI output.

## **2026-04-18**
- **slow-query diagnostics**: added `query-history --min-duration-ms` so recent telemetry can isolate latency regressions and support automation against slow query windows instead of only empty/non-empty checks.
- **test coverage**: added regression coverage for minimum-duration filtering in recorder summaries and CLI output/check flows.

## **2026-04-17**
- **query-history automation gate**: added CLI `query-history --check-empty` so filtered history windows can fail fast in cron/CI when recent matching entries exist, with concise text and JSON validation output for scripted diagnostics.
- **test coverage**: added regression coverage for both clean and failing `--check-empty` query-history checks.

## **2026-04-09**
- **query history source diagnostics**: extended query telemetry with entrypoint source labels (`cli.query`, `openclaw.facade.query`, etc.) and added CLI `query-history --source` filtering plus source/confirmation breakdowns so validation traffic can be separated from real usage windows during daily iteration.
- **test coverage**: added regression coverage for source-aware query history summaries, CLI filtering, and OpenClaw facade query tagging.

## **2026-04-08**
- **repo hygiene clarification**: documented the canonical `.gitignore` rule for `test_files/*` so consistency checks do not keep inventing punctuation as a feature.
- **query history time-window inspection**: extended `query-history` with `--since-hours` and timestamp/relative-age output so recent telemetry windows can be reviewed directly from the CLI during daily iteration.
- **test coverage**: added regression coverage for time-window filtering and query-history CLI output.

## **2026-04-06**
- **query history inspection**: added a first-class `query-history` CLI command and kernel snapshot so recent query telemetry can be reviewed without reading raw JSONL by hand; includes recent entries, success/error counts, and latency summary for daily UX iteration.
- **test coverage**: added regression tests for query-history summaries, corrupt-row handling, and actionable CLI output.

## **2026-04-05**
- **batch progress ETA & throughput diagnostics**: extended batch upload tracking to record per-file processing duration, surface average completed time per file, and estimate remaining wall time in `progress` output/JSON so long-running PDF ingests are easier to monitor and resume.
- **test coverage**: added regression coverage for duration persistence, progress ETA calculation, and enriched sequential batch tracker payloads.

## **2025-07-22**
- **information-collection summary**: Fix the infomration collection summary issue.
- **ai_evaluate_sufficiency-agent**: A new agent that is responsible for judging if the information gathered is sufficient to answer the user's question.







## **CLI Interface Development (2025-07-22)**
- **Parallel Processing**: Added parallel processing support for batch upload.
- **Sequential Processing**: Added sequential processing support for batch upload.
- **Progress Tracking**: Added progress tracking for batch upload.
- **Error Handling**: Added error handling for batch upload.
- **User Feedback**: Added user feedback for batch upload.
- **Logging**: Added logging for batch upload.
- **Documentation**: Added documentation for batch upload.

- **Enhanced CLI Interface**:
  - Interactive multi-turn chat with spinner/progress indicator for AI thinking
  - Environment-based logging (CITEWEAVE_ENV: production/development/test)
  - .env file support for environment variables
  - Batch upload of PDFs via CLI
  - Robust error handling and user feedback in CLI
  - Dynamic debug/info log control for developers
  - Automatic loading of environment variables before CLI runs
  - Improved vector search result aggregation and LLM prompt sampling for LLM context
  - Stricter answer generation for content-based queries (no hallucination if content missing)
  - Cleaner, user-friendly CLI output in production (minimal logs)
  - Modular, extensible CLI command structure for future features
- **Improved Documentation**: Added a new CLI interface for the folder batch upload.

## **MAJOR ARCHITECTURAL OVERHAUL (2025-07-16)**

### **Parallel Structure Redesign - COMPLETED ✅**
- **Revolutionary Change**: Complete redesign of `processed_document.json` structure
- **New Architecture**: `sections[]`, `paragraphs[]`, `sentences[]` as independent parallel arrays
- **Unified Citation Format**: All levels (sections, paragraphs, sentences) use identical citation structure
- **Performance Optimization**: Eliminated nesting for faster querying and better data access
- **Enhanced Statistics**: Added `sections_with_citations`, `paragraphs_with_citations` counters

### **Document Structure Changes**:
```json
{
  "sections": [{"section_index": 0, "citations": [...]}],
  "paragraphs": [{"paragraph_index": 0, "citations": [...]}], 
  "sentences": [{"sentence_index": 0, "citations": [...]}]
}
```

### **Code Cleanup - COMPLETED ✅**
- **Removed Obsolete Methods**: Cleaned up `src/document_processor.py`
  - Removed: `_create_graph_entries()` (old method)
  - Removed: `_group_sentences_into_paragraphs()` (replaced by PDF structure)
  - Removed: `_find_paragraph_for_sentence()` (replaced by mapping logic)
  - Removed: `_determine_section()` (replaced by PDF section detection)
  - Removed: `_aggregate_paragraph_citations()` (integrated into mapping)
  - Removed: Obsolete dataclass definitions
- **File Size Reduction**: Reduced from ~1200 lines to 1004 lines (16% reduction)
- **Maintained Functionality**: All existing features preserved with better architecture

## MinerU Integration - COMPLETED ✅ (2025-07-16)

### **High-Quality PDF Processing**
- **Optional Feature**: MinerU integrated as configurable high-priority PDF parser
- **Superior Accuracy**: 95% vs 85% accuracy compared to traditional methods
- **Markdown Output**: Converts PDF to structured Markdown for simplified processing
- **Smart Detection**: Automatic table/formula processing and header/footer detection

### **Configuration Control**:
```json
{
  "pdf_processing": {
    "enable_mineru": false,  // Default: disabled due to high computational cost
    "mineru_fallback": true,
    "mineru_config": {...}
  }
}
```

### **Installation & Usage**:
```bash
pip install magic-pdf[full]  # Install MinerU
# Edit config/model_config.json to enable
```

## Renewed GraphDB Structure (2025-07-21)
- **Major structural change:** The citation relationships in the graph are now strictly from Sentence→Paper and Paragraph→Paper. The previous Argument→Paper citation relationships have been removed/replaced. This ensures all citation edges are anchored at the sentence or paragraph level, making the graph structure more precise, queryable, and robust.
- Unified paper_id generation using `PaperIDGenerator` (SHA256) for all Paper nodes and embedding payloads
- GraphDB `Paragraph` node now includes `has_citations` attribute
- DocumentProcessor and GraphDB integration: Paragraph creation now sets `has_citations` based on `citation_count`
- Updated `docs/data_structures/README.md` to reflect current graph and embedding database structure
- Confirmed that all graph operations (`MERGE`) are idempotent (no duplicate nodes/edges)
- Added/updated test scripts for graph structure and citation relationships 

## 🧠 Multi-Agent Research System Development (2025-07-21)

- **LLM-Driven Query Analysis:** Replaced all rule-based and regex entity extraction with a dedicated LLM-powered entity extraction agent. The system now uses a configurable language model to extract authors, paper titles, concepts, and other entities from user queries, enabling robust and context-aware intent detection.
- **Sophisticated Stepwise Logging:** Every major step in the multi-agent workflow (entity extraction, LLM intent analysis, tool execution, disambiguation, etc.) now logs both 'step_start' and 'step_finish' events, including results, errors, and request IDs. This enables full traceability and debugging for every research query.
- **Agent Orchestration via LangGraph:** The research system is orchestrated using LangGraph, with each agent (entity extractor, query planner, tool executor, response generator) operating as a modular, traceable step in the workflow.
- **Dynamic Model Configuration:** All LLM agents (query analyzer, response generator, entity extractor, etc.) are now configured via `config/model_config.json`, allowing for easy model swaps and parameter tuning without code changes.
- **Disambiguation and Clarification:** When multiple authors or papers match a query, the system now prompts the user for clarification, rather than making silent or incorrect choices. This logic is applied to both author and paper title searches.
- **LLM-First Query Routing:** The system routes queries to the correct database/tool (graph, vector, PDF) based on LLM-extracted intent and entities, eliminating reliance on brittle rule-based logic.
- **Comprehensive Logging for All Data Retrieval:** All data retrieval steps (tool calls, database queries, LLM responses) are logged with input, output, and error details, ensuring every piece of retrieved data is traceable.
- **Production-Grade Observability:** The logging and modular agent design make the system suitable for production deployment, debugging, and audit.
- **Information Confirmation Layer (2025-07-21):** Added a new layer that shows users what information has been gathered and asks for confirmation before providing the final answer. This includes:
  - **InformationSummaryAgent**: Summarizes gathered data in user-friendly format with confidence assessment
  - **UserConfirmationAgent**: Handles user choices (continue/expand/refine) and routes accordingly
  - **Enhanced Workflow**: New workflow steps between data collection and response generation
  - **Interactive Methods**: `research_question_with_confirmation()` and `continue_with_confirmation()` for interactive usage
  - **User Control**: Users can see what was found and choose to continue, expand search, or refine the approach

### CitationParser Module - COMPLETED ✅ (2025-07-14)

**Status**: Fully implemented and tested with comprehensive coverage

**Key Features Implemented**:
- ✅ **Narrative Citation Detection**: Supports "Smith (2020)" format with various prefixes
- ✅ **Parenthetical Citation Detection**: Handles "(Smith, 2020; Jones, 2019)" format  
- ✅ **Multi-word Author Names**: Correctly processes "Van Der Berg (2020)", "World Health Organization (2021)"
- ✅ **Unicode Character Support**: Full support for international names (Turkish, German, French, Spanish, Polish, Czech, Hungarian, Nordic characters)
- ✅ **Complex Multi-author Citations**: Handles "Smith, Jones, and Brown (2020)" and "et al." formats
- ✅ **Prefix Processing**: Intelligently handles academic prefixes like "According to", "Research by", "As noted by", etc.
- ✅ **False Positive Filtering**: Advanced validation to prevent incorrect matches
- ✅ **Multiple Citations per Sentence**: Detects multiple narrative citations in the same sentence

**Test Results**:
- Standard Citation Tests: **29/29 PASSED** (100%)
- Unicode Character Tests: **24/24 PASSED** (100%)
- Total Test Coverage: **53/53 PASSED** (100%)

**Technical Implementation**:
- Multi-layered regex pattern matching for different citation types
- Unicode-aware character pattern generation
- Smart overlap detection for multiple citations
- Comprehensive prefix handling for academic writing styles
- Integration with GROBID for reference extraction and matching

**Files Modified**:
- `src/citation_parser.py` - Core implementation
- `tests/test_intext_citation_extraction.py` - Comprehensive test suite

### Previous Completed Modules

**PDFProcessor**: Text extraction, metadata extraction via GROBID and CrossRef API
**GraphBuilder**: Citation network construction and analysis  
**VectorIndexer**: Semantic search and sentence embedding 
