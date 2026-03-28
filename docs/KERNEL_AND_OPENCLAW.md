# CiteWeave as a Kernel + Adapter System

This document defines the intended architecture for integrating CiteWeave with
OpenClaw without coupling the research kernel to any one frontend.

## Core idea

Treat **CiteWeave as a kernel**:

- PDF ingestion
- citation graph maintenance
- vector search
- research workflow orchestration
- route diagnostics
- persistence in Neo4j / Qdrant / derived files

Treat **CLI / OpenClaw Skill / future HTTP API** as adapters.

## Layers

### 1. Kernel layer

Location:

- `src/kernel/service.py`
- `src/kernel/batch_tracker.py`

Responsibilities:

- expose stable application operations:
  - `upload_document(pdf_path)`
  - `diagnose_document(pdf_path)`
  - `query(question, confirmation='continue')`
  - `routes_snapshot()`
  - `progress_summary(directory, clear=False)`
- compose heavy internals (`DocumentProcessor`, `LangGraphResearchSystem`)
- remain adapter-agnostic

### 2. CLI adapter

Location:

- `src/core/cli.py`

Responsibilities:

- parse argv
- print human-readable output
- keep interactive `chat` loop in terminal form
- call `CiteWeaveKernel` instead of directly wiring internals everywhere

### 3. OpenClaw adapter

Location:

- `src/adapters/openclaw_facade.py`

Responsibilities:

- expose serializable methods suitable for a future Skill
- avoid terminal scraping
- return structured dictionaries for agent/tool handling

Current facade methods:

- `upload_pdf(pdf_path)`
- `diagnose_pdf(pdf_path)`
- `query(question, confirmation='continue')`
- `routes()`
- `progress(directory, clear=False)`

## Recommended future OpenClaw Skill design

The future skill should **not** reimplement CiteWeave logic.
Instead it should:

1. ensure CiteWeave is bootstrapped
2. check `docker-compose` services and OpenClaw gateway health
3. call the OpenClaw facade (or a future local HTTP API)
4. format results back to the user

### Suggested Skill workflow

1. `bootstrap_openclaw.sh` if missing / first run
2. call `OpenClawCiteWeaveFacade.routes()` for environment diagnostics
3. call `upload_pdf()` / `query()` / `diagnose_pdf()` as needed
4. optionally maintain session memory at the OpenClaw layer, not in the kernel

## Why this split matters

Without this split, three things get tangled together:

- research engine logic
- user interface concerns
- OpenClaw-specific orchestration

That makes docs, testing, and future APIs drift apart. The kernel/adapter split
keeps the database-backed research engine stable while allowing multiple entry
points to evolve independently.

## What remains future work

- add a local HTTP adapter (likely FastAPI)
- make batch upload callable through the OpenClaw facade
- add machine-readable health/status commands beyond current CLI output
- add integration tests that exercise kernel methods directly
