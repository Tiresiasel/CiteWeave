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

Treat **CLI / OpenClaw Package / OpenClaw Skill / future HTTP API** as adapters.

OpenClaw is the entrypoint, query interface, and deployment coordinator. It does
not own the database substrate. CiteWeave owns the local research stack:
Zotero/PDF ingestion, the Docker Compose service layer, Neo4j, Qdrant, GROBID,
embeddings, and the research kernel.

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
  - `batch_upload(directory, resume=True, force_restart=False, clear_progress=False)`
  - `query_history_snapshot(...)`
  - `list_pending_citations_snapshot(limit=10)`
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
- `batch_upload(directory, resume=True, force_restart=False, clear_progress=False)`
- `query_history(...)`
- `list_pending_citations(limit=10)`
- `health()`
- `bootstrap_plan()`
- `chat_turn(user_input, history=None, menu_choice=None, collected_data=None)` — compatibility path, not the preferred OpenClaw package entrypoint.

For the full OpenClaw action and query-routing contract, see
[`docs/openclaw/PACKAGE_INTERFACE.md`](openclaw/PACKAGE_INTERFACE.md).

## Recommended future OpenClaw Skill design

The future skill should **not** reimplement CiteWeave logic.
Instead it should:

1. ensure CiteWeave is bootstrapped
2. check `docker-compose` services and OpenClaw gateway health
3. call the OpenClaw facade (or a future local HTTP API)
4. format results back to the user

### Suggested Skill workflow

1. `bootstrap_openclaw.sh` if missing / first run
2. deploy the local Docker Compose stack via `scripts/deploy_local_stack.sh` when infrastructure repair/restart is needed
3. ask the user for the Zotero data directory and persist it as `CITEWEAVE_ZOTERO_LIBRARY_DIR`
4. dry-run `scripts/sync_zotero_pdfs.py --dry-run --json`
5. schedule `scripts/sync_zotero_pdfs.py --json` for recurring ingestion
6. call `OpenClawCiteWeaveFacade.routes()` for environment diagnostics
7. call `upload_pdf()` / `query()` / `diagnose_pdf()` / `progress()` as needed
8. optionally maintain session memory at the OpenClaw layer, not in the kernel

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
- publish a first-class OpenClaw package installer
- add deeper Zotero metadata enrichment from `zotero.sqlite`
- expand machine-readable health/status coverage beyond the current CLI `health` / `bootstrap-plan` commands
- add integration tests that exercise kernel methods directly
