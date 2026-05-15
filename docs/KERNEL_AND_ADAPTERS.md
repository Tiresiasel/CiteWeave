# CiteWeave as a Kernel + Adapter System

This document defines the architecture for operating CiteWeave through multiple
agent runtimes without coupling the research engine to one frontend.

## Core Idea

Treat **CiteWeave as a research kernel**:

- PDF ingestion;
- document structure extraction;
- citation parsing and citation-context construction;
- Neo4j graph maintenance;
- Qdrant vector indexing and search;
- research query routing and synthesis;
- persistence in local artifacts.

Treat **CLI, Codex workflows, OpenClaw, Claude Code, future HTTP APIs, and other
agent runtimes** as adapters.

The adapter owns the conversation and operational choreography. CiteWeave owns
the local research substrate.

## Layers

### 1. Kernel Layer

Locations:

- `src/kernel/service.py`
- `src/kernel/batch_tracker.py`
- `src/kernel/query_history.py`

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
- compose heavy internals such as `DocumentProcessor` and `LangGraphResearchSystem`;
- remain adapter-agnostic.

### 2. CLI Adapter

Location:

- `src/core/cli.py`

Responsibilities:

- parse command-line arguments;
- print human-readable or JSON output;
- keep the interactive `chat` loop in terminal form;
- call `CiteWeaveKernel` rather than manually wiring internals.

### 3. Agent Template

Location:

- `docs/agent/`

Responsibilities:

- define the generic installation protocol;
- define the generic operating contract;
- describe how any shell-capable Research Agent should apply configuration,
  start services, validate ingestion, and call CiteWeave operations.

The agent template is the canonical documentation for Codex, OpenClaw, Claude
Code, and custom local agents.

### 4. Adapter-Specific Extensions

Locations:

- `docs/agent/<runtime>/`
- `src/adapters/`

Adapter-specific docs and facades live here. For example:

- `docs/agent/openclaw/` documents OpenClaw gateway and facade notes;
- `src/adapters/openclaw_facade.py` exposes a structured OpenClaw-oriented wrapper
  around the kernel.

Adapter-specific code should stay thin. It should translate the runtime's tool or
plugin interface into kernel calls, not reimplement CiteWeave's research logic.

## Operating Boundary

The boundary is simple:

> The agent decides what the user is trying to do. CiteWeave decides how local
> research evidence should be retrieved.

For research questions, adapters pass the user's full question to `query(...)`
first. Low-level Neo4j or Qdrant calls belong to diagnostics and specialized
maintenance workflows.

## Why This Split Matters

Without this split, three things get tangled together:

- research engine logic;
- user interface concerns;
- runtime-specific orchestration.

The kernel/adapter split keeps the database-backed research engine stable while
allowing different agents and frontends to evolve independently.

## Future Work

- add a local HTTP adapter;
- publish first-class adapters for more agent runtimes;
- expand machine-readable health/status coverage;
- add integration tests that exercise kernel methods directly;
- add deeper reference-manager metadata enrichment beyond recursive PDF discovery.
