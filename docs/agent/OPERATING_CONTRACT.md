# Generic Agent Operating Contract

This document describes how a Research Agent should operate CiteWeave after
installation. It applies to Codex, OpenClaw, Claude Code, and custom local agents.

## 1. Operation Selection

The agent should decide the operation. CiteWeave should decide the evidence
route.

Use this rule:

- operational request: call the matching CLI/facade method;
- research question: pass the full question to CiteWeave query handling.

Do not manually query Neo4j or Qdrant from the agent unless the user is asking
for diagnostics or low-level maintenance.

## 2. Common Operations

| User intent | Preferred interface |
|---|---|
| Install or repair local services | `bash scripts/bootstrap_local.sh` or `bash scripts/deploy_local_stack.sh` |
| Check health | `.venv/bin/citeweave health --json` |
| Inspect route configuration | `.venv/bin/citeweave routes --json` |
| Upload one PDF | `.venv/bin/citeweave upload <pdf>` |
| Diagnose one PDF | `.venv/bin/citeweave diagnose <pdf>` |
| Ingest a library or folder | `scripts/sync_literature_pdfs.py --json --skip-failed` |
| Check ingestion progress | `.venv/bin/citeweave progress <source> --json` |
| Ask a research question | `.venv/bin/citeweave query "<question>"` |
| Inspect query telemetry | `.venv/bin/citeweave query-history --json` |
| List missing cited source documents | `.venv/bin/citeweave list_pending_citations --json` |

Agents that use a structured Python boundary can call `CiteWeaveKernel` directly.
Adapter-specific facades, such as the OpenClaw facade, should remain thin wrappers
around that kernel.

## 3. Research Questions

Pass the user's research question intact to CiteWeave for:

- semantic concept queries;
- literature summaries;
- author or paper-specific questions;
- citation-context queries;
- comparative questions;
- compound questions that require graph, vector, and PDF evidence.

Example:

```bash
.venv/bin/citeweave query "Which papers cite Teece 1997, and what are they using it for?"
```

CiteWeave can then choose graph, vector, author, and PDF-content routes internally.

## 4. Scheduling Rules

Recurring sync must be safe:

- check that no active ingestion is already running;
- use resumable ingestion;
- skip failed files unless the user asks to retry;
- never schedule destructive rebuild flags.

Recommended cadence:

- active project: every 5 to 30 minutes;
- normal use: daily;
- very large libraries: daily plus manual runs after large imports.

## 5. Output Expectations

Summarize results for the user instead of dumping raw JSON unless they ask for
debug output.

Good summaries include:

- what operation ran;
- how many PDFs were discovered or processed;
- failed files and the next action;
- whether services are healthy;
- what evidence routes were used;
- which user decision is needed next.

## 6. Embedding Rebuild Rule

If the embedding provider, model, or dimensions change, stop and require explicit
confirmation before rebuilding. A rebuild means recreating or migrating Qdrant
collections, clearing affected progress, and re-ingesting the full corpus.
