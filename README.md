# CiteWeave

**A local citation intelligence stack operated through OpenClaw.**

CiteWeave turns academic PDFs into a searchable research system. OpenClaw is the entrypoint: it helps deploy the stack, keeps a Zotero library synced, and gives the user a natural-language interface for upload, diagnosis, querying, and maintenance. CiteWeave owns the local infrastructure behind that interface: PDF extraction, citation parsing, embeddings, Neo4j, Qdrant, GROBID, and the research query kernel.

[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Start here

This README is for human readers: what the project is, what it can do, and how the pieces fit.

If you are **OpenClaw** or an operator deploying CiteWeave, read:

- [`docs/openclaw/README.md`](docs/openclaw/README.md) — OpenClaw operating index;
- [`docs/openclaw/DEPLOYMENT.md`](docs/openclaw/DEPLOYMENT.md) — step-by-step local deployment;
- [`docs/openclaw/PACKAGE_INTERFACE.md`](docs/openclaw/PACKAGE_INTERFACE.md) — actions, interfaces, and query-routing logic.

That separation is intentional. README explains the product. `docs/openclaw/` explains how to operate it.

---

## What CiteWeave does

CiteWeave builds a local research database from academic PDFs. Once deployed, an OpenClaw user can ask it to:

- keep a Zotero library synced into CiteWeave;
- upload individual PDFs or batch-ingest PDF folders;
- diagnose PDF extraction quality before ingestion;
- build and query a Neo4j citation graph;
- build and query Qdrant semantic vector indexes;
- retrieve evidence from graph, vector, author, and PDF-content routes;
- answer literature, author, citation, argument, and compound research questions;
- inspect health, routes, ingestion progress, unresolved citations, and query history.

Typical use cases:

- literature reviews;
- citation tracing;
- theory lineage;
- author and paper comparison;
- unresolved-reference cleanup;
- “where did this idea come from?” research.

---

## Product model

CiteWeave is designed as an **OpenClaw Package**.

OpenClaw provides:

- the user-facing natural-language entrypoint;
- deployment coordination;
- Docker Compose stack startup for local infrastructure;
- recurring Zotero ingestion automation;
- operation selection: upload, sync, diagnose, query, health, progress, telemetry;
- agent-facing calls into CiteWeave.

CiteWeave provides:

- Zotero/PDF ingestion;
- PDF parsing and citation extraction;
- Neo4j graph storage;
- Qdrant vector storage;
- local or OpenAI embeddings;
- GROBID-backed metadata and structure extraction;
- research query planning and answer synthesis.

The boundary matters:

> OpenClaw decides what operation the user wants. CiteWeave decides how to retrieve and synthesize research evidence.

OpenClaw is not the database layer. It is the entrypoint and coordinator for a local research stack.

---

## Architecture

```text
Human researcher
    │
    ▼
OpenClaw
    │  natural-language entrypoint, deployment coordination,
    │  Zotero sync scheduling, operation selection
    ▼
CiteWeave OpenClaw adapter
    │  src/adapters/openclaw_facade.py
    ▼
CiteWeave kernel
    │  upload, diagnose, route, query, progress, telemetry
    ▼
Local research infrastructure
    ├── Docker Compose stack
    │   ├── Neo4j citation graph
    │   ├── Qdrant vector indexes
    │   └── GROBID PDF extraction
    ├── Zotero PDF source
    └── Embeddings
        ├── local SentenceTransformers   default
        └── OpenAI Embeddings            optional
```

The CLI still exists, but it is an operational adapter. It is useful for verification and debugging; it is not the product center.

---

## Zotero as the persistent data source

A normal deployment starts by telling OpenClaw where the user's Zotero library lives. CiteWeave then treats that library as the continuing PDF source.

OpenClaw persists the source path as:

```env
CITEWEAVE_ZOTERO_LIBRARY_DIR=/path/to/Zotero
```

OpenClaw brings up the local service layer through Docker Compose, then schedules recurring ingestion through:

```bash
.venv/bin/python scripts/sync_zotero_pdfs.py --json
```

The sync script resolves Zotero `storage/`, discovers PDFs recursively, and delegates to CiteWeave's resumable batch uploader. This lets the local research database grow continuously as the Zotero library changes.

For the exact setup procedure, see [`docs/openclaw/DEPLOYMENT.md`](docs/openclaw/DEPLOYMENT.md).

---

## Query model

OpenClaw should not manually query Neo4j or Qdrant for normal research questions. It should pass the user's research question to CiteWeave:

```python
facade.query(question, confirmation="continue")
```

CiteWeave then decides which internal routes to use:

- semantic vector search;
- graph citation traversal;
- author and paper lookup;
- extracted PDF content;
- unresolved citation tracking;
- final answer synthesis.

For operational requests, OpenClaw calls the specific action instead: `upload_pdf`, `batch_upload`, `diagnose_pdf`, `progress`, `health`, `routes`, `query_history`, or `list_pending_citations`.

Full interface documentation: [`docs/openclaw/PACKAGE_INTERFACE.md`](docs/openclaw/PACKAGE_INTERFACE.md).

---

## Embeddings

CiteWeave currently supports two embedding schemes:

| Provider | Default? | Model | Vector size | API key |
|---|---:|---|---:|---|
| `local` | yes | `all-MiniLM-L6-v2` | 384 | no |
| `openai` | no | `text-embedding-3-small` | 1536 | yes |

Default local mode keeps installation self-contained. OpenAI embeddings can be enabled when the user is ready to migrate the vector index.

Switching providers changes vector dimensions, so existing Qdrant collections must be recreated or migrated before mixing providers. Databases do tend to remember their shape. Annoying, but useful.

---

## Documentation map

| Document | Audience | Purpose |
|---|---|---|
| [`README.md`](README.md) | Human reader | Product overview and architecture |
| [`README.zh.md`](README.zh.md) | Chinese human reader | Chinese overview |
| [`docs/openclaw/README.md`](docs/openclaw/README.md) | OpenClaw / operator | Operational entrypoint |
| [`docs/openclaw/DEPLOYMENT.md`](docs/openclaw/DEPLOYMENT.md) | OpenClaw / operator | Local deployment and Zotero sync |
| [`docs/openclaw/PACKAGE_INTERFACE.md`](docs/openclaw/PACKAGE_INTERFACE.md) | OpenClaw / integrator | Actions, interfaces, query logic |
| [`docs/KERNEL_AND_OPENCLAW.md`](docs/KERNEL_AND_OPENCLAW.md) | Developer | Kernel/adapter architecture |

---

## Development

For development gates, run:

```bash
.venv/bin/python -m ruff check src tests scripts/sync_zotero_pdfs.py
.venv/bin/python -m ruff check tests/manual --select F --ignore E501
python3 -m compileall -q src tests scripts/sync_zotero_pdfs.py
.venv/bin/python -m pytest -q
python3 scripts/repo_privacy_audit.py
```

Expected privacy result:

```text
PRIVACY_AUDIT_OK
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
