# CiteWeave

**Turn a Zotero library into a local citation intelligence system.**

Academic PDFs are full of arguments, citations, methods, and theory trails. Most of that structure is invisible until you start reading by hand. CiteWeave extracts it, indexes it, and makes it searchable: not as another chat wrapper, but as a local research stack with a graph, vector indexes, citation contexts, and a query kernel built for scholarly work.

[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## What it is

CiteWeave ingests academic PDFs and builds a local research database around them:

- **PDF extraction** with GROBID and local PDF parsers;
- **citation parsing** from in-text citations to reference entries;
- **Neo4j graph storage** for papers, citations, paragraphs, sentences, and relationships;
- **Qdrant vector indexes** for semantic retrieval over sentences, paragraphs, sections, and citations;
- **local or API-based embeddings**;
- **a research query kernel** that combines graph, vector, author, citation, and PDF-content routes.

The result is a system you can ask questions like:

- “Which papers build on Teece 1997, and how do they use it?”
- “Find work on multimarket competition and summarize the main theoretical split.”
- “What does this paper cite when it discusses competitive response?”
- “Which references are still unresolved or missing from my library?”

This is the useful part: CiteWeave does not just store PDFs. It tries to preserve the scholarly structure inside them. A PDF folder is a pile. A citation graph with retrievable passages is a research instrument.

---

## Why OpenClaw is involved

CiteWeave is designed to run as a local research package operated through [OpenClaw](https://github.com/openclaw/openclaw).

OpenClaw provides the conversational entrypoint and operational loop: deployment checks, Zotero sync scheduling, upload requests, progress monitoring, diagnostics, and user-facing research questions.

CiteWeave owns the actual research machinery: ingestion, extraction, citation analysis, graph/vector storage, and answer synthesis.

A good rule of thumb:

> OpenClaw decides what the user is asking to do. CiteWeave decides how the research evidence should be retrieved.

For normal research questions, OpenClaw should pass the full question to CiteWeave rather than manually poking Neo4j or Qdrant.

```python
facade.query("Which papers discuss platform competition?", confirmation="continue")
```

---

## How it works

```text
Zotero library / PDF folder
        │
        ▼
PDF extraction + metadata
        │
        ▼
Citation and structure parsing
        │
        ├── Neo4j graph
        │     papers, citations, paragraphs, sentences, relationships
        │
        ├── Qdrant vector indexes
        │     sentences, paragraphs, sections, citations
        │
        └── Processed local artifacts
              metadata, JSONL, original PDFs, diagnostics
        │
        ▼
Research query kernel
        │
        ▼
OpenClaw conversational interface
```

The CLI still exists for maintenance and debugging, but the intended product flow is through the OpenClaw facade.

---

## What you can do with it

### Keep a Zotero library indexed

Point CiteWeave at a Zotero library and it will discover PDFs under `storage/`, process them, and track resumable progress.

```env
CITEWEAVE_ZOTERO_LIBRARY_DIR=/path/to/Zotero
```

```bash
.venv/bin/python scripts/sync_zotero_pdfs.py --source "$CITEWEAVE_ZOTERO_LIBRARY_DIR" --json
```

### Upload or diagnose individual PDFs

```bash
.venv/bin/citeweave upload ./papers/example.pdf
.venv/bin/citeweave diagnose ./papers/example.pdf
```

### Ask research questions

```bash
.venv/bin/citeweave query "Which papers discuss competitive dynamics and platform strategy?"
```

### Monitor ingestion

```bash
.venv/bin/citeweave progress /path/to/Zotero/storage
```

---

## Local infrastructure

A typical local deployment uses Docker Compose for:

- **Neo4j** — citation graph and structured research entities;
- **Qdrant** — semantic vector search;
- **GROBID** — scholarly PDF metadata and reference extraction.

Deployment details live in [`docs/openclaw/DEPLOYMENT.md`](docs/openclaw/DEPLOYMENT.md). The short version:

```bash
bash scripts/bootstrap_openclaw.sh
bash scripts/deploy_local_stack.sh
bash scripts/deployment_check.sh
```

Use those scripts rather than hand-assembling services unless you are debugging. Future-you has enough problems.

---

## Embeddings and vector rebuilds

CiteWeave supports local SentenceTransformers embeddings and OpenAI-compatible embedding providers. The current local configuration can be selected through `config/qdrant_config.json` and environment variables such as:

```env
CITEWEAVE_EMBEDDING_PROVIDER=local
CITEWEAVE_EMBEDDING_PROFILE=bge_large_en
CITEWEAVE_EMBEDDING_DEVICE=auto
```

Important operational rule:

> Changing the base embedding model, provider, or vector dimension requires a full vector rebuild and full corpus re-ingest.

Do **not** resume old ingestion progress after changing embeddings. Old vectors live in the old embedding space. Even if two models produce vectors with the same dimension, their distances are not comparable. Mixing them in one Qdrant collection makes retrieval invalid in the quiet, expensive way.

Required rebuild sequence:

1. Stop active ingestion.
2. Update and verify the embedding configuration.
3. Recreate or migrate the Qdrant collections.
4. Clear batch progress for the affected source.
5. Re-ingest the full corpus from scratch.
6. Run representative queries before trusting the index.

More detail: [`docs/openclaw/DEPLOYMENT.md#9-embedding-configuration`](docs/openclaw/DEPLOYMENT.md#9-embedding-configuration).

---

## Documentation

- [`docs/openclaw/README.md`](docs/openclaw/README.md) — operating CiteWeave through OpenClaw;
- [`docs/openclaw/DEPLOYMENT.md`](docs/openclaw/DEPLOYMENT.md) — local deployment, Zotero sync, health checks, rebuild procedures;
- [`docs/openclaw/PACKAGE_INTERFACE.md`](docs/openclaw/PACKAGE_INTERFACE.md) — facade methods, intent routing, output expectations;
- [`docs/KERNEL_AND_OPENCLAW.md`](docs/KERNEL_AND_OPENCLAW.md) — kernel/adapter architecture;
- [`docs/data_structures/README.md`](docs/data_structures/README.md) — graph and vector data model.

Chinese overview: [`README.zh.md`](README.zh.md).

---

## Development

Useful gates before pushing:

```bash
.venv/bin/python -m ruff check src tests scripts/sync_zotero_pdfs.py
.venv/bin/python -m ruff check tests/manual --select F --ignore E501
python3 -m compileall -q src tests scripts/sync_zotero_pdfs.py
.venv/bin/python -m pytest -q
python3 scripts/repo_privacy_audit.py
```

Expected privacy audit result:

```text
PRIVACY_AUDIT_OK
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
