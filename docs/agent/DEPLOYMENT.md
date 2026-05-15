# Generic Agent Deployment Guide

This guide is for any Research Agent installing CiteWeave locally. It is not
specific to OpenClaw, Codex, Claude Code, or any single runtime.

## 1. Prerequisites

Required on the host machine:

- a shell-capable Research Agent;
- Docker with Docker Compose;
- Python 3.9+;
- a Bash-compatible shell.

Optional:

- Git, if the user wants to clone the repository instead of downloading it;
- an embedding API key, only if the user chooses API embeddings;
- MinerU, only if the user wants the optional PDF-to-Markdown parser.

## 2. Apply User Choices

Use [`INSTALL.md`](INSTALL.md) as the interactive protocol. Store choices through
`scripts/apply_install_choices.py` whenever possible.

Example:

```bash
.venv/bin/python scripts/apply_install_choices.py \
  --research-agent Codex \
  --reference-manager zotero \
  --source-location-mode custom \
  --source-dir /path/to/Zotero \
  --embedding-mode local \
  --embedding-profile bge_large_en \
  --sync-schedule every_5_minutes \
  --processors 10 \
  --skip-failed
```

The script writes runtime settings to `.env` and local installation state to
`config/install_session.local.json`.

## 3. Bootstrap Local Environment

Run:

```bash
bash scripts/bootstrap_local.sh
```

The bootstrap path:

- creates `.env` from `.env_template` when needed;
- generates a local Neo4j password if one is missing;
- creates `.venv`;
- installs Python dependencies;
- installs the local `citeweave` command;
- downloads required NLTK data;
- starts the Docker Compose service layer;
- runs deployment checks.

## 4. Local Service Layer

CiteWeave uses Docker Compose for:

| Service | Bind address | Purpose |
|---|---:|---|
| Neo4j HTTP | `127.0.0.1:7474` | graph UI / health checks |
| Neo4j Bolt | `127.0.0.1:7687` | graph driver connection |
| Qdrant REST | `127.0.0.1:6333` | vector database API |
| Qdrant gRPC | `127.0.0.1:6334` | vector database gRPC |
| GROBID | `127.0.0.1:8070` | PDF structure extraction |

For service repair or restart:

```bash
bash scripts/deploy_local_stack.sh
bash scripts/deployment_check.sh
```

Do not expose these ports publicly unless the user explicitly asks for a secured
remote deployment.

## 5. Validate The Literature Source

Before ingestion, run a dry run:

```bash
.venv/bin/python scripts/sync_literature_pdfs.py \
  --source "$CITEWEAVE_LITERATURE_SOURCE_DIR" \
  --reference-manager "$CITEWEAVE_REFERENCE_MANAGER" \
  --dry-run \
  --json
```

The dry run should:

1. resolve the chosen source directory;
2. recursively discover PDFs;
3. report discovered counts and examples;
4. avoid modifying graph or vector indexes.

If no PDFs are found, ask the user to confirm the source path.

## 6. Run Ingestion

```bash
.venv/bin/python scripts/sync_literature_pdfs.py \
  --source "$CITEWEAVE_LITERATURE_SOURCE_DIR" \
  --reference-manager "$CITEWEAVE_REFERENCE_MANAGER" \
  --json \
  --processors 10 \
  --skip-failed
```

The sync script delegates to CiteWeave's resumable batch uploader and records
progress in `data/batch_upload_tracker.json`.

Recurring sync must be resumable. Do not schedule `--force-restart` or
`--clear-progress` unless the user explicitly confirms a rebuild.

## 7. Health Checks

```bash
.venv/bin/citeweave health --json
.venv/bin/citeweave routes --json
.venv/bin/citeweave progress "$CITEWEAVE_LITERATURE_SOURCE_DIR" --json
```

A usable deployment should have Neo4j, Qdrant, and GROBID reachable, a valid
embedding configuration, and available query routes.

## 8. Embedding Rebuild Rule

Changing the embedding provider, model, or vector dimensions invalidates the
existing Qdrant vector space. Stop ingestion, recreate or migrate vector
collections, clear affected batch progress, and re-ingest the full corpus.

Do not resume old ingestion progress after an embedding change.
