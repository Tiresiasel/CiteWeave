# OpenClaw Deployment Guide for CiteWeave

This is the step-by-step local deployment guide OpenClaw should follow when installing and operating CiteWeave.

The top-level README is intentionally human-oriented. This file is operational.

---

## 1. Prerequisites

Required on the host machine:

- OpenClaw local runtime and gateway;
- Git;
- Docker / Docker Compose;
- Python 3.9+;
- Bash-compatible shell.

Recommended:

- Python 3.12;
- at least 8 GB RAM;
- enough disk for PDFs, Neo4j data, Qdrant collections, Docker images, and model caches.

Default local ports:

| Service | Bind address | Purpose |
|---|---:|---|
| Neo4j HTTP | `127.0.0.1:7474` | graph UI / health checks |
| Neo4j Bolt | `127.0.0.1:7687` | graph driver connection |
| Qdrant REST | `127.0.0.1:6333` | vector database API |
| Qdrant gRPC | `127.0.0.1:6334` | vector database gRPC |
| GROBID | `127.0.0.1:8070` | PDF structure extraction |

Do not expose these ports publicly unless the user explicitly wants a secured remote deployment.

---

## 2. Check OpenClaw

Run:

```bash
openclaw gateway status
```

The gateway should be installed and reachable. CiteWeave uses the OpenClaw gateway for LLM query reasoning when:

```env
CITEWEAVE_LLM_PROVIDER=openclaw
```

---

## 3. Install CiteWeave from source

Current package status: source-backed OpenClaw package install.

```bash
git clone https://github.com/Tiresiasel/CiteWeave.git
cd CiteWeave
```

Then bootstrap:

```bash
bash scripts/bootstrap_openclaw.sh
```

The bootstrap script is the all-in-one path OpenClaw should use for first deployment. It:

- creates `.env` from `.env_template` if needed;
- configures LLM calls for the local OpenClaw gateway;
- keeps embeddings on the local provider by default;
- creates `.venv`;
- installs Python dependencies;
- installs the local `citeweave` operational command;
- downloads required NLTK data;
- deploys the local Docker Compose stack;
- runs the deployment smoke check.

---

## 4. Deploy the local database/service stack

The database and local service layer is intentionally a Docker Compose boundary. OpenClaw should not manually install Neo4j, Qdrant, or GROBID.

Use:

```bash
bash scripts/deploy_local_stack.sh
```

This script:

- checks that Docker and Docker Compose are available;
- creates `.env` if needed;
- generates a non-template local Neo4j password before Neo4j starts;
- validates `docker-compose.yml`;
- starts exactly the local infrastructure services:
  - Neo4j;
  - Qdrant;
  - GROBID.

The Compose file binds services to localhost:

| Service | Bind address | Purpose |
|---|---:|---|
| Neo4j HTTP | `127.0.0.1:7474` | graph UI / health checks |
| Neo4j Bolt | `127.0.0.1:7687` | graph driver connection |
| Qdrant REST | `127.0.0.1:6333` | vector database API |
| Qdrant gRPC | `127.0.0.1:6334` | vector database gRPC |
| GROBID | `127.0.0.1:8070` | PDF structure extraction |

Interface wiring is already reserved in CiteWeave configuration and code:

- Neo4j: graph storage and citation traversal;
- Qdrant: vector index storage and semantic retrieval;
- GROBID: PDF structure and metadata extraction;
- CiteWeave kernel/facade: communication boundary OpenClaw should call.

For first deployment, `scripts/bootstrap_openclaw.sh` calls this path indirectly via `scripts/bootstrap_local.sh`. For repair or restart of only the infrastructure layer, call `scripts/deploy_local_stack.sh` directly.

---

## 5. Verify deployment

Run:

```bash
bash scripts/deployment_check.sh
.venv/bin/citeweave health --json
.venv/bin/citeweave routes --json
```

A usable deployment should have:

- OpenClaw gateway reachable;
- Neo4j reachable and authenticated;
- Qdrant reachable;
- GROBID reachable;
- embedding mode configured;
- routes available.

If verification fails, fix the failing service before scheduling ingestion.

---

## 6. Configure the Zotero data source

At first setup, ask the user for their Zotero data directory.

Accept any of these:

- Zotero data directory containing `zotero.sqlite` and `storage/`;
- Zotero `storage/` directory;
- exported folder of PDFs.

Persist the source in `.env`:

```env
CITEWEAVE_ZOTERO_LIBRARY_DIR=/path/to/Zotero
```

Common candidate locations:

```text
~/Zotero
~/Documents/Zotero
~/Library/Application Support/Zotero/Profiles/<profile>/zotero
```

CiteWeave currently ingests PDF attachments recursively. Zotero SQLite metadata can be used by future enrichments, but the current deployment contract is PDF-first.

---

## 7. Dry-run Zotero discovery

Run before scheduling:

```bash
.venv/bin/python scripts/sync_zotero_pdfs.py --dry-run --json
```

Or pass a source explicitly:

```bash
.venv/bin/python scripts/sync_zotero_pdfs.py --source /path/to/Zotero --dry-run --json
```

Expected behavior:

1. resolve the Zotero source to the PDF-bearing directory;
2. count discovered PDFs;
3. show sample PDFs;
4. show the batch upload command that would run;
5. avoid modifying indexes because this is a dry run.

If no PDFs are found, ask the user to confirm the Zotero path.

---

## 8. Run or schedule recurring ingestion

Manual run:

```bash
.venv/bin/python scripts/sync_zotero_pdfs.py --json
```

Explicit source:

```bash
.venv/bin/python scripts/sync_zotero_pdfs.py --source /path/to/Zotero --json
```

The sync script delegates to:

```bash
.venv/bin/citeweave batch-upload <resolved-pdf-source> --resume
```

Progress is stored in:

```text
data/batch_upload_tracker.json
```

Recommended schedule:

- active literature project: every 1-3 hours;
- normal usage: nightly;
- very large library: nightly plus manual sync after large Zotero imports.

Do not schedule destructive rebuild flags:

- do not use `--force-restart` on a recurring job;
- do not use `--clear-progress` on a recurring job.

Use those only after explicit user confirmation.

---

## 9. Embedding configuration

Embeddings are separate from LLM calls.

Supported schemes:

| Provider | Default? | Model | Vector size | API key |
|---|---:|---|---:|---|
| `local` | yes | `all-MiniLM-L6-v2` | 384 | no |
| `openai` | no | `text-embedding-3-small` | 1536 | yes |

Default local mode:

```env
CITEWEAVE_EMBEDDING_PROVIDER=local
CITEWEAVE_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

OpenAI embedding mode:

```env
CITEWEAVE_EMBEDDING_PROVIDER=openai
CITEWEAVE_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your-openai-api-key
```

Important: local and OpenAI embeddings use different vector sizes by default.

Hard rule: changing the base embedding model, provider, or configured vector dimension is a **full re-ingest event**. Do not resume ingestion into existing Qdrant collections after such a change. Old vectors live in the old embedding space; even when dimensions happen to match, semantic distances are no longer comparable. Mixing old and new embeddings makes retrieval results invalid, not merely noisy.

Required procedure after changing the base embedding model:

1. Stop any running Zotero or batch ingestion job.
2. Update the embedding configuration and verify the resolved model/vector size.
3. Recreate or explicitly migrate all Qdrant vector collections.
4. Clear batch-upload progress for the source directory.
5. Re-ingest the complete corpus from scratch with a force restart.
6. Run a representative query before declaring the rebuild complete.

Use `--resume` only when the embedding configuration and Qdrant collection schema are unchanged.

---

## 10. Operational commands

Use the facade documented in [`PACKAGE_INTERFACE.md`](PACKAGE_INTERFACE.md) for structured OpenClaw operation. Use CLI commands only for debugging or maintenance.

Health and routes:

```bash
.venv/bin/citeweave health --json
.venv/bin/citeweave routes --json
```

Upload one PDF:

```bash
.venv/bin/citeweave upload ./papers/example.pdf
```

Diagnose one PDF:

```bash
.venv/bin/citeweave diagnose ./papers/example.pdf
```

Batch upload:

```bash
.venv/bin/citeweave batch-upload ./papers --resume
```

Progress:

```bash
.venv/bin/citeweave progress ./papers
```

Research query:

```bash
.venv/bin/citeweave query "Which papers discuss platform competition?"
```

Query telemetry:

```bash
.venv/bin/citeweave query-history --limit 20
.venv/bin/citeweave query-history --status error --min-duration-ms 2000
```

---

## 11. Troubleshooting

### OpenClaw gateway is not reachable

```bash
openclaw gateway status
```

Repair the local OpenClaw gateway, then rerun:

```bash
bash scripts/bootstrap_openclaw.sh
bash scripts/deployment_check.sh
```

### Docker is not available

```bash
docker info
docker-compose ps
```

Then:

```bash
docker-compose up -d
bash scripts/deployment_check.sh
```

### Neo4j authentication fails after changing `.env`

Neo4j stores the initial password in its Docker volume. For a disposable local deployment:

```bash
docker-compose down
# WARNING: deletes local CiteWeave Neo4j graph data
docker volume rm citeweave_neo4j_data
docker-compose up -d
bash scripts/deployment_check.sh
```

If graph data matters, back it up or migrate it first.

### Qdrant fails after switching embedding providers or models

Existing Qdrant collections keep their original vector size and embedding space. Switching the base embedding provider, model, or dimension requires a full vector rebuild and full corpus re-ingest. For a disposable local index:

```bash
docker-compose down
# WARNING: deletes local CiteWeave Qdrant vector data
docker volume rm citeweave_qdrant_data
docker-compose up -d
bash scripts/deployment_check.sh
```

Then clear batch progress and re-ingest **all** PDFs from scratch. Do not continue with `--resume` from the old embedding run.

### `citeweave` command is missing

```bash
. .venv/bin/activate
python -m pip install -e . --no-deps
.venv/bin/citeweave --help
```

---

## 12. Success criteria

Before telling the user CiteWeave is ready, confirm:

- `scripts/deployment_check.sh` passes or only reports explicitly acceptable environment limitations;
- Zotero source dry-run discovers expected PDFs;
- recurring sync has been scheduled or the user has declined scheduling;
- `health()` and `routes()` return usable results;
- no destructive reset was run without explicit user confirmation.
