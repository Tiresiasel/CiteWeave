# CiteWeave

**Argument-level citation graph + semantic RAG for academic papers.**

CiteWeave extracts sentence-level citation relationships from PDFs, builds a citation graph, and lets you query your paper library through a multi-agent research system. Designed for social-science researchers who need to trace the flow of arguments across literature — but useful anywhere citation-level precision matters.

[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## TL;DR — Local CLI first

```bash
# 1. Clone & configure
git clone https://github.com/Tiresiasel/CiteWeave.git
cd CiteWeave

# 2. Bootstrap the local CLI deployment
bash scripts/bootstrap_local.sh

# 3. Use the CLI
.venv/bin/python -m src.core.cli upload path/to/paper.pdf
.venv/bin/python -m src.core.cli query "Which papers discuss X?"
.venv/bin/python -m src.core.cli routes
.venv/bin/python -m src.core.cli chat

# 4. Or switch to OpenClaw mode
bash scripts/bootstrap_openclaw.sh
# then verify:
.venv/bin/python -m src.core.cli routes
bash scripts/deployment_check.sh
# and call CiteWeave from your OpenClaw session.
```

---

## Part 1 — Use CiteWeave as a CLI application

This is the primary deployment path. Even if you later integrate CiteWeave with
OpenClaw, the base system is still a local CLI application backed by Docker
services.

### What works on the current codebase

Current CLI commands available on this branch:

- `upload`
- `diagnose`
- `batch-upload`
- `progress`
- `chat`
- `query`
- `routes`
- `health`
- `bootstrap-plan`

### Local CLI mode (no OpenClaw needed)

Edit `.env`:

```bash
CITEWEAVE_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...yourkey...
```

Then run CiteWeave through the project virtualenv:

```bash
.venv/bin/python -m src.core.cli upload path/to/paper.pdf
.venv/bin/python -m src.core.cli diagnose path/to/paper.pdf
.venv/bin/python -m src.core.cli batch-upload ./papers --resume
.venv/bin/python -m src.core.cli query "Which papers discuss X?"
.venv/bin/python -m src.core.cli routes
.venv/bin/python -m src.core.cli health
.venv/bin/python -m src.core.cli bootstrap-plan
.venv/bin/python -m src.core.cli chat
```

> Why `.venv/bin/python` instead of plain `python`?
> Because the CLI imports project dependencies at startup. In a clean machine,
> `python -m src.core.cli` will fail unless you installed the requirements in
> the active environment.

---

## Backing services

CiteWeave depends on three backing services. Start them all at once:

```bash
docker-compose up -d
```

| Service | Port | Purpose |
|---------|------|---------|
| **Neo4j** | 7474 / 7687 | Citation graph storage (Bolt + HTTP) |
| **Qdrant** | 6333 / 6334 | Semantic vector index (REST + gRPC) |
| **GROBID** | 8070 | PDF structure extraction (name/author/section parsing) |

Verify everything is healthy:

```bash
bash scripts/deployment_check.sh
```

---

## CLI reference

All interaction with CiteWeave goes through the project environment:

```
.venv/bin/python -m src.core.cli <command> [options]
```

### `upload <pdf_path>`

Parse a PDF and ingest it into the citation graph.

```bash
.venv/bin/python -m src.core.cli upload path/to/paper.pdf
.venv/bin/python -m src.core.cli upload path/to/paper.pdf --diagnose
.venv/bin/python -m src.core.cli upload path/to/paper.pdf --force
```

### `diagnose <pdf_path>`

Run a quality diagnosis on a PDF without ingesting it.

```bash
.venv/bin/python -m src.core.cli diagnose path/to/paper.pdf
```

### `batch-upload <directory>`

Upload and process all PDFs in a directory.

```bash
.venv/bin/python -m src.core.cli batch-upload path/to/papers/
.venv/bin/python -m src.core.cli batch-upload path/to/papers/ --resume
.venv/bin/python -m src.core.cli batch-upload path/to/papers/ --sequential
```

### `progress <directory>`

Inspect or clear batch-upload progress tracking. The report distinguishes between retryable failed files and PDFs that have not been attempted yet, so resume decisions are less guesswork and more engineering.

```bash
.venv/bin/python -m src.core.cli progress path/to/papers/
.venv/bin/python -m src.core.cli progress path/to/papers/ --clear
```

The text output now breaks remaining work into:
- **Retryable failed files**: attempted previously, failed, and will be retried by `batch-upload --resume`
- **Not started yet**: discovered PDFs with no tracker entry yet
- **Pending files**: the union of both groups above

### `chat`

Interactive multi-turn chat with the research system.

```bash
.venv/bin/python -m src.core.cli chat
```

### `query "<question>"`

Single-shot query path into the LangGraph research workflow.
Use `--confirmation` when you want to control how the workflow proceeds after
an information summary step.

```bash
.venv/bin/python -m src.core.cli query "Which papers discuss bandwidth vs. pricing?"
.venv/bin/python -m src.core.cli query "Summarize Michael Porter 1980" --confirmation continue
```

### `routes`

Inspect the active route configuration, including aliases, priority mappings,
and addon/env overrides.

```bash
.venv/bin/python -m src.core.cli routes
```

### `health`

Inspect environment and service health. The human-readable output now leads
with an overall verdict and recommended next actions; use `--json` when you
want the raw machine-readable snapshot.

```bash
.venv/bin/python -m src.core.cli health
.venv/bin/python -m src.core.cli health --json
```

### `bootstrap-plan`

Print the recommended local CLI and OpenClaw bootstrap steps without scraping
other docs.

```bash
.venv/bin/python -m src.core.cli bootstrap-plan
.venv/bin/python -m src.core.cli bootstrap-plan --json
```

---

## Part 2 — Integrate CiteWeave with OpenClaw

For the architectural contract behind this split, see:
`docs/KERNEL_AND_OPENCLAW.md`

### What changes in OpenClaw mode

OpenClaw does not replace CiteWeave's storage or parsing stack. It only becomes
CiteWeave's LLM backend.

The underlying deployment is still the same:

- Neo4j stores the citation graph
- Qdrant stores semantic vectors
- GROBID parses PDFs
- CiteWeave CLI / Python code handles ingestion and chat logic

What changes is that CiteWeave sends LLM calls to the **local OpenClaw
gateway** instead of directly calling OpenAI.

### Architecture

```
OpenClaw Agent (Atlas / any agent)
    │
    │  CITEWEAVE_LLM_PROVIDER=openclaw
    │  All LLM calls → http://localhost:18789/v1
    │
    ├──→ .venv/bin/python -m src.core.cli chat
    │       │
    │       └── Neo4j + Qdrant + GROBID
    │
    └──→ (optional) direct Python import
            LangGraphResearchSystem()
```

### Concrete setup flow

1. First finish the **local CLI deployment** above, or simply run:

```bash
bash scripts/bootstrap_openclaw.sh
```

That script prepares `.env`, ensures the virtualenv exists, installs
requirements, starts Docker services, and keeps the project in OpenClaw mode.

2. If you prefer to set values manually, `.env` should contain:

```bash
CITEWEAVE_LLM_PROVIDER=openclaw
CITEWEAVE_LLM_MODEL=openai-codex/gpt-5.4
CITEWEAVE_LLM_API_BASE=http://localhost:18789/v1
CITEWEAVE_NEO4J_PASSWORD=0xC1735
```

3. Ensure the local OpenClaw gateway is running:

```bash
openclaw gateway status
```

4. Re-run the deployment check:

```bash
bash scripts/deployment_check.sh
```

You should see the gateway connectivity check succeed.

5. Then call CiteWeave from an OpenClaw session. Example:

```text
Atlas, upload these PDFs with CiteWeave and then use chat mode to help me inspect the citation graph.
```

### Important security / behavior note

In `openclaw` mode, CiteWeave does **not** forward your real OpenAI API key to
the gateway. The code replaces it with a harmless placeholder, and OpenClaw
handles authentication through its own local session / gateway flow.

### For autonomous OpenClaw jobs

The `citeweave:daily-iteration-and-push` cron workflow uses the same
OpenClaw-backed mode, so automated iteration does not require a separate
OpenAI key either.

---

## Configuration

### Files

| File | Purpose |
|------|---------|
| `.env` | Runtime configuration (copy from `.env_template`) |
| `config/model_config.json` | Per-agent model, temperature, max_tokens |
| `config/neo4j_config.json` | Neo4j connection (uri, username, password) |
| `config/default_config.yaml` | Default values; sets Neo4j password to `0xC1735` |

### Environment variables (take precedence over JSON config)

| Variable | Default | Description |
|----------|---------|-------------|
| `CITEWEAVE_LLM_PROVIDER` | `openai` | `openclaw` · `openai` · `ollama` |
| `CITEWEAVE_LLM_MODEL` | — | Model name (e.g. `gpt-4o-mini`, `openai-codex/gpt-5.4`) |
| `CITEWEAVE_LLM_API_BASE` | — | API base URL for openclaw/ollama modes |
| `CITEWEAVE_LLM_API_KEY` | — | API key (or any placeholder for openclaw) |
| `CITEWEAVE_NEO4J_PASSWORD` | `0xC1735` | Neo4j password |
| `CITEWEAVE_ENV` | `production` | `production` · `development` (verbose logging) |

### Neo4j password

The default password `0xC1735` is intentionally memorable for local
development. **Change it before any real deployment:**

```bash
# Option 1: set in .env
CITEWEAVE_NEO4J_PASSWORD=your-secure-password

# Option 2: environment variable (takes precedence)
export CITEWEAVE_NEO4J_PASSWORD=your-secure-password
```

---

## Development

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt          # sentence splitting
```

### Run tests

```bash
# All tests
python -m unittest discover -s tests

# Specific test file
python -m unittest discover -s tests -p 'test_routing.py'
```

### Privacy audit (blocking before any commit)

```bash
python3 scripts/repo_privacy_audit.py
```

Any privacy audit failure blocks commit. This checks for:
- Absolute local machine paths (for example, user home directories or private workspace roots)
- Token / secret values in tracked files
- Runtime data tracked in `data/` or `test_files/`

---

## Architecture

```
User question
    │
    ▼
Question Analysis Agent    ──→ Ambiguity? ──→ User Clarification Agent
    │
    ▼
Query Planning Agent
    │
    ├──→ Graph DB Agent  (Neo4j — citation edges, argument nodes)
    ├──→ Vector DB Agent (Qdrant — semantic similarity search)
    └──→ PDF Content Agent (GROBID — full text extraction)

    ◄─────────── Reflection Agent (sufficiency check) ───────────◄
    │                      │                                      │
    │               Still insufficient?                          │
    │                      │ yes                                  │ no
    └──── Additional Query Generation ──────┘                      │
                                                                ▼
                                                    Response Generation Agent
                                                                │
                                                                ▼
                                                        Structured Answer
```

### Argument claim types

CiteWeave classifies every sentence in a paper:

| Type | Description |
|------|-------------|
| `CLAIM_MAIN` | Primary thesis / main claim |
| `CLAIM_SUPPORTING` | Secondary supporting claim |
| `EVIDENCE_EMPIRICAL` | Empirical data / results |
| `EVIDENCE_THEORETICAL` | Theoretical support |
| `EVIDENCE_LITERATURE` | Citation-based support |
| `COUNTERARGUMENT` | Counter-argument / assumption |
| `METHODOLOGY` | Method description |
| `REBUTTAL` | Explicit rebuttal |
| `QUESTION_MOTIVATION` | Research question / motivation |
| `FUTURE_WORK` | Future directions |
| `NON_ARGUMENT` | Neutral / transitional text |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
