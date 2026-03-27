# CiteWeave

**Argument-level citation graph + semantic RAG for academic papers.**

CiteWeave extracts sentence-level citation relationships from PDFs, builds a citation graph, and lets you query your paper library through a multi-agent research system. Designed for social-science researchers who need to trace the flow of arguments across literature — but useful anywhere citation-level precision matters.

[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## TL;DR — Up in 5 minutes

```bash
# 1. Clone & configure
git clone https://github.com/Tiresiasel/CiteWeave.git
cd CiteWeave
cp .env_template .env          # edit .env — see "Choose your LLM mode" below

# 2. Start services
docker-compose up -d

# 3. Verify deployment
bash scripts/deployment_check.sh

# 4. Upload papers and query
python -m src.core.cli upload path/to/paper.pdf
python -m src.core.cli query "Which papers discuss X?"
```

---

## Two ways to run

### Mode A — Local CLI (no OpenClaw needed)

```bash
# Set LLM provider to OpenAI in .env
CITEWEAVE_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...yourkey...

# All features work via CLI; OpenAI handles all LLM calls
python -m src.core.cli upload path/to/paper.pdf
python -m src.core.cli query "Who argues that X?"
python -m src.core.cli chat              # interactive multi-turn chat
```

### Mode B — OpenClaw integration (recommended for OpenClaw users)

```bash
# Set LLM provider to openclaw in .env
CITEWEAVE_LLM_PROVIDER=openclaw
# CITEWEAVE_LLM_MODEL defaults to openai-codex/gpt-5.4
# CITEWEAVE_LLM_API_BASE defaults to http://localhost:18789/v1

# CiteWeave routes all LLM calls through your local OpenClaw gateway.
# No separate OpenAI key needed — OpenClaw handles auth via session.
```

When `CITEWEAVE_LLM_PROVIDER=openclaw`, every agent in the multi-agent system
(`language_processor`, `query_analyzer`, `response_generator`, …) connects to
`http://localhost:18789/v1` automatically. The gateway must be running on the
same host. See [OpenClaw Integration](#openclaw-integration) for details.

---

## Services

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

All interaction with CiteWeave goes through `python -m src.core.cli`.

```
python -m src.core.cli <command> [options]
```

### `upload <pdf_path>`

Parse a PDF and ingest it into the citation graph.

```bash
python -m src.core.cli upload path/to/paper.pdf
python -m src.core.cli upload path/to/paper.pdf --diagnose   # quality report
python -m src.core.cli upload path/to/paper.pdf --force       # re-process even if cached
```

### `query "<question>"`

Ask the citation graph a research question. The multi-agent system plans the
query, retrieves from Neo4j + Qdrant, and returns a structured answer.

```bash
python -m src.core.cli query "Which papers discuss bandwidth vs. pricing?"
```

### `chat`

Interactive multi-turn chat with the research system.

```bash
python -m src.core.cli chat
```

### `batch-upload <directory>`

Upload and process all PDFs in a directory.

```bash
python -m src.core.cli batch-upload path/to/papers/           # 4 parallel workers
python -m src.core.cli batch-upload path/to/papers/ --resume   # skip already done
python -m src.core.cli batch-upload path/to/papers/ --sequential  # one at a time
```

### `diagnose <pdf_path>`

Run a quality diagnosis on a PDF without ingesting it.

```bash
python -m src.core.cli diagnose path/to/paper.pdf
```

### `routes`

Print the active routing configuration (which addon configs and environment
variables are loaded, in priority order).

```bash
python -m src.core.cli routes
```

### `papers [--all | --limit N]`

List all papers currently in the citation database.

```bash
python -m src.core.cli papers --limit 20    # first 20 papers
python -m src.core.cli papers --all          # entire database
```

---

## OpenClaw integration

### Architecture

```
OpenClaw Agent (Atlas / any agent)
    │
    │  CITEWEAVE_LLM_PROVIDER=openclaw
    │  All LLM calls → http://localhost:18789/v1 (OpenClaw gateway)
    │
    ├──→ CLI: python -m src.core.cli query "..."
    │       │
    │       └── Neo4j + Qdrant + GROBID (Docker services)
    │
    └── (optional) Direct Python API import
            from src.agents.multi_agent_research_system import LangGraphResearchSystem
            # uses same EnhancedLLMManager with OpenClaw gateway
```

### Setup

1. Edit `.env`:

```bash
CITEWEAVE_LLM_PROVIDER=openclaw
CITEWEAVE_LLM_MODEL=openai-codex/gpt-5.4          # optional override
CITEWEAVE_LLM_API_BASE=http://localhost:18789/v1  # optional; default is this
CITEWEAVE_NEO4J_PASSWORD=0xC1735                   # change in production
```

2. Ensure the OpenClaw gateway is running:

```bash
openclaw gateway status
```

3. Verify CiteWeave detects the gateway:

```bash
bash scripts/deployment_check.sh
```

4. Start using it from any OpenClaw agent session:

```
Atlas, please upload these PDFs and then help me find papers that discuss X.
```

### For OpenClaw agents doing autonomous iteration

The `citeweave:daily-iteration-and-push` cron job uses OpenClaw mode
automatically — no OpenAI key is needed for any automated workflow.

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
- Absolute local paths (`/home/tiresias`, `.openclaw/workspace`)
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
