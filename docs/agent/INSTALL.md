# CiteWeave AI Installation Protocol

This document is written for the AI agent that is installing CiteWeave.

The agent must ask the user a small set of fixed questions, apply the answers to local configuration, start the local services, validate the installation, and set up resumable synchronization. The agent translates user choices into `.env` updates and shell commands.

All user-facing prompts and installation status messages must be in English.

## Operating Rules

- Treat the selected agent as the **Research Agent**. It owns installation orchestration, future literature queries, and sync scheduling.
- Store installation decisions in `config/install_session.local.json`. This file is local-only and ignored by git.
- Store runtime settings in `.env`. Use `scripts/apply_install_choices.py` whenever possible. Edit `.env` directly only as a fallback.
- Never delete Docker volumes, Qdrant collections, or batch progress unless the user explicitly confirms a rebuild.
- If the embedding provider, model, or dimensions change, stop and warn that a full vector rebuild and full re-ingest are required.
- Recurring sync must be resumable. It must check for active ingestion before starting and must not use `--force-restart` or `--clear-progress`.

## Step 1: Detect And Set Research Agent

Detect the current agent when possible. Known values are `Codex`, `OpenClaw`, `Claude Code`, and `Unknown`.

Ask:

```text
Detected agent: {agent}. Use this as your Research Agent?
```

Options:

```text
Yes
No
```

If the user chooses `Yes`, apply:

```bash
.venv/bin/python scripts/apply_install_choices.py --research-agent "{agent}"
```

If the user chooses `No`, ask:

```text
Choose your Research Agent.
```

Options:

```text
OpenClaw
Codex
Claude Code
Others (experimental; provide an agent command or path)
```

Apply:

```bash
.venv/bin/python scripts/apply_install_choices.py --research-agent "Codex"
```

Use the selected Research Agent for future query orchestration and scheduling. Do not ask a separate "Query LLM Mode" question.

## Step 2: Choose Reference Manager Or Literature Source

Ask:

```text
Choose your reference manager or literature source.
```

Options:

```text
Zotero / Mendeley / EndNote
PDF Folder
Single PDF Test
```

Then ask:

```text
Choose source location.
```

Options:

```text
Default Location
Custom Location
PDF Folder
```

Use this helper text:

```text
CiteWeave will recursively scan the selected location and index every PDF file found in its subfolders.
```

Apply a Zotero default or custom source:

```bash
.venv/bin/python scripts/apply_install_choices.py \
  --reference-manager zotero \
  --source-location-mode default \
  --source-dir "/absolute/path/to/Zotero"
```

Apply a generic PDF folder:

```bash
.venv/bin/python scripts/apply_install_choices.py \
  --reference-manager pdf_folder \
  --source-location-mode pdf_folder \
  --source-dir "/absolute/path/to/pdfs"
```

Validate source discovery before ingestion:

```bash
.venv/bin/python scripts/sync_literature_pdfs.py \
  --source "$CITEWEAVE_LITERATURE_SOURCE_DIR" \
  --reference-manager "$CITEWEAVE_REFERENCE_MANAGER" \
  --dry-run \
  --json
```

## Step 3: Single PDF Test

If the user chooses `Single PDF Test`, ask:

```text
Choose one PDF for a smoke test before indexing a full library.
```

Apply:

```bash
.venv/bin/python scripts/apply_install_choices.py --single-pdf "/absolute/path/to/paper.pdf"
```

After services are running, execute:

```bash
.venv/bin/citeweave upload "/absolute/path/to/paper.pdf"
```

Validation requirements:

- the command returns a paper id;
- `data/papers/<paper_id>/processed_document.json` exists;
- Qdrant contains vectors for the paper;
- Neo4j contains a `Paper` node for the paper.

Single PDF Test disables recurring sync until the user chooses a full library source.

## Step 4: Choose Embedding Model

Ask:

```text
Choose embedding mode for vector-level literature retrieval.
```

Options:

```text
Local
API
```

For the recommended local BGE profile, apply:

```bash
.venv/bin/python scripts/apply_install_choices.py \
  --embedding-mode local \
  --embedding-profile bge_large_en
```

For the fast local MiniLM profile, apply:

```bash
.venv/bin/python scripts/apply_install_choices.py \
  --embedding-mode local \
  --embedding-profile mini_l6_compat
```

For another local model, collect the model name, dimensions, device, and whether `trust_remote_code` is required. Then apply:

```bash
.venv/bin/python scripts/apply_install_choices.py \
  --embedding-mode local \
  --embedding-profile other \
  --embedding-model "model/name" \
  --embedding-dimensions 1024 \
  --embedding-device cpu
```

For API embeddings, ask in this order:

```text
Choose API Provider.
Enter API Key.
Choose embedding model.
```

OpenAI small:

```bash
.venv/bin/python scripts/apply_install_choices.py \
  --embedding-mode api \
  --api-provider openai \
  --embedding-model text-embedding-3-small \
  --api-key "$OPENAI_API_KEY"
```

OpenAI large:

```bash
.venv/bin/python scripts/apply_install_choices.py \
  --embedding-mode api \
  --api-provider openai \
  --embedding-model text-embedding-3-large \
  --api-key "$OPENAI_API_KEY"
```

Other OpenAI-compatible APIs are experimental. Collect `Base URL`, `Model Name`, `Vector Dimensions`, and `API Key`, then run a smoke test before continuing.

## Step 5: Choose Synchronization Strategy

Ask:

```text
Choose how often CiteWeave should update the literature index.
```

Options:

```text
Every 5 minutes
Every half hour
Daily
Others (enter a custom schedule)
```

Apply:

```bash
.venv/bin/python scripts/apply_install_choices.py \
  --sync-schedule every_5_minutes \
  --processors 10 \
  --skip-failed
```

Supported schedule ids:

- `every_5_minutes`
- `every_30_minutes`
- `daily`
- `custom`

All scheduler adapters must obey this rule:

```text
Check for active ingestion first. Never start a second ingest. Never use --force-restart or --clear-progress unless the user explicitly confirms a rebuild.
```

## Step 6: Install And Validate

Show this status message before starting services:

```text
Starting local services. This may take a few minutes.
```

Run:

```bash
bash scripts/bootstrap_local.sh
bash scripts/deployment_check.sh
```

For a full library source, run a dry-run:

```bash
.venv/bin/python scripts/sync_literature_pdfs.py \
  --source "$CITEWEAVE_LITERATURE_SOURCE_DIR" \
  --reference-manager "$CITEWEAVE_REFERENCE_MANAGER" \
  --dry-run \
  --json
```

Then start resumable ingestion:

```bash
.venv/bin/python scripts/sync_literature_pdfs.py \
  --source "$CITEWEAVE_LITERATURE_SOURCE_DIR" \
  --reference-manager "$CITEWEAVE_REFERENCE_MANAGER" \
  --json \
  --processors 10 \
  --skip-failed
```

Use the Research Agent's scheduler adapter for recurring sync:

- `OpenClaw`: OpenClaw automation or gateway-managed task;
- `Codex`: Codex heartbeat automation;
- `Claude Code`: macOS `launchd` fallback unless a native scheduler is available;
- `Others`: best-effort command-capable scheduler, experimental.

Finish with:

```text
Installation complete.
```

Include the resolved source path, embedding model, sync schedule, and progress command in the final report.
