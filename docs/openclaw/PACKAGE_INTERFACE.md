# CiteWeave OpenClaw Package Interface

This document is the operational contract OpenClaw should use when controlling CiteWeave.
For deployment and Zotero setup, read [`DEPLOYMENT.md`](DEPLOYMENT.md) first.

OpenClaw is the user entrypoint, query interface, and deployment coordinator. CiteWeave owns the local citation intelligence stack: Zotero/PDF ingestion, PDF processing, Neo4j, Qdrant, embeddings, GROBID, and the research query kernel.

---

## 1. Data source contract

### Primary data source: Zotero library

At setup time, ask the user for the path to their Zotero data directory.

Accept either:

- the Zotero data directory containing `zotero.sqlite` and `storage/`; or
- the Zotero `storage/` directory itself; or
- an exported folder of PDFs.

Persist it in `.env`:

```env
CITEWEAVE_ZOTERO_LIBRARY_DIR=/path/to/Zotero
```

On many machines the directory is one of:

```text
~/Zotero
~/Documents/Zotero
~/Library/Application Support/Zotero/Profiles/<profile>/zotero
```

CiteWeave currently ingests PDFs from Zotero storage recursively. Zotero SQLite metadata can be used by future enrichments, but the current ingestion contract is PDF-first: discover PDF attachments, process them, and build CiteWeave's own graph/vector/index state.

### Continuous ingestion script

Use:

```bash
.venv/bin/python scripts/sync_zotero_pdfs.py --json
```

or explicitly:

```bash
.venv/bin/python scripts/sync_zotero_pdfs.py --source /path/to/Zotero --json
```

Dry run before scheduling:

```bash
.venv/bin/python scripts/sync_zotero_pdfs.py --source /path/to/Zotero --dry-run --json
```

Behavior:

1. resolve the Zotero source to the PDF-bearing directory;
2. recursively discover `*.pdf` files;
3. call CiteWeave's resumable batch uploader;
4. skip already-completed files unless explicitly forced;
5. record progress in `data/batch_upload_tracker.json`.

Default command delegated by the script:

```bash
.venv/bin/citeweave batch-upload <resolved-pdf-source> --resume
```

### Scheduling rule for OpenClaw

OpenClaw should schedule `scripts/sync_zotero_pdfs.py` as a recurring automation job after the user confirms the Zotero path.

Recommended cadence:

- active literature project: every 1-3 hours;
- normal usage: once per day;
- very large library: nightly, plus manual runs after big Zotero imports.

The sync command is idempotent because it uses resumable batch progress.

Do not run destructive reset flags on a schedule:

- avoid `--force-restart` unless the user explicitly asks to rebuild;
- avoid `--clear-progress` unless debugging or intentionally reprocessing.

---

## 2. OpenClaw action surface

OpenClaw should call the structured facade instead of scraping CLI output.

Python facade:

```python
from src.adapters.openclaw_facade import OpenClawCiteWeaveFacade

facade = OpenClawCiteWeaveFacade()
```

### Available methods

| User intent | Facade method | When to use |
|---|---|---|
| Configure / inspect deployment | `bootstrap_plan()` | User asks how to set up CiteWeave or what to run next. |
| Deploy local infrastructure | `scripts/deploy_local_stack.sh` | User asks to start or repair Neo4j, Qdrant, and GROBID. |
| Health check | `health()` | User asks whether CiteWeave is working, or before first upload/query. |
| Route diagnostics | `routes()` | User asks which query routes/databases are available. |
| Upload one PDF | `upload_pdf(pdf_path)` | User provides one PDF path. |
| Diagnose one PDF | `diagnose_pdf(pdf_path)` | User asks why a PDF may fail or whether it is safe to ingest. |
| Batch upload | `batch_upload(directory, resume=True)` | User provides a folder of PDFs, Zotero storage, or asks to ingest a library. |
| Continuous Zotero sync | `scripts/sync_zotero_pdfs.py` | User has configured a Zotero data source and wants scheduled ingestion. |
| Batch progress | `progress(directory)` | User asks what remains, what failed, or whether ingestion is complete. |
| Research query | `query(question, confirmation="continue")` | User asks a literature, author, citation, argument, semantic, or compound research question. |
| Interactive compatibility | `chat_turn(...)` | Legacy/manual chat flow. Do not prefer this for OpenClaw package operation. |
| Query telemetry | `query_history(...)` | User asks about previous queries, failed queries, slow queries, or route usage. |
| Missing source documents | `list_pending_citations(limit=10)` | User asks which cited papers still need PDFs uploaded. |

---

## 3. Query routing logic

OpenClaw should decide the **operation**. CiteWeave should decide the **research retrieval plan**.

That means:

- If the user asks to upload, sync, diagnose, inspect, or check progress: call the matching operational method.
- If the user asks a research question: call `query(question)` and let CiteWeave choose graph/vector/PDF/author routes internally.
- Do not make OpenClaw manually query Neo4j or Qdrant unless a future low-level diagnostic API explicitly exposes that.

### Intent-to-interface rules

| User wording | Interface |
|---|---|
| “Use my Zotero library”, “set data source”, “keep this synced” | Set `CITEWEAVE_ZOTERO_LIBRARY_DIR`; dry-run `scripts/sync_zotero_pdfs.py`; schedule it. |
| “Upload this paper”, “ingest this PDF” | `upload_pdf(pdf_path)` |
| “Upload my papers folder”, “sync Zotero”, “import my library” | `batch_upload(directory)` or scheduled `scripts/sync_zotero_pdfs.py` |
| “Is this PDF readable?”, “why did extraction fail?” | `diagnose_pdf(pdf_path)` |
| “What has finished?”, “what failed?”, “continue upload” | `progress(directory)` then `batch_upload(directory, resume=True)` |
| “Which papers discuss X?”, “what does author Y argue?”, “trace citations from A to B” | `query(question)` |
| “Which interface should be used?”, “what routes exist?” | `routes()` |
| “Is CiteWeave healthy?” | `health()` |
| “What queries failed recently?” | `query_history(status="error")` |
| “Which cited papers are missing PDFs?” | `list_pending_citations()` |

### Research query categories

OpenClaw should pass the user's research question intact to `query(question)` for these categories:

| Query category | Examples | CiteWeave internals likely involved |
|---|---|---|
| Semantic concept query | “Which papers discuss dynamic capabilities?” | Qdrant vector route + PDF content |
| Literature query | “Summarize the literature on platform competition.” | vector + graph + PDF synthesis |
| Author query | “What does Porter argue about competitive advantage?” | author index + graph + PDF content |
| Citation query | “Which papers cite Teece 1997 and why?” | Neo4j graph + citation contexts |
| Paper-specific query | “What is the main argument of this paper?” | PDF content + metadata |
| Comparative query | “Compare Porter and Teece on strategy.” | author/paper lookup + graph/vector synthesis |
| Compound query | “Find papers on X, group by author, and show citation evidence.” | multiple internal routes, sufficiency assessment, response generation |

OpenClaw should not pre-split compound research questions into several low-level DB calls by default. The safer first move is:

```python
facade.query(full_user_question, confirmation="continue")
```

If the answer is insufficient, OpenClaw may follow up with narrower `query(...)` calls, for example:

1. broad semantic query;
2. author-focused query;
3. citation-evidence query;
4. final synthesis query using the user's requested framing.

---

## 4. Suggested OpenClaw operating loop

### First setup

1. Run `bootstrap_plan()` or `bash scripts/bootstrap_openclaw.sh`.
2. If only the infrastructure layer needs repair, run `bash scripts/deploy_local_stack.sh`.
3. Ask the user for the Zotero data directory.
4. Save it as `CITEWEAVE_ZOTERO_LIBRARY_DIR` in `.env`.
5. Run:

   ```bash
   .venv/bin/python scripts/sync_zotero_pdfs.py --dry-run --json
   ```

6. If PDFs are found, schedule recurring sync.
7. Run `health()` and `routes()`.

### Recurring sync

1. Run `scripts/sync_zotero_pdfs.py --json`.
2. If the command exits non-zero, run `health()` and report the blocker.
3. If failures are recorded, call `progress(...)` for the source directory.
4. Do not reprocess completed PDFs unless the user asks for a rebuild.

### Query handling

1. If the query is operational, call the matching facade method.
2. If the query is research-oriented, call `query(question)`.
3. Inspect the answer for missing-source hints or unresolved citations.
4. If needed, call `list_pending_citations()` and ask the user to add/upload missing PDFs.
5. For repeated issues, inspect `query_history(status="error")`.

---

## 5. Output expectations

OpenClaw should summarize results for the user, not dump raw JSON unless asked.

Good summaries include:

- what operation ran;
- how many PDFs were discovered/processed;
- failed files and next action;
- whether services are healthy;
- which evidence sources were used in research answers;
- missing PDFs or unresolved citations that need user action.

Use raw JSON only for debugging, tests, or explicit user requests.
