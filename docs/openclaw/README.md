# CiteWeave for OpenClaw

This folder is for OpenClaw and operators. The top-level `README.md` explains the project to human readers; this folder explains how OpenClaw should deploy and operate it.

Read in this order:

1. [`DEPLOYMENT.md`](DEPLOYMENT.md) — local setup, Zotero data source, recurring sync, verification, troubleshooting.
2. [`PACKAGE_INTERFACE.md`](PACKAGE_INTERFACE.md) — action surface, method selection, query-routing logic, output expectations.

## Core operating model

OpenClaw is the entrypoint, query interface, and deployment coordinator.

CiteWeave owns the local research stack:

- Zotero/PDF ingestion;
- a Docker Compose service layer for Neo4j, Qdrant, and GROBID;
- Neo4j citation graph;
- Qdrant vector index;
- GROBID PDF extraction;
- local or OpenAI embeddings;
- citation parsing and research query kernel.

Do not treat OpenClaw as the database layer. OpenClaw decides what operation the user wants; CiteWeave decides how to retrieve and synthesize research evidence.

## First-run checklist

1. Confirm OpenClaw gateway is running.
2. Bootstrap CiteWeave.
3. Deploy the local Docker Compose stack: Neo4j, Qdrant, and GROBID.
4. Ask the user for their Zotero data directory.
5. Persist that path in `.env` as `CITEWEAVE_ZOTERO_LIBRARY_DIR`.
6. Dry-run Zotero PDF discovery.
7. Schedule recurring sync.
8. Run health and route checks.
9. Use the facade methods documented in `PACKAGE_INTERFACE.md`.

## Important rule

For research questions, pass the user's full question to `query(question)` unless the user clearly asks for an operational action such as upload, sync, diagnose, health, progress, routes, or telemetry.
