# CiteWeave Agent Template

This directory is the main documentation for installing and operating CiteWeave
through shell-capable Research Agents.

CiteWeave supports Codex, OpenClaw, Claude Code, and custom local agents through
the same installation protocol, service boundary, and query operating contract.

## Read Next

1. [`INSTALL.md`](INSTALL.md) — step-by-step installation protocol for the agent.
2. [`install_manifest.yaml`](install_manifest.yaml) — machine-readable choices, prompts, config patches, commands, and validation checks.
3. [`DEPLOYMENT.md`](DEPLOYMENT.md) — generic local deployment guide for Docker services, Python environment, source discovery, ingestion, and health checks.
4. [`OPERATING_CONTRACT.md`](OPERATING_CONTRACT.md) — how an agent should choose CiteWeave operations and pass research questions to the kernel.

Runtime-specific notes live under this directory. OpenClaw gateway and facade
notes are in [`openclaw/`](openclaw/).

## Operating Model

The Research Agent owns the user-facing workflow:

- ask setup questions;
- apply local configuration;
- start and check local services;
- schedule safe recurring ingestion;
- call CiteWeave operations;
- summarize results for the user.

CiteWeave owns the research substrate:

- PDF parsing and document structure extraction;
- citation parsing and citation-context construction;
- Neo4j graph writes and traversal;
- Qdrant vector indexing and semantic retrieval;
- query route planning and answer synthesis.

The boundary is deliberately simple:

> The agent decides what the user is trying to do. CiteWeave decides how local research evidence should be retrieved.
