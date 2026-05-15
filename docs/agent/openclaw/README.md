# OpenClaw Runtime Notes

This directory documents the OpenClaw runtime path for CiteWeave.

For the shared Research Agent installation and operating flow, start with:

- [`../README.md`](../README.md)
- [`../INSTALL.md`](../INSTALL.md)
- [`../OPERATING_CONTRACT.md`](../OPERATING_CONTRACT.md)

Use this directory for OpenClaw gateway, automation, and facade details.

## Read Next

1. [`DEPLOYMENT.md`](DEPLOYMENT.md) — OpenClaw runtime deployment notes.
2. [`PACKAGE_INTERFACE.md`](PACKAGE_INTERFACE.md) — OpenClaw facade contract.

## Runtime Boundary

OpenClaw handles the conversation and OpenClaw-specific runtime concerns. CiteWeave
handles the research system.

OpenClaw-specific responsibilities:

- check the local OpenClaw gateway when `CITEWEAVE_LLM_PROVIDER=openclaw`;
- call `OpenClawCiteWeaveFacade` when a structured Python facade is preferred;
- map OpenClaw automation or plugin behavior onto the generic agent operating contract.

CiteWeave responsibilities remain generic:

- literature-source ingestion;
- PDF extraction and citation parsing;
- Neo4j graph construction;
- Qdrant vector indexing;
- research query routing and synthesis.

The boundary is:

> OpenClaw chooses the operation. CiteWeave chooses the evidence route.

For normal research questions, pass the user's full question to CiteWeave:

```python
facade.query(question, confirmation="continue")
```

Do not manually query Neo4j or Qdrant from OpenClaw unless the user explicitly
asks for diagnostics or low-level maintenance.
