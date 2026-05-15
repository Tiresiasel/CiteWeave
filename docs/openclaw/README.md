# Operating CiteWeave through OpenClaw

CiteWeave is most useful when it behaves like a local research system that quietly stays in sync with a scholar's library. OpenClaw provides that operating layer: it can deploy the local services, remember the Zotero source, schedule ingestion, monitor progress, and route user requests to the right CiteWeave action.

This directory documents that operating contract.

## Read next

1. [`DEPLOYMENT.md`](DEPLOYMENT.md) — local setup, Zotero source configuration, recurring sync, verification, and troubleshooting.
2. [`PACKAGE_INTERFACE.md`](PACKAGE_INTERFACE.md) — facade methods, intent routing, query behavior, and output expectations.

## Operating model

OpenClaw handles the conversation and the operational choreography. CiteWeave handles the research system.

OpenClaw is responsible for:

- interpreting whether the user wants setup, upload, sync, diagnosis, progress, health, telemetry, or a research answer;
- starting and checking the local Docker Compose services;
- storing the Zotero source path in `.env`;
- scheduling safe recurring ingestion;
- calling the CiteWeave facade instead of reaching directly into databases.

CiteWeave is responsible for:

- Zotero/PDF ingestion;
- PDF extraction and citation parsing;
- Neo4j graph storage;
- Qdrant vector indexing;
- embedding generation;
- research query planning and synthesis.

The boundary is deliberately simple:

> OpenClaw chooses the operation. CiteWeave chooses the evidence route.

## First run

1. Confirm the OpenClaw gateway is available.
2. Bootstrap CiteWeave.
3. Start the local service layer: Neo4j, Qdrant, and GROBID.
4. Configure the Zotero data directory.
5. Dry-run PDF discovery.
6. Schedule recurring sync only after the dry run looks right.
7. Run health and route checks.
8. Use the facade methods in [`PACKAGE_INTERFACE.md`](PACKAGE_INTERFACE.md).

## Non-negotiable rebuild rule

If the base embedding model, embedding provider, or vector dimension changes, the existing Qdrant vector space is no longer valid. Stop ingestion, recreate or migrate vector collections, clear batch progress, and re-ingest the complete corpus. Do not resume from old progress after an embedding change.

This is not ceremony. It is the difference between a semantic index and a drawer full of numerically plausible garbage.

## Research questions

For normal research questions, pass the user's full question to CiteWeave:

```python
facade.query(question, confirmation="continue")
```

Do not manually query Neo4j or Qdrant from OpenClaw unless the user is explicitly asking for diagnostics or a future low-level maintenance API requires it.
