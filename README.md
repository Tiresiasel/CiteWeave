> **This project is licensed under the Apache License 2.0. See the LICENSE file for details.**

# Argument Graph Project

This project provides a pipeline for extracting, classifying, and structuring argumentative claims and their relationships from PDF documents, storing them in a Neo4j graph database, and enabling advanced querying and citation resolution.

## Project Structure

- `PRODUCT_SPEC.md`: Product requirements and specifications.
- `docs/`: Documentation, including data flow and schema definitions.
- `src/`: Source code modules for processing, classification, parsing, graph building, indexing, querying, and CLI.
- `config/`: Configuration files for Neo4j, models, and paths.
- `tests/`: Unit and integration tests for core modules.

See `docs/data_flow.md` and `docs/schema/graph_schema.md` for more details.

## Grobid Deployment for PDF Metadata Extraction

This project uses [Grobid](https://grobid.readthedocs.io/) to extract structured metadata (title, authors, journal, DOI, publisher, etc.) from academic PDF files.

### Deploy Grobid with Docker

1. **Pull the Grobid Docker image:**
   ```bash
   docker pull lfoppiano/grobid:0.8.0
   ```
2. **Run Grobid as a background service:**
   ```bash
   docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0
   ```
   This will start Grobid on [http://localhost:8070](http://localhost:8070).

### Basic Usage
- The Python code can send PDF files to the Grobid API endpoint (e.g., `/api/processHeaderDocument`) to extract metadata.
- Example Python integration is available in the codebase (see `pdf_processor.py` or related modules).

### Stopping Grobid
To stop the Grobid service:
```bash
docker stop grobid
```
To remove the container:
```bash
docker rm grobid
```