> **This project is licensed under the Apache License 2.0. See the LICENSE file for details.**

# CiteWeave - Advanced Citation Analysis System

This project provides a comprehensive pipeline for extracting, analyzing, and structuring citations and their relationships from academic PDF documents. It features dual-layer citation networks stored in Neo4j graph database and multi-level vector embeddings for semantic search and citation analysis.

## 🔄 **NEW PARALLEL STRUCTURE ARCHITECTURE (2025)**

### **Latest Updates**
- **🚀 Unified Data Structure**: Complete redesign with `sections[]`, `paragraphs[]`, `sentences[]` as parallel arrays
- **🎯 Unified Citation Format**: Consistent citation structure across all content levels  
- **⚡ Performance Optimized**: No nesting for faster querying and better data access
- **📊 Enhanced Statistics**: Comprehensive citation tracking at all levels
- **🛠️ Cleaner Codebase**: Removed 200+ lines of obsolete code while maintaining full functionality

## 🚀 Optional: MinerU Integration for High-Quality PDF Processing

**CiteWeave optionally supports [MinerU](https://github.com/opendatalab/MinerU) for enhanced PDF parsing!**

MinerU provides superior PDF-to-Markdown conversion with:
- **95% accuracy** vs 85% traditional methods  
- **Automatic table/formula processing**
- **Smart header/footer detection**
- **Academic document optimization**

### Enable MinerU (Optional)

**Step 1: Install MinerU**
```bash
pip install magic-pdf[full]
```

**Step 2: Enable in Config**
Edit `config/model_config.json`:
```json
{
  "pdf_processing": {
    "enable_mineru": true,
    "mineru_fallback": true
  }
}
```

**Note**: MinerU is **disabled by default** due to high computational requirements. The system uses traditional PDF processors (PyMuPDF, pdfplumber) by default, which work well for most documents.

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