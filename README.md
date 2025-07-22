# CiteWeave - Advanced Citation Analysis System

> **This project is licensed under the Apache License 2.0. See the LICENSE file for details.**

---

## 🚀 Project Overview

CiteWeave is a comprehensive pipeline for extracting, analyzing, and structuring citations and their relationships from academic PDF documents. It features:
- Multi-agent research chat system (ask research questions, get structured answers)
- Dual-layer citation networks (Neo4j graph database)
- Multi-level vector embeddings for semantic search and citation analysis
- High-quality PDF processing (with optional MinerU integration)
- All user interaction is via a powerful command-line interface (CLI)

---

## ⚡ Quick Start

### 1. Environment Setup

- **Python**: Requires Python 3.8+
- **Recommended**: Use a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

- **Install dependencies**:

```bash
pip install -r requirements.txt
```

**System dependencies:**
- [Docker](https://www.docker.com/) (required for Qdrant and GROBID services)
- [Docker Compose](https://docs.docker.com/compose/) (for service orchestration)

**Start core services (Qdrant + GROBID):**
1. Make sure Docker and Docker Compose are installed and running.
2. From your project root, start the services:
   ```bash
   python scripts/start_services.py
   # or manually:
   docker-compose up -d
   ```
3. Wait for both Qdrant (vector DB) and GROBID (PDF metadata extraction) to be ready.
   - Qdrant: http://localhost:6333
   - GROBID: http://localhost:8070

> **You must have Qdrant and GROBID running before using the CLI to upload PDFs or start a chat.**

For full details and troubleshooting, see [docs/QDRANT_SERVER_SETUP.md](docs/setup/QDRANT_SERVER_SETUP.md).

### 2. Initial Configuration

- All config files are in the `config/` directory.
- **Check/edit**:
  - `config/model_config.json` (model and PDF processing settings)
  - `config/neo4j_config.json` (graph DB connection)
  - `config/paths.json` (default data storage path)
  - `config/qdrant_config.json` (vector DB)

### 3. Configure Your AI API Key (Required for Chat/Research)

To use the chat and research features, you must provide an API key for OpenAI (ChatGPT) as your LLM/AI provider.

- **OpenAI Example:**
  - Get your API key from https://platform.openai.com/account/api-keys
  - Set it as an environment variable:
    ```bash
    export OPENAI_API_KEY=sk-...yourkey...
    ```
  - Or, add it to a `.env` file in your project root:
    ```env
    OPENAI_API_KEY=sk-...yourkey...
    ```

**Note:**
- Only OpenAI (ChatGPT) API keys are supported and tested at this time. (In the future, we will support more LLMs)
- The system will not be able to answer research questions or chat unless a valid OpenAI API key is set.
- You may need to restart your terminal or IDE after setting the environment variable.
- For more details, see the comments in `config/model_config.json` and the OpenAI documentation.

---

## 🖥️ CLI Usage (All User Interaction)

**Workflow Summary:**
1. **Upload your PDF files** (required, provides the data for all research and chat)
   - Use `upload` for a single file, or `batch-upload` for all PDFs in a directory
2. **(Optional) Diagnose PDF quality** before upload
3. **Chat with the multi-agent system** to ask research questions about your uploaded documents

All user interaction is via the CLI. Do not use Python APIs or import functions directly.

### 1. Upload and Process PDF Documents (**First Step, Required**)

Before you can chat or ask any research questions, you must upload and process your PDF files. This populates the system with the data needed for analysis and chat.

**Single file upload:**
```bash
python -m src.core.cli upload path/to/your/file.pdf
```
- Add `--diagnose` to run a quality check before processing.
- Add `--force` to force reprocessing even if cached results exist.

**Example:**
```
python -m src.core.cli upload test_files/Porter\ -\ Competitive\ Strategy.pdf --diagnose
```

**Batch upload (process all PDFs in a directory):**
```bash
python -m src.core.cli batch-upload path/to/your/pdf_folder
```
- This will recursively find and process all `.pdf` files in the specified directory and its subdirectories.
- Progress and a summary of successes/failures will be printed.


### 2. Start an Interactive Research Chat (**After Uploading Files**)

Once you have uploaded and processed your PDFs, you can start a multi-turn chat session with the multi-agent research system:

```bash
python -m src.core.cli chat
```
- Type your research questions at the `You:` prompt.
- The AI will respond after each question, using the data from your uploaded files.
- Type `exit` or `quit` to end the chat session.

**Example:**
```
$ python -m src.core.cli chat
🤖 CiteWeave Multi-Agent Research System (Chat Mode)
============================================================
Type 'exit' or 'quit' to end the chat.
============================================================
You: What papers cite Porter's 1980 book?
AI: [answer]
You: exit
Exiting chat.
```



## 📄 PDF Processing & Quality Diagnosis

- **Processing**: The `upload` command will process the PDF, extract sentences, citations, and references, and store results in the data directory.
- **Diagnosis**: Use `--diagnose` or the `diagnose` command to check if a PDF is suitable for processing. The CLI will print quality level, processability, and recommendations.
- **Output**: After processing, you’ll see stats (total sentences, citations, references) and example sentences with citations.

---

## 💬 Citation Analysis & Research Chat

- **Start chat**: `python -m src.core.cli chat`
- **Ask questions**: e.g.,
  - "What papers cite Rivkin's work on strategy?"
  - "List all papers written by Michael Porter."
  - "What is the main idea of Porter's 1980 book?"
- **Supported queries**: Citation relationships, author papers, paper content, concept explanations, etc.
- **System response**: The AI will analyze your question, search the database, and return structured answers.

---

## 🛠️ Advanced Features

### MinerU Integration (Optional)
- **Install MinerU**:
  ```bash
  pip install magic-pdf[full]
  ```
- **Enable in config**: Edit `config/model_config.json`:
  ```json
  {
    "pdf_processing": {
      "enable_mineru": true,
      "mineru_fallback": true
    }
  }
  ```
- MinerU is disabled by default. Use it for best PDF parsing quality.

### Grobid Deployment (Optional, for PDF Metadata Extraction)
- **Pull Grobid Docker image**:
  ```bash
  docker pull lfoppiano/grobid:0.8.0
  ```
- **Run Grobid**:
  ```bash
  docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0
  ```
- **Stop Grobid**:
  ```bash
  docker stop grobid
  docker rm grobid
  ```

---

## ❓ Troubleshooting & FAQ

- **Missing dependencies?**
  - Make sure you’ve run `pip install -r requirements.txt` in your virtual environment.
- **PDF not processing well?**
  - Try `--diagnose` to see recommendations. Enable MinerU for best results (But it is computationally expensive)
- **Database connection errors?**
  - Check your config files in `config/` for correct paths and credentials.
- **Chat/Research not working or AI errors?**
  - Make sure you have set your OpenAI API key as described above (`OPENAI_API_KEY`).
  - Check for typos or missing/expired keys.
  - Restart your terminal after setting the key.
- **Other issues?**
  - Check logs printed in the terminal for error messages.
  - For advanced help, see the `docs/` folder or open an issue on GitHub.

---

## 📜 License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.