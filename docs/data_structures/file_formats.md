# File Format Specifications

## Output File Directory Structure

```
data/papers/{paper_id}/
├── processed_document.json      # Complete processing results
├── sentences_with_citations.jsonl  # Sentence-level data (line-based)
├── metadata.json               # Document metadata
└── quality_report.json         # Quality diagnosis report (optional)
```

Where `paper_id` = SHA256(title + year)

## Detailed File Format Descriptions

### 1. processed_document.json

**Purpose**: Complete document processing results, containing all information
**Format**: Single JSON object
**Size**: Usually 50KB - 5MB (depending on document length)

```json
{
  "metadata": {
    "title": "string",
    "authors": ["string"],
    "year": "string", 
    "doi": "string",
    "journal": "string",
    "abstract": "string"
  },
  "paper_id": "string",
  "sentences_with_citations": [
    {
      "sentence_index": "number",
      "sentence_text": "string",
      "citations": ["object"],
      "argument_analysis": "object",
      "word_count": "number",
      "char_count": "number"
    }
  ],
  "processing_stats": {
    "total_sentences": "number",
    "sentences_with_citations": "number",
    "total_citations": "number",
    "total_references": "number",
    "sentences_with_argument_relations": "number",
    "total_argument_relations": "number",
    "argument_classification_enabled": "boolean",
    "processing_timestamp": "string"
  }
}
```

### 2. sentences_with_citations.jsonl

**Purpose**: Sentence-level data, convenient for streaming processing and querying
**Format**: One JSON object per line (JSONL format)
**Size**: Usually 10KB - 1MB

```jsonl
{"sentence_index": 0, "sentence_text": "Academic research has shown...", "citations": [], "word_count": 8, "char_count": 45}
{"sentence_index": 1, "sentence_text": "Porter (1980) argues that...", "citations": [{"intext": "Porter (1980)", "reference": {...}}], "argument_analysis": {"relations": ["CITES"], "entities": [...]}, "word_count": 12, "char_count": 89}
{"sentence_index": 2, "sentence_text": "This framework supports...", "citations": [], "word_count": 15, "char_count": 67}
```

**Field Descriptions**:
- `sentence_index`: Sentence index in the document (starting from 0)
- `sentence_text`: Cleaned sentence text
- `citations`: List of citations in this sentence (may be empty)
- `argument_analysis`: Argument relation analysis (only when enabled and sentence has citations)
- `word_count`: Word count statistics
- `char_count`: Character count statistics

### 3. metadata.json

**Purpose**: Independent metadata file, convenient for quick access
**Format**: Single JSON object
**Size**: Usually 1-10KB

```json
{
  "title": "Imitation of Complex Strategies",
  "authors": ["Jan W. Rivkin"],
  "year": "2000",
  "doi": "10.1287/mnsc.46.6.824.11940",
  "journal": "Management Science",
  "volume": "46",
  "issue": "6",
  "pages": "824-844",
  "abstract": "This paper examines how organizations imitate complex strategies...",
  "paper_id": "sha256_hash_value",
  "extraction_method": "crossref_api",
  "extraction_timestamp": "2025-01-15T14:30:00.000Z"
}
```

### 4. quality_report.json

**Purpose**: Processing quality diagnosis report
**Format**: Single JSON object
**Size**: Usually 2-20KB

```json
{
  "pdf_path": "test_files/paper.pdf",
  "timestamp": "2025-01-15T14:30:00.000Z",
  "pdf_diagnosis": {
    "best_engine": "pymupdf",
    "best_quality_score": 8.5,
    "engines_tested": ["pymupdf", "pdfplumber"],
    "total_pages": 20,
    "pages_with_text": 20,
    "is_scanned": false,
    "ocr_used": false,
    "extraction_time": 2.3
  },
  "citation_diagnosis": {
    "references_count": 85,
    "references_extraction_success": true,
    "grobid_success": true,
    "metadata_quality": true,
    "has_doi": true
  },
  "overall_assessment": {
    "is_processable": true,
    "quality_level": "excellent",
    "recommendations": [
      "Document appears suitable for citation analysis"
    ]
  }
}
```

## Model-Specific Data Formats

### 1. Training Data Format (models/argument_classifier/datasets/)

#### single_relation.jsonl / multi_relation.jsonl
**Purpose**: Training data for argument relation classifier

```jsonl
{"text": "Porter (1980) argues that competitive advantage stems from strategic positioning.", "labels": ["O", "B-CITES", "I-CITES", "O", "O", "O", "O", "O", "O", "O", "O"], "relations": ["CITES"]}
{"text": "This finding supports our claim about market dynamics.", "labels": ["O", "O", "B-SUPPORTS", "I-SUPPORTS", "I-SUPPORTS", "O", "O", "O"], "relations": ["SUPPORTS"]}
```

#### train_data.jsonl / test_data.jsonl
**Purpose**: Data after train/test split

```jsonl
{"text": "Porter (1980) argues that competitive advantage stems from strategic positioning.", "labels": ["O", "B-CITES", "I-CITES", "O", "O", "O", "O", "O", "O", "O", "O"], "relations": ["CITES"], "tokens": ["Porter", "(", "1980", ")", "argues", "that", "competitive", "advantage", "stems", "from", "strategic", "positioning", "."]}
```

### 2. Test Results Format

#### test_results.json
**Purpose**: Model test results report

```json
{
  "test_data_path": "datasets/test_data.jsonl",
  "model_path": "checkpoints/citation_classifier",
  "test_examples_count": 400,
  "device_used": "mps",
  "token_level_metrics": {
    "accuracy": 1.0,
    "f1_macro": 1.0,
    "f1_weighted": 1.0
  },
  "entity_level_metrics": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "per_relation_metrics": {
    "SUPPORTS": {
      "precision": 0.385,
      "recall": 0.385,
      "f1": 0.385
    }
  }
}
```

## File Naming Conventions

### 1. Output File Naming

- **Fixed Names**: All output files use fixed names for programmatic access
- **Timestamps**: Only include timestamps in filenames when version control is needed
- **Encoding**: All files use UTF-8 encoding

### 2. Paper ID Generation Rules

```python
def generate_paper_id(title: str, year: str) -> str:
    """Generate unique paper identifier"""
    # Clean title
    clean_title = re.sub(r"[^\w\s]", "", title)
    clean_title = re.sub(r"\s+", " ", clean_title).strip()
    
    # Combine and hash
    combined = f"{clean_title}_{year}".lower()
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
```

### 3. Directory Structure Specifications

```
data/
├── papers/                      # Processed paper data
│   ├── {paper_id_1}/
│   ├── {paper_id_2}/
│   └── ...
├── models/                      # Model-related files
│   └── argument_classifier/
│       ├── datasets/           # Training data
│       ├── checkpoints/        # Model checkpoints
│       └── test_results.json   # Test results
└── cache/                      # Temporary cache files (optional)
    ├── grobid_cache/
    └── metadata_cache/
```

## Version Compatibility

### 1. File Format Version

Current Version: **v1.2**

- **v1.0**: Basic PDF processing and citation analysis
- **v1.1**: Added argument relation classification support
- **v1.2**: Unified DocumentProcessor architecture

### 2. Backward Compatibility

- All v1.x formats maintain backward compatibility
- New fields added as optional fields
- Deprecated fields marked as deprecated but retained

### 3. Upgrade Migration

```python
def migrate_v1_to_v2(old_data: dict) -> dict:
    """File format upgrade migration function"""
    # Handle format changes
    # Add default values for new fields
    # Maintain backward compatibility
    pass
``` 