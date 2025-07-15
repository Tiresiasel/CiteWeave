# PDF Processing Data Structures

## PDFProcessor Data Structure Definitions

### 1. PDF Metadata

```json
{
  "title": "string - paper title",
  "authors": ["string - author list"],
  "year": "string - publication year",
  "doi": "string - DOI identifier",
  "abstract": "string - abstract (optional)",
  "journal": "string - journal name (optional)",
  "volume": "string - volume number (optional)",
  "pages": "string - page range (optional)"
}
```

**Example:**
```json
{
  "title": "Imitation of Complex Strategies",
  "authors": ["Jan W. Rivkin"],
  "year": "2000",
  "doi": "10.1287/mnsc.46.6.824.11940",
  "abstract": "This paper examines how organizations imitate...",
  "journal": "Management Science",
  "volume": "46",
  "pages": "824-844"
}
```

### 2. Text Extraction Info

```json
{
  "engine": "string - PDF engine used",
  "total_pages": "number - total page count",
  "pages_with_text": "number - pages with text",
  "quality_score": "number - quality score (0-10)",
  "extraction_time": "number - extraction time (seconds)",
  "is_scanned": "boolean - whether it's a scanned document",
  "ocr_used": "boolean - whether OCR was used",
  "warnings": ["string - warning message list"]
}
```

**Example:**
```json
{
  "engine": "pymupdf",
  "total_pages": 20,
  "pages_with_text": 20,
  "quality_score": 8.5,
  "extraction_time": 2.3,
  "is_scanned": false,
  "ocr_used": false,
  "warnings": []
}
```

### 3. Sentence Parsing Result

```json
{
  "sentences": ["string - cleaned sentence list"],
  "raw_sentence_count": "number - raw sentence count",
  "filtered_sentence_count": "number - filtered sentence count",
  "main_content_chars": "number - main content character count",
  "reference_section_chars": "number - reference section character count"
}
```

**Example:**
```json
{
  "sentences": [
    "Strategic management research has long been concerned with understanding competitive advantage.",
    "Porter (1980) argues that competitive advantage stems from strategic positioning.",
    "This framework has been widely adopted in subsequent research."
  ],
  "raw_sentence_count": 450,
  "filtered_sentence_count": 380,
  "main_content_chars": 52000,
  "reference_section_chars": 8500
}
```

### 4. Quality Diagnosis

```json
{
  "pdf_path": "string - PDF file path",
  "timestamp": "string - diagnosis timestamp",
  "best_engine": "string - best engine",
  "best_quality_score": "number - best quality score",
  "engines_tested": ["string - tested engine list"],
  "is_processable": "boolean - whether processable",
  "recommendations": ["string - recommendation list"],
  "detailed_scores": {
    "engine_name": {
      "quality_score": "number",
      "character_count": "number",
      "word_count": "number",
      "line_count": "number"
    }
  }
}
```

**Example:**
```json
{
  "pdf_path": "test.pdf",
  "timestamp": "2025-01-15T10:30:00",
  "best_engine": "pymupdf",
  "best_quality_score": 8.5,
  "engines_tested": ["pymupdf", "pdfplumber"],
  "is_processable": true,
  "recommendations": ["Document appears suitable for citation analysis"],
  "detailed_scores": {
    "pymupdf": {
      "quality_score": 8.5,
      "character_count": 52000,
      "word_count": 8500,
      "line_count": 1200
    }
  }
}
```

## Internal Data Structures

### 1. Engine Evaluation Result

```python
{
    "engine": str,           # Engine name
    "text": str,            # Extracted text
    "info": dict,           # Extraction info
    "quality_score": float  # Quality score
}
```

### 2. Sentence Filtering Criteria

- Minimum length: 10 characters
- Maximum length: 2000 characters
- Exclusion patterns: Copyright notices, download information, headers/footers
- Format noise detection: Special character ratio > 15%

### 3. File Naming Conventions

```
data/papers/{paper_id}/
├── metadata.json        # PDF metadata
├── sentences.jsonl      # Sentence list
├── quality_report.json  # Quality diagnosis report
└── full_text.txt       # Full text (optional)
```

Where `paper_id` = SHA256(title + year) 