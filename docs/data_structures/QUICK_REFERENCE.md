# Data Structure Quick Reference Index

## 🔍 Common Data Structure Quick Reference (v2.0)

### Core Output Formats

| Data Type | File Location | Primary Use |
|---------|---------|---------|
| Complete Processing Result | `processed_document.json` | Programmatic access to complete data |
| Sentence-level Data | `sentences_with_citations.jsonl` | Streaming processing and querying |
| Metadata | `metadata.json` | Quick access to paper information |
| Quality Report | `quality_report.json` | Processing quality assessment |

### Key Data Fields (Updated v2.0)

| Field Name | Type | Description | Example Value |
|-------|------|------|--------|
| `paper_id` | string | Unique paper identifier | `sha256_hash_value` |
| `sentence_index` | number | Sentence index | `42` |
| `has_citations` | boolean | Whether sentence contains citations | `true` |
| `citations` | array | Citation list with embedded argument analysis | `[{"intext": "Porter (1980)", "argument_analysis": {...}}]` |
| `citation_index` | number | Citation index within sentence | `0`, `1`, `2` |
| `argument_analysis` | object | Citation-level or sentence-level argument analysis | `{"has_argument_relations": true, "entities": [...]}` |
| `relation_type` | string | Argument relation type | `"ELABORATES"`, `"SUPPORTS"` |
| `confidence` | number | Confidence score | `0.856` |
| `entity_text` | string | Detected argument entity text | `"Porter (1980)"` |

## 🚀 What's New in v2.0

### Citation-Level Argument Analysis

**Major Change**: Argument analysis is now embedded directly in each citation, creating a one-to-one mapping.

#### Old Structure (v1.0):
```json
{
  "citations": [{"intext": "Porter (1980)", "paper_id": "..."}],
  "argument_analysis": {
    "entities": [
      {
        "relation_type": "SUPPORTS",
        "matched_citation_index": 0,
        "matched_paper_id": "..."
      }
    ]
  }
}
```

#### New Structure (v2.0):
```json
{
  "has_citations": true,
  "citations": [
    {
      "intext": "Porter (1980)",
      "paper_id": "...",
      "citation_index": 0,
      "argument_analysis": {
        "has_argument_relations": true,
        "entities": [
          {
            "relation_type": "SUPPORTS",
            "confidence": 0.856,
            "entity_text": "Porter (1980)"
          }
        ]
      }
    }
  ],
  "argument_analysis": {
    "has_argument_relations": true
  }
}
```

### Key Benefits:

- ✅ **Direct Access**: `citation.argument_analysis.entities`
- ✅ **No Matching Logic**: No need for `matched_citation_index`
- ✅ **Cleaner Queries**: Find citations by relation type directly
- ✅ **Better Performance**: O(1) access to citation's argument data

### Quick Location Guide

#### 🎯 I want to understand...

**PDF Processing Result Formats** → [`pdf_processing.md`](./pdf_processing.md)
- PDF metadata structure
- Text extraction info
- Sentence parsing results
- Quality diagnosis format

**Citation Analysis Data** → [`citation_analysis.md`](./citation_analysis.md)  
- Citation mapping format
- In-text citation detection
- Reference entries
- Sentence-level citation analysis
- **NEW**: Citation-level argument analysis embedding

**Argument Relation Classification** → [`argument_classification.md`](./argument_classification.md)
- Relation type definitions
- Argument entity format
- Classification result structure
- **NEW**: Citation-level argument distribution
- **NEW**: Entity-citation matching algorithm

**Unified Processing Results** → [`document_processing.md`](./document_processing.md)
- Complete processing results
- Sentence-level analysis
- Processing statistics
- **NEW**: Citation-level argument structure

## 📊 Data Structure Relationship Diagram

```
PDF Document
    ↓ (PDFProcessor)
PDF Metadata + Sentence List
    ↓ (CitationParser)
Citation Mappings
    ↓ (ArgumentClassifier)
Argument Relations
    ↓ (DocumentProcessor)
Unified Processing Results
    ↓
Output Files (JSON/JSONL)
```

## 🚀 Common Usage Scenarios

### Scenario 1: Parse Single PDF Document
```python
# 1. Initialize processor
doc_processor = DocumentProcessor()

# 2. Process document
result = doc_processor.process_document("paper.pdf")

# 3. Access results
sentences = result["sentences_with_citations"]
citations = [s for s in sentences if s["citations"]]
```

### Scenario 2: Analyze Argument Relations
```python
# 1. Initialize classifier
classifier = ArgumentClassifier()

# 2. Classify sentence
result = classifier.classify("Porter (1980) supports our claim...")

# 3. Get relations
relations = result["relations"]  # ["SUPPORTS"]
entities = result["entities"]    # Entity details
```

### Scenario 3: Batch Processing and Querying
```python
# 1. Load JSONL data
sentences = []
with open("sentences_with_citations.jsonl") as f:
    for line in f:
        sentences.append(json.loads(line))

# 2. Query sentences with specific relations
support_sentences = [
    s for s in sentences 
    if s.get("argument_analysis", {}).get("relations") == ["SUPPORTS"]
]
```

## 🔧 Debugging Tips

### Common Field Checking
```python
# Check data integrity
def validate_sentence_data(sentence_data):
    required_fields = ["sentence_index", "sentence_text", "citations"]
    for field in required_fields:
        assert field in sentence_data, f"Missing field: {field}"
    
    # Check argument analysis
    if sentence_data.get("citations"):
        assert "argument_analysis" in sentence_data, "Missing argument analysis"
```

### Data Statistics Script
```python
def analyze_processing_stats(result):
    stats = result["processing_stats"]
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Citation sentences: {stats['sentences_with_citations']}")
    print(f"Argument relations: {stats['total_argument_relations']}")
    
    # Calculate ratios
    citation_rate = stats['sentences_with_citations'] / stats['total_sentences']
    print(f"Citation density: {citation_rate:.2%}")
```

## 📝 Data Validation Checklist

- [ ] Is `paper_id` a valid SHA256 hash value
- [ ] Are `sentence_index` values consecutive and starting from 0
- [ ] Does `citations` array contain valid citation objects
- [ ] Does `argument_analysis` exist only when citations are present
- [ ] Are `confidence` values within 0-1 range
- [ ] Are all required fields present
- [ ] Is JSON format valid
- [ ] Is file encoding UTF-8 