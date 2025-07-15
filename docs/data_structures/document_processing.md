# Unified Document Processing Data Structures

## DocumentProcessor Core Data Structures

### 1. Complete Processing Result

```json
{
  "metadata": "object - document metadata",
  "paper_id": "string - unique paper identifier",
  "sentences_with_citations": ["object - sentence-level analysis results list"],
  "processing_stats": "object - processing statistics"
}
```

**Complete Example:**
```json
{
  "metadata": {
    "title": "Imitation of Complex Strategies",
    "authors": ["Jan W. Rivkin"],
    "year": "2000",
    "doi": "10.1287/mnsc.46.6.824.11940",
    "journal": "Management Science"
  },
  "paper_id": "sha256_hash_of_title_year",
  "sentences_with_citations": [
    {
      "sentence_index": 42,
      "sentence_text": "Porter (1980) argues that competitive advantage stems from strategic positioning.",
      "has_citations": true,
      "citations": [...],
      "argument_analysis": {...},
      "word_count": 12,
      "char_count": 89
    }
  ],
  "processing_stats": {
    "total_sentences": 644,
    "sentences_with_citations": 38,
    "total_citations": 43,
    "total_references": 85,
    "sentences_with_argument_relations": 15,
    "total_argument_relations": 18,
    "argument_classification_enabled": true,
    "processing_timestamp": "2025-01-15T14:30:00.000Z"
  }
}
```

### 2. Sentence-Level Analysis Result

```json
{
  "sentence_index": "number - sentence index",
  "sentence_text": "string - cleaned sentence text",
  "has_citations": "boolean - whether sentence contains citations",
  "citations": ["object - citation analysis results list"],
  "argument_analysis": "object - sentence-level argument summary (optional)",
  "word_count": "number - word count statistics",
  "char_count": "number - character count statistics",
  "error": "string - error message (if any)"
}
```

**Example:**
```json
{
  "sentence_index": 42,
  "sentence_text": "Porter (1980) argues that competitive advantage stems from strategic positioning.",
  "has_citations": true,
  "citations": [
    {
      "intext": "Porter (1980)",
      "reference": {
        "title": "Competitive Strategy",
        "authors": ["Michael E. Porter"],
        "year": "1980"
      },
      "paper_id": "competitive_strategy_1980",
      "citation_index": 0,
      "argument_analysis": {
        "has_argument_relations": true,
        "entities": [
          {
            "relation_type": "CITES",
            "confidence": 0.856,
            "entity_text": "Porter (1980)",
            "start_pos": 0,
            "end_pos": 13
          }
        ]
      }
    }
  ],
  "argument_analysis": {
    "has_argument_relations": true
  },
  "word_count": 12,
  "char_count": 89
}
```

### 3. Citation Structure with Embedded Argument Analysis

**New Structure (v2.0)**: Each citation now contains its own argument analysis, creating a direct one-to-one mapping between citations and their argument relations.

```json
{
  "intext": "string - in-text citation text",
  "reference": "object - reference information",
  "paper_id": "string - cited paper's unique identifier",
  "citation_index": "number - citation index within sentence",
  "argument_analysis": {
    "has_argument_relations": "boolean - whether argument relations were detected",
    "entities": [
      {
        "relation_type": "string - argument relation type",
        "confidence": "number - confidence score (0-1)",
        "entity_text": "string - detected entity text",
        "start_pos": "number - start position in sentence",
        "end_pos": "number - end position in sentence"
      }
    ],
    "error": "string - error message (if classification failed)"
  }
}
```

**Complete Citation Example:**
```json
{
  "intext": "(Topkis, 1998)",
  "reference": {
    "authors": ["D Topkis"],
    "title": "Supermodularity and Complementarity",
    "year": "1998",
    "journal": "Supermodularity and Complementarity",
    "publisher": "Princeton University Press"
  },
  "paper_id": "f25850e1df3efd29f411cc4c335c7eac6cda93a8e82c75a111751911c32d84fc",
  "citation_index": 0,
  "argument_analysis": {
    "has_argument_relations": true,
    "entities": [
      {
        "relation_type": "ELABORATES",
        "confidence": 0.25126200914382935,
        "entity_text": "Topkis 1998",
        "start_pos": 187,
        "end_pos": 198
      }
    ]
  }
}
```

### 4. Multiple Citations Example

When a sentence contains multiple citations, each citation has its own argument analysis:

```json
{
  "sentence_index": 334,
  "sentence_text": "Finally, the NP-completeness of the strategy formulation problem poses a question for the recent, important stream of research on complementarities (e.g., Milgrom and Roberts 1990, 1995; Topkis 1998 and references therein).",
  "has_citations": true,
  "citations": [
    {
      "intext": "(Topkis, 1998)",
      "reference": {...},
      "paper_id": "topkis_1998_id",
      "citation_index": 0,
      "argument_analysis": {
        "has_argument_relations": true,
        "entities": [
          {
            "relation_type": "ELABORATES",
            "confidence": 0.251,
            "entity_text": "Topkis 1998",
            "start_pos": 187,
            "end_pos": 198
          }
        ]
      }
    },
    {
      "intext": "(Milgrom and Roberts 1990, 1995)",
      "reference": {...},
      "paper_id": "milgrom_roberts_1990_id",
      "citation_index": 1,
      "argument_analysis": {
        "has_argument_relations": false,
        "entities": []
      }
    }
  ],
  "argument_analysis": {
    "has_argument_relations": true
  },
  "word_count": 31,
  "char_count": 223
}
```

### 5. Sentence-Level Argument Summary

The sentence-level `argument_analysis` now serves as a summary, indicating whether any citations in the sentence have argument relations:

```json
{
  "has_argument_relations": "boolean - true if any citation has argument relations",
  "error": "string - error message (if sentence-level classification failed)"
}
```

### 6. Processing Statistics

```json
{
  "total_sentences": "number - total number of sentences",
  "sentences_with_citations": "number - number of sentences with citations",
  "total_citations": "number - total number of citations",
  "total_references": "number - total number of references",
  "sentences_with_argument_relations": "number - number of sentences with argument relations",
  "citations_with_argument_relations": "number - number of citations with argument relations",
  "total_argument_relations": "number - total number of argument relations",
  "argument_classification_enabled": "boolean - whether argument classification was enabled",
  "processing_timestamp": "string - processing timestamp"
}
```

### 7. Quality Diagnosis Result

```json
{
  "pdf_path": "string - PDF file path",
  "timestamp": "string - diagnosis timestamp",
  "pdf_diagnosis": {
    "best_quality_score": "number - PDF quality score",
    "best_engine": "string - best PDF engine",
    "is_processable": "boolean - whether processable",
    "total_pages": "number - total pages",
    "pages_with_text": "number - pages with text"
  },
  "citation_diagnosis": {
    "references_count": "number - number of references",
    "references_extraction_success": "boolean - whether reference extraction succeeded",
    "metadata_quality": "boolean - whether metadata quality is good",
    "has_doi": "boolean - whether DOI exists"
  },
  "overall_assessment": {
    "is_processable": "boolean - overall processability",
    "quality_level": "string - quality level (excellent/good/fair/poor)",
    "recommendations": ["string - recommendation list"]
  }
}
```

## Key Improvements in v2.0 Structure

### 1. **Direct Citation-Argument Mapping**
- ✅ Each citation directly contains its argument analysis
- ✅ No need for `matched_citation_index` or `matched_paper_id` fields
- ✅ One-to-one relationship between citations and argument relations

### 2. **Simplified Data Access**
- ✅ To find argument relations for a citation: `citation.argument_analysis.entities`
- ✅ To check if citation has arguments: `citation.argument_analysis.has_argument_relations`
- ✅ No need to search through sentence-level entities

### 3. **Better Logical Structure**
- ✅ Sentence level: Contains all citations and summary of argument relations
- ✅ Citation level: Contains specific argument analysis for that citation
- ✅ Entity level: Contains specific argument relation details

### 4. **Enhanced Querying Capabilities**
```python
# Find all citations with argument relations
citations_with_args = [
    cite for cite in sentence['citations'] 
    if cite['argument_analysis']['has_argument_relations']
]

# Get all argument relation types in sentence
relation_types = [
    entity['relation_type'] 
    for cite in sentence['citations']
    for entity in cite['argument_analysis']['entities']
]

# Find citations by specific relation type
elaborates_citations = [
    cite for cite in sentence['citations']
    for entity in cite['argument_analysis']['entities']
    if entity['relation_type'] == 'ELABORATES'
]
```

## File Output Formats

### 1. Main Results File (processed_document.json)

Contains complete processing results with citation-level argument analysis:

```json
{
  "metadata": {...},
  "paper_id": "string",
  "sentences_with_citations": [...],
  "processing_stats": {...}
}
```

### 2. Sentence-Level Data File (sentences_with_citations.jsonl)

One sentence analysis result per line, with embedded citation-level argument analysis:

```jsonl
{"sentence_index": 0, "sentence_text": "...", "has_citations": false, "citations": [], "word_count": 8}
{"sentence_index": 1, "sentence_text": "...", "has_citations": true, "citations": [{"citation_index": 0, "argument_analysis": {...}}], "argument_analysis": {...}}
```

### 3. Metadata File (metadata.json)

Independent metadata file, convenient for quick access:

```json
{
  "title": "...",
  "authors": [...],
  "year": "...",
  "doi": "...",
  "paper_id": "..."
}
```

## Caching Mechanism

### 1. Metadata Cache

```python
_metadata_cache = {
    "pdf_path": {
        "metadata": dict,
        "extraction_time": float,
        "cache_timestamp": str
    }
}
```

### 2. References Cache

```python
_references_cache = {
    "pdf_path": {
        "references": list,
        "grobid_success": bool,
        "cache_timestamp": str
    }
}
```

## Error Handling Strategy

### 1. Citation-Level Error Recovery

When argument classification fails for a specific citation:

```json
{
  "citation_index": 0,
  "intext": "(Porter, 1980)",
  "argument_analysis": {
    "has_argument_relations": false,
    "entities": [],
    "error": "Classification timeout"
  }
}
```

### 2. Sentence-Level Error Recovery

When citation analysis fails for an entire sentence:

```json
{
  "sentence_index": 42,
  "sentence_text": "cleaned sentence text",
  "has_citations": false,
  "citations": [],
  "error": "Citation parsing failed",
  "word_count": 12,
  "char_count": 89
}
```

### 3. Module Degradation Strategy

- **ArgumentClassifier failure**: Set `has_argument_relations: false` for all citations
- **CitationParser failure**: Return empty citation list, continue processing other sentences
- **PDFProcessor failure**: Complete failure, throw exception

### 4. Processing Quality Assessment

```python
QUALITY_THRESHOLDS = {
    "excellent": {"pdf_score": 7.0, "has_references": True},
    "good": {"pdf_score": 5.0, "has_references": True}, 
    "fair": {"pdf_score": 3.0, "has_references": False},
    "poor": {"pdf_score": 0.0, "has_references": False}
}
``` 