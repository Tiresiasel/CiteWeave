# Citation Analysis Data Structures

## CitationParser Data Structure Definitions

### 1. Citation Mapping (Updated v2.0)

```json
{
  "intext": "string - in-text citation text",
  "reference": {
    "title": "string - cited paper title",
    "authors": ["string - author list"],
    "year": "string - publication year",
    "journal": "string - journal name (optional)",
    "volume": "string - volume number (optional)",
    "pages": "string - page numbers (optional)",
    "doi": "string - DOI (optional)"
  },
  "paper_id": "string - cited paper's paper_id (if known)",
  "citation_index": "number - citation index within sentence",
  "argument_analysis": "object - embedded argument analysis (when enabled)"
}
```

**Example:**
```json
{
  "intext": "(Porter, 1980)",
  "reference": {
    "title": "Competitive Strategy: Techniques for Analyzing Industries and Competitors",
    "authors": ["Michael E. Porter"],
    "year": "1980",
    "journal": null,
    "volume": null,
    "pages": null,
    "doi": null
  },
  "paper_id": "sha256_hash_value",
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
```

### 2. Citation with Embedded Argument Analysis

**New in v2.0**: Each citation now directly contains its argument analysis, eliminating the need for separate entity-citation matching.

```json
{
  "intext": "string - in-text citation text",
  "reference": "object - reference information",
  "paper_id": "string - cited paper's unique identifier",
  "citation_index": "number - citation index within sentence (0-based)",
  "argument_analysis": {
    "has_argument_relations": "boolean - whether argument relations detected",
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

**Complete Example:**
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

### 3. In-text Citation Detection Result

```json
{
  "citation_text": "string - detected citation text",
  "citation_type": "string - citation type",
  "start_pos": "number - start position",
  "end_pos": "number - end position",
  "authors": ["string - extracted authors"],
  "year": "string - extracted year",
  "prefix": "string - prefix text (optional)"
}
```

**Citation Types:**
- `narrative`: Narrative citation, e.g., "Porter (1980) argues that..."
- `parenthetical`: Parenthetical citation, e.g., "(Porter, 1980)"

**Example:**
```json
{
  "citation_text": "Porter (1980)",
  "citation_type": "narrative", 
  "start_pos": 45,
  "end_pos": 58,
  "authors": ["Porter"],
  "year": "1980",
  "prefix": "As discussed by"
}
```

### 4. Reference Entry

```json
{
  "raw_text": "string - raw reference text",
  "parsed": {
    "title": "string - title",
    "authors": ["string - author list"],
    "year": "string - year",
    "journal": "string - journal",
    "volume": "string - volume",
    "issue": "string - issue",
    "pages": "string - pages",
    "doi": "string - DOI",
    "url": "string - URL"
  },
  "grobid_confidence": "number - GROBID parsing confidence",
  "paper_id": "string - generated paper_id"
}
```

**Example:**
```json
{
  "raw_text": "Porter, M. E. (1980). Competitive strategy: Techniques for analyzing industries and competitors. Free Press.",
  "parsed": {
    "title": "Competitive strategy: Techniques for analyzing industries and competitors",
    "authors": ["Michael E. Porter"],
    "year": "1980",
    "journal": null,
    "volume": null,
    "issue": null,
    "pages": null,
    "doi": null,
    "url": null
  },
  "grobid_confidence": 0.92,
  "paper_id": "competitive_strategy_techniques_for_analyzing_industries_and_competitors_1980"
}
```

### 5. Sentence-level Citation Analysis (Updated v2.0)

```json
{
  "sentence": "string - sentence text",
  "has_citations": "boolean - whether sentence contains citations",
  "citations": [
    {
      "intext": "string",
      "reference": "object",
      "paper_id": "string",
      "citation_index": "number",
      "argument_analysis": "object"
    }
  ],
  "citation_count": "number - citation count",
  "has_multiple_citations": "boolean - whether it has multiple citations"
}
```

**Example with Multiple Citations:**
```json
{
  "sentence": "Both Porter (1980) and Barney (1991) emphasize the importance of sustainable competitive advantage.",
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
            "confidence": 0.89,
            "entity_text": "Porter (1980)",
            "start_pos": 5,
            "end_pos": 18
          }
        ]
      }
    },
    {
      "intext": "Barney (1991)",
      "reference": {
        "title": "Firm Resources and Sustained Competitive Advantage",
        "authors": ["Jay Barney"],
        "year": "1991"
      },
      "paper_id": "firm_resources_and_sustained_competitive_advantage_1991",
      "citation_index": 1,
      "argument_analysis": {
        "has_argument_relations": true,
        "entities": [
          {
            "relation_type": "SUPPORTS",
            "confidence": 0.76,
            "entity_text": "Barney (1991)",
            "start_pos": 23,
            "end_pos": 36
          }
        ]
      }
    }
  ],
  "citation_count": 2,
  "has_multiple_citations": true
}
```

### 6. Citation without Argument Relations

When no argument relations are detected for a citation:

```json
{
  "intext": "(Smith, 2020)",
  "reference": {...},
  "paper_id": "smith_2020_id",
  "citation_index": 1,
  "argument_analysis": {
    "has_argument_relations": false,
    "entities": []
  }
}
```

## Key Advantages of v2.0 Structure

### 1. **Direct Citation-Argument Association**
- ✅ No need for `matched_citation_index` lookups
- ✅ Each citation is self-contained with its argument analysis
- ✅ Simplified data access patterns

### 2. **Enhanced Query Performance**
```python
# Find citations with specific relation types
elaborates_citations = [
    cite for cite in sentence['citations']
    for entity in cite['argument_analysis']['entities']
    if entity['relation_type'] == 'ELABORATES'
]

# Count citations with argument relations
citations_with_args = sum(
    1 for cite in sentence['citations']
    if cite['argument_analysis']['has_argument_relations']
)

# Get all paper IDs with argument relations
papers_with_args = [
    cite['paper_id'] for cite in sentence['citations']
    if cite['argument_analysis']['has_argument_relations']
]
```

### 3. **Logical Data Hierarchy**
```
Sentence
├── has_citations: boolean
├── citations[]
│   ├── citation_index: number
│   ├── intext: string
│   ├── reference: object
│   ├── paper_id: string
│   └── argument_analysis
│       ├── has_argument_relations: boolean
│       └── entities[]
│           ├── relation_type: string
│           ├── confidence: number
│           ├── entity_text: string
│           ├── start_pos: number
│           └── end_pos: number
└── argument_analysis (summary)
    └── has_argument_relations: boolean
```

## Internal Data Structures

### 1. Citation Detection Configuration

```python
CITATION_PATTERNS = {
    'narrative': [
        r'(?P<prefix>(?:According to|As noted by|Research by|)\s*)?(?P<authors>[\w\s,&.-]+?)\s*\((?P<year>\d{4}[a-z]?)\)',
        r'(?P<prefix>(?:According to|As noted by|Research by|)\s*)?(?P<authors>[\w\s,&.-]+?)\s*\((?P<year>\d{4}[a-z]?)[,;]\s*[^)]*\)'
    ],
    'parenthetical': [
        r'\((?P<authors>[\w\s,&.-]+?)[,;\s]+(?P<year>\d{4}[a-z]?)\)',
        r'\((?P<authors>[\w\s,&.-]+?)[,;\s]+(?P<year>\d{4}[a-z]?)[,;]\s*[^)]*\)'
    ]
}
```

### 2. Matching Strategy Priority

1. **Exact Match**: Author + year exact match
2. **Fuzzy Match**: Author surname + year match
3. **Partial Match**: First author + year match
4. **Year Match**: Year only match (low confidence)

### 3. Unicode Character Support

Supported international character sets:
- Turkish: İıĞğÜüŞşÖöÇç
- German: ÄäÖöÜüß
- French: ÀàÉéÈèÇç
- Spanish: ÑñÁáÉéÍíÓóÚú
- Nordic: ÅåÆæØø
- Eastern European: ČčŠšŽž

### 4. Error Handling

```json
{
  "error_type": "string - error type",
  "error_message": "string - error message",
  "sentence_index": "number - sentence index",
  "recovery_action": "string - recovery action"
}
```

**Error Types:**
- `grobid_failure`: GROBID service failure
- `parsing_error`: Parsing error
- `encoding_error`: Encoding error
- `regex_timeout`: Regular expression timeout
- `argument_classification_failure`: Argument classification failure

## Migration Guide: v1.0 → v2.0

### Old Structure (v1.0)
```json
{
  "citations": [{"intext": "...", "reference": {...}}],
  "argument_analysis": {
    "entities": [
      {
        "relation_type": "ELABORATES",
        "matched_citation_index": 0,
        "matched_paper_id": "paper_id"
      }
    ]
  }
}
```

### New Structure (v2.0)
```json
{
  "citations": [
    {
      "intext": "...",
      "reference": {...},
      "citation_index": 0,
      "argument_analysis": {
        "has_argument_relations": true,
        "entities": [
          {
            "relation_type": "ELABORATES",
            "confidence": 0.85,
            "entity_text": "..."
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

### Breaking Changes
- ❌ Removed: `matched_citation_index`, `matched_paper_id` fields
- ❌ Removed: Sentence-level `entities` array
- ✅ Added: `citation_index` field to each citation
- ✅ Added: `argument_analysis` object to each citation
- ✅ Added: `has_citations` field to sentence level 