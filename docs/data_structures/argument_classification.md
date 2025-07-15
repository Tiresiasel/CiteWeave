# Argument Relation Classification Data Structures

## ArgumentClassifier Data Structure Definitions

### 1. Relation Type Definition

Based on dynamic loading structure from `docs/schema/relation_types.yaml`:

```json
{
  "id": "string - relation type ID",
  "label": "string - relation label",
  "description": "string - detailed description", 
  "is_default": "boolean - whether it's a default relation",
  "is_deprecated": "boolean - whether it's deprecated",
  "examples": ["string - example sentence list"]
}
```

**Example:**
```json
{
  "id": "SUPPORTS",
  "label": "supports",
  "description": "A relationship where one argument explicitly strengthens or reinforces another argument, claim, or proposition.",
  "is_default": true,
  "is_deprecated": false,
  "examples": [
    "The model result further supports our claim about entry deterring behavior.",
    "This mechanism gives theoretical support to the claim about consumer uncertainty."
  ]
}
```

### 2. Argument Entity (v2.0 - Citation-Embedded)

**New in v2.0**: Argument entities are now embedded directly within citations, eliminating the need for citation matching.

```json
{
  "relation_type": "string - relation type ID",
  "confidence": "number - confidence score (0-1)",
  "entity_text": "string - detected entity text",
  "start_pos": "number - start position in sentence",
  "end_pos": "number - end position in sentence"
}
```

**Example:**
```json
{
  "relation_type": "ELABORATES",
  "confidence": 0.25126200914382935,
  "entity_text": "Topkis 1998",
  "start_pos": 187,
  "end_pos": 198
}
```

### 3. Citation-Level Argument Analysis (v2.0)

Each citation now contains its own argument analysis structure:

```json
{
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
```

**Complete Example:**
```json
{
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
```

### 4. Full Citation with Argument Analysis

```json
{
  "intext": "(Topkis, 1998)",
  "reference": {
    "authors": ["D Topkis"],
    "title": "Supermodularity and Complementarity",
    "year": "1998"
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

### 5. Sentence-Level Argument Summary (v2.0)

The sentence level now contains only a summary of whether any citations have argument relations:

```json
{
  "argument_analysis": {
    "has_argument_relations": "boolean - true if any citation has argument relations",
    "error": "string - error message (if sentence-level classification failed)"
  }
}
```

### 6. Multiple Citations with Different Argument Relations

```json
{
  "sentence_index": 334,
  "sentence_text": "Finally, the NP-completeness of the strategy formulation problem poses a question for the recent, important stream of research on complementarities (e.g., Milgrom and Roberts 1990, 1995; Topkis 1998 and references therein).",
  "has_citations": true,
  "citations": [
    {
      "intext": "(Topkis, 1998)",
      "citation_index": 0,
      "paper_id": "topkis_1998_id",
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
      "citation_index": 1,
      "paper_id": "milgrom_roberts_1990_id",
      "argument_analysis": {
        "has_argument_relations": false,
        "entities": []
      }
    }
  ],
  "argument_analysis": {
    "has_argument_relations": true
  }
}
```

### 7. Raw Classification Result (Internal)

This is the output from the ArgumentClassifier before being distributed to citations:

```json
{
  "relations": ["string - detected relation type list"],
  "entities": [
    {
      "relation_type": "string",
      "start_pos": "number",
      "end_pos": "number",
      "text": "string",
      "confidence": "number"
    }
  ],
  "confidence_scores": ["number - confidence list for all tokens"],
  "sentence": "string - input sentence"
}
```

**Example:**
```json
{
  "relations": ["ELABORATES"],
  "entities": [
    {
      "relation_type": "ELABORATES",
      "start_pos": 187,
      "end_pos": 198,
      "text": "Topkis 1998",
      "confidence": 0.25126200914382935
    }
  ],
  "confidence_scores": [0.98, 0.95, 0.87, 0.92, 0.85, 0.90],
  "sentence": "Finally, the NP-completeness of the strategy formulation problem poses a question for the recent, important stream of research on complementarities (e.g., Milgrom and Roberts 1990, 1995; Topkis 1998 and references therein)."
}
```

## Key Improvements in v2.0

### 1. **Entity-Citation Matching Algorithm**

The DocumentProcessor now includes sophisticated matching logic to associate detected entities with specific citations:

```python
def _match_entity_to_citation(self, entity: Dict, citations: List[Dict], sentence: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Match a detected argument entity to the most relevant citation.
    
    Uses combined scoring:
    - Text similarity (70% weight): Based on token overlap and substring matching
    - Position similarity (30% weight): Based on proximity in sentence
    """
```

**Matching Features:**
- ✅ Text similarity analysis (token overlap, substring matching)
- ✅ Position-based proximity scoring
- ✅ Confidence threshold filtering (0.3 default)
- ✅ Handles multiple citations per sentence correctly

### 2. **Direct Citation Access**

```python
# Find all citations with ELABORATES relations
elaborates_citations = [
    cite for cite in sentence['citations']
    for entity in cite['argument_analysis']['entities']
    if entity['relation_type'] == 'ELABORATES'
]

# Get paper IDs with specific argument relations
papers_that_elaborate = [
    cite['paper_id'] for cite in sentence['citations']
    if any(entity['relation_type'] == 'ELABORATES' 
           for entity in cite['argument_analysis']['entities'])
]

# Count argument relations per citation
for cite in sentence['citations']:
    relation_count = len(cite['argument_analysis']['entities'])
    print(f"Citation {cite['intext']} has {relation_count} argument relations")
```

### 3. **Simplified Data Processing**

**Old v1.0 approach:**
```python
# Find which citation an entity belongs to
for entity in sentence['argument_analysis']['entities']:
    citation_idx = entity['matched_citation_index']
    if citation_idx is not None:
        citation = sentence['citations'][citation_idx]
        paper_id = entity['matched_paper_id']
```

**New v2.0 approach:**
```python
# Direct access to citation's argument analysis
for citation in sentence['citations']:
    if citation['argument_analysis']['has_argument_relations']:
        for entity in citation['argument_analysis']['entities']:
            relation_type = entity['relation_type']
            paper_id = citation['paper_id']
```

## Error Handling in v2.0

### 1. Citation-Level Errors

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

### 2. Sentence-Level Errors

When argument classification fails for the entire sentence:

```json
{
  "argument_analysis": {
    "has_argument_relations": false,
    "error": "ArgumentClassifier initialization failed"
  }
}
```

All citations in this case will have:
```json
{
  "argument_analysis": {
    "has_argument_relations": false,
    "entities": [],
    "error": "Sentence-level classification failed"
  }
}
```

## Training Data Structures

### 1. Training Sample

```json
{
  "text": "string - input text",
  "labels": ["string - BIO label sequence"],
  "relations": ["string - relation type list"],
  "tokens": ["string - tokenization result"],
  "token_labels": ["string - token-level labels"]
}
```

**Example:**
```json
{
  "text": "Porter (1980) argues that competitive advantage stems from strategic positioning.",
  "labels": ["O", "B-CITES", "I-CITES", "O", "O", "O", "O", "O", "O", "O", "O"],
  "relations": ["CITES"],
  "tokens": ["Porter", "(", "1980", ")", "argues", "that", "competitive", "advantage", "stems", "from", "strategic", "positioning", "."],
  "token_labels": ["B-CITES", "I-CITES", "I-CITES", "I-CITES", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

### 2. Test Results

```json
{
  "test_data_path": "string - test data path",
  "model_path": "string - model path",
  "test_examples_count": "number - test sample count",
  "device_used": "string - device used",
  "token_level_metrics": {
    "accuracy": "number - token-level accuracy",
    "f1_macro": "number - macro-average F1",
    "f1_weighted": "number - weighted average F1",
    "precision_macro": "number - macro-average precision",
    "recall_macro": "number - macro-average recall"
  },
  "entity_level_metrics": {
    "precision": "number - entity-level precision",
    "recall": "number - entity-level recall", 
    "f1": "number - entity-level F1",
    "predicted_entities": "number - predicted entity count",
    "actual_entities": "number - actual entity count",
    "correct_entities": "number - correct entity count"
  },
  "per_relation_metrics": {
    "RELATION_TYPE": {
      "precision": "number",
      "recall": "number",
      "f1": "number",
      "predicted_count": "number",
      "actual_count": "number", 
      "correct_count": "number"
    }
  }
}
```

## Model Configuration

### 1. Model Parameters

```python
MODEL_CONFIG = {
    "model_name": "allenai/scibert_scivocab_uncased",
    "num_labels": 21,  # O + 2 * 10 relations
    "max_length": 512,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 100
}
```

### 2. Device Optimization Configuration

```python
DEVICE_CONFIGS = {
    "mps": {
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "fp16": False
    },
    "cuda": {
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "fp16": True
    },
    "cpu": {
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "fp16": False
    }
}
```

### 3. Performance Benchmarks

Current model performance (needs improvement):
- Token-level accuracy: 100%
- Entity-level F1: 37-51% (across relation types)
- Average confidence: 0.495

**Improvement targets**:
- Entity-level F1: >80%
- Average confidence: >0.8
- Support for more complex argument relation detection

## Migration Guide: v1.0 → v2.0

### Breaking Changes

#### Removed Fields:
- ❌ `entity_index` (no longer needed)
- ❌ `matched_citation_index` (replaced by direct embedding)
- ❌ `matched_paper_id` (available as `citation.paper_id`)
- ❌ Sentence-level `entities` array (moved to citation level)
- ❌ Sentence-level `relations` array (replaced by summary)

#### Added Fields:
- ✅ `citation_index` in each citation
- ✅ `argument_analysis` object in each citation
- ✅ `has_citations` boolean in sentence
- ✅ Simplified sentence-level `argument_analysis` summary

#### Data Access Changes:

**Old v1.0:**
```python
# Find entities for a specific citation
citation_entities = [
    entity for entity in sentence['argument_analysis']['entities']
    if entity['matched_citation_index'] == 0
]
```

**New v2.0:**
```python
# Direct access to citation's entities
citation_entities = sentence['citations'][0]['argument_analysis']['entities']
``` 