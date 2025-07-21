# PDF Query System Documentation

## Overview

The PDF Query System extends the CiteWeave research platform with direct access to stored PDF documents. Instead of relying solely on database indices, researchers can now query the complete content of academic papers directly from their processed files.

## Architecture

### Directory Structure

```
data/papers/
├── <paper_id_1>/
│   ├── metadata.json          # Paper metadata (title, authors, year, etc.)
│   ├── processed_document.json # Complete paper content with sections
│   └── sentences_with_citations.jsonl # Citation-specific data
├── <paper_id_2>/
│   ├── metadata.json
│   ├── processed_document.json
│   └── sentences_with_citations.jsonl
└── ...
```

### Key Components

1. **QueryDBAgent** - Extended with PDF query methods
2. **LangGraphResearchSystem** - Integrated PDF tools in the multi-agent workflow
3. **PDF Query Tools** - Five specialized functions for different query types

## PDF Query Functions

### 1. `query_pdf_content(paper_id, query, context_window=500)`

**Purpose**: Search for specific keywords or phrases within a paper's content.

**Parameters**:
- `paper_id`: Unique identifier for the paper
- `query`: Search term or phrase
- `context_window`: Characters to include around matches (default: 500)

**Returns**:
```json
{
  "found": true,
  "paper_id": "abc123...",
  "metadata": {...},
  "query": "competitive advantage",
  "total_matches": 5,
  "data": [
    {
      "section_index": 2,
      "section_title": "Theoretical Framework",
      "section_type": "content",
      "matches": [
        {
          "position": 1234,
          "context": "...competitive advantage stems from...",
          "highlight_start": 150,
          "highlight_end": 170
        }
      ],
      "word_count": 1500,
      "citations": [...]
    }
  ]
}
```

**Use Cases**:
- Finding specific mentions of concepts
- Locating exact quotes or definitions
- Identifying all occurrences of technical terms

### 2. `get_full_pdf_content(paper_id)`

**Purpose**: Retrieve the complete content of a paper for comprehensive analysis.

**Returns**:
```json
{
  "found": true,
  "paper_id": "abc123...",
  "metadata": {...},
  "sections_count": 12,
  "total_word_count": 8500,
  "section_summaries": [...],
  "full_text": "## Introduction\n\n...",
  "data": {...} // Complete processed document
}
```

**Use Cases**:
- Complete paper analysis
- Content summarization
- Comprehensive research overview

### 3. `query_pdf_by_author_and_content(author_name, content_query)`

**Purpose**: Find papers by a specific author and search their content.

**Parameters**:
- `author_name`: Author to search for
- `content_query`: Content to find within author's papers

**Returns**:
```json
{
  "found": true,
  "author_name": "Porter",
  "content_query": "competitive advantage",
  "papers_found": 3,
  "papers_with_content": 2,
  "data": [
    {
      "paper_metadata": {...},
      "content_matches": {...}
    }
  ]
}
```

**Use Cases**:
- Author-specific content analysis
- Tracking concept evolution across an author's work
- Comparative analysis within an author's corpus

### 4. `query_pdf_by_title_and_content(title_query, content_query)`

**Purpose**: Find papers by title pattern and search their content.

**Use Cases**:
- Analyzing specific papers by title matching
- Content verification in known papers
- Targeted paper analysis

### 5. `semantic_search_pdf_content(paper_id, query, similarity_threshold=0.5)`

**Purpose**: Perform semantic similarity search within a paper using sentence transformers.

**Parameters**:
- `paper_id`: Target paper identifier
- `query`: Conceptual query (e.g., "factors affecting strategy imitation")
- `similarity_threshold`: Minimum similarity score (0.0-1.0)

**Returns**:
```json
{
  "found": true,
  "paper_id": "abc123...",
  "query": "factors affecting strategy imitation",
  "similarity_threshold": 0.5,
  "total_chunks_searched": 45,
  "relevant_chunks_found": 8,
  "data": [
    {
      "content": "Complex strategies with interdependent elements...",
      "similarity_score": 0.823,
      "metadata": {
        "section_index": 3,
        "section_title": "Complexity Analysis",
        "paragraph_index": 2,
        "chunk_type": "paragraph"
      }
    }
  ]
}
```

**Use Cases**:
- Conceptual content discovery
- Finding semantically related content
- Advanced research queries beyond keyword matching

## Integration with LangGraph

### New Tools in Multi-Agent System

The PDF query functions are integrated as LangGraph tools:

```python
@tool
def query_pdf_content(paper_id: str, query: str, context_window: int = 500):
    """Query content directly from a PDF paper using keyword search."""

@tool  
def get_full_pdf_content(paper_id: str):
    """Get the complete content of a PDF paper."""

@tool
def query_pdf_by_author_and_content(author_name: str, content_query: str):
    """Find papers by author and search their content."""

@tool
def query_pdf_by_title_and_content(title_query: str, content_query: str):
    """Find papers by title and search their content."""

@tool
def semantic_search_pdf_content(paper_id: str, query: str, similarity_threshold: float = 0.5):
    """Perform semantic search within a specific PDF."""
```

### Updated AI Strategy

The LangGraph system now uses an enhanced strategy:

- **Database tools** for citation relationships and cross-paper analysis
- **PDF tools** for direct content access and comprehensive paper analysis
- **Hybrid approach** combining both for complete research coverage

## Usage Examples

### Example 1: Finding Specific Content in Porter's Work

```python
from query_db_agent import QueryDBAgent

agent = QueryDBAgent()

# Find Porter's papers and search for competitive advantage content
result = agent.query_pdf_by_author_and_content(
    author_name="porter",
    content_query="competitive advantage"
)

if result["found"]:
    for paper_data in result["data"]:
        paper = paper_data["paper_metadata"]
        matches = paper_data["content_matches"]
        print(f"Paper: {paper['title']}")
        print(f"Matches: {matches['total_matches']}")
```

### Example 2: Semantic Analysis of Strategy Papers

```python
# Get Porter's paper ID
papers = agent.get_papers_id_by_author("porter")
paper_id = papers["data"][0]["paper_id"]

# Semantic search for imitation barriers
result = agent.semantic_search_pdf_content(
    paper_id=paper_id,
    query="What makes strategies difficult to imitate?",
    similarity_threshold=0.4
)

for match in result["data"]:
    print(f"Similarity: {match['similarity_score']:.3f}")
    print(f"Content: {match['content'][:200]}...")
```

### Example 3: Complete Paper Analysis

```python
# Get full content of a specific paper
full_content = agent.get_full_pdf_content(paper_id)

if full_content["found"]:
    print(f"Title: {full_content['metadata']['title']}")
    print(f"Sections: {full_content['sections_count']}")
    print(f"Word Count: {full_content['total_word_count']}")
    
    # Access complete text
    full_text = full_content["full_text"]
    
    # Or work with individual sections
    for section in full_content["section_summaries"]:
        print(f"- {section['section_title']} ({section['word_count']} words)")
```

### Example 4: LangGraph Integration

```python
from multi_agent_research_system import LangGraphResearchSystem

system = LangGraphResearchSystem()

# The system automatically selects appropriate PDF tools
response = system.research_question(
    "What specific examples does Porter give of competitive advantages in his papers?"
)

print(response)
```

## Performance Considerations

### Keyword Search (`query_pdf_content`)
- **Speed**: Very fast (string matching)
- **Accuracy**: Exact matches only
- **Best for**: Finding specific terms, quotes, definitions

### Semantic Search (`semantic_search_pdf_content`)
- **Speed**: Moderate (requires embedding computation)
- **Accuracy**: High for conceptual queries
- **Best for**: Understanding content relationships, conceptual analysis

### Full Content Retrieval (`get_full_pdf_content`)
- **Speed**: Fast (direct file access)
- **Memory**: High for large papers
- **Best for**: Comprehensive analysis, complete context

## Error Handling

All PDF query functions include comprehensive error handling:

```json
{
  "found": false,
  "error": "Processed document not found for paper abc123",
  "data": []
}
```

Common error scenarios:
- Paper ID not found
- Corrupted processed document files
- Insufficient permissions
- Missing sentence transformer models (for semantic search)

## Dependencies

### Required Packages
- `sentence-transformers` (for semantic search)
- `scikit-learn` (for similarity computation)
- `numpy` (for numerical operations)

### Optional Fallbacks
- Semantic search falls back to keyword search if transformers unavailable
- System continues functioning with reduced capabilities

## Future Enhancements

1. **Multi-paper Semantic Search**: Search concepts across multiple papers simultaneously
2. **Advanced NLP**: Named entity recognition, concept extraction
3. **Citation Context Analysis**: Enhanced understanding of how papers cite each other
4. **Real-time Indexing**: Automatic processing of new papers
5. **Query Optimization**: Caching and performance improvements

## Conclusion

The PDF Query System provides direct access to the complete content of academic papers, complementing the existing database-driven approach. This hybrid system ensures comprehensive research coverage, enabling both precise citation analysis and deep content exploration. 