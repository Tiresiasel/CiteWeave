# Enhanced RAG Structure for CiteWeave

## Overview

CiteWeave implements an enhanced Retrieval-Augmented Generation (RAG) system that combines multiple levels of document analysis with sophisticated citation tracking and semantic search capabilities.

## Architecture Components

### 1. Multi-Level Document Processing

#### Document Hierarchy
```
Paper
├── Paragraphs (with section attribute)
│   ├── Sentences
│   └── Direct Citations
└── Arguments (Legacy)
    └── Relations to other Arguments/Papers
```

#### Processing Levels
- **Paragraph Level**: Individual paragraphs with section information and citation density metrics
- **Sentence Level**: Fine-grained sentence analysis with precise citation locations
- **Section Level**: Section information stored as paragraph attributes (not as separate nodes)

### 2. Vector Database Collections

#### Sentences Collection
- **Purpose**: Fine-grained semantic search at sentence level
- **Use Cases**: 
  - Finding specific claims or statements
  - Citation context analysis
  - Precise information retrieval
- **Metadata**: sentence_index, sentence_type, has_citations, word_count, char_count

#### Paragraphs Collection
- **Purpose**: Medium-grained semantic search with citation aggregation
- **Use Cases**:
  - Finding paragraphs discussing specific topics
  - Citation density analysis
  - Contextual information retrieval
- **Metadata**: paragraph_index, section, citation_count, sentence_count, has_citations

#### Sections Collection
- **Purpose**: High-level semantic search with structural information
- **Use Cases**:
  - Finding papers with specific methodology sections
  - Structural analysis of academic papers
  - High-level topic exploration
- **Metadata**: section_index, section_title, section_type, paragraph_count
- **Note**: This is for vector search only - sections are not separate nodes in the graph database

#### Citations Collection
- **Purpose**: Direct citation text search and analysis
- **Use Cases**:
  - Finding papers that cite specific works
  - Citation pattern analysis
  - Reference network exploration
- **Metadata**: citation_index, citation_text, cited_paper_id, citation_context, confidence

### 3. Graph Database Integration

#### Citation Network
- **Sentence Citations**: Direct citation relationships with context
- **Paragraph Citations**: Aggregated citation information
- **Cross-Reference Analysis**: Finding papers that cite or are cited by specific works

#### Structural Relationships
- **Hierarchical Organization**: Paper → Paragraph → Sentence
- **Citation Tracking**: Multi-level citation relationships
- **Metadata Association**: Rich metadata at each level
- **Section Information**: Stored as paragraph attributes for efficient querying

## RAG Query Pipeline

### 1. Query Analysis
```python
# Query intent classification
query_intent = analyze_query_intent(user_query)

# Level selection based on intent
if query_intent == "precise_fact":
    target_levels = ["sentences"]
elif query_intent == "contextual_info":
    target_levels = ["paragraphs", "sentences"]
elif query_intent == "structural_analysis":
    target_levels = ["paragraphs"]  # Section info available through paragraphs
```

### 2. Multi-Level Retrieval
```python
# Search across multiple collections
results = {}
for level in target_levels:
    results[level] = vector_indexer.search(
        query=user_query,
        collection_name=level,
        limit=limit_per_level
    )
```

### 3. Citation Context Enrichment
```python
# Enrich results with citation information
for level, level_results in results.items():
    for result in level_results:
        paper_id = result["paper_id"]
        citation_context = graph_db.get_citation_context_for_ai_analysis(paper_id)
        result["citation_network"] = citation_context
```

### 4. Result Ranking and Fusion
```python
# Cross-level result fusion
fused_results = fuse_multi_level_results(results)

# Re-ranking based on citation relevance
ranked_results = rank_by_citation_relevance(fused_results)
```

## Advanced Features

### 1. Citation-Aware Retrieval
- **Citation Density Scoring**: Prioritize results with higher citation relevance
- **Citation Network Analysis**: Consider citation relationships in ranking
- **Stub Paper Handling**: Handle papers that are only referenced, not uploaded

### 2. Multi-Modal Search
- **Semantic Search**: Vector-based similarity search
- **Structural Search**: Section-based queries through paragraph attributes
- **Citation Search**: Direct citation text queries
- **Hybrid Search**: Combination of multiple search strategies

### 3. Context-Aware Generation
- **Citation Context**: Include citation information in generated responses
- **Structural Context**: Consider document structure in response generation
- **Network Context**: Incorporate citation network information

## Implementation Details

### Vector Indexing
```python
class VectorIndexer:
    def index_sentences(self, paper_id: str, sentences: List[str], metadata: dict):
        # Generate embeddings for sentences
        vectors = self.model.encode(sentences, normalize_embeddings=True)
        
        # Create payload with unified metadata structure
        payload = {
            "paper_id": paper_id,
            "sentence_index": i,
            "text": text,
            "sentence_type": claim_types[i] if claim_types else "unspecified",
            # ... other metadata
        }
```

### Graph Database Operations
```python
class GraphDB:
    def create_sentence_citation(self, sentence_id: str, cited_paper_id: str,
                                citation_text: str, citation_context: str = "",
                                confidence: float = 1.0):
        # Create citation relationship with rich metadata
        query = """
        MATCH (s:Sentence {id: $sentence_id})
        MATCH (p:Paper {id: $cited_paper_id})
        MERGE (s)-[c:CITES]->(p)
        SET c.citation_text = $citation_text,
            c.citation_context = $citation_context,
            c.confidence = $confidence,
            c.created_at = datetime()
        """
```

### Database Integration
```python
class DatabaseIntegrator:
    def import_document(self, paper_id: str, force_reimport: bool = False):
        # Coordinate data flow between all storage systems
        # Ensure consistency across Neo4j, Qdrant, and file system
        # Handle multi-level citation tracking
        # Note: Sections are stored as paragraph attributes, not as separate nodes
```

## Section Handling Strategy

### Current Implementation
```python
# Sections are handled as follows:
for paragraph in paragraphs:
    # Create paragraph with section information as attribute
    self.graph_db.create_paragraph(
        paragraph_id=paragraph["id"],
        paper_id=paper_id,
        text=paragraph["text"],
        section=paragraph.get("section", "Unknown"),  # Section as string attribute
        # ... other attributes
    )
    
    # Section information is also indexed in vector database for search
    self.vector_indexer.index_sections(
        paper_id=paper_id,
        sections=section_data,  # For vector search capabilities
        metadata=metadata
    )
```

### Benefits of This Approach
- **Efficient Queries**: Section information is directly accessible without graph traversal
- **Flexible Updates**: Section names can be easily updated without restructuring the graph
- **Performance**: Fewer nodes and relationships to manage
- **Search Capability**: Section-level search still available through vector database

### Limitations
- **No Section Relationships**: Cannot model relationships between sections
- **Limited Section Analysis**: Complex section-level operations require aggregation
- **Section Metadata**: Section-level statistics need to be computed from paragraphs

## Performance Optimizations

### 1. Indexing Strategy
- **Parallel Processing**: Index multiple levels simultaneously
- **Batch Operations**: Bulk insert operations for better performance
- **Incremental Updates**: Support for updating existing documents

### 2. Query Optimization
- **Level-Specific Limits**: Different result limits for different levels
- **Early Termination**: Stop searching when sufficient results are found
- **Caching**: Cache frequently accessed citation networks

### 3. Storage Efficiency
- **Metadata Compression**: Efficient storage of rich metadata
- **Vector Quantization**: Optimize vector storage and retrieval
- **Relationship Indexing**: Fast graph traversal and citation lookups

## Use Cases

### 1. Academic Research
- **Literature Review**: Find relevant papers and citation patterns
- **Gap Analysis**: Identify areas with limited research coverage
- **Impact Assessment**: Analyze citation networks and influence

### 2. Content Analysis
- **Claim Verification**: Find supporting or contradicting evidence
- **Trend Analysis**: Track changes in research focus over time
- **Collaboration Discovery**: Find researchers working on similar topics

### 3. Knowledge Discovery
- **Cross-Domain Connections**: Find unexpected relationships between fields
- **Methodology Transfer**: Identify successful approaches from other domains
- **Research Synthesis**: Combine information from multiple sources

## Future Enhancements

### 1. Advanced NLP Integration
- **Entity Recognition**: Identify people, organizations, and concepts
- **Sentiment Analysis**: Analyze the tone of citations and references
- **Topic Modeling**: Automatic topic discovery and clustering

### 2. Enhanced Citation Analysis
- **Citation Intent**: Classify citations by purpose (support, critique, etc.)
- **Citation Strength**: Measure the strength of citation relationships
- **Temporal Analysis**: Track citation patterns over time

### 3. Interactive Exploration
- **Visual Citation Networks**: Interactive graph visualization
- **Query Refinement**: Iterative query improvement based on results
- **Personalized Search**: User-specific relevance ranking

### 4. Section Node Enhancement (Optional)
If you want to add Section nodes later:
```python
# Migration script example
def migrate_sections_to_nodes(self):
    # Extract unique sections from paragraphs
    # Create Section nodes
    # Establish BELONGS_TO relationships
    # Update existing queries
```

## Conclusion

The enhanced RAG structure in CiteWeave provides a comprehensive framework for academic document analysis and retrieval. By combining multi-level document processing, sophisticated citation tracking, and semantic search capabilities, it enables powerful research tools that go beyond traditional keyword-based search.

The current design choice to store section information as paragraph attributes provides an optimal balance between functionality and performance. While it doesn't support complex section-level graph operations, it enables efficient section-based queries and maintains a clean, performant graph structure.

The system's flexibility and extensibility make it suitable for various academic and research applications, while its performance optimizations ensure efficient operation even with large document collections. 