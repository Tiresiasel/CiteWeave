# API Interface Data Formats

## REST API (Flask) Endpoints

Base URL: `/api/v1`

- POST `/upload`
  - Description: Upload and process a single PDF file.
  - Request: `multipart/form-data` with field `file` (PDF).
  - Response (200):
    ```json
    {
      "success": true,
      "summary": {
        "paper_id": "string",
        "total_sentences": 0,
        "sentences_with_citations": 0,
        "total_citations": 0,
        "total_references": 0
      },
      "result": { }
    }
    ```
  - Errors (4xx/5xx): `Standard Error Response`.

- POST `/diagnose`
  - Description: Diagnose a PDF for processing quality before upload.
  - Request: `multipart/form-data` with field `file` (PDF).
  - Response (200):
    ```json
    {
      "success": true,
      "diagnosis": { }
    }
    ```

- POST `/chat`
  - Description: Stateless chat turn with the multi-agent research system.
  - Request: `application/json`
    ```json
    {
      "user_input": "string",
      "history": [{"user": "string", "ai": "string"}],
      "menu_choice": "1|2|3|4",
      "collected_data": {"results": {}}
    }
    ```
  - Response (200):
    ```json
    {
      "success": true,
      "response": {
        "text": "string",
        "collected_data": {"results": {}},
        "needs_user_choice": true,
        "menu": ["Yes, generate final answer", "No, gather more information", "Tell me what specific information you want", "Exit"]
      }
    }
    ```

- GET `/health`
  - Description: Health check endpoint.
  - Response: `{ "status": "ok" }`

## Core Class Method Interfaces

### 1. DocumentProcessor Interface

#### process_document()
```python
def process_document(pdf_path: str, save_results: bool = True) -> Dict
```

**Input Parameters:**
```json
{
  "pdf_path": "string - PDF file path",
  "save_results": "boolean - whether to save results to disk"
}
```

**Return Result:**
```json
{
  "metadata": "object - document metadata",
  "paper_id": "string - unique identifier", 
  "sentences_with_citations": "array - sentence analysis results",
  "processing_stats": "object - processing statistics"
}
```

#### get_sentences_with_citations()
```python
def get_sentences_with_citations(pdf_path: str, force_reprocess: bool = False) -> List[Dict]
```

**Input Parameters:**
```json
{
  "pdf_path": "string - PDF file path",
  "force_reprocess": "boolean - force reprocessing"
}
```

**Return Result:**
```json
[
  {
    "sentence_index": "number",
    "sentence_text": "string",
    "citations": "array",
    "argument_analysis": "object",
    "word_count": "number",
    "char_count": "number"
  }
]
```

#### diagnose_document_processing()
```python
def diagnose_document_processing(pdf_path: str) -> Dict
```

**Return Result:**
```json
{
  "pdf_diagnosis": "object - PDF quality diagnosis",
  "citation_diagnosis": "object - citation processing diagnosis", 
  "overall_assessment": "object - overall assessment"
}
```

### 2. ArgumentClassifier Interface

#### classify()
```python
def classify(sentence: str, citation_text: str = None) -> Dict
```

**Input Parameters:**
```json
{
  "sentence": "string - input sentence",
  "citation_text": "string - optional specific citation text"
}
```

**Return Result:**
```json
{
  "relations": ["string - detected relation types"],
  "entities": [
    {
      "relation_type": "string",
      "start_pos": "number",
      "end_pos": "number", 
      "text": "string",
      "confidence": "number"
    }
  ],
  "confidence_scores": ["number - token confidence list"],
  "sentence": "string - input sentence"
}
```

#### analyze_citation_context()
```python
def analyze_citation_context(sentence: str, citations: List[str] = None) -> Dict
```

#### get_supported_relations()
```python
def get_supported_relations() -> List[Dict]
```

**Return Result:**
```json
[
  {
    "id": "string - relation ID",
    "label": "string - relation label", 
    "description": "string - description",
    "examples": ["string - example list"]
  }
]
```

#### get_relation_info()
```python
def get_relation_info(relation_id: str) -> Optional[Dict]
```

### 3. CitationParser Interface

#### parse_sentence()
```python
def parse_sentence(sentence: str) -> List[Dict]
```

**Input Parameters:**
```json
{
  "sentence": "string - sentence to analyze"
}
```

**Return Result:**
```json
[
  {
    "intext": "string - in-text citation",
    "reference": {
      "title": "string",
      "authors": ["string"],
      "year": "string"
    },
    "paper_id": "string",
    "match_confidence": "number",
    "match_method": "string"
  }
]
```

#### parse_document()
```python
def parse_document(sentences: List[str]) -> List[Dict]
```

### 4. PDFProcessor Interface

#### parse_sentences()
```python
def parse_sentences(pdf_path: str) -> List[str]
```

#### extract_pdf_metadata()
```python
def extract_pdf_metadata(pdf_path: str) -> Dict
```

**Return Result:**
```json
{
  "title": "string",
  "authors": ["string"],
  "year": "string",
  "doi": "string",
  "journal": "string",
  "abstract": "string"
}
```

#### extract_text_with_best_engine()
```python
def extract_text_with_best_engine(pdf_path: str) -> Tuple[str, Dict]
```

**Return Result:**
```python
(
  "string - extracted text",
  {
    "engine": "string - engine used",
    "quality_score": "number - quality score",
    "total_pages": "number - total pages",
    "extraction_time": "number - extraction time"
  }
)
```

## Error Response Format

### 1. Standard Error Response

```json
{
  "error": true,
  "error_type": "string - error type",
  "error_message": "string - error details",
  "error_code": "string - error code (optional)",
  "timestamp": "string - error timestamp",
  "context": "object - error context information (optional)"
}
```

### 2. Common Error Types

```json
{
  "file_not_found": "PDF file does not exist",
  "pdf_processing_failed": "PDF processing failed",
  "grobid_service_error": "GROBID service error",
  "model_loading_failed": "Model loading failed",
  "citation_parsing_error": "Citation parsing error",
  "argument_classification_error": "Argument classification error",
  "invalid_input": "Invalid input parameters",
  "processing_timeout": "Processing timeout"
}
```

## Batch Processing Interface

### 1. Batch Document Processing

```python
def batch_process_documents(pdf_paths: List[str], options: Dict = None) -> List[Dict]
```

**Input Parameters:**
```json
{
  "pdf_paths": ["string - PDF file path list"],
  "options": {
    "save_results": "boolean - whether to save results",
    "enable_argument_classification": "boolean - whether to enable argument classification",
    "parallel_processing": "boolean - whether to process in parallel",
    "max_workers": "number - maximum worker threads"
  }
}
```

**Return Result:**
```json
[
  {
    "pdf_path": "string - file path",
    "success": "boolean - whether processing succeeded",
    "result": "object - processing result (if successful)",
    "error": "object - error information (if failed)",
    "processing_time": "number - processing time (seconds)"
  }
]
```

### 2. Batch Citation Analysis

```python
def batch_analyze_citations(sentences: List[str]) -> List[Dict]
```

## Configuration Interface

### 1. System Configuration

```python
class SystemConfig:
    def __init__(self):
        self.pdf_engines = ["pymupdf", "pdfplumber"]
        self.grobid_url = "http://localhost:8070"
        self.enable_ocr = True
        self.argument_classification = True
        self.cache_enabled = True
```

### 2. Model Configuration

```python
class ModelConfig:
    def __init__(self):
        self.model_path = "checkpoints/citation_classifier"
        self.device = "auto"  # auto, cpu, cuda, mps
        self.batch_size = 8
        self.max_length = 512
        self.confidence_threshold = 0.5
```

## Monitoring and Logging Interface

### 1. Processing Status Monitoring

```json
{
  "status": "string - processing status",
  "progress": "number - progress percentage (0-100)",
  "current_step": "string - current step description",
  "estimated_time_remaining": "number - estimated remaining time (seconds)",
  "processed_sentences": "number - number of processed sentences",
  "total_sentences": "number - total number of sentences"
}
```

### 2. Performance Metrics

```json
{
  "processing_time": {
    "pdf_extraction": "number - PDF extraction time",
    "sentence_parsing": "number - sentence parsing time", 
    "citation_analysis": "number - citation analysis time",
    "argument_classification": "number - argument classification time",
    "total": "number - total processing time"
  },
  "resource_usage": {
    "memory_peak": "number - peak memory usage (MB)",
    "cpu_utilization": "number - CPU utilization percentage",
    "gpu_utilization": "number - GPU utilization percentage (if applicable)"
  },
  "quality_metrics": {
    "pdf_quality_score": "number - PDF quality score",
    "citation_detection_rate": "number - citation detection rate",
    "argument_classification_confidence": "number - average argument classification confidence"
  }
}
```

## Type Definitions (TypeScript Style)

### 1. Core Types

```typescript
interface PaperMetadata {
  title: string;
  authors: string[];
  year: string;
  doi?: string;
  journal?: string;
  abstract?: string;
}

interface CitationMapping {
  intext: string;
  reference: ReferenceEntry;
  paper_id?: string;
  match_confidence: number;
  match_method: string;
}

interface ArgumentEntity {
  relation_type: string;
  start_pos: number;
  end_pos: number;
  text: string;
  confidence: number;
}

interface SentenceAnalysis {
  sentence_index: number;
  sentence_text: string;
  citations: CitationMapping[];
  argument_analysis?: ArgumentAnalysis;
  word_count: number;
  char_count: number;
}
```

### 2. Configuration Types

```typescript
interface ProcessingOptions {
  save_results?: boolean;
  enable_argument_classification?: boolean;
  preferred_pdf_engine?: string;
  grobid_timeout?: number;
  batch_size?: number;
}

interface QualityThresholds {
  min_pdf_score: number;
  min_citation_count: number;
  min_confidence: number;
}
``` 