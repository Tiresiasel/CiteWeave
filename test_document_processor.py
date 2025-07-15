#!/usr/bin/env python3
"""
Test script for the new DocumentProcessor integration.
Demonstrates unified sentence-level citation analysis.
"""

import sys
import logging
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor

def test_document_processor():
    """Test the DocumentProcessor with sentence-level citation analysis."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== DocumentProcessor Integration Test ===\n")
    
    # Initialize the document processor
    print("1. Initializing DocumentProcessor...")
    doc_processor = DocumentProcessor(storage_root="./test_data/papers/")
    
    # Test with an example PDF (you can replace this with an actual test file)
    pdf_path = "test_files/Porter - Competitive Strategy.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Warning: Test file {pdf_path} not found.")
        print("Please provide a valid PDF path to test the integration.")
        return
    
    try:
        # Step 1: Diagnose document quality
        print(f"\n2. Diagnosing document: {pdf_path}")
        diagnosis = doc_processor.diagnose_document_processing(pdf_path)
        
        print(f"   Quality Level: {diagnosis['overall_assessment']['quality_level']}")
        print(f"   Is Processable: {diagnosis['overall_assessment']['is_processable']}")
        
        if diagnosis['overall_assessment']['recommendations']:
            print("   Recommendations:")
            for rec in diagnosis['overall_assessment']['recommendations']:
                print(f"     - {rec}")
        
        # Step 2: Process the document for sentence-level citation analysis
        print(f"\n3. Processing document with sentence-level citation analysis...")
        results = doc_processor.process_document(pdf_path, save_results=True)
        
        # Display processing statistics
        stats = results['processing_stats']
        print(f"\n=== Processing Results ===")
        print(f"Paper ID: {results['paper_id']}")
        print(f"Document Title: {results['metadata'].get('title', 'Unknown')}")
        print(f"Total Sentences: {stats['total_sentences']}")
        print(f"Sentences with Citations: {stats['sentences_with_citations']}")
        print(f"Total Citations Found: {stats['total_citations']}")
        print(f"Total References: {stats['total_references']}")
        
        # Show example sentences with citations
        sentences_with_cites = [s for s in results['sentences_with_citations'] if s['citations']]
        
        if sentences_with_cites:
            print(f"\n=== Example Sentences with Citations ===")
            for i, sentence in enumerate(sentences_with_cites[:5]):  # Show first 5
                print(f"\n{i+1}. Sentence {sentence['sentence_index']}:")
                print(f"   Text: {sentence['sentence_text'][:150]}...")
                print(f"   Citations found: {len(sentence['citations'])}")
                
                for j, cite in enumerate(sentence['citations']):
                    ref = cite['reference']
                    print(f"     Citation {j+1}: {cite['intext']}")
                    print(f"       → Title: {ref.get('title', 'Unknown')[:60]}...")
                    print(f"       → Year: {ref.get('year', 'Unknown')}")
                    print(f"       → Authors: {', '.join(ref.get('authors', ['Unknown'])[:2])}")
                    if len(ref.get('authors', [])) > 2:
                        print(f"         (and {len(ref.get('authors', [])) - 2} more)")
        else:
            print("\nNo citations found in any sentences.")
        
        # Step 3: Test loading cached results
        print(f"\n4. Testing cached results loading...")
        cached_results = doc_processor.load_processed_document(pdf_path)
        if cached_results:
            print("   ✓ Successfully loaded cached results")
            print(f"   ✓ Cached results match: {len(cached_results['sentences_with_citations']) == len(results['sentences_with_citations'])}")
        else:
            print("   ✗ Failed to load cached results")
        
        # Step 4: Test the simplified interface
        print(f"\n5. Testing simplified interface...")
        simple_results = doc_processor.get_sentences_with_citations(pdf_path, force_reprocess=False)
        print(f"   ✓ Retrieved {len(simple_results)} sentences via simplified interface")
        
        print(f"\n=== Integration Test Completed Successfully ===")
        print(f"The new DocumentProcessor successfully coordinates PDFProcessor and CitationParser")
        print(f"to provide unified sentence-level citation analysis.")
        
    except Exception as e:
        print(f"\nError during document processing: {e}")
        import traceback
        traceback.print_exc()

def demo_architecture_benefits():
    """Demonstrate the benefits of the new architecture."""
    print(f"\n=== Architecture Benefits Demonstration ===")
    
    print(f"""
New DocumentProcessor Architecture Benefits:

1. **Unified Interface**: Single point of entry for document processing
   - doc_processor.process_document(pdf_path) → Complete sentence + citation analysis
   - Eliminates need to manually coordinate PDFProcessor and CitationParser

2. **Efficiency Optimizations**:
   - Shared metadata extraction (no duplicate PDF parsing)
   - Shared reference extraction (GROBID called only once)
   - Intelligent caching to avoid reprocessing

3. **Clear Data Flow**:
   PDFProcessor → extract sentences
        ↓
   CitationParser → analyze citations per sentence  
        ↓
   DocumentProcessor → unified results with both

4. **Robustness**:
   - Combined quality diagnosis for both PDF and citation processing
   - Graceful error handling at the sentence level
   - Comprehensive processing statistics

5. **Extensibility**:
   - Easy to add new processing steps (e.g., ArgumentClassifier)
   - Modular design allows individual component updates
   - Clear interfaces for integration with GraphBuilder, VectorIndexer, etc.

6. **优化的元数据提取策略**:
   - 优先使用 GROBID 提取 DOI
   - 使用 DOI 从 CrossRef 获取高质量元数据
   - 多层次降级策略确保鲁棒性

Recommended Usage Pattern:
```python
# Initialize once
doc_processor = DocumentProcessor()

# Process any PDF with unified sentence-level citation analysis
results = doc_processor.process_document("paper.pdf")

# Access structured results
for sent_data in results["sentences_with_citations"]:
    if sent_data["citations"]:
        print("Sentence:", sent_data['sentence_text'])
        for cite in sent_data["citations"]:
            print("  Citation:", cite['intext'], "→", cite['reference']['title'])
```
""")

if __name__ == "__main__":
    # Run the integration test
    test_document_processor()
    
    # Show architecture benefits
    demo_architecture_benefits() 