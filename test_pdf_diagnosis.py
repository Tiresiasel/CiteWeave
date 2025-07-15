#!/usr/bin/env python3
"""
PDF Diagnosis Tool
Tests and analyzes PDF documents to determine the best processing approach.
"""

import sys
import json
from src.pdf_processor import PDFProcessor

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_pdf_diagnosis.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        processor = PDFProcessor()
        
        print("=" * 60)
        print("PDF DIAGNOSIS REPORT")
        print("=" * 60)
        
        # Run comprehensive diagnosis
        diagnosis = processor.diagnose_pdf_and_recommend(pdf_path)
        
        # Print summary
        print(f"File: {diagnosis['pdf_path']}")
        print(f"Size: {diagnosis['file_size_mb']} MB")
        print(f"Type: {diagnosis['pdf_type']}")
        print(f"Pages: {diagnosis.get('total_pages', 'Unknown')}")
        print(f"Processable: {'✅ Yes' if diagnosis['is_processable'] else '❌ No'}")
        
        if 'recommended_engine' in diagnosis:
            print(f"Best Engine: {diagnosis['recommended_engine']} (score: {diagnosis['best_quality_score']:.2f})")
        
        print("\n" + "-" * 40)
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(diagnosis['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "-" * 40)
        print("ENGINE COMPARISON:")
        for engine, result in diagnosis['engine_results'].items():
            if result['success']:
                print(f"✅ {engine:15} | Score: {result['quality_score']:5.2f} | Text: {result['text_length']:,} chars")
            else:
                print(f"❌ {engine:15} | Error: {result['error']}")
        
        # If processable, show sample extraction
        if diagnosis['is_processable']:
            print("\n" + "-" * 40)
            print("SAMPLE EXTRACTION:")
            
            try:
                text, info = processor.extract_text_with_best_engine(pdf_path)
                sentences = processor.parse_sentences(pdf_path)
                
                print(f"Extracted {len(text)} characters")
                print(f"Parsed {len(sentences)} clean sentences")
                print("\nFirst 2 sentences:")
                for i, sent in enumerate(sentences[:2], 1):
                    print(f"{i}. {sent[:200]}...")
                    
            except Exception as e:
                print(f"Sample extraction failed: {e}")
        
        # Save detailed report
        report_path = pdf_path.replace('.pdf', '_diagnosis.json')
        with open(report_path, 'w') as f:
            json.dump(diagnosis, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")
        
    except Exception as e:
        print(f"Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 