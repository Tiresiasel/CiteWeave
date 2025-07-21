#!/usr/bin/env python3
"""
Test script for PDF-based query functionality
Tests direct PDF content access and analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from multi_agent_research_system import LangGraphResearchSystem
from query_db_agent import QueryDBAgent
import json

def test_pdf_queries():
    """Test various PDF query functionalities"""
    
    print("🔍 Testing PDF Query Functionality")
    print("=" * 60)
    
    # Initialize system
    print("📝 Initializing system...")
    try:
        system = LangGraphResearchSystem()
        query_agent = QueryDBAgent()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # Get available papers
    print(f"\n📄 Available papers in directory:")
    papers_dir = os.path.join(os.path.dirname(__file__), "data", "papers")
    if os.path.exists(papers_dir):
        paper_ids = [d for d in os.listdir(papers_dir) if os.path.isdir(os.path.join(papers_dir, d))]
        for i, paper_id in enumerate(paper_ids, 1):
            print(f"{i}. {paper_id}")
    else:
        print("❌ Papers directory not found")
        return
    
    if not paper_ids:
        print("❌ No papers found in directory")
        return
    
    # Test 1: Get full PDF content
    print(f"\n🔬 Test 1: Getting full PDF content")
    print("-" * 40)
    test_paper_id = paper_ids[0]  # Use first available paper
    
    try:
        full_content = query_agent.get_full_pdf_content(test_paper_id)
        if full_content.get("found"):
            metadata = full_content.get("metadata", {})
            print(f"✅ Paper: {metadata.get('title', 'Unknown')}")
            print(f"📊 Authors: {', '.join(metadata.get('authors', []))}")
            print(f"📅 Year: {metadata.get('year', 'Unknown')}")
            print(f"📖 Sections: {full_content.get('sections_count', 0)}")
            print(f"📝 Word count: {full_content.get('total_word_count', 0)}")
            
            # Show section summaries
            print(f"\n📑 Section overview:")
            for section in full_content.get("section_summaries", [])[:3]:
                print(f"  - {section['section_title']} ({section['word_count']} words)")
                print(f"    Preview: {section['preview'][:100]}...")
        else:
            print(f"❌ Failed to get PDF content: {full_content.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error in full content test: {e}")
    
    # Test 2: Query specific content in PDF
    print(f"\n🔍 Test 2: Querying specific content in PDF")
    print("-" * 40)
    
    search_terms = ["strategy", "imitation", "complexity", "model"]
    
    for term in search_terms:
        try:
            content_result = query_agent.query_pdf_content(test_paper_id, term, context_window=300)
            if content_result.get("found"):
                print(f"✅ Found '{term}': {content_result.get('total_matches', 0)} matches")
                
                # Show first match
                if content_result.get("data"):
                    first_section = content_result["data"][0]
                    first_match = first_section["matches"][0]
                    context = first_match["context"]
                    print(f"   📝 Context: {context[:150]}...")
            else:
                print(f"❌ No matches found for '{term}'")
        except Exception as e:
            print(f"❌ Error searching for '{term}': {e}")
    
    # Test 3: Author-based PDF content search
    print(f"\n👤 Test 3: Author-based PDF content search")
    print("-" * 40)
    
    try:
        # Find papers by Porter and search for content
        author_content = query_agent.query_pdf_by_author_and_content("porter", "competitive advantage")
        if author_content.get("found"):
            print(f"✅ Found {author_content.get('papers_with_content', 0)} papers with relevant content")
            print(f"📊 Total papers by author: {author_content.get('papers_found', 0)}")
            
            # Show details for first paper
            if author_content.get("data"):
                first_result = author_content["data"][0]
                paper_metadata = first_result["paper_metadata"]
                content_matches = first_result["content_matches"]
                
                print(f"📄 Paper: {paper_metadata.get('title', 'Unknown')}")
                print(f"🔍 Matches: {content_matches.get('total_matches', 0)}")
        else:
            print(f"❌ No author content found: {author_content.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error in author content search: {e}")
    
    # Test 4: Semantic search within PDF
    print(f"\n🧠 Test 4: Semantic search within PDF")
    print("-" * 40)
    
    try:
        semantic_result = query_agent.semantic_search_pdf_content(
            test_paper_id, 
            "What factors make strategies difficult to imitate?", 
            similarity_threshold=0.3
        )
        
        if semantic_result.get("found"):
            print(f"✅ Semantic search successful!")
            print(f"🔍 Chunks searched: {semantic_result.get('total_chunks_searched', 0)}")
            print(f"📊 Relevant chunks: {semantic_result.get('relevant_chunks_found', 0)}")
            
            # Show top results
            for i, result in enumerate(semantic_result.get("data", [])[:2], 1):
                similarity = result["similarity_score"]
                content = result["content"]
                metadata = result["metadata"]
                
                print(f"\n🏆 Result {i} (Similarity: {similarity:.3f}):")
                print(f"📍 Section: {metadata['section_title']}")
                print(f"📝 Content: {content[:200]}...")
        else:
            print(f"❌ Semantic search failed: {semantic_result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error in semantic search: {e}")
    
    # Test 5: LangGraph integration with PDF queries
    print(f"\n🤖 Test 5: LangGraph system with PDF queries")
    print("-" * 40)
    
    test_questions = [
        "What does Porter's paper say about competitive strategy?",
        "Find specific examples of imitation barriers in Porter's work",
        "What are the main arguments in Rivkin's complexity paper?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        try:
            response = system.research_question(question)
            print(f"🤖 Response preview: {response[:300]}...")
            print(f"📏 Full response length: {len(response)} characters")
        except Exception as e:
            print(f"❌ Error in LangGraph query: {e}")
    
    print(f"\n🎉 PDF Query Testing Completed!")
    print("=" * 60)

def test_paper_availability():
    """Check what papers are available for testing"""
    
    print("📋 Checking available papers...")
    papers_dir = os.path.join(os.path.dirname(__file__), "data", "papers")
    
    if not os.path.exists(papers_dir):
        print(f"❌ Papers directory not found: {papers_dir}")
        return []
    
    paper_ids = []
    for item in os.listdir(papers_dir):
        paper_path = os.path.join(papers_dir, item)
        if os.path.isdir(paper_path):
            metadata_path = os.path.join(paper_path, "metadata.json")
            processed_doc_path = os.path.join(paper_path, "processed_document.json")
            
            if os.path.exists(metadata_path) and os.path.exists(processed_doc_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    print(f"✅ {item}")
                    print(f"   📄 Title: {metadata.get('title', 'Unknown')}")
                    print(f"   👤 Authors: {', '.join(metadata.get('authors', []))}")
                    print(f"   📅 Year: {metadata.get('year', 'Unknown')}")
                    
                    paper_ids.append(item)
                except Exception as e:
                    print(f"❌ Error reading metadata for {item}: {e}")
            else:
                print(f"⚠️  {item} - Missing required files")
    
    print(f"\n📊 Total available papers: {len(paper_ids)}")
    return paper_ids

if __name__ == "__main__":
    print("🚀 PDF Query Testing Suite")
    print("=" * 60)
    
    # First check available papers
    available_papers = test_paper_availability()
    
    if available_papers:
        print(f"\n✅ Found {len(available_papers)} papers, proceeding with tests...")
        test_pdf_queries()
    else:
        print("\n❌ No papers available for testing")
        print("Please ensure papers are properly processed and stored in data/papers/") 