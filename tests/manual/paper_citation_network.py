"""
Test Script for Paper Citation Network Implementation

This script tests the new paper-to-paper citation relationship functionality
in CiteWeave, including paper ID generation and citation network building.
"""

import os
import json
import logging

from src.utils.paper_id_utils import PaperIDGenerator
from src.storage.database_integrator import DatabaseIntegrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_paper_id_generation():
    """Test the paper ID generation functionality."""
    print("\n🧪 Testing Paper ID Generation")
    print("=" * 50)
    
    generator = PaperIDGenerator()
    
    # Test cases from our existing data
    test_cases = [
        {
            "title": "Imitation of Complex Strategies",
            "year": "2000",
            "authors": ["Jan W. Rivkin"],
            "expected_consistent": True
        },
        {
            "title": "Competitive Strategy",
            "year": "1980", 
            "authors": ["Michael E. Porter"],
            "expected_consistent": True
        },
        {
            "title": "An Evolutionary Theory of Economic Change",
            "year": "1982",
            "authors": ["Richard Nelson", "Sidney Winter"],
            "expected_consistent": True
        }
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"  Title: {case['title']}")
        print(f"  Year: {case['year']}")
        print(f"  Authors: {case['authors']}")
        
        # Generate ID multiple times to ensure consistency
        id1 = generator.generate_paper_id(case["title"], case["year"], case["authors"])
        id2 = generator.generate_paper_id(case["title"], case["year"], case["authors"])
        
        print(f"  Generated ID: {id1}")
        
        # Test consistency
        consistent = id1 == id2
        print(f"  Consistent: {'✅' if consistent else '❌'}")
        if not consistent:
            all_passed = False
        
        # Test validation
        valid = generator.validate_paper_id(id1)
        print(f"  Valid Format: {'✅' if valid else '❌'}")
        if not valid:
            all_passed = False
    
    # Test citation-based generation
    print("\n📚 Testing Citation-Based ID Generation:")
    citation_info = {
        "title": "Capitalism, Socialism, and Democracy",
        "year": "1942",
        "authors": ["J Schumpeter"]
    }
    
    citation_id = generator.generate_from_citation(citation_info)
    print(f"  Citation: {citation_info}")
    print(f"  Generated ID: {citation_id}")
    
    valid_citation = generator.validate_paper_id(citation_id)
    print(f"  Valid: {'✅' if valid_citation else '❌'}")
    if not valid_citation:
        all_passed = False
    
    return all_passed


def test_database_citation_network():
    """Test the database citation network functionality."""
    print("\n🔗 Testing Database Citation Network")
    print("=" * 50)
    
    try:
        # Initialize database integrator
        integrator = DatabaseIntegrator(
            config_path="config",
            storage_root="data/papers"
        )
        
        if not integrator.initialize_connections():
            print("❌ Failed to initialize database connections")
            return False
        
        # Get current network overview
        overview = integrator.get_citation_network_overview()
        
        if "error" in overview:
            print(f"❌ Error getting network overview: {overview['error']}")
            return False
        
        stats = overview["network_stats"]
        print("📊 Current Network Statistics:")
        print(f"  Total Papers: {stats['total_papers']}")
        print(f"  Uploaded Papers: {stats['uploaded_papers']}")
        print(f"  Stub Papers: {stats['stub_papers']}")
        print(f"  Citation Relations: {stats['total_citation_relations']}")
        print(f"  Citation Instances: {stats['total_citation_instances']}")
        
        # Show stub papers
        stub_papers = overview["stub_papers"]
        if stub_papers:
            print("\n🔗 Top Cited Stub Papers:")
            for i, stub in enumerate(stub_papers[:5]):
                print(f"  {i+1}. {stub['title']} ({stub['year']}) - Cited {stub['cited_by_count']} times")
                print(f"     ID: {stub['paper_id'][:16]}...")
        
        integrator.close_connections()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


def test_citation_data_analysis():
    """Analyze existing citation data to understand the structure."""
    print("\n📄 Analyzing Existing Citation Data")
    print("=" * 50)
    
    # Load the existing processed document
    data_dir = "data/papers/babcd89569ffe6cb373ed21a762c1799ace907d68f5cffa189e2d6be77af0504"
    sentences_file = os.path.join(data_dir, "sentences_with_citations.jsonl")
    
    if not os.path.exists(sentences_file):
        print(f"❌ Data file not found: {sentences_file}")
        return False
    
    cited_papers = {}
    total_citations = 0
    generator = PaperIDGenerator()
    
    # Analyze citations in the document
    with open(sentences_file, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                sentence_data = json.loads(line.strip())
                
                if sentence_data.get("has_citations", False):
                    for citation in sentence_data.get("citations", []):
                        reference = citation.get("reference", {})
                        
                        # Generate paper ID for the cited work
                        title = reference.get("title", "")
                        year = reference.get("year", "")
                        authors = reference.get("authors", [])
                        
                        if title and year:
                            paper_id = generator.generate_paper_id(title, year, authors)
                            
                            if paper_id not in cited_papers:
                                cited_papers[paper_id] = {
                                    "title": title,
                                    "year": year,
                                    "authors": authors,
                                    "citation_count": 0,
                                    "argument_relations": []
                                }
                            
                            cited_papers[paper_id]["citation_count"] += 1
                            total_citations += 1
                            
                            # Check for argument relations
                            arg_analysis = citation.get("argument_analysis", {})
                            if arg_analysis.get("has_argument_relations", False):
                                entities = arg_analysis.get("entities", [])
                                for entity in entities:
                                    relation_type = entity.get("relation_type", "UNKNOWN")
                                    confidence = entity.get("confidence", 0.0)
                                    cited_papers[paper_id]["argument_relations"].append({
                                        "relation_type": relation_type,
                                        "confidence": confidence
                                    })
                            
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error on line {line_num + 1}: {e}")
                continue
    
    print("📊 Citation Analysis Results:")
    print(f"  Total Citations Found: {total_citations}")
    print(f"  Unique Cited Papers: {len(cited_papers)}")
    
    # Show most cited papers
    sorted_papers = sorted(cited_papers.items(), 
                          key=lambda x: x[1]["citation_count"], 
                          reverse=True)
    
    print("\n📚 Most Cited Papers:")
    for i, (paper_id, info) in enumerate(sorted_papers[:10]):
        authors_str = ", ".join(info["authors"][:2])
        if len(info["authors"]) > 2:
            authors_str += " et al."
        
        arg_relations = len(info["argument_relations"])
        print(f"  {i+1}. {info['title'][:50]}{'...' if len(info['title']) > 50 else ''}")
        print(f"     Authors: {authors_str}")
        print(f"     Year: {info['year']} | Citations: {info['citation_count']} | Arg Relations: {arg_relations}")
        print(f"     Paper ID: {paper_id[:16]}...")
        
        # Show argument relation types
        if info["argument_relations"]:
            relation_types = {}
            for rel in info["argument_relations"]:
                rel_type = rel["relation_type"]
                if rel_type not in relation_types:
                    relation_types[rel_type] = 0
                relation_types[rel_type] += 1
            
            rel_summary = ", ".join([f"{k}: {v}" for k, v in relation_types.items()])
            print(f"     Relations: {rel_summary}")
        print()
    
    return True


def test_reimport_with_citation_network():
    """Test reimporting the existing document with citation network creation."""
    print("\n🔄 Testing Reimport with Citation Network")
    print("=" * 50)
    
    try:
        integrator = DatabaseIntegrator(
            config_path="config",
            storage_root="data/papers"
        )
        
        if not integrator.initialize_connections():
            print("❌ Failed to initialize database connections")
            return False
        
        # Get the paper ID of our existing document
        paper_id = "babcd89569ffe6cb373ed21a762c1799ace907d68f5cffa189e2d6be77af0504"
        
        print(f"🔄 Reimporting paper: {paper_id}")
        
        # Force reimport to ensure citation network is created
        success = integrator.import_document(paper_id, force_reimport=True)
        
        if success:
            print("✅ Reimport successful")
            
            # Get updated statistics
            overview = integrator.get_citation_network_overview()
            if "error" not in overview:
                stats = overview["network_stats"]
                print("\n📊 Updated Network Statistics:")
                print(f"  Total Papers: {stats['total_papers']}")
                print(f"  Uploaded Papers: {stats['uploaded_papers']}")
                print(f"  Stub Papers: {stats['stub_papers']}")
                print(f"  Citation Relations: {stats['total_citation_relations']}")
                print(f"  Citation Instances: {stats['total_citation_instances']}")
                
                # Show integration stats
                import_stats = integrator.get_import_status()
                print("\n📈 Import Statistics:")
                print(f"  Papers Processed: {import_stats['stats']['papers_processed']}")
                print(f"  Citations Stored: {import_stats['stats']['citations_stored']}")
                print(f"  Argument Relations: {import_stats['stats']['argument_relations_stored']}")
                print(f"  Paper Citations Created: {import_stats['stats']['paper_citations_created']}")
                
        else:
            print("❌ Reimport failed")
            return False
        
        integrator.close_connections()
        return True
        
    except Exception as e:
        print(f"❌ Reimport test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Paper Citation Network Implementation")
    print("=" * 70)
    
    tests = [
        ("Paper ID Generation", test_paper_id_generation),
        ("Citation Data Analysis", test_citation_data_analysis),
        ("Database Citation Network", test_database_citation_network),
        ("Reimport with Citation Network", test_reimport_with_citation_network)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Citation network implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main() 