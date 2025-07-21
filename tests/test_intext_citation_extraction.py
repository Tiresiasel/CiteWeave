import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.citation_parser import CitationParser
import json

def test_unicode_citations():
    """Test cases for Unicode character support in author names"""
    
    # Create a minimal parser instance for testing
    parser = CitationParser.__new__(CitationParser)
    parser.references = []  # We're only testing extraction, not matching
    
    test_cases = [
        # Turkish characters
        {
            "text": "Recent studies by Özcan (2020) show significant results.",
            "expected": ["(Özcan, 2020)"]
        },
        {
            "text": "This finding (Özcan and Güler, 2019) contradicts previous work.",
            "expected": ["(Özcan and Güler, 2019)"]
        },
        
        # French characters  
        {
            "text": "As noted by Dupéré (2021), the methodology is flawed.",
            "expected": ["(Dupéré, 2021)"]
        },
        {
            "text": "Previous research (Lévy and Cézar, 2018) supports this view.",
            "expected": ["(Lévy and Cézar, 2018)"]
        },
        
        # German characters
        {
            "text": "Müller's research (Müller, 2020) provides key insights.",
            "expected": ["(Müller, 2020)"]
        },
        {
            "text": "This approach (Weiß and Höfer, 2019) has proven effective.",
            "expected": ["(Weiß and Höfer, 2019)"]
        },
        
        # Spanish characters
        {
            "text": "According to Peña (2021), these results are significant.",
            "expected": ["(Peña, 2021)"]
        },
        {
            "text": "Studies show (Martínez and Núñez, 2020) clear benefits.",
            "expected": ["(Martínez and Núñez, 2020)"]
        },
        
        # Nordic characters
        {
            "text": "Research by Åberg (2019) confirms these findings.",
            "expected": ["(Åberg, 2019)"]
        },
        {
            "text": "This methodology (Björk and Sørensen, 2021) is widely used.",
            "expected": ["(Björk and Sørensen, 2021)"]
        },
        
        # Mixed Unicode characters
        {
            "text": "Collaborative work (Özcan, Müller, and Dupéré, 2020) demonstrates effectiveness.",
            "expected": ["(Özcan, Müller, and Dupéré, 2020)"]
        },
        
        # More specific Unicode test cases
        {
            "text": "The methodology proposed by Özcan (2022) has been widely adopted.",
            "expected": ["(Özcan, 2022)"]
        },
        {
            "text": "Research findings (Özcan et al., 2021) support this hypothesis.",
            "expected": ["(Özcan et al., 2021)"]
        },
        {
            "text": "Studies by Özcan and Müller (2020) provide comprehensive analysis.",
            "expected": ["(Özcan and Müller, 2020)"]
        },
        
        # Polish characters
        {
            "text": "According to Wójcik (2020), these results are significant.",
            "expected": ["(Wójcik, 2020)"]
        },
        {
            "text": "Previous work (Łukaszewski and Żółć, 2019) demonstrates this approach.",
            "expected": ["(Łukaszewski and Żółć, 2019)"]
        },
        
        # Czech characters  
        {
            "text": "Research by Novák (2021) confirms these findings.",
            "expected": ["(Novák, 2021)"]
        },
        {
            "text": "This methodology (Dvořák and Čech, 2020) is effective.",
            "expected": ["(Dvořák and Čech, 2020)"]
        },
        
        # Hungarian characters
        {
            "text": "The approach developed by Szabó (2019) shows promise.",
            "expected": ["(Szabó, 2019)"]
        },
        {
            "text": "Studies show (Kovács and Tóth, 2021) clear benefits.",
            "expected": ["(Kovács and Tóth, 2021)"]
        },
        
        # Russian/Cyrillic characters (transliterated)
        {
            "text": "According to Petrov (2020), this method is effective.",
            "expected": ["(Petrov, 2020)"]
        },
        
        # More complex Unicode combinations
        {
            "text": "Multi-national research (Özcan, Müller, and Dvořák, 2021) provides insights.",
            "expected": ["(Özcan, Müller, and Dvořák, 2021)"]
        },
        
        # Unicode with apostrophes and hyphens
        {
            "text": "The work by D'Ángelo-Müller (2020) is groundbreaking.",
            "expected": ["(D'Ángelo-Müller, 2020)"]
        },
        {
            "text": "Research findings (Jean-François and Müller, 2019) support this view.",
            "expected": ["(Jean-François and Müller, 2019)"]
        }
    ]
    
    print("Testing Unicode character support in citations...")
    print("=" * 60)
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        text = case["text"]
        expected = case["expected"]
        
        result = parser._extract_intext_citations(text)
        
        print(f"Test {i}: {text}")
        print(f"Expected: {expected}")
        print(f"Got: {result}")
        
        # Convert result to sorted list for comparison
        result_sorted = sorted(result) if result else []
        expected_sorted = sorted(expected) if expected else []
        
        if result_sorted == expected_sorted:
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL")
        print("-" * 40)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    # Run the existing tests first
    print("Running existing citation extraction tests...")
    
    # Create a minimal parser instance for testing
    parser = CitationParser.__new__(CitationParser)
    parser.references = []  # We're only testing extraction, not matching
    
    # Test cases covering various citation formats
    test_cases = [
        # Basic narrative citations
        {
            "text": "Smith (2020) argues that climate change is accelerating.",
            "expected": ["(Smith, 2020)"]
        },
        {
            "text": "According to Jones (2019), this approach is effective.",
            "expected": ["(Jones, 2019)"]
        },
        {
            "text": "In their groundbreaking study from the early 2000s, Smith and Wesson (2001) argued for a new approach.",
            "expected": ["(Smith and Wesson, 2001)"]
        },
        
        # Basic parenthetical citations
        {
            "text": "Climate change is accelerating (Smith, 2020).",
            "expected": ["(Smith, 2020)"]
        },
        {
            "text": "This approach has proven effective (Jones, 2019).",
            "expected": ["(Jones, 2019)"]
        },
        
        # Multiple authors - narrative
        {
            "text": "Smith and Jones (2020) demonstrate significant results.",
            "expected": ["(Smith and Jones, 2020)"]
        },
        {
            "text": "Research by Brown, Davis, and Wilson (2021) supports this view.",
            "expected": ["(Brown, Davis, and Wilson, 2021)"]
        },
        
        # Multiple authors - parenthetical
        {
            "text": "Significant results were found (Smith and Jones, 2020).",
            "expected": ["(Smith and Jones, 2020)"]
        },
        {
            "text": "This view is supported (Brown, Davis, and Wilson, 2021).",
            "expected": ["(Brown, Davis, and Wilson, 2021)"]
        },
        
        # Multiple citations in parentheses - semicolon separated
        {
            "text": "Several studies support this (Smith, 2020; Jones, 2019; Brown, 2021).",
            "expected": ["(Smith, 2020)", "(Jones, 2019)", "(Brown, 2021)"]
        },
        
        # Multiple citations in parentheses - comma separated (FIXED CASE)
        {
            "text": "This also helps to explain recurring efforts to classify strategies into a manageable number of generic types (Miles and Snow 1978, Porter 1980).",
            "expected": ["(Miles and Snow, 1978)", "(Porter, 1980)"]
        },
        
        # Citations with prefixes - should NOT preserve prefix
        {
            "text": "For example, previous research (e.g., Smith, 2020) shows clear benefits.",
            "expected": ["(Smith, 2020)"]
        },
        {
            "text": "This includes several approaches (cf. Jones, 2019; Brown, 2021).",
            "expected": ["(Jones, 2019)", "(Brown, 2021)"]
        },
        
        # Complex author names
        {
            "text": "Van Der Berg (2020) provides comprehensive analysis.",
            "expected": ["(Van Der Berg, 2020)"]
        },
        {
            "text": "Research by O'Connor and MacPherson (2021) confirms these findings.",
            "expected": ["(O'Connor and MacPherson, 2021)"]
        },
        
        # Citations with page numbers
        {
            "text": "This concept is well established (Smith, 2020, p. 45).",
            "expected": ["(Smith, 2020)"]
        },
        {
            "text": "As noted elsewhere (Jones, 2019, pp. 123-125), the evidence is clear.",
            "expected": ["(Jones, 2019)"]
        },
        
        # Square brackets
        {
            "text": "Recent analysis [Smith, 2020] suggests otherwise.",
            "expected": ["(Smith, 2020)"]
        },
        {
            "text": "Multiple studies [Jones, 2019; Brown, 2021] support this view.",
            "expected": ["(Jones, 2019)", "(Brown, 2021)"]
        },
        
        # Corporate/institutional authors
        {
            "text": "According to World Health Organization (2021), guidelines have changed.",
            "expected": ["(World Health Organization, 2021)"]
        },
        {
            "text": "New policies (United Nations, 2020) address these concerns.",
            "expected": ["(United Nations, 2020)"]
        },
        
        # Edge cases
        {
            "text": "The year 2020 was significant for research.",
            "expected": []  # Should not extract standalone year
        },
        {
            "text": "The project was completed in a single year (2023).",
            "expected": [] # Should not extract a year only
        },
        {
            "text": "This was discussed in the final report (see Table 4).",
            "expected": [] # Should not extract table references
        },
        {
            "text": "Smith wrote extensively about this topic.",
            "expected": []  # Should not extract author without year
        },
        {
            "text": "In 2020, research showed (incomplete citation",
            "expected": []  # Should handle malformed citations
        },
        
        # Mixed citation types in same text
        {
            "text": "While Smith (2020) argues for one approach, others disagree (Jones, 2019; Brown, 2021).",
            "expected": ["(Smith, 2020)", "(Jones, 2019)", "(Brown, 2021)"]
        },
        {
            "text": "Recent work by Jones (2019) and research by Åberg (2019) confirm this.",
            "expected": ["(Jones, 2019)", "(Åberg, 2019)"]
        },
        
        # Complex multi-author narrative citation (challenging case)
        {
            "text": "Smith, Jones, and Brown (2020) conducted extensive research on this topic.",
            "expected": ["(Smith, Jones, and Brown, 2020)"]
        },
        
        # Real data cases with page numbers and colons
        {
            "text": '"We also refer to the process of developing these novel replacements as business model innovation." Markides (2006: 20) "Business model innovation is the discovery of a fundamentally different business model in an existing business." Santos et al. (2009: 14)',
            "expected": ["(Markides, 2006)", "(Santos et al., 2009)"]
        },
        {
            "text": '"Business model innovation is a reconfiguration of activities in the existing business model of a firm that is new to the product service market in which the firm competes." Aspara et al. (2010: 47)',
            "expected": ["(Aspara et al., 2010)"]
        },
        {
            "text": '"Initiatives to create novel value by challenging existing industryspecific business models, roles and relations in certain geographical market areas." Gambardella and McGahan (2010: 263)',
            "expected": ["(Gambardella and McGahan, 2010)"]
        },
        {
            "text": '"Business-model innovation occurs when a firm adopts a novel approach to commercializing its underlying assets." Yunus et al. (2010: 312)',
            "expected": ["(Yunus et al., 2010)"]
        },
        {
            "text": '"Business model innovation is about generating new sources of profit by finding novel value proposition/value constellation combinations." Sorescu et al. (2011: S7)',
            "expected": ["(Sorescu et al., 2011)"]
        },
        {
            "text": 'Bucherer et al. (2012: 184) "We define business model innovation as a process that deliberately changes the core elements of a firm and its business logic."',
            "expected": ["(Bucherer et al., 2012)"]
        },
        {
            "text": 'Abdelkafi et al. (2013: 13) "A business model innovation happens when the company modifies or improves at least one of the value dimensions."',
            "expected": ["(Abdelkafi et al., 2013)"]
        },
        {
            "text": '"A BMI can thus be thought of as the introduction of a new business model aimed to create commercial value." Casadesus-Masanell and Zhu (2013: 464)',
            "expected": ["(Casadesus-Masanell and Zhu, 2013)"]
        },
        {
            "text": 'Corporate business model transformation is defined as "a change in the perceived logic of how value is created by the corporation, when it comes to the value-creating links among the corporation\'s portfolio of businesses, from one point of time to another." Berglund and Sandström (2013: 276)',
            "expected": ["(Berglund and Sandström, 2013)"]
        },
        {
            "text": '"repartitioning" (altering the boundaries of the firm), "relocation" (changing the location of units currently performing activities), or "relinking" (altering the linkages among the organizational units that perform activities; see also Amit & Zott, 2012, for an architectural definition).',
            "expected": ["(Amit & Zott, 2012)"]
        }
    ]
    
    print("=" * 60)
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        text = case["text"]
        expected = case["expected"]
        
        result = parser._extract_intext_citations(text)
        
        print(f"Test {i}: {text}")
        print(f"Expected: {expected}")
        print(f"Got: {result}")
        
        # Convert result to sorted list for comparison
        result_sorted = sorted(result) if result else []
        expected_sorted = sorted(expected) if expected else []
        
        if result_sorted == expected_sorted:
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL")
        print("-" * 40)
    
    print(f"\nExisting tests: {passed}/{total} tests passed")
    
    # Now run Unicode tests
    print("\n" + "=" * 60)
    unicode_passed = test_unicode_citations()
    
    if unicode_passed:
        print("\n🎉 All Unicode tests passed!")
    else:
        print("\n⚠️ Some Unicode tests failed.")