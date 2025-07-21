"""
citation_parser.py
Module for detecting and parsing citations in text.
"""

import re
import difflib
import unicodedata
from typing import List, Dict, Optional
from PyPDF2 import PdfReader
import requests
from lxml import etree
import logging
import hashlib
from src.paper_id_utils import PaperIDGenerator

logging.basicConfig(level=logging.INFO)

class CitationParser:
    def __init__(self, pdf_path: str, full_doc_text: Optional[str] = None, references: Optional[List[str]] = None):
        """
        Initialize CitationParser with the path to the PDF file, optional full document text, and optional references.
        """
        self.pdf_path = pdf_path
        if full_doc_text is not None and isinstance(full_doc_text, bytes):
            full_doc_text = full_doc_text.decode("utf-8", errors="ignore")
        self.full_doc_text = full_doc_text
        self.references = references or self._extract_references_with_grobid()
        self.paper_id_generator = PaperIDGenerator()

    def _extract_references_with_grobid(self) -> List[Dict[str, str]]:
        with open(self.pdf_path, "rb") as f:
            files = {"input": (self.pdf_path, f, "application/pdf")}
            headers = {"Accept": "application/xml"}

            response = requests.post(
                "http://localhost:8070/api/processReferences",
                files=files,
                headers=headers
            )

        # Handle different GROBID response codes
        if response.status_code == 204:
            # No Content - PDF has no references or GROBID couldn't extract them
            logging.warning(f"GROBID returned 204 (No Content) for {self.pdf_path} - no references found")
            return []
        elif response.status_code != 200:
            # Other errors - log warning but don't crash
            logging.warning(f"GROBID failed with status {response.status_code} for {self.pdf_path}")
            return []

        try:
            # Check if response has content
            if not response.content.strip():
                logging.warning(f"GROBID returned empty content for {self.pdf_path}")
                return []
            
            root = etree.fromstring(response.content)
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}
            references = []

            for bibl in root.findall(".//tei:biblStruct", namespaces=ns):
                # Authors
                authors = []
                for author in bibl.findall(".//tei:author", namespaces=ns):
                    pers_name = author.find(".//tei:persName", namespaces=ns)
                    if pers_name is not None:
                        forename = pers_name.findtext("tei:forename", default="", namespaces=ns)
                        surname = pers_name.findtext("tei:surname", default="", namespaces=ns)
                        full_name = f"{forename} {surname}".strip()
                        if full_name:
                            authors.append(full_name)

                # Title
                title = bibl.findtext(".//tei:title", default="", namespaces=ns).strip()

                # Year
                date = bibl.find(".//tei:date", namespaces=ns)
                year = date.get("when") if date is not None else ""

                # Journal (may appear as monogr/title)
                journal = bibl.findtext(".//tei:monogr/tei:title", default="", namespaces=ns).strip()

                # Publisher
                publisher = bibl.findtext(".//tei:monogr/tei:imprint/tei:publisher", default="", namespaces=ns).strip()

                # Place of publication
                pub_place = bibl.findtext(".//tei:monogr/tei:imprint/tei:pubPlace", default="", namespaces=ns).strip()

                # Combine raw string
                raw = f"{'; '.join(authors)} ({year}). {title}"
                if journal:
                    raw += f". {journal}"
                if publisher or pub_place:
                    raw += f". {publisher}" + (f", {pub_place}" if pub_place else "")

                references.append({
                    "authors": authors,
                    "title": title,
                    "year": year,
                    "journal": journal,
                    "publisher": publisher,
                    "pub_place": pub_place,
                    "raw_text": raw.strip()
                })

            return references

        except Exception as e:
            raise RuntimeError(f"Failed to parse GROBID XML: {e}")

    def match_intext_to_reference(self, intext: str) -> Optional[dict]:
        """
        Match in-text citation to full reference using strict surname + year match.
        Only exact surname and year match is allowed.
        """
        def _normalize_name(name: str) -> str:
            # Only keep letters and common name characters, remove extra punctuation
            normalized = re.sub(r'[^\w\s\-\']', '', name.lower().split()[-1], flags=re.UNICODE)
            return normalized

        def _parse_intext(intext: str):
            txt = intext.lower().replace('et al.', '')
            letter_pattern = self._create_unicode_letter_pattern()
            pattern = rf'({letter_pattern}+(?:[\s\-\']{letter_pattern}+)*).+?(\d{{4}})'
            match = re.search(pattern, txt)
            if not match:
                return None, None
            surname = _normalize_name(match.group(1))
            year = match.group(2)
            return surname, year

        surname, year = _parse_intext(intext)
        if not surname or not year:
            return None

        for ref in self.references:
            ref_year = ref.get("year") or ""
            ref_authors = ref.get("authors") or []
            author_keys = {_normalize_name(author) for author in ref_authors}
            if year == ref_year and surname in author_keys:
                return ref

        return None
    
    
    def _extract_intext_citations(self, sentence: str) -> List[str]:
        """
        Extract in-text citations from a sentence with comprehensive pattern matching.
        Supports narrative (e.g., "Smith (2020)") and parenthetical (e.g., "(Smith, 2020; Johnson et al., 2019)") formats.
        Returns cleaned citation strings.
        """
        citations = set()

        # --- Pattern 1: Narrative citations ---
        # Enhanced patterns with support for multi-word names like "Van Der Berg" and "World Health Organization"
        letter_char = self._create_unicode_letter_pattern()[1:-1]
        
        # Name component patterns: supports single words, multi-word names, hyphens, apostrophes
        basic_name_chars = f"[{letter_char}'\\.\\-]"
        # Multi-word name: one or more name parts separated by spaces
        name_pattern = f"{basic_name_chars}+(?:\\s+{basic_name_chars}+)*"
        
        # Define patterns with different prefixes
        # Pattern 1: Common sentence-starting prefixes
        sentence_prefix_patterns = [
            rf"(?:^|\.\s+)(?:according\s+to\s+|while\s+)({name_pattern})\s+\((\d{{4}}[a-z]?)\)",
            rf"(?:^|\.\s+)(?:according\s+to\s+|while\s+)({name_pattern}(?:,\s*{name_pattern})*,\s*(?:and|&)\s*{name_pattern})\s+\((\d{{4}}[a-z]?)\)",
            rf"(?:^|\.\s+)(?:according\s+to\s+|while\s+)({name_pattern}\s+(?:and|&)\s+{name_pattern})\s+\((\d{{4}}[a-z]?)\)",
            rf"(?:^|\.\s+)(?:according\s+to\s+|while\s+)({name_pattern}(?:\s+et\s+al\.?))\s+\((\d{{4}}[a-z]?)\)",
        ]
        
        # Pattern 2: "X by Y" patterns that can appear anywhere
        by_prefix_patterns = [
            rf"(?:research|work|study|studies|analysis|findings?|recent\s+work|recent\s+studies?|as\s+noted)\s+by\s+({name_pattern})\s+\((\d{{4}}[a-z]?)\)",
            rf"(?:research|work|study|studies|analysis|findings?|recent\s+work|recent\s+studies?|as\s+noted)\s+by\s+({name_pattern}(?:,\s*{name_pattern})*,\s*(?:and|&)\s*{name_pattern})\s+\((\d{{4}}[a-z]?)\)",
            rf"(?:research|work|study|studies|analysis|findings?|recent\s+work|recent\s+studies?|as\s+noted)\s+by\s+({name_pattern}\s+(?:and|&)\s+{name_pattern})\s+\((\d{{4}}[a-z]?)\)",
            rf"(?:research|work|study|studies|analysis|findings?|recent\s+work|recent\s+studies?|as\s+noted)\s+by\s+({name_pattern}(?:\s+et\s+al\.?))\s+\((\d{{4}}[a-z]?)\)",
        ]
        
        # Pattern 3: "The X by Y" patterns for longer phrases
        the_by_patterns = [
            rf"(?:the\s+(?:methodology|approach|work|study|analysis|findings?)\s+(?:proposed|developed|conducted)\s+by)\s+({name_pattern})\s+\((\d{{4}}[a-z]?)\)",
            rf"(?:the\s+(?:methodology|approach|work|study|analysis|findings?)\s+(?:proposed|developed|conducted)\s+by)\s+({name_pattern}(?:,\s*{name_pattern})*,\s*(?:and|&)\s*{name_pattern})\s+\((\d{{4}}[a-z]?)\)",
            rf"(?:the\s+(?:methodology|approach|work|study|analysis|findings?)\s+(?:proposed|developed|conducted)\s+by)\s+({name_pattern}\s+(?:and|&)\s+{name_pattern})\s+\((\d{{4}}[a-z]?)\)",
            rf"(?:the\s+(?:methodology|approach|work|study|analysis|findings?)\s+(?:proposed|developed|conducted)\s+by)\s+({name_pattern}(?:\s+et\s+al\.?))\s+\((\d{{4}}[a-z]?)\)",
        ]
        
        # Pattern 4: Standard patterns without prefixes
        standard_patterns = [
            rf"\b({name_pattern}(?:,\s*{name_pattern})*,\s*(?:and|&)\s*{name_pattern})\s+\((\d{{4}}[a-z]?)\)",
            rf"\b({name_pattern}(?:\s+et\s+al\.?))\s+\((\d{{4}}[a-z]?)\)",
            rf"\b({name_pattern}\s+(?:and|&)\s+{name_pattern})\s+\((\d{{4}}[a-z]?)\)",
            rf"\b({name_pattern})\s+\((\d{{4}}[a-z]?)\)",
        ]

        all_patterns = sentence_prefix_patterns + by_prefix_patterns + the_by_patterns + standard_patterns
        matched_spans = set()  # Track matched text spans to avoid duplicates
        
        for pattern in all_patterns:
            for match in re.finditer(pattern, sentence, re.UNICODE | re.IGNORECASE):
                start, end = match.span()
                # More precise overlap check: only avoid if there's significant overlap
                # Allow citations that only slightly overlap (like consecutive citations)
                overlap = any(
                    (start < prev_end - 5 and end > prev_start + 5) 
                    for prev_start, prev_end in matched_spans
                )
                
                if not overlap:
                    author = match.group(1).strip()
                    year = match.group(2).strip()
                    
                    if self._is_valid_author_name(author):
                        cleaned_author = re.sub(r'\s+', ' ', author)
                        citations.add(f"({cleaned_author}, {year})")
                        matched_spans.add((start, end))

        # --- Pattern 2: Parenthetical citations ---
        # Find all parenthetical/bracket expressions
        paren_patterns = [
            r'\(([^)]+)\)',  # Regular parentheses
            r'\[([^\]]+)\]'  # Square brackets
        ]
        
        for pattern in paren_patterns:
            for match in re.finditer(pattern, sentence):
                content = match.group(1).strip()
                # Parse the content inside parentheses/brackets
                extracted_citations = self._parse_parenthetical_content(content)
                citations.update(extracted_citations)

        return list(citations)

    def _is_letter(self, char: str) -> bool:
        """Check if a character is a letter using Unicode categories."""
        return unicodedata.category(char).startswith('L')
    
    def _is_uppercase(self, char: str) -> bool:
        """Check if a character is uppercase using Unicode categories."""
        return unicodedata.category(char) == 'Lu'
    
    def _is_lowercase(self, char: str) -> bool:
        """Check if a character is lowercase using Unicode categories."""
        return unicodedata.category(char) == 'Ll'
    
    def _create_unicode_letter_pattern(self) -> str:
        """Create a regex pattern that matches any Unicode letter."""
        # Simple but comprehensive pattern for most European languages
        return r'[a-zA-ZÀ-ÿĀ-žƀ-ǿḀ-ỿ]'

    def _is_valid_author_name(self, author: str) -> bool:
        """
        Validate if a string looks like a valid author name.
        Filters out common false positives.
        """
        author = author.strip()
        
        words = author.split()
        if not words:
            return False

        # Check if the first word is a common non-author prefix/word
        first_word = words[0].lower().strip('.,')
        stop_words = {
            'the', 'a', 'an', 'by', 'in', 'on', 'at', 'of', 'for', 'with', 'from', 'to', 
            'year', 'data', 'study', 'research', 'work', 'paper', 'article', 'results', 'result',
            'findings', 'finding', 'figure', 'table', 'chapter', 'section', 'appendix',
            'e.g', 'i.e', 'see', 'also', 'note', 'equation', 'fig', 'tbl', 'et', 'al', 'vs', 
            'versus', 'cf', 'according', 'recent', 'and', 'as', 'based',
            # Added words from failed tests
            'while', 'noted', 'methodology', 'approach', 'proposed', 'developed'
        }
        if first_word in stop_words:
            return False
        
        # Starts with common non-author phrases (redundant but safe)
        non_author_prefixes = r'^(research by|study by|analysis by|data from|results? from|findings? from|work by|paper by|article by|the year|year|recent studies by|as noted by|according to|the methodology proposed by|studies by|the approach developed by|the work by|as cited in|as seen in|based on|in the work of)\b'
        if re.match(non_author_prefixes, author, re.IGNORECASE):
            return False
        
        # Only digits or mostly digits
        if re.match(r'^\d+$', author) or len(re.findall(r'\d', author)) > len(author) / 2:
            return False
        
        # Must contain at least one letter (using Unicode-aware check)
        if not any(self._is_letter(char) for char in author):
            return False

        # Reject if it ends with " et" but not "et al."
        if author.lower().endswith(" et"):
            return False
            
        # Casing rules: Must have an uppercase letter, unless it's a short (<=5) all-caps acronym.
        is_all_caps = all(self._is_uppercase(c) or not self._is_letter(c) for c in author)
        has_any_uppercase = any(self._is_uppercase(c) for c in author)
        
        if is_all_caps and len(author) > 5 and ' ' not in author:
            # Probably not an acronym, e.g. "ACCORDING"
            return False
        
        if not is_all_caps and not has_any_uppercase:
            # Not an acronym and no uppercase letters, e.g. "smith"
            return False
        
        return True

    def _parse_parenthetical_content(self, content: str) -> List[str]:
        """
        Parse content inside parentheses/brackets to extract individual citations.
        Handles complex cases like: "e.g., Miles and Snow 1978, Porter 1980"
        """
        citations = []
        
        # Check for common prefixes that should be preserved with citations
        prefix_pattern = r'^(see|e\.g\.?,?|cf\.?|i\.e\.?,?|also|for example)\s+(.+)$'
        prefix_match = re.match(prefix_pattern, content, re.IGNORECASE)
        
        if prefix_match:
            # Remove prefix and process normally (don't preserve prefix)
            remaining_content = prefix_match.group(2).strip()
            
            # Handle different separators in the remaining content
            if ';' in remaining_content:
                # First split by semicolon, then handle commas within each part
                semicolon_parts = [part.strip() for part in remaining_content.split(';')]
                for semicolon_part in semicolon_parts:
                    if ',' in semicolon_part and not re.search(r'[A-ZÀ-ÿ][A-Za-zÀ-ÿ\s\-\'\.&,]*,\s*\d{4}', semicolon_part):
                        # This part has commas but not in "Author, Year" format - split by comma
                        comma_parts = self._split_comma_separated_citations(semicolon_part)
                        for comma_part in comma_parts:
                            part_citations = self._extract_citation_from_part(comma_part.strip())
                            citations.extend(part_citations)
                    else:
                        # Treat as single citation or already in "Author, Year" format
                        part_citations = self._extract_citation_from_part(semicolon_part.strip())
                        citations.extend(part_citations)
            else:
                parts = self._split_comma_separated_citations(remaining_content)
                for part in parts:
                    if not part.strip():
                        continue
                    part_citations = self._extract_citation_from_part(part.strip())
                    citations.extend(part_citations)
        else:
            # No prefix found, handle normally
            # Handle different separators: semicolon, comma with year, and others
            if ';' in content:
                # First split by semicolon, then handle commas within each part
                semicolon_parts = [part.strip() for part in content.split(';')]
                for semicolon_part in semicolon_parts:
                    if ',' in semicolon_part and not re.search(r'[A-ZÀ-ÿ][A-Za-zÀ-ÿ\s\-\'\.&,]*,\s*\d{4}', semicolon_part):
                        # This part has commas but not in "Author, Year" format - split by comma
                        comma_parts = self._split_comma_separated_citations(semicolon_part)
                        for comma_part in comma_parts:
                            part_citations = self._extract_citation_from_part(comma_part.strip())
                            citations.extend(part_citations)
                    else:
                        # Treat as single citation or already in "Author, Year" format
                        part_citations = self._extract_citation_from_part(semicolon_part.strip())
                        citations.extend(part_citations)
            else:
                parts = self._split_comma_separated_citations(content)
                for part in parts:
                    if not part.strip():
                        continue
                    part_citations = self._extract_citation_from_part(part.strip())
                    citations.extend(part_citations)
        
        return citations

    def _split_comma_separated_citations(self, content: str) -> List[str]:
        """
        Intelligently split comma-separated citations.
        Handles cases like: "Miles and Snow 1978, Porter 1980"
        """
        # Pattern to identify where one citation ends and another begins
        # Look for: Author YEAR, Author YEAR with comprehensive Unicode support
        letter_pattern = self._create_unicode_letter_pattern()
        citation_boundary_pattern = rf'(\d{{4}}[a-z]?),\s*({letter_pattern})'
        
        # Find all boundaries
        boundaries = []
        for match in re.finditer(citation_boundary_pattern, content):
            boundaries.append(match.start(2))  # Start of next citation
        
        if not boundaries:
            # No clear boundaries found, treat as single citation
            return [content]
        
        # Split at boundaries
        parts = []
        start = 0
        for boundary in boundaries:
            parts.append(content[start:boundary].rstrip(', '))
            start = boundary
        parts.append(content[start:])  # Last part
        
        return [part.strip() for part in parts if part.strip()]

    def _extract_citation_from_part(self, part: str) -> List[str]:
        """
        Extract citation from a single part.
        Handles various formats within one citation string.
        """
        part = part.strip()
        if not part:
            return []
        
        citations = []
        
        # Pattern 1: Standard "Author, Year" format - improved for Unicode and complex names
        letter_pattern = self._create_unicode_letter_pattern()
        standard_pattern = rf'^({letter_pattern}+(?:\s+{letter_pattern}+)*(?:\s+et\s+al\.?)?),\s*(\d{{4}}[a-z]?)$'
        match = re.match(standard_pattern, part)
        if match:
            author = re.sub(r'\s+', ' ', match.group(1).strip())
            year = match.group(2).strip()
            citations.append(f"({author}, {year})")
            return citations
        
        # Pattern 2: "Author Year" format (no comma) - improved for Unicode and complex names  
        no_comma_pattern = rf'^({letter_pattern}+(?:\s+{letter_pattern}+)*(?:\s+et\s+al\.?)?)\s+(\d{{4}}[a-z]?)$'
        match = re.match(no_comma_pattern, part)
        if match:
            author = re.sub(r'\s+', ' ', match.group(1).strip())
            year = match.group(2).strip()
            citations.append(f"({author}, {year})")
            return citations
        
        # Pattern 3: Multiple authors with single year at the end - improved
        multi_author_pattern = r'^(.+?)\s+(\d{4}[a-z]?)$'
        match = re.match(multi_author_pattern, part)
        if match:
            # Strip trailing commas from the author part to prevent duplication
            authors_part = match.group(1).strip().rstrip(',').strip()
            year = match.group(2).strip()
            
            # Check if this looks like multiple authors
            if ' and ' in authors_part or ' & ' in authors_part or ',' in authors_part:
                author = re.sub(r'\s+', ' ', authors_part)
                # Final check for validity before adding
                if self._is_valid_author_name(author):
                    citations.append(f"({author}, {year})")
                    return citations
        
        # Pattern 4: Handle "et al." cases - improved for Unicode
        et_al_pattern = rf'^({letter_pattern}+(?:\s+{letter_pattern}+)*\s+et\s+al\.?),\s*(\d{{4}}[a-z]?)$'
        match = re.match(et_al_pattern, part)
        if match:
            author = match.group(1).strip()
            year = match.group(2).strip()
            citations.append(f"({author}, {year})")
            return citations
        
        # Pattern 5: Year only (should be ignored as incomplete)
        if re.match(r'^\d{4}[a-z]?$', part):
            return []
        
        # Pattern 6: Complex cases - try to find any author-year combination - improved for Unicode
        complex_pattern = r'([A-ZÀ-ÿ][A-Za-zÀ-ÿ\s\-\'\.&,]*?(?:et\s+al\.?)?)\s*,?\s*(\d{4}[a-z]?)'
        matches = list(re.finditer(complex_pattern, part))
        
        if matches:
            for match in matches:
                author = re.sub(r'\s+', ' ', match.group(1).strip().rstrip(',')).strip()
                year = match.group(2).strip()
                if self._is_valid_author_name(author):
                    citations.append(f"({author}, {year})")
        
        return citations

    def _parse_citation_part(self, part: str) -> List[str]:
        """
        Legacy method - redirects to new implementation for backward compatibility.
        """
        return self._extract_citation_from_part(part)
    
    def _generate_paper_id(self, title: str, year: str) -> str:
        """
        Generate a unique paper ID using the unified PaperIDGenerator.
        """
        return self.paper_id_generator.generate_paper_id(title, year)

    def parse_sentence(self, sentence: str) -> List[Dict[str, str]]:
        """
        Extract all in-text citations from a sentence and match them to reference entries.
        Returns list of mappings: [{intext: reference}]
        """
        mappings = []
        intext_citations = self._extract_intext_citations(sentence)
        for intext in intext_citations:
            matched_ref = self.match_intext_to_reference(intext)
            if matched_ref:
                paper_id = self._generate_paper_id(
                    matched_ref["title"], 
                    matched_ref["year"]
                )
                mappings.append({"intext": intext, "reference": matched_ref, "paper_id": paper_id})
        return mappings

    def parse_document(self, sentences: List[str]) -> List[Dict]:
        """
        Process a list of sentences and extract citation mappings for each.
        Returns list of:
        {
            sentence: str,
            citations: [{intext, reference}]
        }
        """
        results = []
        for sentence in sentences:
            mappings = self.parse_sentence(sentence)
            if mappings:
                results.append({
                    "sentence": sentence,
                    "citations": mappings
                })
        return results
    


if __name__ == "__main__":
    # Example usage
    pdf_path = "test_files/Rivkin - 2000 - Imitation of Complex Strategies.pdf"
    pdf_reader = PdfReader(pdf_path)
    pdf_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
    parser = CitationParser(pdf_path)
    reference_section = parser._extract_references_with_grobid()
    print(reference_section)

    print(parser.references)
    test_sentence = "In their discussion of \u201c\ufb01t\u201d\nand complementarity, Milgrom and Roberts (1995)suggest informally that rich interactions among Lin-coln Electric \u2019s many choices may explain why rivals\nhave not replicated that \ufb01rm\u2019s well-documented suc-\ncess.."
    result = parser.parse_sentence(test_sentence)

    from pprint import pprint
    pprint(result)