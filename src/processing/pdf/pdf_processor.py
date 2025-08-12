import os
import re
import shutil
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import requests
from lxml import etree
import logging
import hashlib
import io
import ast

# LLM manager for fallback metadata extraction
from src.llm.enhanced_llm_manager import EnhancedLLMManager
# Multiple PDF parsing engines with fallback support
try:
    import fitz  # pymupdf - best for complex PDFs
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber  # excellent for tables and layout
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from PyPDF2 import PdfReader  # fallback option
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pytesseract  # OCR for scanned PDFs
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

logging.basicConfig(level=logging.INFO)

class PDFProcessor:
    def __init__(self, storage_root: str = "./data/papers/", preferred_engine: str = "auto"):
        self.storage_root = storage_root
        os.makedirs(self.storage_root, exist_ok=True)
        self.preferred_engine = preferred_engine
        
        # Determine available engines and set priority
        self.available_engines = []
        if HAS_PYMUPDF:
            self.available_engines.append(("pymupdf", self._extract_text_with_pymupdf))
            if HAS_OCR:
                self.available_engines.append(("pymupdf_ocr", self._extract_text_with_pymupdf_ocr))
        if HAS_PDFPLUMBER:
            self.available_engines.append(("pdfplumber", self._extract_text_with_pdfplumber))
        if HAS_PYPDF2:
            self.available_engines.append(("pypdf2", self._extract_text_with_pypdf2))
            
        if not self.available_engines:
            raise RuntimeError("No PDF parsing engines available. Install pymupdf, pdfplumber, or PyPDF2")
        
        logging.info(f"Available PDF engines: {[name for name, _ in self.available_engines]}")
        if HAS_OCR:
            logging.info("OCR support available for scanned documents")

    def _generate_paper_id(self, title: str, year: str) -> str:
        """
        Generate a unique paper ID based on the SHA256 hash of the combined paper name and year.
        """
        title = re.sub(r"[^\w\s]", "", title)       # remove punctuation
        title = re.sub(r"\s+", " ", title).strip()  # normalize whitespace
        combined_str = f"{title}_{year}"
        combined_str = combined_str.lower() # lower_case
        hash_sha256 = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()
        return hash_sha256

    def register_pdf(self, pdf_path: str, metadata: Optional[Dict] = None) -> str:
        """
        Save PDF to project structure and store its metadata.
        The paper ID is always generated from the PDF content hash.
        """

        extracted = self.extract_pdf_metadata(pdf_path)
        metadata = metadata or {}

        final_meta = {k: metadata.get(k, extracted.get(k)) for k in extracted}
        paper_id = self._generate_paper_id(final_meta["title"], final_meta["year"])
        final_meta["paper_id"] = paper_id
        final_meta["filename"] = "original.pdf"
        final_meta["upload_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        dest_dir = os.path.join(self.storage_root, paper_id)
        os.makedirs(dest_dir, exist_ok=True)

        new_pdf_path = os.path.join(dest_dir, "original.pdf")
        shutil.copyfile(pdf_path, new_pdf_path)


        with open(os.path.join(dest_dir, "metadata.json"), "w") as f:
            json.dump(final_meta, f, indent=2)

        return new_pdf_path

    def extract_pdf_metadata(self, pdf_path: str) -> Dict:
        """
        Extract metadata using optimized strategy: GROBID for DOI first, then CrossRef for high-quality metadata.
        """
        logging.info(f"Starting metadata extraction for {pdf_path}")
        
        # Strategy 1: GROBID for DOI extraction (most reliable for academic papers)
        doi = None
        grobid_metadata = {}
        
        try:
            logging.info("Attempting DOI extraction with GROBID...")
            grobid_metadata = self._extract_metadata_with_grobid(pdf_path)
            doi = grobid_metadata.get("doi")
            
            if doi and doi != "Unknown DOI":
                logging.info(f"Successfully extracted DOI: {doi}")
                
                # Strategy 2: Use CrossRef API with the DOI for high-quality metadata
                try:
                    logging.info(f"Fetching high-quality metadata from CrossRef using DOI: {doi}")
                    crossref_metadata = self._fetch_metadata_from_crossref(doi)
                    
                    if self._validate_metadata_quality(crossref_metadata):
                        logging.info("Successfully obtained high-quality metadata from CrossRef")
                        crossref_metadata["extraction_method"] = "CrossRef_via_GROBID_DOI"
                        crossref_metadata["doi"] = doi  # Ensure DOI is preserved
                        return crossref_metadata
                    else:
                        logging.warning("CrossRef metadata quality validation failed")
                        
                except Exception as e:
                    logging.warning(f"CrossRef API failed for DOI {doi}: {e}")
            else:
                logging.info("No DOI found in GROBID extraction")
                
        except Exception as e:
            logging.warning(f"GROBID metadata extraction failed: {e}")
        
        # Strategy 3: Fallback to GROBID metadata if available and valid
        if grobid_metadata and self._validate_metadata_quality(grobid_metadata):
            logging.info("Using GROBID metadata as fallback")
            grobid_metadata["extraction_method"] = "GROBID"
            return grobid_metadata
        
        # Strategy 4: Fallback to PyPDF2
        try:
            logging.info("Attempting metadata extraction with PyPDF2...")
            pypdf2_metadata = self._extract_metadata_with_pypdf2(pdf_path)
            
            if self._validate_metadata_quality(pypdf2_metadata):
                logging.info("Successfully extracted metadata using PyPDF2")
                pypdf2_metadata["extraction_method"] = "PyPDF2"
                return pypdf2_metadata
            else:
                logging.warning("PyPDF2 metadata quality validation failed")
                
        except Exception as e:
            logging.warning(f"PyPDF2 extraction failed: {e}")
        
        # Strategy 5: LLM fallback using first three pages text
        try:
            logging.info("Attempting metadata extraction with LLM from first three pages...")
            llm_metadata = self._extract_metadata_with_llm(pdf_path, max_pages=3)
            if llm_metadata and self._validate_metadata_quality(llm_metadata):
                logging.info("Successfully extracted metadata using LLM (first three pages)")
                llm_metadata["extraction_method"] = "LLM_First3Pages"
                # If DOI present, try to enrich via CrossRef but keep LLM as fallback
                doi_candidate = llm_metadata.get("doi")
                if doi_candidate and doi_candidate not in ("Unknown DOI", "", None):
                    try:
                        logging.info(f"Enriching LLM metadata via CrossRef for DOI: {doi_candidate}")
                        crossref_from_llm = self._fetch_metadata_from_crossref(doi_candidate)
                        if self._validate_metadata_quality(crossref_from_llm):
                            crossref_from_llm["extraction_method"] = "CrossRef_via_LLM_DOI"
                            crossref_from_llm["doi"] = doi_candidate
                            return crossref_from_llm
                    except Exception as e:
                        logging.warning(f"CrossRef enrichment failed for LLM DOI {doi_candidate}: {e}")
                return llm_metadata
            else:
                logging.warning("LLM metadata quality validation failed")
        except Exception as e:
            logging.warning(f"LLM-based metadata extraction failed: {e}")
        
        # Strategy 6: Last resort - filename-based metadata
        logging.warning("All primary metadata extraction methods failed, using filename fallback")
        filename_metadata = self._extract_metadata_from_filename(pdf_path)
        filename_metadata["extraction_method"] = "Filename_Fallback"
        
        return filename_metadata

    def _extract_metadata_with_llm(self, pdf_path: str, max_pages: int = 3) -> Dict:
        """
        Use a large language model to extract metadata from the first N pages of a PDF.
        Returns a dict with keys: title, authors (list[str]), year, doi, journal, publisher, abstract.
        """
        preview_text = self._get_first_n_pages_text(pdf_path, max_pages)
        if not preview_text or len(preview_text.strip()) == 0:
            raise RuntimeError("No text available from the first pages for LLM extraction")
        
        llm = EnhancedLLMManager()
        system_prompt = (
            "You are an expert at reading academic papers and extracting bibliographic metadata. "
            "Given the first pages of a paper, extract structured metadata. Do not fabricate data. "
            "If a field cannot be determined from the text, output a clear 'Unknown ...' placeholder. "
            "Prefer exact strings from the document for title and authors."
        )
        user_prompt = (
            "Extract the following fields as strict JSON with keys: \n"
            "- title (string)\n"
            "- authors (array of strings, each 'Given Family')\n"
            "- year (string, 4-digit or 'Unknown Year')\n"
            "- doi (string like '10.xxxx/...' or 'Unknown DOI')\n"
            "- journal (string or 'Unknown Journal')\n"
            "- publisher (string or 'Unknown Publisher')\n"
            "- abstract (string or 'Unknown Abstract')\n\n"
            "Rules:\n"
            "- Return ONLY valid JSON, no commentary, no code fences.\n"
            "- Do not guess a DOI; only include it if explicitly present; otherwise 'Unknown DOI'.\n"
            "- Authors should be captured exactly as listed; include all visible authors on the page(s).\n\n"
            f"Document (first {max_pages} pages) below:\n\n{preview_text}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response_text = llm.generate_response(messages, max_tokens=1200, temperature=0.0)
        parsed = self._parse_llm_metadata_response(response_text)
        if parsed is None:
            # Retry with a stricter follow-up prompt requesting JSON only
            retry_text = self._retry_llm_for_json(llm, preview_text, max_pages)
            parsed = self._parse_llm_metadata_response(retry_text)
        if parsed is None:
            # Heuristic fallback from the LLM response and preview text
            parsed = self._heuristic_extract_metadata_from_text(response_text, preview_text)
        return parsed

    def _parse_llm_metadata_response(self, text: str) -> Optional[Dict]:
        """
        Robustly parse LLM output into a metadata dict. Returns None if parsing fails.
        """
        if not isinstance(text, str):
            return None
        cleaned = text.strip()
        # Remove common code fences and labels
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        # Try direct JSON first
        try:
            obj = json.loads(cleaned)
            return self._normalize_llm_parsed_metadata(obj)
        except Exception:
            pass
        # Try to extract the first JSON object via brace matching
        obj_str = self._extract_first_json_object(cleaned)
        if obj_str:
            for attempt in range(3):
                try:
                    return self._normalize_llm_parsed_metadata(json.loads(obj_str))
                except Exception:
                    # Attempt light repairs: convert single quotes, remove trailing commas
                    obj_str = self._lightly_repair_json_string(obj_str)
            # Try Python literal eval as last resort
            try:
                py_obj = ast.literal_eval(obj_str)
                if isinstance(py_obj, dict):
                    return self._normalize_llm_parsed_metadata(py_obj)
            except Exception:
                pass
        return None

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        stack = []
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append('{')
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start != -1:
                        return text[start:i+1]
        return None

    def _lightly_repair_json_string(self, s: str) -> str:
        # Remove trailing commas before } or ]
        s = re.sub(r",\s*(\}|\])", r"\1", s)
        # Replace Python booleans and None
        s = s.replace("None", "null").replace("True", "true").replace("False", "false")
        return s

    def _normalize_llm_parsed_metadata(self, parsed: Dict) -> Dict:
        # Title
        title = str(parsed.get("title", "Unknown Title")).strip() or "Unknown Title"
        # Authors: handle list of dicts, list of strings, or single string
        authors_field = parsed.get("authors", [])
        authors: List[str] = []
        if isinstance(authors_field, list):
            for a in authors_field:
                if isinstance(a, dict):
                    given = str(a.get("given", "")).strip()
                    family = str(a.get("family", "")).strip()
                    name = f"{given} {family}".strip()
                    if name:
                        authors.append(name)
                elif isinstance(a, str):
                    authors.append(a.strip())
        elif isinstance(authors_field, dict):
            given = str(authors_field.get("given", "")).strip()
            family = str(authors_field.get("family", "")).strip()
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)
        elif isinstance(authors_field, str):
            # Split common separators
            parts = re.split(r"\s*(?:,| and | & |;|\n)\s*", authors_field)
            authors.extend([p for p in (part.strip() for part in parts) if p])
        if not authors:
            authors = ["Unknown Author"]
        # Year
        year_val = parsed.get("year", "Unknown Year")
        if isinstance(year_val, int):
            year = str(year_val)
        else:
            year = str(year_val).strip() or "Unknown Year"
        # DOI and other fields
        doi = str(parsed.get("doi", "Unknown DOI")).strip() or "Unknown DOI"
        journal = str(parsed.get("journal", "Unknown Journal")).strip() or "Unknown Journal"
        publisher = str(parsed.get("publisher", "Unknown Publisher")).strip() or "Unknown Publisher"
        abstract = str(parsed.get("abstract", "Unknown Abstract")).strip() or "Unknown Abstract"
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "journal": journal,
            "publisher": publisher,
            "abstract": abstract,
        }

    def _retry_llm_for_json(self, llm: EnhancedLLMManager, preview_text: str, max_pages: int) -> str:
        system_prompt = (
            "Return ONLY a single JSON object matching the schema. No explanations, no code fences, no extra text."
        )
        schema_hint = (
            "Schema keys: title (string), authors (array of strings), year (string), doi (string), "
            "journal (string), publisher (string), abstract (string)."
        )
        user_prompt = (
            f"{schema_hint}\nIf a value is unknown, use 'Unknown ...' placeholders exactly.\n\n"
            f"Document (first {max_pages} pages):\n\n{preview_text}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return llm.generate_response(messages, max_tokens=800, temperature=0.0)

    def _heuristic_extract_metadata_from_text(self, llm_text: str, preview_text: str) -> Dict:
        # DOI regex
        doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
        doi_match = re.search(doi_pattern, (llm_text or "") + "\n" + (preview_text or ""), re.IGNORECASE)
        doi = doi_match.group(0) if doi_match else "Unknown DOI"
        # Year
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", (llm_text or "") + "\n" + (preview_text or ""))
        year = year_match.group(0) if year_match else "Unknown Year"
        # Title: pick the first non-empty long line from preview
        title = "Unknown Title"
        for line in (preview_text or "").splitlines():
            stripped = line.strip()
            if len(stripped.split()) >= 5 and len(stripped) > 15:
                title = stripped
                break
        # Authors: naive extraction from llm_text lines mentioning 'author'
        authors = ["Unknown Author"]
        m = re.search(r"authors?\s*[:\-]\s*(.+)", llm_text or "", re.IGNORECASE)
        if m:
            parts = re.split(r"\s*(?:,| and | & |;|\n)\s*", m.group(1))
            cleaned = [p for p in (part.strip(' .') for part in parts) if p]
            if cleaned:
                authors = cleaned
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "journal": "Unknown Journal",
            "publisher": "Unknown Publisher",
            "abstract": "Unknown Abstract",
        }

    def _get_first_n_pages_text(self, pdf_path: str, n: int = 3) -> str:
        """
        Extract text from the first N pages using the best available engine.
        """
        try:
            if HAS_PYMUPDF:
                doc = fitz.open(pdf_path)
                pages = []
                for idx in range(min(n, len(doc))):
                    page = doc.load_page(idx)
                    text = page.get_text("text")
                    if text:
                        pages.append(text)
                doc.close()
                return "\n".join(pages)
        except Exception as e:
            logging.warning(f"PyMuPDF failed for first pages extraction: {e}")
        
        try:
            if HAS_PYPDF2:
                reader = PdfReader(pdf_path)
                pages = []
                for idx, page in enumerate(reader.pages[:n]):
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                return "\n".join(pages)
        except Exception as e:
            logging.warning(f"PyPDF2 failed for first pages extraction: {e}")
        
        # Last attempt with pdfplumber
        try:
            if HAS_PDFPLUMBER:
                with pdfplumber.open(pdf_path) as pdf:
                    pages = []
                    for idx, page in enumerate(pdf.pages[:n]):
                        text = page.extract_text() or ""
                        if text:
                            pages.append(text)
                return "\n".join(pages)
        except Exception as e:
            logging.warning(f"pdfplumber failed for first pages extraction: {e}")
        
        return ""

    def _validate_metadata_quality(self, metadata: Dict) -> bool:
        """
        Validate the quality of extracted metadata.
        Returns True if metadata meets minimum quality standards.
        """
        # Check if title is meaningful (not just "Unknown" or too short)
        title = metadata.get("title", "")
        if not title or title == "Unknown Title" or len(title) < 5:
            return False
            
        # Check if we have at least one real author
        authors = metadata.get("authors", [])
        if not authors or authors == ["Unknown Author"]:
            return False
            
        # Check if year is reasonable
        year = metadata.get("year", "")
        try:
            year_int = int(year) if year and year != "Unknown Year" else 0
            if year_int < 1900 or year_int > datetime.now().year + 1:
                return False
        except ValueError:
            return False
            
        return True

    def _extract_metadata_from_filename(self, pdf_path: str) -> Dict:
        """
        Fallback: Extract basic metadata from filename.
        """
        filename = os.path.basename(pdf_path)
        
        # Common patterns: "Author - Year - Title.pdf"
        # or "Author (Year) Title.pdf"
        patterns = [
            r'^([^-]+)\s*-\s*(\d{4})\s*-\s*(.+)\.pdf$',
            r'^([^(]+)\s*\((\d{4})\)\s*(.+)\.pdf$',
            r'^([^-]+)\s*-\s*(.+)\.pdf$'  # No year pattern
        ]
        
        for pattern in patterns:
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:
                    author, year, title = match.groups()
                    return {
                        "title": title.strip(),
                        "authors": [author.strip()],
                        "year": year.strip(),
                        "doi": "Unknown DOI", 
                        "journal": "Unknown Journal",
                        "publisher": "Unknown Publisher",
                        "abstract": "Unknown Abstract"
                    }
                elif len(match.groups()) == 2:
                    author, title = match.groups()
                    return {
                        "title": title.strip(),
                        "authors": [author.strip()],
                        "year": str(datetime.fromtimestamp(os.path.getmtime(pdf_path)).year),
                        "doi": "Unknown DOI",
                        "journal": "Unknown Journal", 
                        "publisher": "Unknown Publisher",
                        "abstract": "Unknown Abstract"
                    }
        
        # Last resort: use filename as title
        return {
            "title": filename.replace('.pdf', ''),
            "authors": ["Unknown Author"],
            "year": str(datetime.fromtimestamp(os.path.getmtime(pdf_path)).year),
            "doi": "Unknown DOI",
            "journal": "Unknown Journal",
            "publisher": "Unknown Publisher", 
            "abstract": "Unknown Abstract"
        }

    def _fetch_metadata_from_crossref(self, doi: str) -> Dict:
        """
        Fetch high-quality metadata from CrossRef API using DOI.
        Returns comprehensive metadata with standardized format.
        """
        if not doi or doi == "Unknown DOI":
            raise ValueError("Invalid DOI provided")
        
        # Clean DOI (remove potential prefixes)
        clean_doi = doi.replace("doi:", "").replace("DOI:", "").strip()
        
        url = f"https://api.crossref.org/works/{clean_doi}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "CiteWeave/1.0 (https://github.com/user/citeweave; mailto:contact@example.com)"
        }
        
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()  # Raises an HTTPError for bad responses
            
            data = resp.json().get("message", {})
            
            # Extract authors
            authors = []
            for author in data.get("author", []):
                given = author.get("given", "")
                family = author.get("family", "")
                if family:
                    full_name = f"{given} {family}".strip()
                    authors.append(full_name)
            
            # Extract publication year
            published = data.get("published-print") or data.get("published-online") or data.get("created")
            year = "Unknown Year"
            if published and "date-parts" in published:
                date_parts = published["date-parts"][0]
                if date_parts:
                    year = str(date_parts[0])
            
            # Extract title (CrossRef returns title as array)
            title_list = data.get("title", [])
            title = title_list[0] if title_list else "Unknown Title"
            
            # Extract journal/container
            container_list = data.get("container-title", [])
            journal = container_list[0] if container_list else "Unknown Journal"
            
            # Extract abstract if available
            abstract = data.get("abstract", "Unknown Abstract")
            
            metadata = {
                "title": title,
                "authors": authors if authors else ["Unknown Author"],
                "year": year,
                "doi": clean_doi,
                "journal": journal,
                "publisher": data.get("publisher", "Unknown Publisher"),
                "abstract": abstract,
                "url": data.get("URL", "Unknown URL"),
                "type": data.get("type", "Unknown Type"),
                "volume": data.get("volume", "Unknown Volume"),
                "issue": data.get("issue", "Unknown Issue"),
                "pages": data.get("page", "Unknown Pages"),
                "issn": data.get("ISSN", ["Unknown ISSN"])[0] if data.get("ISSN") else "Unknown ISSN"
            }
            
            logging.info(f"Successfully fetched metadata from CrossRef for DOI: {clean_doi}")
            return metadata
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"CrossRef API request failed: {e}")
        except (KeyError, IndexError, ValueError) as e:
            raise Exception(f"Failed to parse CrossRef response: {e}")

    def _extract_metadata_with_pypdf2(self, pdf_path: str) -> Dict:
        try:
            reader = PdfReader(pdf_path)
            meta = reader.metadata or {}
            
            # Handle creation_date which might be a datetime object
            creation_date = getattr(meta, "creation_date", None)
            if creation_date and hasattr(creation_date, 'isoformat'):
                creation_date = creation_date.isoformat()
            elif creation_date:
                creation_date = str(creation_date)
            else:
                creation_date = "Unknown Creation Date"
            
            return {
                "title": getattr(meta, "title", None) or "Unknown Title",
                "authors": [getattr(meta, "author", None) or "Unknown Author"],
                "year": str(datetime.fromtimestamp(os.path.getmtime(pdf_path)).year),
                "doi": getattr(meta, "doi", None) or "Unknown DOI",
                "url": getattr(meta, "url", None) or "Unknown URL",
                "publisher": getattr(meta, "publisher", None) or "Unknown Publisher",
                "subject": getattr(meta, "subject", None) or "Unknown Subject",
                "keywords": getattr(meta, "keywords", None) or "Unknown Keywords",
                "language": getattr(meta, "language", None) or "Unknown Language",
                "creator": getattr(meta, "creator", None) or "Unknown Creator",
                "producer": getattr(meta, "producer", None) or "Unknown Producer",
                "creation_date": creation_date,
                "num_pages": len(reader.pages)
            }
        except Exception as e:
            logging.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return {
                "title": "Unknown Title",
                "authors": ["Unknown Author"],
                "year": str(datetime.fromtimestamp(os.path.getmtime(pdf_path)).year),
                "doi": "Unknown DOI",
                "url": "Unknown URL",
                "publisher": "Unknown Publisher",
                "subject": "Unknown Subject",
                "keywords": "Unknown Keywords",
                "language": "Unknown Language",
                "creator": "Unknown Creator",
                "producer": "Unknown Producer",
                "creation_date": "Unknown Creation Date",
                "num_pages": 0,
                "error": str(e)
            }

    def _extract_metadata_with_grobid(self, pdf_path: str) -> Dict:
        try:
            with open(pdf_path, "rb") as f:
                files = {"input": (os.path.basename(pdf_path), f, "application/pdf")}
                headers = {"Accept": "application/xml"}
                response = requests.post(
                    "http://localhost:8070/api/processHeaderDocument",
                    files=files,
                    headers=headers,
                    timeout=30  # Add timeout
                )
        except requests.exceptions.ConnectionError:
            logging.warning(f"GROBID service not available at localhost:8070 - skipping metadata extraction")
            return {}
        except requests.exceptions.Timeout:
            logging.warning(f"GROBID request timed out - skipping metadata extraction")
            return {}
        except Exception as e:
            logging.warning(f"Failed to connect to GROBID: {e}")
            return {}

        if response.status_code != 200 or not response.text.strip().startswith("<"):
            logging.warning(f"Grobid did not return valid XML for {pdf_path}:\n{response.text[:300]}")
            return {}

        try:
            root = etree.fromstring(response.text.strip().encode("utf-8"))
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}

            title = root.findtext(".//tei:titleStmt/tei:title", namespaces=ns) or "Unknown Title"

            authors = []
            for pers in root.findall(".//tei:analytic/tei:author/tei:persName", namespaces=ns):
                surname = pers.findtext("tei:surname", namespaces=ns)
                forenames = pers.findall("tei:forename", namespaces=ns)
                full_name = " ".join([fn.text for fn in forenames if fn.text] + ([surname] if surname else []))
                if full_name:
                    authors.append(full_name)

            doi = root.findtext(".//tei:idno[@type='DOI']", namespaces=ns) or "Unknown DOI"
            publisher = root.findtext(".//tei:monogr/tei:imprint/tei:publisher", namespaces=ns) or "Unknown Publisher"
            year = root.findtext(".//tei:monogr/tei:imprint/tei:date", namespaces=ns) or "Unknown Year"
            journal = root.findtext(".//tei:monogr/tei:title", namespaces=ns) or "Unknown Journal"
            abstract = root.findtext(".//tei:abstract/tei:p", namespaces=ns) or "Unknown Abstract"

            return {
                "title": title,
                "authors": authors or ["Unknown Author"],
                "journal": journal,
                "doi": doi,
                "publisher": publisher,
                "year": year,
                "abstract": abstract
            }

        except Exception as e:
            logging.error(f"Error parsing Grobid XML for {pdf_path}: {e}")
            return {}

    def _clean_sentence_text(self, text: str) -> str:
        """
        Clean sentence text by removing special characters and fixing common formatting issues.
        Specifically handles:
        - \\n newlines and other whitespace
        - Unicode control characters (\\u001, \\u002, etc.)
        - Word breaks (-\\n) by removing them completely
        - Normalizing multiple spaces
        """
        if not text:
            return text
        
        # First handle word breaks: -\n should be removed completely (joining broken words)
        text = re.sub(r'-\s*\n\s*', '', text)
        
        # Remove other newlines and replace with spaces
        text = re.sub(r'\n+', ' ', text)
        
        # Remove unicode control characters (\u0001-\u001F, \u007F-\u009F) and replace with space
        text = re.sub(r'[\u0001-\u001F\u007F-\u009F]', ' ', text)
        
        # Replace other common problematic characters with space
        text = re.sub(r'[\u2000-\u200F\u2028-\u202F\u205F-\u206F]', ' ', text)  # Various spaces and separators
        
        # Normalize multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

    def extract_document_structure(self, pdf_path: str) -> Dict:
        """
        提取PDF文档的真实结构信息，包括段落和章节分割
        
        Returns:
            Dict containing:
            - sections: List of sections with their content
            - paragraphs: List of paragraphs with metadata
            - raw_text_blocks: Original text blocks from PDF
        """
        try:
            if HAS_PYMUPDF:
                return self._extract_structure_with_pymupdf(pdf_path)
            elif HAS_PDFPLUMBER:
                return self._extract_structure_with_pdfplumber(pdf_path)
            else:
                # Fallback to basic text extraction with heuristic parsing
                return self._extract_structure_fallback(pdf_path)
        except Exception as e:
            logging.error(f"Failed to extract document structure: {e}")
            return {"sections": [], "paragraphs": [], "raw_text_blocks": []}
    
    def _extract_structure_with_pymupdf(self, pdf_path: str) -> Dict:
        """使用PyMuPDF提取文档结构"""
        doc = fitz.open(pdf_path)
        sections = []
        paragraphs = []
        raw_text_blocks = []
        
        current_section = None
        section_index = 0
        paragraph_index = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 获取页面的文本块信息，包含字体、位置等
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:  # 跳过图像块
                    continue
                
                # 提取块中的文本和格式信息
                block_text = ""
                font_sizes = []
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = span["size"]
                        font_sizes.append(font_size)
                        line_text += text + " "
                    block_text += line_text.strip() + "\n"
                
                block_text = block_text.strip()
                if not block_text:
                    continue
                
                # 分析文本块是否为标题/章节
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                is_heading = self._is_heading_block(block_text, avg_font_size, font_sizes)
                
                raw_text_blocks.append({
                    "text": block_text,
                    "page": page_num,
                    "font_size": avg_font_size,
                    "is_heading": is_heading,
                    "bbox": block.get("bbox", [])
                })
                
                # 根据分析结果分类
                if is_heading:
                    # 创建新章节
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        "id": f"section_{section_index}",
                        "title": block_text,
                        "text": "",
                        "index": section_index,
                        "section_type": self._classify_section_type(block_text),
                        "paragraphs": [],
                        "page_start": page_num
                    }
                    section_index += 1
                
                else:
                    # 处理段落内容
                    paragraphs_in_block = self._split_block_into_paragraphs(block_text, paragraph_index, page_num)
                    
                    for para in paragraphs_in_block:
                        # 确定段落所属章节
                        section_name = current_section["title"] if current_section else "Unknown"
                        para["section"] = section_name
                        
                        paragraphs.append(para)
                        
                        if current_section:
                            current_section["paragraphs"].append(para)
                            current_section["text"] += para["text"] + "\n\n"
                        
                        paragraph_index += 1
        
        # 添加最后一个章节
        if current_section:
            sections.append(current_section)
        
        doc.close()
        
        # 完善章节信息
        for section in sections:
            section["paragraph_count"] = len(section["paragraphs"])
            section["text"] = section["text"].strip()
        
        return {
            "sections": sections,
            "paragraphs": paragraphs,
            "raw_text_blocks": raw_text_blocks
        }
    
    def _extract_structure_with_pdfplumber(self, pdf_path: str) -> Dict:
        """使用pdfplumber提取文档结构"""
        import pdfplumber
        
        sections = []
        paragraphs = []
        raw_text_blocks = []
        
        current_section = None
        section_index = 0
        paragraph_index = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # 提取文本和字符信息
                chars = page.chars
                
                if not chars:
                    continue
                
                # 按行分组字符
                lines = self._group_chars_into_lines(chars)
                
                # 按段落分组行
                text_blocks = self._group_lines_into_blocks(lines)
                
                for block in text_blocks:
                    block_text = block["text"].strip()
                    if not block_text:
                        continue
                    
                    # 分析是否为标题
                    is_heading = self._is_heading_block_pdfplumber(block)
                    
                    raw_text_blocks.append({
                        "text": block_text,
                        "page": page_num,
                        "font_size": block.get("avg_font_size", 12),
                        "is_heading": is_heading,
                        "bbox": block.get("bbox", [])
                    })
                    
                    if is_heading:
                        # 创建新章节
                        if current_section:
                            sections.append(current_section)
                        
                        current_section = {
                            "id": f"section_{section_index}",
                            "title": block_text,
                            "text": "",
                            "index": section_index,
                            "section_type": self._classify_section_type(block_text),
                            "paragraphs": [],
                            "page_start": page_num
                        }
                        section_index += 1
                    
                    else:
                        # 处理段落
                        paragraphs_in_block = self._split_block_into_paragraphs(block_text, paragraph_index, page_num)
                        
                        for para in paragraphs_in_block:
                            section_name = current_section["title"] if current_section else "Unknown"
                            para["section"] = section_name
                            
                            paragraphs.append(para)
                            
                            if current_section:
                                current_section["paragraphs"].append(para)
                                current_section["text"] += para["text"] + "\n\n"
                            
                            paragraph_index += 1
        
        # 添加最后一个章节
        if current_section:
            sections.append(current_section)
        
        # 完善章节信息
        for section in sections:
            section["paragraph_count"] = len(section["paragraphs"])
            section["text"] = section["text"].strip()
        
        return {
            "sections": sections,
            "paragraphs": paragraphs,
            "raw_text_blocks": raw_text_blocks
        }
    
    def _is_heading_block(self, text: str, avg_font_size: float, font_sizes: List[float]) -> bool:
        """判断文本块是否为标题"""
        text = text.strip()
        
        # 空文本不是标题
        if not text:
            return False
        
        # 通用页眉页脚识别逻辑
        if self._is_header_footer_content(text):
            return False
        
        # 长度判断：标题通常较短
        if len(text) > 200:
            return False
        
        # 字体大小判断：标题通常字体较大
        font_size_threshold = 13  # 可调节
        if avg_font_size > font_size_threshold:
            return True
        
        # 模式匹配：常见标题模式
        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction", "2 Methods"
            r'^[A-Z][A-Z\s]+$',  # "INTRODUCTION", "METHODOLOGY"
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # "Introduction", "Literature Review"
            r'^\d+\.\d+\.?\s+[A-Z]',  # "1.1 Background"
            r'^Abstract$|^Introduction$|^Conclusion$|^References$|^Methodology$',  # 常见章节名
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
        
        # 行数判断：标题通常只有1-2行
        lines = text.split('\n')
        if len(lines) <= 2 and all(len(line.strip()) < 100 for line in lines):
            # 检查是否包含标题关键词
            title_keywords = ['introduction', 'background', 'method', 'result', 'conclusion', 
                            'discussion', 'literature', 'analysis', 'findings', 'summary']
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in title_keywords):
                return True
        
        return False
    
    def _is_header_footer_content(self, text: str) -> bool:
        """
        通用页眉页脚检测方法 - 基于内容特征而非特定期刊信息
        """
        text = text.strip()
        text_lower = text.lower()
        
        # 1. 纯数字（页码）
        if re.match(r'^\d+$', text):
            return True
        
        # 1.5. 页码格式（改进）
        if re.match(r'^\s*page\s+\d+\s*$', text_lower):
            return True
        
        # 2. 版权和法律信息
        copyright_patterns = [
            r'©.*\d{4}',  # 版权符号
            r'copyright.*\d{4}',
            r'all rights reserved',
            r'for personal use only',
            r'downloaded from',
            r'terms and conditions',
            r'unauthorized reproduction',
        ]
        for pattern in copyright_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # 3. 出版商和期刊信息模式（通用）
        publisher_patterns = [
            r'published by',
            r'publication details',
            r'subscription information',
            r'vol\.\s*\d+.*no\.\s*\d+',  # Volume/Issue信息
            r'volume\s+\d+.*number\s+\d+',
            r'issn\s*:?\s*\d{4}-\d{4}',  # ISSN
            r'doi\s*:?\s*10\.\d+',  # DOI
        ]
        for pattern in publisher_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # 4. URL和网址
        if re.search(r'https?://|www\.|\.com|\.org|\.edu', text_lower):
            return True
        
        # 5. 引用格式信息（改进）
        citation_format_patterns = [
            r'to cite this article',
            r'citation format',
            r'how to cite',
            r'cite as:',
            r'citation:\s*[A-Z]',  # "Citation: Author"格式
            r'^citation:\s*\w+',  # 以"Citation:"开头
        ]
        for pattern in citation_format_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # 6. 日期和时间戳
        if re.search(r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', text_lower):
            return True
        if re.search(r'\d{1,2}:\d{2}', text):  # 时间格式
            return True
        
        # 7. IP地址和技术标识符
        if re.search(r'\[?[0-9a-f]{4}:[0-9a-f]{4}:[0-9a-f]{4}', text_lower):  # IPv6片段
            return True
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text):  # IPv4
            return True
        
        # 8. 非常短的文本（可能是页码或标识符）
        if len(text) < 5 and not re.match(r'^[A-Z][a-z]*$', text):  # 排除简单的单词
            return True
        
        # 9. 只包含数字、标点和很少字母的文本
        letter_count = sum(1 for c in text if c.isalpha())
        if len(text) > 10 and letter_count / len(text) < 0.3:  # 字母占比小于30%
            return True
        
        # 10. 期刊特定模式（更通用，改进）
        journal_patterns = [
            r'journal\s+of\s+\w+',
            r'\w+\s+science\s*/?\s*vol',  # 任何科学期刊
            r'\w+\s+review\s*/?\s*vol',   # 任何评论期刊
            r'proceedings\s+of',
            r'transactions\s+on',
            r'annals\s+of',
            r'\w+\s+business\s+review$',  # 商业评论类期刊
            r'harvard\s+business\s+review$',  # 特定知名期刊
            r'nature\s+\w*$',  # Nature系列
            r'science\s*$',  # Science期刊
        ]
        for pattern in journal_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # 11. 引用列表模式（当一行包含太多引用时，可能是页眉）
        citation_count = len(re.findall(r'\(\d{4}[a-z]?\)', text))
        if citation_count >= 4 and len(text) < 300:  # 短文本中包含4个以上引用
            return True
        
        # 12. 作者列表在页眉中的模式
        if re.search(r'^[A-Z][a-z]+\s+et\s+al\.?\s*$', text):  # 只有作者名
            return True
        
        # 13. 多行文本检查（改进）
        lines = text.split('\n')
        if len(lines) > 1:
            # 检查是否包含页码行
            for line in lines:
                line = line.strip()
                if re.match(r'^\s*page\s+\d+\s*$', line.lower()) or re.match(r'^\d+\s*$', line):
                    return True
        
        return False
    
    def _is_heading_block_pdfplumber(self, block: Dict) -> bool:
        """pdfplumber版本的标题判断"""
        text = block["text"].strip()
        avg_font_size = block.get("avg_font_size", 12)
        
        # 基本判断
        if not text:
            return False
        
        # 使用相同的通用页眉页脚检测逻辑
        if self._is_header_footer_content(text):
            return False
        
        # 长度判断
        if len(text) > 200:
            return False
        
        # 字体大小判断
        if avg_font_size > 13:
            return True
        
        # 模式匹配（同PyMuPDF版本）
        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',
            r'^[A-Z][A-Z\s]+$',
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',
            r'^\d+\.\d+\.?\s+[A-Z]',
            r'^Abstract$|^Introduction$|^Conclusion$|^References$|^Methodology$',
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
        
        # 标题关键词检查
        lines = text.split('\n')
        if len(lines) <= 2 and all(len(line.strip()) < 100 for line in lines):
            title_keywords = ['introduction', 'background', 'method', 'result', 'conclusion', 
                            'discussion', 'literature', 'analysis', 'findings', 'summary']
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in title_keywords):
                return True
        
        return False
    
    def _classify_section_type(self, title: str) -> str:
        """根据标题分类章节类型"""
        title_lower = title.lower()
        
        if any(keyword in title_lower for keyword in ['abstract']):
            return 'abstract'
        elif any(keyword in title_lower for keyword in ['introduction', 'background']):
            return 'introduction'
        elif any(keyword in title_lower for keyword in ['literature', 'related work', 'prior']):
            return 'literature_review'
        elif any(keyword in title_lower for keyword in ['method', 'approach', 'design']):
            return 'methodology'
        elif any(keyword in title_lower for keyword in ['result', 'finding', 'analysis']):
            return 'results'
        elif any(keyword in title_lower for keyword in ['discussion']):
            return 'discussion'
        elif any(keyword in title_lower for keyword in ['conclusion', 'summary']):
            return 'conclusion'
        elif any(keyword in title_lower for keyword in ['reference', 'bibliography']):
            return 'references'
        else:
            return 'other'
    
    def _split_block_into_paragraphs(self, block_text: str, start_index: int, page_num: int) -> List[Dict]:
        """将文本块分割为段落"""
        paragraphs = []
        
        # 按空行分割段落
        raw_paragraphs = re.split(r'\n\s*\n', block_text)
        
        for i, para_text in enumerate(raw_paragraphs):
            para_text = para_text.strip()
            if not para_text or len(para_text) < 20:  # 跳过太短的段落
                continue
            
            # 清理段落文本
            para_text = re.sub(r'\s+', ' ', para_text)  # 标准化空格
            para_text = re.sub(r'-\s*\n\s*', '', para_text)  # 移除连字符换行
            
            paragraph = {
                "id": f"para_{start_index + i}",
                "text": para_text,
                "index": start_index + i,
                "page": page_num,
                "word_count": len(para_text.split()),
                "char_count": len(para_text),
                "citation_count": 0,  # 将在后续处理中更新
                "sentence_count": len(re.split(r'[.!?]+', para_text))
            }
            
            paragraphs.append(paragraph)
        
        return paragraphs
    
    def _group_chars_into_lines(self, chars: List[Dict]) -> List[Dict]:
        """将字符分组为行（pdfplumber专用）"""
        if not chars:
            return []
        
        # 按y坐标分组字符
        lines = {}
        for char in chars:
            y = round(char['y0'], 1)  # 四舍五入到小数点后1位
            if y not in lines:
                lines[y] = []
            lines[y].append(char)
        
        # 转换为列表并排序
        line_list = []
        for y in sorted(lines.keys(), reverse=True):  # 从上到下
            line_chars = sorted(lines[y], key=lambda c: c['x0'])  # 从左到右
            line_text = ''.join(char['text'] for char in line_chars)
            
            if line_text.strip():
                line_list.append({
                    "text": line_text,
                    "y": y,
                    "chars": line_chars,
                    "font_sizes": [char.get('size', 12) for char in line_chars]
                })
        
        return line_list
    
    def _group_lines_into_blocks(self, lines: List[Dict]) -> List[Dict]:
        """将行分组为文本块"""
        if not lines:
            return []
        
        blocks = []
        current_block = None
        
        for line in lines:
            # 判断是否开始新块（基于行间距）
            if current_block is None:
                current_block = {
                    "text": line["text"],
                    "lines": [line],
                    "font_sizes": line["font_sizes"]
                }
            else:
                # 计算行间距
                prev_y = current_block["lines"][-1]["y"]
                curr_y = line["y"]
                line_spacing = abs(prev_y - curr_y)
                
                # 如果行间距过大，开始新块
                if line_spacing > 20:  # 可调节的阈值
                    # 完成当前块
                    self._finalize_block(current_block)
                    blocks.append(current_block)
                    
                    # 开始新块
                    current_block = {
                        "text": line["text"],
                        "lines": [line],
                        "font_sizes": line["font_sizes"]
                    }
                else:
                    # 添加到当前块
                    current_block["text"] += "\n" + line["text"]
                    current_block["lines"].append(line)
                    current_block["font_sizes"].extend(line["font_sizes"])
        
        # 添加最后一个块
        if current_block:
            self._finalize_block(current_block)
            blocks.append(current_block)
        
        return blocks
    
    def _finalize_block(self, block: Dict):
        """完善文本块信息"""
        if block["font_sizes"]:
            block["avg_font_size"] = sum(block["font_sizes"]) / len(block["font_sizes"])
        else:
            block["avg_font_size"] = 12
    
    def _extract_structure_fallback(self, pdf_path: str) -> Dict:
        """备用结构提取方法"""
        # 使用现有的文本提取方法
        full_text, _ = self.extract_text_with_best_engine(pdf_path)
        
        # 基于启发式规则分割章节和段落
        sections = []
        paragraphs = []
        
        # 简单的章节分割
        section_patterns = [
            r'\n\s*(\d+\.?\s+[A-Z][^\n]{5,50})\s*\n',
            r'\n\s*([A-Z][A-Z\s]{5,30})\s*\n',
            r'\n\s*(Abstract|Introduction|Methodology|Results|Discussion|Conclusion|References)\s*\n'
        ]
        
        current_pos = 0
        section_index = 0
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
            
            for match in matches:
                section_title = match.group(1).strip()
                section_start = match.end()
                
                # 找到下一个章节的开始
                next_match = None
                for next_pattern in section_patterns:
                    next_matches = list(re.finditer(next_pattern, full_text[section_start:], re.IGNORECASE))
                    if next_matches:
                        if next_match is None or next_matches[0].start() < next_match.start():
                            next_match = next_matches[0]
                
                if next_match:
                    section_end = section_start + next_match.start()
                else:
                    section_end = len(full_text)
                
                section_text = full_text[section_start:section_end].strip()
                
                # 分割段落
                section_paragraphs = []
                para_texts = re.split(r'\n\s*\n', section_text)
                
                for i, para_text in enumerate(para_texts):
                    para_text = para_text.strip()
                    if len(para_text) > 50:  # 只保留有意义的段落
                        para = {
                            "id": f"para_{len(paragraphs)}",
                            "text": para_text,
                            "index": len(paragraphs),
                            "section": section_title,
                            "word_count": len(para_text.split()),
                            "char_count": len(para_text),
                            "citation_count": 0,
                            "sentence_count": len(re.split(r'[.!?]+', para_text))
                        }
                        paragraphs.append(para)
                        section_paragraphs.append(para)
                
                section = {
                    "id": f"section_{section_index}",
                    "title": section_title,
                    "text": section_text,
                    "index": section_index,
                    "section_type": self._classify_section_type(section_title),
                    "paragraphs": section_paragraphs,
                    "paragraph_count": len(section_paragraphs)
                }
                
                sections.append(section)
                section_index += 1
        
        return {
            "sections": sections,
            "paragraphs": paragraphs,
            "raw_text_blocks": []
        }

    def parse_sentences(self, pdf_path: str) -> List[str]:
        """
        Enhanced sentence parsing with content filtering and quality control.
        Uses academic-text-aware sentence splitting to handle citations properly.
        """
        from nltk.tokenize import sent_tokenize
        import re

        # Use the best available PDF extraction engine
        full_text, extraction_info = self.extract_text_with_best_engine(pdf_path)
        
        logging.info(f"Text extraction completed with {extraction_info.get('engine', 'unknown')} engine")
        logging.info(f"Extracted {len(full_text)} characters from {extraction_info.get('pages_with_text', 0)}/{extraction_info.get('total_pages', 0)} pages")
        
        # Identify and separate reference section
        main_content, reference_section = self._separate_main_content_and_references(full_text)
        
        logging.info(f"Main content: {len(main_content)} chars, References: {len(reference_section)} chars")
        
        # Use academic-aware sentence splitting
        sentences = self._split_sentences_academic_aware(main_content)
        
        # Filter out low-quality sentences
        filtered_sentences = self._filter_invalid_sentences(sentences)
        
        # Clean each sentence text
        cleaned_sentences = [self._clean_sentence_text(sent) for sent in filtered_sentences]
        
        # Remove empty sentences after cleaning
        cleaned_sentences = [sent for sent in cleaned_sentences if sent.strip()]
        
        logging.info(f"Parsed {len(sentences)} raw sentences, filtered to {len(filtered_sentences)} sentences, cleaned to {len(cleaned_sentences)} final sentences")
        
        return cleaned_sentences
    
    def _split_sentences_academic_aware(self, text: str) -> List[str]:
        """
        学术文本专用的句子分割方法，正确处理引用
        """
        from nltk.tokenize import sent_tokenize
        
        # 先进行预处理，保护引用不被错误分割
        text = self._protect_citations_for_splitting(text)
        
        # 使用NLTK进行初步分割
        initial_sentences = sent_tokenize(text)
        
        # 后处理：恢复引用并合并被错误分割的句子
        final_sentences = self._merge_broken_citation_sentences(initial_sentences)
        
        return final_sentences
    
    def _protect_citations_for_splitting(self, text: str) -> str:
        """
        在句子分割前保护引用格式，避免被错误分割
        """
        # 保护常见的缩写，特别是在引用中的
        abbreviation_patterns = [
            (r'\bet al\.', 'ETAL_PLACEHOLDER'),
            (r'\bvs\.', 'VS_PLACEHOLDER'),
            (r'\bpp\.', 'PP_PLACEHOLDER'),
            (r'\bVol\.', 'VOL_PLACEHOLDER'),
            (r'\bNo\.', 'NO_PLACEHOLDER'),
            (r'\bFig\.', 'FIG_PLACEHOLDER'),
            (r'\bTab\.', 'TAB_PLACEHOLDER'),
            (r'\bDr\.', 'DR_PLACEHOLDER'),
            (r'\bProf\.', 'PROF_PLACEHOLDER'),
            (r'\bMr\.', 'MR_PLACEHOLDER'),
            (r'\bMs\.', 'MS_PLACEHOLDER'),
            (r'\bMrs\.', 'MRS_PLACEHOLDER'),
        ]
        
        protected_text = text
        for pattern, placeholder in abbreviation_patterns:
            protected_text = re.sub(pattern, placeholder, protected_text, flags=re.IGNORECASE)
        
        return protected_text
    
    def _merge_broken_citation_sentences(self, sentences: List[str]) -> List[str]:
        """
        合并被错误分割的引用句子
        """
        merged_sentences = []
        i = 0
        
        while i < len(sentences):
            current_sentence = sentences[i]
            
            # 恢复占位符
            current_sentence = self._restore_citation_placeholders(current_sentence)
            
            # 检查是否需要与下一个句子合并
            while i + 1 < len(sentences):
                next_sentence = self._restore_citation_placeholders(sentences[i + 1])
                
                # 如果当前句子以引用列表结尾但不完整，或下一句子以引用开头
                if self._should_merge_sentences(current_sentence, next_sentence):
                    current_sentence = current_sentence.rstrip() + " " + next_sentence.lstrip()
                    i += 1
                else:
                    break
            
            if current_sentence.strip():
                merged_sentences.append(current_sentence.strip())
            i += 1
        
        return merged_sentences
    
    def _restore_citation_placeholders(self, text: str) -> str:
        """
        恢复引用占位符为原始文本
        """
        replacements = [
            ('ETAL_PLACEHOLDER', 'et al.'),
            ('VS_PLACEHOLDER', 'vs.'),
            ('PP_PLACEHOLDER', 'pp.'),
            ('VOL_PLACEHOLDER', 'Vol.'),
            ('NO_PLACEHOLDER', 'No.'),
            ('FIG_PLACEHOLDER', 'Fig.'),
            ('TAB_PLACEHOLDER', 'Tab.'),
            ('DR_PLACEHOLDER', 'Dr.'),
            ('PROF_PLACEHOLDER', 'Prof.'),
            ('MR_PLACEHOLDER', 'Mr.'),
            ('MS_PLACEHOLDER', 'Ms.'),
            ('MRS_PLACEHOLDER', 'Mrs.'),
        ]
        
        restored_text = text
        for placeholder, original in replacements:
            restored_text = restored_text.replace(placeholder, original)
        
        return restored_text
    
    def _should_merge_sentences(self, current: str, next_sentence: str) -> bool:
        """
        判断两个句子是否应该合并
        """
        current = current.strip()
        next_sentence = next_sentence.strip()
        
        # 情况1：当前句子以引用列表结尾但不完整（如"et al."）
        if re.search(r'et al\.$', current):
            return True
        
        # 情况2：当前句子以逗号+引用结尾，下一句子看起来是引用的继续
        if re.search(r',\s*\([^)]*\d{4}[^)]*\)\s*$', current):
            # 下一句子以引用开始
            if re.search(r'^\s*\([^)]*\d{4}[^)]*\)', next_sentence):
                return True
            # 或者以逗号+引用开始
            if re.search(r'^\s*,\s*[A-Z][a-z]+', next_sentence):
                return True
        
        # 情况3：当前句子以作者名结尾，下一句子以年份开始
        if re.search(r'[A-Z][a-z]+\s+et al\.$', current):
            if re.search(r'^\s*\(\d{4}\)', next_sentence):
                return True
        
        # 情况4：当前句子不以句号结尾，下一句子不以大写字母开始
        if not current.endswith('.') and not re.match(r'^\s*[A-Z]', next_sentence):
            return True
        
        # 情况5：下一句子看起来是引用的继续部分
        citation_continuation_patterns = [
            r'^\s*\(\d{4}[a-z]?\)',  # (1995a)
            r'^\s*,\s*\d{4}',        # , 1995
            r'^\s*;\s*[A-Z][a-z]+',  # ; Porter
            r'^\s*and\s+[A-Z][a-z]+', # and Smith
            r'^\s*&\s+[A-Z][a-z]+',  # & Smith
        ]
        
        for pattern in citation_continuation_patterns:
            if re.match(pattern, next_sentence):
                return True
        
        return False

    def _separate_main_content_and_references(self, full_text: str) -> tuple:
        """
        Separate main content from reference section using multiple heuristics.
        Returns (main_content, reference_section)
        """
        # Find reference section start using multiple patterns
        ref_patterns = [
            r'\n\s*REFERENCES?\s*\n',
            r'\n\s*Bibliography\s*\n',
            r'\n\s*BIBLIOGRAPHY\s*\n',
            r'\n\s*Literature Cited\s*\n',
            r'\n\s*Works Cited\s*\n'
        ]
        
        ref_start = len(full_text)  # Default: no reference section found
        
        for pattern in ref_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                ref_start = min(ref_start, match.start())
        
        # Additional heuristic: Look for sudden increase in citation density
        # Split text into chunks and analyze citation patterns
        if ref_start == len(full_text):
            ref_start = self._detect_reference_section_by_density(full_text)
        
        main_content = full_text[:ref_start]
        reference_section = full_text[ref_start:]
        
        return main_content, reference_section

    def _detect_reference_section_by_density(self, text: str) -> int:
        """
        Detect reference section by analyzing citation density in text chunks.
        Returns the position where reference section likely starts.
        """
        lines = text.split('\n')
        chunk_size = 10  # Smaller chunks for more precision
        
        # Look for concentrated reference patterns starting from the end
        for i in range(len(lines) - chunk_size, 0, -chunk_size):
            chunk = '\n'.join(lines[i:i + chunk_size])
            
            # More specific reference patterns to avoid false positives
            ref_indicators = [
                r'^[A-Z][a-z]+,\s+[A-Z].*\d{4}',  # Author, Year at line start
                r'^\d+\.\s+[A-Z][a-z]+,.*\d{4}',  # Numbered reference
                r'\b(?:Harvard Business Review|Journal of|Proceedings of)',  # Publication names
                r'\bpp?\.\s*\d+-\d+',  # Page ranges
                r'\bVol\.\s*\d+,\s*No\.',  # Volume/Issue
                r'\bDoi:\s*10\.',  # DOI patterns
                r'^\s*\[[^\]]+\]',  # Bracketed citations
            ]
            
            # Count lines that look like references (not just any indicators)
            ref_like_lines = 0
            for line in lines[i:i + chunk_size]:
                line = line.strip()
                if len(line) < 20:  # Skip very short lines
                    continue
                    
                for pattern in ref_indicators:
                    if re.search(pattern, line):
                        ref_like_lines += 1
                        break
            
            # If most lines in chunk look like references
            ref_ratio = ref_like_lines / max(len([l for l in lines[i:i + chunk_size] if len(l.strip()) > 20]), 1)
            
            if ref_ratio > 0.6:  # 60% of substantial lines look like references
                # Found reference section, but let's be conservative and go back a bit
                return text.find('\n'.join(lines[max(0, i-5):]))
        
        return len(text)  # No reference section detected

    def _filter_invalid_sentences(self, sentences: List[str]) -> List[str]:
        """
        Filter out invalid sentences like headers, footers, and noise.
        """
        filtered = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip empty or very short sentences
            if len(sentence) < 10:
                continue
                
            # Skip common header/footer patterns
            if self._is_header_footer_pattern(sentence):
                continue
                
            # Skip sentences with too many special characters or formatting
            if self._has_excessive_formatting_noise(sentence):
                continue
                
            # Skip copyright and download notices
            if self._is_copyright_or_notice(sentence):
                continue
                
            filtered.append(sentence)
        
        return filtered

    def _is_header_footer_pattern(self, sentence: str) -> bool:
        """
        Detect common header/footer patterns.
        """
        patterns = [
            r'^\d+\s*$',  # Page numbers only
            r'^[A-Z\s]+/Vol\.\s*\d+',  # Journal headers
            r'Management Science.*\d{4}',  # Journal name with year
            r'Downloaded from.*\d{4}',  # Download notices
            r'For personal use only',  # Copyright notices
            r'^\d+\s+[A-Z][a-z]+\s+\d{4}$',  # Date patterns
            r'^Figure\s+\d+',  # Figure captions (should be handled separately)
            r'^Table\s+\d+',   # Table captions
            r'^\s*\d+\.\d+\s',  # Section numbers at start
        ]
        
        for pattern in patterns:
            if re.search(pattern, sentence):
                return True
        return False

    def _has_excessive_formatting_noise(self, sentence: str) -> bool:
        """
        Detect sentences with excessive formatting artifacts.
        """
        # Count special characters and formatting issues
        special_char_ratio = len(re.findall(r'[^\w\s.,;:!?()-]', sentence)) / len(sentence)
        
        # Check for OCR artifacts like H110222, H1154924
        ocr_artifacts = len(re.findall(r'H\d{6}', sentence))
        
        # Check for excessive spacing or line breaks
        whitespace_ratio = len(re.findall(r'\s+', sentence)) / len(sentence)
        
        return (special_char_ratio > 0.15 or 
                ocr_artifacts > 0 or 
                whitespace_ratio > 0.5)

    def _is_copyright_or_notice(self, sentence: str) -> bool:
        """
        Detect copyright notices and download information.
        """
        notice_patterns = [
            r'all rights reserved',
            r'copyright',
            r'downloaded from',
            r'for personal use only',
            r'© \d{4}',
            r'doi:',
            r'ISSN',
            r'ISBN'
        ]
        
        sentence_lower = sentence.lower()
        return any(pattern in sentence_lower for pattern in notice_patterns)

    def save_sentences(self, pdf_path: str, sentences: List[str]):
        """
        Save sentences to a JSONL file using the hash-based paper ID.
        """
        meta = self.extract_pdf_metadata(pdf_path)
        title = meta.get("title", "Unknown Title")
        year = meta.get("year", "Unknown Year")
        paper_id = self._generate_paper_id(title, year)
        out_path = os.path.join(self.storage_root, paper_id, "sentences.jsonl")
        with open(out_path, "w") as f:
            for idx, sent in enumerate(sentences):
                f.write(json.dumps({"index": idx, "text": sent}) + "\n")

    def load_sentences(self, pdf_path: str) -> List[str]:
        """
        Load sentences from a JSONL file using the hash-based paper ID.
        """
        meta = self.extract_pdf_metadata(pdf_path)
        title = meta.get("title", "Unknown Title")
        year = meta.get("year", "Unknown Year")
        paper_id = self._generate_paper_id(title, year)
        path = os.path.join(self.storage_root, paper_id, "sentences.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            return [json.loads(line)["text"] for line in f.readlines()]
        
    def upload_pdf(self, pdf_path: str):
        """
        Main function to upload a PDF file, extract metadata, and save sentences.
        """
        pdf_processor.register_pdf(pdf_path=pdf_path)
        sentences = pdf_processor.parse_sentences(pdf_path)
        pdf_processor.save_sentences(pdf_path, sentences)

    def diagnose_pdf_quality(self, pdf_path: str) -> Dict:
        """
        Diagnose the quality of PDF processing for debugging and monitoring.
        Returns detailed quality metrics and recommendations.
        """
        report = {
            "pdf_path": pdf_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata_quality": {},
            "text_extraction_quality": {},
            "sentence_parsing_quality": {},
            "recommendations": []
        }
        
        try:
            # Test metadata extraction
            metadata = self.extract_pdf_metadata(pdf_path)
            report["metadata_quality"] = {
                "extraction_method": metadata.get("extraction_method", "unknown"),
                "has_valid_title": len(metadata.get("title", "")) > 5,
                "has_valid_authors": len(metadata.get("authors", [])) > 0,
                "has_valid_year": self._is_valid_year(metadata.get("year", "")),
                "has_doi": metadata.get("doi", "") != "Unknown DOI",
                "errors": metadata.get("errors", [])
            }
            
            # Test text extraction
            reader = PdfReader(pdf_path)
            pages_with_text = sum(1 for page in reader.pages if page.extract_text())
            total_pages = len(reader.pages)
            
            full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            main_content, ref_section = self._separate_main_content_and_references(full_text)
            
            report["text_extraction_quality"] = {
                "total_pages": total_pages,
                "pages_with_text": pages_with_text,
                "text_extraction_ratio": pages_with_text / total_pages if total_pages > 0 else 0,
                "total_text_length": len(full_text),
                "main_content_length": len(main_content),
                "reference_section_length": len(ref_section),
                "reference_section_detected": len(ref_section) > 100
            }
            
            # Test sentence parsing
            from nltk.tokenize import sent_tokenize
            sentences = self.parse_sentences(pdf_path)
            raw_sentences = sent_tokenize(full_text)
            
            report["sentence_parsing_quality"] = {
                "raw_sentences_count": len(raw_sentences),
                "filtered_sentences_count": len(sentences),
                "filtering_ratio": (len(raw_sentences) - len(sentences)) / len(raw_sentences) if len(raw_sentences) > 0 else 0,
                "average_sentence_length": sum(len(s) for s in sentences) / len(sentences) if sentences else 0,
                "has_sentences": len(sentences) > 0
            }
            
            # Generate recommendations
            recommendations = []
            if report["metadata_quality"]["extraction_method"] == "fallback":
                recommendations.append("Consider improving filename format for better metadata extraction")
            
            if report["text_extraction_quality"]["text_extraction_ratio"] < 0.8:
                recommendations.append("PDF may have scanning or formatting issues affecting text extraction")
            
            if report["sentence_parsing_quality"]["filtering_ratio"] > 0.5:
                recommendations.append("High sentence filtering ratio - review filtering rules")
            
            if not report["text_extraction_quality"]["reference_section_detected"]:
                recommendations.append("Reference section not automatically detected - manual review recommended")
                
            report["recommendations"] = recommendations
            
        except Exception as e:
            report["error"] = str(e)
            report["recommendations"] = ["PDF processing failed - manual inspection required"]
        
        return report

    def _is_valid_year(self, year_str: str) -> bool:
        """Helper to validate year string."""
        try:
            year = int(year_str)
            return 1900 <= year <= datetime.now().year + 1
        except (ValueError, TypeError):
            return False

    def save_sentences_with_quality_check(self, pdf_path: str, sentences: List[str]):
        """
        Enhanced version of save_sentences that includes quality checks and metadata.
        """
        # Generate quality report
        quality_report = self.diagnose_pdf_quality(pdf_path)
        
        # Get metadata
        meta = self.extract_pdf_metadata(pdf_path)
        title = meta.get("title", "Unknown Title")
        year = meta.get("year", "Unknown Year")
        paper_id = self._generate_paper_id(title, year)
        
        # Create output directory
        paper_dir = os.path.join(self.storage_root, paper_id)
        os.makedirs(paper_dir, exist_ok=True)
        
        # Save sentences with enhanced metadata
        sentences_path = os.path.join(paper_dir, "sentences.jsonl")
        with open(sentences_path, "w") as f:
            for idx, sent in enumerate(sentences):
                sentence_data = {
                    "index": idx,
                    "text": sent,
                    "length": len(sent),
                    "word_count": len(sent.split())
                }
                f.write(json.dumps(sentence_data) + "\n")
        
        # Save quality report
        quality_path = os.path.join(paper_dir, "quality_report.json")
        with open(quality_path, "w") as f:
            json.dump(quality_report, f, indent=2)
        
        logging.info(f"Saved {len(sentences)} sentences for paper {paper_id}")
        logging.info(f"Quality report saved to {quality_path}")
        
        # Log warnings if quality issues detected
        if quality_report.get("recommendations"):
            logging.warning(f"Quality issues detected for {paper_id}:")
            for rec in quality_report["recommendations"]:
                logging.warning(f"  - {rec}")

    def _extract_text_with_pymupdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text using PyMuPDF (fitz) - best for complex layouts and academic papers.
        Returns (full_text, extraction_info)
        """
        doc = fitz.open(pdf_path)
        pages_text = []
        extraction_info = {
            "engine": "pymupdf",
            "total_pages": len(doc),
            "pages_with_text": 0,
            "has_images": False,
            "text_blocks": 0
        }
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text with layout preservation
            text = page.get_text("text")  # or "dict" for more structure
            if text.strip():
                pages_text.append(text)
                extraction_info["pages_with_text"] += 1
            
            # Check for images (useful for scanned documents)
            if page.get_images():
                extraction_info["has_images"] = True
            
            # Count text blocks for layout complexity assessment
            blocks = page.get_text("dict")["blocks"]
            extraction_info["text_blocks"] += len([b for b in blocks if "lines" in b])
        
        doc.close()
        full_text = "\n".join(pages_text)
        return full_text, extraction_info

    def _extract_text_with_pdfplumber(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text using pdfplumber - excellent for tables and precise layout.
        Returns (full_text, extraction_info)
        """
        pages_text = []
        extraction_info = {
            "engine": "pdfplumber",
            "total_pages": 0,
            "pages_with_text": 0,
            "tables_found": 0,
            "figures_found": 0
        }
        
        with pdfplumber.open(pdf_path) as pdf:
            extraction_info["total_pages"] = len(pdf.pages)
            
            for page in pdf.pages:
                # Extract text while preserving layout
                text = page.extract_text()
                if text:
                    pages_text.append(text)
                    extraction_info["pages_with_text"] += 1
                
                # Count tables and figures
                tables = page.find_tables()
                extraction_info["tables_found"] += len(tables)
                
                # Look for figure indicators
                if text and re.search(r'\bfigure\s+\d+\b', text, re.IGNORECASE):
                    extraction_info["figures_found"] += 1
        
        full_text = "\n".join(pages_text)
        return full_text, extraction_info

    def _extract_text_with_pypdf2(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text using PyPDF2 - basic fallback option.
        Returns (full_text, extraction_info)
        """
        reader = PdfReader(pdf_path)
        pages_text = []
        extraction_info = {
            "engine": "pypdf2",
            "total_pages": len(reader.pages),
            "pages_with_text": 0,
            "encrypted": reader.is_encrypted
        }
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
                extraction_info["pages_with_text"] += 1
        
        full_text = "\n".join(pages_text)
        return full_text, extraction_info

    def _extract_text_with_pymupdf_ocr(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text using PyMuPDF + OCR for scanned documents.
        This method is specifically for image-based PDFs.
        """
        doc = fitz.open(pdf_path)
        pages_text = []
        extraction_info = {
            "engine": "pymupdf_ocr",
            "total_pages": len(doc),
            "pages_with_text": 0,
            "pages_with_images": 0,
            "ocr_pages": 0,
            "has_images": False
        }
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # First try normal text extraction
            text = page.get_text("text").strip()
            
            # If no text or very little text, try OCR
            if len(text) < 50:  # Threshold for "no meaningful text"
                try:
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
                    img_data = pix.tobytes("ppm")
                    
                    # Convert to PIL Image
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    ocr_text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
                    
                    if ocr_text.strip():
                        text = ocr_text.strip()
                        extraction_info["ocr_pages"] += 1
                        logging.info(f"OCR extracted {len(ocr_text)} characters from page {page_num + 1}")
                    
                except Exception as e:
                    logging.warning(f"OCR failed for page {page_num + 1}: {e}")
            
            if text:
                pages_text.append(text)
                extraction_info["pages_with_text"] += 1
            
            # Check for images
            if page.get_images():
                extraction_info["has_images"] = True
                extraction_info["pages_with_images"] += 1
        
        doc.close()
        full_text = "\n".join(pages_text)
        
        logging.info(f"OCR extraction completed: {extraction_info['ocr_pages']} pages OCRed, "
                    f"{extraction_info['pages_with_images']} pages with images")
        
        return full_text, extraction_info

    def _detect_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Detect if PDF is likely a scanned document (image-based).
        Returns True if PDF appears to be scanned.
        """
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            pages_with_little_text = 0
            pages_with_images = 0
            
            # Sample first few pages
            sample_pages = min(5, total_pages)
            
            for page_num in range(sample_pages):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                images = page.get_images()
                
                if len(text) < 100:  # Very little extractable text
                    pages_with_little_text += 1
                
                if images:
                    pages_with_images += 1
            
            doc.close()
            
            # Heuristics for scanned PDF detection
            text_ratio = pages_with_little_text / sample_pages
            image_ratio = pages_with_images / sample_pages
            
            is_scanned = (text_ratio > 0.6) and (image_ratio > 0.3)
            
            if is_scanned:
                logging.info(f"PDF appears to be scanned: {text_ratio:.1%} pages with little text, "
                           f"{image_ratio:.1%} pages with images")
            
            return is_scanned
            
        except Exception as e:
            logging.warning(f"Could not analyze PDF for scan detection: {e}")
            return False

    def extract_text_with_best_engine(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Try multiple PDF engines and return the best result based on quality metrics.
        Automatically detects scanned PDFs and prioritizes OCR when needed.
        """
        # First, detect if this is a scanned PDF
        is_scanned = self._detect_scanned_pdf(pdf_path) if HAS_PYMUPDF else False
        
        results = []
        engines_to_try = self.available_engines.copy()
        
        # If scanned PDF detected, prioritize OCR engines
        if is_scanned and HAS_OCR:
            logging.info("Scanned PDF detected - prioritizing OCR extraction")
            ocr_engines = [(name, func) for name, func in engines_to_try if "ocr" in name]
            non_ocr_engines = [(name, func) for name, func in engines_to_try if "ocr" not in name]
            engines_to_try = ocr_engines + non_ocr_engines
        
        for engine_name, engine_func in engines_to_try:
            try:
                logging.info(f"Trying PDF extraction with {engine_name}")
                text, info = engine_func(pdf_path)
                
                # Calculate quality score
                quality_score = self._calculate_text_quality_score(text, info)
                
                results.append({
                    "engine": engine_name,
                    "text": text,
                    "info": info,
                    "quality_score": quality_score
                })
                
                logging.info(f"{engine_name} quality score: {quality_score:.2f}")
                
                # For scanned PDFs, if OCR gives decent results, use it immediately
                if is_scanned and "ocr" in engine_name and quality_score > 3.0:
                    logging.info(f"OCR extraction successful for scanned PDF, using {engine_name}")
                    break
                
            except Exception as e:
                logging.warning(f"{engine_name} extraction failed: {e}")
                continue
        
        if not results:
            raise RuntimeError("All PDF extraction engines failed")
        
        # Select best result based on quality score
        best_result = max(results, key=lambda x: x["quality_score"])
        logging.info(f"Selected {best_result['engine']} as best extraction method")
        
        return best_result["text"], best_result["info"]

    def _calculate_text_quality_score(self, text: str, info: Dict) -> float:
        """
        Calculate quality score for extracted text to determine best engine.
        Higher score = better quality.
        """
        if not text:
            return 0.0
        
        score = 0.0
        
        # Basic text metrics
        text_length = len(text)
        word_count = len(text.split())
        
        # Length bonus (more text usually better, up to a point)
        score += min(text_length / 10000, 5.0)  # Max 5 points for length
        
        # Word density (good text has reasonable word density)
        if text_length > 0:
            word_density = word_count / text_length * 100
            if 10 <= word_density <= 25:  # Reasonable range
                score += 3.0
            else:
                score += max(0, 3.0 - abs(word_density - 17.5) / 5)
        
        # Coverage score (pages with text vs total pages)
        coverage = info.get("pages_with_text", 0) / max(info.get("total_pages", 1), 1)
        score += coverage * 3.0  # Max 3 points for coverage
        
        # Engine-specific bonuses
        if info.get("engine") == "pymupdf":
            score += 1.0  # Slight preference for pymupdf
            if info.get("text_blocks", 0) > 0:
                score += 0.5  # Bonus for structured extraction
        
        if info.get("engine") == "pymupdf_ocr":
            # OCR-specific scoring
            ocr_pages = info.get("ocr_pages", 0)
            total_pages = info.get("total_pages", 1)
            if ocr_pages > 0:
                score += 2.0  # Bonus for successful OCR
                # Extra bonus if OCR was needed for most pages
                if ocr_pages / total_pages > 0.5:
                    score += 1.0
        
        if info.get("engine") == "pdfplumber":
            if info.get("tables_found", 0) > 0:
                score += 1.0  # Bonus for table detection
        
        # Penalty for likely corrupted text
        corruption_indicators = [
            len(re.findall(r'[^\w\s.,;:!?()-]', text)) / len(text),  # Special chars ratio
            len(re.findall(r'\b\w{1,2}\b', text)) / word_count if word_count > 0 else 0,  # Short words ratio
        ]
        
        for indicator in corruption_indicators:
            if indicator > 0.3:
                score -= 2.0
        
        return max(score, 0.0)

    def diagnose_pdf_and_recommend(self, pdf_path: str) -> Dict:
        """
        Comprehensive PDF analysis with processing recommendations.
        Returns detailed diagnosis and suggestions for optimal processing.
        """
        diagnosis = {
            "pdf_path": pdf_path,
            "file_size_mb": round(os.path.getsize(pdf_path) / (1024*1024), 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pdf_type": "unknown",
            "recommendations": [],
            "engine_results": {},
            "quality_scores": {},
            "is_processable": True
        }
        
        try:
            # Basic PDF analysis
            if HAS_PYMUPDF:
                doc = fitz.open(pdf_path)
                diagnosis["total_pages"] = len(doc)
                
                # Analyze first few pages
                sample_pages = min(3, len(doc))
                text_chars = 0
                image_count = 0
                
                for i in range(sample_pages):
                    page = doc.load_page(i)
                    text = page.get_text("text")
                    images = page.get_images()
                    text_chars += len(text)
                    image_count += len(images)
                
                doc.close()
                
                # Determine PDF type
                avg_text_per_page = text_chars / sample_pages if sample_pages > 0 else 0
                avg_images_per_page = image_count / sample_pages if sample_pages > 0 else 0
                
                if avg_text_per_page < 100 and avg_images_per_page > 0:
                    diagnosis["pdf_type"] = "scanned_document"
                    diagnosis["recommendations"].append("Scanned PDF detected - OCR processing recommended")
                elif avg_text_per_page > 1000:
                    diagnosis["pdf_type"] = "native_text"
                    diagnosis["recommendations"].append("Native text PDF - standard extraction sufficient")
                elif avg_images_per_page > 2:
                    diagnosis["pdf_type"] = "mixed_content"
                    diagnosis["recommendations"].append("Mixed content PDF - may need hybrid processing")
                else:
                    diagnosis["pdf_type"] = "low_quality_text"
                    diagnosis["recommendations"].append("Low quality text - may benefit from OCR assistance")
            
            # Test all available engines
            for engine_name, engine_func in self.available_engines:
                try:
                    text, info = engine_func(pdf_path)
                    quality_score = self._calculate_text_quality_score(text, info)
                    
                    diagnosis["engine_results"][engine_name] = {
                        "success": True,
                        "text_length": len(text),
                        "quality_score": round(quality_score, 2),
                        "info": info
                    }
                    diagnosis["quality_scores"][engine_name] = quality_score
                    
                except Exception as e:
                    diagnosis["engine_results"][engine_name] = {
                        "success": False,
                        "error": str(e),
                        "quality_score": 0.0
                    }
                    diagnosis["quality_scores"][engine_name] = 0.0
            
            # Determine best engine
            if diagnosis["quality_scores"]:
                best_engine = max(diagnosis["quality_scores"], key=diagnosis["quality_scores"].get)
                best_score = diagnosis["quality_scores"][best_engine]
                
                diagnosis["recommended_engine"] = best_engine
                diagnosis["best_quality_score"] = best_score
                
                if best_score < 2.0:
                    diagnosis["recommendations"].append("Low quality extraction - consider manual preprocessing")
                    diagnosis["is_processable"] = False
                elif best_score < 4.0:
                    diagnosis["recommendations"].append("Moderate quality - may need post-processing cleanup")
                else:
                    diagnosis["recommendations"].append("Good quality extraction expected")
            
            # Specific recommendations based on PDF type
            if diagnosis["pdf_type"] == "scanned_document":
                if HAS_OCR:
                    diagnosis["recommendations"].append("Use pymupdf_ocr engine for best results")
                else:
                    diagnosis["recommendations"].append("Install pytesseract for OCR support")
                    diagnosis["is_processable"] = False
            
        except Exception as e:
            diagnosis["error"] = str(e)
            diagnosis["recommendations"].append("PDF analysis failed - file may be corrupted")
            diagnosis["is_processable"] = False
        
        return diagnosis

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    pdf_processor = PDFProcessor()
    # pdf_path = "test_files/Rivkin - 2000 - Imitation of Complex Strategies.pdf"
    # pdf_processor.upload_pdf(pdf_path)
    pdf_path = "test_files/Porter - Competitive Strategy.pdf"
    pdf_processor.upload_pdf(pdf_path)
 