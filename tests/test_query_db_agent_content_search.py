#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from src.agents.query_db_agent import QueryDBAgent


class QueryDBAgentContentSearchTests(unittest.TestCase):
    def setUp(self):
        self.agent = QueryDBAgent.__new__(QueryDBAgent)

    def test_extract_search_candidates_handles_single_and_multiple_match_shapes(self):
        single = {
            "status": "single_match",
            "paper_info": {"paper_id": "p1", "title": "Paper One"},
        }
        multiple = {
            "status": "multiple_matches",
            "candidates": [
                {"paper_id": "p1", "title": "Paper One"},
                {"paper_id": "p2", "title": "Paper Two"},
            ],
        }

        self.assertEqual(
            self.agent._extract_search_candidates(single),
            [{"paper_id": "p1", "title": "Paper One"}],
        )
        self.assertEqual(
            self.agent._extract_search_candidates(multiple),
            [
                {"paper_id": "p1", "title": "Paper One"},
                {"paper_id": "p2", "title": "Paper Two"},
            ],
        )
        self.assertEqual(self.agent._extract_search_candidates({"status": "no_match"}), [])

    def test_query_pdf_by_title_and_content_uses_single_match_result_schema(self):
        self.agent.get_papers_id_by_title = lambda title: {
            "status": "single_match",
            "paper_info": {"paper_id": "paper-1", "title": title},
        }
        self.agent.query_pdf_content = lambda paper_id, query: {
            "found": True,
            "paper_id": paper_id,
            "query": query,
            "total_matches": 2,
            "data": [{"section_title": "Intro", "matches": [{"context": "x"}]}],
        }

        result = self.agent.query_pdf_by_title_and_content("Test Paper", "main argument")

        self.assertTrue(result["found"])
        self.assertEqual(result["papers_found"], 1)
        self.assertEqual(result["papers_with_content"], 1)
        self.assertEqual(result["data"][0]["paper_metadata"]["paper_id"], "paper-1")

    def test_query_pdf_by_author_and_content_uses_multiple_match_result_schema(self):
        self.agent.get_papers_id_by_author = lambda author: {
            "status": "multiple_matches",
            "candidates": [
                {"paper_id": "paper-1", "title": "Paper One"},
                {"paper_id": "paper-2", "title": "Paper Two"},
            ],
        }
        self.agent.query_pdf_content = lambda paper_id, query: {
            "found": paper_id == "paper-2",
            "paper_id": paper_id,
            "query": query,
            "total_matches": 1 if paper_id == "paper-2" else 0,
            "data": [],
        }

        result = self.agent.query_pdf_by_author_and_content("Porter", "competitive advantage")

        self.assertTrue(result["found"])
        self.assertEqual(result["papers_found"], 2)
        self.assertEqual(result["papers_with_content"], 1)
        self.assertEqual(result["data"][0]["paper_metadata"]["paper_id"], "paper-2")

    def test_get_full_pdf_content_accepts_alternative_section_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paper_dir = Path(tmpdir) / "paper-1"
            paper_dir.mkdir(parents=True)
            (paper_dir / "processed_document.json").write_text(
                json.dumps(
                    {
                        "metadata": {"title": "Test Paper", "authors": ["A. Author"]},
                        "sections": [
                            {
                                "index": 1,
                                "title": "Abstract",
                                "type": "abstract",
                                "text": "This paper argues that network defenses should adapt.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.agent.papers_dir = tmpdir
            result = self.agent.get_full_pdf_content("paper-1")

        self.assertTrue(result["found"])
        self.assertEqual(result["sections_count"], 1)
        self.assertIn("## Abstract", result["full_text"])
        self.assertGreater(result["total_word_count"], 0)
        self.assertEqual(result["section_summaries"][0]["section_type"], "abstract")


if __name__ == "__main__":
    unittest.main()
