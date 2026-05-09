from src.agents.multi_agent_research_system import LangGraphResearchSystem


class FakeQueryAgent:
    def get_papers_id_by_title(self, title):
        return {"status": "no_match", "query": title}

    def list_uploaded_papers(self, limit=5):
        return [
            {
                "paper_id": "paper-1",
                "title": "Uploaded Sample",
                "citation_sentence_count": 2,
                "citation_relation_count": 1,
            }
        ]

    def get_papers_cited_by_paper(self, paper_id):
        assert paper_id == "paper-1"
        return [{"paper_id": "cited-1", "title": "Cited Paper"}]

    def get_sentences_with_citations_from_paper(self, paper_id, count=50):
        assert paper_id == "paper-1"
        return [{"text": "Example citation sentence", "citations": [{"paper_id": "cited-1"}]}]


def test_citation_analysis_falls_back_to_uploaded_paper_without_tool_calls():
    system = LangGraphResearchSystem.__new__(LangGraphResearchSystem)
    system.query_agent = FakeQueryAgent()

    state = {
        "question": "What citations were extracted from the uploaded sample paper?",
        "request_id": "test-request",
    }
    query_intent = {
        "query_type": "citation_analysis",
        "target_entity": "uploaded sample paper",
        "entity_type": "paper",
        "extracted_entities": {"paper_titles": ["uploaded sample paper"]},
    }

    next_state = system._execute_tools_based_on_intent(state, query_intent)
    results = next_state["collected_data"]["results"]

    assert results["paper_resolution"]["resolution_strategy"] == "uploaded_paper_fallback"
    assert results["source_paper"]["paper_id"] == "paper-1"
    assert results["get_papers_cited_by_paper"]["count"] == 1
    assert results["get_citation_sentences_from_paper"]["count"] == 1
