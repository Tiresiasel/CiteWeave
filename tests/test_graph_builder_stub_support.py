from pathlib import Path
import importlib.util
import uuid


GRAPH_BUILDER_PATH = Path(__file__).resolve().parents[1] / "src" / "storage" / "graph_builder.py"


def _load_module():
    module_name = f"graph_builder_stub_support_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, GRAPH_BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeResult:
    def __init__(self, single_row=None, rows=None):
        self._single_row = single_row
        self._rows = rows or []

    def single(self):
        return self._single_row

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, recorder, result):
        self._recorder = recorder
        self._result = result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        self._recorder.append((query, params))
        return self._result


class _FakeDriver:
    def __init__(self, recorder, result):
        self._recorder = recorder
        self._result = result

    def session(self):
        return _FakeSession(self._recorder, self._result)


def test_graph_builder_lists_stub_papers_with_limit():
    module = _load_module()
    recorder = []
    fake_rows = [{"paper_id": "porter_1980", "title": "Competitive Strategy", "cited_by_count": 3}]
    db = module.GraphDB.__new__(module.GraphDB)
    db.driver = _FakeDriver(recorder, _FakeResult(rows=fake_rows))

    rows = db.list_stub_papers(limit=5)

    assert rows == fake_rows
    assert recorder[0][1]["limit"] == 5
    assert "WHERE coalesce(p.stub, false) = true" in recorder[0][0]


def test_graph_builder_returns_citation_network_stats_dict():
    module = _load_module()
    recorder = []
    expected = {
        "total_papers": 8,
        "uploaded_papers": 5,
        "stub_papers": 3,
        "total_citation_relations": 12,
        "total_citation_instances": 15,
    }
    db = module.GraphDB.__new__(module.GraphDB)
    db.driver = _FakeDriver(recorder, _FakeResult(single_row=expected))

    stats = db.get_citation_network_stats()

    assert stats == expected
    assert "total_citation_instances" in recorder[0][0]


def test_graph_builder_can_resolve_stub_papers():
    module = _load_module()
    recorder = []
    db = module.GraphDB.__new__(module.GraphDB)
    db.driver = _FakeDriver(recorder, _FakeResult())

    db.update_paper_from_stub(
        paper_id="porter_1980",
        title="Competitive Strategy",
        authors=["Michael Porter"],
        year=1980,
        doi="10.1000/porter",
    )

    query, params = recorder[0]
    assert "p.stub = false" in query
    assert params["paper_id"] == "porter_1980"
    assert params["doi"] == "10.1000/porter"


def test_graph_builder_finds_papers_by_author_year_with_stable_shape():
    module = _load_module()
    recorder = []
    fake_rows = [{"paper_id": "porter_1980", "id": "porter_1980", "title": "Competitive Strategy", "authors": ["Michael Porter"], "year": 1980, "stub": False}]
    db = module.GraphDB.__new__(module.GraphDB)
    db.driver = _FakeDriver(recorder, _FakeResult(rows=fake_rows))

    rows = db.find_papers_by_author_year("Porter", year=1980, fuzzy=True, limit=5)

    assert rows == fake_rows
    query, params = recorder[0]
    assert "coalesce(p.authors, [])" in query
    assert "p.year = $year" in query
    assert params == {"author_name": "Porter", "year": 1980, "fuzzy": True, "limit": 5}


def test_graph_builder_finds_citations_by_target_paper():
    module = _load_module()
    recorder = []
    fake_rows = [{"source_id": "s1", "relationship_type": "CITES", "cited_paper_id": "porter_1980"}]
    db = module.GraphDB.__new__(module.GraphDB)
    db.driver = _FakeDriver(recorder, _FakeResult(rows=fake_rows))

    rows = db.find_citations("porter_1980", limit=10)

    assert rows == fake_rows
    query, params = recorder[0]
    assert "CITES|RELATES" in query
    assert params == {"cited_paper_id": "porter_1980", "limit": 10}
