import sqlite3

from src.kernel.service import CiteWeaveKernel
from src.storage.author_paper_index import AuthorPaperIndex


def _insert_paper(cursor, paper_id, title, pdf_path):
    cursor.execute(
        """
        INSERT INTO papers (paper_id, title, year, journal, pdf_path, processed_date)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (paper_id, title, 2024, "Journal", pdf_path, "2026-05-14"),
    )


def test_paper_index_snapshot_filters_by_pdf_status_without_exposing_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    AuthorPaperIndex(index_db_path="data/author_paper_index.db")
    with sqlite3.connect("data/author_paper_index.db") as conn:
        cursor = conn.cursor()
        _insert_paper(cursor, "with-pdf", "Paper With PDF", "/private/local/paper.pdf")
        _insert_paper(cursor, "missing-pdf", "Paper Missing PDF", None)
        conn.commit()

    kernel = CiteWeaveKernel()

    available = kernel.paper_index_snapshot(pdf_status="available", limit=0)
    missing = kernel.paper_index_snapshot(pdf_status="missing", limit=0)

    assert available["pdf_status_filter"] == "available"
    assert [paper["paper_id"] for paper in available["papers"]] == ["with-pdf"]
    assert available["papers"][0]["pdf_available"] is True
    assert "pdf_path" not in available["papers"][0]

    assert missing["pdf_status_filter"] == "missing"
    assert [paper["paper_id"] for paper in missing["papers"]] == ["missing-pdf"]
    assert missing["papers"][0]["pdf_available"] is False
    assert "pdf_path" not in missing["papers"][0]


def test_paper_index_snapshot_filters_title_separately_from_broad_search(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    AuthorPaperIndex(index_db_path="data/author_paper_index.db")
    with sqlite3.connect("data/author_paper_index.db") as conn:
        cursor = conn.cursor()
        _insert_paper(cursor, "network-theory", "Network Theory of Organization", None)
        _insert_paper(cursor, "network-methods", "Relational Network Methods", None)
        conn.commit()

    snapshot = CiteWeaveKernel().paper_index_snapshot(search="network", title="organization", limit=0)

    assert snapshot["search_filter"] == "network"
    assert snapshot["title_filter"] == "organization"
    assert [paper["paper_id"] for paper in snapshot["papers"]] == ["network-theory"]
