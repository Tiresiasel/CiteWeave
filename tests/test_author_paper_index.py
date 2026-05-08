from pathlib import Path

from src.storage.author_paper_index import AuthorPaperIndex


def test_author_paper_index_finds_managed_original_pdf(tmp_path):
    storage_root = tmp_path / "papers"
    paper_dir = storage_root / "paper-1"
    paper_dir.mkdir(parents=True)
    original_pdf = paper_dir / "original.pdf"
    original_pdf.write_bytes(b"%PDF-1.4\n")

    index = AuthorPaperIndex(
        storage_root=str(storage_root),
        index_db_path=str(tmp_path / "index" / "authors.sqlite"),
    )

    assert index._find_original_pdf("paper-1", {}) == str(original_pdf.resolve())


def test_author_paper_index_does_not_depend_on_test_files_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_files_pdf = Path("test_files") / "paper-1.pdf"
    test_files_pdf.parent.mkdir()
    test_files_pdf.write_bytes(b"%PDF-1.4\n")

    index = AuthorPaperIndex(
        storage_root=str(tmp_path / "papers"),
        index_db_path=str(tmp_path / "index" / "authors.sqlite"),
    )

    assert index._find_original_pdf("paper-1", {}) is None
