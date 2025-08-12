import os
import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional


class StateDB:
    """SQLite-backed state store for documents, jobs, and deletions."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    authors_json TEXT,
                    year INTEGER,
                    doi TEXT,
                    original_filename TEXT,
                    new_filename TEXT,
                    local_path TEXT,
                    collection TEXT,
                    status TEXT,
                    file_hash TEXT,
                    norm_title TEXT,
                    created_at INTEGER,
                    updated_at INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
                CREATE UNIQUE INDEX IF NOT EXISTS uniq_documents_collection_pid ON documents(collection, paper_id);

                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    created_at INTEGER,
                    updated_at INTEGER
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT,
                    filename TEXT,
                    progress INTEGER,
                    stage TEXT,
                    done INTEGER,
                    success INTEGER,
                    error TEXT,
                    updated_at INTEGER,
                    FOREIGN KEY(paper_id) REFERENCES documents(paper_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS deletions (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT,
                    requested_at INTEGER,
                    finished_at INTEGER,
                    ok INTEGER,
                    error TEXT
                );
                """
            )
            # Lightweight migrations for newly added columns
            try:
                cols = [r[1] for r in conn.execute("PRAGMA table_info('documents');").fetchall()]
                if 'original_filename' not in cols:
                    conn.execute("ALTER TABLE documents ADD COLUMN original_filename TEXT;")
                if 'file_hash' not in cols:
                    conn.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT;")
                if 'norm_title' not in cols:
                    conn.execute("ALTER TABLE documents ADD COLUMN norm_title TEXT;")
                # Create hash index only when column exists (older DBs won't break)
                if 'file_hash' in cols:
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection_hash ON documents(collection, file_hash);")
            except Exception:
                pass

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # Documents
    def upsert_document(self, doc: Dict[str, Any]) -> None:
        with self._lock, self._conn() as conn:
            fields = [
                'paper_id','title','authors_json','year','doi','original_filename','new_filename','local_path',
                'collection','status','file_hash','norm_title','created_at','updated_at'
            ]
            values = [doc.get(k) for k in fields]
            placeholders = ','.join('?' for _ in fields)
            updates = ','.join(f"{k}=excluded.{k}" for k in fields if k != 'paper_id')
            conn.execute(
                f"INSERT INTO documents({','.join(fields)}) VALUES({placeholders}) "
                f"ON CONFLICT(paper_id) DO UPDATE SET {updates};",
                values,
            )

    def set_document_status(self, paper_id: str, status: str, updated_at: int) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE documents SET status=?, updated_at=? WHERE paper_id=?;", (status, updated_at, paper_id))

    def list_documents(self, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM documents"
        args: List[Any] = []
        if collection:
            query += " WHERE collection=?"
            args.append(collection)
        query += " ORDER BY updated_at DESC;"
        with self._conn() as conn:
            rows = conn.execute(query, args).fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, paper_id: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE paper_id=?;", (paper_id,))

    # Collections
    def ensure_collection(self, name: str, ts: Optional[int] = None) -> None:
        if not name:
            return
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO collections(name, created_at, updated_at) VALUES(?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET updated_at=excluded.updated_at;",
                (name, ts or 0, ts or 0),
            )

    def list_collections(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.name AS name,
                       c.created_at AS created_at,
                       c.updated_at AS updated_at,
                       COUNT(d.paper_id) AS document_count
                FROM collections c
                LEFT JOIN documents d ON d.collection = c.name
                GROUP BY c.name
                ORDER BY c.created_at ASC, c.name ASC;
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def rename_collection(self, old_name: str, new_name: str, ts: Optional[int] = None) -> None:
        if not old_name or not new_name or old_name == new_name:
            return
        with self._lock, self._conn() as conn:
            conn.execute("BEGIN;")
            try:
                # Preserve original created_at from old_name strictly
                row = conn.execute("SELECT created_at FROM collections WHERE name=?;", (old_name,)).fetchone()
                original_created = row[0] if row else 0
                conn.execute(
                    "INSERT INTO collections(name, created_at, updated_at) VALUES(?, ?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET updated_at=excluded.updated_at;",
                    (new_name, original_created, ts or original_created),
                )
                conn.execute("UPDATE documents SET collection=? WHERE collection=?;", (new_name, old_name))
                conn.execute("DELETE FROM collections WHERE name=?;", (old_name,))
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    # Jobs
    def upsert_job(self, job: Dict[str, Any]) -> None:
        with self._lock, self._conn() as conn:
            fields = ['id','paper_id','filename','progress','stage','done','success','error','updated_at']
            values = [job.get(k) for k in fields]
            placeholders = ','.join('?' for _ in fields)
            updates = ','.join(f"{k}=excluded.{k}" for k in fields if k != 'id')
            conn.execute(
                f"INSERT INTO jobs({','.join(fields)}) VALUES({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates};",
                values,
            )

    def delete_jobs_by_paper(self, paper_id: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM jobs WHERE paper_id=?;", (paper_id,))

    # Deletions
    def create_deletion(self, deletion: Dict[str, Any]) -> None:
        with self._lock, self._conn() as conn:
            fields = ['id','paper_id','requested_at','finished_at','ok','error']
            values = [deletion.get(k) for k in fields]
            placeholders = ','.join('?' for _ in fields)
            conn.execute(
                f"INSERT INTO deletions({','.join(fields)}) VALUES({placeholders});",
                values,
            )

    def finish_deletion(self, deletion_id: str, ok: bool, finished_at: int, error: Optional[str] = None) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE deletions SET finished_at=?, ok=?, error=? WHERE id=?;", (finished_at, 1 if ok else 0, error, deletion_id))

    def get_deletion(self, deletion_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM deletions WHERE id=?;", (deletion_id,)).fetchone()
            return dict(row) if row else None


