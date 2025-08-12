"""
Flask API server exposing CiteWeave core features via HTTP endpoints.
Endpoints:
- POST /api/v1/upload: Upload and process a PDF
- POST /api/v1/diagnose: Diagnose a PDF for processing quality
- POST /api/v1/chat: Stateless interactive research chat turn
- GET  /api/v1/health: Health check
"""

import os
import json
import time
import threading
from typing import List
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.processing.pdf.document_processor import DocumentProcessor
from src.agents.multi_agent_research_system import LangGraphResearchSystem
from src.storage.state_db import StateDB


def create_app() -> Flask:
    """Create and configure the Flask app instance."""
    app = Flask(__name__, static_folder=None)
    CORS(app)

    # Secret key for sessions (basic use; not storing sensitive auth server-side in this version)
    app.secret_key = os.environ.get("CITEWEAVE_FLASK_SECRET", "change-me-dev-secret")

    # Data dir and settings
    data_dir = Path(os.environ.get("CITEWEAVE_DATA_DIR", os.path.join(os.getcwd(), "data")))
    data_dir.mkdir(parents=True, exist_ok=True)
    settings_path = Path(os.environ.get("CITEWEAVE_SETTINGS_PATH", str(data_dir / "settings.json")))
    model_config_path = Path(os.environ.get("CITEWEAVE_MODEL_CONFIG_PATH", os.path.join(os.getcwd(), "config", "model_config.json")))
    env_file_path = Path(os.environ.get("CITEWEAVE_ENV_FILE", str(data_dir / ".env")))
    jobs_state_path = Path(os.environ.get("CITEWEAVE_JOBS_FILE", str(data_dir / "jobs.json")))
    db_path = Path(os.environ.get("CITEWEAVE_DB_PATH", str(data_dir / "state.db")))
    default_collection = os.environ.get("CITEWEAVE_DEFAULT_COLLECTION", "Default")

    def _read_settings() -> dict:
        try:
            if settings_path.exists():
                with open(settings_path, 'r', encoding='utf-8') as f:
                    return json.load(f) or {}
        except Exception:
            pass
        return {}

    def _write_settings(d: dict):
        try:
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to persist settings: {e}")

    def _read_model_config() -> dict:
        try:
            if model_config_path.exists():
                with open(model_config_path, 'r', encoding='utf-8') as f:
                    return json.load(f) or {}
        except Exception:
            pass
        return {}

    def _write_model_config(d: dict):
        try:
            model_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_config_path, 'w', encoding='utf-8') as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to write model_config.json: {e}")

    # Jobs persistence helpers
    def _read_jobs_file() -> dict:
        try:
            if jobs_state_path.exists():
                with open(jobs_state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f) or {}
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _write_jobs_file(d: dict):
        try:
            jobs_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(jobs_state_path, 'w', encoding='utf-8') as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # .env helpers (simple parser/writer)
    def _read_env_file() -> dict:
        env: Dict[str, str] = {}
        try:
            if env_file_path.exists():
                for line in env_file_path.read_text(encoding='utf-8').splitlines():
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        env[k.strip()] = v.strip()
        except Exception:
            pass
        return env

    def _write_env_file(env: Dict[str, str]):
        try:
            lines = [f"{k}={v}" for k, v in env.items()]
            env_file_path.parent.mkdir(parents=True, exist_ok=True)
            env_file_path.write_text("\n".join(lines) + "\n", encoding='utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to write .env: {e}")

    def _set_env_var(key: str, value: Optional[str]):
        if value:
            os.environ[key] = value
            env = _read_env_file()
            env[key] = value
            _write_env_file(env)

    # Load .env first, then overlay with settings.json for compatibility
    env_vars = _read_env_file()
    if env_vars.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = env_vars['OPENAI_API_KEY']
    if env_vars.get('CITEWEAVE_PUBLIC_API_BASE'):
        os.environ['CITEWEAVE_PUBLIC_API_BASE'] = env_vars['CITEWEAVE_PUBLIC_API_BASE']
    persisted = _read_settings()
    if isinstance(persisted, dict):
        openai_key = persisted.get('openai_key')
        if openai_key:
            _set_env_var('OPENAI_API_KEY', openai_key)

    # Watcher state
    watch_state_path = data_dir / "watch_state.json"
    _watch_lock = threading.Lock()
    _scan_now_event = threading.Event()
    _scan_thread_started = False

    def _read_watch_state() -> dict:
        try:
            if watch_state_path.exists():
                with open(watch_state_path, 'r', encoding='utf-8') as f:
                    return json.load(f) or {}
        except Exception:
            pass
        return {"last_scan": None, "files": {}}

    def _write_watch_state(d: dict):
        try:
            with open(watch_state_path, 'w', encoding='utf-8') as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _list_pdfs(root: str) -> List[str]:
        try:
            pdfs: List[str] = []
            for current, _dirs, files in os.walk(root):
                for name in files:
                    if name.lower().endswith('.pdf'):
                        pdfs.append(os.path.join(current, name))
            return pdfs
        except Exception:
            return []

    def _scan_once():
        nonlocal persisted
        with _watch_lock:
            settings = _read_settings()
            watch_enabled = settings.get('watch_enabled', False)
            watch_dirs = settings.get('watch_directories', []) or []
            # Support structured mapping: [{ path, collection }]
            watch_map = settings.get('watch_map') or []
            if not watch_enabled or not watch_dirs:
                # If directories are not provided the legacy way, derive from watch_map
                if isinstance(watch_map, list) and watch_map:
                    watch_dirs = [e.get('path') for e in watch_map if isinstance(e, dict) and e.get('path')]
                if not watch_dirs:
                    return {"scanned": 0, "processed": 0}
            state = _read_watch_state()
            files_state = state.get('files', {})
            scanned = 0
            processed = 0
            processor = get_doc_processor()
            # Build mapping path->collection (fallback to default)
            path_to_collection: Dict[str, str] = {}
            path_paused: Dict[str, bool] = {}
            try:
                for e in (watch_map if isinstance(watch_map, list) else []):
                    if isinstance(e, dict) and e.get('path'):
                        abp = os.path.abspath(e['path'])
                        path_to_collection[abp] = e.get('collection') or default_collection
                        path_paused[abp] = bool(e.get('paused'))
            except Exception:
                pass
            for d in watch_dirs:
                if not d or not os.path.isdir(d):
                    continue
                for path in _list_pdfs(d):
                    try:
                        scanned += 1
                        st = os.stat(path)
                        key = os.path.abspath(path)
                        # skip paused folders
                        for root, paused in list(path_paused.items()):
                            if paused and key.startswith(root.rstrip(os.sep) + os.sep):
                                raise Exception("paused")
                        last_mtime = files_state.get(key, {}).get('mtime')
                        if last_mtime is not None and float(last_mtime) >= st.st_mtime:
                            continue
                        # Create a job entry for this file processing (progress shows in UI)
                        try:
                            job_id = f"watch_{int(time.time()*1000)}_{os.path.basename(path)}"
                            now_ts = int(time.time())
                            app.config["JOB_STATUS"][job_id] = {"done": False, "success": None, "progress": 5, "stage": "processing", "filename": os.path.basename(path), "collection": path_to_collection.get(d, default_collection), "updated_at": now_ts}
                            app.config["JOBS_DIRTY"] = True
                            try:
                                get_state_db().upsert_job({"id": job_id, "paper_id": None, "filename": os.path.basename(path), "progress": 5, "stage": "processing", "done": 0, "success": None, "error": None, "updated_at": now_ts})
                            except Exception:
                                pass
                        except Exception:
                            pass
                        # Process new or updated file
                        result = processor.process_document(path, save_results=True)
                        files_state[key] = {"mtime": st.st_mtime, "paper_id": result.get('paper_id')}
                        # After processing, upsert into DB with collection mapping
                        try:
                            pid = result.get('paper_id')
                            meta = result.get('metadata', {}) or {}
                            if pid:
                                pdir = _paper_dir(pid)
                                orig = pdir / "original.pdf"
                                target = None
                                if orig.exists():
                                    target = pdir / _sanitize_filename(meta.get("title", "Untitled"), meta.get("authors", []))
                                    if target.name != orig.name:
                                        if target.exists():
                                            target = pdir / (target.stem + f"_{pid}" + target.suffix)
                                        try:
                                            orig.rename(target)
                                        except Exception:
                                            target = orig
                                now_ts2 = int(time.time())
                                sdb = get_state_db()
                                # Resolve collection by longest matching watched root
                                col = default_collection
                                try:
                                    abs_path = os.path.abspath(path)
                                    best = None
                                    for root, c in path_to_collection.items():
                                        if abs_path.startswith(root.rstrip(os.sep) + os.sep):
                                            if best is None or len(root) > len(best):
                                                best = root; col = c
                                except Exception:
                                    col = default_collection
                                sdb.ensure_collection(col, now_ts2)
                                # Optional hash for de-duplication
                                import hashlib
                                file_hash = None
                                try:
                                    with open(path, 'rb') as fh:
                                        file_hash = hashlib.sha256(fh.read()).hexdigest()
                                except Exception:
                                    pass
                                norm = _norm_title(meta.get("title"))
                                doc = {
                                    "paper_id": pid,
                                    "title": meta.get("title"),
                                    "authors_json": json.dumps(meta.get("authors", []), ensure_ascii=False),
                                    "year": meta.get("year"),
                                    "doi": meta.get("doi"),
                                    "original_filename": os.path.basename(path),
                                    "new_filename": (target.name if target else "original.pdf"),
                                    "local_path": str((target if target else orig)),
                                    "collection": col,
                                    "status": "done",
                                    "file_hash": file_hash,
                                    "norm_title": norm,
                                    "created_at": now_ts2,
                                    "updated_at": now_ts2,
                                }
                                # Try to reuse existing doc by hash or norm_title
                                try:
                                    with sdb._conn() as conn:
                                        row = None
                                        if file_hash:
                                            row = conn.execute("SELECT paper_id FROM documents WHERE collection=? AND file_hash=? LIMIT 1;", (col, file_hash)).fetchone()
                                        if not row and norm:
                                            row = conn.execute("SELECT paper_id FROM documents WHERE collection=? AND norm_title=? LIMIT 1;", (col, norm)).fetchone()
                                        if row:
                                            doc["paper_id"] = row[0]
                                except Exception:
                                    pass
                                sdb.upsert_document(doc)
                        except Exception:
                            pass
                        # finalize job
                        try:
                            now_ts2 = int(time.time())
                            app.config["JOB_STATUS"][job_id].update({"done": True, "success": True, "progress": 100, "stage": "done", "updated_at": now_ts2, "paper_id": result.get('paper_id')})
                            app.config["JOBS_DIRTY"] = True
                            try:
                                get_state_db().upsert_job({"id": job_id, "paper_id": result.get('paper_id'), "filename": os.path.basename(path), "progress": 100, "stage": "done", "done": 1, "success": 1, "error": None, "updated_at": now_ts2})
                            except Exception:
                                pass
                        except Exception:
                            pass
                        processed += 1
                        # brief pause to avoid overwhelming services
                        time.sleep(0.2)
                    except Exception as e:
                        files_state[key] = {"error": str(e), "mtime": st.st_mtime if 'st' in locals() else None}
                        continue
            state['files'] = files_state
            state['last_scan'] = int(time.time())
            _write_watch_state(state)
            return {"scanned": scanned, "processed": processed, "last_scan": state['last_scan']}

    def _watch_loop():
        while True:
            try:
                settings = _read_settings()
                if settings.get('watch_enabled', False):
                    interval = int(settings.get('watch_interval_seconds', 300) or 300)
                    _scan_once()
                else:
                    interval = 10
                # Sleep in small steps so we can respond to scan-now
                for _ in range(interval):
                    if _scan_now_event.is_set():
                        _scan_now_event.clear()
                        break
                    time.sleep(1)
            except Exception:
                time.sleep(5)

    # Lazy init controls via env
    lazy_init = os.environ.get("CITEWEAVE_API_LAZY_INIT", "1") not in ("0", "false", "False")
    enable_graph_db = os.environ.get("CITEWEAVE_ENABLE_GRAPHDB", "0") in ("1", "true", "True")

    app.config["DOC_PROCESSOR"] = None
    app.config["RESEARCH_SYSTEM"] = None
    app.config["JOB_STATUS"] = {}
    app.config["JOBS_DIRTY"] = False
    jobs_ttl_hours = int(os.environ.get("CITEWEAVE_JOBS_TTL_HOURS", "72") or 72)
    from concurrent.futures import ThreadPoolExecutor
    max_workers = int(os.environ.get("CITEWEAVE_MAX_UPLOAD_WORKERS", "2"))
    app.config["EXECUTOR"] = ThreadPoolExecutor(max_workers=max_workers)
    app.config["STATE_DB"] = StateDB(str(db_path))
    # ensure default collection exists
    try:
        app.config["STATE_DB"].ensure_collection(default_collection, int(time.time()))
    except Exception:
        pass

    def get_doc_processor() -> DocumentProcessor:
        dp = app.config.get("DOC_PROCESSOR")
        if dp is None:
            # Avoid heavy graph DB init unless explicitly enabled
            dp = DocumentProcessor(enable_graph_db=enable_graph_db)
            app.config["DOC_PROCESSOR"] = dp
        return dp

    def get_research_system() -> LangGraphResearchSystem:
        rs = app.config.get("RESEARCH_SYSTEM")
        if rs is None:
            rs = LangGraphResearchSystem()
            app.config["RESEARCH_SYSTEM"] = rs
        return rs

    def get_state_db() -> StateDB:
        return app.config["STATE_DB"]

    # Restore jobs from disk and start autosave/cleanup
    def _init_jobs_state():
        try:
            saved = _read_jobs_file()
            if isinstance(saved, dict):
                # Mark in-flight jobs as interrupted after restart
                now_ts = int(time.time())
                for jid, st in saved.items():
                    if not st.get("done"):
                        st.update({"done": True, "success": False, "stage": "interrupted", "error": "Server restarted", "updated_at": now_ts})
                app.config["JOB_STATUS"] = saved
        except Exception:
            app.config["JOB_STATUS"] = {}

        def _autosave_loop():
            while True:
                try:
                    # cleanup
                    ttl = jobs_ttl_hours * 3600
                    now = time.time()
                    pruned = False
                    js = app.config.get("JOB_STATUS", {})
                    to_del = []
                    for jid, st in js.items():
                        ts = st.get("updated_at") or st.get("timestamp") or now
                        if st.get("done") and (now - float(ts)) > ttl:
                            to_del.append(jid)
                    for jid in to_del:
                        js.pop(jid, None)
                        pruned = True
                    if app.config.get("JOBS_DIRTY") or pruned:
                        _write_jobs_file(js)
                        app.config["JOBS_DIRTY"] = False
                except Exception:
                    pass
                time.sleep(2)

        threading.Thread(target=_autosave_loop, daemon=True).start()

    _init_jobs_state()

    # Eager init if not lazy
    if not lazy_init:
        try:
            get_doc_processor()
            get_research_system()
        except Exception as e:
            # Do not crash app creation; requests will surface errors
            print(f"[CiteWeave API] Warning: eager initialization failed: {e}")

    @app.errorhandler(Exception)
    def handle_exception(e):
        return (
            jsonify(
                {
                    "error": True,
                    "error_type": e.__class__.__name__,
                    "error_message": str(e),
                }
            ),
            500,
        )

    @app.get("/api/v1/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.get("/api/v1/settings")
    def get_settings() -> Any:
        s = _read_settings() or {}
        # Derive key presence from .env or settings
        env_vars_local = _read_env_file()
        env_key = env_vars_local.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
        present = bool(env_key or s.get('openai_key'))
        # Prepare safe response copy
        out = dict(s)
        # prefer .env value for api_base if present
        if env_vars_local.get('CITEWEAVE_PUBLIC_API_BASE'):
            out['api_base'] = env_vars_local['CITEWEAVE_PUBLIC_API_BASE']
        if 'openai_key' in out:
            del out['openai_key']
        out['openai_key_present'] = present
        if present:
            out['openai_key_preview'] = '********'
        return jsonify({"success": True, "settings": out, "path": str(settings_path)})

    @app.post("/api/v1/settings")
    def post_settings() -> Any:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": True, "error_type": "invalid_input", "error_message": "JSON body expected"}), 400
        s = _read_settings()
        # allow list (include watch_map for folder monitoring)
        for key in ("api_base", "theme", "display_name", "openai_key", "watch_enabled", "watch_interval_seconds", "watch_directories", "watch_map", "embedding_provider"):
            if key in payload:
                s[key] = payload[key]
        _write_settings(s)
        # apply sensitive settings to env
        if s.get('openai_key'):
            _set_env_var('OPENAI_API_KEY', s['openai_key'])
        if s.get('api_base'):
            _set_env_var('CITEWEAVE_PUBLIC_API_BASE', s['api_base'])
        return jsonify({"success": True})

    @app.get("/api/v1/config/embedding")
    def get_embedding_config() -> Any:
        cfg = _read_model_config()
        emb = (cfg.get("embedding") or {}) if isinstance(cfg, dict) else {}
        provider = emb.get("provider")
        model = emb.get("model")
        return jsonify({"success": True, "provider": provider, "model": model, "path": str(model_config_path)})

    @app.post("/api/v1/config/embedding")
    def set_embedding_config() -> Any:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": True, "error_type": "invalid_input", "error_message": "JSON body expected"}), 400
        provider = payload.get("provider")
        model = payload.get("model")
        if not provider or not model:
            return jsonify({"error": True, "error_message": "provider and model are required"}), 400
        cfg = _read_model_config()
        if not isinstance(cfg, dict):
            cfg = {}
        cfg.setdefault("embedding", {})
        cfg["embedding"]["provider"] = provider
        cfg["embedding"]["model"] = model
        _write_model_config(cfg)
        return jsonify({"success": True})

    @app.get("/api/v1/secret/openai-key")
    def get_openai_secret() -> Any:
        """Return the stored OpenAI API key (dev convenience)."""
        # Optional guard
        if os.environ.get("CITEWEAVE_ALLOW_SECRET_VIEW", "1") not in ("1", "true", "True"):
            return jsonify({"error": True, "error_message": "Secret view disabled"}), 403
        env_vars_local = _read_env_file()
        key = env_vars_local.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if not key:
            return jsonify({"error": True, "error_message": "Not set"}), 404
        return jsonify({"success": True, "key": key})

    @app.post("/api/v1/admin/init")
    def admin_init() -> Any:
        """Dangerous: Reset local state to a clean slate and keep only Default collection.
        Request body: { "confirm": true }
        This clears:
          - SQLite tables: documents, jobs, deletions, collections (except Default)
          - In-memory jobs cache and jobs file
          - Local filesystem under data/papers and watch_state.json
          - Optional vector index folders if present (best-effort)
        """
        payload = request.get_json(silent=True) or {}
        if not bool(payload.get("confirm")):
            return jsonify({"error": True, "error_message": "confirmation required"}), 400
        summary: Dict[str, Any] = {"fs": {}, "sqlite": {}, "jobs": {}}
        # 1) Clear SQLite rows and leave only Default collection
        try:
            sdb = get_state_db()
            with sdb._conn() as conn:
                conn.execute("DELETE FROM documents;")
                conn.execute("DELETE FROM jobs;")
                conn.execute("DELETE FROM deletions;")
                # Remove all collections then recreate Default
                conn.execute("DELETE FROM collections;")
            # ensure Default exists
            sdb.ensure_collection(default_collection, int(time.time()))
            summary["sqlite"]["ok"] = True
        except Exception as e:
            summary["sqlite"]["ok"] = False
            summary["sqlite"]["error"] = str(e)

        # 2) Clear in-memory jobs and persist empty jobs file
        try:
            app.config["JOB_STATUS"] = {}
            _write_jobs_file({})
            app.config["JOBS_DIRTY"] = False
            summary["jobs"]["ok"] = True
        except Exception as e:
            summary["jobs"]["ok"] = False
            summary["jobs"]["error"] = str(e)

        # 3) Clear filesystem: data/papers, watch_state.json, optional vector indexes
        try:
            import shutil
            papers_root = data_dir / "papers"
            if papers_root.exists():
                shutil.rmtree(papers_root)
            papers_root.mkdir(parents=True, exist_ok=True)
            # optional vector indices
            for folder in (data_dir / "vector_index", data_dir / "vector_index_backup"):
                try:
                    if folder.exists():
                        shutil.rmtree(folder)
                except Exception:
                    pass
            # watch state
            try:
                if watch_state_path.exists():
                    watch_state_path.unlink()
            except Exception:
                pass
            summary["fs"]["ok"] = True
        except Exception as e:
            summary["fs"]["ok"] = False
            summary["fs"]["error"] = str(e)

        return jsonify({"success": True, "summary": summary})

    # Backward/alias endpoints for convenience and to avoid 404s in older clients
    @app.post("/api/v1/reset")
    def admin_reset_alias_post() -> Any:
        return admin_init()

    @app.get("/api/v1/admin/init")
    def admin_init_get() -> Any:
        # Allow GET for quick manual testing from browser
        return admin_init()

    # Extra aliases to avoid path prefix mismatches in different deployments
    @app.post("/admin/init")
    def admin_init_noapi_post() -> Any:
        return admin_init()

    @app.get("/admin/init")
    def admin_init_noapi_get() -> Any:
        return admin_init()

    @app.post("/api/reset")
    def admin_reset_short_post() -> Any:
        return admin_init()

    @app.get("/")
    def index_root() -> Any:
        # Serve frontend index if available
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return send_from_directory(static_dir, "index.html")
        return jsonify({"status": "ok", "message": "API is running"})

    @app.get("/static/logo.png")
    def serve_logo_png():
        """Serve the app logo from static dir if present; otherwise fall back to project root or data dir.
        This ensures the favicon and AI avatar work even if the file was dropped at project root as logo.png.
        """
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        candidate_paths = [
            os.path.join(static_dir, "logo.png"),
            os.path.join(os.getcwd(), "logo.png"),
            os.path.join(str(data_dir), "logo.png"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                return send_from_directory(os.path.dirname(path), os.path.basename(path))
        # Transparent 1x1 PNG fallback to avoid broken images
        import base64
        from flask import Response
        transparent_png = base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAoMBgZ1T4nsAAAAASUVORK5CYII="
        )
        return Response(transparent_png, mimetype="image/png")

    @app.get("/static/<path:filename>")
    def serve_static(filename: str):
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        return send_from_directory(static_dir, filename)

    @app.post("/api/v1/watch/scan-now")
    def watch_scan_now():
        _scan_now_event.set()
        # trigger immediate scan in background but return quickly
        threading.Thread(target=_scan_once, daemon=True).start()
        return jsonify({"success": True})

    # Start watcher background thread once
    def _ensure_watch_thread():
        nonlocal _scan_thread_started
        if not _scan_thread_started:
            t = threading.Thread(target=_watch_loop, daemon=True)
            t.start()
            _scan_thread_started = True
    _ensure_watch_thread()

    def _sanitize_filename(title: str, authors: list) -> str:
        """Create a safe filename like 'Title - Author.pdf'. Fallbacks included."""
        def clean(text: str) -> str:
            # Remove forbidden characters and trim
            bad = '\\/:*?"<>|\n\r\t'
            for ch in bad:
                text = text.replace(ch, ' ')
            text = ' '.join(text.split())
            return text[:120]
        safe_title = clean(title or "")
        primary_author = clean((authors or ["Unknown"])[0])
        if not safe_title:
            safe_title = "Untitled"
        if not primary_author:
            primary_author = "Unknown"
        return f"{safe_title} - {primary_author}.pdf"

    def _norm_title(title: Optional[str]) -> Optional[str]:
        if not title:
            return None
        t = title.lower().strip()
        for ch in "\t\n\r":
            t = t.replace(ch, ' ')
        while '  ' in t:
            t = t.replace('  ', ' ')
        return t

    def _paper_dir(paper_id: str) -> Path:
        return Path(os.path.join("data", "papers", paper_id))

    def _status_path(paper_id: str) -> Path:
        return _paper_dir(paper_id) / "status.json"

    def _read_status(paper_id: str) -> dict:
        p = _status_path(paper_id)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                return {}
        return {}

    def _write_status(paper_id: str, status: dict):
        try:
            _status_path(paper_id).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

    # Simple cascade deletion helper (FS + SQLite)
    def _cascade_delete_paper(paper_id: str) -> bool:
        ok = True
        try:
            import shutil
            target = _paper_dir(paper_id)
            if target.exists():
                shutil.rmtree(target)
        except Exception:
            ok = False
        try:
            sdb = get_state_db()
            sdb.delete_jobs_by_paper(paper_id)
            sdb.delete_document(paper_id)
        except Exception:
            ok = False
        return ok

    @app.post("/api/v1/upload")
    def upload() -> Any:
        """Upload and process a PDF. Expects multipart/form-data with field 'file'."""
        if "file" not in request.files:
            return jsonify({"error": True, "error_type": "invalid_input", "error_message": "Missing 'file'"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": True, "error_type": "invalid_input", "error_message": "Empty filename"}), 400

        # Persist to a temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            processor: DocumentProcessor = get_doc_processor()
            result: Dict = processor.process_document(tmp_path, save_results=True)
            # Summarize response to essentials and include full result
            stats = result.get("processing_stats", {})
            summary = {
                "paper_id": result.get("paper_id"),
                "total_sentences": stats.get("total_sentences", 0),
                "sentences_with_citations": stats.get("sentences_with_citations", 0),
                "total_citations": stats.get("total_citations", 0),
                "total_references": stats.get("total_references", 0),
            }
            # Try to rename original.pdf to a canonical name derived from metadata
            try:
                pid = result.get("paper_id")
                meta = result.get("metadata", {}) or {}
                if pid:
                    pdir = _paper_dir(pid)
                    orig = pdir / "original.pdf"
                    if orig.exists():
                        target = pdir / _sanitize_filename(meta.get("title", "Untitled"), meta.get("authors", []))
                        if target.name != orig.name:
                            # Avoid overwriting if same name exists; append id suffix
                            if target.exists():
                                target = pdir / (target.stem + f"_{pid}" + target.suffix)
                            orig.rename(target)
                        # persist to DB (default collection)
                        try:
                            now_ts = int(time.time())
                            doc = {
                                "paper_id": pid,
                                "title": meta.get("title"),
                                "authors_json": json.dumps(meta.get("authors", []), ensure_ascii=False),
                                "year": meta.get("year"),
                                "doi": meta.get("doi"),
                                "original_filename": file.filename,
                                "new_filename": target.name,
                                "local_path": str(target),
                                "collection": request.form.get("collection") or request.args.get("collection") or default_collection,
                                "status": "done",
                                "created_at": now_ts,
                                "updated_at": now_ts,
                            }
                            sdb = get_state_db()
                            sdb.ensure_collection(doc["collection"], now_ts)
                            sdb.upsert_document(doc)
                        except Exception:
                            pass
                # initialize status if not present
                if pid and not _status_path(pid).exists():
                    _write_status(pid, {"done": False, "updated_at": int(time.time())})
            except Exception:
                pass
            return jsonify({"success": True, "summary": summary, "result": result})
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @app.post("/api/v1/upload_async")
    def upload_async() -> Any:
        """Queue a PDF for processing. Returns a job_id immediately."""
        if "file" not in request.files:
            return jsonify({"error": True, "error_type": "invalid_input", "error_message": "Missing 'file'"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": True, "error_type": "invalid_input", "error_message": "Empty filename"}), 400
        collection = request.form.get("collection") or request.args.get("collection") or default_collection
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        job_id = str(int(time.time() * 1000)) + os.path.basename(tmp_path)
        now_ts = int(time.time())
        app.config["JOB_STATUS"][job_id] = {"done": False, "success": None, "progress": 5, "stage": "queued", "filename": file.filename, "collection": collection, "updated_at": now_ts}
        app.config["JOBS_DIRTY"] = True
        # persist queued job to DB
        try:
            get_state_db().upsert_job({
                "id": job_id,
                "paper_id": None,
                "filename": file.filename,
                "progress": 5,
                "stage": "queued",
                "done": 0,
                "success": None,
                "error": None,
                "updated_at": now_ts,
            })
        except Exception:
            pass

        def _job():
            try:
                processor: DocumentProcessor = get_doc_processor()
                # naive progress ticker up to 90 while processing
                def _tick():
                    try:
                        while not app.config["JOB_STATUS"][job_id]["done"]:
                            st = app.config["JOB_STATUS"].get(job_id, {})
                            p = int(st.get("progress", 5))
                            if p < 90:
                                app.config["JOB_STATUS"][job_id]["progress"] = p + 1
                            time.sleep(1)
                    except Exception:
                        pass
                threading.Thread(target=_tick, daemon=True).start()
                app.config["JOB_STATUS"][job_id]["stage"] = "processing"
                app.config["JOB_STATUS"][job_id]["updated_at"] = int(time.time())
                app.config["JOBS_DIRTY"] = True
                try:
                    get_state_db().upsert_job({
                        "id": job_id,
                        "paper_id": None,
                        "filename": file.filename,
                        "progress": app.config["JOB_STATUS"][job_id]["progress"],
                        "stage": "processing",
                        "done": 0,
                        "success": None,
                        "error": None,
                        "updated_at": int(time.time()),
                    })
                except Exception:
                    pass
                # deduplicate: if the same paper already exists in this collection by hash or norm title, reuse
                # compute basic hash to identify file sameness
                import hashlib
                file_hash = None
                try:
                    with open(tmp_path, 'rb') as fh:
                        file_hash = hashlib.sha256(fh.read()).hexdigest()
                except Exception:
                    pass
                result: Dict = processor.process_document(tmp_path, save_results=True)
                stats = result.get("processing_stats", {})
                summary = {
                    "paper_id": result.get("paper_id"),
                    "total_sentences": stats.get("total_sentences", 0),
                }
                # rename and init status as in sync path
                try:
                    pid = result.get("paper_id")
                    meta = result.get("metadata", {}) or {}
                    if pid:
                        pdir = _paper_dir(pid)
                        orig = pdir / "original.pdf"
                        if orig.exists():
                            target = pdir / _sanitize_filename(meta.get("title", "Untitled"), meta.get("authors", []))
                            if target.name != orig.name:
                                if target.exists():
                                    target = pdir / (target.stem + f"_{pid}" + target.suffix)
                                orig.rename(target)
                        if pid and not _status_path(pid).exists():
                            _write_status(pid, {"done": True, "updated_at": int(time.time()), "collection": collection})
                        # persist document to DB (upsert; avoid duplicates by collection+hash)
                        try:
                            now_ts2 = int(time.time())
                            sdb = get_state_db()
                            sdb.ensure_collection(collection, now_ts2)
                            norm = _norm_title(meta.get("title"))
                            # Build document record consistently (spaces only; no tabs)
                            doc = {
                                "paper_id": pid,
                                "title": meta.get("title"),
                                "authors_json": json.dumps(meta.get("authors", []), ensure_ascii=False),
                                "year": meta.get("year"),
                                "doi": meta.get("doi"),
                                "original_filename": file.filename,
                                "new_filename": (target.name if 'target' in locals() else "original.pdf"),
                                "local_path": str((target if 'target' in locals() and target.exists() else orig)),
                                "collection": collection,
                                "status": "done",
                                "file_hash": file_hash,
                                "norm_title": norm,
                                "created_at": now_ts2,
                                "updated_at": now_ts2,
                            }
                            # If a row exists for this collection with same file hash, reuse paper_id and only update fields
                            try:
                                with sdb._conn() as conn:
                                    row = None
                                    if file_hash:
                                        row = conn.execute("SELECT paper_id FROM documents WHERE collection=? AND file_hash=? LIMIT 1;", (collection, file_hash)).fetchone()
                                    if not row and norm:
                                        row = conn.execute("SELECT paper_id FROM documents WHERE collection=? AND norm_title=? LIMIT 1;", (collection, norm)).fetchone()
                                    if row:
                                        # reuse existing paper_id, overwrite doc target paths and trigger reindex already done
                                        doc["paper_id"] = row[0]
                            except Exception:
                                pass
                            sdb.upsert_document(doc)
                            # finalize job record
                            sdb.upsert_job({
                                "id": job_id,
                                "paper_id": pid,
                                "filename": file.filename,
                                "progress": 100,
                                "stage": "done",
                                "done": 1,
                                "success": 1,
                                "error": None,
                                "updated_at": now_ts2,
                            })
                        except Exception:
                            pass
                except Exception:
                    pass
                app.config["JOB_STATUS"][job_id].update({"done": True, "success": True, "summary": summary, "result": result, "progress": 100, "stage": "done", "updated_at": int(time.time())})
                app.config["JOBS_DIRTY"] = True
            except Exception as e:
                app.config["JOB_STATUS"][job_id].update({"done": True, "success": False, "error": str(e), "progress": 100, "stage": "error", "updated_at": int(time.time())})
                app.config["JOBS_DIRTY"] = True
                try:
                    get_state_db().upsert_job({
                        "id": job_id,
                        "paper_id": None,
                        "filename": file.filename,
                        "progress": 100,
                        "stage": "error",
                        "done": 1,
                        "success": 0,
                        "error": str(e),
                        "updated_at": int(time.time()),
                    })
                except Exception:
                    pass
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        app.config["EXECUTOR"].submit(_job)
        return jsonify({"success": True, "job_id": job_id})

    @app.get("/api/v1/jobs/<job_id>")
    def job_status(job_id: str) -> Any:
        st = app.config["JOB_STATUS"].get(job_id)
        if not st:
            return jsonify({"error": True, "error_message": "job not found"}), 404
        return jsonify({"success": True, "job": st})

    @app.get("/api/v1/jobs")
    def list_jobs() -> Any:
        return jsonify({"success": True, "jobs": app.config.get("JOB_STATUS", {})})

    @app.get("/api/v1/documents")
    def documents_list() -> Any:
        """List documents from SQLite with optional collection filter."""
        try:
            default_collection = os.environ.get("CITEWEAVE_DEFAULT_COLLECTION", "Default")
            collection_q = request.args.get("collection")
            # default to Default collection if not provided
            collection = collection_q or default_collection
            docs = get_state_db().list_documents(collection=collection)
            # Ensure any currently in-progress jobs with known filename but no paper_id appear as pending items
            try:
                js = app.config.get("JOB_STATUS", {})
                for jid, st in js.items():
                    if st and not st.get('done') and st.get('filename') and (st.get('collection') == collection):
                        docs.insert(0, {
                            "paper_id": st.get('paper_id'),
                            "title": st.get('filename'),
                            "authors": [],
                            "year": None,
                            "doi": None,
                            "original_filename": st.get('filename'),
                            "new_filename": st.get('filename'),
                            "local_path": None,
                            "collection": collection,
                            "status": st.get('stage') or 'queued',
                            "created_at": st.get('updated_at'),
                            "updated_at": st.get('updated_at'),
                        })
            except Exception:
                pass
            return jsonify({"success": True, "documents": docs, "collection": collection})
        except Exception as e:
            return jsonify({"error": True, "error_message": str(e)}), 500

    @app.get("/api/v1/collections")
    def collections_list() -> Any:
        try:
            # always ensure default exists at read time
            try:
                get_state_db().ensure_collection(os.environ.get("CITEWEAVE_DEFAULT_COLLECTION", "Default"), int(time.time()))
            except Exception:
                pass
            cols = get_state_db().list_collections()
            # guarantee Default in response even if DB empty
            names = {c.get('name') for c in cols if isinstance(c, dict)}
            if 'Default' not in names:
                cols = [{"name": "Default", "created_at": None, "updated_at": None, "document_count": 0}] + cols
            return jsonify({"success": True, "collections": cols})
        except Exception as e:
            return jsonify({"error": True, "error_message": str(e)}), 500

    @app.post("/api/v1/collections")
    def collections_create() -> Any:
        payload = request.get_json(silent=True) or {}
        name = (payload.get("name") or "").strip()
        if not name:
            return jsonify({"error": True, "error_message": "name required"}), 400
        try:
            ts = int(time.time())
            get_state_db().ensure_collection(name, ts)
            return jsonify({"success": True, "name": name})
        except Exception as e:
            return jsonify({"error": True, "error_message": str(e)}), 500

    @app.post("/api/v1/collections/rename")
    def collections_rename() -> Any:
        payload = request.get_json(silent=True) or {}
        old_name = payload.get("old_name")
        new_name = payload.get("new_name")
        if not old_name or not new_name:
            return jsonify({"error": True, "error_message": "old_name and new_name required"}), 400
        try:
            ts = int(time.time())
            get_state_db().rename_collection(old_name, new_name, ts)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": True, "error_message": str(e)}), 500

    @app.post("/api/v1/collections/delete")
    def collections_delete() -> Any:
        payload = request.get_json(silent=True) or {}
        name = (payload.get("name") or "").strip()
        confirm = bool(payload.get("confirm"))
        if not name:
            return jsonify({"error": True, "error_message": "name required"}), 400
        if not confirm:
            return jsonify({"error": True, "error_message": "confirmation required"}), 400
        try:
            sdb = get_state_db()
            docs = sdb.list_documents(collection=name)
            count = 0
            for d in docs:
                pid = d.get("paper_id")
                if not pid:
                    continue
                deletion_id = f"del_{int(time.time()*1000)}_{pid}"
                try:
                    sdb.create_deletion({
                        "id": deletion_id,
                        "paper_id": pid,
                        "requested_at": int(time.time()),
                        "finished_at": None,
                        "ok": None,
                        "error": None,
                    })
                except Exception:
                    pass
                def _do(pid_inner=pid, del_id=deletion_id):
                    ok = _cascade_delete_paper(pid_inner)
                    try:
                        sdb.finish_deletion(del_id, ok=ok, finished_at=int(time.time()), error=None if ok else "partial failure")
                    except Exception:
                        pass
                app.config["EXECUTOR"].submit(_do)
                count += 1
            # Remove collection row after scheduling
            try:
                with sdb._conn() as conn:
                    conn.execute("DELETE FROM collections WHERE name=?;", (name,))
            except Exception:
                pass
            return jsonify({"success": True, "scheduled": count})
        except Exception as e:
            return jsonify({"error": True, "error_message": str(e)}), 500

    @app.post("/api/v1/diagnose")
    def diagnose() -> Any:
        """Diagnose a PDF. Expects multipart/form-data with field 'file'."""
        if "file" not in request.files:
            return jsonify({"error": True, "error_type": "invalid_input", "error_message": "Missing 'file'"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": True, "error_type": "invalid_input", "error_message": "Empty filename"}), 400

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            processor: DocumentProcessor = get_doc_processor()
            diagnosis: Dict = processor.diagnose_document_processing(tmp_path)
            return jsonify({"success": True, "diagnosis": diagnosis})
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @app.post("/api/v1/chat")
    def chat() -> Any:
        """
        Stateless chat turn with the research system.
        Body (application/json):
        {
          "user_input": string,
          "history": list[dict],            # optional; format [{"user": str, "ai": str}, ...]
          "menu_choice": string|null,       # optional ("1"|"2"|"3"|"4")
          "collected_data": object|null     # optional; pass-through between turns
        }
        Returns the system response with potential next-step flags.
        """
        data = request.get_json(silent=True) or {}
        user_input: Optional[str] = data.get("user_input")
        if not user_input or not isinstance(user_input, str):
            return jsonify({"error": True, "error_type": "invalid_input", "error_message": "Field 'user_input' is required"}), 400

        history = data.get("history") or []
        menu_choice = data.get("menu_choice")
        collected_data = data.get("collected_data")

        system: LangGraphResearchSystem = get_research_system()
        response: Dict = system.interactive_research_chat(
            user_input=user_input,
            history=history,
            menu_choice=menu_choice,
            collected_data=collected_data,
        )
        return jsonify({"success": True, "response": response})

    @app.get("/api/v1/docs")
    def list_docs() -> Any:
        """List processed documents by scanning data/papers directory."""
        papers_root = os.path.join("data", "papers")
        results = []
        try:
            if not os.path.isdir(papers_root):
                return jsonify({"success": True, "documents": results})
            for pid in os.listdir(papers_root):
                meta_path = os.path.join(papers_root, pid, "metadata.json")
                if os.path.isfile(meta_path):
                    try:
                        import json
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        st = _read_status(pid)
                        results.append({
                            "paper_id": pid,
                            "title": meta.get("title", "Unknown Title"),
                            "authors": meta.get("authors", []),
                            "year": meta.get("year", "Unknown Year"),
                            "doi": meta.get("doi", "Unknown DOI"),
                            "done": bool(st.get("done", False)),
                            "collection": st.get("collection", os.environ.get("CITEWEAVE_DEFAULT_COLLECTION", "Default"))
                        })
                    except Exception:
                        continue
        except Exception as e:
            return jsonify({"error": True, "error_type": "list_error", "error_message": str(e)}), 500
        return jsonify({"success": True, "documents": results})

    @app.post("/api/v1/watch/pause")
    def watch_pause() -> Any:
        payload = request.get_json(silent=True) or {}
        path = (payload.get("path") or "").strip()
        paused = bool(payload.get("paused"))
        if not path:
            return jsonify({"error": True, "error_message": "path required"}), 400
        s = _read_settings()
        arr = s.get('watch_map') or []
        changed = False
        out = []
        for e in arr:
            if isinstance(e, dict) and e.get('path') == path:
                e['paused'] = paused
                changed = True
            out.append(e)
        if not changed:
            return jsonify({"error": True, "error_message": "path not found"}), 404
        s['watch_map'] = out
        _write_settings(s)
        return jsonify({"success": True})

    @app.post("/api/v1/watch/delete")
    def watch_delete() -> Any:
        payload = request.get_json(silent=True) or {}
        path = (payload.get("path") or "").strip()
        if not path:
            return jsonify({"error": True, "error_message": "path required"}), 400
        s = _read_settings()
        arr = s.get('watch_map') or []
        next_map = [e for e in arr if not (isinstance(e, dict) and e.get('path') == path)]
        s['watch_map'] = next_map
        _write_settings(s)
        return jsonify({"success": True})

    @app.post("/api/v1/docs/mark")
    def mark_doc() -> Any:
        data = request.get_json(silent=True) or {}
        pid = data.get("paper_id")
        done = bool(data.get("done", False))
        if not pid:
            return jsonify({"error": True, "error_message": "paper_id required"}), 400
        st = _read_status(pid)
        st["done"] = done
        st["updated_at"] = int(time.time())
        _write_status(pid, st)
        return jsonify({"success": True})

    @app.post("/api/v1/docs/reprocess")
    def reprocess_doc() -> Any:
        """Reprocess an existing paper asynchronously and return a job_id for UI polling."""
        data = request.get_json(silent=True) or {}
        pid = data.get("paper_id")
        if not pid:
            return jsonify({"error": True, "error_message": "paper_id required"}), 400
        pdir = _paper_dir(pid)
        # Choose the most likely PDF under the paper dir
        pdf_path = None
        try:
            candidates = [p for p in pdir.glob("*.pdf")]
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
                pdf_path = str(candidates[0])
        except Exception:
            pass
        if not pdf_path:
            return jsonify({"error": True, "error_message": "No PDF found for paper"}), 404

        job_id = f"re_{int(time.time()*1000)}_{pid}"
        now_ts = int(time.time())
        app.config["JOB_STATUS"][job_id] = {
            "done": False,
            "success": None,
            "progress": 5,
            "stage": "queued",
            "paper_id": pid,
            "filename": os.path.basename(pdf_path),
            "updated_at": now_ts,
        }
        app.config["JOBS_DIRTY"] = True
        try:
            get_state_db().upsert_job({
                "id": job_id,
                "paper_id": pid,
                "filename": os.path.basename(pdf_path),
                "progress": 5,
                "stage": "queued",
                "done": 0,
                "success": None,
                "error": None,
                "updated_at": now_ts,
            })
        except Exception:
            pass

        def _job():
            try:
                processor: DocumentProcessor = get_doc_processor()
                # tick progress while working
                def _tick():
                    try:
                        while not app.config["JOB_STATUS"][job_id]["done"]:
                            st = app.config["JOB_STATUS"].get(job_id, {})
                            p = int(st.get("progress", 5))
                            if p < 90:
                                app.config["JOB_STATUS"][job_id]["progress"] = p + 1
                            time.sleep(1)
                    except Exception:
                        pass
                threading.Thread(target=_tick, daemon=True).start()
                app.config["JOB_STATUS"][job_id]["stage"] = "processing"
                app.config["JOB_STATUS"][job_id]["updated_at"] = int(time.time())
                app.config["JOBS_DIRTY"] = True
                try:
                    get_state_db().upsert_job({
                        "id": job_id,
                        "paper_id": pid,
                        "filename": os.path.basename(pdf_path),
                        "progress": app.config["JOB_STATUS"][job_id]["progress"],
                        "stage": "processing",
                        "done": 0,
                        "success": None,
                        "error": None,
                        "updated_at": int(time.time()),
                    })
                except Exception:
                    pass

                # process
                result: Dict = processor.process_document(pdf_path, save_results=True)
                # Update DB document status and timestamps
                try:
                    get_state_db().set_document_status(pid, "done", int(time.time()))
                except Exception:
                    pass
                # finalize job
                app.config["JOB_STATUS"][job_id].update({
                    "done": True,
                    "success": True,
                    "progress": 100,
                    "stage": "done",
                    "updated_at": int(time.time()),
                    "result": result,
                    "paper_id": pid,
                })
                app.config["JOBS_DIRTY"] = True
                try:
                    get_state_db().upsert_job({
                        "id": job_id,
                        "paper_id": pid,
                        "filename": os.path.basename(pdf_path),
                        "progress": 100,
                        "stage": "done",
                        "done": 1,
                        "success": 1,
                        "error": None,
                        "updated_at": int(time.time()),
                    })
                except Exception:
                    pass
            except Exception as e:
                app.config["JOB_STATUS"][job_id].update({
                    "done": True,
                    "success": False,
                    "progress": 100,
                    "stage": "error",
                    "error": str(e),
                    "updated_at": int(time.time()),
                    "paper_id": pid,
                })
                app.config["JOBS_DIRTY"] = True
                try:
                    get_state_db().upsert_job({
                        "id": job_id,
                        "paper_id": pid,
                        "filename": os.path.basename(pdf_path),
                        "progress": 100,
                        "stage": "error",
                        "done": 1,
                        "success": 0,
                        "error": str(e),
                        "updated_at": int(time.time()),
                    })
                except Exception:
                    pass

        app.config["EXECUTOR"].submit(_job)
        return jsonify({"success": True, "job_id": job_id})

    @app.delete("/api/v1/docs/<paper_id>")
    def delete_doc(paper_id: str):
        try:
            import shutil
            target = _paper_dir(paper_id)
            if target.exists():
                shutil.rmtree(target)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": True, "error_message": str(e)}), 500

    @app.post("/api/v1/documents/<paper_id>/delete")
    def delete_document_cascade(paper_id: str):
        """Schedule a cascading deletion job: DB rows and local filesystem. External stores can be integrated later."""
        deletion_id = f"del_{int(time.time()*1000)}_{paper_id}"

        def _do_delete():
            ok = True
            err = None
            try:
                # delete local FS
                try:
                    import shutil
                    target = _paper_dir(paper_id)
                    if target.exists():
                        shutil.rmtree(target)
                except Exception as e:
                    ok = False
                    err = str(e)
                # delete from DB
                try:
                    sdb = get_state_db()
                    sdb.delete_jobs_by_paper(paper_id)
                    sdb.delete_document(paper_id)
                except Exception as e:
                    ok = False
                    err = (err or "") + f"; db: {e}"
            finally:
                try:
                    get_state_db().finish_deletion(deletion_id, ok=ok, finished_at=int(time.time()), error=err)
                except Exception:
                    pass

        try:
            get_state_db().create_deletion({
                "id": deletion_id,
                "paper_id": paper_id,
                "requested_at": int(time.time()),
                "finished_at": None,
                "ok": None,
                "error": None,
            })
        except Exception:
            pass
        app.config["EXECUTOR"].submit(_do_delete)
        return jsonify({"success": True, "deletion_id": deletion_id})

    @app.post("/api/v1/mounts/overlay")
    def generate_mounts_overlay() -> Any:
        """Generate a docker-compose overlay file content for bind-mounting watch_map paths.
        Body: { update_watch_map: bool }
        Returns overlay YAML content and planned host->container mappings.
        """
        try:
            payload = request.get_json(silent=True) or {}
            update = bool(payload.get('update_watch_map'))
            s = _read_settings() or {}
            wm = s.get('watch_map') or []
            if not isinstance(wm, list):
                wm = []
            # Build host->container mapping with safe slugs
            def slugify(p: str) -> str:
                import re
                return re.sub(r"[^A-Za-z0-9._-]+", "_", p.strip().strip('/')).strip('_') or "path"
            plans = []
            for e in wm:
                if not isinstance(e, dict) or not e.get('path'):
                    continue
                # Prefer original host_path if present (when watch_map was previously rewritten to container paths)
                host_path = e.get('host_path') or e['path']
                container_path = f"/data/host/{slugify(host_path)}"
                plans.append({
                    'host': host_path,
                    'container': container_path,
                    'collection': e.get('collection') or default_collection,
                })
            # Compose YAML text (include app_data and all binds, :ro)
            lines = [
                "services:",
                "  citeweave:",
                "    volumes:",
                "      - app_data:/app/data",
            ]
            for p in plans:
                lines.append(f"      - {p['host']}:{p['container']}:ro")
            lines.append("")
            overlay = "\n".join(lines)
            if update:
                # Rewrite watch_map to use container paths for runtime scanning, preserving real host_path
                next_map = []
                for e in wm:
                    if not isinstance(e, dict) or not e.get('path') and not e.get('host_path'):
                        continue
                    host = e.get('host_path') or e.get('path')
                    ne = dict(e)
                    ne['host_path'] = host
                    ne['path'] = f"/data/host/{slugify(host)}"
                    next_map.append(ne)
                s['watch_map'] = next_map
                _write_settings(s)
            return jsonify({"success": True, "overlay": overlay, "mounts": plans, "updated": update})
        except Exception as e:
            return jsonify({"error": True, "error_message": str(e)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("CITEWEAVE_API_HOST", "0.0.0.0")
    # Default to 31415 (pi-inspired, academic-themed) to reduce common port conflicts
    port = int(os.environ.get("CITEWEAVE_API_PORT", "31415"))
    debug = os.environ.get("CITEWEAVE_ENV", "production").lower() in ("dev", "development", "test")
    print(f"[CiteWeave API] Starting server on http://{host}:{port} (debug={debug})")
    print("[CiteWeave API] Endpoints: /api/v1/health, /api/v1/upload, /api/v1/diagnose, /api/v1/chat, /api/v1/docs")
    print("[CiteWeave API] Web UI: / (serves static) ")
    app.run(host=host, port=port, debug=debug)


