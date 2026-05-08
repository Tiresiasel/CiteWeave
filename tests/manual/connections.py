#!/usr/bin/env python3
"""Manual connection smoke checks for local Neo4j and Qdrant services.

This script never guesses passwords and never writes config files. Set
CITEWEAVE_NEO4J_PASSWORD in `.env` or the environment before running it.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.storage.graph_builder import GraphDB
from src.storage.vector_indexer import VectorIndexer


def test_neo4j_connection() -> bool:
    print("🔗 Testing Neo4j connection...")
    load_dotenv(Path(".env"))

    password = os.environ.get("CITEWEAVE_NEO4J_PASSWORD")
    if not password:
        print("  ❌ CITEWEAVE_NEO4J_PASSWORD is not set")
        return False

    db = None
    try:
        db = GraphDB(
            uri=os.environ.get("CITEWEAVE_NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("CITEWEAVE_NEO4J_USERNAME", "neo4j"),
            password=password,
        )
        with db.driver.session() as session:
            record = session.run("RETURN 1 as test").single()
        ok = bool(record and record["test"] == 1)
        print("  ✅ Connected successfully" if ok else "  ❌ Unexpected Neo4j response")
        return ok
    except Exception as exc:
        print(f"  ❌ Neo4j connection failed: {exc}")
        return False
    finally:
        if db is not None:
            db.close()


def test_vector_database() -> bool:
    print("\n🔗 Testing vector database connection...")
    try:
        VectorIndexer(config_path="config/qdrant_config.json")
        print("  ✅ Vector indexer initialized successfully")
        return True
    except Exception as exc:
        print(f"  ❌ Vector indexer failed: {exc}")
        return False


def main() -> None:
    print("=" * 50)
    print("🧪 Manual Connection Smoke Check")
    print("=" * 50)

    neo4j_success = test_neo4j_connection()
    vector_success = test_vector_database()

    print("\n📊 Summary:")
    print(f"  Neo4j: {'✅ Connected' if neo4j_success else '❌ Failed'}")
    print(f"  Vector DB: {'✅ Connected' if vector_success else '❌ Failed'}")


if __name__ == "__main__":
    main()
