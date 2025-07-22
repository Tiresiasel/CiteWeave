#!/usr/bin/env python3
"""
Setup script for initializing Neo4j and Qdrant services using project config files.
"""
import os
import json
from pathlib import Path
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')

# --- Utility functions ---
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_neo4j(config):
    print("\n🔗 Setting up Neo4j...")
    uri = config["uri"]
    user = config["username"]
    password = config["password"]
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        # Example constraints (customize as needed)
        print("  - Ensuring unique constraint on Paper.id ...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
        print("  - Ensuring unique constraint on Author.name ...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
        print("  - Ensuring unique constraint on Claim.id ...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE")
        # Add more as needed
    driver.close()
    print("✅ Neo4j setup complete.")

def setup_qdrant(config):
    print("\n🔗 Setting up Qdrant...")
    host = config.get("host", "localhost")
    port = config.get("port", 6333)
    client = QdrantClient(host=host, port=port)
    collections = config.get("collections", {})
    for name, spec in collections.items():
        print(f"  - Ensuring collection '{name}' exists ...")
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=spec["vector_size"], distance=Distance[spec["distance"].upper()])
            )
            print(f"    Created collection '{name}' (size={spec['vector_size']}, distance={spec['distance']})")
        else:
            print(f"    Collection '{name}' already exists.")
    print("✅ Qdrant setup complete.")

def main():
    print("\n🚀 Setting up all services (Neo4j + Qdrant)...")
    neo4j_config_path = os.path.join(CONFIG_DIR, "neo4j_config.json")
    qdrant_config_path = os.path.join(CONFIG_DIR, "qdrant_config.json")
    if not os.path.exists(neo4j_config_path):
        print(f"❌ Neo4j config not found: {neo4j_config_path}")
        return
    if not os.path.exists(qdrant_config_path):
        print(f"❌ Qdrant config not found: {qdrant_config_path}")
        return
    neo4j_config = load_json(neo4j_config_path)
    qdrant_config = load_json(qdrant_config_path)
    setup_neo4j(neo4j_config)
    setup_qdrant(qdrant_config)
    print("\n🎉 All services are initialized and ready!")

if __name__ == "__main__":
    main() 