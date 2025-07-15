"""
vector_indexer.py
Module for generating embeddings and storing them in an index.
"""

# vector_indexer.py
import os
import uuid
import json
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


class VectorIndexer:
    def __init__(self, paper_root: str = "./data/papers/", index_path: str = "./data/vector_index",
                 collection_name: str = "claims"):
        self.paper_root = paper_root
        self.collection_name = collection_name
        self.client = QdrantClient(path=index_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    def index_sentences(self, paper_id: str, sentences: List[str], metadata: dict,
                        claim_types: Optional[List[str]] = None):
        vectors = self.model.encode(sentences, normalize_embeddings=True)
        points = []
        for i, (vector, text) in enumerate(zip(vectors, sentences)):
            payload = {
                "paper_id": paper_id,
                "sentence_index": i,
                "text": text,
                "claim_type": claim_types[i] if claim_types else "unspecified",
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", []),
                "year": metadata.get("year", "Unknown"),
                "doi": metadata.get("doi", "Unknown"),
                "journal": metadata.get("journal", "Unknown"),
                "publisher": metadata.get("publisher", "Unknown"),
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ Indexed {len(points)} sentences for paper {paper_id}")

    def index_all_from_data(self):
        for paper_id in os.listdir(self.paper_root):
            paper_dir = os.path.join(self.paper_root, paper_id)
            if not os.path.isdir(paper_dir):
                continue

            sentences_path = os.path.join(paper_dir, "sentences.jsonl")
            metadata_path = os.path.join(paper_dir, "metadata.json")

            if not os.path.exists(sentences_path) or not os.path.exists(metadata_path):
                print(f"[WARN] Missing files for {paper_id}, skipping...")
                continue

            with open(sentences_path, "r") as f:
                sentence_objs = [json.loads(line.strip()) for line in f if line.strip()]
            sentences = [s["text"] for s in sentence_objs]
            claim_types = [s.get("claim_type", "unspecified") for s in sentence_objs]

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.index_sentences(paper_id=paper_id, sentences=sentences, metadata=metadata, claim_types=claim_types)

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]
        results = self.client.search(collection_name=self.collection_name, query_vector=query_vector, limit=limit)
        return [{
            "score": round(r.score, 3),
            "text": r.payload.get("text"),
            "claim_type": r.payload.get("claim_type"),
            "title": r.payload.get("title"),
            "authors": r.payload.get("authors"),
            "year": r.payload.get("year"),
            "doi": r.payload.get("doi"),
            "journal": r.payload.get("journal")
        } for r in results]


if __name__ == "__main__":
    indexer = VectorIndexer()
    indexer.index_all_from_data()
    results = indexer.search("strategic behavior under uncertainty")
    print(json.dumps(results, indent=2))