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
from src.utils.paper_id_utils import PaperIDGenerator


class VectorIndexer:
    def __init__(self, paper_root: str = "./data/papers/", config_path: str = "config/qdrant_config.json"):
        self.paper_root = paper_root
        self.config_path = config_path
        
        # Load Qdrant configuration
        self.qdrant_config = self._load_qdrant_config()
        
        # Initialize Qdrant client with server connection
        client_kwargs = {
            "host": self.qdrant_config.get("host", "localhost"),
            "port": self.qdrant_config.get("port", 6333),
            "prefer_grpc": self.qdrant_config.get("prefer_grpc", False),
            "https": self.qdrant_config.get("https", False),
            "timeout": self.qdrant_config.get("timeout", 60.0)
        }
        
        # Add optional parameters only if they exist
        if self.qdrant_config.get("api_key"):
            client_kwargs["api_key"] = self.qdrant_config["api_key"]
        if self.qdrant_config.get("prefix"):
            client_kwargs["prefix"] = self.qdrant_config["prefix"]
        
        self.client = QdrantClient(**client_kwargs)
        
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.paper_id_generator = PaperIDGenerator()
        
        # Ensure all necessary collections exist
        self._ensure_collections()
    
    def _load_qdrant_config(self) -> dict:
        """Load Qdrant configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Qdrant config file {self.config_path} not found, using defaults")
            return {
                "host": "localhost",
                "port": 6333,
                "grpc_port": 6334,
                "prefer_grpc": False,
                "https": False,
                "timeout": 60.0
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing Qdrant config: {e}")
            return {
                "host": "localhost", 
                "port": 6333,
                "grpc_port": 6334,
                "prefer_grpc": False,
                "https": False,
                "timeout": 60.0
            }
    
    def _ensure_collections(self):
        """Ensure all necessary collections exist with proper configuration"""
        collections_config = self.qdrant_config.get("collections", {})
        default_collections = ["sentences", "paragraphs", "sections", "citations"]
        
        for collection_name in default_collections:
            try:
                # Check if collection exists
                self.client.get_collection(collection_name)
                print(f"Collection '{collection_name}' already exists")
            except Exception as e:
                # If collection doesn't exist, create it
                collection_config = collections_config.get(collection_name, {})
                vector_size = collection_config.get("vector_size", 384)
                distance = Distance.COSINE
                
                if collection_config.get("distance", "Cosine").lower() == "euclidean":
                    distance = Distance.EUCLID
                elif collection_config.get("distance", "Cosine").lower() == "dot":
                    distance = Distance.DOT
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance)
                )
                print(f"Created collection '{collection_name}' with vector_size={vector_size}, distance={distance}")

    def index_sentences(self, paper_id: str, sentences: List[str], metadata: dict,
                        claim_types: Optional[List[str]] = None):
        """使用统一的paper_id（由PaperIDGenerator生成）来索引句子"""
        vectors = self.model.encode(sentences, normalize_embeddings=True)
        points = []
        for i, (vector, text) in enumerate(zip(vectors, sentences)):
            payload = {
                "paper_id": paper_id,  # 现在这个paper_id是由PaperIDGenerator生成的
                "sentence_index": i,
                "text": text,
                "sentence_type": claim_types[i] if claim_types else "unspecified",
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", []),
                "year": metadata.get("year", "Unknown"),
                "doi": metadata.get("doi", "Unknown"),
                "journal": metadata.get("journal", "Unknown"),
                "publisher": metadata.get("publisher", "Unknown"),
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

        self.client.upsert(collection_name="sentences", points=points)
        print(f"✅ Indexed {len(points)} sentences for paper {paper_id}")

    def index_paragraphs(self, paper_id: str, paragraphs: List[Dict], metadata: dict):
        """索引段落级向量"""
        if not paragraphs:
            return
            
        paragraph_texts = [p.get("text", "") for p in paragraphs]
        vectors = self.model.encode(paragraph_texts, normalize_embeddings=True)
        points = []
        
        for i, (vector, paragraph) in enumerate(zip(vectors, paragraphs)):
            payload = {
                "paper_id": paper_id,
                "paragraph_index": i,
                "text": paragraph.get("text", ""),
                "section": paragraph.get("section", ""),
                "citation_count": paragraph.get("citation_count", 0),
                "sentence_count": paragraph.get("sentence_count", 0),
                "has_citations": paragraph.get("has_citations", False),
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", []),
                "year": metadata.get("year", "Unknown"),
                "doi": metadata.get("doi", "Unknown"),
                "journal": metadata.get("journal", "Unknown"),
                "publisher": metadata.get("publisher", "Unknown"),
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

        self.client.upsert(collection_name="paragraphs", points=points)
        print(f"✅ Indexed {len(points)} paragraphs for paper {paper_id}")

    def index_sections(self, paper_id: str, sections: List[Dict], metadata: dict):
        """索引章节级向量"""
        if not sections:
            return
            
        section_texts = [s.get("text", "") for s in sections]
        vectors = self.model.encode(section_texts, normalize_embeddings=True)
        points = []
        
        for i, (vector, section) in enumerate(zip(vectors, sections)):
            payload = {
                "paper_id": paper_id,
                "section_index": i,
                "text": section.get("text", ""),
                "section_title": section.get("title", ""),
                "section_type": section.get("type", ""),
                "paragraph_count": section.get("paragraph_count", 0),
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", []),
                "year": metadata.get("year", "Unknown"),
                "doi": metadata.get("doi", "Unknown"),
                "journal": metadata.get("journal", "Unknown"),
                "publisher": metadata.get("publisher", "Unknown"),
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

        self.client.upsert(collection_name="sections", points=points)
        print(f"✅ Indexed {len(points)} sections for paper {paper_id}")

    def index_citations(self, paper_id: str, citations: List[Dict], metadata: dict):
        """索引引用文本向量"""
        if not citations:
            return
            
        citation_texts = [c.get("text", "") for c in citations]
        vectors = self.model.encode(citation_texts, normalize_embeddings=True)
        points = []
        
        for i, (vector, citation) in enumerate(zip(vectors, citations)):
            payload = {
                "paper_id": paper_id,
                "citation_index": i,
                "text": citation.get("text", ""),
                "cited_paper_id": citation.get("cited_paper_id", ""),
                "citation_context": citation.get("context", ""),
                "confidence": citation.get("confidence", 0.0),
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", []),
                "year": metadata.get("year", "Unknown"),
                "doi": metadata.get("doi", "Unknown"),
                "journal": metadata.get("journal", "Unknown"),
                "publisher": metadata.get("publisher", "Unknown"),
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

        self.client.upsert(collection_name="citations", points=points)
        print(f"✅ Indexed {len(points)} citations for paper {paper_id}")

    def index_all_from_data(self):
        """从data/papers目录索引所有数据，使用PaperIDGenerator确保paper_id一致性"""
        for folder_name in os.listdir(self.paper_root):
            paper_dir = os.path.join(self.paper_root, folder_name)
            if not os.path.isdir(paper_dir):
                continue

            sentences_path = os.path.join(paper_dir, "sentences.jsonl")
            metadata_path = os.path.join(paper_dir, "metadata.json")

            if not os.path.exists(sentences_path) or not os.path.exists(metadata_path):
                print(f"[WARN] Missing files for {folder_name}, skipping...")
                continue

            # 读取metadata来生成正确的paper_id
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # 使用PaperIDGenerator生成paper_id，而不是直接使用文件夹名
            paper_id = self.paper_id_generator.generate_paper_id(
                title=metadata.get("title", ""),
                year=str(metadata.get("year", "")),
                authors=metadata.get("authors", [])
            )
            
            print(f"Processing {folder_name} -> paper_id: {paper_id}")

            with open(sentences_path, "r") as f:
                sentence_objs = [json.loads(line.strip()) for line in f if line.strip()]
            sentences = [s["text"] for s in sentence_objs]
            sentence_types = [s.get("sentence_type", s.get("claim_type", "unspecified")) for s in sentence_objs]

            self.index_sentences(paper_id=paper_id, sentences=sentences, metadata=metadata, claim_types=sentence_types)

    def search(self, query: str, collection_name: str = "sentences", limit: int = 5) -> List[Dict]:
        """在指定collection中搜索"""
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]
        results = self.client.search(collection_name=collection_name, query_vector=query_vector, limit=limit)
        return [{
            "score": round(r.score, 3),
            "text": r.payload.get("text"),
            "collection": collection_name,
            "sentence_type": r.payload.get("sentence_type"),
            "section": r.payload.get("section"),
            "citation_context": r.payload.get("citation_context"),
            "title": r.payload.get("title"),
            "authors": r.payload.get("authors"),
            "year": r.payload.get("year"),
            "doi": r.payload.get("doi"),
            "journal": r.payload.get("journal"),
            "paper_id": r.payload.get("paper_id")
        } for r in results]

    def search_all_collections(self, query: str, limit_per_collection: int = 3) -> Dict[str, List[Dict]]:
        """在所有collection中搜索"""
        results = {}
        collections = ["sentences", "paragraphs", "sections", "citations"]
        
        for collection in collections:
            try:
                results[collection] = self.search(query, collection, limit_per_collection)
            except Exception as e:
                print(f"Error searching in {collection}: {e}")
                results[collection] = []
        
        return results


if __name__ == "__main__":
    indexer = VectorIndexer()
    indexer.index_all_from_data()
    
    # 测试跨collection搜索
    print("\n=== 测试跨collection搜索 ===")
    results = indexer.search_all_collections("strategic behavior under uncertainty", limit_per_collection=2)
    for collection, coll_results in results.items():
        print(f"\n{collection.upper()}:")
        if coll_results:
            for result in coll_results:
                print(f"  Score: {result['score']} - {result['text'][:100]}...")
        else:
            print("  No results found")
    
    # 测试单个collection搜索
    print("\n=== 测试sentences collection搜索 ===")
    sentence_results = indexer.search("strategic behavior under uncertainty", "sentences")
    print(json.dumps(sentence_results, indent=2))