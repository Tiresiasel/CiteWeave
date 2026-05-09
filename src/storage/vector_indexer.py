"""
vector_indexer.py
Module for generating embeddings and storing them in Qdrant.
"""

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.utils.env_config import (
    LOCAL_EMBEDDING_DEFAULT_MODEL,
    OPENAI_EMBEDDING_DEFAULT_MODEL,
    get_embedding_api_key,
    get_embedding_dimensions,
    get_embedding_model,
    get_embedding_provider,
)
from src.utils.paper_id_utils import PaperIDGenerator

logging.basicConfig(level=logging.WARNING)

DEFAULT_QDRANT_CONFIG: Dict[str, Any] = {
    "host": "localhost",
    "port": 6333,
    "grpc_port": 6334,
    "prefer_grpc": False,
    "https": False,
    "timeout": 60.0,
    "embedding": {
        "provider": "local",
        "local": {
            "model": LOCAL_EMBEDDING_DEFAULT_MODEL,
            "vector_size": 384,
            "normalize": True,
        },
        "openai": {
            "model": OPENAI_EMBEDDING_DEFAULT_MODEL,
            "vector_size": 1536,
            "dimensions": None,
            "batch_size": 128,
        },
    },
    "collections": {
        "sentences": {"distance": "Cosine"},
        "paragraphs": {"distance": "Cosine"},
        "sections": {"distance": "Cosine"},
        "citations": {"distance": "Cosine"},
    },
}


class LocalSentenceTransformerEmbedder:
    """SentenceTransformers-backed local embedding provider."""

    provider = "local"

    def __init__(self, model_name: str, vector_size: Optional[int] = None, normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)
        inferred_size = None
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            inferred_size = self.model.get_sentence_embedding_dimension()
        self.vector_size = int(vector_size or inferred_size or 384)

    def encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=self.normalize)
        return [_as_float_list(vector) for vector in vectors]


class OpenAIEmbeddingClient:
    """OpenAI Embeddings API provider."""

    provider = "openai"

    def __init__(
        self,
        model_name: str,
        vector_size: Optional[int] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 128,
        api_key: str = "",
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.dimensions = dimensions
        self.vector_size = int(vector_size or dimensions or 1536)
        self.batch_size = int(batch_size or 128)
        if self.batch_size < 1:
            raise ValueError("OpenAI embedding batch_size must be >= 1")
        if not api_key:
            raise ValueError(
                "OpenAI embeddings require OPENAI_API_KEY or CITEWEAVE_EMBEDDING_API_KEY. "
                "Set CITEWEAVE_EMBEDDING_PROVIDER=local to use local embeddings."
            )
        self.client = OpenAI(api_key=api_key)

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            request: Dict[str, Any] = {"model": self.model_name, "input": chunk}
            if self.dimensions:
                request["dimensions"] = self.dimensions
            response = self.client.embeddings.create(**request)
            embeddings.extend(_as_float_list(item.embedding) for item in response.data)
        return embeddings


def _as_float_list(vector: Any) -> List[float]:
    """Convert numpy arrays, tensors, and plain iterables into JSON/Qdrant-safe vectors."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _deep_merge(defaults: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested configuration dictionaries without mutating the defaults."""
    merged = dict(defaults)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalise_embedding_provider(provider: str) -> str:
    provider = (provider or "local").strip().lower().replace("-", "_")
    aliases = {
        "sentence_transformer": "local",
        "sentence_transformers": "local",
        "transformer": "local",
        "local_transformer": "local",
    }
    return aliases.get(provider, provider)


class VectorIndexer:
    def __init__(self, paper_root: str = "./data/papers/", config_path: str = "config/qdrant_config.json"):
        self.paper_root = paper_root
        self.config_path = config_path

        # Load Qdrant and embedding configuration.
        self.qdrant_config = self._load_qdrant_config()
        self.embedding_config = self._resolve_embedding_config()
        self.embedder = self._build_embedder()

        # Initialize Qdrant client with server connection.
        client_kwargs = {
            "host": self.qdrant_config.get("host", "localhost"),
            "port": self.qdrant_config.get("port", 6333),
            "prefer_grpc": self.qdrant_config.get("prefer_grpc", False),
            "https": self.qdrant_config.get("https", False),
            "timeout": self.qdrant_config.get("timeout", 60.0),
        }

        # Add optional parameters only if they exist.
        if self.qdrant_config.get("api_key"):
            client_kwargs["api_key"] = self.qdrant_config["api_key"]
        if self.qdrant_config.get("prefix"):
            client_kwargs["prefix"] = self.qdrant_config["prefix"]

        self.client = QdrantClient(**client_kwargs)
        self.paper_id_generator = PaperIDGenerator()

        # Ensure all necessary collections exist.
        self._ensure_collections()

    def _load_qdrant_config(self) -> dict:
        """Load Qdrant configuration from JSON file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            return _deep_merge(DEFAULT_QDRANT_CONFIG, loaded)
        except FileNotFoundError:
            logging.warning("Warning: Qdrant config file %s not found, using defaults", self.config_path)
            return dict(DEFAULT_QDRANT_CONFIG)
        except json.JSONDecodeError as e:
            logging.error("Error parsing Qdrant config: %s", e)
            return dict(DEFAULT_QDRANT_CONFIG)

    def _resolve_embedding_config(self) -> Dict[str, Any]:
        """Resolve embedding provider configuration from qdrant_config.json plus env overrides."""
        configured = self.qdrant_config.get("embedding", {})
        provider = _normalise_embedding_provider(get_embedding_provider() or configured.get("provider", "local"))
        if provider not in {"local", "openai"}:
            raise ValueError(f"Unsupported embedding provider '{provider}'. Use 'local' or 'openai'.")

        provider_config = dict(configured.get(provider, {}))
        provider_config["provider"] = provider

        model_override = get_embedding_model()
        if model_override:
            provider_config["model"] = model_override

        dimensions_override = get_embedding_dimensions()
        if dimensions_override is not None:
            provider_config["dimensions"] = dimensions_override
            provider_config["vector_size"] = dimensions_override

        if provider == "local":
            provider_config.setdefault("model", LOCAL_EMBEDDING_DEFAULT_MODEL)
            provider_config.setdefault("vector_size", 384)
            provider_config.setdefault("normalize", True)
        else:
            provider_config.setdefault("model", OPENAI_EMBEDDING_DEFAULT_MODEL)
            provider_config.setdefault("vector_size", provider_config.get("dimensions") or 1536)
            provider_config.setdefault("dimensions", None)
            provider_config.setdefault("batch_size", 128)
            provider_config["api_key"] = get_embedding_api_key()

        return provider_config

    def _build_embedder(self):
        """Instantiate the configured embedding provider."""
        provider = self.embedding_config["provider"]
        if provider == "local":
            return LocalSentenceTransformerEmbedder(
                model_name=self.embedding_config["model"],
                vector_size=self.embedding_config.get("vector_size"),
                normalize=bool(self.embedding_config.get("normalize", True)),
            )
        if provider == "openai":
            return OpenAIEmbeddingClient(
                model_name=self.embedding_config["model"],
                vector_size=self.embedding_config.get("vector_size"),
                dimensions=self.embedding_config.get("dimensions"),
                batch_size=int(self.embedding_config.get("batch_size", 128)),
                api_key=self.embedding_config.get("api_key", ""),
            )
        raise ValueError(f"Unsupported embedding provider '{provider}'")

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for non-empty text strings."""
        clean_texts = [text or "" for text in texts]
        if not clean_texts:
            return []
        vectors = self.embedder.encode(clean_texts)
        expected_size = self.embedder.vector_size
        for vector in vectors:
            if len(vector) != expected_size:
                raise ValueError(
                    f"Embedding provider '{self.embedder.provider}' returned vector size {len(vector)}; "
                    f"expected {expected_size}. Check config/qdrant_config.json."
                )
        return vectors

    def _ensure_collections(self):
        """Ensure all necessary collections exist with the configured embedding dimensions."""
        collections_config = self.qdrant_config.get("collections", {})
        default_collections = ["sentences", "paragraphs", "sections", "citations"]

        for collection_name in default_collections:
            collection_config = collections_config.get(collection_name, {})
            try:
                collection_info = self.client.get_collection(collection_name)
                existing_size = self._collection_vector_size(collection_info)
                if existing_size and existing_size != self.embedder.vector_size:
                    logging.warning(
                        "Collection '%s' has vector_size=%s but configured embedding provider '%s' outputs %s. "
                        "Recreate the collection before switching embedding providers.",
                        collection_name,
                        existing_size,
                        self.embedder.provider,
                        self.embedder.vector_size,
                    )
                logging.debug("Collection '%s' already exists", collection_name)
            except Exception:
                vector_size = int(collection_config.get("vector_size") or self.embedder.vector_size)
                distance = Distance.COSINE

                if collection_config.get("distance", "Cosine").lower() == "euclidean":
                    distance = Distance.EUCLID
                elif collection_config.get("distance", "Cosine").lower() == "dot":
                    distance = Distance.DOT

                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance),
                )
                logging.info(
                    "Created collection '%s' with vector_size=%s, distance=%s",
                    collection_name,
                    vector_size,
                    distance,
                )

    @staticmethod
    def _collection_vector_size(collection_info: Any) -> Optional[int]:
        """Best-effort extraction of vector size from Qdrant collection metadata."""
        try:
            vectors = collection_info.config.params.vectors
            if hasattr(vectors, "size"):
                return int(vectors.size)
            if isinstance(vectors, dict):
                first = next(iter(vectors.values()))
                if hasattr(first, "size"):
                    return int(first.size)
        except Exception:
            return None
        return None

    def _point_id(self, collection_name: str, paper_id: str, index: int) -> str:
        """Return a stable Qdrant point id for idempotent re-indexing."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"citeweave:{collection_name}:{paper_id}:{index}"))

    def _delete_existing_points_for_paper(self, collection_name: str, paper_id: str) -> None:
        """Remove previous vectors for a paper before re-indexing it.

        Older CiteWeave builds used random Qdrant point ids. Reprocessing the
        same PDF could therefore append duplicate vectors even though Neo4j
        graph writes are keyed by paper id. Deleting by payload keeps resumed
        Zotero ingestion idempotent across both old random ids and the stable
        ids used below.
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
                ),
                wait=True,
            )
        except Exception as exc:  # pragma: no cover - defensive around external Qdrant
            logging.warning(
                "Failed to clear existing %s vectors for paper %s before re-indexing: %s",
                collection_name,
                paper_id,
                exc,
            )

    def index_sentences(
        self,
        paper_id: str,
        sentences: List[str],
        metadata: dict,
        claim_types: Optional[List[str]] = None,
    ):
        """Use the canonical paper_id (generated by PaperIDGenerator) to index sentences."""
        if not sentences:
            return
        self._delete_existing_points_for_paper("sentences", paper_id)
        vectors = self._embed_texts(sentences)
        points = []
        for i, (vector, text) in enumerate(zip(vectors, sentences)):
            payload = {
                "paper_id": paper_id,
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
            points.append(PointStruct(id=self._point_id("sentences", paper_id, i), vector=vector, payload=payload))

        self.client.upsert(collection_name="sentences", points=points)
        logging.info("✅ Indexed %s sentences for paper %s", len(points), paper_id)

    def index_paragraphs(self, paper_id: str, paragraphs: List[Dict], metadata: dict):
        """Index paragraph-level vectors."""
        if not paragraphs:
            return
        self._delete_existing_points_for_paper("paragraphs", paper_id)

        paragraph_texts = [p.get("text", "") for p in paragraphs]
        vectors = self._embed_texts(paragraph_texts)
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
            points.append(PointStruct(id=self._point_id("paragraphs", paper_id, i), vector=vector, payload=payload))

        self.client.upsert(collection_name="paragraphs", points=points)
        logging.info("✅ Indexed %s paragraphs for paper %s", len(points), paper_id)

    def index_sections(self, paper_id: str, sections: List[Dict], metadata: dict):
        """Index section-level vectors."""
        if not sections:
            return
        self._delete_existing_points_for_paper("sections", paper_id)

        section_texts = [s.get("text", "") for s in sections]
        vectors = self._embed_texts(section_texts)
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
            points.append(PointStruct(id=self._point_id("sections", paper_id, i), vector=vector, payload=payload))

        self.client.upsert(collection_name="sections", points=points)
        logging.info("✅ Indexed %s sections for paper %s", len(points), paper_id)

    def index_citations(self, paper_id: str, citations: List[Dict], metadata: dict):
        """Index citation text vectors."""
        if not citations:
            return
        self._delete_existing_points_for_paper("citations", paper_id)

        citation_texts = [c.get("text", "") for c in citations]
        vectors = self._embed_texts(citation_texts)
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
            points.append(PointStruct(id=self._point_id("citations", paper_id, i), vector=vector, payload=payload))

        self.client.upsert(collection_name="citations", points=points)
        logging.info("✅ Indexed %s citations for paper %s", len(points), paper_id)

    def index_all_from_data(self):
        """Index all data from data/papers using PaperIDGenerator for consistent paper_id values."""
        for folder_name in os.listdir(self.paper_root):
            paper_dir = os.path.join(self.paper_root, folder_name)
            if not os.path.isdir(paper_dir):
                continue

            sentences_path = os.path.join(paper_dir, "sentences.jsonl")
            metadata_path = os.path.join(paper_dir, "metadata.json")

            if not os.path.exists(sentences_path) or not os.path.exists(metadata_path):
                logging.warning("[WARN] Missing files for %s, skipping...", folder_name)
                continue

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            paper_id = self.paper_id_generator.generate_paper_id(
                title=metadata.get("title", ""),
                year=str(metadata.get("year", "")),
                authors=metadata.get("authors", []),
            )

            logging.info("Processing %s -> paper_id: %s", folder_name, paper_id)

            with open(sentences_path, "r", encoding="utf-8") as f:
                sentence_objs = [json.loads(line.strip()) for line in f if line.strip()]
            sentences = [s["text"] for s in sentence_objs]
            sentence_types = [s.get("sentence_type", s.get("claim_type", "unspecified")) for s in sentence_objs]

            self.index_sentences(paper_id=paper_id, sentences=sentences, metadata=metadata, claim_types=sentence_types)

    def search(self, query: str, collection_name: str = "sentences", limit: int = 5) -> List[Dict]:
        """Search a specific Qdrant collection."""
        query_vector = self._embed_texts([query])[0]
        results = self.client.search(collection_name=collection_name, query_vector=query_vector, limit=limit)
        return [
            {
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
                "paper_id": r.payload.get("paper_id"),
            }
            for r in results
        ]

    def search_all_collections(self, query: str, limit_per_collection: int = 3) -> Dict[str, List[Dict]]:
        """Search all configured content collections."""
        results = {}
        collections = ["sentences", "paragraphs", "sections", "citations"]

        for collection in collections:
            try:
                results[collection] = self.search(query, collection, limit_per_collection)
            except Exception as e:
                logging.error("Error searching in %s: %s", collection, e)
                results[collection] = []

        return results


if __name__ == "__main__":
    indexer = VectorIndexer()
    indexer.index_all_from_data()

    logging.info("\n=== Cross-collection search smoke test ===")
    results = indexer.search_all_collections("strategic behavior under uncertainty", limit_per_collection=2)
    for collection, coll_results in results.items():
        logging.info("\n%s:", collection.upper())
        if coll_results:
            for result in coll_results:
                logging.info("  Score: %s - %s...", result["score"], result["text"][:100])
        else:
            logging.info("  No results found")

    logging.info("\n=== Sentences collection search smoke test ===")
    sentence_results = indexer.search("strategic behavior under uncertainty", "sentences")
    print(json.dumps(sentence_results, indent=2))
