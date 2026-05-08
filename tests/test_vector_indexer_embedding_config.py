import json
import sys
from types import SimpleNamespace

from src.storage import vector_indexer as vi


class FakeQdrantClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.created = []
        self.upserts = []
        self.searches = []
        FakeQdrantClient.instances.append(self)

    def get_collection(self, collection_name):
        raise RuntimeError("collection missing")

    def create_collection(self, collection_name, vectors_config):
        self.created.append((collection_name, vectors_config.size))

    def upsert(self, collection_name, points):
        self.upserts.append((collection_name, points))

    def search(self, collection_name, query_vector, limit):
        self.searches.append((collection_name, query_vector, limit))
        return [SimpleNamespace(score=0.98, payload={"text": "hit"})]


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_sentence_embedding_dimension(self):
        return 7

    def encode(self, texts, normalize_embeddings=True):
        return [[float(i + 1)] * 7 for i, _ in enumerate(texts)]


def write_config(path, provider, vector_size, dimensions=None):
    path.write_text(
        json.dumps(
            {
                "embedding": {
                    "provider": provider,
                    "local": {"model": "fake-local", "vector_size": vector_size, "normalize": True},
                    "openai": {
                        "model": "fake-openai",
                        "vector_size": vector_size,
                        "dimensions": dimensions,
                        "batch_size": 2,
                    },
                },
                "collections": {
                    "sentences": {"distance": "Cosine"},
                    "paragraphs": {"distance": "Cosine"},
                    "sections": {"distance": "Cosine"},
                    "citations": {"distance": "Cosine"},
                },
            }
        )
    )


def test_vector_indexer_defaults_to_local_embedding_config(tmp_path, monkeypatch):
    FakeQdrantClient.instances = []
    monkeypatch.setattr(vi, "QdrantClient", FakeQdrantClient)
    monkeypatch.setattr(vi, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.delenv("CITEWEAVE_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("CITEWEAVE_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("CITEWEAVE_EMBEDDING_DIMENSIONS", raising=False)

    config_path = tmp_path / "qdrant_config.json"
    write_config(config_path, provider="local", vector_size=7)

    indexer = vi.VectorIndexer(paper_root=str(tmp_path), config_path=str(config_path))

    assert indexer.embedder.provider == "local"
    assert indexer.embedding_config["model"] == "fake-local"
    assert indexer.embedder.vector_size == 7
    client = FakeQdrantClient.instances[-1]
    assert client.created == [("sentences", 7), ("paragraphs", 7), ("sections", 7), ("citations", 7)]

    indexer.index_sentences("paper-1", ["alpha", "beta"], metadata={})
    assert len(client.upserts[0][1]) == 2
    assert len(client.upserts[0][1][0].vector) == 7


def test_vector_indexer_supports_openai_embedding_config(tmp_path, monkeypatch):
    FakeQdrantClient.instances = []
    monkeypatch.setattr(vi, "QdrantClient", FakeQdrantClient)
    monkeypatch.setenv("CITEWEAVE_EMBEDDING_API_KEY", "test-key")
    monkeypatch.delenv("CITEWEAVE_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("CITEWEAVE_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("CITEWEAVE_EMBEDDING_DIMENSIONS", raising=False)

    class FakeOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.embeddings = self

        def create(self, **request):
            dimensions = request.get("dimensions") or 5
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.25] * dimensions) for _ in request["input"]]
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    config_path = tmp_path / "qdrant_config.json"
    write_config(config_path, provider="openai", vector_size=5, dimensions=5)

    indexer = vi.VectorIndexer(paper_root=str(tmp_path), config_path=str(config_path))

    assert indexer.embedder.provider == "openai"
    assert indexer.embedding_config["model"] == "fake-openai"
    assert indexer.embedder.vector_size == 5
    client = FakeQdrantClient.instances[-1]
    assert client.created == [("sentences", 5), ("paragraphs", 5), ("sections", 5), ("citations", 5)]

    results = indexer.search("strategic uncertainty", collection_name="sentences", limit=1)
    assert results[0]["text"] == "hit"
    assert len(client.searches[0][1]) == 5
