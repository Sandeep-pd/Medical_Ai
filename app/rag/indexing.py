"""Persistent ChromaDB vector indexing and retrieval."""

from functools import lru_cache
from typing import Any, Sequence

from app import config
from app.rag.embeddings import get_embedding_model


@lru_cache(maxsize=2)
def _get_collection(path: str) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Vector indexing requires chromadb.") from exc

    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name="medical_knowledge",
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: Sequence[dict]) -> None:
    if not chunks:
        return

    collection = _get_collection(str(config.CHROMA_DIR))
    embeddings = get_embedding_model().encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                key: value
                for key, value in chunk.items()
                if key not in {"chunk_id", "text"} and value is not None
            }
            for chunk in chunks
        ],
    )


def collection_count() -> int:
    return _get_collection(str(config.CHROMA_DIR)).count()


def search_chroma(question: str, top_k: int = 10) -> list[dict]:
    if top_k <= 0:
        return []

    collection = _get_collection(str(config.CHROMA_DIR))
    count = collection.count()
    if not count:
        return []

    query_embedding = get_embedding_model().encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].tolist()
    matches = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["metadatas", "documents", "distances"],
    )
    results = []
    documents = matches["documents"][0] or []
    metadatas = matches["metadatas"][0] or []
    ids = matches["ids"][0] or []
    distances = matches["distances"][0] or []
    for index, document in enumerate(documents):
        result = dict(metadatas[index])
        result["text"] = document
        result["chunk_id"] = ids[index]
        result["score"] = 1.0 - distances[index]
        results.append(result)
    return results
