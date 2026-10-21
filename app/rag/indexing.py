"""Vector index adapters."""

from typing import Any, Sequence


def build_qdrant_index(chunks: Sequence[dict], embeddings: Any) -> Any:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError("Vector indexing requires qdrant-client.") from exc
    client = QdrantClient(":memory:")
    collection_name = "medical_knowledge"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings.shape[1], distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(
            id=index,
            vector=embeddings[index].tolist(),
            payload=dict(chunk),
        )
        for index, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=collection_name, points=points)
    return client
