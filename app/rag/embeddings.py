"""Sentence-transformer embedding model loader."""

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Embeddings require sentence-transformers.") from exc
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
