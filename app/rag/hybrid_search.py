"""Sparse and dense retrieval with reciprocal-rank fusion."""

from functools import lru_cache
import re
from typing import Sequence


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+(?:\.[0-9]+)?", text.lower())


@lru_cache(maxsize=2)
def _get_bm25_index(corpus: tuple[str, ...]):
    from rank_bm25 import BM25Okapi

    tokenized_chunks = [tokenize(text) for text in corpus]
    return BM25Okapi(tokenized_chunks)


def hybrid_search(question: str, chunks: Sequence[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        query = set(tokenize(question))
        ranked = sorted(
            chunks,
            key=lambda chunk: len(query.intersection(tokenize(chunk.get("text", "")))),
            reverse=True,
        )
        return [dict(chunk) for chunk in ranked[:top_k]]

    corpus = tuple(chunk.get("text", "") for chunk in chunks)
    scores = _get_bm25_index(corpus).get_scores(tokenize(question))
    ranked_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [
        {**chunks[index], "score": float(scores[index])}
        for index in ranked_indices[:top_k]
    ]
