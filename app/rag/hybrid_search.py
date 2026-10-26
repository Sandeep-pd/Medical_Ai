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


def hybrid_search(
    question: str,
    chunks: Sequence[dict],
    top_k: int = 5,
    dense_results: Sequence[dict] = (),
) -> list[dict]:
    if not chunks:
        lexical_results = []
    else:
        lexical_results = _lexical_search(question, chunks, top_k)

    if not dense_results:
        return lexical_results
    return _reciprocal_rank_fusion(lexical_results, dense_results, top_k)


def _lexical_search(question: str, chunks: Sequence[dict], top_k: int) -> list[dict]:
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


def _reciprocal_rank_fusion(
    lexical_results: Sequence[dict],
    dense_results: Sequence[dict],
    top_k: int,
) -> list[dict]:
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}
    for results in (lexical_results, dense_results):
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["chunk_id"]
            chunks_by_id[chunk_id] = chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)

    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [
        {**chunks_by_id[chunk_id], "score": scores[chunk_id]}
        for chunk_id in ranked_ids[:top_k]
    ]
