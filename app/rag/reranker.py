"""Optional cross-encoder reranking."""

from functools import lru_cache
from typing import Sequence


@lru_cache(maxsize=1)
def _get_reranker():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_results(question: str, candidates: Sequence[dict], top_k: int = 3) -> list[dict]:
    if not candidates:
        return []
    reranker = _get_reranker()
    if reranker is None:
        return [dict(item) for item in candidates[:top_k]]

    scores = reranker.predict([[question, item["text"]] for item in candidates])
    ranked = [{**item, "rerank_score": float(score)} for item, score in zip(candidates, scores)]
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:top_k]
