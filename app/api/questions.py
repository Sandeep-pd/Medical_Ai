"""Question answering endpoints."""

from fastapi import APIRouter, HTTPException

from app.api.documents import indexed_chunks
from app.llm.generator import generate_answer
from app.rag.citations import format_context_with_citations, format_sources
from app.rag.hybrid_search import hybrid_search
from app.rag.reranker import rerank_results
from app.schemas.response import QuestionRequest, QuestionResponse

router = APIRouter(tags=["questions"])


@router.post("/questions", response_model=QuestionResponse)
def ask_question(request: QuestionRequest) -> QuestionResponse:
    if not indexed_chunks:
        raise HTTPException(status_code=404, detail="Upload a document before asking a question.")

    candidates = hybrid_search(request.question, indexed_chunks, top_k=10)
    results = rerank_results(request.question, candidates, top_k=3)
    sources = format_sources(results)
    context = format_context_with_citations(results)
    answer = generate_answer(request.question, context)
    return QuestionResponse(question=request.question, answer=answer, sources=sources)
