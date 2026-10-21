"""FastAPI application and command-line entry point."""

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import config
from app.api.documents import router as documents_router
from app.api.predictions import router as predictions_router
from app.api.questions import router as questions_router
from app.llm.crew import CrewAIAnswerError
from app.ml.predictor import evaluate_demo_model
from app.ml.validation import summarize_healthcare_csv

app = FastAPI(
    title="Medical AI RAG",
    description="Healthcare document ingestion, retrieval, and analysis API.",
    version="1.0.0",
)
app.include_router(documents_router, prefix="/api")
app.include_router(questions_router, prefix="/api")
app.include_router(predictions_router, prefix="/api")


@app.exception_handler(CrewAIAnswerError)
async def crewai_answer_error_handler(
    request: Request,
    exc: CrewAIAnswerError,
) -> JSONResponse:
    """Return a concise service error instead of an unhandled traceback."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "detail": str(exc),
            "retryable": exc.retryable,
        },
    )


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical AI RAG application")
    parser.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--evaluate-model",
        action="store_true",
        help="Train and evaluate the synthetic demo model, then exit.",
    )
    args = parser.parse_args()

    if args.summary_only and args.evaluate_model:
        parser.error("--summary-only and --evaluate-model cannot be used together.")

    config.ensure_directories()
    if args.evaluate_model:
        print(json.dumps(evaluate_demo_model(), indent=2))
        return

    if args.dataset.exists():
        print(json.dumps(summarize_healthcare_csv(args.dataset), indent=2, default=str))
    elif args.summary_only:
        parser.error(f"Dataset does not exist: {args.dataset}")

    if not args.summary_only:
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
