"""Demonstration prediction endpoint."""

from fastapi import APIRouter, HTTPException

from app.ml.predictor import evaluate_demo_model, predict_demo
from app.schemas.response import PredictionRequest, PredictionResponse

router = APIRouter(tags=["predictions"])


@router.post("/predictions", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        score = predict_demo(request.features)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(score=score)


@router.get("/predictions/evaluation")
def evaluate_prediction_model() -> dict:
    """Return holdout metrics and feature importance for the demo model."""
    try:
        return evaluate_demo_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
