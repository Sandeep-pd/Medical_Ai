"""API request and response models."""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class QuestionSource(BaseModel):
    source_label: str
    document_id: str
    page: int
    chunk_id: str
    text: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[QuestionSource]


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)


class PredictionResponse(BaseModel):
    score: float
