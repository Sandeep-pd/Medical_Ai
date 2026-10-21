"""Document metadata schemas."""

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    laboratory_results: list[dict]
