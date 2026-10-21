"""Laboratory result schema."""

from typing import Optional

from pydantic import BaseModel, Field


class LabResult(BaseModel):
    test_name: str
    value: float
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    page: int = Field(default=1, ge=1)
    source_text: str
    extraction_method: str
