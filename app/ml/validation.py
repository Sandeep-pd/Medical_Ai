"""Healthcare CSV inspection and request validation."""

from pathlib import Path
from typing import Any

import pandas as pd


def summarize_healthcare_csv(path: str | Path) -> dict[str, Any]:
    dataset = pd.read_csv(path)
    summary: dict[str, Any] = {
        "rows": len(dataset),
        "columns": list(dataset.columns),
        "dtypes": dataset.dtypes.astype(str).to_dict(),
        "missing_values": dataset.isnull().sum().to_dict(),
        "sample": dataset.head(5).to_dict(orient="records"),
    }
    for column, key in (
        ("Medical Condition", "medical_condition_distribution"),
        ("Test Results", "test_results_distribution"),
    ):
        if column in dataset.columns:
            summary[key] = dataset[column].value_counts().to_dict()
    return summary
