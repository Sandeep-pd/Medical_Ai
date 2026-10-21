"""Extract simple reported lab values and reference intervals from text."""

import re
from typing import Sequence

from app.schemas.laboratory import LabResult

LAB_PATTERN = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z /()_-]{1,50}?)\s+"
    r"(?P<value>-?\d+(?:\.\d+)?)\s+"
    r"(?P<unit>[A-Za-z%µμ/\.-]+)\s+"
    r"(?P<low>-?\d+(?:\.\d+)?)\s*(?:-|–|to)\s*"
    r"(?P<high>-?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)


def extract_lab_results(documents: Sequence[dict]) -> list[dict]:
    results = []
    for document in documents:
        for line in document.get("text", "").splitlines():
            match = LAB_PATTERN.match(line)
            if match is None:
                continue
            result = LabResult(
                test_name=match.group("name").strip(),
                value=float(match.group("value")),
                unit=match.group("unit"),
                reference_range=f"{match.group('low')} - {match.group('high')}",
                page=document.get("page", 1),
                source_text=line.strip(),
                extraction_method=document.get("method", "unknown"),
            )
            results.append(result.model_dump())
    return results
