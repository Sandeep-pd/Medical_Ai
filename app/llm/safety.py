"""Question and citation validation helpers."""

import re


def validate_question(question: str) -> tuple[bool, str]:
    if not isinstance(question, str):
        return False, "Question must be text."
    question = question.strip()
    if not question:
        return False, "Please enter a question."
    if len(question) > 2000:
        return False, "Question is too long."
    return True, "Input accepted."


def validate_source_references(answer: str, sources: list[dict]) -> dict:
    valid_labels = {f"[{source['source_label']}]" for source in sources}
    cited_labels = set(re.findall(r"\[Source\s+\d+\]", answer))
    invalid_labels = cited_labels - valid_labels
    return {
        "valid": not invalid_labels,
        "cited_labels": sorted(cited_labels),
        "invalid_labels": sorted(invalid_labels),
    }
