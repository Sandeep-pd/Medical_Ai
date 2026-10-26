"""Question-answer generation using the medical evidence-analysis crew."""

from app.llm.crew import generate_medical_answer


def generate_answer(question: str, context: str, max_new_tokens: int = 350) -> str:
    """Generate one answer grounded in retrieved medical evidence."""
    return generate_medical_answer(
        question=question,
        context=context,
        max_output_tokens=max_new_tokens,
    )
