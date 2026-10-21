"""Question-answer generation using the medical evidence-analysis crew."""

from app.llm.crew import run_medical_crew


def generate_answer(question: str, context: str, max_new_tokens: int = 350) -> str:
    """Generate an answer after three sequential CrewAI evidence tasks."""
    return run_medical_crew(
        question=question,
        context=context,
        max_output_tokens=max_new_tokens,
    )
