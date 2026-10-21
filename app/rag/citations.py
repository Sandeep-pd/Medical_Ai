"""Source labels and citation context formatting."""

from typing import Sequence


def format_sources(results: Sequence[dict]) -> list[dict]:
    return [
        {
            "source_label": f"Source {index}",
            "document_id": item["document_id"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
        }
        for index, item in enumerate(results, start=1)
    ]


def format_context_with_citations(results: Sequence[dict]) -> str:
    return "\n\n".join(
        f"[Source {index}: Document {item['document_id']}, Page {item['page']}]\n{item['text']}"
        for index, item in enumerate(results, start=1)
    )
