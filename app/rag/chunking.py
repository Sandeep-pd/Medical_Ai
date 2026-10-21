"""Split extracted documents into chunks suitable for retrieval."""


def chunk_documents(
    documents: list[dict],
    document_id: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0.")

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        splitter = None
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    chunks = []
    for document in documents:
        text = document.get("text", "").strip()
        if not text:
            continue
        pieces = splitter.split_text(text) if splitter else _split_text(text, chunk_size, chunk_overlap)
        for index, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{document_id}_p{document.get('page', 1)}_c{index}",
                "document_id": document_id,
                "page": document.get("page", 1),
                "text": piece,
                "extraction_method": document.get("method", "unknown"),
            })
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return [chunk for chunk in chunks if chunk]
