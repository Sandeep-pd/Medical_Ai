"""Document upload and ingestion endpoints."""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import config
from app.ingestion.layout_parser import ingest_document
from app.ingestion.lab_extractor import extract_lab_results
from app.rag.chunking import chunk_documents
from app.rag.indexing import index_chunks

router = APIRouter(tags=["documents"])
indexed_chunks: list[dict] = []


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Save an uploaded document, extract text and make chunks searchable."""
    config.ensure_directories()
    extension = Path(file.filename or "").suffix.lower()
    if extension not in config.ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Upload a CSV, DOCX, PDF, PNG, JPG, or JPEG file.",
        )

    content = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    destination = config.DOCUMENTS_DIR / f"{uuid4().hex}{extension}"
    destination.write_bytes(content)
    try:
        documents = ingest_document(destination)
    except (RuntimeError, ValueError, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not process document: {exc}") from exc

    document_id = destination.stem
    chunks = chunk_documents(documents, document_id=document_id)
    index_chunks(chunks)
    indexed_chunks.extend(chunks)
    return {
        "document_id": document_id,
        "filename": file.filename,
        "pages": len(documents),
        "chunks": len(chunks),
        "laboratory_results": extract_lab_results(documents),
    }
