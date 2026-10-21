"""Application settings loaded from environment variables."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", DATA_DIR / "documents"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
DEFAULT_DATASET = Path(os.getenv("DATASET_PATH", DATA_DIR / "medical.csv"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
ALLOWED_DOCUMENT_EXTENSIONS = {".csv", ".docx", ".pdf", ".png", ".jpg", ".jpeg"}


def ensure_directories() -> None:
    """Create writable application data directories."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
