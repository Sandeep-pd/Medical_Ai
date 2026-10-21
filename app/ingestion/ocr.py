"""OCR helpers. OCR dependencies are imported only when OCR is requested."""

from typing import Any

import numpy as np
from PIL import Image


def create_ocr_engine() -> Any:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("OCR requires paddleocr and paddlepaddle.") from exc
    return PaddleOCR(use_textline_orientation=True, lang="en")


def extract_image_text(image: Image.Image, engine: Any = None) -> list[dict]:
    if engine is None:
        engine = create_ocr_engine()
    results = engine.predict(np.asarray(image.convert("RGB")))
    extracted: list[dict] = []
    for result in results or []:
        if not isinstance(result, dict):
            continue
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        for index, text in enumerate(texts):
            extracted.append({
                "text": text,
                "confidence": scores[index] if index < len(scores) else None,
            })
    return extracted
