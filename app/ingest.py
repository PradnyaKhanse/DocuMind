"""
PDF Ingestion and Chunking Module.

Handles extracting raw text from uploaded PDF documents and slicing
it into overlapping windows for downstream embedding and semantic retrieval.
"""

import io
from typing import List, Dict, Any
import pypdf
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts plain text from raw PDF bytes using pypdf.

    Validates:
    1. The PDF is not empty (contains at least one page).
    2. The PDF is not a scanned/image-only document lacking OCR text.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        Extracted plain text across all pages.

    Raises:
        ValueError: If PDF is empty, unreadable, or contains no extractable text.
    """
    if not pdf_bytes:
        raise ValueError("The uploaded file is empty (0 bytes).")

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        raise ValueError(f"Failed to parse PDF document: {str(e)}")

    if len(reader.pages) == 0:
        raise ValueError("The uploaded PDF contains no pages.")

    extracted_pages: List[str] = []
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        # Normalize whitespace while preserving line/paragraph boundaries
        cleaned_text = page_text.strip()
        if cleaned_text:
            extracted_pages.append(cleaned_text)

    full_text = "\n\n".join(extracted_pages).strip()

    # If all pages returned empty string, document is likely scanned or image-only
    if not full_text:
        raise ValueError(
            "The uploaded PDF contains no extractable text. It may be scanned or image-based."
        )

    return full_text


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Splits continuous text into overlapping word windows.

    WHY CHUNKING EXISTS:
    Embedding models and LLMs operate on finite context windows. More importantly,
    a single vector cannot effectively capture the nuance of an entire 50-page book.
    Chunking divides the document into semantically cohesive units (passages) so that
    retrieval can pinpoint the exact passage containing the relevant answer.

    WHY CHUNK OVERLAP EXISTS:
    If a sentence or key definition falls across the boundary between chunk N and
    chunk N+1, splitting cleanly without overlap would sever that thought in half.
    Sliding the window forward with overlap ensures that boundary concepts are
    preserved intact in at least one (often both) adjacent chunks.

    Args:
        text: Normalized plain text to chunk.
        chunk_size: Maximum words per chunk (default: 500).
        chunk_overlap: Number of words shared between adjacent chunks (default: 50).

    Returns:
        List of dicts: [{"id": int, "text": str, "word_count": int}]
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    # Split on whitespace to get individual words
    words = text.split()
    if not words:
        return []

    # If the entire document fits within a single chunk, return it directly
    if len(words) <= chunk_size:
        return [
            {
                "id": 0,
                "text": " ".join(words),
                "word_count": len(words),
            }
        ]

    chunks: List[Dict[str, Any]] = []
    step = chunk_size - chunk_overlap
    chunk_id = 0

    for start_idx in range(0, len(words), step):
        end_idx = min(start_idx + chunk_size, len(words))
        chunk_words = words[start_idx:end_idx]
        chunk_str = " ".join(chunk_words).strip()

        if chunk_str:
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_str,
                    "word_count": len(chunk_words),
                }
            )
            chunk_id += 1

        # Stop if we reached the end of the text
        if end_idx >= len(words):
            break

    return chunks
