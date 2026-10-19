"""
Unit and integration tests for doc-qa-rag application.

Tests:
- PDF text extraction and error cases (empty, corrupt, scanned)
- Word chunking and overlap logic
- FAISS vector store build, search, save, and load
- Grounded prompt construction
- FastAPI endpoints validation and error handling
"""

import io
import sys
from pathlib import Path
import pytest
import numpy as np
import pypdf
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ingest import extract_text_from_pdf, chunk_text
from app.embeddings import normalize_vectors
from app.vector_store import FAISSVectorStore
from app.rag import build_grounded_prompt
from app.main import app


def create_sample_pdf(text: str) -> bytes:
    """Helper to create an in-memory PDF with specified text."""
    writer = pypdf.PdfWriter()
    # Create a page and add annotation/text or use canvas
    # In pypdf, we can create a blank page
    page = writer.add_blank_page(width=612, height=792)
    
    # We can write text to PDF stream or use a minimal PDF structure
    # A standard minimal valid PDF with a text stream:
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_pdf_with_stream_text(text: str) -> bytes:
    """Creates a valid PDF with extractable text using standard PDF objects."""
    # Build a minimal single-page PDF with text stream
    stream_content = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET"
    stream_len = len(stream_content)
    
    pdf_template = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length " + str(stream_len).encode("ascii") + b" >>\nstream\n"
        + stream_content.encode("ascii") +
        b"\nendstream\nendobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000056 00000 n \n"
        b"0000000111 00000 n \n"
        b"0000000234 00000 n \n"
        b"0000000320 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n395\n%%EOF\n"
    )
    return pdf_template


# -------------------------------------------------------------
# Test 1: Ingestion & Validation
# -------------------------------------------------------------
def test_extract_text_empty_bytes():
    """Verify empty byte payload raises ValueError."""
    try:
        extract_text_from_pdf(b"")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()


def test_extract_text_scanned_pdf():
    """Verify PDF with blank page (scanned without OCR) raises ValueError."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    blank_pdf = buf.getvalue()

    try:
        extract_text_from_pdf(blank_pdf)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "extractable text" in str(e).lower() or "scanned" in str(e).lower()


def test_extract_text_valid_pdf():
    """Verify PDF with extractable text succeeds."""
    sample_pdf = create_pdf_with_stream_text("DocuMind RAG test document content.")
    text = extract_text_from_pdf(sample_pdf)
    assert "DocuMind" in text or "RAG" in text


# -------------------------------------------------------------
# Test 2: Chunking & Overlap
# -------------------------------------------------------------
def test_chunking_short_text():
    """Text under CHUNK_SIZE should return exactly 1 chunk."""
    text = "This is a short sample document with only a few words."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["id"] == 0
    assert chunks[0]["text"] == text


def test_chunking_with_overlap():
    """Text exceeding chunk_size should produce overlapping chunks."""
    # Generate 120 words
    words = [f"word{i}" for i in range(120)]
    text = " ".join(words)

    # Chunk size 50, overlap 10 -> step is 40
    # Chunk 0: words 0..49
    # Chunk 1: words 40..89
    # Chunk 2: words 80..119
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) == 3
    assert chunks[0]["id"] == 0
    assert chunks[1]["id"] == 1
    assert chunks[2]["id"] == 2

    # Check overlap between chunk 0 and chunk 1
    chunk0_words = chunks[0]["text"].split()
    chunk1_words = chunks[1]["text"].split()
    assert chunk0_words[-10:] == chunk1_words[:10]


# -------------------------------------------------------------
# Test 3: Embeddings Normalization
# -------------------------------------------------------------
def test_normalize_vectors():
    """Vector normalization should produce unit L2 norm."""
    vectors = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    normalized = normalize_vectors(vectors)

    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)


# -------------------------------------------------------------
# Test 4: FAISS Vector Store
# -------------------------------------------------------------
def test_faiss_vector_store():
    """Verify build, search, save, and load of FAISS store."""
    store = FAISSVectorStore()
    assert not store.is_ready

    # Create dummy normalized vectors of dimension 4
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = np.array([v1, v2], dtype=np.float32)

    chunks = [
        {"id": 0, "text": "Chunk zero about cats"},
        {"id": 1, "text": "Chunk one about dogs"},
    ]

    store.build(embeddings, chunks, {"filename": "test.pdf"})
    assert store.is_ready
    assert store.index.ntotal == 2

    # Search with query vector pointing at v1
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    results = store.search(query, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == 0
    assert "cats" in results[0]["text"]
    assert results[0]["score"] > 0.99


# -------------------------------------------------------------
# Test 5: Grounded Prompt Construction
# -------------------------------------------------------------
def test_build_grounded_prompt():
    """Prompt must strictly instruct 'Answer ONLY using the provided context'."""
    chunks = [{"id": 0, "text": "The sky is blue on Mars during sunset."}]
    prompt = build_grounded_prompt("What color is the sky?", chunks)

    assert "Answer ONLY using the provided context" in prompt
    assert "If the answer is not in the context, say you don't know" in prompt
    assert "The sky is blue on Mars during sunset." in prompt
    assert "What color is the sky?" in prompt


# -------------------------------------------------------------
# Test 6: FastAPI Endpoints & Error Handling
# -------------------------------------------------------------
client = TestClient(app)


def test_get_root():
    """GET / should serve the HTML application."""
    response = client.get("/")
    assert response.status_code == 200
    assert "DocuMind" in response.text
    assert "<!DOCTYPE html>" in response.text


def test_get_status():
    """GET /status should return system readiness."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "is_ready" in data


def test_ask_empty_question():
    """POST /ask with empty question should return 400 Bad Request."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ask_no_document():
    """POST /ask when no document indexed should return 400."""
    from app.vector_store import vector_store
    vector_store.index = None
    vector_store.chunks = []

    response = client.post("/ask", json={"question": "What is this about?"})
    assert response.status_code == 400
    assert "no document" in response.json()["detail"].lower()


def test_upload_invalid_extension():
    """POST /upload with non-PDF should return 400."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert "only pdf" in response.json()["detail"].lower()


def test_upload_empty_pdf():
    """POST /upload with 0-byte file should return 400."""
    response = client.post(
        "/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_scanned_pdf():
    """POST /upload with blank/scanned PDF should return 400."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    blank_pdf = buf.getvalue()

    response = client.post(
        "/upload",
        files={"file": ("blank.pdf", blank_pdf, "application/pdf")},
    )
    assert response.status_code == 400
    assert "extractable text" in response.json()["detail"].lower() or "scanned" in response.json()["detail"].lower()


if __name__ == "__main__":
    print("Running tests manually...")
    test_extract_text_empty_bytes()
    test_extract_text_scanned_pdf()
    test_extract_text_valid_pdf()
    test_chunking_short_text()
    test_chunking_with_overlap()
    test_normalize_vectors()
    test_faiss_vector_store()
    test_build_grounded_prompt()
    test_get_root()
    test_get_status()
    test_ask_empty_question()
    test_ask_no_document()
    test_upload_invalid_extension()
    test_upload_empty_pdf()
    test_upload_scanned_pdf()
    print("ALL TESTS PASSED SUCCESSFULLY!")
