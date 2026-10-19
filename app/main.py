"""
FastAPI Main Application Module.

Defines HTTP REST endpoints for PDF upload and processing, semantic Q&A (/ask),
static asset serving, CORS headers, and lifecycle initialization.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import (
    STATIC_DIR,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    DOCUMENT_INFO_PATH,
)
from app.ingest import extract_text_from_pdf, chunk_text
from app.embeddings import embed_chunks
from app.vector_store import vector_store
from app.rag import answer_question


# Pydantic Schemas
class AskRequest(BaseModel):
    question: str = Field(..., description="User's question to answer using the indexed PDF.")


class SourceChunk(BaseModel):
    id: int
    text: str
    score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class UploadResponse(BaseModel):
    filename: str
    chunks_count: int
    message: str


class StatusResponse(BaseModel):
    is_ready: bool
    filename: Optional[str] = None
    chunks_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Attempts to restore the FAISS index and metadata persisted from previous runs.
    """
    if FAISS_INDEX_PATH.exists() and CHUNKS_METADATA_PATH.exists():
        loaded = vector_store.load_from_disk(
            index_path=FAISS_INDEX_PATH,
            chunks_path=CHUNKS_METADATA_PATH,
            doc_info_path=DOCUMENT_INFO_PATH,
        )
        if loaded:
            print(
                f"[Startup] Restored FAISS vector index with {len(vector_store.chunks)} chunks "
                f"for document '{vector_store.document_info.get('filename', 'Unknown')}'."
            )
    yield


app = FastAPI(
    title="DocuMind - Document Q&A RAG",
    description="Production-grade RAG pipeline powered by FastAPI, FAISS, and Google Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the polished single-page frontend application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend template static/index.html not found.",
        )
    return FileResponse(index_file)


@app.get("/status", response_model=StatusResponse, summary="Check index and document readiness")
async def get_status():
    """Returns whether a PDF document is currently indexed and ready for Q&A."""
    if vector_store.is_ready:
        return StatusResponse(
            is_ready=True,
            filename=vector_store.document_info.get("filename", "Uploaded Document"),
            chunks_count=len(vector_store.chunks),
        )
    return StatusResponse(is_ready=False, filename=None, chunks_count=0)


@app.post("/upload", response_model=UploadResponse, summary="Upload and index a PDF document")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file, extracts plain text, partitions text into overlapping chunks,
    generates dense embeddings via Gemini text-embedding-004, and builds the FAISS vector index.
    """
    filename = file.filename or "uploaded.pdf"

    # Validate file extension
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF (.pdf) documents are supported.",
        )

    # Read binary content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if not content or len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes).",
        )

    # 1. Text Extraction
    try:
        raw_text = extract_text_from_pdf(content)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while extracting text: {str(e)}",
        )

    # 2. Overlapping Chunking
    try:
        chunks = chunk_text(raw_text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to chunk document text: {str(e)}",
        )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text was insufficient to generate searchable chunks.",
        )

    # 3. Embedding Generation
    try:
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_chunks(chunk_texts)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings with Gemini API: {str(e)}",
        )

    # 4. Build and Persist FAISS Vector Store
    try:
        doc_info = {
            "filename": filename,
            "total_chunks": len(chunks),
            "total_words": sum(c.get("word_count", 0) for c in chunks),
        }
        vector_store.build(embeddings, chunks, document_info=doc_info)
        vector_store.save_to_disk()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build vector index: {str(e)}",
        )

    return UploadResponse(
        filename=filename,
        chunks_count=len(chunks),
        message="Document successfully processed and indexed into FAISS.",
    )


@app.post("/ask", response_model=AskResponse, summary="Ask a grounded question about the document")
async def ask_question_endpoint(payload: AskRequest):
    """
    Retrieves the top nearest chunks matching the user's question,
    formats a strictly grounded prompt, and synthesizes the answer with Gemini.
    """
    question = payload.question.strip() if payload.question else ""
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty. Please enter a valid question.",
        )

    if not vector_store.is_ready:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document has been uploaded yet. Please upload a PDF first.",
        )

    try:
        result = answer_question(question)
        return AskResponse(
            answer=result["answer"],
            sources=[
                SourceChunk(
                    id=s["id"],
                    text=s["text"],
                    score=s.get("score"),
                )
                for s in result.get("sources", [])
            ],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG pipeline: {str(e)}",
        )
