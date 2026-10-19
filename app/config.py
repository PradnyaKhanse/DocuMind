"""
Configuration module for doc-qa-rag.

Defines model constants, chunking parameters, file persistence paths,
and API credentials loaded from environment variables (.env).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file at the project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ==========================================
# Google Gemini API & Model Configuration
# ==========================================
# GEMINI_API_KEY: Authentication key for Google Gemini API services.
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()

# EMBEDDING_MODEL: Google's current active embedding model for the GenAI SDK.
# "gemini-embedding-001" generates dense 3072-dimensional vector representations.
EMBEDDING_MODEL: str = "gemini-embedding-001"

# GENERATION_MODEL: Fast, active flash model for Google GenAI free tier.
# "gemini-3.1-flash-lite" provides fast, reliable grounded reasoning.
GENERATION_MODEL: str = "gemini-3.1-flash-lite"

# ==========================================
# RAG Chunking & Retrieval Configuration
# ==========================================
# CHUNK_SIZE: Number of words per text chunk.
# WHY 500 words: A 500-word block represents roughly 1-2 complete paragraphs or self-contained
# ideas. Shorter chunks lose context; larger chunks dilute the embedding vector's focus.
CHUNK_SIZE: int = 500

# CHUNK_OVERLAP: Number of shared words between consecutive chunks.
# WHY 50 words: Overlap prevents boundary truncation where a critical sentence or fact
# would otherwise be sliced in half between two distinct chunks.
CHUNK_OVERLAP: int = 50

# TOP_K: Number of nearest neighbor chunks retrieved from FAISS.
# WHY 4 chunks: 4 * 500 words = ~2000 words (~2600 tokens). This delivers sufficient multi-part
# context for complex questions while eliminating unrelated text noise and keeping latency low.
TOP_K: int = 4

# ==========================================
# Data Persistence Paths
# ==========================================
DATA_DIR: Path = BASE_DIR / "data"
FAISS_INDEX_PATH: Path = DATA_DIR / "faiss_index.bin"
CHUNKS_METADATA_PATH: Path = DATA_DIR / "chunks_metadata.json"
DOCUMENT_INFO_PATH: Path = DATA_DIR / "document_info.json"
STATIC_DIR: Path = BASE_DIR / "static"

# Ensure runtime data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
