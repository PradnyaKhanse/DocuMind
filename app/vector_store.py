"""
Vector Store Module using FAISS (faiss-cpu) and NumPy.

Provides high-performance nearest-neighbor indexing, similarity search,
and disk persistence for document chunks.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from app.config import (
    TOP_K,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    DOCUMENT_INFO_PATH,
)


class FAISSVectorStore:
    """
    Manages a FAISS index and associated chunk metadata.

    WHY FAISS & VECTOR STORE EXIST:
    Once documents are converted into dense mathematical vectors (e.g. 768 dimensions),
    we must find which document passages are closest to the user's question vector.
    Scanning through vectors linearly in Python is $O(N \\times d)$ and slow.
    FAISS provides C++ optimized, hardware-accelerated matrix multiplication (BLAS/AVX)
    to compute Cosine Similarities across thousands of chunks in mere milliseconds.
    """

    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []
        self.document_info: Dict[str, Any] = {}

    @property
    def is_ready(self) -> bool:
        """Returns True if an index is built, loaded, and contains vectors."""
        return self.index is not None and self.index.ntotal > 0 and len(self.chunks) > 0

    def build(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
        document_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Constructs a new FAISS index from normalized embedding vectors.

        Uses IndexFlatIP (Inner Product). Because embeddings are L2-normalized,
        Inner Product is identical to Cosine Similarity (values between -1 and 1).

        Args:
            embeddings: 2D numpy array of shape (N, dimension) of float32.
            chunks: List of chunk metadata dicts corresponding to each vector row.
            document_info: Optional metadata about the indexed PDF document.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch between number of embeddings ({len(embeddings)}) and chunks ({len(chunks)})."
            )

        if len(embeddings) == 0:
            raise ValueError("Cannot build FAISS index with zero embeddings.")

        dimension = embeddings.shape[1]

        # faiss.IndexFlatIP computes exact inner-product (cosine similarity for normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        self.index = index
        self.chunks = chunks
        self.document_info = document_info or {}

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Performs nearest-neighbor search to retrieve the most relevant chunks.

        Args:
            query_embedding: Normalized query vector of shape (1, dimension).
            top_k: Number of highest-similarity chunks to return.

        Returns:
            List of chunk dicts including chunk 'id', 'text', and 'similarity_score'.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Vector store is empty. Please upload and index a PDF document first."
            )

        # Ensure top_k does not exceed total indexed chunks
        effective_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, effective_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                chunk_data = self.chunks[idx].copy()
                chunk_data["score"] = float(score)
                results.append(chunk_data)

        return results

    def save_to_disk(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        chunks_path: Path = CHUNKS_METADATA_PATH,
        doc_info_path: Path = DOCUMENT_INFO_PATH,
    ) -> None:
        """Persists the FAISS index and chunk metadata to disk."""
        if not self.is_ready:
            raise RuntimeError("Cannot save an uninitialized vector store.")

        # Ensure directory exists
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        with open(doc_info_path, "w", encoding="utf-8") as f:
            json.dump(self.document_info, f, ensure_ascii=False, indent=2)

    def load_from_disk(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        chunks_path: Path = CHUNKS_METADATA_PATH,
        doc_info_path: Path = DOCUMENT_INFO_PATH,
    ) -> bool:
        """
        Loads the persisted FAISS index and metadata if available on disk.

        Returns:
            True if successfully loaded, False otherwise.
        """
        if not (index_path.exists() and chunks_path.exists()):
            return False

        try:
            self.index = faiss.read_index(str(index_path))
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

            if doc_info_path.exists():
                with open(doc_info_path, "r", encoding="utf-8") as f:
                    self.document_info = json.load(f)
            else:
                self.document_info = {}

            return True
        except Exception:
            self.index = None
            self.chunks = []
            self.document_info = {}
            return False


# Global singleton instance for the FastAPI application
vector_store = FAISSVectorStore()
