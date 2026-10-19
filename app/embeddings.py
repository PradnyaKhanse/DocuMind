"""
Embeddings Generation Module using Google GenAI SDK.

Handles converting document chunks and user queries into dense vector
representations using Google's text-embedding-004 model.
"""

from typing import List, Optional
import numpy as np
from google import genai
from google.genai import errors as genai_errors
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL


def get_genai_client() -> genai.Client:
    """
    Initializes and returns an authenticated Google GenAI client instance.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not configured in .env.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Please provide your Google Gemini API key in the .env file."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizes a 2D array of vectors to unit L2 norm.

    WHY NORMALIZATION MATTERS:
    When vectors are unit-normalized (length = 1.0), their dot product (Inner Product)
    is mathematically identical to Cosine Similarity. This ensures that longer text chunks
    do not dominate retrieval simply due to larger vector magnitudes.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1e-12, norms)
    return (vectors / norms).astype(np.float32)


def embed_chunks(chunk_texts: List[str], batch_size: int = 50) -> np.ndarray:
    """
    Batch-embeds a list of document chunk strings into a 2D float32 numpy array.

    WHY EMBEDDINGS EXIST:
    Traditional keyword search fails when users ask questions using different vocabulary
    than the document (e.g., searching for "revenue" when the PDF says "sales receipts").
    Embeddings map text into a continuous 768-dimensional semantic space where words
    and sentences with similar meanings reside near one another, enabling true
    conceptual/semantic search.

    Args:
        chunk_texts: List of text passages to embed.
        batch_size: Maximum chunks to send per API request (avoids payload size limits).

    Returns:
        2D numpy array of shape (len(chunk_texts), embedding_dimension), normalized to unit length.

    Raises:
        RuntimeError: On API errors or invalid credentials.
    """
    if not chunk_texts:
        return np.empty((0, 3072), dtype=np.float32)

    client = get_genai_client()
    all_embeddings: List[List[float]] = []

    try:
        # Process in batches to respect Google GenAI payload limits
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                )
            except Exception:
                # Fallback to gemini-embedding-2 if primary has an issue
                response = client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=batch,
                )

            if not response.embeddings:
                raise RuntimeError("Gemini embedding model returned empty embeddings list.")

            for emb in response.embeddings:
                all_embeddings.append(emb.values)

    except genai_errors.APIError as e:
        raise RuntimeError(f"Gemini Embedding API error: {e.message}") from e
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e

    embedding_matrix = np.array(all_embeddings, dtype=np.float32)
    return normalize_vectors(embedding_matrix)


def embed_query(query: str) -> np.ndarray:
    """
    Generates a normalized embedding for a single user question.

    Args:
        query: User's question string.

    Returns:
        2D numpy array of shape (1, embedding_dimension).
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty.")

    client = get_genai_client()
    try:
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=query.strip(),
            )
        except Exception:
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=query.strip(),
            )

        if not response.embeddings:
            raise RuntimeError("Gemini embedding model returned empty embedding for query.")

        embedding_vector = response.embeddings[0].values
        vector_2d = np.array([embedding_vector], dtype=np.float32)
        return normalize_vectors(vector_2d)

    except genai_errors.APIError as e:
        raise RuntimeError(f"Gemini Embedding API error: {e.message}") from e
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Failed to generate query embedding: {str(e)}") from e
