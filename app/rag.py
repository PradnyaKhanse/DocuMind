"""
RAG (Retrieval-Augmented Generation) Orchestration Module.

Combines semantic retrieval via FAISS with constrained generative reasoning
via Google Gemini to produce fully grounded, hallucination-free document answers.
"""

from typing import Dict, Any, List
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from app.config import GENERATION_MODEL, TOP_K
from app.embeddings import embed_query, get_genai_client
from app.vector_store import vector_store


def build_grounded_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Constructs a strictly constrained prompt containing the retrieved context.

    WHY GROUNDING EXISTS:
    Without strict grounding, Large Language Models default to their pre-trained
    parametric weights, which frequently causes subtle hallucinations, speculation,
    or referencing information outside the provided document.
    By enforcing strict instructions and negative constraints ("If the answer is not in the context,
    say you don't know"), the model becomes a verifiable synthesis engine for the document alone.
    """
    formatted_context = []
    for chunk in context_chunks:
        chunk_id = chunk.get("id", "N/A")
        chunk_text = chunk.get("text", "").strip()
        formatted_context.append(f"[Source Chunk ID: {chunk_id}]\n{chunk_text}")

    context_str = "\n\n".join(formatted_context)

    prompt = f"""You are a precise, truthful Document Q&A assistant.
Answer ONLY using the provided context. If the answer is not in the context, say you don't know. Never answer from outside knowledge.

Context from uploaded document:
----------------------------------------
{context_str}
----------------------------------------

Question: {question.strip()}

Instructions:
1. Answer ONLY using the facts stated in the context above.
2. If the answer is not contained in the context, explicitly respond: "I don't know based on the provided document."
3. Do not assume, extrapolate, or use outside world knowledge.
4. Keep the answer clear, direct, and well-structured.

Answer:"""
    return prompt


def answer_question(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Executes the end-to-end RAG pipeline for a user question.

    Step 1: Embed Query (Convert user question into vector)
    Step 2: Semantic Retrieval (Query FAISS to obtain top_k nearest chunk neighbors)
    Step 3: Grounded Synthesis (Format prompt with retrieved context and invoke Gemini)

    Args:
        question: User's question string.
        top_k: Number of chunks to retrieve for context.

    Returns:
        Dict with "answer": str and "sources": List[Dict[str, Any]]

    Raises:
        ValueError: If question is empty or document is not ready.
        RuntimeError: If Gemini API or vector retrieval fails.
    """
    clean_question = question.strip() if question else ""
    if not clean_question:
        raise ValueError("Question cannot be empty. Please ask a valid question.")

    if not vector_store.is_ready:
        raise ValueError(
            "No document has been uploaded or indexed yet. Please upload a PDF first."
        )

    # -------------------------------------------------------------
    # Step 1: Query Embedding
    # -------------------------------------------------------------
    query_vector = embed_query(clean_question)

    # -------------------------------------------------------------
    # Step 2: Semantic Retrieval (FAISS)
    # -------------------------------------------------------------
    # WHY RETRIEVAL MATTERS:
    # LLMs cannot take unbounded document lengths, and feeding entire 100-page books
    # degrades reasoning quality (needle-in-a-haystack problem).
    # FAISS pinpoints the TOP_K most semantically similar chunks.
    retrieved_chunks = vector_store.search(query_vector, top_k=top_k)

    if not retrieved_chunks:
        return {
            "answer": "I don't know based on the provided document.",
            "sources": [],
        }

    # -------------------------------------------------------------
    # Step 3: Grounded Prompt Building & Generation
    # -------------------------------------------------------------
    prompt = build_grounded_prompt(clean_question, retrieved_chunks)
    client = get_genai_client()
    candidate_models = [GENERATION_MODEL, "gemini-3.5-flash-lite", "gemma-4-26b-a4b-it"]
    answer_text = None
    last_error = None

    for model_name in candidate_models:
        try:
            # Use temperature=0.0 for deterministic, factual, grounded answers
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            if response and response.text:
                answer_text = response.text.strip()
                break
        except Exception as e:
            last_error = e
            continue

    if answer_text is None:
        if last_error:
            raise RuntimeError(f"Gemini Generation API error: {str(last_error)}")
        answer_text = "I don't know based on the provided document."

    # Package response matching exact contract
    sources = [
        {
            "id": chunk.get("id"),
            "text": chunk.get("text"),
            "score": round(chunk.get("score", 0.0), 4),
        }
        for chunk in retrieved_chunks
    ]

    return {
        "answer": answer_text,
        "sources": sources,
    }
