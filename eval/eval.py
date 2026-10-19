"""
Evaluation Script for doc-qa-rag.

Loads test question/expected_keywords pairs from eval/questions.json,
executes each against the RAG pipeline (via HTTP endpoint or direct module invocation),
validates keyword presence case-insensitively, and calculates accuracy and latency metrics.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import httpx

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

QUESTIONS_FILE = ROOT_DIR / "eval" / "questions.json"
DEFAULT_API_URL = "http://127.0.0.1:8000/ask"


def query_via_http(question: str, url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Queries the running FastAPI backend over HTTP."""
    response = httpx.post(url, json={"question": question}, timeout=60.0)
    response.raise_for_status()
    return response.json()


def query_via_module(question: str) -> Dict[str, Any]:
    """Fallback: Directly invokes the RAG pipeline module if server is offline."""
    from app.vector_store import vector_store
    from app.config import FAISS_INDEX_PATH, CHUNKS_METADATA_PATH, DOCUMENT_INFO_PATH
    from app.rag import answer_question

    if not vector_store.is_ready:
        loaded = vector_store.load_from_disk(
            FAISS_INDEX_PATH, CHUNKS_METADATA_PATH, DOCUMENT_INFO_PATH
        )
        if not loaded:
            raise RuntimeError(
                "No index found on disk. Please upload a PDF first via the UI or endpoint."
            )

    return answer_question(question)


def evaluate_question(
    item: Dict[str, Any], query_func
) -> Dict[str, Any]:
    """
    Evaluates a single question item against expected keywords.

    An answer is marked PASS if ALL expected keywords are present in the answer
    string (case-insensitive).
    """
    question = item["question"]
    expected_keywords = [kw.strip().lower() for kw in item.get("expected_keywords", [])]

    start_time = time.perf_counter()
    try:
        result = query_func(question)
        elapsed_sec = time.perf_counter() - start_time
        answer = result.get("answer", "")
        error_msg = None
    except Exception as e:
        elapsed_sec = time.perf_counter() - start_time
        answer = ""
        error_msg = str(e)

    answer_lower = answer.lower()
    matched_keywords = [kw for kw in expected_keywords if kw in answer_lower]
    missing_keywords = [kw for kw in expected_keywords if kw not in answer_lower]

    is_pass = len(missing_keywords) == 0 and error_msg is None

    return {
        "id": item.get("id"),
        "question": question,
        "expected_keywords": expected_keywords,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "is_pass": is_pass,
        "elapsed_sec": elapsed_sec,
        "answer": answer,
        "error": error_msg,
    }


def run_evaluation():
    """Runs the full evaluation benchmark across all configured questions."""
    print("=" * 70)
    print("  Document Q&A RAG Evaluation Benchmark (doc-qa-rag)")
    print("=" * 70)

    if not QUESTIONS_FILE.exists():
        print(f"[Error] Questions file not found at: {QUESTIONS_FILE}")
        sys.exit(1)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    total_questions = len(questions_data)
    if total_questions == 0:
        print("[Error] No questions found in questions.json.")
        sys.exit(1)

    # Determine query mode (HTTP endpoint vs direct module)
    use_http = False
    try:
        check = httpx.get("http://127.0.0.1:8000/status", timeout=2.0)
        if check.status_code == 200 and check.json().get("is_ready"):
            use_http = True
            print("[Mode] Connected to running FastAPI server at http://127.0.0.1:8000")
        else:
            print("[Mode] FastAPI server responding but no document indexed yet.")
    except Exception:
        print("[Mode] FastAPI server not running on port 8000; executing via internal RAG module.")

    query_fn = query_via_http if use_http else query_via_module

    results: List[Dict[str, Any]] = []
    print(f"\nRunning evaluation on {total_questions} questions...\n" + "-" * 70)

    for i, item in enumerate(questions_data, start=1):
        q_text = item.get("question", "")
        print(f"[{i}/{total_questions}] Evaluating: \"{q_text}\"")

        res = evaluate_question(item, query_fn)
        results.append(res)

        status_label = "PASS" if res["is_pass"] else "FAIL"
        time_str = f"{res['elapsed_sec']:.2f}s"

        if res["is_pass"]:
            print(f"      Result: [{status_label}] ({time_str})")
            print(f"      Matched keywords: {res['matched_keywords']}")
        else:
            print(f"      Result: [{status_label}] ({time_str})")
            if res["error"]:
                print(f"      Error: {res['error']}")
            else:
                print(f"      Missing keywords: {res['missing_keywords']}")
                print(f"      Answer excerpt: \"{res['answer'][:120]}...\"")
        print()

    # Calculate metrics
    passed_count = sum(1 for r in results if r["is_pass"])
    accuracy_pct = (passed_count / total_questions) * 100.0
    avg_latency = sum(r["elapsed_sec"] for r in results) / total_questions

    print("=" * 70)
    print("                     EVALUATION SUMMARY")
    print("=" * 70)
    print(f" Total Questions Tested : {total_questions}")
    print(f" Passed                 : {passed_count}")
    print(f" Failed                 : {total_questions - passed_count}")
    print(f" Accuracy               : {passed_count}/{total_questions} ({accuracy_pct:.1f}%)")
    print(f" Average Response Time  : {avg_latency:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
