# DocuMind: Document Q&A RAG Assistant (`doc-qa-rag`)

> **What is RAG in 3 Lines?**  
> Retrieval-Augmented Generation (RAG) is an AI architecture that equips Large Language Models with external reference knowledge.  
> Instead of relying on static training weights, relevant excerpts are retrieved dynamically from private documents using vector similarity search.  
> The model then synthesizes a strictly grounded, factual answer based exclusively on those verified excerpts, eliminating hallucinations.

---

## 🏗️ Architecture: The 5 Core Steps

```
[ Upload PDF ] ────► [ 1. Ingest & Extract ] ────► [ 2. Word Chunking + Overlap ]
                                                                 │
                                                                 ▼
[ User Question ] ──► [ 4. Retrieval (FAISS) ] ◄── [ 3. Batch Embeddings (Gemini) ]
        │                       │
        ▼                       ▼
 [ 5. Grounded Prompt ] ──► [ Gemini Flash ] ──► [ Verified Answer + Sources ]
```

### 1. Ingestion & Text Extraction (`app/ingest.py`)
- **What it does**: Reads binary PDF streams using `pypdf`, extracts plain text across all pages, and validates that the document is neither empty nor an image-only scanned PDF lacking OCR text.
- **Why it matters**: Clean text normalization is the foundation of high-quality downstream retrieval. Catching unreadable or scanned documents early prevents silent failures.

### 2. Overlapping Chunking (`app/ingest.py`)
- **What it does**: Slices the text into `CHUNK_SIZE = 500` word segments with a sliding window `CHUNK_OVERLAP = 50` words.
- **Why chunking exists**: A single 768-dimensional embedding vector cannot represent the nuance of an entire 50-page document without severe information loss. Chunking segments content into self-contained semantic passages.
- **Why overlap exists**: Splitting text blindly risks severing a sentence, definition, or fact across boundary borders. A 50-word overlap guarantees that thoughts spanning a cut point are preserved intact in at least one adjacent chunk.

### 3. Dense Vector Embeddings (`app/embeddings.py`)
- **What it does**: Converts chunk text into dense 768-dimensional numerical vectors using Google's official `google-genai` SDK and the `text-embedding-004` model. Embeddings are L2-normalized.
- **Why embeddings exist**: Lexical/keyword search fails when queries use synonyms (e.g. searching "revenue" when the PDF says "gross earnings"). Dense embeddings map conceptual meaning into geometric vector space where semantic neighbors cluster together. Normalization ensures cosine similarity equals inner product.

### 4. Vector Storage & Nearest-Neighbor Search (`app/vector_store.py`)
- **What it does**: Indexes chunk vectors into a FAISS (`faiss-cpu`) `IndexFlatIP` structure and saves both the binary index and chunk metadata to disk.
- **Why FAISS exists**: Comparing a question vector against thousands of document passages linearly in pure Python is $O(N \times d)$ and slow. FAISS utilizes C++ hardware acceleration (SIMD/BLAS) to compute exact cosine similarity searches in under 2 milliseconds.

### 5. Grounded Prompt Synthesis (`app/rag.py`)
- **What it does**: Retrieves the `TOP_K = 4` nearest chunks, builds an explicit grounding prompt, and queries Google Gemini (`gemini-2.5-flash` at temperature `0.0`).
- **Why grounding exists**: Large Language Models are prone to reciting stale parametric memory or hallucinating plausible fiction. Strict negative constraints ("*Answer ONLY using the provided context. If the answer is not in the context, say you don't know.*") turn the LLM into a reliable, verifiable reasoning engine over your document alone.

---

## 🚀 Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2
- **Vector Search**: FAISS (`faiss-cpu`), NumPy
- **Document Processing**: `pypdf`
- **AI Models (Google GenAI SDK)**:
  - Embeddings: `text-embedding-004`
  - Generation: `gemini-2.5-flash` (or `gemini-1.5-flash`)
- **Configuration**: `python-dotenv`
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (no external JS frameworks)

---

## 📁 Project Structure

```
doc-qa-rag/
├── app/
│   ├── __init__.py
│   ├── config.py           # Model constants, chunk size/overlap, persistence paths
│   ├── ingest.py           # PDF extraction, validation, word chunking with overlap
│   ├── embeddings.py       # Google GenAI client, batch embeddings, normalization
│   ├── vector_store.py     # FAISS IndexFlatIP, cosine search, disk persistence
│   ├── rag.py              # Prompt construction, top-k retrieval, grounded answer
│   └── main.py             # FastAPI REST endpoints (/upload, /ask, /status, /)
├── static/
│   └── index.html          # Polished, modern lavender SaaS UI (drag & drop, sources)
├── eval/
│   ├── eval.py             # Benchmark script testing accuracy & latency
│   └── questions.json      # 10 question and expected keyword pairs
├── data/                   # Auto-created directory for FAISS index and chunk metadata
├── requirements.txt        # Pinned dependencies
├── .env.example            # Environment variable template
├── .env                    # Local secrets (GEMINI_API_KEY)
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone & Navigate to Project Directory
```bash
cd doc-qa-rag
```

### 2. Create and Activate a Virtual Environment
```bash
# Using Python 3.11+
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your Gemini API Key
Create a `.env` file in the project root (or copy from `.env.example`):
```bash
cp .env.example .env
```
Edit `.env` and paste your key:
```ini
GEMINI_API_KEY=AIzaSy...your_actual_gemini_api_key...
```
*(Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/)).*

---

## 🏃 Running the Application

Start the FastAPI application with Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 📡 API Endpoints & Example `curl` Commands

### 1. Check Document Readiness
```bash
curl -X GET http://127.0.0.1:8000/status
```

### 2. Upload and Index a PDF
```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@/path/to/your/document.pdf"
```

**Example Response**:
```json
{
  "filename": "document.pdf",
  "chunks_count": 18,
  "message": "Document successfully processed and indexed into FAISS."
}
```

### 3. Ask a Grounded Question
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the primary conclusion of this document?"}'
```

**Example Response**:
```json
{
  "answer": "The primary conclusion is that implementing retrieval-augmented generation reduced hallucination rates by 84% compared to vanilla zero-shot prompting.",
  "sources": [
    {
      "id": 14,
      "text": "In our experimental evaluation, implementing retrieval-augmented generation reduced hallucination rates by 84% compared to vanilla zero-shot prompting across all benchmark tasks.",
      "score": 0.8842
    },
    {
      "id": 15,
      "text": "Furthermore, factual attribution scores reached 96.2% when restricting generation to top-4 retrieved passages.",
      "score": 0.8120
    }
  ]
}
```

---

## 📊 Evaluation Benchmark

The evaluation harness in `eval/eval.py` loads `eval/questions.json`, executes each question through the RAG pipeline, and verifies that all expected key facts appear in the response case-insensitively.

### Running the Evaluation
With the server running (or using direct local module execution):
```bash
python eval/eval.py
```

### Evaluation Results Table (Placeholder Template)

| # | Question | Expected Keywords | Status | Response Time |
|---|---|---|:---:|:---:|
| 1 | What is the primary topic or purpose? | `overview`, `purpose` | -- | -- |
| 2 | Who is the author or organization? | `author` | -- | -- |
| 3 | What key methodology is outlined? | `method`, `process` | -- | -- |
| 4 | What are the primary objectives? | `objective`, `goal` | -- | -- |
| 5 | What key findings are reported? | `result`, `conclusion` | -- | -- |
| 6 | What technologies are utilized? | `tool`, `technology` | -- | -- |
| 7 | What challenges or limitations are noted?| `challenge`, `limitation` | -- | -- |
| 8 | What future steps are proposed? | `recommendation`, `future` | -- | -- |
| 9 | What specific metrics are discussed? | `metric`, `evaluation` | -- | -- |
| 10 | What is the summary of the conclusion?| `summary`, `conclusion` | -- | -- |
| **Total** | **Overall Accuracy** | | **-- / 10 (--%)** | **Avg: -- s** |

*Note: Update `eval/questions.json` with questions and expected keywords tailored to your specific PDF document before running.*
