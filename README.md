## AI PDF Analyzer (RAG)

RAG-based PDF analysis system built with FastAPI, Inngest, Qdrant, and Ollama for local embeddings and text generation.

---

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for analyzing PDF documents:

- **PDF ingestion**: PDFs are split into text chunks, converted into embeddings, and stored in a vector database (Qdrant).
- **Querying PDFs**: User questions are embedded, relevant context is retrieved from Qdrant, and a final answer is generated using Ollama.

The architecture ensures **local-first execution** without sending data to external services.

---

## Architecture

The main orchestration is defined in `main.py` with two Inngest functions:

### 1. PDF Ingestion (`rag_ingest_pdf`)

- **Trigger**: `"rag/ingest_pdf"`
- **Steps**:
  - Load PDF (`load_and_chunk_pdf` in `data_loader.py`) and split into overlapping text chunks.
  - Generate embeddings using `OllamaAdapter` (`ollama.py`).
  - Insert vectors and metadata into Qdrant using `QdrantStorage` (`vector_db.py`).

### 2. PDF Query (`rag_query_pdf_ai`)

- **Trigger**: `"rag/query_pdf_ai"`
- **Steps**:
  - Embed the user question using `OllamaAdapter`.
  - Retrieve the `top_k` most similar chunks from Qdrant.
  - Build a prompt with retrieved context.
  - Generate an answer using Ollama’s text model.

FastAPI integrates with Inngest via:

```python
inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])
```

---

## Prerequisites

- **OS**: macOS, Linux, or Windows (WSL recommended)
- **Python**: ≥ 3.10
- **Ollama**:
  - Running locally at `http://localhost:11434`
  - Models required:
    - Text: `llama3.2`
    - Embeddings: `nomic-embed-text`
- **Qdrant**:
  - Running locally at `http://localhost:6333`
  - Docker recommended
- **Inngest CLI**: for local event orchestration

---

## Installation

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 2. Dependencies

```bash
pip install fastapi uvicorn[standard] httpx python-dotenv
pip install qdrant-client
pip install "llama-index-core" "llama-index-readers-file"
pip install inngest
```

Optionally, install from `requirements.txt` :

```bash
pip install -r r.txt
```

---

## Ollama Setup

1. Install Ollama: `https://ollama.com`
2. Start Ollama service:

```bash
ollama serve
```

3. Pull required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

The `OllamaAdapter` connects to:

- `base_url="http://localhost:11434"`
- `model="llama3.2"`
- `embed_model="nomic-embed-text"`

---

## Qdrant Setup

### Docker Example

```bash
docker run -d --name qdrantRagDb -p 6333:6333 \
  -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Collection Configuration

- Collection: `"docs"`
- Vector dimension: `768` (must match embedding model)

If switching embedding models, reset the collection:

```bash
python reset_qdrant.py
```

---

## Environment Variables

Use `.env` to configure service URLs:

```bash
QDRANT_URL=http://localhost:6333
OLLAMA_BASE_URL=http://localhost:11434
```

Load variables in code using `python-dotenv` (if you want):

```python
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")
```

---

## Running the API

```bash
uvicorn main:app --reload
```

- OpenAPI docs: `http://127.0.0.1:8000/docs`
- FastAPI endpoints trigger Inngest functions via `inngest.fast_api.serve`.

---

## Inngest Orchestration

**Functions:**

- `rag_ingest_pdf` (`"rag/ingest_pdf"`)
- `rag_query_pdf_ai` (`"rag/query_pdf_ai"`)

**Typical workflow:**

1. Start FastAPI and the Inngest dev server.
2. Trigger events using the Inngest UI, CLI, or HTTP requests.
3. Monitor logs and retries in the dev server.

To run the Inngest dev server locally:

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

---

## Ingesting PDFs

Example payload:

```json
{
  "pdf_path": "pdf/dracula.pdf",
  "source_id": "dracula_pdf" (optional)
}
```

- The PDF is split into chunks, embedded, and upserted into Qdrant.

Example response:

```json
{
  "ingested": 42
}
```

---

## Querying PDFs

Example payload:

```json
{
  "question": "Who is Dracula and where does he live?",
  "top_k": 5
}
```

Returns top-k relevant context and a generated answer:

```json
{
  "answer": "Dracula is a vampire who resides in Transylvania...",
  "sources": ["dracula_pdf"],
  "num_contexts": 5
}
```

---

## Project Structure

```text
.
├── main.py            # Inngest orchestration + FastAPI app
├── data_loader.py     # PDF loading and chunking
├── vector_db.py       # QdrantStorage wrapper
├── ollama.py          # OllamaAdapter for embeddings & generation
├── custom_types.py    # Pydantic models for RAG objects
├── reset_qdrant.py    # Utility to delete Qdrant collection
├── pdf/               # Sample PDF files
└── .env               # Environment configuration
```

---

## Notes

- Ensure embedding dimension matches Qdrant configuration.
- Paths to PDFs must be valid.
- Designed for local execution with sensitive data remaining on your machine.
- Inngest supports scalable workflows with retries, queues, and cron jobs.

#Views

## Server Side (dev)
<img width="1346" height="363" alt="Screenshot 2026-02-23 at 21 20 06" src="https://github.com/user-attachments/assets/7e1223e9-f411-487a-87ae-613fc76b2f38" />

## FrontEnd (simple design due to is not the focus)
- Uploading and ingesting a pdf
<img width="1136" height="605" alt="Screenshot 2026-02-23 at 21 25 32" src="https://github.com/user-attachments/assets/d3ee4b43-8d11-422d-ba30-1d65ab2a5d2a" />

- AI Questions
<img width="1089" height="722" alt="Screenshot 2026-02-23 at 21 13 04" src="https://github.com/user-attachments/assets/479e27ed-e2d9-4450-b5a8-c00ca01aa071" />
