"""
RAG Application using Inngest + FastAPI + Qdrant + Ollama

This module defines:
- A PDF ingestion pipeline (chunk → embed → upsert)
- A query pipeline (embed → search → generate answer)
- FastAPI server integration for Inngest orchestration

Architecture:
Event-driven orchestration using Inngest.
Vector storage handled by Qdrant.
Embeddings and generation handled by Ollama.
"""

from fastapi import FastAPI
from dotenv import load_dotenv
import logging
import uuid
import os
import inngest
import inngest.fast_api

from data_loader import load_and_chunk_pdf
from vector_db import QdrantStorage
from custom_types import (
    RAGUpsertResult,
    RAGSearchResult,
    RAGChunkAndSrc,
    RAGQueryResult,
)
from ollama import OllamaAdapter


# -------------------------------------------------------------------
# Environment Configuration
# -------------------------------------------------------------------

load_dotenv()

logger = logging.getLogger("uvicorn")


# -------------------------------------------------------------------
# Inngest Client Initialization
# -------------------------------------------------------------------

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# -------------------------------------------------------------------
# RAG: PDF INGESTION FUNCTION
# -------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context) -> dict:
    """
    Ingest a PDF file into the vector database.

    Pipeline:
    1. Load and chunk PDF
    2. Generate embeddings
    3. Upsert into Qdrant

    Expected Event Data:
    {
        "pdf_path": str,
        "source_id": Optional[str]
    }

    Returns:
        dict: JSON-serializable ingestion result
    """

    def load_pdf_step() -> RAGChunkAndSrc:
        """
        Load PDF from disk and split into chunks.
        """
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)

        chunks = load_and_chunk_pdf(pdf_path)

        return RAGChunkAndSrc(
            chunks=chunks,
            source_id=source_id
        )

    async def embed_and_upsert_step(
        chunk_data: RAGChunkAndSrc,
    ) -> RAGUpsertResult:
        """
        Generate embeddings for chunks and store them in Qdrant.
        """
        adapter = OllamaAdapter()
        store = QdrantStorage()

        embeddings = await adapter.embed(chunk_data.chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk_data.source_id}:{i}"))
            for i in range(len(chunk_data.chunks))
        ]

        payloads = [
            {
                "source": chunk_data.source_id,
                "text": chunk_data.chunks[i],
            }
            for i in range(len(chunk_data.chunks))
        ]

        store.upsert(ids=ids, vectors=embeddings, payloads=payloads)

        return RAGUpsertResult(ingested=len(chunk_data.chunks))

    # Step 1: Load + Chunk
    chunk_data = await ctx.step.run(
        "load-and-chunk",
        load_pdf_step,
        output_type=RAGChunkAndSrc,
    )

    # Step 2: Embed + Upsert
    ingestion_result = await ctx.step.run(
        "embed-and-upsert",
        lambda: embed_and_upsert_step(chunk_data),
        output_type=RAGUpsertResult,
    )

    return ingestion_result.model_dump()


# -------------------------------------------------------------------
# RAG: QUERY FUNCTION
# -------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> dict:
    """
    Query the RAG system using semantic search.

    Pipeline:
    1. Embed user question
    2. Search similar chunks in Qdrant
    3. Build prompt with retrieved context
    4. Generate final answer

    Expected Event Data:
    {
        "question": str,
        "top_k": Optional[int]
    }

    Returns:
        dict: {
            "answer": str,
            "sources": List[str],
            "num_contexts": int
        }
    """

    async def embed_and_search_step(
        question: str,
        top_k: int,
    ) -> RAGSearchResult:
        """
        Embed the user question and search the vector database.
        """
        adapter = OllamaAdapter()
        store = QdrantStorage()

        query_vector = (await adapter.embed([question]))[0]

        search_results = store.search(query_vector, top_k=top_k)

        return RAGSearchResult(
            context=search_results["contexts"],
            sources=search_results["sources"],
        )

    question: str = ctx.event.data["question"]
    top_k: int = int(ctx.event.data.get("top_k", 5))

    # Step 1: Embed + Search
    search_result = await ctx.step.run(
        "embed-and-search",
        lambda: embed_and_search_step(question, top_k),
        output_type=RAGSearchResult,
    )

    # Build context block for prompt
    context_block = "\n\n".join(
        f"- {chunk}" for chunk in search_result.context
    )

    # Prompt Engineering (minimal but structured)
    prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{question}\n\n"
        "Answer concisely and only using the provided context."
    )

    # Step 2: Generate Answer
    adapter = OllamaAdapter()
    answer = await adapter.generate(prompt)

    if isinstance(answer, dict):
        answer = answer.get("response", "")

    result = RAGQueryResult(
        answer=answer,
        sources=search_result.sources,
        num_contexts=len(search_result.context),
    )

    return result.model_dump()


# -------------------------------------------------------------------
# FastAPI Application
# -------------------------------------------------------------------

app = FastAPI(
    title="RAG Orchestration API",
    description="Event-driven RAG system using Inngest",
    version="1.0.0",
)

inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf_ai],
)
