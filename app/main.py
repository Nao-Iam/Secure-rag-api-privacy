"""FastAPI app: GET /health, GET /sources, POST /ask, POST /evaluate.

Run:
    uvicorn app.main:app --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs

Environment:
    RAG_EMBEDDER=auto|tfidf|dense   (default auto: dense, TF-IDF fallback)
    RAG_USE_LLM=1 + OPENAI_API_KEY  (optional generative rewrite, never required)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .answer import RAGService
from .evaluate import render_markdown, run_evaluation
from .ingest import list_sources

BASE_DIR = Path(__file__).resolve().parent.parent


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about Canadian privacy / AI governance")
    include_untrusted: bool = Field(
        False,
        description="Also load the synthetic tier-2 test passages (flagged, never cited). Default false.",
    )
    top_k: int = Field(3, ge=1, le=5)


class Citation(BaseModel):
    doc_id: str
    title: str
    source: str
    trust_tier: int


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    flags: list[str]
    refused: bool
    embedder: str
    retrieved_trusted: int
    retrieved_untrusted: int


def create_app(embedder: str | None = None) -> FastAPI:
    preference = embedder or os.environ.get("RAG_EMBEDDER", "auto")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = RAGService(BASE_DIR, embedder_preference=preference)
        yield

    app = FastAPI(
        title="secure-rag-api-privacy",
        description=(
            "Small RAG API on Canadian privacy & AI governance (PIPEDA, OPC "
            "generative-AI principles, Treasury Board AI guidance) with "
            "trust-tier defences against indirect prompt injection and "
            "knowledge poisoning. Learning project for the CIS*6560 "
            "application, University of Guelph."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        service: RAGService = app.state.service
        return {
            "status": "ok",
            "embedder": service.embedder_name,
            "embedder_fallback": service.embedder_fallback,
            "trusted_chunks": len(service.trusted_chunks),
            "untrusted_chunks": len(service.untrusted_chunks),
        }

    @app.get("/sources")
    def sources() -> dict:
        return {"sources": list_sources(BASE_DIR / "data" / "corpus")}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        service: RAGService = app.state.service
        return AskResponse(
            **service.ask(req.question, include_untrusted=req.include_untrusted, top_k=req.top_k)
        )

    @app.post("/evaluate")
    def evaluate() -> dict:
        service: RAGService = app.state.service
        return run_evaluation(service)

    @app.get("/evaluation.md", include_in_schema=False)
    def evaluation_markdown() -> str:
        service: RAGService = app.state.service
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(render_markdown(run_evaluation(service)))

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
