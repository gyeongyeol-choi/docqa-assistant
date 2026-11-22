"""FastAPI server and web UI for DocQA Assistant."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from docqa.service import DocQAService

STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=20)


class IndexRequest(BaseModel):
    docs_dir: str | None = Field(default=None, description="Directory path to index")


def create_app(service: DocQAService | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="DocQA Assistant",
        description="Personal & technical document Q&A assistant with source citations.",
        version="0.1.0",
    )
    svc = service or DocQAService()

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_INDEX)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "provider": svc.llm.provider_name,
            "indexed_chunks": len(svc.retriever.chunks),
        }

    @app.get("/api/docs")
    def list_docs() -> dict:
        return {"documents": svc.list_sources()}

    @app.post("/api/index")
    def index_docs(req: IndexRequest) -> dict:
        target = Path(req.docs_dir) if req.docs_dir else None
        try:
            return svc.index_directory(target)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/ask")
    def ask_question(req: AskRequest) -> dict:
        result = svc.ask(req.question, top_k=req.top_k)
        return result.to_dict()

    return app


app = create_app()
