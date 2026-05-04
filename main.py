"""
main.py — FastAPI backend for the RAG Codebase Q&A system.

"""


import json
import logging
import time
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ingestion.chunkers.fixed_size as fixed_chunker
import ingestion.chunkers.recursive as recursive_chunker
import ingestion.chunkers.semantic as semantic_chunker
from ingestion.embedder import check_if_indexed, embed_and_store, get_collection_name
from ingestion.file_parser import parse_documents
from ingestion.repo_loader import RepoLoader
import ingestion.chunkers.ast_based as ast_chunker

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Codebase Q&A API",
    description="Ask natural language questions about any indexed GitHub repository.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    repo_url: str
    chunk_strategy: Literal["fixed", "recursive", "ast", "semantic"] = "ast"
    persist_dir: str = "./chroma_db"
    bm25_dir: str = "./bm25_indexes"


class IndexResponse(BaseModel):
    status: str
    repo_name: str
    num_chunks: int
    time_taken_s: float
    estimated_cost_usd: float
    collection_name: str


class QueryRequest(BaseModel):
    question: str
    repo_name: str
    chunk_strategy: Literal["fixed", "recursive", "ast", "semantic"] = "ast"
    top_k: int = 5
    alpha: float = 0.7
    model: str = "gpt-4o-mini"
    persist_dir: str = "./chroma_db"
    bm25_dir: str = "./bm25_indexes"


class SourceInfo(BaseModel):
    file_path: str
    function_name: str
    start_line: str | None = None
    end_line: str | None = None
    language: str
    chunk_preview: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    tokens_used: int
    latency_ms: int
    model: str


def _extract_repo_name(repo_url: str) -> str:
    """Derive a ``owner_repo`` identifier from a GitHub URL."""
    parts = repo_url.rstrip("/").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1]


@app.get("/health", tags=["ops"])
def health_check():
    """Liveness probe — returns ``{"status": "ok"}``."""
    logger.debug("Health check called")
    return {"status": "ok", "version": "1.0.0"}


@app.post("/index", response_model=IndexResponse, tags=["ingestion"])
def index_repository(req: IndexRequest):
    """
    Clone a GitHub repository, chunk it with the chosen strategy, embed chunks,
    """

    CHUNKERS = {
        "fixed": fixed_chunker,
        "recursive": recursive_chunker,
        "ast": ast_chunker,
        "semantic": semantic_chunker,
    }

    logger.info(
        "Index request received: repo_url=%s strategy=%s",
        req.repo_url,
        req.chunk_strategy,
    )

    repo_name = _extract_repo_name(req.repo_url)

    if check_if_indexed(req.repo_url, req.chunk_strategy):
        collection_name = get_collection_name(repo_name, req.chunk_strategy)
        logger.info(
            "Repository already indexed — returning early: collection=%s",
            collection_name,
        )
        return IndexResponse(
            status="already_indexed",
            repo_name=repo_name,
            num_chunks=0,
            time_taken_s=0.0,
            estimated_cost_usd=0.0,
            collection_name=collection_name,
        )

    start = time.perf_counter()

    try:
        loader = RepoLoader()
        documents = loader.load(req.repo_url)
        logger.info("Repo loaded: files=%d", len(documents))

        documents = parse_documents(documents)

        chunker = CHUNKERS[req.chunk_strategy]
        chunks = chunker.chunk_documents(documents)
        logger.info(
            "Chunks created: count=%d strategy=%s", len(chunks), req.chunk_strategy
        )

        result = embed_and_store(
            documents=chunks,
            repo_name=repo_name,
            chunk_strategy=req.chunk_strategy,
            repo_url=req.repo_url,
            persist_dir=req.persist_dir,
            bm25_dir=req.bm25_dir,
        )

        elapsed = time.perf_counter() - start
        logger.info(
            "Indexing complete: repo=%s strategy=%s chunks=%d elapsed_s=%.2f",
            repo_name,
            req.chunk_strategy,
            result["num_chunks"],
            elapsed,
        )

        return IndexResponse(
            status="indexed",
            repo_name=repo_name,
            num_chunks=result["num_chunks"],
            time_taken_s=round(elapsed, 2),
            estimated_cost_usd=result["estimated_cost_usd"],
            collection_name=result["collection_name"],
        )

    except Exception as exc:
        logger.error(
            "Indexing failed: repo_url=%s error=%s", req.repo_url, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(exc))