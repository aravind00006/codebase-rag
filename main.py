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
from generation.answer_generator import generate_answer
from generation.prompt_builder import build_prompt
from ingestion.embedder import get_collection_name
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker


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
    """
    A query must identify the target repo via *either* ``repo_url`` (full
    GitHub URL) *or* ``repo_name`` (the ``owner_repo`` slug). If both are
    given, ``repo_url`` wins and the slug is recomputed from it.
    """
    question: str
    repo_url: str | None = None
    repo_name: str | None = None
    chunk_strategy: Literal["fixed", "recursive", "ast", "semantic"] = "ast"
    top_k: int = 5
    alpha: float = 0.7
    model: str = "gpt-4o-mini"
    persist_dir: str = "./chroma_db"
    bm25_dir: str = "./bm25_indexes"


class SourceInfo(BaseModel):
    file_path: str
    function_name: str
    start_line: str |  int | None = None
    end_line: str |  int | None = None
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
    """Derive a ``owner_repo`` identifier from a GitHub URL.

    Thin wrapper around :py:meth:`RepoLoader._extract_repo_name` so the URL
    parsing logic lives in exactly one place.
    """
    return RepoLoader._extract_repo_name(repo_url)


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
    

@app.post("/query", response_model=QueryResponse, tags=["query"])
def query_repository(req: QueryRequest):
    """
    Answer a natural language question about an indexed repository.

    """

    # Resolve the repo identifier from whichever field the client provided.
    # repo_url wins if both are given, since it is the canonical source.
    if req.repo_url:
        repo_name = _extract_repo_name(req.repo_url)
    elif req.repo_name:
        repo_name = req.repo_name
    else:
        raise HTTPException(
            status_code=422,
            detail="Either 'repo_url' or 'repo_name' must be provided.",
        )

    collection_name = get_collection_name(repo_name, req.chunk_strategy)

    logger.info(
        "Query received: repo=%s strategy=%s question_preview='%s...'",
        repo_name,
        req.chunk_strategy,
        req.question[:60],
    )

    try:
        retriever = HybridRetriever(
            collection_name=collection_name,
            persist_dir=req.persist_dir,
            bm25_dir=req.bm25_dir,
            alpha=req.alpha,
        )
    except Exception as exc:
        logger.warning(
            "Collection not found: collection=%s error=%s", collection_name, exc
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"Collection '{collection_name}' not found. "
                "Index the repository first via POST /index."
            ),
        )

    reranker = Reranker(top_k=req.top_k)

    raw_results = retriever.retrieve(req.question, top_k=20)
    reranked = reranker.rerank(req.question, raw_results)

    logger.info(
        "Retrieval complete: raw=%d reranked=%d", len(raw_results), len(reranked)
    )

    sys_prompt, user_prompt = build_prompt(req.question, reranked)
    result = generate_answer(sys_prompt, user_prompt, reranked, model=req.model)

    logger.info(
        "Response ready: latency_ms=%d tokens=%d sources=%d",
        result["latency_ms"],
        result.get("tokens_used", 0),
        len(result["sources"]),
    )

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        tokens_used=result.get("tokens_used", 0),
        latency_ms=result["latency_ms"],
        model=result["model"],
    )


@app.get("/eval/results", tags=["evaluation"])
def get_eval_results():
    """Return stored RAGAS scores and ablation study results."""
    logger.info("Eval results requested")
    results: dict = {}

    ragas_path = Path("evaluation/results/ragas_scores.json")
    if ragas_path.exists():
        with open(ragas_path) as f:
            results["ragas_scores"] = json.load(f)
        logger.debug("RAGAS scores loaded from %s", ragas_path)

    ablation_path = Path("evaluation/results/ablation_results.json")
    if ablation_path.exists():
        with open(ablation_path) as f:
            results["ablation_table"] = json.load(f)
        logger.debug("Ablation results loaded from %s", ablation_path)

    if not results:
        logger.warning("No evaluation results found on disk")
        raise HTTPException(
            status_code=404,
            detail=(
                "No evaluation results found. "
                "Run evaluation/ragas_eval.py or evaluation/ablation_study.py first."
            ),
        )

    return results


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)