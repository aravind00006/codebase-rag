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