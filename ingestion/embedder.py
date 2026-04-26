"""
Generate embeddings and persist to ChromaDB and BM25.

"""

import json
import time
import pickle
import hashlib
import logging
import chromadb
from pathlib import Path
from typing import Final
from __future__ import annotations
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

COST_PER_MILLION_TOKENS: Final[float] = 0.02
BATCH_SIZE: Final[int] = 100
INDEX_REGISTRY_FILE: Final[str] = ".indexed_repos.json"

_embeddings_model: OpenAIEmbeddings | None = None
_chroma_client: chromadb.ClientAPI | None = None


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set. Check your .env file.")
        logger.debug("Initialising OpenAI embeddings model: text-embedding-3-small")
        _embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )
    return _embeddings_model


def _get_chroma_client(persist_dir: str = "./chroma_db") -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        logger.debug("Creating ChromaDB PersistentClient: dir=%s", persist_dir)
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_collection_name(repo_name: str, chunk_strategy: str) -> str:
    """
    Return a ChromaDB-safe collection name for *(repo_name, chunk_strategy)*.

    ChromaDB collection names must be alphanumeric + underscores and <= 63 chars.
    """
    raw = f"{repo_name}_{chunk_strategy}"
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in raw)
    return safe[:63]


def check_if_indexed(repo_url: str, chunk_strategy: str) -> bool:
    """
    Return ``True`` if *(repo_url, chunk_strategy)* has already been indexed.

    Reads the local registry file written by :func:`embed_and_store`.
    """
    registry = _load_registry()
    key = _registry_key(repo_url, chunk_strategy)
    exists = key in registry
    logger.debug(
        "Index check: repo_url=%s strategy=%s indexed=%s",
        repo_url,
        chunk_strategy,
        exists,
    )
    return exists