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

# ---------------------------------------------------------------------------
# Core: embed_and_store
# ---------------------------------------------------------------------------

def embed_and_store(
    documents: list[Document],
    repo_name: str,
    chunk_strategy: str,
    repo_url: str = "",
    persist_dir: str = "./chroma_db",
    bm25_dir: str = "./bm25_indexes",
) -> dict:
    """
    Embed *documents* and persist to ChromaDB and a BM25 pickle.

    Args:
        documents:      Chunked Documents from any chunking strategy.
        repo_name:      Human-readable name used as the collection prefix.
        chunk_strategy: One of ``'fixed'``, ``'recursive'``, ``'ast'``, ``'semantic'``.
        repo_url:       Original GitHub URL stored in the index registry.
        persist_dir:    ChromaDB persistence directory.
        bm25_dir:       Directory for pickled BM25 indexes.

    Returns:
        ``dict`` with keys ``num_chunks``, ``time_taken_s``,
        ``estimated_cost_usd``, ``collection_name``.
    """
    start_time = time.perf_counter()
    Path(bm25_dir).mkdir(parents=True, exist_ok=True)

    collection_name = get_collection_name(repo_name, chunk_strategy)
    logger.info(
        "Starting embedding: chunks=%d collection=%s", len(documents), collection_name
    )

    client = _get_chroma_client(persist_dir)
    embeddings_fn = _get_embeddings()

    try:
        client.delete_collection(collection_name)
        logger.debug("Deleted existing collection: %s", collection_name)
    except Exception:
        logger.debug("Collection does not exist yet — skipping delete: %s", collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug("Created ChromaDB collection: %s", collection_name)

    texts = [doc.page_content for doc in documents]
    ids = [
        _sanitize_id(doc.metadata.get("chunk_id", f"chunk_{i}"))
        for i, doc in enumerate(documents)
    ]

    estimated_tokens = sum(len(t) for t in texts) // 4
    estimated_cost = (estimated_tokens / 1_000_000) * COST_PER_MILLION_TOKENS
    logger.info(
        "Cost estimate: tokens=%d cost_usd=%.4f", estimated_tokens, estimated_cost
    )

    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, batch_start in enumerate(range(0, len(texts), BATCH_SIZE)):
        batch = texts[batch_start : batch_start + BATCH_SIZE]
        all_embeddings.extend(embeddings_fn.embed_documents(batch))
        logger.debug(
            "Embedding progress: batch=%d/%d chunks=%d/%d",
            batch_idx + 1,
            total_batches,
            min(batch_start + BATCH_SIZE, len(texts)),
            len(texts),
        )

    logger.info("All embeddings generated: count=%d", len(all_embeddings))

    metadatas = []
    for doc in documents:
        meta: dict = {}
        for k, v in doc.metadata.items():
            meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        metadatas.append(meta)

    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info("Stored vectors in ChromaDB: collection=%s", collection_name)

    tokenized_corpus = [text.lower().split() for text in texts]
    bm25_data = {
        "index": BM25Okapi(tokenized_corpus),
        "texts": texts,
        "ids": ids,
        "metadatas": metadatas,
    }
    bm25_path = Path(bm25_dir) / f"{collection_name}.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)
    logger.info("BM25 index persisted: path=%s", bm25_path)

    elapsed = time.perf_counter() - start_time

    if repo_url:
        _update_registry(
            repo_url,
            chunk_strategy,
            {
                "repo_name":          repo_name,
                "num_chunks":         len(documents),
                "collection_name":    collection_name,
                "time_taken_s":       round(elapsed, 2),
                "estimated_cost_usd": round(estimated_cost, 6),
            },
        )

    logger.info(
        "Indexing complete: chunks=%d elapsed_s=%.2f cost_usd=%.4f",
        len(documents),
        elapsed,
        estimated_cost,
    )

    return {
        "num_chunks":         len(documents),
        "time_taken_s":       round(elapsed, 2),
        "estimated_cost_usd": round(estimated_cost, 6),
        "collection_name":    collection_name,
    }


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _registry_key(repo_url: str, chunk_strategy: str) -> str:
    return hashlib.md5(f"{repo_url}:{chunk_strategy}".encode()).hexdigest()


def _load_registry() -> dict:
    if not Path(INDEX_REGISTRY_FILE).exists():
        return {}
    with open(INDEX_REGISTRY_FILE) as f:
        return json.load(f)


def _update_registry(repo_url: str, chunk_strategy: str, info: dict) -> None:
    registry = _load_registry()
    key = _registry_key(repo_url, chunk_strategy)
    registry[key] = {"repo_url": repo_url, "chunk_strategy": chunk_strategy, **info}
    with open(INDEX_REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)
    logger.debug("Registry updated: key=%s", key)


def _sanitize_id(id_: str) -> str:
    """Return a ChromaDB-safe ID (alphanumeric + ``-_.``, max 200 chars)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in id_)[:200]