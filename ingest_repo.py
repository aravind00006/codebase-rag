"""
ingest_repo.py — CLI to index a GitHub repository with one or all chunking strategies.

"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dotenv import load_dotenv
from ingestion.chunkers import get_chunker
from ingestion.embedder import check_if_indexed, embed_and_store
from ingestion.file_parser import parse_documents
from ingestion.repo_loader import RepoLoader

load_dotenv()

logger = logging.getLogger(__name__)


def ingest(
    repo_url: str,
    strategy: str,
    persist_dir: str,
    bm25_dir: str,
) -> None:


    logger.info("Ingestion started: repo=%s strategy=%s", repo_url, strategy)

    if check_if_indexed(repo_url, strategy):
        logger.info(
            "Already indexed — skipping: repo_url=%s strategy=%s",
            repo_url,
            strategy,
        )
        print(f"  [skip] Already indexed ({strategy}).")
        return

    pipeline_start = time.perf_counter()

    # Stage 1 — Load
    logger.info("Stage 1/4 — Loading repository")
    loader = RepoLoader()
    docs = loader.load(repo_url)
    logger.info("Repository loaded: files=%d", len(docs))

    # Stage 2 — Parse
    logger.info("Stage 2/4 — Parsing file metadata")
    docs = parse_documents(docs)
    logger.info("Metadata parsed: documents=%d", len(docs))

    # Stage 3 — Chunk
    logger.info("Stage 3/4 — Chunking: strategy=%s", strategy)
    chunker = get_chunker(strategy)
    chunks = chunker.chunk_documents(docs)
    logger.info("Chunking complete: chunks=%d", len(chunks))

    # Stage 4 — Embed + Store
    logger.info("Stage 4/4 — Embedding and storing")

    # Note: repo_name derivation is duplicated from api/main.py.
    # TODO: extract to ingestion/utils.py once all callers are identified.
    url_parts = repo_url.rstrip("/").split("/")
    repo_name = (
        f"{url_parts[-2]}_{url_parts[-1]}" if len(url_parts) >= 2 else url_parts[-1]
    )

    result = embed_and_store(
        documents=chunks,
        repo_name=repo_name,
        chunk_strategy=strategy,
        repo_url=repo_url,
        persist_dir=persist_dir,
        bm25_dir=bm25_dir,
    )

    elapsed = time.perf_counter() - pipeline_start

    logger.info(
        "Ingestion complete: repo=%s strategy=%s chunks=%d elapsed_s=%.2f cost_usd=%.4f",
        repo_name,
        strategy,
        result["num_chunks"],
        elapsed,
        result["estimated_cost_usd"],
    )

    print(
        f"  [done] {strategy:<10}  {result['num_chunks']:>6,} chunks  "
        f"{elapsed:>5.1f}s  approx. ${result['estimated_cost_usd']:.4f}  "
        f"->  {result['collection_name']}"
    )