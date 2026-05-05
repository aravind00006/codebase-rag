"""
ingest_repo.py — CLI to index a GitHub repository with one or all chunking strategies.

"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from dotenv import load_dotenv

load_dotenv()


def ingest(
    repo_url: str,
    strategy: str,
    persist_dir: str,
    bm25_dir: str,
) -> None:
    pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Index a GitHub repository into ChromaDB + BM25 for RAG Q&A."
    )
    parser.add_argument("--url", required=True, help="GitHub repository URL")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "ast", "semantic"],
        default="ast",
        help="Chunking strategy (default: ast)",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Index with all 4 strategies (needed for ablation study)",
    )
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--bm25-dir", default="./bm25_indexes")
    args = parser.parse_args()

    strategies = (
        ["fixed", "recursive", "ast", "semantic"]
        if args.all_strategies
        else [args.strategy]
    )

    print(f"\n  Indexing: {args.url}")
    print(f"  Strategies: {', '.join(strategies)}\n")

    for s in strategies:
        ingest(args.url, s, args.persist_dir, args.bm25_dir)