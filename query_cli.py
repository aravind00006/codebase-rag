"""
query_cli.py — Interactive REPL for querying an indexed repository.

"""

from __future__ import annotations

import argparse
import logging
import sys
from dotenv import load_dotenv
from generation.answer_generator import generate_answer
from generation.prompt_builder import build_prompt
from ingestion.embedder import get_collection_name
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

load_dotenv()

logger = logging.getLogger(__name__)


def run_repl(
    repo: str,
    strategy: str,
    alpha: float,
    model: str,
    persist_dir: str,
    bm25_dir: str,
) -> None:
    """
    Start an interactive query loop against an indexed repository.

    """

    collection_name = get_collection_name(repo, strategy)

    logger.info(
        "Starting REPL: repo=%s strategy=%s collection=%s model=%s alpha=%.2f",
        repo,
        strategy,
        collection_name,
        model,
        alpha,
    )

    try:
        retriever = HybridRetriever(
            collection_name=collection_name,
            persist_dir=persist_dir,
            bm25_dir=bm25_dir,
            alpha=alpha,
        )
        reranker = Reranker(top_k=5)
    except Exception as exc:
        logger.error("Index not found — run ingest_repo.py first: %s", exc)
        raise ValueError(
            f"Collection '{collection_name}' not found. "
            f"Run: python ingest_repo.py --url <repo_url> --strategy {strategy}"
        ) from exc
    
    # implement interactive question loop with retrieval, reranking, and answer display

    print(
        f"\n  {'-'*52}\n"
        f"  RAG Codebase Q&A — Interactive CLI\n"
        f"  {'-'*52}\n"
        f"  Repo       : {repo}\n"
        f"  Strategy   : {strategy}\n"
        f"  Collection : {collection_name}\n"
        f"  Model      : {model}\n"
        f"  Alpha      : {alpha}  (dense/sparse balance)\n"
        f"  {'-'*52}\n"
        f"  Type a question and press Enter.  'quit' to exit.\n"
    )

    while True:
        try:
            question = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!\n")
            logger.info("REPL terminated by user")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye!\n")
            logger.info("REPL terminated by user command")
            break

        logger.info("Processing question: preview='%s...'", question[:60])

        try:
            raw = retriever.retrieve(question, top_k=20)
            reranked = reranker.rerank(question, raw)

            sys_p, user_p = build_prompt(question, reranked)
            result = generate_answer(sys_p, user_p, reranked, model=model)

            logger.info(
                "Answer generated: latency_ms=%d tokens=%d sources=%d",
                result["latency_ms"],
                result.get("tokens_used", 0),
                len(result["sources"]),
            )

            _print_result(result)

        except Exception as exc:
            logger.error("Query failed: %s", exc, exc_info=True)
            print(f"\n  [error] {exc}\n")

#add answer/source printer and argparse entrypoint with clean error handling

def _print_result(result: dict) -> None:
    """Print the answer and source citations to stdout."""
    print(
        f"\n  {'-'*60}\n"
        f"  Answer  ({result['latency_ms']} ms · "
        f"{result.get('tokens_used', '?')} tokens · {result['model']})\n"
        f"  {'-'*60}\n"
    )
    for line in result["answer"].splitlines():
        print(f"  {line}")

    if result["sources"]:
        print(f"\n  Sources ({len(result['sources'])}):")
        for i, src in enumerate(result["sources"]):
            fn = f" -> {src['function_name']}()" if src.get("function_name") else ""
            score = f"  [{src['score']:.3f}]" if src.get("score") else ""
            print(f"    [{i+1}] {src['file_path']}{fn}{score}")

    print(f"\n  {'-'*60}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Interactive Q&A CLI for indexed repositories."
    )
    parser.add_argument(
        "--repo", required=True, help="Repository name (e.g. tiangolo_fastapi)"
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "ast", "semantic"],
        default="ast",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Dense/sparse balance: 0.0 = pure BM25, 1.0 = pure semantic",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--bm25-dir", default="./bm25_indexes")
    args = parser.parse_args()

    try:
        run_repl(
            repo=args.repo,
            strategy=args.strategy,
            alpha=args.alpha,
            model=args.model,
            persist_dir=args.persist_dir,
            bm25_dir=args.bm25_dir,
        )
    except ValueError as exc:
        logger.error("Startup failed: %s", exc)
        print(f"\n  [error] {exc}\n")
        sys.exit(1)