"""
ablation_study.py — Compare all 4 chunking strategies on the same question set.

"""


import argparse
import csv
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

STRATEGIES: list[str] = ["fixed", "recursive", "ast", "semantic"]
RESULTS_DIR: Path = Path("evaluation/results")

def run_ablation_study(
    repo_name: str,
    questions_file: str = "evaluation/test_questions.json",
    persist_dir: str = "./chroma_db",
    bm25_dir: str = "./bm25_indexes",
    output_csv: str = "evaluation/results/ablation_results.csv",
    output_json: str = "evaluation/results/ablation_results.json",
) -> list[dict]:
    """
    Run the same ground-truth questions through each chunking strategy and
    record RAGAS metrics + latency for each.

    Args:
        repo_name:      Repository identifier used to filter questions
                        (e.g. 'fastapi').
        questions_file: Path to the ground-truth JSON file.
        persist_dir:    ChromaDB persistence directory.
        bm25_dir:       Directory containing BM25 pickle indexes.
        output_csv:     Destination path for the CSV comparison table.
        output_json:    Destination path for the JSON results.

    Returns:
        List of result dicts, one per strategy.

    Raises:
        ValueError: If no questions match the given repo_name.
    """
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from generation.answer_generator import generate_answer
    from generation.prompt_builder import build_prompt
    from ingestion.embedder import get_collection_name
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.reranker import Reranker

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading questions: file=%s repo_filter=%s", questions_file, repo_name
    )
    with open(questions_file) as f:
        all_questions: list[dict] = json.load(f)

    questions = [q for q in all_questions if q.get("repo") == repo_name]
    logger.info(
        "Questions ready: total_in_file=%d filtered=%d",
        len(all_questions),
        len(questions),
    )

    if not questions:
        raise ValueError(
            f"No questions matched repo '{repo_name}' in {questions_file}. "
            "Check the 'repo' field in your test_questions.json."
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    all_results: list[dict] = []

    for strategy in STRATEGIES:
        logger.info("Ablation run starting: strategy=%s", strategy)
        collection_name = get_collection_name(repo_name, strategy)

        try:
            retriever = HybridRetriever(collection_name, persist_dir, bm25_dir)
            reranker = Reranker(top_k=5)
        except Exception as exc:
            logger.warning(
                "Skipping strategy — collection not found: strategy=%s error=%s",
                strategy,
                exc,
            )
            all_results.append(
                {
                    "strategy": strategy,
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "context_recall": None,
                    "avg_retrieval_latency_ms": None,
                    "avg_answer_latency_ms": None,
                    "total_tokens": None,
                    "error": str(exc),
                }
            )
            continue

        eval_data: dict[str, list] = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        retrieval_latencies: list[int] = []
        answer_latencies: list[int] = []
        total_tokens: int = 0

        for i, q in enumerate(questions):
            logger.debug(
                "Question %d/%d: strategy=%s preview='%s...'",
                i + 1,
                len(questions),
                strategy,
                q["question"][:55],
            )

            t0 = time.perf_counter()
            raw_results = retriever.retrieve(q["question"], top_k=10)
            reranked = reranker.rerank(q["question"], raw_results)
            retrieval_latencies.append(int((time.perf_counter() - t0) * 1000))

            t1 = time.perf_counter()
            sys_p, user_p = build_prompt(q["question"], reranked)
            result = generate_answer(sys_p, user_p, reranked)
            answer_latencies.append(int((time.perf_counter() - t1) * 1000))
            total_tokens += result.get("tokens_used", 0)

            eval_data["question"].append(q["question"])
            eval_data["answer"].append(result["answer"])
            eval_data["contexts"].append([c["text"] for c in reranked])
            eval_data["ground_truth"].append(q["ground_truth_answer"])

        logger.info(
            "Computing RAGAS metrics: strategy=%s questions=%d",
            strategy,
            len(questions),
        )
        dataset = Dataset.from_dict(eval_data)
        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )

        avg_retrieval = round(sum(retrieval_latencies) / len(retrieval_latencies))
        avg_answer = round(sum(answer_latencies) / len(answer_latencies))

        strategy_result = {
            "strategy": strategy,
            "faithfulness": round(float(ragas_result["faithfulness"]), 4),
            "answer_relevancy": round(float(ragas_result["answer_relevancy"]), 4),
            "context_precision": round(float(ragas_result["context_precision"]), 4),
            "context_recall": round(float(ragas_result["context_recall"]), 4),
            "avg_retrieval_latency_ms": avg_retrieval,
            "avg_answer_latency_ms": avg_answer,
            "total_tokens": total_tokens,
        }
        all_results.append(strategy_result)

        logger.info(
            "Strategy complete: strategy=%s faithfulness=%.3f relevancy=%.3f "
            "retrieval_ms=%d answer_ms=%d tokens=%d",
            strategy,
            strategy_result["faithfulness"],
            strategy_result["answer_relevancy"],
            avg_retrieval,
            avg_answer,
            total_tokens,
        )

    _write_csv(all_results, output_csv)
    _write_json(all_results, output_json)
    _print_summary(all_results, output_csv)

    return all_results