"""
ragas_eval.py — Run RAGAS evaluation metrics on the RAG pipeline.

"""

import argparse
import json
import logging
import time
from pathlib import Path
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from generation.answer_generator import generate_answer
from generation.prompt_builder import build_prompt
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from dotenv import load_dotenv
from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

load_dotenv()

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = Path("evaluation/results")

def run_ragas_evaluation(
    collection_name: str,
    chunk_strategy: str = "ast",
    questions_file: str = "evaluation/test_questions.json",
    repo_filter: str = "fastapi",
    persist_dir: str = "./chroma_db",
    bm25_dir: str = "./bm25_indexes",
    output_file: str = "evaluation/results/ragas_scores.json",
) -> dict:
    """
    Evaluate the full RAG pipeline against ground-truth Q&A pairs.

    """
   
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading questions: file=%s repo_filter=%s", questions_file, repo_filter
    )
    with open(questions_file) as f:
        all_questions: list[dict] = json.load(f)

    questions = [q for q in all_questions if q.get("repo") == repo_filter]

    if not questions:
        raise ValueError(
            f"No questions found for repo='{repo_filter}' in {questions_file}"
        )

    logger.info(
        "Evaluation set ready: questions=%d collection=%s strategy=%s",
        len(questions),
        collection_name,
        chunk_strategy,
    )

    retriever = HybridRetriever(collection_name, persist_dir, bm25_dir)
    reranker = Reranker(top_k=5)

    eval_data: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    latencies: list[int] = []

    for i, q in enumerate(questions):
        logger.debug(
            "Processing: idx=%d/%d preview='%s...'",
            i + 1,
            len(questions),
            q["question"][:60],
        )

        start = time.perf_counter()

        raw_results = retriever.retrieve(q["question"], top_k=10)
        reranked = reranker.rerank(q["question"], raw_results)

        sys_prompt, user_prompt = build_prompt(q["question"], reranked)
        result = generate_answer(sys_prompt, user_prompt, reranked)

        latency_ms = int((time.perf_counter() - start) * 1000)
        latencies.append(latency_ms)

        eval_data["question"].append(q["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([chunk["text"] for chunk in reranked])
        eval_data["ground_truth"].append(q["ground_truth_answer"])

        logger.debug(
            "Question complete: idx=%d latency_ms=%d", i + 1, latency_ms
        )

    logger.info(
        "All questions processed: count=%d avg_latency_ms=%.0f",
        len(questions),
        sum(latencies) / len(latencies),
    )

    dataset = Dataset.from_dict(eval_data)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    logger.info("Running RAGAS evaluation: metrics=4")
    ragas_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings_model,
    )

    scores = {
        "chunk_strategy": chunk_strategy,
        "collection_name": collection_name,
        "num_questions": len(questions),
        "faithfulness": round(float(ragas_result["faithfulness"]), 4),
        "answer_relevancy": round(float(ragas_result["answer_relevancy"]), 4),
        "context_precision": round(float(ragas_result["context_precision"]), 4),
        "context_recall": round(float(ragas_result["context_recall"]), 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies)),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
    }

    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2)

    logger.info(
        "RAGAS evaluation complete: faithfulness=%.3f relevancy=%.3f "
        "precision=%.3f recall=%.3f avg_latency_ms=%d saved=%s",
        scores["faithfulness"],
        scores["answer_relevancy"],
        scores["context_precision"],
        scores["context_recall"],
        scores["avg_latency_ms"],
        output_file,
    )

    print(
        f"\n{'─'*52}\n"
        f"  RAGAS Evaluation Complete\n"
        f"{'─'*52}\n"
        f"  Strategy          : {chunk_strategy}\n"
        f"  Questions         : {len(questions)}\n"
        f"  Faithfulness      : {scores['faithfulness']:.3f}\n"
        f"  Answer Relevancy  : {scores['answer_relevancy']:.3f}\n"
        f"  Context Precision : {scores['context_precision']:.3f}\n"
        f"  Context Recall    : {scores['context_recall']:.3f}\n"
        f"  Avg Latency       : {scores['avg_latency_ms']} ms\n"
        f"{'─'*52}\n"
        f"  Saved → {output_file}\n"
    )

    return scores