"""
sparse_retriever.py — BM25 keyword-based retrieval.

"""
import re
import pickle
import logging
import numpy as np
from pathlib import Path
from __future__ import annotations


logger = logging.getLogger(__name__)


class SparseRetriever:
    """
    Keyword retrieval via a pre-built BM25Okapi index.

    """

    def __init__(self, collection_name: str, bm25_dir: str = "./bm25_indexes") -> None:
        bm25_path = Path(bm25_dir) / f"{collection_name}.pkl"

        if not bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {bm25_path}\n"
                "Run the ingestion pipeline (ingest_repo.py) to build the index."
            )

        logger.info("Loading BM25 index: path=%s", bm25_path)
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["index"]
        self.texts: list[str] = data["texts"]
        self.ids: list[str] = data["ids"]
        self.metadatas: list[dict] = data["metadatas"]

        logger.info(
            "BM25 index loaded: collection=%s corpus_size=%d",
            collection_name,
            len(self.texts),
        )

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Retrieve the top_k highest-scoring BM25 results for query.

        """
        logger.debug(
            "Sparse retrieval: query_preview='%s...' top_k=%d", query[:50], top_k
        )

        tokenized_query = re.findall(r"\w+", query.lower())
        raw_scores: np.ndarray = self.bm25.get_scores(tokenized_query)

        max_score = float(raw_scores.max())
        if max_score > 0:
            normalised = raw_scores / max_score
        else:
            logger.warning("All BM25 scores are zero — query may be out-of-vocabulary")
            normalised = raw_scores

        top_indices = np.argsort(normalised)[::-1][:top_k]

        results: list[dict] = []
        for rank, idx in enumerate(top_indices):
            if normalised[idx] == 0:
                break
            results.append(
                {
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(normalised[idx]),
                    "rank": rank,
                    "retriever": "sparse",
                }
            )

        logger.debug(
            "Sparse retrieval complete: returned=%d top_score=%.4f",
            len(results),
            results[0]["score"] if results else 0.0,
        )
        return results