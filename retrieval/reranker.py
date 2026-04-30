"""
reranker.py — Cross-encoder re-ranking for precision boost.
A cross-encoder jointly encodes (query, passage)
pairs — unlike bi-encoders that embed them independently.

"""

import logging
import numpy as np
from __future__ import annotations
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH: int = 512  # Maximum token length per pair

# Module-level — loaded once per process to avoid repeated
# disk reads and model initialisation overhead
_model = None


def _get_model() -> "CrossEncoder":
    global _model
    if _model is None:
        logger.info("Loading cross-encoder model: %s", RERANKER_MODEL)
        _model = CrossEncoder(RERANKER_MODEL, max_length=MAX_LENGTH)
        logger.info("Cross-encoder model loaded successfully")
    return _model

class Reranker:
    """
    Re-ranks a list of retrieved chunks using a cross-encoder.

    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.model = _get_model()
        logger.debug(
            "Reranker initialised: top_k=%d model=%s",
            top_k,
            RERANKER_MODEL,
        )

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        """
        Score and re-rank results against query.

        """
        if not results:
            logger.warning("Reranker received empty results list")
            return []

        logger.debug(
            "Reranking %d candidates: query_preview='%s...'",
            len(results),
            query[:50],
        )

        pairs = [(query, result["text"]) for result in results]

        # np.atleast_1d handles edge case where predict() returns a single float
        raw = self.model.predict(pairs)
        scores: list[float] = np.atleast_1d(raw).tolist()

        scored = [
            {
                **result,
                "rerank_score": float(score),
                "original_rank": result.get("rank", i),
            }
            for i, (result, score) in enumerate(zip(results, scores))
        ]

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        top = scored[: self.top_k]
        for rank, result in enumerate(top):
            result["rank"] = rank

        logger.debug(
            "Reranking complete: input=%d output=%d top_score=%.4f",
            len(results),
            len(top),
            top[0]["rerank_score"] if top else 0.0,
        )
        return top