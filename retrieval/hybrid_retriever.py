"""
hybrid_retriever.py — Reciprocal Rank Fusion 
combining dense and sparse signals.

"""



import logging
from __future__ import annotations
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)

RRF_K: int = 60  # RRF smoothing constant

class HybridRetriever:
    """
    Fuses dense and sparse retrieval rankings with Reciprocal Rank Fusion.

    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: str = "./chroma_db",
        bm25_dir: str = "./bm25_indexes",
        alpha: float = 0.7,
    ) -> None:
        """
        Args:
            collection_name: ChromaDB collection and BM25 index name.
            persist_dir:     ChromaDB persistence directory.
            bm25_dir:        Directory containing pickled BM25 indexes.
            alpha:           Dense retrieval weight ∈ [0, 1].
        """
        self.alpha = alpha
        self.dense = DenseRetriever(collection_name, persist_dir)
        self.sparse = SparseRetriever(collection_name, bm25_dir)

        logger.info(
            "HybridRetriever ready: collection=%s alpha=%.2f",
            collection_name,
            alpha,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        doc_type_filter: str | None = None,
    ) -> list[dict]:
        """
        Retrieve and RRF-fuse results from dense and sparse retrievers.

        Both retrievers fetch 20 candidates each before fusion.

        Args:
            query:           Natural language question.
            top_k:           Number of fused results to return.
            doc_type_filter: Optionally restrict dense retrieval to
                             ``'code'``, ``'documentation'``, or ``'config'``.

        Returns:
            List of up to *top_k* result dicts, sorted by descending RRF score.
        """
        logger.debug(
            "Hybrid retrieval: query_preview='%s...' top_k=%d alpha=%.2f",
            query[:50],
            top_k,
            self.alpha,
        )

        dense_results = self.dense.retrieve(
            query, top_k=20, doc_type_filter=doc_type_filter
        )
        sparse_results = self.sparse.retrieve(query, top_k=20)

        logger.debug(
            "Candidates fetched: dense=%d sparse=%d",
            len(dense_results),
            len(sparse_results),
        )

        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        def _chunk_id(result: dict) -> str:
            return result["metadata"].get("chunk_id", result["text"][:100])

        for rank, result in enumerate(dense_results):
            cid = _chunk_id(result)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.alpha / (rank + RRF_K)
            chunk_data.setdefault(cid, result)

        for rank, result in enumerate(sparse_results):
            cid = _chunk_id(result)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1 - self.alpha) / (rank + RRF_K)
            chunk_data.setdefault(cid, result)

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused: list[dict] = []
        for rank, (cid, rrf_score) in enumerate(sorted_chunks[:top_k]):
            result = {
                **chunk_data[cid],
                "rrf_score": rrf_score,
                "rank": rank,
                "retriever": "hybrid",
            }
            fused.append(result)

        logger.debug(
            "RRF fusion complete: unique_chunks=%d returning=%d top_rrf_score=%.6f",
            len(rrf_scores),
            len(fused),
            fused[0]["rrf_score"] if fused else 0.0,
        )
        return fused