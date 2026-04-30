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