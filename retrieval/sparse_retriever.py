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

        if not bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {bm25_path}\n"
                "Run the ingestion pipeline to build the index."
            )
    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Retrieve the *top_k* highest-scoring BM25 results for query.

        """
        logger.debug(
            "Sparse retrieval: query_preview='%s...' top_k=%d", query[:50], top_k
        )
        return []  # placeholder 