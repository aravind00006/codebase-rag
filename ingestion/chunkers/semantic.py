"""
semantic.py — Strategy 4: Semantic similarity-based chunking.
Groups sentences into chunks by embedding each sentence with
sentence-transformers.
"""

import logging
import numpy as np
from __future__ import annotations
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD: float = 0.85
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

_embed_model = None


def _get_embed_model() :
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformer: model=%s", EMBEDDING_MODEL)
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Sentence-transformer loaded")
    return _embed_model


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Group sentences by embedding similarity into semantic chunks.

    """
    logger.info(
        "Semantic chunking: documents=%d threshold=%.2f",
        len(documents),
        SIMILARITY_THRESHOLD,
    )

    model = _get_embed_model()
    all_chunks: list[Document] = []

    for doc in documents:
        file_path = doc.metadata.get("file_path", "unknown")
        chunks = _semantic_split(doc, model)
        all_chunks.extend(chunks)
        logger.debug("Semantic chunks created: file=%s chunks=%d", file_path, len(chunks))

    logger.info(
        "Semantic chunking complete: input_docs=%d output_chunks=%d",
        len(documents),
        len(all_chunks),
    )
    return all_chunks
