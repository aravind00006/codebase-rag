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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _semantic_split(doc: Document, model) -> list[Document]:
    """Split a single document by cosine similarity between adjacent sentences."""
    file_path = doc.metadata.get("file_path", "unknown")

    sentences = [s.strip() for s in doc.page_content.split("\n") if s.strip()]
    if not sentences:
        return []

    if len(sentences) == 1:
        return [_make_chunk(doc, sentences, 0, file_path, 1)]

    embeddings: np.ndarray = model.encode(sentences, convert_to_numpy=True)

    groups: list[list[str]] = []
    current_group: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = float(_cosine_similarity(embeddings[i - 1], embeddings[i]))
        if sim >= SIMILARITY_THRESHOLD:
            current_group.append(sentences[i])
        else:
            groups.append(current_group)
            current_group = [sentences[i]]

    groups.append(current_group)

    return [
        _make_chunk(doc, group, idx, file_path, len(groups))
        for idx, group in enumerate(groups)
    ]


def _make_chunk(
    doc: Document,
    sentences: list[str],
    index: int,
    file_path: str,
    total: int,
) -> Document:
    return Document(
        page_content="\n".join(sentences),
        metadata={
            **doc.metadata,
            "chunk_strategy":       "semantic",
            "chunk_id":             f"{file_path}_semantic_{index}",
            "chunk_index":          index,
            "total_chunks":         total,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        },
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)