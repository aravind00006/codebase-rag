"""
ast_based.py — Strategy 3: AST-aware chunking at function/class boundaries.

Uses Python's built-in ``ast`` module to split source files at *semantic*
boundaries — one chunk per top-level function or class definition — so that
each chunk is always a complete, syntactically valid unit.

"""



import ast
import logging
import textwrap
from __future__ import annotations
from langchain_core.documents import Document
from ingestion.chunkers import recursive as recursive_chunker

logger = logging.getLogger(__name__)


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split Python documents at AST function/class boundaries.

    Non-Python files are handled by the recursive fallback chunker.

    """
    

    logger.info("AST-based chunking: documents=%d", len(documents))

    all_chunks: list[Document] = []
    fallback_docs: list[Document] = []

    for doc in documents:
        language = doc.metadata.get("language", "")
        if language != "python":
            fallback_docs.append(doc)
            continue

        chunks = _chunk_python(doc)
        if chunks is None:
            # Syntax error in file — hand off to recursive chunker
            logger.warning(
                "AST parse failed — using recursive fallback: file=%s",
                doc.metadata.get("file_path", "unknown"),
            )
            fallback_docs.append(doc)
        else:
            all_chunks.extend(chunks)

    # Process non-Python and failed-parse files with recursive chunker
    if fallback_docs:
        logger.debug("Applying recursive fallback: files=%d", len(fallback_docs))
        fallback_chunks = recursive_chunker.chunk_documents(fallback_docs)

        # Re-tag so the ablation study still groups these under "ast"
        for chunk in fallback_chunks:
            chunk.metadata["chunk_strategy"] = "ast"
            chunk.metadata.setdefault("node_type", "fallback")

        all_chunks.extend(fallback_chunks)

    logger.info(
        "AST chunking complete: input_docs=%d output_chunks=%d fallback_files=%d",
        len(documents),
        len(all_chunks),
        len(fallback_docs),
    )
    return all_chunks