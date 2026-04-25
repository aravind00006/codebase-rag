"""fixed_size.py — strategy 1: dumb but reliable 
token-window chunking splits everything into 512-token 
windows with 50-token overlap main downside: doesn't care
about code structure, can cut mid-function used as the baseline.
"""

import logging
import tiktoken
from __future__ import annotations
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE: int = 512    # tokens
CHUNK_OVERLAP: int = 50  # tokens


def chunk_documents(documents: list[Document]) -> list[Document]:
    # using tiktoken cl100k_base to match text-embedding-3-small's tokenizer

    logger.info(
        "Fixed-size chunking: documents=%d chunk_size=%d overlap=%d",
        len(documents),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    encoding = tiktoken.get_encoding("cl100k_base")
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        file_path = doc.metadata.get("file_path", "unknown")
        splits = splitter.split_documents([doc])

        for i, chunk in enumerate(splits):
            chunk.metadata.update({
                "chunk_strategy": "fixed",
                "chunk_id": f"{file_path}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(splits),
            })
            all_chunks.append(chunk)

        logger.debug("Fixed chunks created: file=%s chunks=%d", file_path, len(splits))

    logger.info(
        "Fixed-size chunking complete: input_docs=%d output_chunks=%d",
        len(documents),
        len(all_chunks),
    )
    return all_chunks