# recursive.py — splits documents using langchain's recursive character splitter
# tries separators in order: class → def → blank lines → lines → words → chars


import logging
from __future__ import annotations
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE: int = 1000    # chars
CHUNK_OVERLAP: int = 200  # chars

# order matters here — tries top to bottom, falls back if chunk still too big
_CODE_SEPARATORS: list[str] = [
    "\nclass ",
    "\ndef ",
    "\n\n",
    "\n",
    " ",
    "",
]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Chunk documents with a code-aware recursive character splitter.
    """
    logger.info(
        "Recursive chunking: documents=%d chunk_size=%d overlap=%d",
        len(documents),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    splitter = RecursiveCharacterTextSplitter(
        separators=_CODE_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        file_path = doc.metadata.get("file_path", "unknown")
        splits = splitter.split_documents([doc])

        for i, chunk in enumerate(splits):
            chunk.metadata.update({
                "chunk_strategy": "recursive",
                "chunk_id": f"{file_path}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(splits),
            })
            all_chunks.append(chunk)

        logger.debug("Recursive chunks created: file=%s chunks=%d", file_path, len(splits))

    logger.info(
        "Recursive chunking complete: input_docs=%d output_chunks=%d",
        len(documents),
        len(all_chunks),
    )
    return all_chunks