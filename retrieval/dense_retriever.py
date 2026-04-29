"""
dense_retriever.py — Semantic search via ChromaDB cosine similarity.

Embeds the user query with the same ``text-embedding-3-small`` model 
used at ingestion time.
"""

import os
import logging
import chromadb
from typing import Optional
from __future__ import annotations
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# Module-level singletons — initialised lazily
_embeddings_model: OpenAIEmbeddings | None = None

def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        logger.debug("Initialising embeddings singleton: text-embedding-3-small")
        _embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
    return _embeddings_model

_chroma_client: chromadb.ClientAPI | None = None

def _get_chroma_client(persist_dir: str = "./chroma_db") -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        logger.debug("Initialising ChromaDB client: dir=%s", persist_dir)
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client