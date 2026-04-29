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

class DenseRetriever:
    """
    Retrieves documents via cosine similarity in a ChromaDB collection.
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: str = "./chroma_db",
        top_k: int = 20,
    ) -> None:
        self.collection_name = collection_name
        self.top_k = top_k
        self.embeddings = _get_embeddings()

        client = _get_chroma_client(persist_dir)
        self.collection = client.get_collection(collection_name)

        logger.info(
            "DenseRetriever ready: collection=%s default_top_k=%d",
            collection_name,
            top_k,
        )
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_type_filter: Optional[str] = None,
    ) -> list[dict]:
        k = top_k or self.top_k

        logger.debug(
            "Dense retrieval: query_preview='%s...' top_k=%d filter=%s",
            query[:50],
            k,
            doc_type_filter,
        )

        query_embedding = self.embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs: list[dict] = []
        for i, (text, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            score = max(0.0, 1.0 - dist / 2.0)   # basic score, will fix next commit
            docs.append(
                {
                    "text": text,
                    "metadata": meta,
                    "score": score,
                    "rank": i,
                    "retriever": "dense",
                }
            )

        return docs