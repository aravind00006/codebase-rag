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