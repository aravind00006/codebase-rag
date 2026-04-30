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