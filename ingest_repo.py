"""
ingest_repo.py — CLI to index a GitHub repository with one or all chunking strategies.

"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dotenv import load_dotenv
from ingestion.chunkers import get_chunker
from ingestion.embedder import check_if_indexed, embed_and_store
from ingestion.file_parser import parse_documents
from ingestion.repo_loader import RepoLoader

load_dotenv()

logger = logging.getLogger(__name__)


def ingest(
    repo_url: str,
    strategy: str,
    persist_dir: str,
    bm25_dir: str,
) -> None:

