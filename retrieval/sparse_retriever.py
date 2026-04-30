"""
sparse_retriever.py — BM25 keyword-based retrieval.

"""

import re
import pickle
import logging
import numpy as np
from pathlib import Path
from __future__ import annotations


logger = logging.getLogger(__name__)