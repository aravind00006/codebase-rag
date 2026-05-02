"""
ragas_eval.py — Run RAGAS evaluation metrics on the RAG pipeline.

"""

import argparse
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = Path("evaluation/results")