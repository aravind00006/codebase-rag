"""
ablation_study.py — Compare all 4 chunking strategies on the same question set.

"""


import argparse
import csv
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

STRATEGIES: list[str] = ["fixed", "recursive", "ast", "semantic"]
RESULTS_DIR: Path = Path("evaluation/results")