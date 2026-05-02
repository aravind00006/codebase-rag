"""
answer_generator.py — LLM call with streaming, retry logic, and observability.

"""

import logging
import os
import time

logger = logging.getLogger(__name__)

_RETRY_WAIT_BASE: int = 5   # seconds — doubles on each attempt
_MAX_RETRIES: int = 3