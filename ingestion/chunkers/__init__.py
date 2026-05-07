"""
chunkers — Four chunking strategies for the RAG ingestion pipeline.

"""

from __future__ import annotations

import importlib
import logging
import types

logger = logging.getLogger(__name__)

_STRATEGY_MODULES: dict[str, str] = {
    "fixed": "ingestion.chunkers.fixed_size",
    "recursive": "ingestion.chunkers.recursive",
    "ast": "ingestion.chunkers.ast_based",
    "semantic": "ingestion.chunkers.semantic",
}


def get_chunker(strategy: str) -> types.ModuleType:
    """
    Return the chunker module for strategy.

    """
    if strategy not in _STRATEGY_MODULES:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Valid options: {list(_STRATEGY_MODULES)}"
        )

    module_path = _STRATEGY_MODULES[strategy]
    module = importlib.import_module(module_path)
    logger.debug("Chunker loaded: strategy=%s module=%s", strategy, module_path)
    return module