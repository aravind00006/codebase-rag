"""
repo_loader.py — Clone and walk GitHub repositories.

"""

import git
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps file extension → language label stored in Document metadata
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".jsx":  "javascript",
    ".tsx":  "typescript",
    ".md":   "markdown",
    ".txt":  "text",
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".json": "json",
    ".java": "java",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".c":    "c",
    ".go":   "go",
    ".rs":   "rust",
    ".rb":   "ruby",
    ".php":  "php",
    ".sh":   "bash",
    ".toml": "toml",
    ".ini":  "ini",
    ".cfg":  "ini",
    ".env":  "env",
}

# Directory names that are always skipped during the repo walk
SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules", ".git", "__pycache__", ".pytest_cache",
    "venv", ".venv", "env", "dist", "build", ".next",
    ".nuxt", "target", "vendor", ".idea", ".vscode",
    "coverage", ".coverage", "htmlcov",
})

# File suffixes / name endings that are always skipped
SKIP_FILE_PATTERNS: frozenset[str] = frozenset({
    ".min.js", ".min.css", ".map", ".lock", ".sum",
    "-lock.json", ".pyc", ".pyo", ".pyd",
})

# Files larger than this are skipped (likely auto-generated)
MAX_FILE_SIZE_BYTES: int = 500_000  # 500 KB

