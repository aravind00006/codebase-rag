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

# ---------------------------------------------------------------------------
# RepoLoader
# ---------------------------------------------------------------------------

class RepoLoader:
    """
    Loads a GitHub repository and returns Documents ready for chunking.

    Each returned Document carries metadata:
        file_path, language, repo_name, file_size,
        last_modified, file_name, extension.

    Example usage::

        loader = RepoLoader()
        docs = loader.load("https://github.com/tiangolo/fastapi")
    """

    def __init__(self, repos_dir: str = "./repos") -> None:
        self.repos_dir = Path(repos_dir)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("RepoLoader initialised — repos_dir=%s", self.repos_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source: str) -> list[Document]:
        """
        Load a repo from a GitHub URL or a local filesystem path.

        Args:
            source: GitHub URL (``https://github.com/owner/repo``)
                    or a local directory path.

        Returns:
            List of Document objects, one per supported source file.

        Raises:
            git.GitCommandError: If cloning fails.
            FileNotFoundError: If a local path does not exist.
        """
        if source.startswith("https://github.com") or source.startswith("git@"):
            local_path = self._clone_repo(source)
            repo_name = self._extract_repo_name(source)
        else:
            local_path = Path(source)
            if not local_path.exists():
                raise FileNotFoundError(f"Local path does not exist: {local_path}")
            repo_name = local_path.name

        logger.info("Walking repo tree: repo=%s path=%s", repo_name, local_path)
        documents = self._walk_repo(local_path, repo_name)
        logger.info(
            "Repo loaded: repo=%s total_files=%d", repo_name, len(documents)
        )
        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clone_repo(self, url: str) -> Path:
        """Clone *url* into ``repos_dir`` (shallow, depth=1) if not cached."""
        repo_name = self._extract_repo_name(url)
        local_path = self.repos_dir / repo_name

        if local_path.exists():
            logger.info("Repo already on disk — skipping clone: path=%s", local_path)
            return local_path

        logger.info("Cloning repo: url=%s target=%s", url, local_path)
        start = time.perf_counter()

        try:
            git.Repo.clone_from(url, local_path, depth=1)
        except git.GitCommandError:
            logger.exception("Clone failed: url=%s", url)
            raise

        elapsed = time.perf_counter() - start
        logger.info("Clone complete: elapsed_s=%.2f", elapsed)
        return local_path

    @staticmethod
    def _extract_repo_name(url: str) -> str:
        """Return ``owner_repo`` slug from a GitHub URL."""
        url = url.rstrip("/").removesuffix(".git")
        parts = url.split("/")
        return f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else parts[-1]

    def _walk_repo(self, local_path: Path, repo_name: str) -> list[Document]:
        """Recursively walk *local_path* and return one Document per valid file."""
        documents: list[Document] = []
        skipped_large = 0
        skipped_unreadable = 0

        for file_path in local_path.rglob("*"):
            if file_path.is_dir():
                continue

            # Skip blacklisted directories
            rel_parts = file_path.relative_to(local_path).parts
            if any(part in SKIP_DIRS for part in rel_parts):
                continue

            # Skip blacklisted file patterns
            if any(file_path.name.endswith(pat) for pat in SKIP_FILE_PATTERNS):
                continue

            # Skip unsupported extensions
            suffix = file_path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue

            # Size guard
            try:
                file_size = file_path.stat().st_size
            except OSError:
                logger.warning("Cannot stat file — skipping: path=%s", file_path)
                skipped_unreadable += 1
                continue

            if file_size > MAX_FILE_SIZE_BYTES:
                logger.debug(
                    "Skipping oversized file: name=%s size_kb=%d",
                    file_path.name,
                    file_size // 1024,
                )
                skipped_large += 1
                continue

            if file_size == 0:
                continue

            # Read content
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                logger.debug("Cannot read file — skipping: path=%s", file_path)
                skipped_unreadable += 1
                continue

            if not content.strip():
                continue

            # Build last-modified timestamp
            try:
                last_modified = datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).isoformat()
            except OSError:
                last_modified = ""

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "file_path": str(file_path.relative_to(local_path)),
                        "language":  SUPPORTED_EXTENSIONS[suffix],
                        "repo_name": repo_name,
                        "file_size": file_size,
                        "last_modified": last_modified,
                        "file_name": file_path.name,
                        "extension": suffix,
                    },
                )
            )

        logger.info(
            "Walk complete: accepted=%d skipped_large=%d skipped_unreadable=%d",
            len(documents),
            skipped_large,
            skipped_unreadable,
        )
        return documents


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_repo_id(source: str) -> str:
    """Return a stable 12-character hex ID for *source* (URL or path)."""
    return hashlib.md5(source.encode()).hexdigest()[:12]