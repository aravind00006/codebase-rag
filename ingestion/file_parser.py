# file_parser.py — adds language metadata to documents
# can give better citations like "defined in solve_dependencies()"

import re
import ast
import logging
from typing import Final
from __future__ import annotations
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

# maps extensions to broad categories
_DOC_TYPE_MAP: Final[dict[str, set[str]]] = {
    "code": {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".cpp", ".cc", ".c", ".go",
        ".rs", ".rb", ".php", ".sh",
    },
    "documentation": {".md", ".txt", ".rst"},
    "config": {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".env"},
}

def classify_doc_type(extension: str) -> str:
    # default to "code" if we don't recognize it
    for doc_type, exts in _DOC_TYPE_MAP.items():
        if extension in exts:
            return doc_type
    return "code"