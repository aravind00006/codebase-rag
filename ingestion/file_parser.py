# file_parser.py — adds language metadata to documents

import re
import ast
import logging
from typing import Final
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
    "config": {".yaml", ".yml", ".json", ".toml",
                ".ini", ".cfg", ".env"},
            }

def classify_doc_type(extension: str) -> str:
    # default to "code" if we don't recognize it
    for doc_type, exts in _DOC_TYPE_MAP.items():
        if extension in exts:
            return doc_type
    return "code"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_document(doc: Document) -> Document:
    # enriches a single doc with functions, classes, headings etc.
    # chromadb only takes strings so we flatten lists at the end
    ext = doc.metadata.get("extension", "")
    language = doc.metadata.get("language", "")
    content = doc.page_content

    metadata: dict = {**doc.metadata}
    metadata["doc_type"] = classify_doc_type(ext)
    metadata["functions"] = []
    metadata["classes"] = []
    metadata["headings"] = []

    parser_map = {
        "python":     _parse_python,
        "markdown":   _parse_markdown,
        "javascript": _parse_js_ts,
        "typescript": _parse_js_ts,
        "java":       _parse_java,
        "go":         _parse_go,
    }

    parser = parser_map.get(language, _parse_generic)
    parser(content, metadata)

    # flatten — chroma doesn't support list metadata values
    metadata["functions"] = ",".join(metadata["functions"][:20])
    metadata["classes"]   = ",".join(metadata["classes"][:10])
    metadata["headings"]  = ",".join(metadata["headings"][:10])

    logger.debug(
        "Parsed document: file=%s lang=%s functions=%d classes=%d",
        metadata.get("file_path", "unknown"),
        language,
        len(metadata["functions"].split(",")) if metadata["functions"] else 0,
        len(metadata["classes"].split(",")) if metadata["classes"] else 0,
    )
    return Document(page_content=content, metadata=metadata)


def parse_documents(docs: list[Document]) -> list[Document]:
    # runs parse_document on a list — failed docs get minimal metadata
    # we never drop docs silently, always keep them with fallback fields
    logger.info("Parsing %d documents", len(docs))
    parsed: list[Document] = []

    for doc in docs:
        try:
            parsed.append(parse_document(doc))
        except Exception:
            logger.warning(
                "Parse failed — falling back to minimal metadata: file=%s",
                doc.metadata.get("file_path", "unknown"),
                exc_info=True,
            )
            fallback = {**doc.metadata}
            fallback.setdefault("doc_type", classify_doc_type(fallback.get("extension", "")))
            fallback.setdefault("functions", "")
            fallback.setdefault("classes", "")
            fallback.setdefault("headings", "")
            parsed.append(Document(page_content=doc.page_content, metadata=fallback))

    logger.info("Document parsing complete: total=%d", len(parsed))
    return parsed

