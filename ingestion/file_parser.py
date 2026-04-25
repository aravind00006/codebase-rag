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

# ---------------------------------------------------------------------------
# Language-specific parsers (internal)
# ---------------------------------------------------------------------------

def _parse_python(content: str, metadata: dict) -> None:
    # use ast for accuracy, fall back to regex if the file has syntax errors
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metadata["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                metadata["classes"].append(node.name)
    except SyntaxError:
        logger.debug("AST parse failed — using regex fallback for Python file")
        _parse_generic(content, metadata)


def _parse_markdown(content: str, metadata: dict) -> None:
    # grab ATX headings (# through ######)
    for line in content.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+)", line)
        if m:
            metadata["headings"].append(m.group(2).strip())


def _parse_js_ts(content: str, metadata: dict) -> None:
    # js/ts has too many ways to define a function, so we need multiple patterns
    func_patterns = [
        r"\bfunction\s+(\w+)\s*\(",
        r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
        r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\w+\s*=>",
        r"async\s+function\s+(\w+)\s*\(",
        r"^\s*(\w+)\s*\([^)]*\)\s*\{",
    ]
    for p in func_patterns:
        metadata["functions"].extend(re.findall(p, content, re.MULTILINE))

    metadata["classes"].extend(re.findall(r"\bclass\s+(\w+)", content))

    # dedupe but keep order
    metadata["functions"] = list(dict.fromkeys(metadata["functions"]))
    metadata["classes"]   = list(dict.fromkeys(metadata["classes"]))


def _parse_java(content: str, metadata: dict) -> None:
    metadata["classes"].extend(
        re.findall(r"\b(?:class|interface|enum)\s+(\w+)", content)
    )
    metadata["functions"].extend(
        re.findall(
            r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(",
            content,
        )
    )


def _parse_go(content: str, metadata: dict) -> None:
    metadata["functions"].extend(
        re.findall(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", content, re.MULTILINE)
    )
    # structs are the closest thing go has to classes
    metadata["classes"].extend(
        re.findall(r"^type\s+(\w+)\s+struct", content, re.MULTILINE)
    )


def _parse_generic(content: str, metadata: dict) -> None:
    # last resort — works for most c-style langs
    metadata["functions"].extend(re.findall(r"\bfunction\s+(\w+)\s*\(", content))
    metadata["classes"].extend(re.findall(r"\bclass\s+(\w+)", content))