"""
ast_based.py — Strategy 3: AST-aware chunking at function/class boundaries.

Uses Python's built-in ``ast`` module to split source files at *semantic*
boundaries — one chunk per top-level function or class definition — so that
each chunk is always a complete, syntactically valid unit.

"""



import ast
import logging
import textwrap
from __future__ import annotations
from langchain_core.documents import Document
from ingestion.chunkers import recursive as recursive_chunker

logger = logging.getLogger(__name__)


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split Python documents at AST function/class boundaries.

    Non-Python files are handled by the recursive fallback chunker.

    """
    

    logger.info("AST-based chunking: documents=%d", len(documents))

    all_chunks: list[Document] = []
    fallback_docs: list[Document] = []

    for doc in documents:
        language = doc.metadata.get("language", "")
        if language != "python":
            fallback_docs.append(doc)
            continue

        chunks = _chunk_python(doc)
        if chunks is None:
            logger.warning(
                "AST parse failed — using recursive fallback: file=%s",
                doc.metadata.get("file_path", "unknown"),
            )
            fallback_docs.append(doc)
        else:
            all_chunks.extend(chunks)

    # Process non-Python and failed-parse files with recursive chunker
    if fallback_docs:
        logger.debug("Applying recursive fallback: files=%d", len(fallback_docs))
        fallback_chunks = recursive_chunker.chunk_documents(fallback_docs)

        # Re-tag so the ablation study still groups these under "ast"
        for chunk in fallback_chunks:
            chunk.metadata["chunk_strategy"] = "ast"
            chunk.metadata.setdefault("node_type", "fallback")

        all_chunks.extend(fallback_chunks)

    logger.info(
        "AST chunking complete: input_docs=%d output_chunks=%d fallback_files=%d",
        len(documents),
        len(all_chunks),
        len(fallback_docs),
    )
    return all_chunks

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_python(doc: Document) -> list[Document] | None:
    """
    Parse *doc* with ``ast`` and return one chunk per top-level node.

    Returns ``None`` if the file cannot be parsed (syntax error).
    """
    source = doc.page_content
    file_path = doc.metadata.get("file_path", "unknown")
    lines = source.splitlines(keepends=True)

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.debug(
            "SyntaxError in %s (line %s): %s", file_path, exc.lineno, exc.msg
        )
        return None

    chunks: list[Document] = []
    covered_lines: set[int] = set()

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start = node.lineno - 1   # convert to 0-indexed
        end = node.end_lineno     # exclusive upper bound for slicing

        node_source = "".join(lines[start:end])
        covered_lines.update(range(start, end))

        # .get() with fallback so an unexpected node type never crashes here
        node_type = {
            ast.FunctionDef:      "function",
            ast.AsyncFunctionDef: "async_function",
            ast.ClassDef:         "class",
        }.get(type(node), "unknown")

        # Grab first line of docstring only — keeps metadata compact
        docstring = ""
        try:
            raw = ast.get_docstring(node)
            if raw:
                docstring = raw.splitlines()[0][:120]
        except Exception:
            pass

        chunks.append(
            Document(
                page_content=textwrap.dedent(node_source),
                metadata={
                    **doc.metadata,
                    "chunk_strategy": "ast",
                    "node_name":      node.name,
                    "node_type":      node_type,
                    "start_line":     node.lineno,
                    "end_line":       node.end_lineno,
                    "docstring":      docstring,
                    "chunk_id":       f"{file_path}::{node.name}",
                },
            )
        )

    # Collect module-level code not inside any function or class
    module_lines = [
        line
        for i, line in enumerate(lines)
        if i not in covered_lines and line.strip()
    ]
    if module_lines:
        module_source = "".join(module_lines)
        if len(module_source.strip()) > 20:  # skip trivially short remnants
            chunks.append(
                Document(
                    page_content=module_source,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "ast",
                        "node_name":  "__module__",
                        "node_type":  "module_level",
                        "start_line": 1,
                        "end_line":   len(lines),
                        "docstring":  "",
                        "chunk_id":   f"{file_path}::__module__",
                    },
                )
            )

    logger.debug("AST chunks created: file=%s chunks=%d", file_path, len(chunks))
    return chunks if chunks else None