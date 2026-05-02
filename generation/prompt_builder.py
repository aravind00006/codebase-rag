"""
prompt_builder.py — Construct grounded RAG prompts with inline source citations.

"""
import logging


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — immutable instruction set for the LLM
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """
You are an expert code analysis assistant. Your role is to answer \
questions about software codebases accurately and precisely.

CRITICAL RULES:
1. Answer ONLY using the provided code context below. Do not use external knowledge.
2. For EVERY factual claim, cite the exact source: `[Source: filename → function_name()]`
3. If the context is insufficient, say: "I don't have enough context to answer this accurately."
4. Format ALL code examples in markdown code blocks with the appropriate language tag.
5. Keep answers concise (under 300 words) unless the user explicitly requests detail.
6. If multiple sources contradict each other, note the discrepancy explicitly.

CITATION FORMAT:
  File-level:     [Source: path/to/file.py]
  Function-level: [Source: path/to/file.py → function_name()]
  Line-level:     [Source: path/to/file.py, Lines 42–67]
  """

def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into labelled context blocks for the LLM."""
    blocks: list[str] = []

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        file_path = meta.get("file_path", "unknown")
        node_name = meta.get("node_name", meta.get("functions", ""))
        start_line = meta.get("start_line", "?")
        end_line = meta.get("end_line", "?")
        language = meta.get("language", "")
        rerank_score = chunk.get("rerank_score", chunk.get("score", 0.0))

        header_parts = [f"[CHUNK {i + 1}]", f"Source: {file_path}"]
        if start_line != "?":
            header_parts.append(f"Lines: {start_line}–{end_line}")
        if node_name:
            header_parts.append(f"Function/Class: {node_name}")
        header_parts.append(f"Relevance: {rerank_score:.3f}")

        block = (
            " | ".join(header_parts)
            + f"\n```{language}\n{chunk['text']}\n```"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    retrieved_chunks: list[dict],
) -> tuple[str, str]:
    """
    Build the *(system_prompt, user_prompt)* pair for the LLM.

    Args:
        question:         The user's natural language question.
        retrieved_chunks: Re-ranked chunks from the retrieval pipeline.

    Returns:
        Tuple of ``(system_prompt, user_prompt)`` strings.
    """
    if not retrieved_chunks:
        logger.warning("build_prompt called with zero retrieved chunks")

    context_blocks = _format_context(retrieved_chunks)

    user_prompt = (
        "CODEBASE CONTEXT:\n"
        f"{context_blocks}\n\n"
        f"QUESTION: {question}\n\n"
        "Please answer the question using ONLY the context provided above. "
        "Cite your sources for every claim."
    )

    logger.debug(
        "Prompt built: chunks=%d context_chars=%d",
        len(retrieved_chunks),
        len(context_blocks),
    )

    return _SYSTEM_PROMPT, user_prompt