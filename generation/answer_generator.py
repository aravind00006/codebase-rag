"""
answer_generator.py — LLM call with streaming, retry logic, and observability.

"""

import logging
import os
import time
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_RETRY_WAIT_BASE: int = 5   # seconds — doubles on each attempt
_MAX_RETRIES: int = 3

def generate_answer(
    system_prompt: str,
    user_prompt: str,
    retrieved_chunks: list[dict],
    model: str = "gpt-4o-mini",
    stream: bool = False,
) -> dict:
    """
    Generate a grounded answer from the LLM with automatic retry on rate limits.

    """
    

    logger.info(
        "Generating answer: model=%s chunks=%d", model, len(retrieved_chunks)
    )

    llm = ChatOpenAI(
        model=model,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        streaming=False,
        temperature=0,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(_MAX_RETRIES):
        try:
            start = time.perf_counter()
            response = llm.invoke(messages)
            latency_ms = int((time.perf_counter() - start) * 1000)

            tokens_used = (
                response.usage_metadata.get("total_tokens", 0)
                if hasattr(response, "usage_metadata") and response.usage_metadata
                else 0
            )

            logger.info(
                "Answer generated: model=%s latency_ms=%d tokens=%d",
                model,
                latency_ms,
                tokens_used,
            )

            return {
                "answer": response.content,
                "sources": _extract_sources(retrieved_chunks),
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
                "model": model,
            }

        except Exception as exc:
            error_str = str(exc).lower()
            is_rate_limit = "rate limit" in error_str or "429" in error_str

            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                wait = _RETRY_WAIT_BASE * (2 ** attempt)
                logger.warning(
                    "Rate limit hit — retrying: attempt=%d/%d wait_s=%d",
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue

            logger.error(
                "LLM call failed: attempt=%d/%d error=%s",
                attempt + 1,
                _MAX_RETRIES,
                exc,
                exc_info=True,
            )
            raise

    raise RuntimeError(f"LLM call failed after {_MAX_RETRIES} retries")

def stream_answer(
    system_prompt: str,
    user_prompt: str,
    retrieved_chunks: list[dict],
    model: str = "gpt-4o-mini",
):
    """
    Stream answer tokens from the LLM as a generator of strings.

    Args:
        system_prompt:    Instruction prompt.
        user_prompt:      Context + question prompt.
        retrieved_chunks: Unused here — consumed by the caller after streaming.
        model:            OpenAI model identifier.

    Yields:
        Successive token strings from the LLM response stream.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    logger.debug("Starting streaming response: model=%s", model)

    llm = ChatOpenAI(
        model=model,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        streaming=True,
        temperature=0,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    token_count = 0
    for chunk in llm.stream(messages):
        if chunk.content:
            token_count += 1
            yield chunk.content

    logger.debug("Streaming complete: token_chunks_yielded=%d", token_count)