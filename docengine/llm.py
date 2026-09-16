"""Answer generation with OpenAI chat models."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from docengine.chunker import Chunk
from docengine.config import DEFAULT_MODELS, LLM_PROVIDERS

logger = logging.getLogger(__name__)

MODEL = DEFAULT_MODELS["openai"]
_KEY_ENV = {"openai": "OPENAI_API_KEY"}


class LLMError(RuntimeError):
    """Raised when an answer cannot be generated."""


def format_context(chunks: Sequence[Chunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        where = f" (page {c.page})" if c.page is not None else ""
        parts.append(f"[{i}]{where} {c.text}")
    return "\n\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    return f"""
Answer the question based ONLY on the context below.
If the answer is not in the context, say you don't know.

Be simple, clear, and structured.

Context:
{context}

Question:
{query}

Answer:
"""


def generate_answer(
    query: str,
    context: str,
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> str:
    if provider not in LLM_PROVIDERS:
        raise LLMError(f"Unknown LLM provider {provider!r}")
    if not query.strip():
        raise LLMError("Question is empty.")
    env = _KEY_ENV[provider]
    api_key = os.getenv(env)
    if not api_key:
        raise LLMError(f"{env} is not set")
    model = model or DEFAULT_MODELS[provider]
    prompt = build_prompt(query, context)

    try:
        from openai import OpenAI

        response = OpenAI(api_key=api_key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
    except LLMError:
        raise
    except Exception as exc:
        logger.error("%s request failed: %s", provider, exc)
        raise LLMError(f"{provider} request failed: {exc}") from exc

    return text.strip()
