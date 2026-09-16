"""Answer generation.

The client is built on first use rather than at import, so importing this module
does not require OPENAI_API_KEY to be set. Previously the key was read at import
time and a missing one produced a failure far from its cause.

Context passages arrive labelled with their page number, and the prompt asks for
those page numbers in the answer. Without that, a grounded answer is
indistinguishable from a confident guess.
"""

from __future__ import annotations

import os

from src.retriever import Retrieved

MODEL = os.getenv("DOCENGINE_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You answer questions about a single document.

Rules:
- Use only the supplied passages. Do not use outside knowledge.
- Cite the page number for each claim, like (p. 4).
- If the passages do not contain the answer, say exactly:
  Not clearly found in document
- Do not reproduce long verbatim runs from the passages; summarise instead.
- Prefer short prose. Use bullets only for genuinely list-like answers."""

_client = None


class AnswerError(RuntimeError):
    """Raised when the model call fails."""


def get_client():
    global _client
    if _client is None:
        # Checked before the import so a missing key is reported as a missing
        # key, whatever else is wrong with the environment.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise AnswerError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file."
            )
        from openai import OpenAI

        _client = OpenAI(api_key=api_key)
    return _client


def set_client(client) -> None:
    """Install a client explicitly. Used by tests."""
    global _client
    _client = client


def build_context(results: list[Retrieved]) -> str:
    """Format retrieved chunks as page-labelled passages."""
    return "\n\n".join(
        f"[page {item.chunk.page}] {item.chunk.text}" for item in results
    )


def generate_answer(query: str, results: list[Retrieved], model: str | None = None) -> str:
    """Answer a question from retrieved passages."""
    if not results:
        return "Not clearly found in document"

    user_prompt = (
        f"Passages:\n\n{build_context(results)}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the passages above, citing page numbers."
    )

    model = model or MODEL
    try:
        response = get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
    except AnswerError:
        raise
    except Exception as exc:
        raise AnswerError(f"Model request failed ({model}): {exc}") from exc

    return (response.choices[0].message.content or "").strip()
