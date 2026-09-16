"""Answer generation.

The client is built on first use rather than at import, so importing this module
does not require OPENAI_API_KEY to be set. Previously the key was read at import
time and a missing one produced a failure far from its cause.

Context passages arrive labelled with their page number, and the prompt asks for
those page numbers in the answer. Without that, a grounded answer is
indistinguishable from a confident guess.
"""

from __future__ import annotations

import json
import logging
import os

from src.retriever import Retrieved

logger = logging.getLogger(__name__)

MODEL = os.getenv("DOCENGINE_MODEL", "gpt-4o-mini")

# The fixed reply the prompt asks for when the passages do not answer the question.
ABSTAIN = "Not clearly found in document"

SYSTEM_PROMPT = """You answer questions about a single document, using only the supplied passages.

Rules:
- Begin with a direct answer to exactly what was asked, in one or two sentences.
  Then add only the detail needed to support it.
- Use only the passages. Do not use outside knowledge.
- Cite the page for each claim, like (p. 4). Cite only pages whose passages
  support that claim.
- Keep numbers, units and names exactly as the passages state them.
- Passages often describe several similar things, such as a method and its
  variant, or different scenarios. Check that every value you give belongs to
  the thing the question asks about, and never mix them.
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
                "The answering service is not configured: OPENAI_API_KEY is not set."
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
        return ABSTAIN

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
        raise AnswerError(f"The answer could not be generated ({model}): {exc}") from exc

    return (response.choices[0].message.content or "").strip()


EXPAND_PROMPT = """You help search a document. Here is how the document begins:

{opening}

A reader asked: {question}

1. Rewrite the question as a standalone search query that names what "it", "this"
   or "they" refer to and uses the document's own terminology.
2. Write one sentence that could plausibly appear in the document and answer it.
   It does not need to be correct; it is only used to find similar passages.

Reply with JSON only: {{"query": "...", "passage": "..."}}"""


def expand_query(question: str, opening: str, model: str | None = None) -> list[str]:
    """Alternative phrasings of a question for retrieval: a rewrite and a hypothetical passage.

    Readers ask about "it" and use their own words; documents use their own
    terms. The opening of the document gives the model enough context to name
    the subject and borrow the document's vocabulary. On the gold set this
    raised recall@8 from 84% to 92% (see evaluation/ACCURACY.md).

    Retrieval must not depend on this call: any failure returns no expansions
    and the search proceeds with the original question alone.
    """
    if not question.strip():
        return []
    try:
        response = get_client().chat.completions.create(
            model=model or MODEL,
            messages=[{
                "role": "user",
                "content": EXPAND_PROMPT.format(opening=opening[:1200], question=question),
            }],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001 - expansion is an optimisation, never a failure
        logger.warning("query expansion skipped: %s", exc)
        return []
    return [
        str(data[key]).strip() for key in ("query", "passage") if str(data.get(key, "")).strip()
    ]
