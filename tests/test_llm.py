"""Answer generation and embedding.

The OpenAI client is stubbed: these tests check what we send and how we handle
what comes back, not whether the model is any good.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src import llm
from src.chunker import Chunk
from src.embedder import embed
from src.llm import AnswerError, build_context, generate_answer
from src.retriever import Retrieved


def make_results(*specs) -> list[Retrieved]:
    return [
        Retrieved(
            chunk=Chunk(text=text, page=page, index=position),
            score=1.0 / (position + 1),
            keyword_rank=position + 1,
            semantic_rank=None,
        )
        for position, (text, page) in enumerate(specs)
    ]


class FakeClient:
    def __init__(self, content="An answer (p. 2).", error=None):
        self.content = content
        self.error = error
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


@pytest.fixture
def client(monkeypatch):
    def install(content="An answer (p. 2).", error=None):
        fake = FakeClient(content, error)
        llm.set_client(fake)
        return fake

    yield install
    llm._client = None


# ── context construction ─────────────────────────────────────────────────────


def test_passages_are_labelled_with_their_page():
    context = build_context(make_results(("Revenue was 42.7 crore.", 4)))
    assert "[page 4]" in context
    assert "Revenue was 42.7 crore." in context


def test_every_passage_appears_in_the_context():
    context = build_context(make_results(("alpha", 1), ("beta", 2), ("gamma", 3)))
    for term in ("alpha", "beta", "gamma"):
        assert term in context


# ── generation ───────────────────────────────────────────────────────────────


def test_the_answer_is_returned_stripped(client):
    client(content="  An answer (p. 2).  ")
    assert generate_answer("q", make_results(("text", 2))) == "An answer (p. 2)."


def test_no_results_short_circuits_without_calling_the_model(client):
    fake = client()
    assert generate_answer("q", []) == "Not clearly found in document"
    assert fake.calls == []


def test_the_prompt_carries_the_question_and_the_passages(client):
    fake = client()
    generate_answer("What was revenue?", make_results(("Revenue was 42.7 crore.", 4)))

    messages = fake.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "What was revenue?" in messages[1]["content"]
    assert "[page 4]" in messages[1]["content"]


def test_the_system_prompt_constrains_the_model_to_the_passages(client):
    fake = client()
    generate_answer("q", make_results(("text", 1)))

    system = fake.calls[0]["messages"][0]["content"]
    assert "only the supplied passages" in system
    assert "Not clearly found in document" in system
    assert "Cite the page for each claim" in system


def test_temperature_is_low_for_an_extraction_task(client):
    fake = client()
    generate_answer("q", make_results(("text", 1)))
    assert fake.calls[0]["temperature"] <= 0.3


def test_an_empty_completion_does_not_crash(client):
    client(content=None)
    assert generate_answer("q", make_results(("text", 1))) == ""


def test_a_transport_failure_becomes_a_named_error(client):
    client(error=RuntimeError("connection reset"))
    with pytest.raises(AnswerError, match="could not be generated"):
        generate_answer("q", make_results(("text", 1)))


def test_a_missing_api_key_is_reported_clearly(monkeypatch):
    """The key used to be read at import, so a missing one failed far from here."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm._client = None
    with pytest.raises(AnswerError, match="not configured: OPENAI_API_KEY is not set"):
        generate_answer("q", make_results(("text", 1)))


# ── embedding ────────────────────────────────────────────────────────────────


def test_embeddings_are_unit_length():
    """Cosine similarity via inner product depends on this."""
    vectors = embed(["total revenue", "employee headcount"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embedding_nothing_returns_an_empty_array():
    assert embed([]).shape[0] == 0


def test_a_zero_vector_does_not_divide_by_zero():
    """Text with no scoreable tokens embeds to zeros; it must not become NaN."""
    vectors = embed(["the and of"])
    assert not np.isnan(vectors).any()


def test_the_model_is_not_loaded_at_import():
    """Importing must not download 90MB before any logic runs."""
    import importlib

    module = importlib.import_module("src.embedder")
    importlib.reload(module)
    assert module._model is None


def test_model_can_be_chosen_per_call(client):
    fake = client()
    generate_answer("q", make_results(("text", 1)), model="gpt-4.1-nano")
    assert fake.calls[-1]["model"] == "gpt-4.1-nano"
    generate_answer("q", make_results(("text", 1)))
    assert fake.calls[-1]["model"] == llm.MODEL
