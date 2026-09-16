import sys
import types

import pytest

from docengine.chunker import Chunk
from docengine.llm import LLMError, build_prompt, format_context, generate_answer


def test_format_context_includes_pages():
    ctx = format_context([Chunk("alpha", 2), Chunk("beta", None)])
    assert ctx == "[1] (page 2) alpha\n\n[2] beta"


def test_prompt_contains_parts():
    p = build_prompt("Why?", "CTX")
    assert "Why?" in p and "CTX" in p and "ONLY" in p


def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        generate_answer("q", "c")


def test_unknown_provider():
    with pytest.raises(LLMError):
        generate_answer("q", "c", provider="gemini")


def test_empty_query(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with pytest.raises(LLMError, match="empty"):
        generate_answer("  ", "c")


def _fake_openai(monkeypatch, content=" answer ", raises=None):
    calls = {}

    class Completions:
        def create(self, **kw):
            calls.update(kw)
            if raises:
                raise raises
            msg = types.SimpleNamespace(content=content)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class OpenAI:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return calls


def test_openai_success(monkeypatch):
    calls = _fake_openai(monkeypatch)
    assert generate_answer("q?", "ctx") == "answer"
    assert calls["model"] == "gpt-4o-mini"
    assert calls["api_key"] == "test-key"
    assert "ctx" in calls["messages"][0]["content"]


def test_openai_none_content(monkeypatch):
    _fake_openai(monkeypatch, content=None)
    assert generate_answer("q?", "ctx") == ""


def test_openai_failure_wrapped(monkeypatch):
    _fake_openai(monkeypatch, raises=ConnectionError("network down"))
    with pytest.raises(LLMError, match="network down"):
        generate_answer("q?", "ctx")
