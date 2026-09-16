from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_boots_without_upload():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value.endswith("Document Q&A Engine")
    assert [s.label for s in at.sidebar.selectbox] == ["Retrieval"]
    assert [t.label for t in at.sidebar.text_input] == ["OpenAI model"]


def test_app_reports_bad_config(monkeypatch):
    monkeypatch.setenv("DOCENGINE_TOP_K", "zero")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Invalid configuration" in at.error[0].value
