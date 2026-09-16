"""Scoring for the answer-model A/B. No API calls: answers are plain strings."""

from __future__ import annotations

from evaluation.answer_ab import cited_pages, score_answer, summarise

QUESTION = {"id": "q", "question": "?", "pages": [2, 4]}


def test_cited_pages_accepts_common_forms():
    assert cited_pages("Defaults are given (p. 2) and again on page 4, pp. 9") == {2, 4, 9}
    assert cited_pages("No citation here") == set()


def test_gold_citation_counts():
    assert score_answer("Use 0.001 (p. 2).", QUESTION) == {"abstained": False, "cites_gold": True}


def test_wrong_page_does_not_count():
    assert score_answer("Use 0.001 (p. 7).", QUESTION)["cites_gold"] is False


def test_abstention_is_recognised_and_never_cites():
    scored = score_answer("Not clearly found in document.", QUESTION)
    assert scored == {"abstained": True, "cites_gold": False}


def test_control_question_has_no_gold_pages():
    control = {"id": "c", "question": "?", "pages": []}
    assert score_answer("Canberra (p. 2).", control) == {"abstained": False, "cites_gold": False}


def test_summarise_rates():
    answered = [
        (QUESTION, {"abstained": False, "cites_gold": True}),
        (QUESTION, {"abstained": True, "cites_gold": False}),
    ]
    controls = [
        {"abstained": True, "cites_gold": False},
        {"abstained": False, "cites_gold": False},
    ]
    row = summarise("m", answered, controls, errors=1, seconds=[1.0, 3.0])
    assert row["cites_gold"] == 0.5
    assert row["abstained"] == 0.5
    assert row["control_abstained"] == 0.5
    assert row["errors"] == 1
    assert row["median_s"] == 2.0
