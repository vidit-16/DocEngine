"""Page styling for the Streamlit app.

The app had no styling at all, so the status line, the answer and the eight
source passages all carried the same weight and the page read as one column of
text. Colour is used to separate those three things and to mark the page
citations, which are the point of the app.

One accent (indigo), white canvas, hairline borders. Nothing here changes what
the pipeline returns.
"""

from __future__ import annotations

import html
import re

import streamlit as st

ACCENT = "#3b4d8f"

CSS = f"""
<style>
:root {{
  --accent: {ACCENT};
  --accent-tint: #eef1fa;
  --ink: #14181f;
  --muted: #5b6472;
  --surface: #f7f8f9;
  --hairline: #e5e7eb;
}}

/* Streamlit's own red-to-yellow bar sits above every page; make it ours. */
[data-testid="stDecoration"] {{
  background: linear-gradient(90deg, var(--accent), #5a6cc0);
}}

.stMain h1 {{
  padding-bottom: 0.3rem;
  border-bottom: 3px solid var(--accent);
  display: inline-block;
}}

/* The document is ready: a quiet accent strip, not a green alert. */
.doc-status {{
  display: flex;
  align-items: baseline;
  gap: 18px;
  border: 1px solid var(--hairline);
  border-left: 4px solid var(--accent);
  border-radius: 10px;
  background: var(--accent-tint);
  padding: 10px 16px;
  margin: 4px 0 18px;
}}
.doc-status-figure {{ font-weight: 700; color: var(--accent); }}
.doc-status-label {{ color: var(--muted); font-size: 13px; }}

/* The answer is the result, so it gets a card of its own. */
.answer-card {{
  border: 1px solid var(--hairline);
  border-left: 4px solid var(--accent);
  border-radius: 10px;
  padding: 16px 20px;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(20, 24, 31, 0.04);
}}
.answer-label {{
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); margin-bottom: 6px;
}}
.answer-body {{ color: var(--ink); font-size: 16px; line-height: 1.6; }}

/* Page citations, in the answer and beside each passage. */
.page-chip {{
  display: inline-block;
  background: var(--accent-tint);
  color: var(--accent);
  border: 1px solid #d7dcf2;
  border-radius: 999px;
  padding: 0 8px;
  font-size: 13px;
  font-weight: 600;
  white-space: nowrap;
}}

/* Each source passage as its own block, with the match reason set apart. */
.passage {{
  border: 1px solid var(--hairline);
  border-radius: 10px;
  padding: 12px 16px;
  margin-bottom: 10px;
  background: var(--surface);
}}
.passage-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
.passage-rank {{ color: var(--muted); font-size: 13px; font-weight: 600; }}
.passage-why {{ color: var(--muted); font-size: 12px; }}
.passage-text {{ color: var(--ink); font-size: 14px; line-height: 1.55; }}

[data-testid="stTable"] thead th {{
  background: var(--surface);
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  border-bottom: 2px solid var(--accent);
}}
</style>
"""

# "(p. 5)" or "(pp. 4-5)" as the answer prompt asks for them.
_CITATION = re.compile(r"\((pp?\.\s*[\d,\s–-]+)\)")


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def status(pages: int, passages: int) -> str:
    return (
        '<div class="doc-status">'
        f'<span><span class="doc-status-figure">{pages}</span>'
        ' <span class="doc-status-label">pages read</span></span>'
        f'<span><span class="doc-status-figure">{passages}</span>'
        ' <span class="doc-status-label">passages indexed</span></span>'
        '<span class="doc-status-label">Ask a question below.</span>'
        "</div>"
    )


def answer_card(answer: str) -> str:
    """The answer, with its page citations marked."""
    body = _CITATION.sub(
        lambda m: f'<span class="page-chip">{html.escape(m.group(1))}</span>',
        html.escape(answer),
    )
    return (
        '<div class="answer-card"><div class="answer-label">Answer</div>'
        f'<div class="answer-body">{body}</div></div>'
    )


def passage(rank: int, page: int, why: str, text: str) -> str:
    return (
        '<div class="passage"><div class="passage-head">'
        f'<span class="passage-rank">{rank}</span>'
        f'<span class="page-chip">Page {page}</span>'
        f'<span class="passage-why">{html.escape(why)}</span>'
        f'</div><div class="passage-text">{html.escape(text)}</div></div>'
    )
