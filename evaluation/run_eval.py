"""Reproducible A/B evaluation of retrieval strategies and (optionally) LLMs.

Usage:
    python -m evaluation.run_eval               # retrieval grid only (no API calls)
    python -m evaluation.run_eval --llm         # + LLM A/B when OPENAI_API_KEY is set

Writes evaluation/results/retrieval.md (+ llm.md) and JSON copies.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from docengine.chunker import chunk_pages
from docengine.config import RETRIEVAL_MODES
from docengine.embedder import get_embeddings, load_model
from docengine.llm import LLMError, format_context, generate_answer
from docengine.loader import Page
from docengine.retriever import Retriever

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CHUNK_CONFIGS = [(300, 50), (500, 100), (800, 150)]
K_VALUES = (1, 3)

LLM_CANDIDATES = {
    "openai": ["gpt-4o-mini", "gpt-4.1-nano"],
}

log = logging.getLogger("eval")


def load_dataset() -> dict:
    return json.loads((HERE / "dataset.json").read_text(encoding="utf-8"))


def evaluate_retrieval(model, data: dict) -> list[dict]:
    pages = [Page(p["page"], p["text"]) for p in data["pages"]]
    questions = data["questions"]
    rows = []
    for size, overlap in CHUNK_CONFIGS:
        chunks = chunk_pages(pages, size, overlap, 50)
        retriever = Retriever(chunks, model, get_embeddings(model, [c.text for c in chunks]))
        for mode in RETRIEVAL_MODES:
            hits = {k: 0 for k in K_VALUES}
            rr = 0.0
            start = time.perf_counter()
            for item in questions:
                ranked = [c.page for c in retriever.search(item["q"], k=max(K_VALUES), mode=mode)]
                for k in K_VALUES:
                    hits[k] += item["page"] in ranked[:k]
                if item["page"] in ranked:
                    rr += 1.0 / (ranked.index(item["page"]) + 1)
            ms = (time.perf_counter() - start) * 1000 / len(questions)
            n = len(questions)
            rows.append(
                {
                    "chunk_size": size,
                    "overlap": overlap,
                    "chunks": len(chunks),
                    "mode": mode,
                    "hit@1": hits[1] / n,
                    "hit@3": hits[3] / n,
                    "mrr@3": rr / n,
                    "ms_per_query": ms,
                }
            )
    return rows


def evaluate_llms(model, data: dict, size: int, overlap: int, mode: str) -> list[dict]:
    pages = [Page(p["page"], p["text"]) for p in data["pages"]]
    retriever = Retriever(chunk_pages(pages, size, overlap, 50), model)
    rows = []
    for provider, models in LLM_CANDIDATES.items():
        key = "OPENAI_API_KEY"
        if not os.getenv(key):
            log.info("Skipping %s: %s not set", provider, key)
            continue
        for name in models:
            correct, errors, latency = 0, 0, 0.0
            for item in data["questions"]:
                context = format_context(retriever.search(item["q"], k=3, mode=mode))
                start = time.perf_counter()
                try:
                    answer = generate_answer(
                        item["q"], context, provider=provider, model=name, max_tokens=150
                    )
                except LLMError as exc:
                    log.warning("%s/%s failed: %s", provider, name, exc)
                    errors += 1
                    if errors >= 3 and correct == 0:
                        log.error("Aborting %s/%s after repeated failures", provider, name)
                        break
                    continue
                latency += time.perf_counter() - start
                correct += item["answer"].lower() in answer.lower()
            answered = len(data["questions"]) - errors
            if errors >= 3 and correct == 0:
                errors = len(data["questions"])
            rows.append(
                {
                    "provider": provider,
                    "model": name,
                    "accuracy": correct / len(data["questions"]),
                    "errors": errors,
                    "avg_latency_s": latency / answered if answered else 0.0,
                }
            )
    return rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "_no rows_\n"
    cols = list(rows[0])

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="also run the LLM A/B")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data = load_dataset()
    model = load_model()
    RESULTS.mkdir(exist_ok=True)

    rows = evaluate_retrieval(model, data)
    (RESULTS / "retrieval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    header = f"# Retrieval A/B ({len(data['questions'])} questions, {len(data['pages'])} pages)\n\n"
    (RESULTS / "retrieval.md").write_text(header + to_markdown(rows), encoding="utf-8")
    print(to_markdown(rows))

    if args.llm:
        best = max(rows, key=lambda r: (r["hit@3"], r["mrr@3"]))
        llm_rows = evaluate_llms(model, data, best["chunk_size"], best["overlap"], best["mode"])
        setup = (
            f"# LLM A/B\n\nRetrieval: {best['mode']}, chunk {best['chunk_size']}/"
            f"{best['overlap']}, k=3. Accuracy = expected answer string appears in the reply.\n\n"
        )
        llm_rows = [r for r in llm_rows if r["errors"] < len(data["questions"]) / 2]
        if not llm_rows:
            log.error("No LLM produced usable results (missing/invalid keys?); nothing written.")
            return
        (RESULTS / "llm.json").write_text(json.dumps(llm_rows, indent=2), encoding="utf-8")
        (RESULTS / "llm.md").write_text(setup + to_markdown(llm_rows), encoding="utf-8")
        print(to_markdown(llm_rows))


if __name__ == "__main__":
    main()
