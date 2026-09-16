"""A/B test of answer models on the same retrieved passages.

Retrieval is held fixed (the app's hybrid search, or keyword-only with
--skip-semantic) so the only thing that differs between rows is the model.
Each answer is scored on the two properties the prompt asks for:

  cites gold page   the answer cites at least one page the evidence is on
  abstained         the answer is the fixed "Not clearly found" sentence

Three off-document questions are added as a control. For those, abstaining is
the correct behaviour and anything else is an answer from outside knowledge.

Usage:

    set OPENAI_API_KEY (environment or .env), then
    python evaluation/answer_ab.py
    python evaluation/answer_ab.py --models gpt-4o-mini gpt-4.1-mini --limit 10

Writes evaluation/ANSWER_AB.md. This makes real API calls: with the defaults
that is about 45 calls per model on small models, a few cents in total.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import QUESTION_SETS, fetch_document, load_questions  # noqa: E402
from src.chunker import chunk_pages  # noqa: E402
from src.llm import AnswerError, generate_answer  # noqa: E402
from src.loader import load_pdf  # noqa: E402
from src.retriever import DEFAULT_K, search  # noqa: E402

ABSTAIN = "not clearly found in document"
DEFAULT_MODELS = ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano")
OFF_DOCUMENT = [
    {"id": "c01", "question": "What is the capital of Australia?", "pages": []},
    {"id": "c02", "question": "Who won the 2018 FIFA World Cup?", "pages": []},
    {"id": "c03", "question": "What is the usual adult dose of paracetamol?", "pages": []},
]

_PAGE = re.compile(r"\bp(?:p|age)?\.?\s*(\d+)", re.IGNORECASE)


def cited_pages(answer: str) -> set[int]:
    """Page numbers cited in an answer, e.g. '(p. 4)', 'page 12', 'pp. 3'."""
    return {int(number) for number in _PAGE.findall(answer)}


def score_answer(answer: str, question: dict) -> dict:
    abstained = answer.strip().lower().rstrip(".") == ABSTAIN
    gold = set(question["pages"])
    return {
        "abstained": abstained,
        "cites_gold": bool(gold) and not abstained and bool(cited_pages(answer) & gold),
    }


def summarise(
    model: str,
    answered: list[tuple[dict, dict]],
    controls: list[dict],
    errors: int,
    seconds: list[float],
) -> dict:
    n = len(answered)
    return {
        "model": model,
        "questions": n,
        "cites_gold": sum(s["cites_gold"] for _, s in answered) / n if n else 0.0,
        "abstained": sum(s["abstained"] for _, s in answered) / n if n else 0.0,
        "control_abstained": (
            sum(s["abstained"] for s in controls) / len(controls) if controls else 0.0
        ),
        "errors": errors,
        "median_s": statistics.median(seconds) if seconds else 0.0,
    }


def run_model(model: str, questions: list[dict], retrieve) -> dict:
    answered, controls, seconds, errors = [], [], [], 0
    for question in questions + OFF_DOCUMENT:
        results = retrieve(question["question"])
        start = time.perf_counter()
        try:
            answer = generate_answer(question["question"], results, model=model)
        except AnswerError as exc:
            errors += 1
            print(f"  {model} {question['id']}: {exc}")
            if errors >= 3 and not answered and not controls:
                raise SystemExit(f"{model}: repeated failures, stopping. Check the key.") from exc
            continue
        seconds.append(time.perf_counter() - start)
        scored = score_answer(answer, question)
        if question["pages"]:
            answered.append((question, scored))
        else:
            controls.append(scored)
    return summarise(model, answered, controls, errors, seconds)


def write_report(rows: list[dict], retrieval: str, path: Path) -> None:
    lines = [
        "# Answer model A/B",
        "",
        f"Retrieval fixed at {retrieval}, k={DEFAULT_K}. Same passages for every model.",
        "",
        "| Model | Questions | Cites a gold page | Abstained | Off-document abstained "
        "| Errors | Median latency |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['questions']} | {row['cites_gold']:.0%} "
            f"| {row['abstained']:.0%} | {row['control_abstained']:.0%} "
            f"| {row['errors']} | {row['median_s']:.2f}s |"
        )
    lines += [
        "",
        "- **Cites a gold page**: the answer cites at least one page the evidence is on. "
        "It shows the answer is grounded where it should be, not that it is correct.",
        "- **Off-document abstained**: questions the paper cannot answer. Anything "
        "below 100% means the model answered from outside knowledge.",
        "",
        "Regenerate with `python evaluation/answer_ab.py`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--limit", type=int, default=0, help="questions per set (0 = all)")
    parser.add_argument(
        "--skip-semantic", action="store_true", help="keyword-only retrieval, no embedding model"
    )
    parser.add_argument("--output", default=str(Path(__file__).parent / "ANSWER_AB.md"))
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    chunks = chunk_pages(load_pdf(fetch_document()))
    index = None
    if not args.skip_semantic:
        from src.embedder import embed
        from src.vector_store import create_index

        index = create_index(embed([chunk.text for chunk in chunks]))

    questions = []
    for path in QUESTION_SETS.values():
        loaded = load_questions(path, chunks)
        questions += loaded[: args.limit] if args.limit else loaded

    def retrieve(query):
        return search(query, chunks, index=index, k=DEFAULT_K)

    rows = []
    for model in args.models:
        print(f"{model}: {len(questions)} questions + {len(OFF_DOCUMENT)} controls")
        rows.append(run_model(model, questions, retrieve))
        print(f"  {rows[-1]}")

    retrieval = "keyword-only" if args.skip_semantic else "hybrid"
    write_report(rows, retrieval, Path(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
