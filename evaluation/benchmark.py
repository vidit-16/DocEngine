"""End-to-end accuracy benchmark over evaluation/gold/.

Retrieval (free, no API calls):
    python evaluation/benchmark.py retrieval --split dev

Answers (makes API calls, logged against a spending cap):
    python evaluation/benchmark.py answers --split dev

Retrieval is scored per question as recall@k: a hit when any of the top k
passages contains one of the question's evidence strings. Unanswerable
questions are left out of retrieval scoring.

Answers are scored on four things:
    correct         judged against the reference answer by a separate model call
    cites gold page the answer cites at least one page the evidence is on
    citation prec.  share of cited pages that are gold pages
    abstention      unanswerable questions should get the fixed "not found" reply

The dev split is for development; the test split is only for the final run.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.build_gold import DOCUMENTS  # noqa: E402
from src.chunker import Chunk, chunk_pages  # noqa: E402
from src.loader import load_pdf  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
GOLD = HERE / "gold"
RESULTS = HERE / "results"
LEDGER = CACHE / "spend.json"

K_VALUES = (1, 3, 5, 8)
SPEND_CAP_USD = 1.20
# USD per million tokens (input, output).
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}
ABSTAIN = "not clearly found in document"


# ── gold set ─────────────────────────────────────────────────────────────────


def load_gold(split: str) -> list[dict]:
    rows = []
    for path in sorted(GOLD.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if split == "all" or row["split"] == split:
                    rows.append(row)
    return rows


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def is_hit(chunk: Chunk, question: dict) -> bool:
    text = normalise(chunk.text)
    return any(normalise(ev) in text for ev in question["evidence"])


# ── documents and retrievers ─────────────────────────────────────────────────


@dataclass
class Document:
    name: str
    pages: list
    chunks: list[Chunk] = field(default_factory=list)
    state: dict = field(default_factory=dict)


def load_documents(chunker=chunk_pages) -> dict[str, Document]:
    docs = {}
    for name, filename in DOCUMENTS.items():
        pages = load_pdf((CACHE / filename).read_bytes())
        docs[name] = Document(name, pages, chunker(pages))
    return docs


def hybrid_retriever():
    """The retriever the app ships with: keyword + semantic, fused."""
    from src.embedder import embed
    from src.retriever import search
    from src.vector_store import create_index

    def prepare(doc: Document) -> None:
        doc.state["index"] = create_index(embed([c.text for c in doc.chunks]))

    def retrieve(doc: Document, query: str, k: int) -> list[Chunk]:
        return [r.chunk for r in search(query, doc.chunks, index=doc.state["index"], k=k)]

    return prepare, retrieve


RETRIEVERS = {"hybrid": hybrid_retriever}


# ── retrieval scoring ────────────────────────────────────────────────────────


def score_retrieval(docs: dict[str, Document], questions: list[dict], retriever: str) -> dict:
    prepare, retrieve = RETRIEVERS[retriever]()
    for doc in docs.values():
        prepare(doc)

    per_question = []
    depth = max(K_VALUES)
    for q in questions:
        if q["kind"] == "unanswerable":
            continue
        ranked = retrieve(docs[q["doc"]], q["question"], depth)
        first = next((i for i, c in enumerate(ranked, start=1) if is_hit(c, q)), None)
        per_question.append({"id": q["id"], "doc": q["doc"], "kind": q["kind"], "first_hit": first})

    def summarise(rows: list[dict]) -> dict:
        n = len(rows)
        out = {"n": n}
        for k in K_VALUES:
            out[f"recall@{k}"] = sum(r["first_hit"] is not None and r["first_hit"] <= k for r in rows) / n
        out["mrr"] = sum(1 / r["first_hit"] for r in rows if r["first_hit"]) / n
        return out

    groups = {"all": per_question}
    for key in ("kind", "doc"):
        for value in sorted({r[key] for r in per_question}):
            groups[value] = [r for r in per_question if r[key] == value]
    return {
        "summary": {name: summarise(rows) for name, rows in groups.items()},
        "misses": [r["id"] for r in per_question if r["first_hit"] is None],
    }


# ── spending ledger ──────────────────────────────────────────────────────────


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Accumulates API spend on disk so the cap holds across separate runs."""

    def __init__(self, cap: float = SPEND_CAP_USD) -> None:
        self.cap = cap
        self.data = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"usd": 0.0, "calls": 0}

    @property
    def spent(self) -> float:
        return self.data["usd"]

    def check(self) -> None:
        if self.spent >= self.cap:
            raise BudgetExceeded(f"spending cap reached: ${self.spent:.4f} of ${self.cap:.2f}")

    def record(self, model: str, usage) -> float:
        price_in, price_out = PRICES[model]
        cost = (usage.prompt_tokens * price_in + usage.completion_tokens * price_out) / 1_000_000
        self.data["usd"] += cost
        self.data["calls"] += 1
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(self.data))
        return cost


# ── answer scoring ───────────────────────────────────────────────────────────

_PAGE = re.compile(r"\bp(?:p|age)?\.?\s*(\d+)", re.IGNORECASE)

JUDGE_PROMPT = """You grade answers to questions about a document.

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Is the answer to grade correct? It is correct if it states the key facts of the
reference answer without contradicting it. Extra detail is fine if it is not
wrong. Missing the central fact, giving a different value, or declining to
answer is incorrect. Ignore page citations and wording.

Reply with JSON only: {{"correct": true or false, "reason": "<one short sentence>"}}"""


def cited_pages(answer: str) -> set[int]:
    return {int(n) for n in _PAGE.findall(answer)}


def is_abstention(answer: str) -> bool:
    return answer.strip().lower().rstrip(".") == ABSTAIN


def judge(client, ledger: Ledger, model: str, question: dict, answer: str) -> tuple[bool, str]:
    ledger.check()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question["question"], reference=question["answer"], answer=answer)}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    ledger.record(model, response.usage)
    verdict = json.loads(response.choices[0].message.content)
    return bool(verdict.get("correct")), verdict.get("reason", "")


def score_answers(docs, questions, retriever, answer_model, judge_model, generate) -> dict:
    """`generate(question, results) -> (answer, usage)` so answering strategies can be swapped."""
    from openai import OpenAI

    from src.retriever import DEFAULT_K

    client = OpenAI()
    ledger = Ledger()
    prepare, retrieve = RETRIEVERS[retriever]()
    for doc in docs.values():
        prepare(doc)

    rows = []
    for q in questions:
        ledger.check()
        chunks = retrieve(docs[q["doc"]], q["question"], DEFAULT_K)
        start = time.perf_counter()
        answer, usage = generate(q, chunks)
        seconds = time.perf_counter() - start
        if usage is not None:
            ledger.record(answer_model, usage)
        row = {"id": q["id"], "doc": q["doc"], "kind": q["kind"], "answer": answer, "seconds": seconds}
        if q["kind"] == "unanswerable":
            row["abstained"] = is_abstention(answer)
        else:
            gold = set(q["pages"])
            cited = cited_pages(answer)
            row["retrieved_gold"] = any(is_hit(c, q) for c in chunks)
            row["abstained"] = is_abstention(answer)
            row["cites_gold"] = bool(cited & gold)
            row["citation_precision"] = len(cited & gold) / len(cited) if cited else None
            row["correct"], row["judge_reason"] = (
                (False, "abstained") if row["abstained"] else judge(client, ledger, judge_model, q, answer)
            )
        rows.append(row)
        print(f"  {q['id']}: {'abstain' if row['abstained'] else ''} "
              f"{'correct' if row.get('correct') else ''} (${ledger.spent:.3f})")

    answerable = [r for r in rows if r["kind"] != "unanswerable"]
    unanswerable = [r for r in rows if r["kind"] == "unanswerable"]
    precisions = [r["citation_precision"] for r in answerable if r["citation_precision"] is not None]

    def rate(items, key):
        return sum(bool(r[key]) for r in items) / len(items) if items else 0.0

    by_kind = defaultdict(list)
    for r in answerable:
        by_kind[r["kind"]].append(r)
    return {
        "summary": {
            "answerable": len(answerable),
            "correct": rate(answerable, "correct"),
            "correct_when_retrieved": rate([r for r in answerable if r["retrieved_gold"]], "correct"),
            "retrieved_gold": rate(answerable, "retrieved_gold"),
            "cites_gold": rate(answerable, "cites_gold"),
            "citation_precision": statistics.mean(precisions) if precisions else 0.0,
            "false_abstention": rate(answerable, "abstained"),
            "unanswerable": len(unanswerable),
            "correct_abstention": rate(unanswerable, "abstained"),
            "median_seconds": statistics.median(r["seconds"] for r in rows),
            **{f"correct_{k}": rate(v, "correct") for k, v in by_kind.items()},
        },
        "spent_usd": ledger.spent,
        "rows": rows,
    }


def app_generate(model: str):
    from src.llm import generate_answer, get_client
    from src.retriever import Retrieved

    captured = {}
    client = get_client()
    original = client.chat.completions.create

    def create(**kwargs):
        response = original(**kwargs)
        captured["usage"] = response.usage
        return response

    client.chat.completions.create = create

    def generate(question: dict, chunks: list[Chunk]):
        captured.pop("usage", None)
        results = [Retrieved(chunk=c, score=0.0, keyword_rank=None, semantic_rank=None) for c in chunks]
        answer = generate_answer(question["question"], results, model=model)
        return answer, captured.get("usage")

    return generate


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["retrieval", "answers"])
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--retriever", default="hybrid", choices=sorted(RETRIEVERS))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--judge", default="gpt-4o-mini")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    questions = load_gold(args.split)
    docs = load_documents()
    RESULTS.mkdir(exist_ok=True)
    name = f"{args.mode}-{args.retriever}-{args.split}{'-' + args.label if args.label else ''}"

    if args.mode == "retrieval":
        result = score_retrieval(docs, questions, args.retriever)
    else:
        result = score_answers(docs, questions, args.retriever, args.model, args.judge,
                               app_generate(args.model))
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    if "misses" in result:
        print("misses:", " ".join(result["misses"]))


if __name__ == "__main__":
    main()
