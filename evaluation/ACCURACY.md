# Accuracy

How DocEngine's answers were measured, what was changed, and what the changes
did to the numbers. Everything here is reproducible with the scripts in this
folder.

## Result

Measured once, at the end, on a held-out test split of 70 questions that were
not used while making any decision below.

| | Before (previous `main`) | After | Change |
| --- | --- | --- | --- |
| **Correct answers** | 75.0% | **86.8%** | +11.8 pts |
| Correct, reworded questions | 69.7% | 84.8% | +15.1 |
| Correct, questions in the document's own words | 80.0% | 88.6% | +8.6 |
| Right passage among the 8 retrieved | 76.5% | 89.7% | +13.2 |
| Right passage ranked first | 48.5% | 54.4% | +5.9 |
| Answer cites a page the evidence is on | 80.9% | 86.8% | +5.9 |
| Citation precision (cited pages that are correct) | 79.0% | 83.6% | +4.6 |
| Wrongly answered "not found" | 2.9% | 1.5% | -1.4 |
| Declined questions the document cannot answer | 2 of 2 | 2 of 2 | |
| Median time per answer | 1.04 s | 1.19 s | +0.15 s |

Answers by `gpt-4o-mini` in both columns. Raw per-question results:
`results/final-test-before-main.json`, `results/final-test-after-branch.json`.

## The gold set

`gold/*.jsonl`, built and validated by `build_gold.py`.

| Document | Kind | Pages | Questions |
| --- | --- | --- | --- |
| Kingma & Ba, *Adam* (arXiv 1412.6980) | technical paper | 15 | 44 |
| Vaswani et al., *Attention Is All You Need* (arXiv 1706.03762) | technical paper, tables | 15 | 30 |
| NIST AI 100-1, *AI Risk Management Framework* | policy framework, landscape pages, tables | 48 | 32 |
| IPCC AR6 Synthesis Report, *Summary for Policymakers* | report, dense figures | 42 | 32 |

Each question has exact evidence strings, the pages they are on, a reference
answer, and a kind:

- **lexical**: asked in the document's own words (57)
- **paraphrase**: reworded away from the document's vocabulary (73)
- **unanswerable**: the document does not contain the answer (8)

`build_gold.py` refuses to write the set if an evidence string is missing from
the extracted text or found only on pages other than the ones labelled.
The split into **dev** (68) and **test** (70) is fixed by a hash of each
question id. All development used dev; test was run once, for the table above.

The documents are downloaded into `.cache/` and are not committed.

## How answers are judged

`benchmark.py answers` asks a separate model whether each answer states the
central fact of the reference answer without contradicting it. The judge is
checked against 62 answers labelled by hand (`judge_calibration.jsonl`):

| Judge | Agreement with hand labels | Disagreements |
| --- | --- | --- |
| `gpt-4o-mini` | 90.3% | 3 too lenient, 3 too strict |
| **`gpt-4.1-mini`** (used) | **93.5%** | 0 too lenient, 4 too strict |

On the calibration set the judge used never passed an answer marked wrong by
hand; its four errors were correct answers marked wrong (6.5% of answers). Its
errors therefore understate accuracy rather than inflate it.

## What changed, and what each change did

All figures below are on the dev split (62 answerable questions). On a set this
size one question is 1.6 points, so a difference of one or two questions is
noise; changes were kept only when they helped clearly or on several measures.

### 1. Text extraction

| Problem | Where | Fix |
| --- | --- | --- |
| Text set at 90 degrees came out reversed with no spaces ("gnitsettuoba") | NIST landscape figure pages | Rebuild rotated lines from character positions (`loader._rotated_text`) |
| Words hyphenated across line breaks ("man- agement") | 233 in NIST, 28 in Adam | Join or keep the hyphen using the document's own vocabulary (`loader.clean_pages`) |
| `(cid:NN)` placeholders for unmapped glyphs | Adam | Removed |

Garbled tokens across the four documents went from 6 to 0. Retrieval on NIST,
the most affected document: recall@8 67% -> 73%.

### 2. Retrieval

| Configuration (dev) | recall@1 | recall@8 | Paraphrase recall@8 | MRR |
| --- | --- | --- | --- | --- |
| Previous: MiniLM + keyword, 400-char windows | 45% | 77% | 68% | 0.57 |
| MiniLM, embeddings only (keyword search removed) | 42% | 69% | 60% | 0.52 |
| bge-small + keyword | 47% | 77% | 70% | 0.59 |
| e5-small + keyword | 48% | 81% | 75% | 0.60 |
| MiniLM + keyword + cross-encoder rerank | 58% | 79% | 72% | 0.65 |
| bge-small + keyword + rerank | 60% | 84% | 80% | 0.68 |
| ... with sentence chunks up to 400 chars | 66% | 77% | 70% | 0.70 |
| ... with sentence chunks up to 600 chars | 65% | 84% | 80% | 0.71 |
| ... with a 60-passage rerank pool instead of 30 | 65% | 84% | 80% | 0.71 |
| **... plus query expansion** (chosen) | 60% | **92%** | **88%** | 0.70 |
| ... expansion, reranked once against the original question | 65% | 84% | 80% | 0.71 |
| ... expansion, reranked once against the rewritten question | 47% | 85% | 80% | 0.60 |

Rows above the sentence-chunk rows were measured partly before and partly after
the extraction fixes in section 1; those fixes moved these figures by at most
one or two questions, so the comparisons hold.

- **Keyword search stays.** Removing it cost 8 points of recall@8.
- **The cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) was the
  largest single gain in putting the right passage first.
- **Sentence chunks** end on sentence boundaries instead of cutting sentences in
  half. Chunks of up to 400 characters lowered recall@8 by 7 points; up to 600
  kept recall and improved ranking.
- **Query expansion** (`llm.expand_query`) asks the model to rewrite the question
  as a standalone query in the document's terms and to write a hypothetical
  answer sentence, using the document's opening as context. Each phrasing is
  searched and reranked on its own and the rankings are fused. This fixed
  questions such as "How does *it* cope when the objective keeps changing?".
  Reranking all candidates once instead undid the gain: the cross-encoder scores
  passages against the vague original wording and rejects exactly what the
  rewrite found.

The app implements the chosen row in `src/retriever.search` (fusing all
phrasings in one step rather than pairwise, as the experiment did). Run through
the app's own code, dev recall@8 is 90% against the experiment's 92%, a
difference of one question.

### 3. Answers

| Configuration (dev, same judge) | Correct | Retrieved gold | Cites gold page | Citation precision | False "not found" |
| --- | --- | --- | --- | --- | --- |
| Previous pipeline | 80.6% | 77.4% | 83.9% | 82.6% | 8.1% |
| bge-small + rerank, 400-char windows | 77.4% | 83.9% | 87.1% | 83.1% | 3.2% |
| bge-small + rerank, sentence chunks 600 | 83.9% | 83.9% | 91.9% | 87.4% | 1.6% |
| + query expansion | 87.1% | 91.9% | 91.9% | 88.1% | 1.6% |
| **+ revised answer prompt** (chosen) | 85.5% | 91.9% | **96.8%** | **91.7%** | **0.0%** |
| + only 5 passages instead of 8 | 87.1% | 88.7% | 96.8% | 89.5% | 0.0% |

The revised prompt asks for a direct answer first, numbers and names exactly as
written, citations only for pages that support the claim, and care not to mix
values between similar things (a method and its variant, two scenarios). Its
correctness differs from the previous row by one question, which is noise; the
citation measures improved clearly, so it was kept. Passing 5 passages instead
of 8 lowered retrieval and citation precision without improving correctness.

## Remaining errors

On dev, the judge marked 9 of the final configuration's 62 answers wrong. Read
by hand, they fall into three groups:

- **2 wrong because the passage was not retrieved.** One answer is a sentence
  split across a page break (page boundaries are kept on purpose so citations
  stay honest); the other sits in a table whose columns extraction interleaves.
- **3 wrong with the right passage retrieved.** Temporal averaging described as
  "averaging the last iterate"; Adam's default settings listed alongside
  AdaMax's from a nearby passage, with ε omitted; and the wrong parser named
  from a table of results.
- **4 correct answers the judge marked wrong,** for omitting a secondary detail
  of the reference, adding a related figure, or writing "GtCO" where the
  reference has "GtCO2" (the subscript is lost in PDF extraction).

## Limits

- Four documents and 138 questions. Differences of a few points between close
  configurations are not significant; the before/after difference is.
- Questions were written with the documents open, which is an optimistic bias.
  The paraphrase questions reduce it but do not remove it.
- The test split has only 2 unanswerable questions; dev had 6. All 8 were
  declined correctly, but that is too few to estimate a rate.
- The previous pipeline's retrieval figures are slightly understated: its loader
  did not repair hyphenation, so a few evidence strings cannot match its text.
  Answer correctness, judged from the answer alone, is unaffected.
- Scanned PDFs are still not supported (no OCR).

## Reproducing

```bash
pip install -r requirements.txt
python evaluation/build_gold.py                      # downloads nothing; expects PDFs in .cache/
python evaluation/benchmark.py retrieval --split dev # free
python evaluation/benchmark.py answers --split dev   # API calls, capped at $1.20 in .cache/spend.json
```

The PDFs are fetched from the URLs listed in the table above into
`evaluation/.cache/` (`adam.pdf`, `attention.pdf`, `nist_ai_rmf.pdf`,
`ipcc_ar6_spm.pdf`). The whole study, including every configuration above and
the judge calibration, cost $0.23 in API calls.
