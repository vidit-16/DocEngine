# Answer model A/B

Retrieval fixed at hybrid, k=8. Same passages for every model.

| Model | Questions | Cites a gold page | Abstained | Off-document abstained | Errors | Median latency |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-4o-mini | 42 | 62% | 7% | 100% | 0 | 1.15s |
| gpt-4.1-mini | 42 | 64% | 0% | 100% | 0 | 1.18s |
| gpt-4.1-nano | 42 | 57% | 7% | 100% | 0 | 0.92s |

- **Cites a gold page**: the answer cites at least one page the evidence is on. It shows the answer is grounded where it should be, not that it is correct.
- **Off-document abstained**: questions the paper cannot answer. Anything below 100% means the model answered from outside knowledge.

Regenerate with `python evaluation/answer_ab.py`.
