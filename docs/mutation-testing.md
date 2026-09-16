# Mutation testing

`mutmut` needs `fork()` and does not run on Windows, so `scripts/mutation_test.py`
does the same job on a small scale. It walks the AST of each core module, applies
one mutation at a time (swap an arithmetic or comparison operator, nudge a numeric
constant, flip a unary sign), runs that module's tests in an isolated temporary
copy of the repository, and counts the mutant as killed if any test fails.

```bash
python scripts/mutation_test.py
```

| Module | Tests run per mutant |
| --- | --- |
| `docengine/chunker.py` | `tests/test_chunker.py` |
| `docengine/bm25.py` | `tests/test_bm25.py` |
| `docengine/retriever.py` | `tests/test_retriever.py` |

## Result

**68 of 74 mutants killed (91.9%).** The first run killed 67; the survivor
`chunk_size <= 0` → `< 0` still raised a `ValueError` (from the overlap check), so a
test now asserts on the message as well.

The remaining 6 survivors are equivalent mutants, which no test can kill because
the observable behaviour does not change:

| Location | Mutation | Why it is equivalent |
| --- | --- | --- |
| `bm25.py:23` | `avgdl` fallback `0.0` → `1.0` | Only used when there are no documents, and then nothing is scored |
| `retriever.py:23` | RRF start score `0.0` → `1.0` | Adds the same constant to every document, so the ranking is unchanged |
| `retriever.py:23` | RRF numerator `1.0` → `2.0` | Scales every score equally, so the ranking is unchanged |
| `retriever.py:35`, `:54` | FAISS sentinel `-1` → `-2` | `k` is clamped to the number of chunks, so FAISS never returns a padding id |
| `retriever.py:62` | `k <= 0` → `k < 0` | `k == 0` falls through to searches that also return an empty list |
