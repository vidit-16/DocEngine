"""Lightweight AST mutation testing (mutmut needs fork() and does not run on Windows).

Each mutant flips one operator / constant / comparison in a core module, runs the
relevant tests in an isolated temp copy of the repo, and counts it as killed if the
tests fail. Usage:  python scripts/mutation_test.py
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "docengine/chunker.py": ["tests/test_chunker.py"],
    "docengine/bm25.py": ["tests/test_bm25.py"],
    "docengine/retriever.py": ["tests/test_retriever.py"],
}

BINOP_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
CMP_SWAP = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


def mutation_sites(tree: ast.AST) -> list[tuple[ast.AST, str]]:
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in BINOP_SWAP:
            sites.append((node, "binop"))
        elif isinstance(node, ast.Compare) and type(node.ops[0]) in CMP_SWAP:
            sites.append((node, "compare"))
        elif (
            isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
            and not isinstance(node.value, bool)
        ):
            sites.append((node, "const"))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            sites.append((node, "neg"))
    return sites


def apply(node: ast.AST, kind: str) -> str:
    if kind == "binop":
        old = type(node.op)
        node.op = BINOP_SWAP[old]()
        return f"{old.__name__}->{type(node.op).__name__}"
    if kind == "compare":
        old = type(node.ops[0])
        node.ops[0] = CMP_SWAP[old]()
        return f"{old.__name__}->{type(node.ops[0]).__name__}"
    if kind == "const":
        old = node.value
        node.value = old + 1
        return f"{old}->{node.value}"
    node.op = ast.UAdd()
    return "-x->+x"


def run_tests(workdir: Path, tests: list[str]) -> bool:
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests]
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, timeout=600)
    return result.returncode == 0


def main() -> int:
    total = killed = 0
    survivors = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(ROOT / "docengine", work / "docengine")
        shutil.copytree(ROOT / "tests", work / "tests")
        shutil.copy(ROOT / "pyproject.toml", work / "pyproject.toml")
        for rel, tests in TARGETS.items():
            original = (ROOT / rel).read_text(encoding="utf-8")
            count = len(mutation_sites(ast.parse(original)))
            for i in range(count):
                tree = ast.parse(original)
                node, kind = mutation_sites(tree)[i]
                desc = apply(node, kind)
                (work / rel).write_text(ast.unparse(tree), encoding="utf-8")
                passed = run_tests(work, tests)
                total += 1
                if passed:
                    survivors.append(f"{rel}:{node.lineno} {desc}")
                else:
                    killed += 1
                print(f"[{'SURVIVED' if passed else 'killed  '}] {rel}:{node.lineno} {desc}")
            (work / rel).write_text(original, encoding="utf-8")
    score = killed / total if total else 0.0
    print(f"\nMutation score: {killed}/{total} = {score:.1%}")
    for s in survivors:
        print("  survivor:", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
