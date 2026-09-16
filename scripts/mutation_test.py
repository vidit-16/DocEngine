"""Mutation testing for the retrieval core.

mutmut relies on fork() and does not run on Windows, so this does the same job
at a smaller scale. For each target module it walks the AST, applies one
mutation at a time (swap a comparison or arithmetic operator, nudge a numeric
constant, drop a `not`), runs the suite, and records whether any test failed.
A mutant that survives is a change to the code that no test noticed.

    python scripts/mutation_test.py
    python scripts/mutation_test.py --target src/chunker.py

The target file is always restored, including on Ctrl+C.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["src/chunker.py", "src/retriever.py", "src/loader.py", "src/llm.py"]

SWAPS: dict[type, type] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
}


def sites(tree: ast.AST) -> list[tuple[int, str]]:
    """Every mutable node, as (walk position, description)."""
    found = []
    for position, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.Compare) and type(node.ops[0]) in SWAPS:
            found.append((position, f"line {node.lineno}: {type(node.ops[0]).__name__}"))
        elif isinstance(node, ast.BinOp) and type(node.op) in SWAPS:
            found.append((position, f"line {node.lineno}: {type(node.op).__name__}"))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            found.append((position, f"line {node.lineno}: drop not"))
        elif (
            isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
            and hasattr(node, "lineno")
        ):
            found.append((position, f"line {node.lineno}: {node.value!r} + 1"))
    return found


class _Mutator(ast.NodeTransformer):
    def __init__(self, target: int, positions: dict[int, int]) -> None:
        self.target = target
        self.positions = positions

    def generic_visit(self, node):
        node = super().generic_visit(node)
        if self.positions.get(id(node)) != self.target:
            return node
        if isinstance(node, ast.Compare):
            node.ops[0] = SWAPS[type(node.ops[0])]()
        elif isinstance(node, ast.BinOp):
            node.op = SWAPS[type(node.op)]()
        elif isinstance(node, ast.UnaryOp):
            return node.operand
        elif isinstance(node, ast.Constant):
            node.value = node.value + 1
        return node


def mutate(source: str, target: int) -> str:
    tree = ast.parse(source)
    positions = {id(node): position for position, node in enumerate(ast.walk(tree))}
    return ast.unparse(_Mutator(target, positions).visit(tree))


def suite_passes() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", action="append", help="module to mutate (repeatable)")
    args = parser.parse_args()

    if not suite_passes():
        print("The suite fails before any mutation. Fix that first.")
        return 2

    killed, survivors = 0, []
    for relative in args.target or TARGETS:
        path = ROOT / relative
        original = path.read_text(encoding="utf-8")
        mutable = sites(ast.parse(original))
        try:
            for position, label in mutable:
                path.write_text(mutate(original, position), encoding="utf-8")
                if suite_passes():
                    survivors.append(f"{relative} {label}")
                else:
                    killed += 1
        finally:
            path.write_text(original, encoding="utf-8")
        print(f"{relative}: {len(mutable)} mutants")

    total = killed + len(survivors)
    print(f"\nMutation score: {killed}/{total} killed ({100 * killed / total:.1f}%)")
    for survivor in survivors:
        print(f"  survived: {survivor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
