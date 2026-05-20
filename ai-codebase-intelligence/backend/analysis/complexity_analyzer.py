"""Lightweight complexity analysis helpers."""

from __future__ import annotations

import ast


_BRANCH_NODES = (
	ast.If,
	ast.For,
	ast.AsyncFor,
	ast.While,
	ast.Try,
	ast.ExceptHandler,
	ast.With,
	ast.AsyncWith,
	ast.BoolOp,
	ast.IfExp,
	ast.Match,
)


def cyclomatic_complexity(source: str) -> int:
	"""Approximate McCabe complexity: 1 + number of branch/decision nodes."""
	try:
		tree = ast.parse(source or "")
	except SyntaxError:
		return 1

	decisions = sum(1 for node in ast.walk(tree) if isinstance(node, _BRANCH_NODES))
	return 1 + decisions


def analyze_chunk_complexity(name: str, file_path: str, source: str) -> dict:
	complexity = cyclomatic_complexity(source)
	risk = "low"
	if complexity >= 20:
		risk = "high"
	elif complexity >= 10:
		risk = "medium"

	return {
		"name": name,
		"file": file_path,
		"cyclomatic_complexity": complexity,
		"risk": risk,
	}
