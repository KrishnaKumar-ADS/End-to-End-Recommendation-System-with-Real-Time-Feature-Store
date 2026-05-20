"""Utilities for normalizing LLM output into API response contracts."""

from __future__ import annotations


def clean_answer_text(answer: str) -> str:
	"""Normalize whitespace and ensure a non-empty answer body."""
	text = (answer or "").strip()
	if not text:
		return "I could not produce an answer from the available context."
	return text


def normalize_sources(sources: list[dict]) -> list[dict]:
	"""Return de-duplicated source list with required keys present."""
	seen: set[tuple[str, str | None, str | None]] = set()
	normalized: list[dict] = []

	for src in sources:
		file_path = src.get("file", "")
		fn = src.get("function")
		lines = src.get("lines")
		key = (file_path, fn, lines)
		if key in seen:
			continue
		seen.add(key)
		normalized.append(
			{
				"file": file_path,
				"function": fn,
				"lines": lines,
			}
		)

	return normalized
