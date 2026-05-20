"""Helpers for turning retrieval hits into prompt context and source metadata."""

from __future__ import annotations


def _format_lines(start_line: int | None, end_line: int | None) -> str | None:
	if start_line and end_line and start_line > 0 and end_line >= start_line:
		return f"{start_line}-{end_line}"
	return None


def build_context_and_sources(results: list[dict]) -> tuple[str, list[dict]]:
	"""
	Build a compact context string for the LLM and a normalized `sources` list.

	Args:
		results: retrieval hits with fields like file_path, display_name, name,
				 content_preview, start_line, end_line.

	Returns:
		(context_text, sources)
	"""
	if not results:
		return "", []

	context_parts: list[str] = []
	sources: list[dict] = []

	for idx, item in enumerate(results, start=1):
		file_path = item.get("file_path", "")
		name = item.get("display_name") or item.get("name") or "unknown_symbol"
		preview = (item.get("content_preview") or "").strip()
		lines = _format_lines(item.get("start_line"), item.get("end_line"))

		context_parts.append(
			"\n".join(
				[
					f"[Source {idx}]",
					f"File: {file_path}",
					f"Symbol: {name}",
					f"Lines: {lines or 'unknown'}",
					f"Snippet: {preview}",
				]
			)
		)

		sources.append(
			{
				"file": file_path,
				"function": name,
				"lines": lines,
			}
		)

	return "\n\n".join(context_parts), sources
