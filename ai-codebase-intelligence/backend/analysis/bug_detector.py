"""Rule-based bug detector for quick repository diagnostics."""

from __future__ import annotations

import re


_RULES: list[tuple[str, str, str]] = [
	(
		r"except\s*:\s*\n\s*pass",
		"Swallowed exception",
		"Bare except with pass hides failures and makes debugging difficult.",
	),
	(
		r"except\s+Exception\s*:\s*\n\s*pass",
		"Generic exception swallowed",
		"Avoid catching Exception and ignoring it; log or re-raise with context.",
	),
	(
		r"assert\s+[^\n]+",
		"Assertion used in runtime path",
		"Assertions may be stripped with optimization flags; use explicit validation.",
	),
	(
		r"TODO|FIXME|HACK",
		"Pending fragile code marker",
		"Code contains TODO/FIXME/HACK marker that may indicate unfinished logic.",
	),
]


def detect_bug_patterns(source: str, file_path: str, symbol_name: str) -> list[dict]:
	findings: list[dict] = []
	text = source or ""
	for pattern, title, detail in _RULES:
		if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
			findings.append(
				{
					"title": title,
					"detail": detail,
					"file": file_path,
					"symbol": symbol_name,
					"severity": "medium",
				}
			)
	return findings
