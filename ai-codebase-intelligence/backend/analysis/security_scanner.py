"""Rule-based security scanner with CWE-tagged findings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"


@dataclass(frozen=True)
class SecurityRule:
	rule_id: str
	category: str
	description: str
	cwe_id: str
	severity: Severity
	pattern: str


@dataclass
class SecurityFinding:
	file_path: str
	function: str
	line_number: int
	severity: Severity
	category: str
	description: str
	matched_text: str
	rule_id: str
	cwe_id: str
	llm_analysis: str = ""
	false_positive: bool = False


_SEVERITY_WEIGHT = {
	Severity.CRITICAL: 4,
	Severity.HIGH: 3,
	Severity.MEDIUM: 2,
	Severity.LOW: 1,
}


class SecurityScanner:
	"""Static scanner for common application security issues."""

	def __init__(self) -> None:
		self._rules = self._build_rules()
		self._compiled = [
			(rule, re.compile(rule.pattern, re.IGNORECASE))
			for rule in self._rules
		]

	def scan_chunk(
		self,
		content: str,
		file_path: str,
		function_name: str,
		chunk_offset: int,
	) -> list[SecurityFinding]:
		findings: list[SecurityFinding] = []
		lines = (content or "").splitlines() or [content or ""]

		for index, line in enumerate(lines):
			line_number = chunk_offset + index
			for rule, regex in self._compiled:
				match = regex.search(line)
				if not match:
					continue

				findings.append(
					SecurityFinding(
						file_path=file_path,
						function=function_name,
						line_number=line_number,
						severity=rule.severity,
						category=rule.category,
						description=rule.description,
						matched_text=(line.strip()[:200]),
						rule_id=rule.rule_id,
						cwe_id=rule.cwe_id,
					)
				)

		return findings

	def scan_chunks_batch(self, chunks: list[dict]) -> list[SecurityFinding]:
		all_findings: list[SecurityFinding] = []
		for chunk in chunks:
			all_findings.extend(
				self.scan_chunk(
					content=chunk.get("content", ""),
					file_path=chunk.get("file_path", ""),
					function_name=chunk.get("name", ""),
					chunk_offset=chunk.get("start_line", 1),
				)
			)

		all_findings.sort(
			key=lambda finding: (
				-_SEVERITY_WEIGHT[finding.severity],
				finding.file_path,
				finding.line_number,
				finding.rule_id,
			)
		)
		return all_findings

	def get_rules_summary(self) -> list[dict]:
		return [
			{
				"rule_id": rule.rule_id,
				"category": rule.category,
				"description": rule.description,
				"severity": rule.severity.value,
				"cwe_id": rule.cwe_id,
			}
			for rule in self._rules
		]

	@staticmethod
	def _build_rules() -> list[SecurityRule]:
		return [
			SecurityRule(
				rule_id="SEC001",
				category="SQL_INJECTION",
				description="Potential SQL injection via dynamic query composition.",
				cwe_id="CWE-89",
				severity=Severity.CRITICAL,
				pattern=r"(SELECT|INSERT|UPDATE|DELETE).*(\{.+\}|\+\s*\w+|%\s*\w+)",
			),
			SecurityRule(
				rule_id="SEC002",
				category="HARDCODED_SECRET",
				description="Hardcoded secret or password found in source.",
				cwe_id="CWE-798",
				severity=Severity.HIGH,
				pattern=r"(password|secret|api[_-]?key|token)\s*=\s*['\"][^'\"]{4,}['\"]",
			),
			SecurityRule(
				rule_id="SEC003",
				category="UNSAFE_DESERIALIZATION",
				description="Unsafe deserialization primitive detected.",
				cwe_id="CWE-502",
				severity=Severity.HIGH,
				pattern=r"(pickle\.loads|yaml\.load\(|marshal\.loads)",
			),
			SecurityRule(
				rule_id="SEC004",
				category="COMMAND_INJECTION",
				description="Command execution with shell=True can lead to injection.",
				cwe_id="CWE-78",
				severity=Severity.CRITICAL,
				pattern=r"subprocess\.(run|Popen|call)\([^\n]*shell\s*=\s*True",
			),
			SecurityRule(
				rule_id="SEC005",
				category="PATH_TRAVERSAL",
				description="Potential path traversal from unvalidated user input.",
				cwe_id="CWE-22",
				severity=Severity.HIGH,
				pattern=r"(open\(|Path\().*(\.\./|request\.|params\[|input\()",
			),
			SecurityRule(
				rule_id="SEC006",
				category="WEAK_CRYPTOGRAPHY",
				description="Weak cryptographic hash function in security-sensitive code.",
				cwe_id="CWE-327",
				severity=Severity.HIGH,
				pattern=r"hashlib\.(md5|sha1)\(",
			),
			SecurityRule(
				rule_id="SEC007",
				category="INSECURE_RANDOM",
				description="Non-cryptographic random source used for security-sensitive value.",
				cwe_id="CWE-338",
				severity=Severity.MEDIUM,
				pattern=r"random\.(random|randint|choice|choices)\(",
			),
			SecurityRule(
				rule_id="SEC008",
				category="DEBUG_MODE",
				description="Debug mode should not be enabled in production.",
				cwe_id="CWE-489",
				severity=Severity.MEDIUM,
				pattern=r"(debug\s*=\s*True|FLASK_DEBUG\s*=\s*['\"]?1)",
			),
			SecurityRule(
				rule_id="SEC009",
				category="JWT_WITHOUT_VERIFY",
				description="JWT decode without signature verification detected.",
				cwe_id="CWE-347",
				severity=Severity.CRITICAL,
				pattern=r"jwt\.decode\([^\n]*verify_signature\s*=\s*False",
			),
			SecurityRule(
				rule_id="SEC010",
				category="INSECURE_CORS",
				description="Overly permissive CORS configuration.",
				cwe_id="CWE-942",
				severity=Severity.MEDIUM,
				pattern=r"allow_origins\s*=\s*\[\s*['\"]\*['\"]\s*\]",
			),
			SecurityRule(
				rule_id="SEC011",
				category="XSS_RAW_HTML",
				description="Raw HTML rendering with user input can enable XSS.",
				cwe_id="CWE-79",
				severity=Severity.HIGH,
				pattern=r"(mark_safe\(|innerHTML\s*=|render_template_string\()",
			),
			SecurityRule(
				rule_id="SEC012",
				category="SSRF",
				description="Network request appears to use untrusted URL input.",
				cwe_id="CWE-918",
				severity=Severity.HIGH,
				pattern=r"requests\.(get|post|put|delete)\([^\n]*(request\.|input\(|params\[)",
			),
			SecurityRule(
				rule_id="SEC013",
				category="TEMPLATE_INJECTION",
				description="Template rendering with dynamic templates detected.",
				cwe_id="CWE-1336",
				severity=Severity.HIGH,
				pattern=r"(Template\(|Environment\().*(request\.|input\(|params\[)",
			),
			SecurityRule(
				rule_id="SEC014",
				category="OPEN_REDIRECT",
				description="Open redirect using unvalidated destination input.",
				cwe_id="CWE-601",
				severity=Severity.MEDIUM,
				pattern=r"(redirect\(|Response\.redirect).*(request\.|params\[|query_params)",
			),
			SecurityRule(
				rule_id="SEC015",
				category="XXE",
				description="XML parser configured in unsafe mode (possible XXE).",
				cwe_id="CWE-611",
				severity=Severity.HIGH,
				pattern=r"(xml\.etree|lxml).*resolve_entities\s*=\s*True",
			),
		]


def scan_security_patterns(source: str, file_path: str, symbol_name: str) -> list[dict]:
	"""Backward-compatible helper used by legacy analysis route."""
	scanner = SecurityScanner()
	findings = scanner.scan_chunk(
		content=source,
		file_path=file_path,
		function_name=symbol_name,
		chunk_offset=1,
	)
	return [
		{
			"title": finding.category,
			"detail": finding.description,
			"file": finding.file_path,
			"symbol": finding.function,
			"severity": finding.severity.value,
		}
		for finding in findings
	]
