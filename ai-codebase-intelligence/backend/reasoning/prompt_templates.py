"""Prompt templates and helpers for Week 6 reasoning flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from reasoning.llm_router import TaskType

TEMPLATE_VERSION = "1.0.0"
RESPONSE_WORD_MIN = 100
RESPONSE_WORD_MAX = 200


@dataclass
class ContextChunkForPrompt:
	file_path: str
	name: str
	display_name: str
	chunk_type: str
	start_line: int
	end_line: int
	score: float
	content: str
	docstring: str = ""
	language: str = "python"


@dataclass
class GraphContextForPrompt:
	call_chain: list[str] = field(default_factory=list)
	callers: list[str] = field(default_factory=list)
	callees: list[str] = field(default_factory=list)
	class_parents: list[str] = field(default_factory=list)
	related_files: list[str] = field(default_factory=list)


class BuiltPrompt(NamedTuple):
	system_prompt: str
	user_prompt: str
	task_type: TaskType
	template_version: str
	estimated_tokens: int


_SYSTEM_CODE_QA = """You are an expert software engineer and code analyst.
Answer using only the supplied code context and graph context.
If evidence is missing, state that explicitly and avoid guessing.
Always cite concrete files and lines when possible."""

_SYSTEM_REASONING = """You are a senior software architect.
Ground design claims in the supplied context.
Explain tradeoffs clearly and avoid unsupported assumptions."""

_SYSTEM_SECURITY = """You are a senior application security engineer.
Only report issues visible in the supplied context.
For each issue include risk, impact, and remediation."""

_SYSTEM_SUMMARIZE = """You are a technical documentation specialist.
Summarize only what appears in supplied context and keep it concise."""

_SYSTEM_ARCHITECTURE = """You are a principal software architect.
Explain architecture and relationships grounded in supplied evidence."""


_SYSTEM_PROMPTS: dict[TaskType, str] = {
	TaskType.CODE_QA: _SYSTEM_CODE_QA,
	TaskType.REASONING: _SYSTEM_REASONING,
	TaskType.SECURITY: _SYSTEM_SECURITY,
	TaskType.SUMMARIZE: _SYSTEM_SUMMARIZE,
	TaskType.ARCHITECTURE: _SYSTEM_ARCHITECTURE,
}

_RESPONSE_LENGTH_RULE = (
	f"Return the final answer in {RESPONSE_WORD_MIN}-{RESPONSE_WORD_MAX} words. "
	"Keep it specific and grounded in the supplied context."
)


def build_prompt(
	task_type: TaskType,
	question: str,
	context_chunks: list[ContextChunkForPrompt],
	graph_context: GraphContextForPrompt | None = None,
	repo_name: str = "the repository",
	history_block: str = "",
) -> BuiltPrompt:
	"""Build a fully structured prompt for LLM execution."""
	system_prompt = (
		_SYSTEM_PROMPTS.get(task_type, _SYSTEM_CODE_QA)
		+ "\n"
		+ _RESPONSE_LENGTH_RULE
	)
	context_block = _build_context_block(context_chunks)
	graph_block = _build_graph_block(graph_context) if graph_context else ""
	question_block = _build_question_block(question, task_type, repo_name)

	user_prompt = "\n\n".join(
		filter(None, [history_block, context_block, graph_block, question_block])
	)
	estimated_tokens = _estimate_tokens(system_prompt + user_prompt)

	return BuiltPrompt(
		system_prompt=system_prompt,
		user_prompt=user_prompt,
		task_type=task_type,
		template_version=TEMPLATE_VERSION,
		estimated_tokens=estimated_tokens,
	)


def _build_context_block(chunks: list[ContextChunkForPrompt]) -> str:
	if not chunks:
		return "<context>\nNo code context available.\n</context>"

	chunk_blocks: list[str] = []
	for chunk in chunks:
		content = _truncate_content(chunk.content, max_chars=3000)
		attrs = [
			f'file="{chunk.file_path}"',
			f'{chunk.chunk_type}="{chunk.display_name}"',
			f'lines="{chunk.start_line}-{chunk.end_line}"',
			f'language="{chunk.language}"',
			f'score="{chunk.score:.2f}"',
		]
		if chunk.docstring:
			summary = chunk.docstring.split("\n")[0][:120].strip()
			attrs.append(f'summary="{summary}"')

		chunk_blocks.append(f"<chunk {' '.join(attrs)}>\n{content}\n</chunk>")

	return "<context>\n" + "\n\n".join(chunk_blocks) + "\n</context>"


def _build_graph_block(graph: GraphContextForPrompt) -> str:
	if not any([graph.call_chain, graph.callers, graph.callees, graph.class_parents, graph.related_files]):
		return ""

	lines = ["<graph>"]

	if graph.call_chain:
		lines.append(f"  <call_chain>{' -> '.join(graph.call_chain)}</call_chain>")
	if graph.callers:
		lines.append(f"  <callers>Called by: {', '.join(graph.callers[:5])}</callers>")
	if graph.callees:
		lines.append(f"  <callees>Calls: {', '.join(graph.callees[:5])}</callees>")
	if graph.class_parents:
		lines.append(f"  <hierarchy>Inherits from: {' -> '.join(graph.class_parents)}</hierarchy>")
	if graph.related_files:
		lines.append(f"  <imports>Related files: {', '.join(graph.related_files[:5])}</imports>")

	lines.append("</graph>")
	return "\n".join(lines)


def _build_question_block(question: str, task_type: TaskType, repo_name: str) -> str:
	labels = {
		TaskType.CODE_QA: "Code Question",
		TaskType.REASONING: "Software Design Question",
		TaskType.SECURITY: "Security Analysis Request",
		TaskType.SUMMARIZE: "Summarization Request",
		TaskType.ARCHITECTURE: "Architecture Question",
	}
	label = labels.get(task_type, "Question")
	tag = label.lower().replace(" ", "_")
	return (
		f"<{tag}>\n"
		f"Repository: {repo_name}\n"
		f"Response length: {RESPONSE_WORD_MIN}-{RESPONSE_WORD_MAX} words\n\n"
		f"{question}\n"
		f"</{tag}>"
	)


def _truncate_content(content: str, max_chars: int) -> str:
	if len(content) <= max_chars:
		return content

	lines = content.splitlines()
	kept: list[str] = []
	total = 0
	for line in lines:
		if total + len(line) + 1 > max_chars:
			break
		kept.append(line)
		total += len(line) + 1

	truncated_lines = len(lines) - len(kept)
	note = f"\n... [{truncated_lines} lines truncated - content too large for context window]"
	return "\n".join(kept) + note


def _estimate_tokens(text: str) -> int:
	return len(text) // 4


def get_task_type_from_question(question: str) -> TaskType:
	q = (question or "").lower()

	security_keywords = {
		"security",
		"vulnerability",
		"exploit",
		"sql injection",
		"command injection",
		"ldap injection",
		"prompt injection",
		"xss",
		"csrf",
		"unsafe",
		"insecure",
		"attack",
	}
	summarize_keywords = {
		"summarise",
		"summarize",
		"overview",
		"what does",
		"purpose",
	}
	architecture_keywords = {
		"architecture",
		"design",
		"pattern",
		"structure",
		"high level",
		"components",
	}

	if any(keyword in q for keyword in security_keywords):
		return TaskType.SECURITY
	if any(keyword in q for keyword in architecture_keywords):
		return TaskType.ARCHITECTURE
	if any(keyword in q for keyword in summarize_keywords):
		return TaskType.SUMMARIZE
	return TaskType.CODE_QA


# Backward-compatible aliases used by older route code.
SYSTEM_PROMPT_CODE_QA = _SYSTEM_CODE_QA


def build_code_qa_prompt(question: str, context: str) -> str:
	return (
		"<context>\n"
		f"{context or 'No code context available.'}\n"
		"</context>\n\n"
		f"<code_question>\n{question}\n</code_question>"
	)
