"""Unit tests for reasoning.prompt_templates."""

import pytest

from reasoning.llm_router import TaskType
from reasoning.prompt_templates import (
    RESPONSE_WORD_MAX,
    RESPONSE_WORD_MIN,
    TEMPLATE_VERSION,
    BuiltPrompt,
    ContextChunkForPrompt,
    GraphContextForPrompt,
    _build_context_block,
    _build_graph_block,
    _truncate_content,
    build_prompt,
    get_task_type_from_question,
)


@pytest.fixture
def sample_chunk() -> ContextChunkForPrompt:
    return ContextChunkForPrompt(
        file_path="auth/service.py",
        name="verify_password",
        display_name="AuthService.verify_password",
        chunk_type="function",
        start_line=45,
        end_line=67,
        score=0.934,
        content="def verify_password(plain: str, hashed: str) -> bool:\n    return bcrypt.checkpw(plain.encode(), hashed.encode())",
        docstring="Verify plain password against hash.",
        language="python",
    )


@pytest.fixture
def sample_graph() -> GraphContextForPrompt:
    return GraphContextForPrompt(
        call_chain=["login_controller", "auth_service", "verify_password"],
        callers=["handle_login"],
        callees=["bcrypt.checkpw"],
        class_parents=[],
        related_files=["utils/crypto.py"],
    )


def test_template_version_semver_format():
    parts = TEMPLATE_VERSION.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_get_task_type_security_keyword():
    assert get_task_type_from_question("Find vulnerability in auth") == TaskType.SECURITY


def test_get_task_type_architecture_keyword():
    assert get_task_type_from_question("Explain architecture of auth") == TaskType.ARCHITECTURE


def test_get_task_type_summarize_keyword():
    assert get_task_type_from_question("Please summarize this module") == TaskType.SUMMARIZE


def test_get_task_type_defaults_to_code_qa():
    assert get_task_type_from_question("How does verify_password work?") == TaskType.CODE_QA


def test_get_task_type_dependency_injection_not_security():
    assert get_task_type_from_question("How does dependency injection work?") == TaskType.CODE_QA


def test_context_block_empty_list():
    block = _build_context_block([])
    assert "No code context available" in block


def test_context_block_contains_file_and_lines(sample_chunk):
    block = _build_context_block([sample_chunk])
    assert 'file="auth/service.py"' in block
    assert 'lines="45-67"' in block


def test_context_block_contains_score_and_content(sample_chunk):
    block = _build_context_block([sample_chunk])
    assert 'score="0.93"' in block
    assert "bcrypt.checkpw" in block


def test_context_block_multiple_chunks(sample_chunk):
    chunk2 = ContextChunkForPrompt(
        file_path="utils/crypto.py",
        name="hash_password",
        display_name="hash_password",
        chunk_type="function",
        start_line=10,
        end_line=20,
        score=0.82,
        content="def hash_password(x):\n    pass",
        language="python",
    )
    block = _build_context_block([sample_chunk, chunk2])
    assert "auth/service.py" in block
    assert "utils/crypto.py" in block


def test_graph_block_empty_is_blank():
    assert _build_graph_block(GraphContextForPrompt()) == ""


def test_graph_block_contains_call_chain(sample_graph):
    block = _build_graph_block(sample_graph)
    assert "login_controller" in block
    assert "verify_password" in block
    assert "->" in block


def test_graph_block_contains_callers_and_callees(sample_graph):
    block = _build_graph_block(sample_graph)
    assert "Called by" in block
    assert "Calls:" in block


def test_truncate_short_content_no_change():
    text = "def foo():\n    return 1"
    assert _truncate_content(text, max_chars=200) == text


def test_truncate_long_content_adds_note():
    text = "\n".join([f"line {i} {'x' * 50}" for i in range(100)])
    out = _truncate_content(text, max_chars=200)
    assert "truncated" in out.lower()
    assert len(out) < len(text)


def test_truncate_at_line_boundary():
    text = "line1\nline2\nline3\nline4\nline5"
    out = _truncate_content(text, max_chars=15)
    lines = [line for line in out.split("\n") if "truncated" not in line.lower() and line and not line.startswith("...")]
    for line in lines:
        assert line in text


def test_build_prompt_returns_built_prompt(sample_chunk, sample_graph):
    result = build_prompt(
        task_type=TaskType.CODE_QA,
        question="Why login fails?",
        context_chunks=[sample_chunk],
        graph_context=sample_graph,
        repo_name="my-app",
    )
    assert isinstance(result, BuiltPrompt)
    assert result.system_prompt
    assert result.user_prompt
    assert f"{RESPONSE_WORD_MIN}-{RESPONSE_WORD_MAX} words" in result.system_prompt


def test_build_prompt_includes_question(sample_chunk):
    q = "How does password hashing work?"
    result = build_prompt(TaskType.CODE_QA, q, [sample_chunk])
    assert q in result.user_prompt


def test_build_prompt_includes_code(sample_chunk):
    result = build_prompt(TaskType.CODE_QA, "test", [sample_chunk])
    assert "bcrypt.checkpw" in result.user_prompt


def test_build_prompt_includes_repo_name(sample_chunk):
    result = build_prompt(TaskType.CODE_QA, "test", [sample_chunk], repo_name="flask-backend")
    assert "flask-backend" in result.user_prompt
    assert f"Response length: {RESPONSE_WORD_MIN}-{RESPONSE_WORD_MAX} words" in result.user_prompt


def test_build_prompt_without_graph_excludes_graph_block(sample_chunk):
    result = build_prompt(TaskType.CODE_QA, "test", [sample_chunk], graph_context=None)
    assert "<graph>" not in result.user_prompt
