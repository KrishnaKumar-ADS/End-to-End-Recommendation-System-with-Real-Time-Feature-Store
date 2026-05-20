"""Unit tests for conversation/history_builder.py."""

from conversation.history_builder import build_history_block, _format_single_turn


def test_empty_turns_returns_empty_string():
    result = build_history_block([])
    assert result == ""


def test_single_user_turn_contains_role_and_content():
    turns = [{"role": "user", "content": "How does login work?", "timestamp": "2024-01-01T00:00:00Z"}]
    block = build_history_block(turns)

    assert "<conversation_history>" in block
    assert "</conversation_history>" in block
    assert 'role="user"' in block
    assert "How does login work?" in block


def test_assistant_turn_includes_sources():
    turns = [
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": "Answer here",
            "sources": [{"file": "auth.py", "function": "login"}],
            "timestamp": "2024-01-01T00:00:01Z",
        },
    ]
    block = build_history_block(turns)
    assert "auth.py:login" in block
    assert "[Sources:" in block


def test_format_single_turn_role_in_tag():
    turn = {"role": "user", "content": "Hello?", "timestamp": ""}
    result = _format_single_turn(turn)
    assert 'role="user"' in result
    assert "Hello?" in result
    assert "<turn" in result
    assert "</turn>" in result


def test_format_single_assistant_turn_no_sources():
    turn = {"role": "assistant", "content": "Answer text.", "timestamp": ""}
    result = _format_single_turn(turn)
    assert "[Sources:" not in result


def test_token_budget_drops_oldest_turns():
    turns = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Turn {i}: " + ("word " * 200),
        }
        for i in range(20)
    ]

    block = build_history_block(turns, max_tokens=500)

    assert "Turn 0:" not in block
    assert "Turn 18:" in block or "Turn 19:" in block


def test_history_block_has_correct_wrapper_tags():
    turns = [{"role": "user", "content": "Test"}]
    block = build_history_block(turns)
    assert block.startswith("<conversation_history>")
    assert block.strip().endswith("</conversation_history>")


def test_timestamp_appears_in_turn_tag():
    turns = [{"role": "user", "content": "Hi", "timestamp": "2024-06-01T12:00:00Z"}]
    block = build_history_block(turns)
    assert "2024-06-01T12:00:00Z" in block
