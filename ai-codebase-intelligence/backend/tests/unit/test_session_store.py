"""Unit tests for conversation/session_store.py."""

import pytest
from unittest.mock import patch

from conversation.session_store import SessionStore


@pytest.fixture
def fake_redis():
    import fakeredis

    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def store(fake_redis):
    with patch("conversation.session_store._get_redis_client", return_value=fake_redis):
        yield SessionStore()


def test_create_session_returns_uuid(store):
    session_id = store.create_session()
    assert len(session_id) == 36
    assert session_id.count("-") == 4


def test_create_session_each_call_unique(store):
    ids = [store.create_session() for _ in range(10)]
    assert len(set(ids)) == 10


def test_append_and_get_user_turn(store):
    sid = store.create_session()
    result = store.append_turn(sid, role="user", content="How does login work?")
    assert result is True

    turns = store.get_turns(sid)
    assert len(turns) == 1
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == "How does login work?"
    assert "timestamp" in turns[0]


def test_append_assistant_turn_with_sources(store):
    sid = store.create_session()
    store.append_turn(sid, role="user", content="Question")
    store.append_turn(
        sid,
        role="assistant",
        content="The answer is...",
        sources=[{"file": "auth.py", "function": "login"}],
        provider_used="openrouter",
        model_used="qwen/qwen-2.5-coder-32b-instruct",
    )

    turns = store.get_turns(sid)
    assert len(turns) == 2
    assistant = turns[1]
    assert assistant["role"] == "assistant"
    assert assistant["sources"] == [{"file": "auth.py", "function": "login"}]
    assert assistant["provider_used"] == "openrouter"
    assert assistant["model_used"] == "qwen/qwen-2.5-coder-32b-instruct"


def test_append_invalid_role_returns_false(store):
    sid = store.create_session()
    result = store.append_turn(sid, role="system", content="You are a bot.")
    assert result is False


def test_get_turns_missing_session_returns_empty(store):
    assert store.get_turns("nonexistent-uuid-1234") == []


def test_session_exists_after_first_turn(store):
    sid = store.create_session()
    assert not store.session_exists(sid)

    store.append_turn(sid, role="user", content="Hello")
    assert store.session_exists(sid)


def test_delete_session_removes_data(store):
    sid = store.create_session()
    store.append_turn(sid, role="user", content="Hello")
    assert store.session_exists(sid)

    deleted = store.delete_session(sid)
    assert deleted is True
    assert not store.session_exists(sid)
    assert store.get_turns(sid) == []


def test_delete_nonexistent_session_returns_false(store):
    assert store.delete_session("does-not-exist") is False


def test_turn_count(store):
    sid = store.create_session()
    assert store.turn_count(sid) == 0

    store.append_turn(sid, role="user", content="Q1")
    store.append_turn(sid, role="assistant", content="A1")
    store.append_turn(sid, role="user", content="Q2")
    assert store.turn_count(sid) == 3


def test_long_content_is_truncated(store):
    sid = store.create_session()
    content = "x" * 10_000
    store.append_turn(sid, role="assistant", content=content)

    turns = store.get_turns(sid)
    assert len(turns[0]["content"]) < 10_000
    assert turns[0]["content"].endswith("[truncated]")


def test_session_accumulates_multiple_turns(store):
    sid = store.create_session()
    for i in range(5):
        store.append_turn(sid, role="user", content=f"Question {i}")
        store.append_turn(sid, role="assistant", content=f"Answer {i}")

    turns = store.get_turns(sid)
    assert len(turns) == 10
    assert turns[0]["content"] == "Question 0"
    assert turns[-1]["content"] == "Answer 4"
