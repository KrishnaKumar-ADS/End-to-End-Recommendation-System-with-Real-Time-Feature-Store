"""Unit tests for ask route word-limit helpers."""

from __future__ import annotations

from api.routes.ask import (
    _REPLY_WORD_MAX,
    _REPLY_WORD_MIN,
    _count_words,
    _ensure_fallback_word_floor,
    _trim_chunk_to_remaining_words,
    _truncate_to_max_words,
)


def test_truncate_to_max_words_caps_output():
    text = " ".join([f"word{i}" for i in range(260)])
    truncated = _truncate_to_max_words(text)
    assert _count_words(truncated) == _REPLY_WORD_MAX


def test_ensure_fallback_word_floor_reaches_minimum():
    short_text = "Providers are unavailable right now."
    expanded = _ensure_fallback_word_floor(short_text)
    assert _count_words(expanded) >= _REPLY_WORD_MIN


def test_trim_chunk_to_remaining_words_truncates_partial_chunk():
    chunk = "one two three four five"
    trimmed, emitted_words, hit_limit = _trim_chunk_to_remaining_words(chunk, remaining_words=3)

    assert emitted_words == 3
    assert hit_limit is True
    assert _count_words(trimmed) == 3
