"""
Unit tests for the Gemini embedder.

All Gemini API calls are mocked — these tests run with no API key
and make no network requests.
"""

import pytest
from unittest.mock import patch, MagicMock
import embeddings.gemini_embedder as gemini_embedder
from embeddings.gemini_embedder import (
    embed_text,
    embed_query,
    embed_chunk,
    embed_chunks_batch,
    _prepare_text,
    _build_embedding_text,
    VECTOR_DIMENSIONS,
    TASK_TYPE_DOCUMENT,
    TASK_TYPE_QUERY,
    MAX_INPUT_CHARS,
)
from core.exceptions import EmbeddingError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_embed(*args, **kwargs):
    """Return a fake 768-dimensional vector."""
    return {"embedding": [0.1] * VECTOR_DIMENSIONS}


SAMPLE_CHUNK = {
    "id": "chunk-abc-123",
    "chunk_type": "function",
    "name": "verify_password",
    "display_name": "AuthService.verify_password",
    "docstring": "Check if plain password matches the hash.",
    "content": "def verify_password(self, plain, hashed):\n    return bcrypt.checkpw(plain.encode(), hashed.encode())",
}


# ── _prepare_text tests ───────────────────────────────────────────────────────

def test_prepare_text_strips_whitespace():
    result = _prepare_text("  hello world  ")
    assert result == "hello world"


def test_prepare_text_removes_null_bytes():
    result = _prepare_text("hello\x00world")
    assert "\x00" not in result
    assert "helloworld" in result


def test_prepare_text_truncates_long_input():
    long_text = "x" * (MAX_INPUT_CHARS + 500)
    result = _prepare_text(long_text)
    assert len(result) == MAX_INPUT_CHARS


def test_prepare_text_returns_empty_for_blank():
    assert _prepare_text("") == ""
    assert _prepare_text("   ") == ""


def test_prepare_text_short_input_unchanged():
    text = "def hello(): return 'world'"
    result = _prepare_text(text)
    assert result == text


# ── _build_embedding_text tests ───────────────────────────────────────────────

def test_build_text_includes_chunk_type_and_name():
    text = _build_embedding_text(SAMPLE_CHUNK)
    assert "Function" in text
    assert "AuthService.verify_password" in text


def test_build_text_includes_docstring():
    text = _build_embedding_text(SAMPLE_CHUNK)
    assert "Check if plain password matches the hash" in text


def test_build_text_includes_code():
    text = _build_embedding_text(SAMPLE_CHUNK)
    assert "def verify_password" in text
    assert "bcrypt.checkpw" in text


def test_build_text_skips_empty_docstring():
    chunk = {**SAMPLE_CHUNK, "docstring": ""}
    text = _build_embedding_text(chunk)
    assert "Docstring" not in text


def test_build_text_uses_display_name_over_name():
    chunk = {**SAMPLE_CHUNK, "display_name": "MyClass.my_method"}
    text = _build_embedding_text(chunk)
    assert "MyClass.my_method" in text


def test_build_text_falls_back_to_name():
    chunk = {**SAMPLE_CHUNK, "display_name": ""}
    text = _build_embedding_text(chunk)
    assert "verify_password" in text


# ── embed_text tests ──────────────────────────────────────────────────────────

@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=_fake_embed)
@patch("embeddings.gemini_embedder._configure_genai")
def test_embed_text_returns_correct_dimensions(mock_config, mock_embed):
    vector = embed_text("def hello(): pass")
    assert len(vector) == VECTOR_DIMENSIONS


@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=_fake_embed)
@patch("embeddings.gemini_embedder._configure_genai")
def test_embed_text_returns_floats(mock_config, mock_embed):
    vector = embed_text("some code here")
    assert all(isinstance(v, float) for v in vector)


@patch("embeddings.gemini_embedder._configure_genai")
def test_embed_text_raises_for_empty_string(mock_config):
    # FIX: EmbeddingError is now excluded from tenacity retry, so it
    # propagates immediately instead of being wrapped in RetryError.
    with pytest.raises(EmbeddingError, match="empty"):
        embed_text("")


@patch("embeddings.gemini_embedder._configure_genai")
def test_embed_text_raises_for_blank_string(mock_config):
    # FIX: Same as above — EmbeddingError propagates directly, no RetryError.
    with pytest.raises(EmbeddingError):
        embed_text("    ")


@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=Exception("429 quota exceeded"))
@patch("embeddings.gemini_embedder._configure_genai")
def test_embed_text_falls_back_on_quota_error(mock_config, mock_embed):
    with patch.object(gemini_embedder.settings, "embedding_allow_local_fallback", True):
        vector = embed_text("def hello(): pass")
    assert len(vector) == VECTOR_DIMENSIONS
    assert any(v != 0.0 for v in vector)


@patch("embeddings.gemini_embedder._configure_genai", side_effect=EmbeddingError("api key missing"))
def test_embed_text_falls_back_on_provider_config_error(mock_config):
    with patch.object(gemini_embedder.settings, "embedding_allow_local_fallback", True):
        vector = embed_text("def hello(): pass")
    assert len(vector) == VECTOR_DIMENSIONS


@patch("embeddings.gemini_embedder._configure_genai", side_effect=EmbeddingError("api key missing"))
def test_embed_text_raises_when_fallback_disabled(mock_config):
    with patch.object(gemini_embedder.settings, "embedding_allow_local_fallback", False):
        with pytest.raises(EmbeddingError, match="api key"):
            embed_text("def hello(): pass")


# ── embed_query tests ─────────────────────────────────────────────────────────

@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=_fake_embed)
@patch("embeddings.gemini_embedder._configure_genai")
def test_embed_query_uses_retrieval_query_task(mock_config, mock_embed):
    with patch("embeddings.gemini_embedder.get_cache_manager") as mock_cache_factory:
        mock_cache = MagicMock()
        mock_cache.get_embedding.return_value = None
        mock_cache_factory.return_value = mock_cache
        embed_query("how does authentication work?")
    _, kwargs = mock_embed.call_args
    assert kwargs["task_type"] == TASK_TYPE_QUERY


@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=_fake_embed)
@patch("embeddings.gemini_embedder._configure_genai")
def test_embed_query_returns_768_dims(mock_config, mock_embed):
    vector = embed_query("find password hashing")
    assert len(vector) == VECTOR_DIMENSIONS


# ── embed_chunk tests ─────────────────────────────────────────────────────────

@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=_fake_embed)
@patch("embeddings.gemini_embedder._configure_genai")
def test_embed_chunk_uses_document_task(mock_config, mock_embed):
    with patch("embeddings.gemini_embedder.get_cache_manager") as mock_cache_factory:
        mock_cache = MagicMock()
        mock_cache.get_embedding.return_value = None
        mock_cache_factory.return_value = mock_cache
        embed_chunk(SAMPLE_CHUNK)
    _, kwargs = mock_embed.call_args
    assert kwargs["task_type"] == TASK_TYPE_DOCUMENT


@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=_fake_embed)
@patch("embeddings.gemini_embedder._configure_genai")
def test_embed_chunk_returns_768_dims(mock_config, mock_embed):
    vector = embed_chunk(SAMPLE_CHUNK)
    assert len(vector) == VECTOR_DIMENSIONS


# ── embed_chunks_batch tests ──────────────────────────────────────────────────

@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=_fake_embed)
@patch("embeddings.gemini_embedder._configure_genai")
@patch("embeddings.gemini_embedder.time.sleep")
def test_batch_embeds_all_chunks(mock_sleep, mock_config, mock_embed):
    chunks = [
        {**SAMPLE_CHUNK, "id": f"chunk-{i}",
         "content": f"def func_{i}(): return {i}"}
        for i in range(5)
    ]
    results = embed_chunks_batch(chunks)
    assert len(results) == 5


@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=_fake_embed)
@patch("embeddings.gemini_embedder._configure_genai")
@patch("embeddings.gemini_embedder.time.sleep")
def test_batch_returns_chunk_id_vector_pairs(mock_sleep, mock_config, mock_embed):
    chunks = [{**SAMPLE_CHUNK, "id": "my-id",
               "content": "def f(): return 1\n    pass"}]
    results = embed_chunks_batch(chunks)
    assert results[0][0] == "my-id"
    assert len(results[0][1]) == VECTOR_DIMENSIONS


@patch("embeddings.gemini_embedder.genai.embed_content")
@patch("embeddings.gemini_embedder._configure_genai")
@patch("embeddings.gemini_embedder.time.sleep")
def test_batch_skips_failed_chunks(mock_sleep, mock_config, mock_embed):
    """
    chunk-1 must be skipped. The mock always raises for chunk-1's content
    so all 3 tenacity retry attempts fail, tenacity gives up with RetryError,
    and embed_chunks_batch catches that and increments failed count.

    OLD APPROACH (broken):
        call_num[0] += 1
        if call_num[0] == 2: raise ...
      → tenacity retries the call, call_num becomes 3 on the retry → succeeds.
      → all 3 chunks succeed, len(results) == 3, test fails.

    FIX: key the failure on the CONTENT of the chunk, not call order.
    chunk-1's content always contains "func_1", so we raise for every call
    where that string appears. Tenacity retries 3 times, all 3 fail,
    tenacity raises RetryError, batch catches it → chunk-1 is skipped.
    """
    def always_fail_for_chunk_1(*args, **kwargs):
        content = kwargs.get("content", "")
        if not content and args:
            content = args[0]

        # "func_1" is unique to chunk-1's content — fail on every attempt
        if "func_1" in content:
            raise Exception("Rate limit")
        return {"embedding": [0.1] * VECTOR_DIMENSIONS}

    mock_embed.side_effect = always_fail_for_chunk_1

    chunks = [
        {**SAMPLE_CHUNK, "id": f"chunk-{i}",
         "content": f"def func_{i}(): return {i}\n    pass"}
        for i in range(3)
    ]
    with patch("embeddings.gemini_embedder.get_cache_manager") as mock_cache_factory:
        mock_cache = MagicMock()
        mock_cache.get_embedding.return_value = None
        mock_cache_factory.return_value = mock_cache
        with patch.object(gemini_embedder.settings, "embedding_allow_local_fallback", False):
            results = embed_chunks_batch(chunks)

    # chunk-0 and chunk-2 succeed, chunk-1 is skipped
    assert len(results) == 2
    succeeded_ids = [chunk_id for chunk_id, _ in results]
    assert "chunk-0" in succeeded_ids
    assert "chunk-1" not in succeeded_ids
    assert "chunk-2" in succeeded_ids


@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=_fake_embed)
@patch("embeddings.gemini_embedder._configure_genai")
@patch("embeddings.gemini_embedder.time.sleep")
def test_batch_calls_progress_callback(mock_sleep, mock_config, mock_embed):
    calls = []

    def track(done, total):
        calls.append((done, total))

    chunks = [
        {**SAMPLE_CHUNK, "id": f"chunk-{i}",
         "content": f"def func_{i}(): pass\n    return {i}"}
        for i in range(5)
    ]
    embed_chunks_batch(chunks, progress_callback=track)
    assert len(calls) >= 1
    assert calls[-1][0] == 5   # last call shows all 5 done


@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=_fake_embed)
@patch("embeddings.gemini_embedder._configure_genai")
@patch("embeddings.gemini_embedder.time.sleep")
def test_batch_sleeps_between_batches(mock_sleep, mock_config, mock_embed):
    """Sleep should be called between batches (not after the last one)."""
    # 25 chunks with batch_size=20 means 2 batches, 1 sleep
    chunks = [
        {**SAMPLE_CHUNK, "id": f"chunk-{i}",
         "content": f"def func_{i}(): pass\n    return {i}"}
        for i in range(25)
    ]
    embed_chunks_batch(chunks)
    # Should sleep once (between batch 1 and batch 2, but not after batch 2)
    assert mock_sleep.call_count == 1


@patch("embeddings.gemini_embedder.genai.embed_content", side_effect=_fake_embed)
@patch("embeddings.gemini_embedder._configure_genai")
@patch("embeddings.gemini_embedder.time.sleep")
def test_batch_empty_input_returns_empty(mock_sleep, mock_config, mock_embed):
    results = embed_chunks_batch([])
    assert results == []
    mock_embed.assert_not_called()