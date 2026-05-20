"""
Unit tests for the Qdrant vector store.

All Qdrant client calls are mocked — no running Qdrant needed.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from embeddings.vector_store import (
    ensure_collection_exists,
    upsert_chunk,
    upsert_chunks_batch,
    search,
    delete_repo_vectors,
    count_repo_vectors,
    COLLECTION_NAME,
    VECTOR_SIZE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_qdrant():
    """
    Patch get_client() to return a mock Qdrant client.
    All tests use this fixture.
    """
    with patch("embeddings.vector_store.get_client") as mock_get:
        client = MagicMock()
        mock_get.return_value = client
        yield client


def _make_vector(value: float = 0.1) -> list[float]:
    return [value] * VECTOR_SIZE


# ── ensure_collection_exists tests ───────────────────────────────────────────

def test_creates_collection_when_missing(mock_qdrant):
    """Should create collection if it does not exist."""
    mock_qdrant.get_collections.return_value.collections = []

    ensure_collection_exists()

    mock_qdrant.create_collection.assert_called_once()
    call_kwargs = mock_qdrant.create_collection.call_args[1]
    assert call_kwargs["collection_name"] == COLLECTION_NAME


def test_skips_creation_when_already_exists(mock_qdrant):
    """Should not recreate an existing collection."""
    existing = MagicMock()
    existing.name = COLLECTION_NAME
    mock_qdrant.get_collections.return_value.collections = [existing]

    ensure_collection_exists()

    mock_qdrant.create_collection.assert_not_called()


def test_collection_name_is_correct(mock_qdrant):
    mock_qdrant.get_collections.return_value.collections = []
    ensure_collection_exists()
    call_kwargs = mock_qdrant.create_collection.call_args[1]
    assert call_kwargs["collection_name"] == "code_chunks"


def test_collection_vector_size_is_768(mock_qdrant):
    mock_qdrant.get_collections.return_value.collections = []
    ensure_collection_exists()
    call_kwargs = mock_qdrant.create_collection.call_args[1]
    assert call_kwargs["vectors_config"].size == 768


# ── upsert_chunk tests ────────────────────────────────────────────────────────

def test_upsert_chunk_calls_qdrant_upsert(mock_qdrant):
    upsert_chunk("chunk-123", _make_vector(), {"repo_id": "r1"})
    mock_qdrant.upsert.assert_called_once()


def test_upsert_chunk_passes_correct_collection(mock_qdrant):
    upsert_chunk("chunk-123", _make_vector(), {"repo_id": "r1"})
    call_kwargs = mock_qdrant.upsert.call_args[1]
    assert call_kwargs["collection_name"] == COLLECTION_NAME


def test_upsert_chunk_sets_correct_id(mock_qdrant):
    upsert_chunk("my-chunk-id", _make_vector(), {"repo_id": "r1"})
    call_kwargs = mock_qdrant.upsert.call_args[1]
    point = call_kwargs["points"][0]
    assert point.id == "my-chunk-id"


def test_upsert_chunk_rejects_wrong_vector_size(mock_qdrant):
    bad_vector = [0.1] * 100  # wrong size
    with pytest.raises(ValueError, match="768"):
        upsert_chunk("chunk-123", bad_vector, {"repo_id": "r1"})


def test_upsert_chunk_stores_payload(mock_qdrant):
    payload = {"repo_id": "r1", "name": "my_func", "file_path": "app.py"}
    upsert_chunk("chunk-123", _make_vector(), payload)
    call_kwargs = mock_qdrant.upsert.call_args[1]
    point = call_kwargs["points"][0]
    assert point.payload == payload


# ── upsert_chunks_batch tests ─────────────────────────────────────────────────

def test_batch_upsert_100_chunks_in_one_call(mock_qdrant):
    chunks = [
        (f"id-{i}", _make_vector(), {"repo_id": "r1"})
        for i in range(100)
    ]
    upsert_chunks_batch(chunks, batch_size=100)
    assert mock_qdrant.upsert.call_count == 1


def test_batch_upsert_splits_250_into_3_calls(mock_qdrant):
    """250 chunks with batch_size=100 → 3 upsert calls: 100+100+50."""
    chunks = [
        (f"id-{i}", _make_vector(), {"repo_id": "r1"})
        for i in range(250)
    ]
    upsert_chunks_batch(chunks, batch_size=100)
    assert mock_qdrant.upsert.call_count == 3


def test_batch_upsert_returns_total_count(mock_qdrant):
    chunks = [
        (f"id-{i}", _make_vector(), {"repo_id": "r1"})
        for i in range(47)
    ]
    count = upsert_chunks_batch(chunks, batch_size=100)
    assert count == 47


def test_batch_upsert_empty_list(mock_qdrant):
    count = upsert_chunks_batch([], batch_size=100)
    assert count == 0
    mock_qdrant.upsert.assert_not_called()


# ── search tests ──────────────────────────────────────────────────────────────

def test_search_calls_qdrant_search(mock_qdrant):
    mock_qdrant.search.return_value = []
    search(_make_vector(), repo_id="r1")
    mock_qdrant.search.assert_called_once()


def test_search_uses_correct_collection(mock_qdrant):
    mock_qdrant.search.return_value = []
    search(_make_vector(), repo_id="r1")
    call_kwargs = mock_qdrant.search.call_args[1]
    assert call_kwargs["collection_name"] == COLLECTION_NAME


def test_search_always_filters_by_repo_id(mock_qdrant):
    mock_qdrant.search.return_value = []
    search(_make_vector(), repo_id="my-special-repo")
    call_kwargs = mock_qdrant.search.call_args[1]
    conditions = call_kwargs["query_filter"].must
    repo_cond = next(c for c in conditions if c.key == "repo_id")
    assert repo_cond.match.value == "my-special-repo"


def test_search_adds_language_filter_when_provided(mock_qdrant):
    mock_qdrant.search.return_value = []
    search(_make_vector(), repo_id="r1", language="python")
    call_kwargs = mock_qdrant.search.call_args[1]
    conditions = call_kwargs["query_filter"].must
    lang_cond = next((c for c in conditions if c.key == "language"), None)
    assert lang_cond is not None
    assert lang_cond.match.value == "python"


def test_search_no_language_filter_when_not_provided(mock_qdrant):
    mock_qdrant.search.return_value = []
    search(_make_vector(), repo_id="r1")
    call_kwargs = mock_qdrant.search.call_args[1]
    conditions = call_kwargs["query_filter"].must
    lang_conds = [c for c in conditions if c.key == "language"]
    assert len(lang_conds) == 0


def test_search_adds_chunk_type_filter(mock_qdrant):
    mock_qdrant.search.return_value = []
    search(_make_vector(), repo_id="r1", chunk_type="function")
    call_kwargs = mock_qdrant.search.call_args[1]
    conditions = call_kwargs["query_filter"].must
    type_cond = next((c for c in conditions if c.key == "chunk_type"), None)
    assert type_cond is not None
    assert type_cond.match.value == "function"


def test_search_returns_scored_points(mock_qdrant):
    point = MagicMock()
    point.id = "chunk-uuid"
    point.score = 0.92
    point.payload = {"name": "hash_password", "file_path": "crypto.py"}
    mock_qdrant.search.return_value = [point]

    results = search(_make_vector(), repo_id="r1")
    assert len(results) == 1
    assert results[0].score == 0.92
    assert results[0].payload["name"] == "hash_password"


def test_search_respects_top_k(mock_qdrant):
    mock_qdrant.search.return_value = []
    search(_make_vector(), repo_id="r1", top_k=3)
    call_kwargs = mock_qdrant.search.call_args[1]
    assert call_kwargs["limit"] == 3


def test_search_rejects_wrong_vector_size(mock_qdrant):
    bad_vector = [0.1] * 100
    with pytest.raises(ValueError, match="768"):
        search(bad_vector, repo_id="r1")


# ── delete tests ──────────────────────────────────────────────────────────────

def test_delete_calls_qdrant_delete(mock_qdrant):
    delete_repo_vectors("repo-to-delete")
    mock_qdrant.delete.assert_called_once()


def test_delete_uses_correct_collection(mock_qdrant):
    delete_repo_vectors("repo-to-delete")
    call_kwargs = mock_qdrant.delete.call_args[1]
    assert call_kwargs["collection_name"] == COLLECTION_NAME


def test_delete_filters_by_repo_id(mock_qdrant):
    delete_repo_vectors("specific-repo-id")
    call_kwargs = mock_qdrant.delete.call_args[1]
    selector = call_kwargs["points_selector"]
    condition = selector.must[0]
    assert condition.key == "repo_id"
    assert condition.match.value == "specific-repo-id"


# ── count tests ───────────────────────────────────────────────────────────────

def test_count_returns_integer(mock_qdrant):
    mock_qdrant.count.return_value.count = 42
    result = count_repo_vectors("r1")
    assert result == 42


def test_count_filters_by_repo_id(mock_qdrant):
    mock_qdrant.count.return_value.count = 0
    count_repo_vectors("test-repo")
    call_kwargs = mock_qdrant.count.call_args[1]
    condition = call_kwargs["count_filter"].must[0]
    assert condition.match.value == "test-repo"