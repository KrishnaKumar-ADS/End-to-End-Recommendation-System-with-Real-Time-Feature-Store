"""Unit tests for search.bm25_index."""

import pytest

from search.bm25_index import BM25Index, _split_camel_case, tokenize_code


class TestTokenizeCode:
    def test_splits_camel_case(self):
        tokens = tokenize_code("hashPassword")
        assert "hash" in tokens
        assert "password" in tokens

    def test_splits_snake_case(self):
        tokens = tokenize_code("verify_user_token")
        assert "verify" in tokens
        assert "user" in tokens
        assert "token" in tokens

    def test_lowercases_tokens(self):
        tokens = tokenize_code("HTTPSConnection")
        assert all(token == token.lower() for token in tokens)

    def test_discards_single_char_tokens(self):
        assert tokenize_code("a + b = c") == []

    def test_handles_empty_or_none(self):
        assert tokenize_code("") == []
        assert tokenize_code(None) == []

    def test_split_helper(self):
        out = _split_camel_case("HTMLParser")
        assert "HTML" in out
        assert "Parser" in out


@pytest.fixture
def sample_chunks():
    return [
        {
            "id": "chunk-1",
            "name": "hash_password",
            "content": "def hash_password(plain: str) -> str: return bcrypt.hashpw(plain)",
            "chunk_type": "function",
            "language": "python",
            "file_path": "auth/crypto.py",
            "start_line": 1,
            "end_line": 2,
            "docstring": "Hash password",
        },
        {
            "id": "chunk-2",
            "name": "verify_token",
            "content": "def verify_token(token: str) -> dict: return jwt.decode(token, SECRET)",
            "chunk_type": "function",
            "language": "python",
            "file_path": "auth/jwt.py",
            "start_line": 10,
            "end_line": 12,
            "docstring": "Verify token",
        },
        {
            "id": "chunk-3",
            "name": "database_connect",
            "content": "def database_connect(): return psycopg2.connect(DSN)",
            "chunk_type": "function",
            "language": "python",
            "file_path": "db/conn.py",
            "start_line": 20,
            "end_line": 21,
            "docstring": "",
        },
    ]


@pytest.fixture
def built_index(sample_chunks):
    idx = BM25Index()
    idx.build(sample_chunks, repo_id="repo-1")
    return idx


class TestBM25Index:
    def test_build_sets_state(self, sample_chunks):
        idx = BM25Index()
        assert not idx.is_built
        idx.build(sample_chunks)
        assert idx.is_built

    def test_build_empty_keeps_unbuilt(self):
        idx = BM25Index()
        idx.build([])
        assert not idx.is_built

    def test_search_before_build_empty(self):
        idx = BM25Index()
        assert idx.search("password") == []

    def test_search_ranks_expected_chunk_first(self, built_index):
        results = built_index.search("password hashing", top_k=3)
        assert results
        assert results[0]["id"] == "chunk-1"

    def test_search_results_have_rank_and_score(self, built_index):
        results = built_index.search("password", top_k=3)
        assert all("bm25_score" in result for result in results)
        assert all("bm25_rank" in result for result in results)

    def test_search_filters(self, built_index):
        results = built_index.search("password", top_k=3, chunk_type="function", language="python")
        assert all(result["chunk_type"] == "function" for result in results)
        assert all(result["language"] == "python" for result in results)

    def test_vocab_and_top_terms(self, built_index):
        assert built_index.get_vocab_size() > 0
        assert len(built_index.get_top_terms(5)) <= 5

    def test_serialize_roundtrip(self, built_index):
        payload = built_index.serialize()
        restored = BM25Index.deserialize(payload)
        assert restored.is_built
        original = [item["id"] for item in built_index.search("password", top_k=3)]
        reloaded = [item["id"] for item in restored.search("password", top_k=3)]
        assert original == reloaded
