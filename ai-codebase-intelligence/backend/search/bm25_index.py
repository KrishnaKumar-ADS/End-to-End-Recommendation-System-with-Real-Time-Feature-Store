"""BM25 index builder for code chunks with code-aware tokenization."""

from __future__ import annotations

import pickle
import re
from typing import Any

from rank_bm25 import BM25Okapi

MIN_BM25_SCORE = 0.01
MAX_TOKENS_PER_CHUNK = 512


def _split_camel_case(text: str) -> str:
    """Insert spaces inside camelCase and PascalCase identifiers."""
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", text)
    return text


def tokenize_code(text: str | None) -> list[str]:
    """Tokenize code text into BM25-friendly tokens."""
    if not text:
        return []

    text = _split_camel_case(text)
    text = text.replace("_", " ").replace("-", " ")
    tokens = re.split(r"[^a-zA-Z0-9]+", text)
    tokens = [tok.lower() for tok in tokens if len(tok) >= 2]
    return tokens[:MAX_TOKENS_PER_CHUNK]


def _build_index_text(chunk: dict[str, Any]) -> str:
    name = chunk.get("name") or chunk.get("function_name") or ""
    docstring = chunk.get("docstring") or ""
    content = chunk.get("content") or ""

    parts: list[str] = []
    if name:
        parts.append(f"{name} {name} {name}")
    if docstring:
        parts.append(str(docstring))
    if content:
        parts.append(str(content))

    return " ".join(parts)


class BM25Index:
    """In-memory BM25 index for one repository."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []
        self._repo_id: str | None = None

    def build(self, chunks: list[dict[str, Any]], repo_id: str = "") -> None:
        if not chunks:
            return

        self._repo_id = repo_id
        self._chunks = chunks
        self._tokenized_corpus = [tokenize_code(_build_index_text(chunk)) for chunk in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 20,
        chunk_type: str | None = None,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._bm25 is None:
            return []

        query_tokens = tokenize_code(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        scored_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        output: list[dict[str, Any]] = []
        rank = 1
        for idx in scored_indices:
            if rank > top_k * 3:
                break

            score = float(scores[idx])
            if score < MIN_BM25_SCORE:
                break

            chunk = self._chunks[idx]
            if chunk_type and chunk.get("chunk_type") != chunk_type:
                continue
            if language and chunk.get("language") != language:
                continue

            result = dict(chunk)
            result["bm25_score"] = score
            result["bm25_rank"] = rank
            output.append(result)
            rank += 1

            if len(output) >= top_k:
                break

        return output

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    @property
    def repo_id(self) -> str | None:
        return self._repo_id

    def get_vocab_size(self) -> int:
        if self._bm25 is None:
            return 0
        return len(self._bm25.idf)

    def get_top_terms(self, n: int = 20) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        return sorted(self._bm25.idf.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def serialize(self) -> bytes:
        return pickle.dumps(
            {
                "bm25": self._bm25,
                "chunks": self._chunks,
                "tokenized_corpus": self._tokenized_corpus,
                "repo_id": self._repo_id,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "BM25Index":
        payload = pickle.loads(data)
        instance = cls()
        instance._bm25 = payload["bm25"]
        instance._chunks = payload["chunks"]
        instance._tokenized_corpus = payload["tokenized_corpus"]
        instance._repo_id = payload["repo_id"]
        return instance
