"""Redis-backed BM25 index cache and lifecycle helpers."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from db.models import CodeChunk, SourceFile
from search.bm25_index import BM25Index

settings = get_settings()

_KEY_INDEX = "bm25:index:{repo_id}"
_KEY_META = "bm25:meta:{repo_id}"
_KEY_LOCK = "bm25:lock:{repo_id}"

_INDEX_TTL_SECONDS = 86_400
_LOCK_TTL_SECONDS = 120

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=10,
        )
    return _redis_client


def get_cached_index(repo_id: str) -> BM25Index | None:
    key = _KEY_INDEX.format(repo_id=repo_id)
    try:
        data = _get_redis().get(key)
        if data is None:
            return None
        return BM25Index.deserialize(data)
    except Exception:
        return None


def cache_index(repo_id: str, index: BM25Index) -> None:
    redis_client = _get_redis()
    data = index.serialize()
    size_kb = len(data) / 1024

    redis_client.set(_KEY_INDEX.format(repo_id=repo_id), data, ex=_INDEX_TTL_SECONDS)

    meta = {
        "repo_id": repo_id,
        "built_at": time.time(),
        "chunk_count": index.size,
        "vocab_size": index.get_vocab_size(),
        "size_kb": round(size_kb, 1),
    }
    redis_client.set(
        _KEY_META.format(repo_id=repo_id),
        json.dumps(meta).encode("utf-8"),
        ex=_INDEX_TTL_SECONDS,
    )


def invalidate_index(repo_id: str) -> None:
    redis_client = _get_redis()
    redis_client.delete(_KEY_INDEX.format(repo_id=repo_id))
    redis_client.delete(_KEY_META.format(repo_id=repo_id))


def get_index_meta(repo_id: str) -> dict[str, Any] | None:
    key = _KEY_META.format(repo_id=repo_id)
    try:
        raw = _get_redis().get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None


def is_index_cached(repo_id: str) -> bool:
    return _get_redis().exists(_KEY_INDEX.format(repo_id=repo_id)) == 1


async def _load_chunks_from_db(db: AsyncSession, repo_id: str) -> list[dict[str, Any]]:
    query = (
        select(
            CodeChunk.id,
            CodeChunk.name,
            CodeChunk.chunk_type,
            CodeChunk.content,
            CodeChunk.docstring,
            CodeChunk.start_line,
            CodeChunk.end_line,
            CodeChunk.parent_name,
            SourceFile.file_path,
            SourceFile.language,
        )
        .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
        .where(SourceFile.repository_id == repo_id)
        .where(CodeChunk.content.is_not(None))
        .where(CodeChunk.content != "")
        .order_by(SourceFile.file_path, CodeChunk.start_line)
    )

    rows = (await db.execute(query)).all()
    return [
        {
            "id": str(row.id),
            "name": row.name,
            "chunk_type": row.chunk_type,
            "content": row.content,
            "docstring": row.docstring,
            "start_line": row.start_line,
            "end_line": row.end_line,
            "parent_name": row.parent_name,
            "file_path": row.file_path,
            "language": row.language,
        }
        for row in rows
    ]


async def get_or_build_index(
    db: AsyncSession,
    repo_id: str,
    force_rebuild: bool = False,
) -> BM25Index | None:
    if not force_rebuild:
        cached = get_cached_index(repo_id)
        if cached is not None:
            return cached

    redis_client = _get_redis()
    lock_key = _KEY_LOCK.format(repo_id=repo_id)
    lock_acquired = redis_client.set(lock_key, b"building", nx=True, ex=_LOCK_TTL_SECONDS)

    if not lock_acquired:
        for _ in range(30):
            await asyncio.sleep(1)
            cached = get_cached_index(repo_id)
            if cached is not None:
                return cached
        return None

    try:
        chunks = await _load_chunks_from_db(db, repo_id)
        if not chunks:
            return None

        loop = asyncio.get_event_loop()
        index = await loop.run_in_executor(None, _build_index_sync, chunks, repo_id)
        cache_index(repo_id, index)
        return index
    finally:
        redis_client.delete(lock_key)


def _build_index_sync(chunks: list[dict[str, Any]], repo_id: str) -> BM25Index:
    index = BM25Index()
    index.build(chunks=chunks, repo_id=repo_id)
    return index
