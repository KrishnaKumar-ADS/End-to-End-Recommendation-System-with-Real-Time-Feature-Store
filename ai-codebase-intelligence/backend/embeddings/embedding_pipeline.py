"""
Embedding Pipeline — orchestrates the full embed + index flow.

This is the only file the Celery task calls for the embedding stage.

What it does:
  1. Loads all unembedded code chunks from PostgreSQL
     (chunks where qdrant_point_id IS NULL)
  2. Builds enriched text for each chunk (name + docstring + code)
  3. Sends chunks to Gemini in batches to get vectors
  4. Uploads all vectors to Qdrant with metadata payload
  5. Updates qdrant_point_id in PostgreSQL to mark chunks as embedded

The qdrant_point_id in PostgreSQL is intentionally set to the same
value as the chunk UUID — this makes cross-database lookups trivial.
"""

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker

from db.models import CodeChunk, SourceFile
from embeddings.gemini_embedder import embed_chunks_batch
from embeddings.vector_store import (
    ensure_collection_exists,
    upsert_chunks_batch,
    count_repo_vectors,
    get_collection_info,
)
from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Sync engine — this pipeline runs inside Celery (sync context)
_sync_engine = create_engine(
    settings.sync_database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)
_SyncSession = sessionmaker(bind=_sync_engine)


# ── Main pipeline function ────────────────────────────────────────────────────

def embed_repository(
    repo_id: str,
    progress_callback=None,
) -> dict:
    """
    Full embedding pipeline for one repository.

    Stages:
      1. Ensure Qdrant collection exists
      2. Load all unembedded chunks from PostgreSQL
      3. Batch embed with Gemini
      4. Upsert vectors to Qdrant
      5. Mark as embedded in PostgreSQL

    Args:
        repo_id:           UUID string of the repository to embed
        progress_callback: optional function(done, total, message)
                           called periodically to report progress

    Returns:
        dict {
          "total_chunks": int,    — all chunks for this repo
          "embedded": int,        — successfully embedded
          "failed": int,          — failed to embed
          "already_embedded": int — skipped (already had a vector)
        }
    """
    # Stage 1: Ensure Qdrant collection exists
    ensure_collection_exists()

    # Stage 2: Load chunks from PostgreSQL
    if progress_callback:
        progress_callback(0, 1, "Loading chunks from database...")

    all_chunks   = _load_all_chunks(repo_id)
    to_embed     = [c for c in all_chunks if not c.get("already_embedded")]
    already_done = [c for c in all_chunks if c.get("already_embedded")]

    total = len(to_embed)

    if total == 0:
        logger.info(
            "embed_pipeline_nothing_to_do",
            repo_id=repo_id,
            already_embedded=len(already_done),
        )
        return {
            "total_chunks": len(all_chunks),
            "embedded": 0,
            "failed": 0,
            "already_embedded": len(already_done),
        }

    logger.info(
        "embed_pipeline_start",
        repo_id=repo_id,
        to_embed=total,
        already_embedded=len(already_done),
    )

    # Stage 3: Batch embed with Gemini
    def on_embed_progress(done, chunk_total):
        if progress_callback:
            # Embedding takes 70% of total progress budget
            pct_through = done / chunk_total if chunk_total > 0 else 0
            message = f"Embedding chunk {done}/{chunk_total}..."
            progress_callback(done, chunk_total, message)

    embedded_pairs = embed_chunks_batch(
        to_embed,
        progress_callback=on_embed_progress,
    )

    # Stage 4: Build Qdrant points with full metadata payload
    if progress_callback:
        progress_callback(total, total, "Uploading vectors to Qdrant...")

    chunk_lookup  = {c["id"]: c for c in to_embed}
    qdrant_points = []
    ids_to_mark   = []

    for chunk_id, vector in embedded_pairs:
        chunk = chunk_lookup.get(chunk_id)
        if not chunk:
            continue

        payload = _build_payload(chunk, repo_id)
        qdrant_points.append((chunk_id, vector, payload))
        ids_to_mark.append(chunk_id)

    # Upload to Qdrant
    upserted = upsert_chunks_batch(qdrant_points, batch_size=100)

    # Stage 5: Mark as embedded in PostgreSQL
    if progress_callback:
        progress_callback(total, total, "Updating database records...")

    _mark_chunks_embedded(ids_to_mark)

    failed = total - len(embedded_pairs)

    result = {
        "total_chunks": len(all_chunks),
        "embedded": upserted,
        "failed": failed,
        "already_embedded": len(already_done),
    }

    logger.info("embed_pipeline_complete", repo_id=repo_id, **result)
    return result


# ── Database helpers ──────────────────────────────────────────────────────────

def _load_all_chunks(repo_id: str) -> list[dict]:
    """
    Load all code chunks for a repository.
    Chunks with qdrant_point_id set are flagged as already_embedded.
    """
    session = _SyncSession()
    try:
        rows = session.execute(
            select(
                CodeChunk.id,
                CodeChunk.chunk_type,
                CodeChunk.name,
                CodeChunk.display_name,
                CodeChunk.content,
                CodeChunk.start_line,
                CodeChunk.end_line,
                CodeChunk.docstring,
                CodeChunk.qdrant_point_id,
                SourceFile.file_path,
                SourceFile.language,
            )
            .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
            .where(SourceFile.repository_id == repo_id)
            .order_by(SourceFile.file_path, CodeChunk.start_line)
        ).fetchall()

        chunks = []
        for row in rows:
            chunks.append({
                "id":               str(row.id),
                "chunk_type":       row.chunk_type,
                "name":             row.name,
                "display_name":     row.display_name or row.name,
                "content":          row.content or "",
                "start_line":       row.start_line,
                "end_line":         row.end_line,
                "docstring":        row.docstring or "",
                "file_path":        row.file_path,
                "language":         row.language,
                "already_embedded": row.qdrant_point_id is not None,
            })
        return chunks
    finally:
        session.close()


def _build_payload(chunk: dict, repo_id: str) -> dict:
    """
    Build the Qdrant payload for a chunk.
    This is the metadata stored alongside the vector.
    It's used for filtering searches and displaying results.
    """
    return {
        "repo_id":         repo_id,
        "file_path":       chunk["file_path"],
        "language":        chunk["language"],
        "chunk_type":      chunk["chunk_type"],
        "name":            chunk["name"],
        "display_name":    chunk["display_name"],
        "start_line":      chunk["start_line"],
        "end_line":        chunk["end_line"],
        "docstring":       (chunk["docstring"] or "")[:500],
        "content_preview": (chunk["content"] or "")[:300],
    }


def _mark_chunks_embedded(chunk_ids: list[str]) -> None:
    """
    Set qdrant_point_id on all successfully embedded chunks.
    We use the chunk's own UUID as the qdrant_point_id.
    This creates a 1:1 link between PostgreSQL and Qdrant.
    """
    if not chunk_ids:
        return

    session = _SyncSession()
    try:
        for chunk_id in chunk_ids:
            session.execute(
                update(CodeChunk)
                .where(CodeChunk.id == chunk_id)
                .values(qdrant_point_id=chunk_id)
            )
        session.commit()
        logger.debug("marked_chunks_embedded", count=len(chunk_ids))
    except Exception as e:
        session.rollback()
        logger.error("mark_embedded_failed", error=str(e), count=len(chunk_ids))
    finally:
        session.close()


# ── Stats helpers ─────────────────────────────────────────────────────────────

def get_embedding_stats(repo_id: str) -> dict:
    """
    Return embedding progress stats for a repository.
    Called by GET /api/v1/search/stats/{repo_id}
    """
    session = _SyncSession()
    try:
        from sqlalchemy import func

        # Total chunks in PostgreSQL
        total = session.execute(
            select(func.count(CodeChunk.id))
            .join(SourceFile)
            .where(SourceFile.repository_id == repo_id)
        ).scalar() or 0

        # Embedded chunks (qdrant_point_id is set)
        embedded = session.execute(
            select(func.count(CodeChunk.id))
            .join(SourceFile)
            .where(SourceFile.repository_id == repo_id)
            .where(CodeChunk.qdrant_point_id.is_not(None))
        ).scalar() or 0

        # Vectors in Qdrant
        qdrant_count = count_repo_vectors(repo_id)

        return {
            "total_chunks":        total,
            "embedded_chunks":     embedded,
            "pending_chunks":      total - embedded,
            "qdrant_vector_count": qdrant_count,
            "embedding_progress":  round(embedded / total * 100, 1) if total > 0 else 0.0,
            "in_sync":             embedded == qdrant_count,
        }
    finally:
        session.close()