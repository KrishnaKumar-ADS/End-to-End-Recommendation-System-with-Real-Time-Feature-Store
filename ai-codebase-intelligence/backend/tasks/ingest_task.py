"""
Ingestion Task — full pipeline from GitHub URL to stored chunks.

Pipeline stages:
  1. Clone     → GitPython clones the repo to data/raw/{repo_id}/
  2. Scan      → file_scanner finds all source files
  3. Parse     → metadata_extractor extracts functions/classes/chunks → PostgreSQL
  4. Embed     → embedding_pipeline embeds all chunks → Qdrant
  5. Graph     → graph_builder builds call graph → Neo4j   (non-fatal if it fails)
  6. Complete  → status set to COMPLETED
"""
import asyncio
import sys
import os
from pathlib import Path

from celery.utils.log import get_task_logger
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from tasks.celery_app import celery_app
from db.database import AsyncSessionLocal as PgAsyncSessionLocal
from db.models import Repository, SourceFile, CodeChunk, IngestionStatus
from ingestion.repo_loader import clone_repository
from ingestion.file_scanner import scan_repository, ScannedFile
from parsing.metadata_extractor import extract_metadata
from graph.graph_builder import build_repo_graph
from graph.hierarchy_builder import build_class_hierarchy
from graph.neo4j_writer import write_graph_to_neo4j, delete_repo_graph
from graph.schema import create_indexes
from search.bm25_store import invalidate_index
from core.config import get_settings
from core.exceptions import RepoNotFoundError

# Ensure sibling packages like `embeddings` are always importable in worker runtime.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

logger   = get_task_logger(__name__)
settings = get_settings()

sync_engine = create_engine(
    settings.sync_database_url,
    pool_pre_ping=True,
    echo=False,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _update_state(task, progress: int, message: str) -> None:
    """Push a progress update to Celery's result backend."""
    task.update_state(
        state="STARTED",
        meta={"progress": progress, "message": message},
    )


def _set_status(
    repo_id: str,
    status: IngestionStatus,
    total_files: int = 0,
    processed_files: int = 0,
    error: str | None = None,
) -> None:
    """Write a status update for the repo row in PostgreSQL."""
    with SyncSessionLocal() as session:
        repo = session.execute(
            select(Repository).where(Repository.id == repo_id)
        ).scalar_one_or_none()

        if not repo:
            raise RepoNotFoundError(f"Repository {repo_id} not found")

        repo.status = status

        if total_files:
            repo.total_files = total_files

        if processed_files:
            repo.processed_files = processed_files

        if error:
            repo.error_message = error

        session.commit()


def _store_file_and_chunks(repo_id: str, scanned: ScannedFile, chunks: list) -> int:
    """Persist one scanned file and all its extracted chunks to PostgreSQL."""
    with SyncSessionLocal() as session:
        source_file = SourceFile(
            repository_id=repo_id,
            file_path=scanned.relative_path,
            language=scanned.language,
            size_bytes=scanned.size_bytes,
            line_count=scanned.line_count,
        )

        session.add(source_file)
        session.flush()

        for chunk in chunks:
            session.add(
                CodeChunk(
                    source_file_id=source_file.id,
                    chunk_type=chunk.chunk_type,
                    name=chunk.name,
                    display_name=chunk.display_name,
                    content=chunk.content,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    docstring=chunk.docstring,
                    parent_name=chunk.parent_name,
                )
            )

        session.commit()
        return len(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.ingest_task.run_ingestion_task",
    max_retries=3,
    soft_time_limit=900,
)
def run_ingestion_task(self, repo_id: str, github_url: str, branch: str = "main"):
    logger.info(f"[{repo_id}] Starting ingestion pipeline")

    try:
        # ── Stage 1: Clone ────────────────────────────────────────────────────
        _update_state(self, 5, "Cloning repository...")
        _set_status(repo_id, IngestionStatus.CLONING)

        repo_path = clone_repository(github_url, repo_id, branch)
        logger.info(f"[{repo_id}] Cloned to {repo_path}")

        # ── Stage 2: Scan ─────────────────────────────────────────────────────
        _update_state(self, 20, "Scanning source files...")
        _set_status(repo_id, IngestionStatus.SCANNING)

        scanned_files = scan_repository(repo_path)
        total_files   = len(scanned_files)

        logger.info(f"[{repo_id}] Found {total_files} files")

        if total_files == 0:
            _set_status(repo_id, IngestionStatus.COMPLETED)
            return {
                "repo_id":      repo_id,
                "total_files":  0,
                "total_chunks": 0,
            }

        # ── Stage 3: Parse + Store ────────────────────────────────────────────
        _update_state(self, 30, f"Parsing {total_files} files...")
        _set_status(repo_id, IngestionStatus.PARSING)

        total_chunks = 0

        for i, scanned in enumerate(scanned_files):
            if i % 10 == 0:
                progress = 30 + int((i / max(total_files, 1)) * 35)
                _update_state(
                    self,
                    progress,
                    f"Parsing {i + 1}/{total_files}: {scanned.relative_path}",
                )

            try:
                chunks = extract_metadata(
                    file_path=scanned.path,
                    relative_path=scanned.relative_path,
                    language=scanned.language,
                )
            except Exception as e:
                logger.warning(f"[{repo_id}] Parse failed: {scanned.relative_path} → {e}")
                chunks = []

            total_chunks += _store_file_and_chunks(repo_id, scanned, chunks)

        logger.info(f"[{repo_id}] Stored {total_chunks} chunks")

        # ── Stage 4: Embedding ────────────────────────────────────────────────
        _update_state(self, 65, f"Embedding {total_chunks} chunks...")
        _set_status(repo_id, IngestionStatus.EMBEDDING)

        total_embedded  = 0
        embedding_error: str | None = None

        try:
            # Guard against worker runtime CWD / sys.path drift before dynamic import.
            os.chdir(str(BACKEND_ROOT))
            if str(BACKEND_ROOT) not in sys.path:
                sys.path.insert(0, str(BACKEND_ROOT))

            from embeddings.embedding_pipeline import embed_repository

            def _embedding_progress(done: int, chunk_total: int, message: str) -> None:
                if chunk_total > 0:
                    progress = 65 + int((done / chunk_total) * 20)   # 65 → 85
                    _update_state(self, progress, message)

            result = embed_repository(
                repo_id=repo_id,
                progress_callback=_embedding_progress,
            )

            total_embedded = result.get("embedded", 0)

            attempted = result.get("total_chunks", total_chunks) - result.get("already_embedded", 0)
            if attempted > 0 and total_embedded == 0:
                embedding_error = (
                    "Embedding produced 0 vectors. "
                    "Check Gemini embedding model / API key and Qdrant connectivity."
                )
            elif result.get("failed", 0) > 0:
                logger.warning(
                    f"[{repo_id}] Partial embedding failures: "
                    f"embedded={total_embedded}, failed={result.get('failed', 0)}"
                )

        except Exception as e:
            embedding_error = str(e)

        if embedding_error:
            logger.error(f"[{repo_id}] Embedding failed: {embedding_error}")
            failure_message = f"Failed during embedding: {embedding_error}"
            _set_status(
                repo_id,
                IngestionStatus.FAILED,
                total_files=total_files,
                processed_files=total_files,
                error=embedding_error,
            )
            _update_state(self, 100, failure_message)
            return {
                "repo_id":        repo_id,
                "status":         "failed",
                "error":          embedding_error,
                "progress":       100,
                "message":        failure_message,
                "total_files":    total_files,
                "total_chunks":   total_chunks,
                "total_embedded": total_embedded,
            }

        # ── Stage 5: Graph Building ───────────────────────────────────────────
        # Non-fatal: if the graph build fails the repo is still fully searchable
        # via Qdrant vector search. We log the error and proceed to completion.
        _update_state(self, 88, "Building call graph in Neo4j...")
        logger.info(f"[{repo_id}] Graph stage start")

        graph_stats: dict = {}
        hierarchy_stats: dict = {}

        try:
            # Create Neo4j indexes on first run (idempotent).
            create_indexes()

            # Delete any existing graph for this repo (handles re-ingestion).
            delete_repo_graph(repo_id)

            # Build the in-memory graph from PostgreSQL using an async DB session.
            async def _build_repo_graph_async() -> object:
                async with PgAsyncSessionLocal() as async_db:
                    return await build_repo_graph(repo_id, async_db)

            repo_graph = asyncio.run(_build_repo_graph_async())

            # Write nodes first, then edges.
            graph_stats = write_graph_to_neo4j(repo_graph)

            logger.info(
                f"[{repo_id}] Graph complete — "
                f"nodes={graph_stats.get('nodes_written', 0)}, "
                f"edges={graph_stats.get('edges_written', 0)}, "
                f"calls={graph_stats.get('calls_edges', 0)}"
            )

            _update_state(
                self,
                95,
                f"Graph built — "
                f"{graph_stats.get('function_nodes', 0)} functions, "
                f"{graph_stats.get('calls_edges', 0)} call edges.",
            )

        except Exception as graph_err:
            logger.error(
                f"[{repo_id}] Graph build failed (non-fatal): {graph_err}",
                exc_info=True,
            )
            # Do NOT return or raise here — fall through to completion.

        # ── Stage 5b: Class hierarchy edges (Week 5) ───────────────────────
        try:
            _update_state(self, 97, "Building class hierarchy graph...")
            with SyncSessionLocal() as sync_db:
                hierarchy_result = build_class_hierarchy(repo_id=repo_id, db=sync_db)
            hierarchy_stats = {
                "classes_processed": hierarchy_result.classes_processed,
                "inherits_from_edges": hierarchy_result.inherits_from_edges,
                "implements_edges": hierarchy_result.implements_edges,
                "mixes_in_edges": hierarchy_result.mixes_in_edges,
                "classes_updated": hierarchy_result.classes_updated,
            }
        except Exception as hierarchy_err:
            logger.error(
                f"[{repo_id}] Hierarchy build failed (non-fatal): {hierarchy_err}",
                exc_info=True,
            )

        # ── Stage 6: Complete ─────────────────────────────────────────────────
        _set_status(
            repo_id,
            IngestionStatus.COMPLETED,
            total_files=total_files,
            processed_files=total_files,
        )

        try:
            invalidate_index(str(repo_id))
        except Exception:
            logger.warning(f"[{repo_id}] BM25 index invalidation failed (non-fatal)")

        completion_message = (
            f"Done — {total_files} files, "
            f"{total_chunks} chunks, "
            f"{total_embedded} embedded, "
            f"{graph_stats.get('function_nodes', 0)} functions in graph, "
            f"{hierarchy_stats.get('classes_processed', 0)} classes in hierarchy."
        )

        _update_state(self, 100, completion_message)
        logger.info(f"[{repo_id}] Pipeline completed")

        return {
            "repo_id":        repo_id,
            "status":         "completed",
            "total_files":    total_files,
            "total_chunks":   total_chunks,
            "total_embedded": total_embedded,
            "graph_stats":    graph_stats,
            "hierarchy_stats": hierarchy_stats,
        }

    except RepoNotFoundError as e:
        logger.error(f"[{repo_id}] Repo not found: {e}")
        raise   # Re-raise directly — no retry for a missing repo row.

    except Exception as e:
        logger.error(f"[{repo_id}] FAILED: {e}")

        try:
            _set_status(repo_id, IngestionStatus.FAILED, error=str(e))
        except Exception:
            logger.error(f"[{repo_id}] Could not update failure status")

        raise self.retry(exc=e)
        # NOTE: raise self.retry() terminates this call by raising Retry.
        # Nothing after this line runs. Keep this as the absolute last statement
        # in the except block.