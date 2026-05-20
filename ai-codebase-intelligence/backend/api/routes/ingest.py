from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select
import uuid
from db.models import Repository, SourceFile, CodeChunk, IngestionStatus
from db.database import get_db
from api.schemas.ingest_schema import IngestRequest, IngestResponse, StatusResponse
from tasks.ingest_task import run_ingestion_task
from core.logging import get_logger
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])
logger = get_logger(__name__)


class RepositorySummaryResponse(BaseModel):
    id: str
    github_url: str
    name: str
    branch: str
    status: str
    task_id: str | None = None
    error_message: str | None = None
    total_files: int
    processed_files: int
    total_chunks: int
    created_at: str
    updated_at: str


class RepositoryFileResponse(BaseModel):
    id: str
    file_path: str
    language: str
    size_bytes: int
    line_count: int
    chunk_count: int


class RepositoryDetailResponse(RepositorySummaryResponse):
    files: list[RepositoryFileResponse] = Field(default_factory=list)


class DeleteRepositoryResponse(BaseModel):
    repo_id: str
    repo_name: str
    deleted_files: int
    deleted_chunks: int
    deleted_vectors: int
    deleted_graph_nodes: int
    deleted_cache_keys: int
    warnings: list[str] = Field(default_factory=list)
    message: str


def _to_repo_summary_payload(row) -> dict:
    return {
        "id": str(row.id),
        "github_url": row.github_url,
        "name": row.name,
        "branch": row.branch,
        "status": row.status.value,
        "task_id": row.task_id,
        "error_message": row.error_message,
        "total_files": int(row.total_files or 0),
        "processed_files": int(row.processed_files or 0),
        "total_chunks": int(row.total_chunks or 0),
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=202,
    summary="Ingest a GitHub repository",
    description=(
        "Queues asynchronous ingestion for a repository URL. "
        "The background task clones, scans, parses, chunks, embeds, and indexes repository data."
    ),
    responses={
        202: {
            "description": "Ingestion queued successfully.",
        },
        409: {
            "description": "Repository was already ingested.",
        },
        503: {
            "description": "Task queue is unavailable or enqueue failed.",
        },
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "requests": {
                            "summary": "Python repository",
                            "value": {
                                "github_url": "https://github.com/psf/requests",
                                "branch": "main",
                            },
                        },
                        "express": {
                            "summary": "JavaScript repository",
                            "value": {
                                "github_url": "https://github.com/expressjs/express",
                                "branch": "master",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def ingest_repository(request: IngestRequest, db: AsyncSession = Depends(get_db)):
    repo_name = request.github_url.rstrip("/").split("/")[-1]
    repo_id = uuid.uuid4()
    task_id = str(uuid.uuid4())

    existing = await db.execute(select(Repository).where(Repository.github_url == request.github_url))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Repository already ingested.")

    repo = Repository(
        id=repo_id,
        github_url=request.github_url,
        name=repo_name,
        branch=request.branch,
        status=IngestionStatus.QUEUED,
        task_id=task_id,
    )
    db.add(repo)
    await db.commit()

    try:
        run_ingestion_task.apply_async(
            kwargs={
                "repo_id": str(repo_id),
                "github_url": request.github_url,
                "branch": request.branch,
            },
            task_id=task_id,
        )
    except Exception as exc:
        logger.exception("Failed to enqueue ingestion task for repo_id=%s", repo_id)

        failed_repo = await db.get(Repository, repo_id)
        if failed_repo is not None:
            failed_repo.status = IngestionStatus.FAILED
            failed_repo.error_message = "Failed to queue ingestion task."
            await db.commit()

        raise HTTPException(status_code=503, detail="Failed to queue ingestion task.") from exc

    return IngestResponse(
        repo_id=repo_id,
        task_id=task_id,
        status="queued",
        message=f"Repository '{repo_name}' queued for ingestion.",
    )


@router.get(
    "/status/{task_id}",
    response_model=StatusResponse,
    summary="Get ingestion task status",
    description=(
        "Returns current ingestion status for a task id. "
        "Possible states include queued, cloning, scanning, parsing, embedding, completed, and failed."
    ),
    responses={
        200: {
            "description": "Status returned successfully.",
        },
        404: {
            "description": "Task not found.",
        },
    },
)
async def get_task_status(task_id: str, db: AsyncSession = Depends(get_db)):
    from celery.result import AsyncResult
    from tasks.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)
    repo_result = await db.execute(select(Repository).where(Repository.task_id == task_id))
    repo = repo_result.scalar_one_or_none()

    if repo is None and result.state == "PENDING":
        raise HTTPException(status_code=404, detail="Task not found.")

    meta = result.info if isinstance(result.info, dict) else {}
    result_state = result.state.upper()

    status_value = repo.status.value if repo else result_state.lower()
    repo_failed = bool(repo and repo.status == IngestionStatus.FAILED)
    repo_completed = bool(repo and repo.status == IngestionStatus.COMPLETED)
    repo_in_progress = bool(
        repo and repo.status in {
            IngestionStatus.QUEUED,
            IngestionStatus.CLONING,
            IngestionStatus.SCANNING,
            IngestionStatus.PARSING,
            IngestionStatus.EMBEDDING,
        }
    )
    meta_failed = meta.get("status") == "failed"
    celery_failed = result_state == "FAILURE"

    if repo_in_progress and celery_failed:
        status_value = IngestionStatus.FAILED.value

    progress = meta.get("progress")
    if progress is None:
        # Reflect terminal repository status even if Celery meta has no progress.
        progress = 100 if (repo_failed or repo_completed or celery_failed) else 0

    error: str | None = None
    if repo_failed:
        error = repo.error_message or meta.get("error")
    elif celery_failed:
        error = str(result.info) if result.info else meta.get("error")
    elif meta_failed:
        error = meta.get("error")

    message = meta.get("message", "")
    if repo_completed and (meta_failed or celery_failed):
        # Historical task metadata may still contain old failure info after recovery.
        message = "Recovered from previous failed task metadata."
    elif not message and error:
        message = "Ingestion failed."

    # Count chunks stored so far for this repo
    total_chunks = 0
    if repo:
        chunk_count_result = await db.execute(
            select(func.count(CodeChunk.id))
            .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
            .where(SourceFile.repository_id == repo.id)
        )
        total_chunks = chunk_count_result.scalar() or 0

    return StatusResponse(
        task_id=task_id,
        status=status_value,
        progress=progress,
        message=message,
        repo_id=str(repo.id) if repo else None,
        error=error,
        total_files=repo.total_files if repo else 0,
        processed_files=repo.processed_files if repo else 0,
        total_chunks=total_chunks,
    )


@router.get("/repos", response_model=list[RepositorySummaryResponse])
async def list_repositories(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            select(
                Repository.id,
                Repository.github_url,
                Repository.name,
                Repository.branch,
                Repository.status,
                Repository.task_id,
                Repository.error_message,
                Repository.total_files,
                Repository.processed_files,
                Repository.created_at,
                Repository.updated_at,
                func.count(CodeChunk.id).label("total_chunks"),
            )
            .outerjoin(SourceFile, SourceFile.repository_id == Repository.id)
            .outerjoin(CodeChunk, CodeChunk.source_file_id == SourceFile.id)
            .group_by(
                Repository.id,
                Repository.github_url,
                Repository.name,
                Repository.branch,
                Repository.status,
                Repository.task_id,
                Repository.error_message,
                Repository.total_files,
                Repository.processed_files,
                Repository.created_at,
                Repository.updated_at,
            )
            .order_by(Repository.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
    ).all()

    return [RepositorySummaryResponse(**_to_repo_summary_payload(row)) for row in rows]


@router.get("/repos/{repo_id}", response_model=RepositoryDetailResponse)
async def get_repository_detail(repo_id: str, db: AsyncSession = Depends(get_db)):
    try:
        repo_uuid = uuid.UUID(repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid repository id.") from exc

    repo = await db.get(Repository, repo_uuid)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repository not found.")

    chunk_count = (
        await db.execute(
            select(func.count(CodeChunk.id))
            .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
            .where(SourceFile.repository_id == repo_uuid)
        )
    ).scalar() or 0

    file_rows = (
        await db.execute(
            select(
                SourceFile.id,
                SourceFile.file_path,
                SourceFile.language,
                SourceFile.size_bytes,
                SourceFile.line_count,
                func.count(CodeChunk.id).label("chunk_count"),
            )
            .outerjoin(CodeChunk, CodeChunk.source_file_id == SourceFile.id)
            .where(SourceFile.repository_id == repo_uuid)
            .group_by(
                SourceFile.id,
                SourceFile.file_path,
                SourceFile.language,
                SourceFile.size_bytes,
                SourceFile.line_count,
            )
            .order_by(SourceFile.file_path.asc())
        )
    ).all()

    files = [
        RepositoryFileResponse(
            id=str(row.id),
            file_path=row.file_path,
            language=row.language,
            size_bytes=int(row.size_bytes or 0),
            line_count=int(row.line_count or 0),
            chunk_count=int(row.chunk_count or 0),
        )
        for row in file_rows
    ]

    summary_payload = {
        "id": str(repo.id),
        "github_url": repo.github_url,
        "name": repo.name,
        "branch": repo.branch,
        "status": repo.status.value,
        "task_id": repo.task_id,
        "error_message": repo.error_message,
        "total_files": int(repo.total_files or 0),
        "processed_files": int(repo.processed_files or 0),
        "total_chunks": int(chunk_count),
        "created_at": repo.created_at.isoformat(),
        "updated_at": repo.updated_at.isoformat(),
    }

    return RepositoryDetailResponse(**summary_payload, files=files)


@router.delete(
    "/repos/{repo_id}",
    response_model=DeleteRepositoryResponse,
    summary="Delete an ingested repository",
    description=(
        "Deletes repository records from PostgreSQL and attempts best-effort cleanup of "
        "related vectors, graph nodes, cache entries, and local cloned files."
    ),
    responses={
        200: {
            "description": "Repository and associated indexed artifacts were deleted.",
        },
        404: {
            "description": "Repository not found.",
        },
        422: {
            "description": "Invalid repository id.",
        },
        500: {
            "description": "Database deletion failed.",
        },
    },
)
async def delete_repository_by_id(repo_id: str, db: AsyncSession = Depends(get_db)):
    try:
        repo_uuid = uuid.UUID(repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid repository id.") from exc

    repo = await db.get(Repository, repo_uuid)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repository not found.")

    file_count = (
        await db.execute(
            select(func.count(SourceFile.id)).where(SourceFile.repository_id == repo_uuid)
        )
    ).scalar() or 0

    chunk_count = (
        await db.execute(
            select(func.count(CodeChunk.id))
            .join(SourceFile, CodeChunk.source_file_id == SourceFile.id)
            .where(SourceFile.repository_id == repo_uuid)
        )
    ).scalar() or 0

    deleted_vectors = 0
    deleted_graph_nodes = 0
    deleted_cache_keys = 0
    warnings: list[str] = []

    try:
        from embeddings.vector_store import count_repo_vectors, delete_repo_vectors

        deleted_vectors = int(count_repo_vectors(repo_id))
        delete_repo_vectors(repo_id)
    except Exception as exc:
        logger.warning("repo_delete_vector_cleanup_failed", repo_id=repo_id, error=str(exc))
        warnings.append("Vector cleanup failed; stale vectors may remain.")

    try:
        from graph.neo4j_writer import delete_repo_graph

        deleted_graph_nodes = int(delete_repo_graph(repo_id))
    except Exception as exc:
        logger.warning("repo_delete_graph_cleanup_failed", repo_id=repo_id, error=str(exc))
        warnings.append("Graph cleanup failed; stale graph nodes may remain.")

    try:
        from caching.cache_manager import get_cache_manager
        from semantic_cache.answer_cache import get_semantic_answer_cache

        deleted_cache_keys = int(get_cache_manager().invalidate_repo(repo_id))
        deleted_cache_keys += int(await get_semantic_answer_cache().invalidate_repo(repo_id))
    except Exception as exc:
        logger.warning("repo_delete_cache_cleanup_failed", repo_id=repo_id, error=str(exc))
        warnings.append("Cache cleanup failed.")

    try:
        from ingestion.repo_loader import delete_repository as delete_raw_repository

        delete_raw_repository(repo_id)
    except Exception as exc:
        logger.warning("repo_delete_raw_cleanup_failed", repo_id=repo_id, error=str(exc))
        warnings.append("Local repository directory cleanup failed.")

    repo_name = repo.name

    try:
        await db.delete(repo)
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("repo_delete_db_failed", repo_id=repo_id)
        raise HTTPException(status_code=500, detail="Failed to delete repository from database.") from exc

    return DeleteRepositoryResponse(
        repo_id=repo_id,
        repo_name=repo_name,
        deleted_files=int(file_count),
        deleted_chunks=int(chunk_count),
        deleted_vectors=deleted_vectors,
        deleted_graph_nodes=deleted_graph_nodes,
        deleted_cache_keys=deleted_cache_keys,
        warnings=warnings,
        message=f"Repository '{repo_name}' deleted successfully.",
    )