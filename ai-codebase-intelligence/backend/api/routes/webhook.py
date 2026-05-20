"""GitHub webhook endpoint for push-triggered ingestion."""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from db.database import get_db
from db.models import IngestionStatus, Repository
from tasks.ingest_task import run_ingestion_task

router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


class WebhookResponse(BaseModel):
    triggered: bool
    task_id: str | None = None
    repo_url: str | None = None
    branch: str | None = None
    message: str


def _verify_github_signature(payload: bytes, signature_header: str | None) -> bool:
    settings = get_settings()
    secret = (settings.webhook_secret or "").strip()

    if not secret:
        return True

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    provided = signature_header.split("=", 1)[1]
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, provided)


@router.post(
    "/webhook",
    response_model=WebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Handle GitHub push events and queue ingestion.",
)
async def github_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> WebhookResponse:
    payload_bytes = await request.body()
    signature_header = request.headers.get("X-Hub-Signature-256")

    if not _verify_github_signature(payload_bytes, signature_header):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook signature.")

    try:
        payload = json.loads(payload_bytes.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body.") from exc

    event = (request.headers.get("X-GitHub-Event") or "").strip().lower()
    if event == "ping":
        return WebhookResponse(
            triggered=False,
            message="Ignored GitHub ping event.",
        )

    if event != "push":
        return WebhookResponse(
            triggered=False,
            message=f"Ignored unsupported GitHub event '{event or 'unknown'}'.",
        )

    repository_payload = payload.get("repository") if isinstance(payload, dict) else None
    repository_payload = repository_payload if isinstance(repository_payload, dict) else {}

    clone_url = str(
        repository_payload.get("clone_url")
        or repository_payload.get("ssh_url")
        or repository_payload.get("html_url")
        or ""
    ).strip()
    if not clone_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract repository URL from payload.",
        )

    ref = str(payload.get("ref") or "").strip()
    branch = ref.replace("refs/heads/", "", 1) if ref.startswith("refs/heads/") else (ref or "main")

    repo_result = await db.execute(select(Repository).where(Repository.github_url == clone_url))
    repository = repo_result.scalar_one_or_none()

    if repository is None:
        repository = Repository(
            id=uuid.uuid4(),
            github_url=clone_url,
            name=clone_url.rstrip("/").split("/")[-1].replace(".git", ""),
            branch=branch,
            status=IngestionStatus.QUEUED,
        )
        db.add(repository)
        await db.flush()
    else:
        repository.branch = branch
        repository.status = IngestionStatus.QUEUED
        repository.error_message = None

    task = run_ingestion_task.delay(
        repo_id=str(repository.id),
        github_url=clone_url,
        branch=branch,
    )
    repository.task_id = task.id

    await db.commit()

    return WebhookResponse(
        triggered=True,
        task_id=task.id,
        repo_url=clone_url,
        branch=branch,
        message=f"Ingestion task queued for {clone_url} (branch: {branch})",
    )
