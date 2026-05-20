from pydantic import BaseModel, field_validator
from uuid import UUID


class IngestRequest(BaseModel):
    github_url: str
    branch: str = "main"

    @field_validator("github_url")
    @classmethod
    def must_be_github(cls, v: str) -> str:
        if "github.com" not in v:
            raise ValueError("Only GitHub URLs are supported.")
        return v.rstrip("/")


class IngestResponse(BaseModel):
    repo_id: UUID
    task_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    repo_id: str | None = None
    error: str | None = None
    # ── New in Week 2 ─────────────────────────
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0