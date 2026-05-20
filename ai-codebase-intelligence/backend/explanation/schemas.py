"""Pydantic schemas for /api/v1/explain endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExplainRequest(BaseModel):
    repo_id: str = Field(..., description="Repository UUID.")
    function_name: str | None = Field(default=None)
    file_path: str | None = Field(default=None)
    chunk_id: str | None = Field(default=None)
    max_callers: int = Field(default=5, ge=1, le=20)
    max_callees: int = Field(default=5, ge=1, le=20)


class ParameterInfo(BaseModel):
    name: str
    type_annotation: str | None = None
    default_value: str | None = None
    description: str | None = None


class ReturnInfo(BaseModel):
    type_annotation: str | None = None
    description: str = ""


class CallerCalleeInfo(BaseModel):
    function_name: str
    file_path: str
    node_id: str


class ExplainResponse(BaseModel):
    function_name: str
    file_path: str
    start_line: int
    end_line: int
    summary: str
    parameters: list[ParameterInfo] = Field(default_factory=list)
    returns: ReturnInfo = Field(default_factory=ReturnInfo)
    side_effects: list[str] = Field(default_factory=list)
    callers: list[CallerCalleeInfo] = Field(default_factory=list)
    callees: list[CallerCalleeInfo] = Field(default_factory=list)
    complexity_score: int = 1
    provider_used: str = ""
    model_used: str = ""
    explanation_ms: float = 0.0
