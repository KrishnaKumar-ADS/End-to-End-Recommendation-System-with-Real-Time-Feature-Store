"""Schemas for the Week 6 ask endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    repo_id: str = Field(..., description="Repository UUID.")
    question: str = Field(..., min_length=5, max_length=2000)
    session_id: str | None = Field(default=None, description="Conversation session UUID.")
    task_type: Literal["code_qa", "reasoning", "security", "summarize", "architecture"] | None = None
    stream: bool = False
    top_k: int = Field(default=8, ge=1, le=20)
    include_graph: bool = True
    language_filter: str | None = None
    chunk_type_filter: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace only.")
        return value.strip()


class SourceReference(BaseModel):
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    score: float
    chunk_type: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    session_id: str | None = None
    provider_used: str
    model_used: str
    task_type: str
    sources: list[SourceReference]
    graph_path: list[str]
    context_chunks_used: int
    estimated_tokens: int
    vector_search_ms: float
    graph_expansion_ms: float
    total_latency_ms: float
    top_result_score: float
    quality_score: dict | None = None
    cached: bool = False
    cache_similarity: float = 0.0
    intent: str | None = None


class DeepAskRequest(BaseModel):
    repo_id: str = Field(..., description="Repository UUID.")
    question: str = Field(..., min_length=5, max_length=3000)
    session_id: str | None = Field(default=None, description="Optional conversation session UUID.")


class DeepAskResponse(BaseModel):
    answer: str
    sub_questions: list[str]
    partial_answers: list[str]
    all_sources: list[dict]
    providers_used: list[str]
    decomposed: bool
    total_ms: float
