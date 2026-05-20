from pydantic import BaseModel
from uuid import UUID


class QueryRequest(BaseModel):
    repo_id: UUID
    question: str
    stream: bool = False


class SourceReference(BaseModel):
    file: str
    function: str | None = None
    lines: str | None = None


class QueryResponse(BaseModel):
    answer: str
    provider_used: str           # Which LLM answered: gemini / deepseek / openrouter
    model_used: str              # Exact model name
    sources: list[SourceReference] = []
    graph_path: list[str] = []