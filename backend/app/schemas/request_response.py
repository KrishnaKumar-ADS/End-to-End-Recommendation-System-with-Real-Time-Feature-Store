from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# /recommend ENDPOINT SCHEMAS
# ─────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    """A single recommended item."""
    rank:     int   = Field(..., description="Rank (1 = best)", ge=1)
    item_id:  int   = Field(..., description="Internal item index")
    movie_id: Optional[int] = Field(default=None, description="Raw MovieLens movie ID when available")
    title:    str   = Field(..., description="Movie title from MovieLens")
    score:    float = Field(..., description="LightGBM relevance score", ge=0.0, le=1.0)
    bandit_score: Optional[float] = Field(default=None, description="Bandit reranking score when available")
    similarity_score: Optional[float] = Field(default=None, description="Retriever similarity score when available")
    genres:   Optional[List[str]] = Field(default=[], description="Movie genres")

    class Config:
        json_schema_extra = {
            "example": {
                "rank": 1,
                "item_id": 4886,
                "title": "Iron Man (2008)",
                "score": 0.9412,
                "genres": ["Action", "Adventure", "Sci-Fi"]
            }
        }


class RecommendationResponse(BaseModel):
    """Full response from /recommend endpoint."""
    user_id:         int                     = Field(..., description="Input user ID")
    top_k:           int                     = Field(..., description="Number of recommendations returned")
    recommendations: List[RecommendationItem] = Field(..., description="Ranked recommendation list")
    variant:         Optional[str]           = Field(default=None, description="A/B variant assigned for this request")
    source_type:     Optional[str]           = Field(default="user", description="Recommendation source: user or movie")
    seed_title:      Optional[str]           = Field(default=None, description="Seed movie title when source_type=movie")
    seed_item_id:    Optional[int]           = Field(default=None, description="Seed item index when source_type=movie")
    latency_ms:      float                   = Field(..., description="Total server-side latency in ms")
    cache_hit:       bool                    = Field(..., description="True if result was served from Redis cache")
    cached:          Optional[bool]          = Field(default=None, description="Backward-compatible cache flag")
    request_id:      Optional[str]           = Field(default=None, description="Request correlation ID when available")
    timestamp:       str                     = Field(..., description="UTC timestamp of response")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "top_k": 10,
                "recommendations": [
                    {"rank": 1, "item_id": 4886, "title": "Iron Man (2008)",
                     "score": 0.94, "genres": ["Action", "Sci-Fi"]},
                    {"rank": 2, "item_id": 1032, "title": "Toy Story (1995)",
                     "score": 0.87, "genres": ["Animation", "Children"]}
                ],
                "variant": "sasrec",
                "source_type": "user",
                "latency_ms": 28.4,
                "cached": False,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class MovieSuggestionResponse(BaseModel):
    """Response from movie-title suggestion endpoint."""
    query: str = Field(..., description="Original query text from user input")
    suggestions: List[str] = Field(
        default_factory=list,
        description="Ordered movie title suggestions",
    )


# ─────────────────────────────────────────────────────────────
# /feedback ENDPOINT SCHEMAS
# ─────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """User click feedback — used for future online learning."""
    user_id:    int   = Field(..., description="User who clicked", ge=1)
    item_id:    int   = Field(..., description="Item that was clicked", ge=1)
    reward:     float = Field(
        default=1.0,
        description="Reward signal: 1.0=click, 0.0=skip, 0.5=partial",
        ge=0.0, le=1.0
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for grouping feedback events"
    )
    variant: Optional[str] = Field(
        default=None,
        description="A/B variant associated with the recommendation exposure"
    )
    experiment_id: Optional[str] = Field(
        default="retrieval_v1",
        description="Experiment identifier used for A/B analytics"
    )
    context:    Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context (position, page, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "item_id": 4886,
                "reward": 1.0,
                "session_id": "sess_abc123",
                "context": {"position": 1, "page": "home"}
            }
        }


class FeedbackResponse(BaseModel):
    """Response after receiving feedback."""
    status:    str   = Field(..., description="'ok' or 'error'")
    user_id:   int
    item_id:   int
    message:   str
    bandit_updated: Optional[bool] = Field(default=None)
    ab_logged:      Optional[bool] = Field(default=None)
    variant:        Optional[str] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "accepted",
                "user_id": 123,
                "item_id": 4886,
                "message": "Feedback logged successfully",
                "bandit_updated": True,
                "ab_logged": True,
                "variant": "sasrec"
            }
        }


# ─────────────────────────────────────────────────────────────
# /health ENDPOINT SCHEMAS
# ─────────────────────────────────────────────────────────────

class ComponentStatus(BaseModel):
    """Status of a single backend component."""
    name:    str
    status:  str   = Field(..., description="'ok', 'degraded', or 'down'")
    latency_ms: Optional[float] = None
    detail:  Optional[str] = None


class HealthResponse(BaseModel):
    """Full system health check."""
    status:     str              = Field(..., description="Overall: 'healthy', 'degraded', or 'unhealthy'")
    version:    str              = Field(default="1.0.0")
    components: Dict[str, Dict[str, Any]]
    metrics:    Dict[str, Any]
    uptime_seconds: float        = Field(..., description="Server uptime in seconds")
    timestamp:  Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": [
                    {"name": "two_tower", "status": "healthy", "detail": "cuda:0"},
                    {"name": "faiss_index", "status": "healthy", "detail": "26744 vectors"},
                    {"name": "lgbm_ranker", "status": "healthy"},
                    {"name": "redis", "status": "healthy", "detail": "localhost:6379"},
                    {"name": "feature_store", "status": "healthy", "detail": "120K users loaded"}
                ],
                "uptime_s": 3600.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# ─────────────────────────────────────────────────────────────
# ERROR SCHEMAS
# ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Structured error response."""
    error:   str   = Field(..., description="Error type")
    detail:  str   = Field(..., description="Human-readable error message")
    user_id: Optional[int] = None
    movie_title: Optional[str] = None
    suggestions: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "error": "UserNotFound",
                "detail": "user_id=99999 not found in training data. Cold-start users are not supported yet.",
                "user_id": 99999
            }
        }