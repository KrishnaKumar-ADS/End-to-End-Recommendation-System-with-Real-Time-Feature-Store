from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from loguru import logger

from backend.app.schemas.request_response import (
    RecommendationResponse,
    ErrorResponse,
    MovieSuggestionResponse,
)
from backend.app.services.pipeline_service import PipelineService
from backend.app.core.config import DEFAULT_TOP_K
from backend.app.metrics import (
    ACTIVE_AB_EXPERIMENTS,
    CACHE_HIT_RATIO,
    RECOMMENDATION_LATENCY,
    RECOMMENDATION_REQUESTS,
)

router = APIRouter()
_recommend_requests = 0
_recommend_cache_hits = 0


def get_pipeline_service() -> PipelineService:
    """
    FastAPI dependency — returns the singleton PipelineService.
    The actual service is stored in app.state and accessed via this function.
    This function is overridden at startup (see main.py).
    """
    raise RuntimeError("PipelineService not initialized. Check startup.")


@router.get(
    "/recommend",
    response_model    = RecommendationResponse,
    summary           = "Get recommendations for a user",
    description       = """
    Returns top-k movie recommendations for the given user_id.

    **How it works internally:**
    1. Checks Redis cache (returns instantly if cached)
    2. Encodes user sequence via Two-Tower model (GPU)
    3. Retrieves 100 candidates via FAISS (ANN search)
    4. Re-ranks using LightGBM LambdaRank
    5. Returns top-k with movie titles and scores

    **Latency:**
    - Cache HIT: ~1-3ms
    - Cache MISS: ~25-40ms
    """,
    responses={
        200: {"description": "Successful recommendation response"},
        404: {"model": ErrorResponse, "description": "User not found (cold-start)"},
        422: {"description": "Invalid input parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["recommendations"],
)
async def get_recommendations(
    request: Request,
    http_response: Response,
    user_id: int = Query(
        ...,
        description = "User ID from the MovieLens dataset",
        ge          = 1,
        example     = 1,
    ),
    top_k: int = Query(
        default     = DEFAULT_TOP_K,
        description = "Number of recommendations to return",
        ge          = 1,
        le          = 50,
        example     = 10,
    ),
    experiment_id: str = Query(
        default="retrieval_v1",
        description="A/B experiment identifier for variant assignment"
    ),
    pipeline: PipelineService = Depends(get_pipeline_service),
) -> RecommendationResponse:
    """
    Main recommendation endpoint.

    Query Parameters:
        user_id (int, required): User ID. Must be a known user from training data.
        top_k   (int, optional): Number of recommendations. Default=10, Max=50.

    Returns:
        RecommendationResponse with ranked movie recommendations.

    Example:
        GET /recommend?user_id=1&top_k=10

    Example response:
        {
            "user_id": 1,
            "top_k": 10,
            "recommendations": [
                {"rank": 1, "item_id": 4886, "title": "Iron Man (2008)", "score": 0.94},
                ...
            ],
            "latency_ms": 28.4,
            "cached": false
        }
    """
    logger.info(
        f"→ /recommend user_id={user_id} top_k={top_k} "
        f"experiment_id={experiment_id}"
    )

    try:
        from backend.app.main import get_ab_router, get_ab_logger, get_bandit_service

        ab_router = get_ab_router()
        ab_logger = get_ab_logger()
        bandit_service = get_bandit_service()

        variant = ab_router.assign(user_id=user_id, experiment_id=experiment_id)

        response = await pipeline.recommend(user_id=user_id, top_k=top_k)

        # Apply Thompson Sampling rerank over the current top-k list.
        # If reranking fails, return the base ranking instead of failing the request.
        try:
            candidate_ids = [r.item_id for r in response.recommendations]
            lgbm_scores = [float(r.score) for r in response.recommendations]
            reranked = bandit_service.rerank(candidate_ids, lgbm_scores=lgbm_scores)

            item_lookup = {r.item_id: r for r in response.recommendations}
            reranked_items = []
            for rank_i, (item_id, bandit_score) in enumerate(reranked[:len(candidate_ids)], start=1):
                base_item = item_lookup.get(int(item_id))
                if base_item is None:
                    continue

                bounded_score = max(0.0, min(1.0, float(bandit_score)))
                reranked_items.append(base_item.model_copy(update={
                    "rank": rank_i,
                    "score": round(bounded_score, 4),
                    "bandit_score": round(float(bandit_score), 4),
                }))

            if len(reranked_items) == len(response.recommendations):
                response = response.model_copy(update={"recommendations": reranked_items})
        except Exception as e:
            logger.warning(f"Bandit rerank failed; returning base ranking: {e}")

        cache_hit = bool(response.cache_hit)
        request_id = getattr(request.state, "request_id", None)
        response_with_variant = response.model_copy(update={
            "variant": variant,
            "cache_hit": cache_hit,
            "cached": cache_hit,
            "request_id": request_id,
        })

        global _recommend_requests, _recommend_cache_hits
        _recommend_requests += 1
        if cache_hit:
            _recommend_cache_hits += 1

        CACHE_HIT_RATIO.set(_recommend_cache_hits / _recommend_requests)
        RECOMMENDATION_LATENCY.labels(variant=variant).observe(max(response_with_variant.latency_ms, 0.0) / 1000.0)
        RECOMMENDATION_REQUESTS.labels(
            variant=variant,
            status="cache_hit" if cache_hit else "success",
        ).inc()
        ACTIVE_AB_EXPERIMENTS.set(1)

        http_response.headers["X-AB-Variant"] = variant
        http_response.headers["X-Cache-Status"] = "hit" if cache_hit else "miss"
        http_response.headers["X-Cache"] = "hit" if cache_hit else "miss"

        try:
            ab_logger.log_exposure(
                user_id=user_id,
                variant=variant,
                experiment_id=experiment_id,
                n_recs=top_k,
                extra={
                    "cached": response_with_variant.cached,
                    "latency_ms": response_with_variant.latency_ms,
                },
            )
        except Exception as e:
            logger.warning(f"A/B exposure logging failed for user_id={user_id}: {e}")

        return response_with_variant

    except ValueError as e:
        # User not found (cold-start)
        logger.warning(f"  User not found: {e}")
        RECOMMENDATION_REQUESTS.labels(variant="unknown", status="error").inc()
        raise HTTPException(
            status_code = 404,
            detail      = {
                "error":   "UserNotFound",
                "detail":  str(e),
                "user_id": user_id,
            }
        )

    except RuntimeError as e:
        # System error (FAISS failure, model error, etc.)
        logger.error(f"  Runtime error: {e}")
        RECOMMENDATION_REQUESTS.labels(variant="unknown", status="error").inc()
        raise HTTPException(
            status_code = 500,
            detail      = {
                "error":  "InternalError",
                "detail": str(e),
            }
        )

    except Exception as e:
        # Unexpected errors
        logger.exception(f"  Unexpected error for user_id={user_id}: {e}")
        RECOMMENDATION_REQUESTS.labels(variant="unknown", status="error").inc()
        raise HTTPException(
            status_code = 500,
            detail      = {
                "error":  "UnexpectedError",
                "detail": "An unexpected error occurred. Check server logs.",
            }
        )


@router.get(
    "/recommend/by-movie",
    response_model=RecommendationResponse,
    summary="Get movie-seeded recommendations",
    description="""
    Returns top-k movies similar to a provided movie title.

    This endpoint is useful for discovery flows like "If you liked this movie..."
    and does not require a user_id.
    """,
    responses={
        200: {"description": "Successful movie-seeded recommendation response"},
        404: {"model": ErrorResponse, "description": "Movie title not found"},
        422: {"description": "Invalid input parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["recommendations"],
)
async def get_recommendations_by_movie(
    movie_title: str = Query(
        ...,
        min_length=1,
        description="Movie title seed for similar-movie retrieval",
        example="Toy Story",
    ),
    top_k: int = Query(
        default=DEFAULT_TOP_K,
        description="Number of similar movies to return",
        ge=1,
        le=50,
        example=10,
    ),
    pipeline: PipelineService = Depends(get_pipeline_service),
) -> RecommendationResponse:
    logger.info(f"→ /recommend/by-movie movie_title='{movie_title}' top_k={top_k}")

    try:
        return await pipeline.recommend_by_movie_title(movie_title=movie_title, top_k=top_k)

    except ValueError as e:
        logger.warning(f"  Movie not found: {e}")
        suggestions = pipeline.suggest_movie_titles(movie_title=movie_title, limit=6)
        raise HTTPException(
            status_code=404,
            detail={
                "error": "MovieNotFound",
                "detail": str(e),
                "movie_title": movie_title,
                "suggestions": suggestions,
            },
        )

    except RuntimeError as e:
        logger.error(f"  Runtime error in movie-seeded recommendation: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalError",
                "detail": str(e),
            },
        )

    except Exception as e:
        logger.exception(f"  Unexpected movie-seeded error for title='{movie_title}': {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "UnexpectedError",
                "detail": "An unexpected error occurred. Check server logs.",
            },
        )


@router.get(
    "/recommend/movie-suggestions",
    response_model=MovieSuggestionResponse,
    summary="Suggest movie titles from a keyword",
    description="""
    Returns keyword-based movie title suggestions.

    Use this endpoint while users type in the movie search box.
    """,
    responses={
        200: {"description": "Keyword suggestions returned"},
        422: {"description": "Invalid input parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["recommendations"],
)
async def get_movie_title_suggestions(
    query: str = Query(
        ...,
        min_length=1,
        description="Keyword fragment typed by the user",
        example="spider",
    ),
    limit: int = Query(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of title suggestions to return",
        example=8,
    ),
    pipeline: PipelineService = Depends(get_pipeline_service),
) -> MovieSuggestionResponse:
    try:
        suggestions = pipeline.suggest_movie_titles(movie_title=query, limit=limit)
        return MovieSuggestionResponse(query=query, suggestions=suggestions)

    except Exception as e:
        logger.exception(f"  Unexpected suggestion error for query='{query}': {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "UnexpectedError",
                "detail": "Could not fetch movie suggestions.",
            },
        )