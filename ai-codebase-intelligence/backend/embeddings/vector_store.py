"""
Qdrant Vector Store — manages the code_chunks collection.

This file handles everything related to Qdrant:
  - Creating the collection at startup
  - Inserting/updating vectors (upsert)
  - Searching by vector similarity
  - Filtering by repo, language, chunk type
  - Deleting vectors when a repo is re-indexed

The collection name is "code_chunks".
Each point in the collection has:
  - id:      UUID string (same as CodeChunk.id in PostgreSQL)
  - vector:  768 floats from Gemini text-embedding-004
  - payload: metadata dict for filtering
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
)
from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION_NAME = "code_chunks"
VECTOR_SIZE = 768           # Gemini text-embedding-004 output dimension
DISTANCE = Distance.COSINE  # Cosine similarity — standard for text embeddings

# ── Client singleton ──────────────────────────────────────────────────────────

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """
    Return a singleton Qdrant client.
    Creates one connection and reuses it — safe because Qdrant
    handles concurrent connections internally.
    """
    global _client
    if _client is None:
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )
        logger.info(
            "qdrant_client_created",
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    return _client


# ── Collection management ─────────────────────────────────────────────────────

def ensure_collection_exists() -> None:
    """
    Create the code_chunks collection if it does not already exist.

    Called at:
      - Application startup (in main.py lifespan)
      - Before any embedding pipeline run

    Safe to call multiple times — fully idempotent.
    """
    client = get_client()

    existing_names = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=DISTANCE,
                on_disk=False,  # keep vectors in RAM for fast search
            ),
        )
        logger.info(
            "qdrant_collection_created",
            collection=COLLECTION_NAME,
            vector_size=VECTOR_SIZE,
            distance=str(DISTANCE),
        )
    else:
        logger.debug("qdrant_collection_already_exists", collection=COLLECTION_NAME)


def get_collection_info() -> dict:
    """
    Return basic statistics about the collection.
    Used by the /health and /search/stats endpoints.
    """
    client = get_client()
    info = client.get_collection(COLLECTION_NAME)
    return {
        "name": COLLECTION_NAME,
        "vectors_count": info.vectors_count or 0,
        "points_count": info.points_count or 0,
        "status": str(info.status),
        "vector_size": VECTOR_SIZE,
        "distance": str(DISTANCE),
    }


# ── Upsert operations ─────────────────────────────────────────────────────────

def upsert_chunk(
    chunk_id: str,
    vector: list[float],
    payload: dict,
) -> None:
    """
    Add or update a single chunk vector in Qdrant.

    If a point with this chunk_id already exists, it will be
    overwritten (upsert = insert or update).

    Args:
        chunk_id: UUID string matching CodeChunk.id in PostgreSQL
        vector:   768 floats from Gemini embedding
        payload:  metadata dict (repo_id, file_path, name, etc.)
    """
    if len(vector) != VECTOR_SIZE:
        raise ValueError(
            f"Vector must be {VECTOR_SIZE}-dimensional, got {len(vector)}"
        )

    client = get_client()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload=payload,
            )
        ],
    )
    logger.debug("qdrant_chunk_upserted", chunk_id=chunk_id)


def upsert_chunks_batch(
    chunks: list[tuple[str, list[float], dict]],
    batch_size: int = 100,
) -> int:
    """
    Efficiently upload many chunk vectors to Qdrant in batches.

    Qdrant recommends batches of 100-500 points for best performance.
    We default to 100 to stay memory-efficient.

    Args:
        chunks:     list of (chunk_id, vector, payload) tuples
        batch_size: number of points per upload request

    Returns:
        Total number of points successfully upserted
    """
    client = get_client()
    total_upserted = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        points = [
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload=payload,
            )
            for chunk_id, vector, payload in batch
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,  # wait for indexing to complete before returning
        )
        total_upserted += len(batch)

        logger.debug(
            "qdrant_batch_upserted",
            batch_start=i,
            batch_size=len(batch),
            total_so_far=total_upserted,
        )

    logger.info(
        "qdrant_full_upsert_complete",
        total=total_upserted,
        batches=len(range(0, len(chunks), batch_size)),
    )
    return total_upserted


# ── Search operations ─────────────────────────────────────────────────────────

def search(
    query_vector: list[float],
    repo_id: str,
    top_k: int = 10,
    language: str | None = None,
    chunk_type: str | None = None,
    score_threshold: float = 0.3,
) -> list[ScoredPoint]:
    """
    Find the most similar code chunks to a query vector.

    The query vector is the Gemini embedding of the user's question.
    We always filter by repo_id so results only come from the right repo.

    Args:
        query_vector:    768-dimensional vector from embed_query()
        repo_id:         only return chunks from this repository
        top_k:           maximum number of results (default 10, max 20)
        language:        optional — filter to one language only
        chunk_type:      optional — "function", "class", "method"
        score_threshold: minimum cosine similarity (0.0 to 1.0)
                         0.3 = loosely related, 0.7 = strongly similar

    Returns:
        List of ScoredPoint, sorted by score descending (best first)
        Each ScoredPoint has: .id, .score, .payload
    """
    if len(query_vector) != VECTOR_SIZE:
        raise ValueError(
            f"Query vector must be {VECTOR_SIZE}-dimensional, got {len(query_vector)}"
        )

    client = get_client()

    # Build filter conditions — always filter by repo_id
    must_conditions = [
        FieldCondition(
            key="repo_id",
            match=MatchValue(value=repo_id),
        )
    ]

    # Add optional language filter
    if language:
        must_conditions.append(
            FieldCondition(
                key="language",
                match=MatchValue(value=language),
            )
        )

    # Add optional chunk_type filter
    if chunk_type:
        must_conditions.append(
            FieldCondition(
                key="chunk_type",
                match=MatchValue(value=chunk_type),
            )
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=Filter(must=must_conditions),
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False,  # don't return vectors in results (saves bandwidth)
    )

    logger.info(
        "qdrant_search_complete",
        repo_id=repo_id,
        query_language=language,
        query_chunk_type=chunk_type,
        results_count=len(results),
        top_score=results[0].score if results else 0.0,
    )

    return results


def search_similar_chunks(
    chunk_id: str,
    repo_id: str,
    top_k: int = 5,
) -> list[ScoredPoint]:
    """
    Find chunks similar to a specific chunk (by its id).
    Used for "find related functions" feature.

    Args:
        chunk_id: UUID of the reference chunk
        repo_id:  only return chunks from the same repo
        top_k:    number of results

    Returns:
        List of similar ScoredPoint objects (excluding the query chunk itself)
    """
    client = get_client()

    # Qdrant's recommend endpoint finds similar points by id
    results = client.recommend(
        collection_name=COLLECTION_NAME,
        positive=[chunk_id],
        negative=[],
        limit=top_k + 1,  # +1 because results may include the chunk itself
        query_filter=Filter(
            must=[
                FieldCondition(key="repo_id", match=MatchValue(value=repo_id))
            ]
        ),
        with_payload=True,
    )

    # Remove the query chunk itself from results
    return [r for r in results if str(r.id) != chunk_id][:top_k]


# ── Delete operations ─────────────────────────────────────────────────────────

def delete_repo_vectors(repo_id: str) -> None:
    """
    Delete all vectors belonging to a repository.
    Called when a repo is being re-indexed from scratch.
    """
    client = get_client()

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="repo_id",
                    match=MatchValue(value=repo_id),
                )
            ]
        ),
        wait=True,
    )

    logger.info("qdrant_repo_vectors_deleted", repo_id=repo_id)


def count_repo_vectors(repo_id: str) -> int:
    """
    Count how many vectors are stored for a given repository.
    Used to verify embedding completeness.
    """
    client = get_client()

    result = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="repo_id",
                    match=MatchValue(value=repo_id),
                )
            ]
        ),
        exact=True,
    )

    return result.count