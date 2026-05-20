"""
Gemini Embedder — generates vector embeddings using Google's
gemini-embedding-001 model via the generativeai SDK.

Key facts about this model:
    Output dimensions : 768 floats (requested via output_dimensionality)
  Max input tokens  : 2048 per request
  Free tier         : 100 requests/minute, 1500 requests/day
  Task types        :
    "retrieval_document" — used when indexing code chunks
    "retrieval_query"    — used when embedding user search queries
    (using the right task type improves search accuracy by ~10%)

Rate limiting strategy we use:
  - Process chunks in batches of 20
  - Sleep 0.8 seconds between batches
  - This keeps us at ~25 requests/second, well under 100/min limit
  - Automatic retry with exponential backoff on errors
  - Truncate inputs that are too long before sending

How we build the embedding text for a code chunk:
  We don't just embed the raw code. We build a richer text that includes:
    1. The chunk type and name (e.g. "Function: hash_password")
    2. The docstring if available
    3. The actual source code
  This gives the model more context and produces better embeddings.
"""

import logging
import hashlib
import math
import re
import time
import google.generativeai as genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,       # ← takes predicate (exception) -> bool
    before_sleep_log,
)
from caching.cache_manager import get_cache_manager
from core.config import get_settings
from core.logging import get_logger
from core.exceptions import EmbeddingError

logger = get_logger(__name__)
settings = get_settings()

# ── Model constants ───────────────────────────────────────────────────────────

EMBEDDING_MODEL     = settings.resolved_embedding_model
VECTOR_DIMENSIONS   = settings.embedding_vector_dim
MAX_INPUT_CHARS     = 8000    # 2048 tokens ≈ 8000 chars (rough estimate)
BATCH_SIZE          = 20      # chunks per batch before sleeping
BATCH_DELAY_SECS    = 0.8     # seconds to sleep between batches

TASK_TYPE_DOCUMENT  = "retrieval_document"  # for indexing code
TASK_TYPE_QUERY     = "retrieval_query"     # for user search queries

_PROVIDER_FAILURE_HINTS = (
    "429",
    "quota",
    "resourceexhausted",
    "rate limit",
    "deadline",
    "timeout",
    "timed out",
    "connection",
    "unavailable",
    "api key",
    "permission denied",
)


# ── Client setup ──────────────────────────────────────────────────────────────

def _configure_genai() -> None:
    """
    Configure the Gemini SDK with the API key from settings.
    Raises EmbeddingError if the key is not set.
    """
    if not settings.gemini_api_key:
        raise EmbeddingError(
            "GEMINI_API_KEY is not set in your .env file. "
            "Get your free key at: https://aistudio.google.com/app/apikey"
        )
    genai.configure(api_key=settings.gemini_api_key)


# ── Text preprocessing ────────────────────────────────────────────────────────

def _prepare_text(text: str) -> str:
    """
    Clean and truncate text before sending to Gemini.

    What we do:
      1. Strip leading/trailing whitespace
      2. Remove null bytes (they cause API errors)
      3. Truncate to MAX_INPUT_CHARS if too long
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Strip whitespace
    text = text.strip()

    # Truncate if too long
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    return text


def _build_embedding_text(chunk: dict) -> str:
    """
    Build an enriched text representation of a code chunk for embedding.

    Format:
        {ChunkType}: {display_name}
        Docstring: {docstring}
        Code:
        {content}

    Example output:
        Function: AuthService.verify_password
        Docstring: Check if plain password matches stored hash using bcrypt.
        Code:
        def verify_password(self, plain: str, hashed: str) -> bool:
            return bcrypt.checkpw(plain.encode(), hashed.encode())

    Why this format:
        - Including the name helps find functions by name
        - Including docstring captures human-readable intent
        - Including code captures implementation details
        - The model understands all three together
    """
    parts = []

    chunk_type  = chunk.get("chunk_type", "")
    display     = chunk.get("display_name") or chunk.get("name", "")
    docstring   = chunk.get("docstring", "").strip()
    content     = chunk.get("content", "").strip()

    if chunk_type and display:
        parts.append(f"{chunk_type.capitalize()}: {display}")

    if docstring:
        parts.append(f"Docstring: {docstring}")

    if content:
        parts.append(f"Code:\n{content}")

    return "\n".join(parts)


def _local_hash_embedding(text: str) -> list[float]:
    """
    Build a deterministic local embedding when remote provider calls fail.

    This uses feature hashing over tokens, then L2-normalizes the vector.
    It is not as semantically strong as Gemini embeddings, but keeps the
    ingestion pipeline operational and queryable in offline/quota-exhausted
    conditions.
    """
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", text.lower())
    vector = [0.0] * VECTOR_DIMENSIONS

    if not tokens:
        vector[0] = 1.0
        return vector

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:4], byteorder="big", signed=False) % VECTOR_DIMENSIONS
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        magnitude = 0.5 + (digest[5] / 255.0)
        vector[index] += sign * magnitude

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        vector[0] = 1.0
        return vector

    return [v / norm for v in vector]


def _should_use_local_fallback(exc: Exception) -> bool:
    """
    Decide whether provider failures should fall back to local embeddings.
    """
    if not settings.embedding_allow_local_fallback:
        return False

    error_text = str(exc).lower()
    return any(hint in error_text for hint in _PROVIDER_FAILURE_HINTS)


# ── Single embedding functions ────────────────────────────────────────────────

# WHY retry_if_exception(lambda) and NOT retry_if_exception_type(Exception):
#
#   retry_if_exception_type(Exception) retries on ALL exceptions including
#   EmbeddingError — our own validation error. Retrying those never helps.
#
#   retry_if_exception(predicate) calls predicate(exception) -> bool.
#   Returning False for EmbeddingError makes it propagate immediately.
#   Returning True for everything else (network errors, 429s, 500s) retries.
#
# WHY NOT a plain function passed to retry= directly:
#
#   retry= requires a retry_base instance. A plain callable is misinterpreted
#   — tenacity calls it with retry_state (not the exception), it always
#   evaluates truthy, so tenacity retries even on SUCCESS, exhausts all
#   3 attempts, then raises RetryError even when the call worked fine.

@retry(
    retry=retry_if_exception(lambda e: not isinstance(e, EmbeddingError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(
        logging.getLogger("embeddings.gemini_embedder"), logging.WARNING
    ),
)
def embed_text(
    text: str,
    task_type: str = TASK_TYPE_DOCUMENT,
) -> list[float]:
    """
    Embed a single piece of text using Gemini embedding models.

    Retries automatically up to 3 times with exponential backoff.
    This handles:
      - Temporary network errors
      - 429 rate limit errors (waits before retrying)
      - Transient 500 server errors

    Does NOT retry EmbeddingError — those are validation failures
    (empty text, missing API key) that will never succeed on retry.

    Args:
        text:      Raw text to embed (code, query, etc.)
        task_type: "retrieval_document" for indexing, "retrieval_query" for search

    Returns:
        List of 768 floats representing the semantic meaning of the text

    Raises:
        EmbeddingError: immediately (no retry) if text is empty or blank
        EmbeddingError: if the API returns wrong number of dimensions
        tenacity.RetryError: if all retries exhausted on a transient error
    """
    prepared = _prepare_text(text)
    if not prepared:
        raise EmbeddingError("Cannot embed empty or blank text")

    try:
        _configure_genai()

        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=prepared,
            task_type=task_type,
            output_dimensionality=VECTOR_DIMENSIONS,
        )

        vector = result["embedding"]
        if len(vector) != VECTOR_DIMENSIONS:
            raise EmbeddingError(
                f"Expected {VECTOR_DIMENSIONS} dimensions, got {len(vector)}"
            )

        return vector
    except EmbeddingError as exc:
        if _should_use_local_fallback(exc):
            logger.warning("embedding_local_fallback", reason="provider_config", error=str(exc))
            return _local_hash_embedding(prepared)
        raise
    except Exception as exc:
        if _should_use_local_fallback(exc):
            logger.warning("embedding_local_fallback", reason="provider_runtime", error=str(exc))
            return _local_hash_embedding(prepared)
        raise


def embed_query(query: str) -> list[float]:
    """
    Embed a user search query.

    Uses "retrieval_query" task type which is tuned for search queries
    (as opposed to documents). This gives better results than
    using the document task type for queries.

    Args:
        query: The user's natural language search query

    Returns:
        768-dimensional vector ready for Qdrant similarity search
    """
    normalized = query.strip()
    cache = get_cache_manager()
    cached = cache.get_embedding(normalized)
    if cached is not None:
        logger.debug("embedding_cache_hit_query", preview=normalized[:60])
        return cached

    vector = embed_text(normalized, task_type=TASK_TYPE_QUERY)
    cache.set_embedding(normalized, vector)
    return vector


def embed_chunk(chunk: dict) -> list[float]:
    """
    Embed a single code chunk dict.

    Builds enriched text (name + docstring + code) then embeds it.
    Uses "retrieval_document" task type.

    Args:
        chunk: dict with keys: chunk_type, name, display_name, docstring, content

    Returns:
        768-dimensional vector
    """
    text = _build_embedding_text(chunk)
    cache = get_cache_manager()
    cached = cache.get_embedding(text)
    if cached is not None:
        logger.debug("embedding_cache_hit_chunk", chunk_id=chunk.get("id", "unknown"))
        return cached

    vector = embed_text(text, task_type=TASK_TYPE_DOCUMENT)
    cache.set_embedding(text, vector)
    return vector


# ── Batch embedding ───────────────────────────────────────────────────────────

def embed_chunks_batch(
    chunks: list[dict],
    progress_callback=None,
) -> list[tuple[str, list[float]]]:
    """
    Embed a list of code chunks in rate-limit-safe batches.

    Processing:
      - Groups chunks into batches of BATCH_SIZE (20)
      - Embeds each chunk in the batch one by one
      - Sleeps BATCH_DELAY_SECS (0.8s) between batches
      - Skips and logs any chunks that fail (non-fatal)
      - Calls progress_callback(done, total) after each batch

    Args:
        chunks:            list of chunk dicts — must have "id" field
        progress_callback: optional function(done: int, total: int)
                           called after each batch completes

    Returns:
        List of (chunk_id, vector) tuples for successful embeddings.
        Failed chunks are NOT included — check logs for details.
    """
    results: list[tuple[str, list[float]]] = []
    total   = len(chunks)
    failed  = 0

    if total == 0:
        return results

    logger.info("embed_batch_start", total=total, batch_size=BATCH_SIZE)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        for chunk in batch:
            chunk_id = chunk.get("id", "unknown")
            try:
                vector = embed_chunk(chunk)
                results.append((chunk_id, vector))
            except Exception as e:
                logger.warning(
                    "chunk_embed_failed",
                    chunk_id=chunk_id,
                    name=chunk.get("name", "?"),
                    error=str(e),
                )
                failed += 1

        done = min(batch_start + BATCH_SIZE, total)

        if progress_callback:
            progress_callback(done, total)

        logger.info(
            "embed_batch_progress",
            done=done,
            total=total,
            succeeded=len(results),
            failed=failed,
        )

        # Rate limit delay — skip for the very last batch
        if batch_start + BATCH_SIZE < total:
            time.sleep(BATCH_DELAY_SECS)

    logger.info(
        "embed_batch_complete",
        total=total,
        succeeded=len(results),
        failed=failed,
        success_rate=f"{len(results)/total*100:.1f}%",
    )
    return results