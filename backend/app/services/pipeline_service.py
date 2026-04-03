import time
import re
import difflib
from datetime import datetime, timezone
from loguru import logger

from backend.app.core.model_loader import ModelLoader
from backend.app.core.config import TOP_K_RETRIEVAL, DEFAULT_TOP_K
from backend.app.schemas.request_response import RecommendationResponse, RecommendationItem
from backend.app.services.cache_service import CacheService
from backend.app.services.feature_service import FeatureService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.ranking_service import RankingService
class PipelineService:
    """
    End-to-end recommendation pipeline.
    All sub-services are injected (created once, shared across requests).
    """

    def __init__(
        self,
        model_loader:     ModelLoader,
        cache_service:    CacheService,
        feature_service:  FeatureService,
        retrieval_service: RetrievalService,
        ranking_service:  RankingService,
    ):
        self.ml        = model_loader
        self.cache     = cache_service
        self.features  = feature_service
        self.retrieval = retrieval_service
        self.ranking   = ranking_service

        self._title_search_rows: list[tuple[int, str, str]] = []
        self._title_lookup_by_normalized: dict[str, list[tuple[int, str]]] = {}

        for raw_item_idx, raw_title in self.ml.idx2title.items():
            item_idx = int(raw_item_idx)
            title = str(raw_title)
            normalized = self._normalize_title(title)
            if not normalized:
                continue

            self._title_search_rows.append((item_idx, title, normalized))
            self._title_lookup_by_normalized.setdefault(normalized, []).append(
                (item_idx, title)
            )

        self._normalized_title_keys = sorted(self._title_lookup_by_normalized.keys())

    @staticmethod
    def _normalize_title(title: str) -> str:
        cleaned = (title or "").strip().lower()
        cleaned = re.sub(r"\s*\(\d{4}\)$", "", cleaned)
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _suggest_movie_matches(
        self,
        movie_title: str,
        limit: int = 8,
    ) -> list[tuple[int, str]]:
        query = self._normalize_title(movie_title)
        if not query or limit <= 0:
            return []

        matches: list[tuple[int, str]] = []
        seen_item_idxs: set[int] = set()

        def add_match(item_idx: int, title: str) -> None:
            if item_idx in seen_item_idxs:
                return
            seen_item_idxs.add(item_idx)
            matches.append((item_idx, title))

        for item_idx, title in self._title_lookup_by_normalized.get(query, []):
            add_match(item_idx, title)
            if len(matches) >= limit:
                return matches

        query_tokens = query.split()
        ranked_partial: list[tuple[int, int, str, int, str]] = []

        for item_idx, title, normalized in self._title_search_rows:
            score = 0
            if normalized.startswith(query):
                score = 300 - max(0, len(normalized) - len(query))
            elif query in normalized:
                score = 220 - normalized.index(query)
            elif query_tokens:
                tokens = normalized.split()
                if all(any(token.startswith(q_token) for token in tokens) for q_token in query_tokens):
                    score = 180

            if score > 0:
                ranked_partial.append((score, len(normalized), title.lower(), item_idx, title))

        ranked_partial.sort(key=lambda row: (-row[0], row[1], row[2]))
        for _, _, _, item_idx, title in ranked_partial:
            add_match(item_idx, title)
            if len(matches) >= limit:
                return matches

        fuzzy_keys = difflib.get_close_matches(
            query,
            self._normalized_title_keys,
            n=max(limit * 2, 10),
            cutoff=0.65,
        )
        for normalized in fuzzy_keys:
            for item_idx, title in self._title_lookup_by_normalized.get(normalized, []):
                add_match(item_idx, title)
                if len(matches) >= limit:
                    return matches

        return matches

    def suggest_movie_titles(self, movie_title: str, limit: int = 8) -> list[str]:
        return [title for _, title in self._suggest_movie_matches(movie_title=movie_title, limit=limit)]

    def _find_item_idx_by_title(self, movie_title: str) -> tuple[int, str]:
        if not self._normalize_title(movie_title):
            raise ValueError("movie_title must not be empty")

        matches = self._suggest_movie_matches(movie_title=movie_title, limit=1)
        if matches:
            return matches[0]

        raise ValueError(f"No movie found matching title '{movie_title}'")

    def _resolve_seed_item_idx_for_retrieval(self, seed_item_idx: int) -> int:
        """
        Ensure the selected seed item index exists in the FAISS map.

        Some pipelines use a +1 offset for retrieval item indices where 0 is reserved.
        """
        retrieval_map = self.retrieval.item_idx_to_faiss_pos

        if seed_item_idx in retrieval_map:
            return seed_item_idx

        if (seed_item_idx + 1) in retrieval_map:
            return seed_item_idx + 1

        if seed_item_idx > 0 and (seed_item_idx - 1) in retrieval_map:
            return seed_item_idx - 1

        raise ValueError(
            f"seed_item_idx={seed_item_idx} not found in FAISS map"
        )

    async def recommend(
        self,
        user_id: int,
        top_k:   int = DEFAULT_TOP_K,
    ) -> RecommendationResponse:
        """
        Generate top-k recommendations for a user.
        This is the main method called by the API endpoint.

        Args:
            user_id: raw integer user ID (from the API request)
            top_k:   number of final recommendations to return

        Returns:
            RecommendationResponse with ranked recommendations + metadata

        Raises:
            ValueError: if user_id is not found (cold-start)
        """
        request_start = time.time()

        # ── Step 1: Cache Lookup ─────────────────────────────
        cached_result = self.cache.get(user_id, top_k)
        if cached_result is not None:
            # Cache HIT — reconstruct response with updated latency/timestamp
            total_ms = (time.time() - request_start) * 1000
            return RecommendationResponse(
                user_id         = user_id,
                top_k           = top_k,
                recommendations = [RecommendationItem(**item) for item in cached_result],
                source_type     = "user",
                latency_ms      = round(total_ms, 2),
                cache_hit       = True,
                cached          = True,
                timestamp       = datetime.now(timezone.utc).isoformat(),
            )

        # ── Step 2: Get User Sequence ────────────────────────
        sequence_tensor, is_known_user = self.features.get_user_sequence(user_id)

        if not is_known_user:
            raise ValueError(
                f"user_id={user_id} not found in training data. "
                f"Cold-start users are not yet supported. "
                f"Valid user IDs are in the range used during training."
            )

        # ── Step 3: Two-Tower Retrieval (GPU) ────────────────
        candidate_idxs, faiss_scores = self.retrieval.retrieve(
            sequence_tensor, top_k=TOP_K_RETRIEVAL
        )

        if len(candidate_idxs) == 0:
            raise RuntimeError(
                f"FAISS retrieval returned 0 candidates for user_id={user_id}. "
                f"Check FAISS index integrity."
            )

        # ── Step 4: Feature Construction ────────────────────
        feature_df = self.features.build_ranking_features(
            user_id        = user_id,
            candidate_idxs = candidate_idxs,
            faiss_scores   = faiss_scores,
        )

        # ── Step 5: LightGBM Ranking ─────────────────────────
        top_k_item_idxs, top_k_scores = self.ranking.rank(
            feature_df     = feature_df,
            candidate_idxs = candidate_idxs,
            top_k          = top_k,
        )

        # ── Step 6: Build Response ───────────────────────────
        recommendations = []
        for rank_i, (item_idx, score) in enumerate(
            zip(top_k_item_idxs, top_k_scores), start=1
        ):
            item_idx = int(item_idx)
            recommendations.append(RecommendationItem(
                rank    = rank_i,
                item_id = item_idx,
                title   = self.features.get_movie_title(item_idx),
                score   = round(float(score), 4),
                genres  = self.features.get_movie_genres(item_idx),
            ))

        total_ms = (time.time() - request_start) * 1000

        # ── Step 7: Cache Store ──────────────────────────────
        cache_payload = [item.model_dump() for item in recommendations]
        self.cache.set(user_id, top_k, cache_payload)

        # ── Step 8: Return Response ──────────────────────────
        response = RecommendationResponse(
            user_id         = user_id,
            top_k           = top_k,
            recommendations = recommendations,
            source_type     = "user",
            latency_ms      = round(total_ms, 2),
            cache_hit       = False,
            cached          = False,
            timestamp       = datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            f"✅ recommend user_id={user_id} top_k={top_k} "
            f"latency={total_ms:.1f}ms cached=False"
        )

        return response

    async def recommend_by_movie_title(
        self,
        movie_title: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> RecommendationResponse:
        """
        Generate top-k similar movies based on a seed movie title.
        """
        request_start = time.time()

        seed_item_idx, matched_title = self._find_item_idx_by_title(movie_title)
        resolved_seed_item_idx = self._resolve_seed_item_idx_for_retrieval(seed_item_idx)

        similar_item_idxs, similarity_scores = self.retrieval.retrieve_similar_items(
            seed_item_idx=resolved_seed_item_idx,
            top_k=top_k,
        )

        if len(similar_item_idxs) == 0:
            raise RuntimeError(
                f"No similar movies found for seed title '{matched_title}'"
            )

        recommendations = []
        for rank_i, (item_idx, sim_score) in enumerate(
            zip(similar_item_idxs, similarity_scores), start=1
        ):
            item_idx = int(item_idx)
            raw_similarity = float(sim_score)
            normalized_score = max(0.0, min(1.0, (raw_similarity + 1.0) / 2.0))
            recommendations.append(
                RecommendationItem(
                    rank=rank_i,
                    item_id=item_idx,
                    title=self.features.get_movie_title(item_idx),
                    score=round(normalized_score, 4),
                    similarity_score=round(raw_similarity, 4),
                    genres=self.features.get_movie_genres(item_idx),
                )
            )

        total_ms = (time.time() - request_start) * 1000

        return RecommendationResponse(
            user_id=0,
            top_k=len(recommendations),
            recommendations=recommendations,
            variant="item_seed",
            source_type="movie",
            seed_title=matched_title,
            seed_item_id=int(resolved_seed_item_idx),
            latency_ms=round(total_ms, 2),
            cache_hit=False,
            cached=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def health_status(self) -> dict:
        """Aggregate health from all sub-services."""
        return {
            "two_tower":     self.retrieval.health_check(),
            "faiss_index":   self.retrieval.health_check(),
            "lgbm_ranker":   self.ranking.health_check(),
            "redis":         self.cache.health_check(),
            "feature_store": {
                "status": "healthy",
                "detail": (
                    f"{len(self.ml.user_features_dict):,} users, "
                    f"{len(self.ml.item_features_dict):,} items loaded"
                )
            }
        }
