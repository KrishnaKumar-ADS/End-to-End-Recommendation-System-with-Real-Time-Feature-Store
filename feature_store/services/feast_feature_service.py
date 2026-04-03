import json
import time
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION (inline defaults — override via config.py)
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
REPO_PATH       = PROJECT_ROOT / "feature_store" / "feature_repo"
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
RAW_DIR         = PROJECT_ROOT / "data" / "raw"
MAX_SEQ_LEN     = 50
PAD_IDX         = 0


# ─────────────────────────────────────────────────────────────
# FEAST FEATURE NAMES
# ─────────────────────────────────────────────────────────────

USER_FEATURE_REFS = [
    "user_features_view:avg_rating",
    "user_features_view:total_interactions",
    "user_features_view:rating_std",
    "user_features_view:min_rating",
    "user_features_view:max_rating",
    "user_features_view:high_rating_ratio",
    "user_features_view:active_days",
    "user_features_view:interaction_density",
    "user_features_view:temporal_spread_days",
    "user_features_view:recency_days",
    "user_features_view:session_count",
    "user_features_view:avg_session_length",
    "user_features_view:genre_diversity",
    "user_features_view:top_genre_idx",
    "user_features_view:popularity_bias",
]

ITEM_FEATURE_REFS = [
    "item_features_view:global_popularity",
    "item_features_view:popularity_rank",
    "item_features_view:niche_score",
    "item_features_view:avg_ratings_per_day",
    "item_features_view:avg_item_rating",
    "item_features_view:rating_count",
    "item_features_view:rating_std",
    "item_features_view:high_rating_ratio",
    "item_features_view:genre_count",
    "item_features_view:primary_genre_idx",
    "item_features_view:release_year",
    "item_features_view:item_age_days",
]

# Shortened names for DataFrame columns (strip the view prefix)
USER_FEAT_COLS = [f.split(":")[1] for f in USER_FEATURE_REFS]
ITEM_FEAT_COLS = [f.split(":")[1] for f in ITEM_FEATURE_REFS]

# Rename clashes: both user and item have 'rating_std', 'high_rating_ratio'
USER_FEAT_RENAME = {c: f"user_{c}" for c in USER_FEAT_COLS}
ITEM_FEAT_RENAME = {c: f"item_{c}" for c in ITEM_FEAT_COLS}


# ─────────────────────────────────────────────────────────────
# FEAST FEATURE SERVICE CLASS
# ─────────────────────────────────────────────────────────────

class FeastFeatureService:
    """
    Drop-in replacement for Week 6 FeatureService.
    Uses Feast + Redis for all feature lookups.
    """

    def __init__(self):
        self._store        = None    # Feast FeatureStore (lazy load)
        self._interactions  = None   # user_idx → item_idx list
        self._user2idx      = None   # userId → user_idx
        self._idx2item      = None   # item_idx → movieId
        self._idx2title     = None   # item_idx → title string
        self._item2idx      = None   # movieId → item_idx
        self._n_items       = None

        self._user_feat_cache  = {}  # user_idx → feast result dict (hot cache)
        self._item_feat_cache  = {}  # item_idx → feast result dict (hot cache)
        self._cache_max_size   = 10_000

        logger.info("FeastFeatureService initialized (lazy loading)")

    # ─────────────────────────────────────────────────────────
    # LAZY INITIALIZATION
    # ─────────────────────────────────────────────────────────

    def _ensure_store(self):
        """Lazy-loads the Feast FeatureStore."""
        if self._store is not None:
            return
        from feast import FeatureStore
        logger.info(f"Loading Feast FeatureStore from {REPO_PATH}")
        t0 = time.time()
        self._store = FeatureStore(repo_path=str(REPO_PATH))
        logger.info(f"Feast store loaded in {time.time()-t0:.2f}s")

    def _ensure_interactions(self):
        """Lazy-loads user interaction sequences."""
        if self._interactions is not None:
            return
        logger.info("Loading user interactions...")
        t0 = time.time()

        df = pd.read_csv(
            PROCESSED_DIR / "interactions.csv",
            dtype={"user_idx": np.int32, "item_idx": np.int32, "timestamp": np.int64}
        )
        # Build user_idx → sorted list of item_idxs
        self._interactions = (
            df.sort_values("timestamp")
            .groupby("user_idx")["item_idx"]
            .apply(list)
            .to_dict()
        )
        logger.info(f"Interactions loaded: {len(self._interactions):,} users in {time.time()-t0:.2f}s")

    def _ensure_mappings(self):
        """Lazy-loads all index mappings and title lookup."""
        if self._idx2title is not None:
            return
        logger.info("Loading mappings...")

        # item_idx → movieId
        with open(PROCESSED_DIR / "idx2item.json") as f:
            self._idx2item = {int(k): int(v) for k, v in json.load(f).items()}

        # movieId → item_idx
        with open(PROCESSED_DIR / "item2idx.json") as f:
            self._item2idx = {int(k): int(v) for k, v in json.load(f).items()}

        # user userId → user_idx
        user2idx_path = PROCESSED_DIR / "user2idx.json"
        if user2idx_path.exists():
            with open(user2idx_path) as f:
                self._user2idx = {int(k): int(v) for k, v in json.load(f).items()}
        else:
            self._user2idx = {}

        # item_idx → movie title
        movies = pd.read_csv(RAW_DIR / "movies.csv", usecols=["movieId", "title"])
        movieid2title = dict(zip(movies["movieId"], movies["title"]))
        self._idx2title = {
            idx: movieid2title.get(mid, f"Unknown(movieId={mid})")
            for idx, mid in self._idx2item.items()
        }

        meta_path = PROCESSED_DIR / "dataset_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._n_items = meta.get("n_items", len(self._item2idx))
        else:
            self._n_items = len(self._item2idx)

        logger.info(f"Mappings loaded: {self._n_items:,} items")

    # ─────────────────────────────────────────────────────────
    # PUBLIC API — SEQUENCE RETRIEVAL (unchanged from Week 6)
    # ─────────────────────────────────────────────────────────

    def get_user_sequence(
        self,
        user_id: int,
        as_tensor: bool = True,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Returns the user's interaction history as a padded sequence tensor.
        Shape: [1, MAX_SEQ_LEN]
        Identical interface to Week 6 FeatureService.
        """
        self._ensure_interactions()
        self._ensure_mappings()

        # Map userId → user_idx if needed
        user_idx = self._user2idx.get(user_id, user_id)

        history = self._interactions.get(user_idx, [])

        # Take the most recent MAX_SEQ_LEN items
        history = history[-MAX_SEQ_LEN:]

        # Pad with PAD_IDX (0) at the front
        if len(history) < MAX_SEQ_LEN:
            padded = [PAD_IDX] * (MAX_SEQ_LEN - len(history)) + history
        else:
            padded = history

        if as_tensor:
            return torch.tensor([padded], dtype=torch.long).to(device)
        return np.array([padded], dtype=np.int32)

    # ─────────────────────────────────────────────────────────
    # PUBLIC API — FEAST FEATURE RETRIEVAL
    # ─────────────────────────────────────────────────────────

    def get_feast_user_features(self, user_idx: int) -> Dict[str, float]:
        """
        Fetches all 15 user features from Redis via Feast.
        
        Uses a local hot cache to avoid repeat Redis calls
        for the same user within a single request lifecycle.
        
        Returns: dict with keys matching USER_FEAT_COLS
        Falls back to default values if user not in store.
        """
        self._ensure_store()

        # Hot cache check
        if user_idx in self._user_feat_cache:
            return self._user_feat_cache[user_idx]

        try:
            result = self._store.get_online_features(
                entity_rows=[{"user_idx": user_idx}],
                features=USER_FEATURE_REFS
            ).to_dict()

            # Strip view prefix from keys
            features = {}
            for ref, col in zip(USER_FEATURE_REFS, USER_FEAT_COLS):
                val = result.get(col, [None])[0]
                if val is None:
                    val = self._default_user_feature(col)
                features[col] = float(val) if val is not None else 0.0

        except Exception as e:
            logger.warning(f"Feast user features fetch failed for user_idx={user_idx}: {e}")
            features = {col: self._default_user_feature(col) for col in USER_FEAT_COLS}

        # Update hot cache (LRU-style eviction)
        if len(self._user_feat_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._user_feat_cache))
            del self._user_feat_cache[oldest_key]
        self._user_feat_cache[user_idx] = features

        return features

    def get_feast_item_features(
        self,
        item_idxs: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """
        Batch-fetches item features from Redis via Feast.
        Returns: {item_idx: {feature_name: value}}
        
        Batch retrieval is more efficient than per-item calls.
        Feast internally pipelines the Redis GET requests.
        """
        self._ensure_store()

        # Split into cached and uncached
        cached = {idx: self._item_feat_cache[idx] for idx in item_idxs
                  if idx in self._item_feat_cache}
        uncached = [idx for idx in item_idxs if idx not in self._item_feat_cache]

        if not uncached:
            return cached

        try:
            entity_rows = [{"item_idx": idx} for idx in uncached]
            result = self._store.get_online_features(
                entity_rows=entity_rows,
                features=ITEM_FEATURE_REFS
            ).to_dict()

            new_features = {}
            n = len(uncached)
            for i, item_idx in enumerate(uncached):
                item_dict = {}
                for ref, col in zip(ITEM_FEATURE_REFS, ITEM_FEAT_COLS):
                    vals = result.get(col, [None] * n)
                    val = vals[i] if i < len(vals) else None
                    if val is None:
                        val = self._default_item_feature(col)
                    item_dict[col] = float(val) if val is not None else 0.0
                new_features[item_idx] = item_dict

        except Exception as e:
            logger.warning(f"Feast item features batch fetch failed: {e}")
            new_features = {
                idx: {col: self._default_item_feature(col) for col in ITEM_FEAT_COLS}
                for idx in uncached
            }

        # Update hot cache
        for item_idx, features in new_features.items():
            if len(self._item_feat_cache) >= self._cache_max_size:
                oldest = next(iter(self._item_feat_cache))
                del self._item_feat_cache[oldest]
            self._item_feat_cache[item_idx] = features

        return {**cached, **new_features}

    # ─────────────────────────────────────────────────────────
    # PUBLIC API — RANKING FEATURE BUILDER
    # ─────────────────────────────────────────────────────────

    def build_ranking_features(
        self,
        user_id: int,
        candidate_item_idxs: np.ndarray,
        faiss_scores: np.ndarray,
        user_history: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Builds the full feature matrix [n_candidates × n_features]
        for LightGBM ranking.

        Feature groups:
          - 15 user features (from Feast/Redis)
          - 12 item features (from Feast/Redis, batched)
          - 2 FAISS scores (similarity score + rank)
          - 1 genre overlap (computed inline)
          - Total: ~30 features

        This method is the KEY difference from Week 6:
          Week 6: features loaded from in-memory Pandas dict (12ms)
          Week 7: features fetched from Redis via Feast (2-4ms)
                  AND same features used in training (no skew)
        """
        self._ensure_mappings()

        n = len(candidate_item_idxs)
        user_idx = self._user2idx.get(user_id, user_id)

        # ── 1. Fetch user features (single Redis call) ──
        user_feats = self.get_feast_user_features(user_idx)

        # ── 2. Fetch item features (batch Redis call) ──
        item_feats_dict = self.get_feast_item_features(candidate_item_idxs.tolist())

        # ── 3. Get user interaction history for genre overlap ──
        self._ensure_interactions()
        if user_history is None:
            history = self._interactions.get(user_idx, [])[-50:]
        else:
            history = user_history

        # User's genre profile (from their interaction history)
        user_genres = self._get_user_genre_profile(history)

        # ── 4. Build feature DataFrame ──
        rows = []
        for i, item_idx in enumerate(candidate_item_idxs):
            item_f = item_feats_dict.get(int(item_idx), {})

            # Genre overlap: fraction of item's genre in user's genre profile
            item_genre_idx = int(item_f.get("primary_genre_idx", -1))
            genre_overlap = float(user_genres.get(item_genre_idx, 0.0))

            row = {
                # User features (prefixed to avoid column collisions)
                "u_avg_rating":           user_feats.get("avg_rating", 0.0),
                "u_total_interactions":   user_feats.get("total_interactions", 0.0),
                "u_rating_std":           user_feats.get("rating_std", 0.0),
                "u_min_rating":           user_feats.get("min_rating", 0.0),
                "u_max_rating":           user_feats.get("max_rating", 0.0),
                "u_high_rating_ratio":    user_feats.get("high_rating_ratio", 0.0),
                "u_active_days":          user_feats.get("active_days", 0.0),
                "u_interaction_density":  user_feats.get("interaction_density", 0.0),
                "u_temporal_spread_days": user_feats.get("temporal_spread_days", 0.0),
                "u_recency_days":         user_feats.get("recency_days", 0.0),
                "u_session_count":        user_feats.get("session_count", 0.0),
                "u_avg_session_length":   user_feats.get("avg_session_length", 0.0),
                "u_genre_diversity":      user_feats.get("genre_diversity", 0.0),
                "u_top_genre_idx":        user_feats.get("top_genre_idx", 0.0),
                "u_popularity_bias":      user_feats.get("popularity_bias", 9999.0),

                # Item features
                "i_global_popularity":    item_f.get("global_popularity", 0.0),
                "i_popularity_rank":      item_f.get("popularity_rank", 99999.0),
                "i_niche_score":          item_f.get("niche_score", 1.0),
                "i_avg_ratings_per_day":  item_f.get("avg_ratings_per_day", 0.0),
                "i_avg_item_rating":      item_f.get("avg_item_rating", 0.0),
                "i_rating_count":         item_f.get("rating_count", 0.0),
                "i_rating_std":           item_f.get("rating_std", 0.0),
                "i_high_rating_ratio":    item_f.get("high_rating_ratio", 0.0),
                "i_genre_count":          item_f.get("genre_count", 0.0),
                "i_primary_genre_idx":    item_f.get("primary_genre_idx", 0.0),
                "i_release_year":         item_f.get("release_year", 0.0),
                "i_item_age_days":        item_f.get("item_age_days", 0.0),

                # Retrieval signal
                "faiss_score":            float(faiss_scores[i]),
                "faiss_rank":             float(i + 1),

                # Cross features
                "genre_overlap":          genre_overlap,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────
    # PUBLIC API — TITLE LOOKUP
    # ─────────────────────────────────────────────────────────

    def get_movie_titles(self, item_idxs: List[int]) -> List[str]:
        """Maps item_idx → movie title string."""
        self._ensure_mappings()
        return [
            self._idx2title.get(int(idx), f"Unknown(item_idx={idx})")
            for idx in item_idxs
        ]

    # ─────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────

    def _get_user_genre_profile(self, history: List[int]) -> Dict[int, float]:
        """
        Computes a normalized genre frequency dict from user's item history.
        Returns: {genre_idx: frequency_fraction}
        """
        self._ensure_interactions()
        genre_counts: Dict[int, int] = {}
        total = 0
        for item_idx in history:
            # We need genre info for items in history — use cached item features
            item_f = self._item_feat_cache.get(item_idx)
            if item_f is not None:
                g = int(item_f.get("primary_genre_idx", -1))
                if g >= 0:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
                    total += 1

        if total == 0:
            return {}
        return {g: count / total for g, count in genre_counts.items()}

    def _default_user_feature(self, col: str) -> float:
        """Returns a sensible default value for missing user features."""
        defaults = {
            "avg_rating": 3.5,
            "total_interactions": 0.0,
            "rating_std": 0.5,
            "min_rating": 1.0,
            "max_rating": 5.0,
            "high_rating_ratio": 0.5,
            "active_days": 1.0,
            "interaction_density": 1.0,
            "temporal_spread_days": 0.0,
            "recency_days": 365.0,
            "session_count": 1.0,
            "avg_session_length": 1.0,
            "genre_diversity": 5.0,
            "top_genre_idx": 4.0,  # Comedy
            "popularity_bias": 5000.0,
        }
        return defaults.get(col, 0.0)

    def _default_item_feature(self, col: str) -> float:
        """Returns a sensible default value for missing item features."""
        defaults = {
            "global_popularity": 0.0,
            "popularity_rank": 99999.0,
            "niche_score": 1.0,
            "avg_ratings_per_day": 0.0,
            "avg_item_rating": 3.0,
            "rating_count": 0.0,
            "rating_std": 0.5,
            "high_rating_ratio": 0.5,
            "genre_count": 1.0,
            "primary_genre_idx": 7.0,  # Drama
            "release_year": 2000.0,
            "item_age_days": 8760.0,   # 24 years
        }
        return defaults.get(col, 0.0)

    def clear_hot_cache(self):
        """Clears the in-memory hot cache (use between requests if needed)."""
        self._user_feat_cache.clear()
        self._item_feat_cache.clear()
        logger.info("Hot cache cleared")