import time
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from loguru import logger

from backend.app.core.config import (
    MAX_SEQ_LEN, PAD_TOKEN, RANKING_FEATURE_COLS, ALL_GENRES
)
from backend.app.core.model_loader import ModelLoader


class FeatureService:
    """
    Provides fast in-memory feature operations for the recommendation pipeline.
    Depends on ModelLoader being initialized first.
    """

    def __init__(self, model_loader: ModelLoader):
        self.ml           = model_loader
        self.device       = model_loader.device

    # ─────────────────────────────────────────────────────────
    # 1. USER SEQUENCE LOOKUP
    # ─────────────────────────────────────────────────────────

    def get_user_sequence(self, user_id: int) -> Tuple[torch.Tensor, bool]:
        """
        Retrieve and pad the user's interaction sequence.

        Args:
            user_id: raw user ID (integer)

        Returns:
            (sequence_tensor [1, MAX_SEQ_LEN], is_known_user)
            sequence_tensor: padded integer sequence ready for Two-Tower input
            is_known_user: False if user_id not in training data (cold-start)

        Padding rule:
            Left-pad with PAD_TOKEN=0, keep most recent MAX_SEQ_LEN items.
            This matches the training-time padding exactly.
        """
        user_idx_str = str(user_id)

        # Check if user exists
        if user_idx_str not in self.ml.user2idx:
            logger.warning(f"Unknown user_id={user_id} (cold-start)")
            # Return a zero sequence (will produce a generic user embedding)
            seq = [PAD_TOKEN] * MAX_SEQ_LEN
            return torch.tensor([seq], dtype=torch.long), False

        user_idx = self.ml.user2idx[user_idx_str]

        # Get interaction history
        if user_idx not in self.ml.interactions_dict:
            logger.warning(f"user_idx={user_idx} has no interactions")
            seq = [PAD_TOKEN] * MAX_SEQ_LEN
            return torch.tensor([seq], dtype=torch.long), False

        history = self.ml.interactions_dict[user_idx]  # list of item_idxs

        # Use only the train portion (exclude last 2 items = val + test)
        # During serving, we use all known interactions
        train_history = history[:-1] if len(history) > 1 else history

        # Truncate to MAX_SEQ_LEN (keep most recent)
        if len(train_history) >= MAX_SEQ_LEN:
            seq = train_history[-MAX_SEQ_LEN:]
        else:
            # Left-pad with PAD_TOKEN
            pad_len = MAX_SEQ_LEN - len(train_history)
            seq     = [PAD_TOKEN] * pad_len + train_history

        tensor = torch.tensor([seq], dtype=torch.long)  # shape: [1, MAX_SEQ_LEN]
        return tensor, True

    def get_user_idx(self, user_id: int) -> Optional[int]:
        """Returns internal user_idx for a raw user_id, or None if unknown."""
        user_idx_str = str(user_id)
        if user_idx_str in self.ml.user2idx:
            return self.ml.user2idx[user_idx_str]
        return None

    # ─────────────────────────────────────────────────────────
    # 2. RANKING FEATURE CONSTRUCTION
    # ─────────────────────────────────────────────────────────

    def build_ranking_features(
        self,
        user_id:        int,
        candidate_idxs: np.ndarray,
        faiss_scores:   np.ndarray,
    ) -> pd.DataFrame:
        n_candidates = len(candidate_idxs)
        user_idx = self.get_user_idx(user_id)

        # ── User features ───────────────────────────────────────
        u_feats = {}
        if user_idx is not None and user_idx in self.ml.user_features_dict:
            u_feats = self.ml.user_features_dict[user_idx]

        # Map actual column names to expected names
        user_interaction_count_log = u_feats.get("user_rating_count", 0.0)
        user_avg_rating            = u_feats.get("user_avg_rating", 3.5)
        user_recency_days          = u_feats.get("user_days_since_last", 365.0)
        user_seq_length            = u_feats.get("user_sequence_length", 10.0)

        # ── User genre profile ──────────────────────────────────
        user_genres = self._get_user_genre_profile(user_id)

        # ── Build one row per candidate ─────────────────────────
        rows = []
        for rank_i, (item_idx, faiss_score) in enumerate(zip(candidate_idxs, faiss_scores)):
            item_idx = int(item_idx)

            i_feats = {}
            if item_idx in self.ml.item_features_dict:
                i_feats = self.ml.item_features_dict[item_idx]

            # Map actual item feature names
            item_log_popularity     = i_feats.get("item_log_popularity", i_feats.get("item_rating_count", 0.0))
            item_avg_rating         = i_feats.get("item_avg_rating", 3.0)
            item_rating_count_log   = i_feats.get("item_rating_count_log", i_feats.get("item_rating_count", 0.0))
            item_recency_days       = i_feats.get("item_recency_days", 730.0)
            item_year               = i_feats.get("item_year", 2000.0)
            item_active_days        = i_feats.get("item_active_days", 365.0)
            mf_score                = i_feats.get("mf_score", 0.0)

            # Genre overlap
            item_genres = self.ml.item_genres_dict.get(item_idx, [])
            genre_overlap = self._compute_genre_overlap(user_genres, item_genres)
            is_same_top_genre = self._is_same_top_genre(user_genres, item_genres)

            row = {
                # Retrieval features
                "faiss_score":             float(faiss_score),
                "faiss_rank_normalized":   (rank_i + 1) / n_candidates,

                # Item features
                "item_log_popularity":     item_log_popularity,
                "item_avg_rating":         item_avg_rating,
                "item_rating_count_log":   item_rating_count_log,
                "item_recency_days":       item_recency_days,
                "item_year":               item_year,
                "item_active_days":        item_active_days,

                # User features
                "user_interaction_count_log": user_interaction_count_log,
                "user_avg_rating":            user_avg_rating,
                "user_recency_days":          user_recency_days,
                "user_seq_length":            user_seq_length,

                # Interaction features
                "genre_overlap":           genre_overlap,
                "is_same_top_genre":       int(is_same_top_genre),
                "mf_score":                mf_score,
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Ensure all required columns exist
        for col in RANKING_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        return df[RANKING_FEATURE_COLS]

    def _get_user_genre_profile(self, user_id: int) -> Dict[str, int]:
        """
        Build a genre frequency profile from the user's interaction history.
        Returns: {genre_name: interaction_count}
        """
        user_idx = self.get_user_idx(user_id)
        if user_idx is None:
            return {}

        history  = self.ml.interactions_dict.get(user_idx, [])
        genre_counts: Dict[str, int] = {}
        for item_idx in history[-50:]:  # Use last 50 interactions for recency
            genres = self.ml.item_genres_dict.get(int(item_idx), [])
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + 1

        return genre_counts

    def _compute_genre_overlap(
        self,
        user_genres: Dict[str, int],
        item_genres: List[str]
    ) -> float:
        """
        Weighted genre overlap score.
        Each genre the user has interacted with before contributes proportionally.
        Returns: float in [0, 1]
        """
        if not user_genres or not item_genres:
            return 0.0

        total_user_interactions = sum(user_genres.values())
        overlap_score = 0.0
        for genre in item_genres:
            if genre in user_genres:
                overlap_score += user_genres[genre] / total_user_interactions

        # Normalize by number of item genres to avoid bias toward multi-genre items
        return overlap_score / max(len(item_genres), 1)

    def _is_same_top_genre(
        self,
        user_genres: Dict[str, int],
        item_genres: List[str]
    ) -> bool:
        """
        Returns True if the item belongs to the user's most-interacted genre.
        """
        if not user_genres or not item_genres:
            return False
        top_genre = max(user_genres, key=user_genres.get)
        return top_genre in item_genres

    # ─────────────────────────────────────────────────────────
    # 3. TITLE LOOKUP
    # ─────────────────────────────────────────────────────────

    def get_movie_title(self, item_idx: int) -> str:
        """Returns the movie title for a given item_idx."""
        return self.ml.idx2title.get(item_idx, f"Unknown Movie (idx={item_idx})")

    def get_movie_genres(self, item_idx: int) -> List[str]:
        """Returns the genre list for a given item_idx."""
        return self.ml.item_genres_dict.get(item_idx, [])