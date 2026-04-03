import time
import numpy as np
import pandas as pd
from typing import Tuple
from loguru import logger

from backend.app.core.config import TOP_K_FINAL, RANKING_FEATURE_COLS
from backend.app.core.model_loader import ModelLoader


class RankingService:
    """
    LightGBM-based re-ranking of FAISS candidates.
    """

    def __init__(self, model_loader: ModelLoader):
        self.ml = model_loader

    def _to_float_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert a feature DataFrame to a float32 matrix robustly.

        Some online feature tables may contain timedelta/datetime/object columns.
        LightGBM expects numeric values only.
        """
        clean_df = df.copy()

        for col in clean_df.columns:
            series = clean_df[col]

            if pd.api.types.is_timedelta64_dtype(series):
                clean_df[col] = series.dt.total_seconds()
                continue

            if pd.api.types.is_datetime64_any_dtype(series):
                # Convert datetimes to unix seconds for numeric model input.
                clean_df[col] = series.view("int64") / 1_000_000_000
                continue

            if pd.api.types.is_bool_dtype(series):
                clean_df[col] = series.astype(np.float32)
                continue

            if not pd.api.types.is_numeric_dtype(series):
                clean_df[col] = pd.to_numeric(series, errors="coerce")

        clean_df = clean_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return clean_df.to_numpy(dtype=np.float32)

    def rank(
        self,
        feature_df:     pd.DataFrame,
        candidate_idxs: np.ndarray,
        top_k:          int = TOP_K_FINAL,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rank candidates using LightGBM and return top-k.

        Args:
            feature_df:     DataFrame [n_candidates × len(RANKING_FEATURE_COLS)]
                            Must have exactly the columns in RANKING_FEATURE_COLS
            candidate_idxs: np.ndarray [n_candidates] — item_idxs from FAISS
            top_k:          int — how many final items to return

        Returns:
            top_k_item_idxs: np.ndarray [top_k] — final ranked item indices
            top_k_scores:    np.ndarray [top_k] — LightGBM relevance scores (in [0,1])

        Pipeline:
            1. Validate feature columns
            2. LightGBM.predict(feature_matrix) → raw scores
            3. Normalize scores to [0, 1]
            4. Sort descending
            5. Return top-k indices and scores
        """
        t0 = time.time()

        n_candidates = len(candidate_idxs)

        if n_candidates == 0:
            logger.warning("RankingService: No candidates to rank!")
            return np.array([]), np.array([])

        # Align to the model's training schema when available.
        # This keeps inference robust even if online feature names evolved.
        expected_cols = []
        if hasattr(self.ml.lgbm_model, "feature_name"):
            expected_cols = list(self.ml.lgbm_model.feature_name() or [])

        if expected_cols:
            aligned_df = feature_df.copy()

            rename_map = {
                "user_interaction_count_log": "u_user_rating_count",
                "user_avg_rating": "u_user_avg_rating",
                "user_recency_days": "u_user_days_since_last",
                "user_seq_length": "u_user_sequence_length",
                "item_avg_rating": "i_item_avg_rating",
                "item_log_popularity": "i_item_log_popularity",
                "item_year": "i_item_year",
                "item_active_days": "i_item_active_days",
                "item_recency_days": "i_item_days_since_last_rating",
                "is_same_top_genre": "genre_max_match",
            }
            aligned_df = aligned_df.rename(columns=rename_map)

            if "i_item_rating_count" not in aligned_df.columns and "item_rating_count_log" in aligned_df.columns:
                # Some pipelines persist count values in this column instead of log-counts.
                # Clip before expm1 to avoid numeric overflow.
                rating_count_log = aligned_df["item_rating_count_log"].astype(np.float32).to_numpy()
                aligned_df["i_item_rating_count"] = np.expm1(np.clip(rating_count_log, -20.0, 20.0))

            if "faiss_rank" not in aligned_df.columns:
                aligned_df["faiss_rank"] = np.arange(1, n_candidates + 1, dtype=np.float32)

            if "faiss_score_normalized" not in aligned_df.columns:
                faiss_vals = aligned_df["faiss_score"].astype(np.float32).to_numpy()
                if len(faiss_vals) > 0:
                    vmin = float(np.min(faiss_vals))
                    vmax = float(np.max(faiss_vals))
                    denom = vmax - vmin
                    if denom > 1e-8:
                        aligned_df["faiss_score_normalized"] = (faiss_vals - vmin) / denom
                    else:
                        aligned_df["faiss_score_normalized"] = np.zeros_like(faiss_vals)
                else:
                    aligned_df["faiss_score_normalized"] = 0.0

            if "faiss_score_gap" not in aligned_df.columns:
                faiss_vals = aligned_df["faiss_score"].astype(np.float32).to_numpy()
                if len(faiss_vals) > 0:
                    aligned_df["faiss_score_gap"] = float(faiss_vals[0]) - faiss_vals
                else:
                    aligned_df["faiss_score_gap"] = 0.0

            if "item_popularity_pct" not in aligned_df.columns:
                if "i_item_log_popularity" in aligned_df.columns:
                    pop_vals = aligned_df["i_item_log_popularity"].astype(np.float32).to_numpy()
                    if len(pop_vals) > 0:
                        pmin = float(np.min(pop_vals))
                        pmax = float(np.max(pop_vals))
                        pden = pmax - pmin
                        if pden > 1e-8:
                            aligned_df["item_popularity_pct"] = (pop_vals - pmin) / pden
                        else:
                            aligned_df["item_popularity_pct"] = np.zeros_like(pop_vals)
                    else:
                        aligned_df["item_popularity_pct"] = 0.0
                else:
                    aligned_df["item_popularity_pct"] = 0.0

            if "group_id" not in aligned_df.columns:
                aligned_df["group_id"] = 0.0

            for col in expected_cols:
                if col not in aligned_df.columns:
                    aligned_df[col] = 0.0

            feature_matrix = self._to_float_matrix(aligned_df[expected_cols])
        else:
            feature_matrix = self._to_float_matrix(feature_df[RANKING_FEATURE_COLS])

        # LightGBM predict
        raw_scores = self.ml.lgbm_model.predict(feature_matrix)  # [n_candidates]

        # Normalize to [0, 1] using sigmoid
        # LambdaRank scores are not bounded — sigmoid maps them to a probability-like range
        normalized_scores = 1.0 / (1.0 + np.exp(-raw_scores))

        # Sort descending by score
        sorted_indices = np.argsort(normalized_scores)[::-1]  # highest score first

        # Take top-k
        top_k = min(top_k, n_candidates)
        top_k_indices = sorted_indices[:top_k]

        top_k_item_idxs = candidate_idxs[top_k_indices].astype(int)
        top_k_scores    = normalized_scores[top_k_indices]

        ranking_ms = (time.time() - t0) * 1000
        logger.debug(
            f"Ranking: {n_candidates} → top-{top_k} "
            f"in {ranking_ms:.1f}ms "
            f"(top score: {top_k_scores[0]:.4f})"
        )

        return top_k_item_idxs, top_k_scores

    def health_check(self) -> dict:
        """Check ranking model status."""
        try:
            n_trees = self.ml.lgbm_model.num_trees()
            n_features = (
                len(self.ml.lgbm_model.feature_name())
                if hasattr(self.ml.lgbm_model, "feature_name")
                else len(RANKING_FEATURE_COLS)
            )
            return {
                "status": "healthy",
                "detail": f"{n_trees} trees, {n_features} features"
            }
        except Exception as e:
            return {"status": "unavailable", "detail": str(e)}