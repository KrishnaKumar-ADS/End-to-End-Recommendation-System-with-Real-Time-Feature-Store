import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_PATH    = PROJECT_ROOT / "feature_store" / "feature_repo"
USER_PARQUET = PROJECT_ROOT / "feature_store" / "data" / "user_features.parquet"
ITEM_PARQUET = PROJECT_ROOT / "feature_store" / "data" / "item_features.parquet"

# PSI thresholds (industry standard):
PSI_SLIGHT_SHIFT = 0.1    # < 0.1: no significant change
PSI_MODERATE_SHIFT = 0.2  # 0.1-0.2: slight change, monitor
PSI_SEVERE_SHIFT = 0.25   # > 0.2: significant change, investigate


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI).
    
    PSI = Σ (actual% - expected%) × ln(actual% / expected%)
    
    PSI < 0.1:  No significant change
    PSI 0.1-0.2: Slight change, monitor
    PSI > 0.2:  Significant change, investigate
    
    Used in industry to monitor feature drift between training and serving.
    """
    # Create bins from expected distribution
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    if max_val == min_val:
        return 0.0  # Constant feature — no drift possible

    bins = np.linspace(min_val, max_val, n_bins + 1)
    bins[0]  -= 1e-10  # Include minimum
    bins[-1] += 1e-10  # Include maximum

    # Compute proportions
    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts   = np.histogram(actual, bins=bins)[0]

    # Add small epsilon to avoid division by zero / log(0)
    epsilon = 1e-6
    n_expected = len(expected)
    n_actual   = len(actual)

    expected_pct = (expected_counts + epsilon) / (n_expected + epsilon * n_bins)
    actual_pct   = (actual_counts   + epsilon) / (n_actual   + epsilon * n_bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_mean_diff(expected: np.ndarray, actual: np.ndarray) -> float:
    """Normalized mean absolute difference: |E[train] - E[serve]| / std[train]"""
    std = np.std(expected)
    if std < 1e-10:
        return abs(np.mean(expected) - np.mean(actual))
    return abs(np.mean(expected) - np.mean(actual)) / std


class SkewDetector:
    """
    Detects training-serving skew by comparing:
      - Offline Parquet (training distribution, sampled)
      - Online Redis via Feast (serving distribution, sampled)
    """

    def __init__(self, n_serving_samples: int = 1000):
        self.n_serving_samples = n_serving_samples
        self._store = None

    def _ensure_store(self):
        if self._store is not None:
            return
        from feast import FeatureStore
        self._store = FeatureStore(repo_path=str(REPO_PATH))

    def _load_training_distribution(
        self,
        parquet_path: Path,
        feature_cols: List[str],
        entity_col: str,
        n_samples: int = 5000
    ) -> pd.DataFrame:
        """
        Samples n_samples rows from the offline Parquet file.
        Returns DataFrame with feature_cols columns.
        """
        df = pd.read_parquet(parquet_path, columns=[entity_col] + feature_cols)
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42)
        return df

    def _fetch_serving_distribution(
        self,
        entity_col: str,
        entity_ids: List[int],
        feature_refs: List[str],
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Fetches n_serving_samples entities' features from online store.
        These represent the "serving distribution" — what the model sees in production.
        """
        self._ensure_store()

        sample_ids = entity_ids[:self.n_serving_samples]
        entity_rows = [{entity_col: eid} for eid in sample_ids]

        try:
            result = self._store.get_online_features(
                entity_rows=entity_rows,
                features=feature_refs
            ).to_dict()

            df = pd.DataFrame({col: result.get(col, [None] * len(sample_ids))
                               for col in feature_cols})
            df[entity_col] = sample_ids
            return df
        except Exception as e:
            print(f"  ⚠️  Failed to fetch serving features: {e}")
            return pd.DataFrame()

    def run_user_feature_skew_report(self) -> Dict:
        """
        Computes skew report for all user features.
        Returns dict with PSI and mean_diff for each feature.
        """
        print("\n  Running user feature skew detection...")

        USER_FEATURE_COLS = [
            "avg_rating", "total_interactions", "rating_std",
            "high_rating_ratio", "active_days", "interaction_density",
            "recency_days", "session_count", "genre_diversity", "popularity_bias"
        ]
        USER_FEATURE_REFS = [f"user_features_view:{c}" for c in USER_FEATURE_COLS]

        # Load training distribution
        print("  Loading training distribution from Parquet...")
        train_df = self._load_training_distribution(
            USER_PARQUET, USER_FEATURE_COLS, "user_idx"
        )

        # Fetch serving distribution
        print(f"  Fetching serving distribution from Redis ({self.n_serving_samples} samples)...")
        sample_user_idxs = train_df["user_idx"].tolist()[:self.n_serving_samples]
        serve_df = self._fetch_serving_distribution(
            "user_idx", sample_user_idxs, USER_FEATURE_REFS, USER_FEATURE_COLS
        )

        if serve_df.empty:
            print("  ❌ Could not fetch serving features — skipping skew detection")
            return {}

        # Compute skew metrics per feature
        results = {}
        print(f"\n  {'Feature':<30} {'PSI':>6}  {'MeanDiff':>9}  {'Status'}")
        print("  " + "-" * 70)

        for col in USER_FEATURE_COLS:
            if col not in train_df.columns or col not in serve_df.columns:
                continue

            train_vals = train_df[col].dropna().values.astype(float)
            serve_vals = serve_df[col].dropna().values.astype(float)

            if len(train_vals) == 0 or len(serve_vals) == 0:
                continue

            psi = compute_psi(train_vals, serve_vals)
            mean_diff = compute_mean_diff(train_vals, serve_vals)

            if psi < PSI_SLIGHT_SHIFT:
                status = "✅ OK"
            elif psi < PSI_MODERATE_SHIFT:
                status = "⚠️  SLIGHT"
            elif psi < PSI_SEVERE_SHIFT:
                status = "🔶 MODERATE"
            else:
                status = "❌ SEVERE"

            results[col] = {"psi": psi, "mean_diff": mean_diff, "status": status}
            print(f"  {col:<30} {psi:>6.3f}  {mean_diff:>9.3f}  {status}")

        return results

    def run_item_feature_skew_report(self) -> Dict:
        """Computes skew report for all item features."""
        print("\n  Running item feature skew detection...")

        ITEM_FEATURE_COLS = [
            "global_popularity", "avg_item_rating", "rating_count",
            "niche_score", "primary_genre_idx", "release_year", "item_age_days"
        ]
        ITEM_FEATURE_REFS = [f"item_features_view:{c}" for c in ITEM_FEATURE_COLS]

        print("  Loading training distribution from Parquet...")
        train_df = self._load_training_distribution(
            ITEM_PARQUET, ITEM_FEATURE_COLS, "item_idx"
        )

        print(f"  Fetching serving distribution from Redis...")
        sample_item_idxs = train_df["item_idx"].tolist()[:self.n_serving_samples]
        serve_df = self._fetch_serving_distribution(
            "item_idx", sample_item_idxs, ITEM_FEATURE_REFS, ITEM_FEATURE_COLS
        )

        if serve_df.empty:
            return {}

        results = {}
        print(f"\n  {'Feature':<30} {'PSI':>6}  {'MeanDiff':>9}  {'Status'}")
        print("  " + "-" * 70)

        for col in ITEM_FEATURE_COLS:
            if col not in train_df.columns or col not in serve_df.columns:
                continue

            train_vals = train_df[col].dropna().values.astype(float)
            serve_vals = serve_df[col].dropna().values.astype(float)

            if len(train_vals) == 0 or len(serve_vals) == 0:
                continue

            psi = compute_psi(train_vals, serve_vals)
            mean_diff = compute_mean_diff(train_vals, serve_vals)

            if psi < PSI_SLIGHT_SHIFT:
                status = "✅ OK"
            elif psi < PSI_MODERATE_SHIFT:
                status = "⚠️  SLIGHT"
            elif psi < PSI_SEVERE_SHIFT:
                status = "🔶 MODERATE"
            else:
                status = "❌ SEVERE"

            results[col] = {"psi": psi, "mean_diff": mean_diff, "status": status}
            print(f"  {col:<30} {psi:>6.3f}  {mean_diff:>9.3f}  {status}")

        return results


def main():
    print("=" * 70)
    print("  DS19 Week 7 — Training-Serving Skew Detection Report")
    print("=" * 70)
    print()
    print("  PSI Legend:")
    print("    PSI < 0.10: ✅ OK — no significant change")
    print("    PSI 0.10-0.20: ⚠️  SLIGHT — monitor and investigate")
    print("    PSI 0.20-0.25: 🔶 MODERATE — consider re-engineering features")
    print("    PSI > 0.25: ❌ SEVERE — re-run feature pipelines immediately")

    detector = SkewDetector(n_serving_samples=500)

    print("\n" + "═" * 70)
    print("  USER FEATURES")
    print("═" * 70)
    user_results = detector.run_user_feature_skew_report()

    print("\n" + "═" * 70)
    print("  ITEM FEATURES")
    print("═" * 70)
    item_results = detector.run_item_feature_skew_report()

    # Summary
    all_results = {**user_results, **item_results}
    severe = [f for f, r in all_results.items() if r.get("psi", 0) >= PSI_SEVERE_SHIFT]
    moderate = [f for f, r in all_results.items()
                if PSI_MODERATE_SHIFT <= r.get("psi", 0) < PSI_SEVERE_SHIFT]

    print("\n" + "=" * 70)
    print("  SKEW DETECTION SUMMARY")
    print("=" * 70)

    if not severe and not moderate:
        print("  ✅ NO SIGNIFICANT SKEW DETECTED")
        print("  Training and serving distributions are consistent.")
        print("  Feast feature store is working correctly.")
    else:
        if severe:
            print(f"  ❌ SEVERE SKEW ({len(severe)} features): {severe}")
            print("  Action: Re-run feature engineering pipelines + re-materialize")
        if moderate:
            print(f"  ⚠️  MODERATE SKEW ({len(moderate)} features): {moderate}")
            print("  Action: Monitor these features; consider re-engineering if persists")

    print("=" * 70)


if __name__ == "__main__":
    main()