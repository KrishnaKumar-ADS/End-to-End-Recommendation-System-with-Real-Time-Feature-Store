import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FILES = {
    "user": PROJECT_ROOT / "feature_store" / "data" / "user_features.parquet",
    "item": PROJECT_ROOT / "feature_store" / "data" / "item_features.parquet",
}

REQUIRED_USER_COLS = [
    "user_idx", "event_timestamp",
    "avg_rating", "total_interactions", "rating_std", "min_rating", "max_rating",
    "high_rating_ratio", "active_days", "interaction_density",
    "temporal_spread_days", "recency_days", "session_count",
    "avg_session_length", "genre_diversity", "top_genre_idx", "popularity_bias",
]

REQUIRED_ITEM_COLS = [
    "item_idx", "event_timestamp",
    "global_popularity", "popularity_rank", "niche_score", "avg_ratings_per_day",
    "avg_item_rating", "rating_count", "rating_std", "high_rating_ratio",
    "genre_count", "primary_genre_idx", "release_year", "item_age_days",
]


def validate_parquet(path: Path, required_cols: list, label: str):
    print(f"\n  Validating {label} Parquet: {path.name}")

    if not path.exists():
        print(f"  ❌ File not found: {path}")
        print(f"     Run: python feature_store/pipelines/{label}_features_pipeline.py")
        return False

    # Load metadata (fast, no data read)
    meta = pq.read_metadata(path)
    print(f"  Row groups: {meta.num_row_groups}")
    print(f"  Total rows: {meta.num_rows:,}")

    # Load actual data
    df = pd.read_parquet(path)
    print(f"  Columns:    {len(df.columns)}")
    print(f"  Memory:     {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Check required columns
    missing = [c for c in required_cols if c not in df.columns]
    extra   = [c for c in df.columns if c not in required_cols]

    if missing:
        print(f"  ❌ Missing columns: {missing}")
        return False
    else:
        print(f"  ✅ All required columns present")

    if extra:
        print(f"  ℹ️  Extra columns (OK): {extra}")

    # Check event_timestamp
    if "event_timestamp" in df.columns:
        ts = df["event_timestamp"]
        if ts.dtype.tz is None:
            print("  ⚠️  event_timestamp has no timezone — Feast needs UTC")
            print("     Fix: df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], utc=True)")
        else:
            print(f"  ✅ event_timestamp timezone: {ts.dtype.tz}")

    # Check entity column
    entity_col = "user_idx" if label == "user" else "item_idx"
    if entity_col in df.columns:
        nulls = df[entity_col].isnull().sum()
        dups  = df[entity_col].duplicated().sum()
        print(f"  ✅ {entity_col}: {df[entity_col].nunique():,} unique values")
        if nulls > 0:
            print(f"  ⚠️  {entity_col} has {nulls} null values")
        if dups > 0:
            print(f"  ⚠️  {entity_col} has {dups} duplicate values")
            print(f"     Feast expects one row per entity — deduplicate if needed")

    # Check for NaN in feature columns
    feature_cols = [c for c in df.columns if c not in [entity_col, "event_timestamp"]]
    nan_cols = {c: df[c].isnull().sum() for c in feature_cols if df[c].isnull().sum() > 0}
    if nan_cols:
        print(f"  ⚠️  Columns with NaN values (should be 0):")
        for col, count in nan_cols.items():
            print(f"       {col}: {count} NaN values")
    else:
        print(f"  ✅ No NaN values in feature columns")

    print(f"  ✅ {label.capitalize()} Parquet validation PASSED")
    return True


def main():
    print("=" * 65)
    print("  DS19 Week 7 — Parquet Schema Validation")
    print("=" * 65)

    user_ok = validate_parquet(FILES["user"], REQUIRED_USER_COLS, "user")
    item_ok = validate_parquet(FILES["item"], REQUIRED_ITEM_COLS, "item")

    print("\n" + "=" * 65)
    if user_ok and item_ok:
        print("✅ Both Parquet files validated — Ready for Feast apply")
        print("   Next: cd feature_store/feature_repo && feast apply")
    else:
        print("❌ Validation failed — Fix issues above before running feast apply")
        sys.exit(1)
    print("=" * 65)


if __name__ == "__main__":
    main()