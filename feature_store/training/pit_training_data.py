import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from feast import FeatureStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_PATH    = PROJECT_ROOT / "feature_store" / "feature_repo"
PROCESSED_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH  = PROJECT_ROOT / "feature_store" / "data" / "pit_training_data.parquet"


def load_training_samples() -> pd.DataFrame:
    """
    Loads the training split and converts to entity_df format for Feast.
    
    entity_df format required by feast.get_historical_features():
      Columns: [user_idx, item_idx, event_timestamp, label]
      - event_timestamp must be timezone-aware UTC datetime
      - label: 1 for positive interaction, 0 for negative sample
    
    Returns DataFrame ready for point-in-time feature retrieval.
    """
    print("  Loading training sequences...")

    # Load training sequences (from Week 1 pipeline)
    train_seqs_path = PROCESSED_DIR / "splits" / "train_seqs.pkl"
    val_labels_path = PROCESSED_DIR / "splits" / "val_labels.pkl"

    with open(train_seqs_path, "rb") as f:
        train_seqs = pickle.load(f)
    with open(val_labels_path, "rb") as f:
        val_labels = pickle.load(f)

    print(f"  Training users: {len(train_seqs):,}")

    # Load interactions with timestamps
    interactions = pd.read_csv(
        PROCESSED_DIR/ "processed" / "interactions.csv",
        dtype={"user_idx": np.int32, "item_idx": np.int32, "timestamp": np.int64}
    )
    interactions["event_timestamp"] = pd.to_datetime(
        interactions["timestamp"], unit="s", utc=True
    )

    print("  Building entity DataFrame for point-in-time join...")

    # Build positive samples with their actual timestamps
    rows = []

    # Sample a subset to keep it manageable (PIT join can be slow for 13M rows)
    max_samples_per_user = 5   # positive + negatives per user
    n_neg_per_pos = 4

    user_interactions = (
        interactions
        .sort_values("timestamp")
        .groupby("user_idx")
        .apply(lambda x: list(zip(x["item_idx"], x["event_timestamp"])))
        .to_dict()
    )

    # Get set of all item_idxs for negative sampling
    all_items = set(interactions["item_idx"].unique())

    import random
    random.seed(42)
    np.random.seed(42)

    for user_idx, seq_items in list(train_seqs.items())[:5000]:  # first 5K users
        if user_idx not in user_interactions:
            continue
        user_ts_pairs = user_interactions[user_idx]

        # Use the LAST positive interaction as our training sample
        if len(user_ts_pairs) < 2:
            continue

        # Positive sample: last interaction
        pos_item_idx, pos_ts = user_ts_pairs[-1]
        rows.append({
            "user_idx":        int(user_idx),
            "item_idx":        int(pos_item_idx),
            "event_timestamp": pos_ts,
            "label":           1
        })

        # Negative samples: random uninteracted items at the same timestamp
        user_items_set = {item for item, _ in user_ts_pairs}
        neg_pool = list(all_items - user_items_set)
        neg_samples = random.sample(neg_pool, min(n_neg_per_pos, len(neg_pool)))

        for neg_item in neg_samples:
            rows.append({
                "user_idx":        int(user_idx),
                "item_idx":        int(neg_item),
                "event_timestamp": pos_ts,  # same timestamp as positive
                "label":           0
            })

    entity_df = pd.DataFrame(rows)
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True)

    print(f"  Entity DataFrame: {len(entity_df):,} samples "
          f"({entity_df['label'].sum():,} positive, "
          f"{(entity_df['label'] == 0).sum():,} negative)")

    return entity_df


def generate_pit_training_data():
    """
    Generates point-in-time correct training features.
    
    Uses feast.get_historical_features() which automatically:
    1. Takes each row in entity_df with its event_timestamp
    2. Finds the feature snapshot closest to (but not after) that timestamp
    3. Attaches those historical feature values to the row
    
    The result is a training DataFrame with zero data leakage.
    """
    print("=" * 65)
    print("  DS19 Week 7 — Point-in-Time Training Data Generation")
    print("=" * 65)

    # Load Feast store
    print("\n[1/5] Loading Feast store...")
    store = FeatureStore(repo_path=str(REPO_PATH))
    print("  ✅ Feast store loaded")

    # Build entity DataFrame
    print("\n[2/5] Building entity DataFrame...")
    entity_df = load_training_samples()

    # Run point-in-time correct historical feature retrieval
    print("\n[3/5] Running Feast get_historical_features() (may take 1-3 minutes)...")
    print("  This reads offline Parquet and performs point-in-time joins.")
    print("  For each (user_idx, item_idx, timestamp), it finds the")
    print("  feature values that were valid AT that timestamp.")

    FEATURE_REFS = [
        "user_features_view:avg_rating",
        "user_features_view:total_interactions",
        "user_features_view:high_rating_ratio",
        "user_features_view:recency_days",
        "user_features_view:genre_diversity",
        "user_features_view:top_genre_idx",
        "user_features_view:popularity_bias",
        "item_features_view:global_popularity",
        "item_features_view:avg_item_rating",
        "item_features_view:niche_score",
        "item_features_view:primary_genre_idx",
        "item_features_view:release_year",
        "item_features_view:item_age_days",
    ]

    try:
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=FEATURE_REFS
        ).to_df()

        print(f"  ✅ Historical features retrieved: {len(training_df):,} rows")
        print(f"  Columns: {list(training_df.columns)}")

    except Exception as e:
        print(f"  ❌ Historical feature retrieval failed: {e}")
        print("  Note: For the MovieLens dataset with a single timestamp snapshot,")
        print("  this is expected to return features for the snapshot timestamp.")
        print("  In production with rolling data, this would return true PIT features.")
        raise

    # Fill NaN from point-in-time join (items with no historical data)
    print("\n[4/5] Post-processing training DataFrame...")
    print(f"  NaN values before fill: {training_df.isnull().sum().sum()}")
    training_df = training_df.fillna(0.0)
    print(f"  NaN values after fill: {training_df.isnull().sum().sum()}")

    print(f"\n  Feature statistics:")
    feature_cols = [c for c in training_df.columns
                    if c not in ["user_idx", "item_idx", "event_timestamp", "label"]]
    print(training_df[feature_cols].describe().round(3).to_string())

    # Write to Parquet
    print(f"\n[5/5] Writing PIT training data to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  ✅ Written: {len(training_df):,} rows, {len(training_df.columns)} columns")

    print("\n" + "=" * 65)
    print("✅ Point-in-time correct training data generated!")
    print(f"   {OUTPUT_PATH}")
    print("   Use this for future LightGBM model retraining (no data leakage).")
    print("=" * 65)

    return training_df


if __name__ == "__main__":
    generate_pit_training_data()