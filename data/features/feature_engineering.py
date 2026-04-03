"""
DS19 — Week 5: Feature Engineering
Precomputing user + item features for ranking model.

Run: python data/features/feature_engineering.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Fixed paths to match Week 1-2 pipeline output
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
FEATURES_DIR = Path("data/features")
META_PATH = PROCESSED_DIR / "dataset_meta.json"  # FIXED: Was models/saved/

os.makedirs(FEATURES_DIR, exist_ok=True)


def build_train_only_ratings(ratings_df, train_sequences, meta):
    """
    Restrict ratings to each user's training window only.

    Why this matters:
    - Week 5 ranking features must not use interactions after the train cutoff.
    - This prevents temporal leakage into validation/test ranking labels.
    """
    print("  Building train-only ratings view (leakage-safe)...")

    interactions_df = pd.read_csv(PROCESSED_DIR / "interactions.csv")
    interactions_df["timestamp"] = pd.to_datetime(
        interactions_df["timestamp"], errors="coerce"
    )

    train_len = pd.Series(
        {int(uid): len(seq) for uid, seq in train_sequences.items()},
        name="train_len",
    )

    interactions_df = interactions_df.merge(
        train_len.rename_axis("user_idx").reset_index(),
        on="user_idx",
        how="inner",
    )

    interactions_df = interactions_df.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)
    interactions_df["interaction_rank"] = interactions_df.groupby("user_idx").cumcount() + 1

    train_interactions = interactions_df[
        interactions_df["interaction_rank"] <= interactions_df["train_len"]
    ].copy()

    ratings_subset = ratings_df[["userId", "movieId", "rating", "timestamp"]].copy()
    ratings_subset["timestamp"] = pd.to_datetime(
        ratings_subset["timestamp"], unit="s", errors="coerce"
    )
    ratings_subset = ratings_subset.drop_duplicates(
        subset=["userId", "movieId", "timestamp"],
        keep="first",
    )

    train_df = train_interactions.merge(
        ratings_subset,
        on=["userId", "movieId", "timestamp"],
        how="left",
    )

    train_df["encoded_user_id"] = train_df["user_idx"].astype(int)
    train_df["encoded_item_id"] = train_df["item_idx"].astype(int)

    missing_ratings = int(train_df["rating"].isna().sum())

    dropped = len(interactions_df) - len(train_df)
    print(f"     Full rows:      {len(interactions_df):,}")
    print(f"     Train-only rows: {len(train_df):,}")
    print(f"     Dropped future rows: {dropped:,}")
    if missing_ratings > 0:
        print(f"     ⚠️ Missing ratings after join: {missing_ratings:,}")
        train_df["rating"] = train_df["rating"].fillna(0.0)

    return train_df


# ─────────────────────────────────────────────────────────────
# 1. USER FEATURES
# ─────────────────────────────────────────────────────────────

def compute_user_features(ratings_df, train_sequences, meta):
    """
    Compute per-user features from the ratings dataframe and sequences.

    Args:
        ratings_df: pd.DataFrame with columns [userId, movieId, rating, timestamp]
        train_sequences: dict {user_id: [item_id, item_id, ...]} (integer encoded)
        meta: dict with user2id, id2user mappings

    Returns:
        pd.DataFrame indexed by user_id (integer encoded) with user features
    """
    print("  Computing user features...")

    if "encoded_user_id" in ratings_df.columns:
        ratings_filtered = ratings_df.copy()
    else:
        # Backward-compatible path if pre-encoded columns are absent.
        user2id = {str(k): int(v) for k, v in meta["user2id"].items()}
        ratings_df = ratings_df.copy()
        ratings_df["userId_str"] = ratings_df["userId"].astype(str)
        ratings_filtered = ratings_df[ratings_df["userId_str"].isin(user2id)].copy()
        ratings_filtered["encoded_user_id"] = ratings_filtered["userId_str"].map(user2id)

    # ── Compute per-user stats from ratings ───────────────────────
    user_stats = ratings_filtered.groupby("encoded_user_id").agg(
        user_rating_count=("rating", "count"),
        user_avg_rating=("rating", "mean"),
        user_min_rating=("rating", "min"),
        user_max_rating=("rating", "max"),
        user_rating_std=("rating", "std"),
        user_last_timestamp=("timestamp", "max"),
        user_first_timestamp=("timestamp", "min"),
    ).reset_index()

    # ── Compute activity duration (days between first and last interaction) ──
    user_stats["user_active_days"] = (
        (user_stats["user_last_timestamp"] - user_stats["user_first_timestamp"])
        / 86400
    )

    # ── Compute sequence length from training sequences ───────────
    seq_lengths = {
        encoded_uid: len(seq)
        for encoded_uid, seq in train_sequences.items()
    }
    user_stats["user_sequence_length"] = user_stats["encoded_user_id"].map(
        seq_lengths
    ).fillna(0).astype(int)

    # ── Fill NaN (users with single rating have no std) ──────────
    user_stats["user_rating_std"] = user_stats["user_rating_std"].fillna(0.0)

    # ── Compute recency: days since last interaction ───────────────
    MAX_TIMESTAMP = ratings_filtered["timestamp"].max()
    user_stats["user_days_since_last"] = (
        (MAX_TIMESTAMP - user_stats["user_last_timestamp"]) / 86400
    )

    # ── Set index to encoded_user_id for fast lookup ───────────────
    user_stats = user_stats.set_index("encoded_user_id")

    # ── Log info ──────────────────────────────────────────────────
    print(f"     Users with features: {len(user_stats):,}")
    print(f"     Feature columns: {list(user_stats.columns)}")
    print(f"     Avg interactions per user: {user_stats['user_rating_count'].mean():.1f}")

    return user_stats


# ─────────────────────────────────────────────────────────────
# 2. ITEM FEATURES
# ─────────────────────────────────────────────────────────────

def compute_item_features(ratings_df, movies_df, meta):
    """
    Compute per-item features from ratings and movie metadata.

    Args:
        ratings_df: pd.DataFrame [userId, movieId, rating, timestamp]
        movies_df:  pd.DataFrame [movieId, title, genres]
        meta:       dict with item2id, id2item mappings

    Returns:
        pd.DataFrame indexed by encoded_item_id with item features
    """
    print("  Computing item features...")

    item_map = meta.get("item2id") or meta.get("item2idx")
    if item_map is None:
        with open(PROCESSED_DIR / "item2idx.json", "r") as f:
            item_map = json.load(f)
    item2id = {str(k): int(v) for k, v in item_map.items()}

    if "encoded_item_id" in ratings_df.columns:
        ratings_filtered = ratings_df.copy()
    else:
        ratings_df = ratings_df.copy()
        ratings_df["movieId_str"] = ratings_df["movieId"].astype(str)
        ratings_filtered = ratings_df[ratings_df["movieId_str"].isin(item2id)].copy()
        ratings_filtered["encoded_item_id"] = ratings_filtered["movieId_str"].map(item2id)

    # ── Compute per-item stats ────────────────────────────────────
    item_stats = ratings_filtered.groupby("encoded_item_id").agg(
        item_rating_count=("rating", "count"),
        item_avg_rating=("rating", "mean"),
        item_rating_std=("rating", "std"),
        item_last_timestamp=("timestamp", "max"),
        item_first_timestamp=("timestamp", "min"),
    ).reset_index()

    # ── Popularity score (log-scaled to reduce long-tail effect) ──
    item_stats["item_popularity"] = item_stats["item_rating_count"]
    item_stats["item_log_popularity"] = np.log1p(item_stats["item_popularity"])

    # ── Item recency (days since last rating) ─────────────────────
    MAX_TIMESTAMP = ratings_filtered["timestamp"].max()
    item_stats["item_days_since_last_rating"] = (
        (MAX_TIMESTAMP - item_stats["item_last_timestamp"]) / 86400
    )

    # ── Item age (days between first and last rating) ─────────────
    item_stats["item_active_days"] = (
        (item_stats["item_last_timestamp"] - item_stats["item_first_timestamp"])
        / 86400
    )

    # ── Fill NaN ───────────────────────────────────────────────────
    item_stats["item_rating_std"] = item_stats["item_rating_std"].fillna(0.0)

    # ── Merge movie metadata (genres, year) ───────────────────────
    movies_df["movieId_str"] = movies_df["movieId"].astype(str)
    movies_filtered = movies_df[movies_df["movieId_str"].isin(item2id)].copy()
    movies_filtered["encoded_item_id"] = movies_filtered["movieId_str"].map(item2id)

    # ── Extract year from title (e.g. "Toy Story (1995)") ─────────
    movies_filtered["item_year"] = (
        movies_filtered["title"]
        .str.extract(r"\((\d{4})\)")
        .astype(float)
        .fillna(0)
    )

    # ── Genre features ────────────────────────────────────────────
    all_genres = set()
    for genres in movies_filtered["genres"].dropna():
        for g in genres.split("|"):
            if g != "(no genres listed)":
                all_genres.add(g)

    print(f"     Unique genres found: {len(all_genres)}")
    print(f"     Genres: {sorted(all_genres)}")

    # Create binary genre columns
    for genre in sorted(all_genres):
        movies_filtered[f"genre_{genre.replace(' ', '_').replace('-', '_')}"] = (
            movies_filtered["genres"].str.contains(genre, regex=False).astype(int)
        )

    # ── Number of genres per item ─────
    genre_cols = [c for c in movies_filtered.columns if c.startswith("genre_")]
    movies_filtered["item_num_genres"] = movies_filtered[genre_cols].sum(axis=1)

    # ── Merge item stats with movie metadata ──────────────────────
    item_features = item_stats.merge(
        movies_filtered[["encoded_item_id", "item_year", "item_num_genres"] + genre_cols],
        on="encoded_item_id",
        how="left"
    )

    # ── Fill movies without stats ──
    item_features["item_rating_count"] = item_features["item_rating_count"].fillna(0)
    item_features["item_avg_rating"] = item_features["item_avg_rating"].fillna(
        item_stats["item_avg_rating"].mean()
    )
    item_features["item_year"] = item_features["item_year"].fillna(0)
    item_features["item_num_genres"] = item_features["item_num_genres"].fillna(0)
    for gc in genre_cols:
        item_features[gc] = item_features[gc].fillna(0)

    # ── Popularity rank ───────────
    item_features["item_popularity_rank"] = item_features["item_rating_count"].rank(
        method="dense", ascending=False
    )

    item_features = item_features.set_index("encoded_item_id")

    print(f"     Items with features: {len(item_features):,}")
    print(f"     Feature columns: {[c for c in item_features.columns if not c.startswith('genre_')]}")

    return item_features, genre_cols


# ─────────────────────────────────────────────────────────────
# 3. RETRIEVAL FEATURES (from FAISS output)
# ─────────────────────────────────────────────────────────────

def build_retrieval_features(faiss_scores, faiss_item_ids):
    """
    Build features from the raw FAISS retrieval output.
    """
    num_candidates = len(faiss_scores)

    # ── Rank normalization ────────────────────────────────────────
    if num_candidates > 1:
        rank_normalized = 1.0 - (np.arange(num_candidates) / (num_candidates - 1))
    else:
        rank_normalized = np.array([1.0])

    # ── Score normalization ───────────────────────────────────────
    score_min = faiss_scores.min()
    score_max = faiss_scores.max()
    if score_max > score_min:
        scores_normalized = (faiss_scores - score_min) / (score_max - score_min)
    else:
        scores_normalized = np.ones(num_candidates) * 0.5

    # ── Score gap features ────────────────────────────────────────
    score_gaps = np.zeros(num_candidates)
    score_gaps[1:] = faiss_scores[:-1] - faiss_scores[1:]

    return {
        "faiss_score": faiss_scores,
        "faiss_rank": np.arange(1, num_candidates + 1),
        "faiss_rank_normalized": rank_normalized,
        "faiss_score_normalized": scores_normalized,
        "faiss_score_gap": score_gaps,
    }


# ─────────────────────────────────────────────────────────────
# 4. INTERACTION FEATURES (user × item)
# ─────────────────────────────────────────────────────────────

def build_interaction_features(user_id, candidate_item_ids,
                                user_features, item_features,
                                genre_cols, train_sequences):
    """
    Build user × item interaction features for a list of candidates.
    """
    num_cands = len(candidate_item_ids)

    # ── Get user's genre preferences from their history ───────────
    user_genre_counts = np.zeros(len(genre_cols))

    if user_id in train_sequences:
        user_history = train_sequences[user_id]
        for hist_item_id in user_history:
            if hist_item_id in item_features.index:
                item_genre_row = item_features.loc[hist_item_id, genre_cols].values
                user_genre_counts += item_genre_row.astype(float)

    # Normalize to get genre preference probabilities
    total_genre_count = user_genre_counts.sum()
    if total_genre_count > 0:
        user_genre_prefs = user_genre_counts / total_genre_count
    else:
        user_genre_prefs = np.zeros(len(genre_cols))

    # ── Compute interaction features per candidate ─────────────────
    genre_overlap = np.zeros(num_cands)
    genre_max_overlap = np.zeros(num_cands)
    item_popularity_pct = np.zeros(num_cands)

    max_popularity = item_features["item_rating_count"].max()

    for idx, item_id in enumerate(candidate_item_ids):
        if item_id in item_features.index:
            item_genre_row = item_features.loc[item_id, genre_cols].values.astype(float)

            genre_overlap[idx] = np.dot(user_genre_prefs, item_genre_row)

            if item_genre_row.sum() > 0:
                matching_prefs = user_genre_prefs * item_genre_row
                genre_max_overlap[idx] = matching_prefs.max()

            item_pop = item_features.loc[item_id, "item_rating_count"]
            item_popularity_pct[idx] = item_pop / max_popularity if max_popularity > 0 else 0

    return {
        "genre_overlap": genre_overlap,
        "genre_max_match": genre_max_overlap,
        "item_popularity_pct": item_popularity_pct,
    }


# ─────────────────────────────────────────────────────────────
# 5. SAVE / LOAD FEATURE CACHES
# ─────────────────────────────────────────────────────────────

def save_features(user_features, item_features, genre_cols):
    """Cache precomputed user and item features to parquet."""
    print("\n  Saving feature caches...")

    user_path = os.path.join(FEATURES_DIR, "user_features.parquet")
    item_path = os.path.join(FEATURES_DIR, "item_features.parquet")
    genre_cols_path = os.path.join(FEATURES_DIR, "genre_cols.json")

    user_features.to_parquet(user_path)
    item_features.to_parquet(item_path)
    with open(genre_cols_path, "w") as f:
        json.dump(genre_cols, f)

    print(f"     ✅ Saved user_features: {user_path}  ({len(user_features):,} users)")
    print(f"     ✅ Saved item_features: {item_path}  ({len(item_features):,} items)")
    print(f"     ✅ Saved genre_cols:    {genre_cols_path}  ({len(genre_cols)} genres)")


def load_features():
    """Load cached user and item features from parquet."""
    user_path = os.path.join(FEATURES_DIR, "user_features.parquet")
    item_path = os.path.join(FEATURES_DIR, "item_features.parquet")
    genre_cols_path = os.path.join(FEATURES_DIR, "genre_cols.json")

    if not all(os.path.exists(p) for p in [user_path, item_path, genre_cols_path]):
        return None, None, None

    user_features = pd.read_parquet(user_path)
    item_features = pd.read_parquet(item_path)
    with open(genre_cols_path, "r") as f:
        genre_cols = json.load(f)

    return user_features, item_features, genre_cols


# ─────────────────────────────────────────────────────────────
# MAIN — Precompute and cache all features
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DS19 — Week 5: Feature Engineering")
    print("  Precomputing user + item features")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    print("\n📂 Loading data...")

    # Load ratings (with original userId, movieId)
    ratings_df = pd.read_csv(os.path.join(PROCESSED_DIR, "ratings_filtered.csv"))
    
    # Load movies (filtered to our vocabulary)
    movies_df = pd.read_csv(os.path.join(PROCESSED_DIR, "movies.csv"))
    
    print(f"  Ratings: {len(ratings_df):,} rows")
    print(f"  Movies:  {len(movies_df):,} rows")

    # ── Load meta (FIXED: Handle both key styles) ─────────────────
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    
    # Normalize meta keys for compatibility
    if 'n_users' in meta and 'num_users' not in meta:
        meta['num_users'] = meta['n_users']
    if 'n_items' in meta and 'num_items' not in meta:
        meta['num_items'] = meta['n_items']
    if 'user2idx' in meta and 'user2id' not in meta:
        meta['user2id'] = meta['user2idx']
    if 'item2idx' in meta and 'item2id' not in meta:
        meta['item2id'] = meta['item2idx']
    if 'idx2user' in meta and 'id2user' not in meta:
        meta['id2user'] = meta['idx2user']
    if 'idx2item' in meta and 'id2item' not in meta:
        meta['id2item'] = meta['idx2item']
        
    print(f"  Users in meta: {meta['num_users']:,}")
    print(f"  Items in meta: {meta['num_items']:,}")

    # ── Load sequences (FIXED: Try both filenames) ────────────────
    seq_path = os.path.join(SPLITS_DIR, "train_sequences.pkl")
    if not os.path.exists(seq_path):
        seq_path = os.path.join(SPLITS_DIR, "train_seqs.pkl")
        
    with open(seq_path, "rb") as f:
        train_sequences = pickle.load(f)
    print(f"  Train sequences: {len(train_sequences):,} users")

    # ── Compute features ──────────────────────────────────────────
    print("\n🔧 Computing features...")

    train_ratings_df = build_train_only_ratings(ratings_df, train_sequences, meta)

    user_features = compute_user_features(train_ratings_df, train_sequences, meta)
    item_features, genre_cols = compute_item_features(train_ratings_df, movies_df, meta)

    # ── Save ───────────────────────────────────────────────────────
    save_features(user_features, item_features, genre_cols)

    print("\n✅ Feature engineering complete!")
    print(f"   User feature shape: {user_features.shape}")
    print(f"   Item feature shape: {item_features.shape}")
    print(f"   Genre columns: {len(genre_cols)}")


if __name__ == "__main__":
    main()