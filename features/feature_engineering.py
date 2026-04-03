import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"
SPLITS_DIR = "data/splits"
FEATURES_DIR = "data/features"
META_PATH = "models/saved/dataset_meta.json"

os.makedirs(FEATURES_DIR, exist_ok=True)


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

    # ── Map original userId to encoded user_id ────────────────────
    # ratings_df uses original movieLens userId (string/int)
    # We need to work in encoded space (integer IDs from our pipeline)
    user2id = meta["user2id"]  # original_userId → encoded_user_id
    id2user = meta["id2user"]  # encoded_user_id → original_userId

    # Convert id2user keys from string (JSON) to int
    id2user = {int(k): v for k, v in id2user.items()}
    user2id = {str(k): v for k, v in user2id.items()}

    # Filter ratings to only users in our dataset
    valid_original_users = set(user2id.keys())
    # ratings userId might be int or string — normalize
    ratings_df["userId_str"] = ratings_df["userId"].astype(str)
    ratings_filtered = ratings_df[ratings_df["userId_str"].isin(valid_original_users)].copy()
    ratings_filtered["encoded_user_id"] = ratings_filtered["userId_str"].map(user2id)

    # ── Compute per-user stats from ratings ───────────────────────
    user_stats = ratings_filtered.groupby("encoded_user_id").agg(
        user_rating_count=("rating", "count"),         # total interactions
        user_avg_rating=("rating", "mean"),            # average explicit rating
        user_min_rating=("rating", "min"),             # lowest rating given
        user_max_rating=("rating", "max"),             # highest rating given
        user_rating_std=("rating", "std"),             # rating variance (taste breadth)
        user_last_timestamp=("timestamp", "max"),      # most recent interaction time
        user_first_timestamp=("timestamp", "min"),     # first interaction time
    ).reset_index()

    # ── Compute activity duration (days between first and last interaction) ──
    user_stats["user_active_days"] = (
        (user_stats["user_last_timestamp"] - user_stats["user_first_timestamp"])
        / 86400  # convert seconds to days
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
    # Use max timestamp in dataset as "now"
    MAX_TIMESTAMP = ratings_df["timestamp"].max()
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

    item2id = meta["item2id"]  # original_movieId → encoded_item_id
    item2id = {str(k): v for k, v in item2id.items()}

    # ── Filter to items in our vocabulary ────────────────────────
    ratings_df["movieId_str"] = ratings_df["movieId"].astype(str)
    ratings_filtered = ratings_df[ratings_df["movieId_str"].isin(item2id)].copy()
    ratings_filtered["encoded_item_id"] = ratings_filtered["movieId_str"].map(item2id)

    # ── Compute per-item stats ────────────────────────────────────
    item_stats = ratings_filtered.groupby("encoded_item_id").agg(
        item_rating_count=("rating", "count"),     # total ratings received
        item_avg_rating=("rating", "mean"),        # average rating
        item_rating_std=("rating", "std"),         # rating variance (controversial?)
        item_last_timestamp=("timestamp", "max"),  # most recently rated
        item_first_timestamp=("timestamp", "min"), # first rated
    ).reset_index()

    # ── Popularity score (log-scaled to reduce long-tail effect) ──
    item_stats["item_popularity"] = item_stats["item_rating_count"]
    item_stats["item_log_popularity"] = np.log1p(item_stats["item_popularity"])

    # ── Item recency (days since last rating) ─────────────────────
    MAX_TIMESTAMP = ratings_df["timestamp"].max()
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
    # Split pipe-separated genres into separate columns
    # Example: "Action|Comedy|Drama"
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

    # ── Number of genres per item (breadth of genre coverage) ─────
    genre_cols = [c for c in movies_filtered.columns if c.startswith("genre_")]
    movies_filtered["item_num_genres"] = movies_filtered[genre_cols].sum(axis=1)

    # ── Merge item stats with movie metadata ──────────────────────
    item_features = item_stats.merge(
        movies_filtered[["encoded_item_id", "item_year", "item_num_genres"] + genre_cols],
        on="encoded_item_id",
        how="left"
    )

    # ── Fill movies without stats (items in vocab but no valid ratings) ──
    item_features["item_rating_count"] = item_features["item_rating_count"].fillna(0)
    item_features["item_avg_rating"] = item_features["item_avg_rating"].fillna(
        item_stats["item_avg_rating"].mean()
    )
    item_features["item_year"] = item_features["item_year"].fillna(0)
    item_features["item_num_genres"] = item_features["item_num_genres"].fillna(0)
    for gc in genre_cols:
        item_features[gc] = item_features[gc].fillna(0)

    # ── Popularity rank (dense rank — 1 = most popular) ───────────
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

    Args:
        faiss_scores:    np.array [top_k] — cosine similarity scores (higher = more relevant)
        faiss_item_ids:  np.array [top_k] — encoded item IDs (0-indexed in FAISS)

    Returns:
        dict — retrieval features per (position in list)
    """
    num_candidates = len(faiss_scores)

    # ── FAISS cosine score (primary retrieval signal) ─────────────
    # These are inner-product scores for L2-normalized embeddings → cosine similarity
    # Range: [-1, 1] but in practice [0, 1] for recommendation embeddings

    # ── Rank normalization ────────────────────────────────────────
    # Convert rank to a normalized score: rank 1 = 1.0, rank 100 = 0.0
    # This captures "FAISS thinks this is the most relevant" as a continuous feature
    rank_normalized = 1.0 - (np.arange(num_candidates) / (num_candidates - 1))

    # ── Score normalization ───────────────────────────────────────
    # Min-max normalize FAISS scores within this user's candidates
    # (relative score matters more than absolute score)
    score_min = faiss_scores.min()
    score_max = faiss_scores.max()
    if score_max > score_min:
        scores_normalized = (faiss_scores - score_min) / (score_max - score_min)
    else:
        scores_normalized = np.ones(num_candidates) * 0.5

    # ── Score gap features ────────────────────────────────────────
    # Gap between consecutive scores: large gap → top items are clearly better
    score_gaps = np.zeros(num_candidates)
    score_gaps[1:] = faiss_scores[:-1] - faiss_scores[1:]

    return {
        "faiss_score": faiss_scores,                    # raw cosine score
        "faiss_rank": np.arange(1, num_candidates + 1), # 1-indexed rank
        "faiss_rank_normalized": rank_normalized,        # 1.0 (best) → 0.0 (worst)
        "faiss_score_normalized": scores_normalized,     # min-max within user
        "faiss_score_gap": score_gaps,                   # score drop from previous
    }


# ─────────────────────────────────────────────────────────────
# 4. INTERACTION FEATURES (user × item)
# ─────────────────────────────────────────────────────────────

def build_interaction_features(user_id, candidate_item_ids,
                                user_features, item_features,
                                genre_cols, train_sequences):
    """
    Build user × item interaction features for a list of candidates.

    Args:
        user_id:           int — encoded user ID
        candidate_item_ids: list[int] — encoded item IDs to compute features for
        user_features:     pd.DataFrame indexed by encoded_user_id
        item_features:     pd.DataFrame indexed by encoded_item_id
        genre_cols:        list[str] — genre column names in item_features
        train_sequences:   dict — {encoded_user_id: [encoded_item_ids]}

    Returns:
        dict of np.arrays — one value per candidate
    """
    num_cands = len(candidate_item_ids)

    # ── Get user's genre preferences from their history ───────────
    # Count how often each genre appears in user's past interactions
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
    genre_overlap = np.zeros(num_cands)        # genre overlap score
    genre_max_overlap = np.zeros(num_cands)    # max genre preference score
    item_popularity_pct = np.zeros(num_cands)  # item popularity percentile

    # Get max popularity for percentile computation
    max_popularity = item_features["item_rating_count"].max()

    for idx, item_id in enumerate(candidate_item_ids):
        if item_id in item_features.index:
            item_genre_row = item_features.loc[item_id, genre_cols].values.astype(float)

            # ── Genre overlap: dot product of user prefs and item genres ──
            # High value = item's genres align well with user's history
            genre_overlap[idx] = np.dot(user_genre_prefs, item_genre_row)

            # ── Max genre preference: strongest matching genre ─────────────
            if item_genre_row.sum() > 0:
                matching_prefs = user_genre_prefs * item_genre_row
                genre_max_overlap[idx] = matching_prefs.max()

            # ── Popularity percentile ─────────────────────────────────────
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

    ratings_df = pd.read_csv(os.path.join(PROCESSED_DIR, "ratings_filtered.csv"))
    movies_df = pd.read_csv(os.path.join(PROCESSED_DIR, "movies.csv"))
    print(f"  Ratings: {len(ratings_df):,} rows")
    print(f"  Movies:  {len(movies_df):,} rows")

    with open(META_PATH, "r") as f:
        meta = json.load(f)
    print(f"  Users in meta: {meta['num_users']:,}")
    print(f"  Items in meta: {meta['num_items']:,}")

    with open(os.path.join(SPLITS_DIR, "train_sequences.pkl"), "rb") as f:
        train_sequences = pickle.load(f)
    print(f"  Train sequences: {len(train_sequences):,} users")

    # ── Compute features ──────────────────────────────────────────
    print("\n🔧 Computing features...")

    user_features = compute_user_features(ratings_df, train_sequences, meta)
    item_features, genre_cols = compute_item_features(ratings_df, movies_df, meta)

    # ── Save ───────────────────────────────────────────────────────
    save_features(user_features, item_features, genre_cols)

    print("\n✅ Feature engineering complete!")
    print(f"   User feature shape: {user_features.shape}")
    print(f"   Item feature shape: {item_features.shape}")
    print(f"   Genre columns: {len(genre_cols)}")


if __name__ == "__main__":
    main()