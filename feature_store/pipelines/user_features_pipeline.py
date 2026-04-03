import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT      = Path(__file__).resolve().parent.parent.parent
INTERACTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "interactions.csv"
RATINGS_PATH      = PROJECT_ROOT / "data" / "raw" / "ratings.csv"
ITEM2IDX_PATH     = PROJECT_ROOT / "data" / "processed" / "item2idx.json"
MOVIES_PATH       = PROJECT_ROOT / "data" / "raw" / "movies.csv"
OUTPUT_PATH       = PROJECT_ROOT / "feature_store" / "data" / "user_features.parquet"

# Feast requires an event_timestamp column (datetime with timezone)
# We use the MAX timestamp in the dataset as "now" for offline features
# In production: this would be the actual computation timestamp

CHUNK_SIZE        = 500_000  # Process ratings in chunks if RAM is tight


# ─────────────────────────────────────────────────────────────
# GENRE UTILITIES
# ─────────────────────────────────────────────────────────────

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western", "(no genres listed)"
]

GENRE2IDX = {g: i for i, g in enumerate(ALL_GENRES)}


def parse_genres(genre_str: str) -> list:
    """
    Parses MovieLens genre string 'Action|Comedy|Drama' → ['Action', 'Comedy', 'Drama']
    Returns [] if string is nan or empty.
    """
    if not isinstance(genre_str, str) or genre_str == "(no genres listed)":
        return []
    return [g.strip() for g in genre_str.split("|") if g.strip() in GENRE2IDX]


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD RAW DATA
# ─────────────────────────────────────────────────────────────

def load_raw_data():
    """
    Loads ratings.csv and movies.csv, applies implicit feedback filter.
    Returns merged DataFrame with columns:
      [userId, movieId, rating, timestamp, genres]
    """
    print("  Loading ratings.csv...")
    ratings = pd.read_csv(
        RATINGS_PATH,
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
        usecols=["userId", "movieId", "rating", "timestamp"]
    )
    print(f"  Raw ratings: {len(ratings):,} rows")

    # Apply implicit feedback filter (same as training pipeline)
    ratings = ratings[ratings["rating"] >= 4.0].copy()
    print(f"  After implicit filter (>=4): {len(ratings):,} rows")

    print("  Loading movies.csv...")
    movies = pd.read_csv(MOVIES_PATH, usecols=["movieId", "genres"])
    movies["genres"] = movies["genres"].fillna("(no genres listed)")

    # Merge to attach genres
    ratings = ratings.merge(movies, on="movieId", how="left")
    ratings["genres"] = ratings["genres"].fillna("(no genres listed)")

    # Convert timestamp (Unix seconds → datetime)
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    ratings["date"]     = ratings["datetime"].dt.date

    print(f"  Merged ratings shape: {ratings.shape}")
    return ratings


# ─────────────────────────────────────────────────────────────
# STEP 2: LOAD ITEM POPULARITY (for popularity_bias feature)
# ─────────────────────────────────────────────────────────────

def compute_item_popularity(ratings: pd.DataFrame) -> pd.Series:
    """
    Returns a Series: movieId → popularity_rank (1 = most popular)
    Used to compute avg popularity rank of items a user has interacted with.
    """
    popularity = ratings.groupby("movieId").size()
    # Rank: most popular gets rank 1
    popularity_rank = popularity.rank(method="min", ascending=False).astype(int)
    return popularity_rank


# ─────────────────────────────────────────────────────────────
# STEP 3: COMPUTE SESSION FEATURES
# ─────────────────────────────────────────────────────────────

def compute_session_features(user_ratings: pd.DataFrame) -> dict:
    """
    Given a single user's ratings (sorted by timestamp), computes:
      - session_count: number of sessions (gap > 3600 seconds = new session)
      - avg_session_length: average items per session
    
    This is an expensive operation — we compute it per-user in a groupby apply.
    """
    if len(user_ratings) == 0:
        return {"session_count": 1, "avg_session_length": 0.0}

    ts = user_ratings["timestamp"].sort_values().values
    
    # Session boundary: gap > 3600 seconds (1 hour)
    SESSION_GAP_SECONDS = 3600
    gaps = np.diff(ts)
    session_boundaries = np.where(gaps > SESSION_GAP_SECONDS)[0]
    session_count = len(session_boundaries) + 1

    # Session lengths
    boundaries = np.concatenate([[-1], session_boundaries, [len(ts) - 1]])
    session_lengths = np.diff(boundaries)
    avg_session_length = float(np.mean(session_lengths)) if len(session_lengths) > 0 else 0.0

    return {
        "session_count": int(session_count),
        "avg_session_length": round(avg_session_length, 2)
    }


# ─────────────────────────────────────────────────────────────
# STEP 4: MAIN FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_user_features(ratings: pd.DataFrame, popularity_rank: pd.Series) -> pd.DataFrame:
    """
    Core function: computes all 15 user features.
    
    Strategy:
      - Use pandas groupby for vectorized operations (fast)
      - Session features computed separately (requires sorting per user)
      - Merge all feature groups on userId
    
    Returns DataFrame with columns:
      [user_idx, event_timestamp, feat1, feat2, ..., feat15]
    """
    print("\n  Computing user features (this takes 2-8 minutes)...")

    # ── GROUP 1: Basic rating statistics (fast, vectorized) ──
    print("    [1/7] Basic rating statistics...")
    grp = ratings.groupby("userId")

    basic_stats = grp["rating"].agg(
        avg_rating="mean",
        total_interactions="count",
        rating_std="std",
        min_rating="min",
        max_rating="max"
    ).reset_index()
    basic_stats["rating_std"] = basic_stats["rating_std"].fillna(0.0)
    basic_stats.columns = ["userId", "avg_rating", "total_interactions",
                           "rating_std", "min_rating", "max_rating"]
    basic_stats["avg_rating"]         = basic_stats["avg_rating"].astype(np.float32)
    basic_stats["total_interactions"] = basic_stats["total_interactions"].astype(np.int32)
    basic_stats["rating_std"]         = basic_stats["rating_std"].astype(np.float32)
    basic_stats["min_rating"]         = basic_stats["min_rating"].astype(np.float32)
    basic_stats["max_rating"]         = basic_stats["max_rating"].astype(np.float32)

    # ── GROUP 2: High rating ratio ──
    print("    [2/7] High rating ratio...")
    high_rating = grp.apply(
        lambda x: (x["rating"] >= 4.0).sum() / len(x)
    ).reset_index()
    high_rating.columns = ["userId", "high_rating_ratio"]
    high_rating["high_rating_ratio"] = high_rating["high_rating_ratio"].astype(np.float32)

    # ── GROUP 3: Temporal features ──
    print("    [3/7] Temporal features...")
    temporal = grp["timestamp"].agg(
        first_ts="min",
        last_ts="max"
    ).reset_index()
    temporal.columns = ["userId", "first_ts", "last_ts"]

    # Max timestamp across entire dataset = "current time" for offline features
    max_ts = int(ratings["timestamp"].max())

    temporal["temporal_spread_days"] = (
        (temporal["last_ts"] - temporal["first_ts"]) / 86400
    ).clip(lower=0).astype(np.float32)

    temporal["recency_days"] = (
        (max_ts - temporal["last_ts"]) / 86400
    ).clip(lower=0).astype(np.float32)

    # ── GROUP 4: Active days ──
    print("    [4/7] Active days...")
    active_days = grp["date"].nunique().reset_index()
    active_days.columns = ["userId", "active_days"]
    active_days["active_days"] = active_days["active_days"].astype(np.int32)

    # ── GROUP 5: Interaction density ──
    # = total_interactions / active_days
    # (computed after merging basic_stats + active_days)

    # ── GROUP 6: Genre features ──
    print("    [5/7] Genre features...")

    def user_genre_stats(user_df):
        """
        For a single user's ratings, returns:
          genre_diversity: int (number of unique genres)
          top_genre_idx: int (index of most common genre)
        """
        genre_counts = {}
        for genre_str in user_df["genres"]:
            for genre in parse_genres(genre_str):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        if not genre_counts:
            return pd.Series({"genre_diversity": 0, "top_genre_idx": -1})

        top_genre = max(genre_counts, key=genre_counts.get)
        return pd.Series({
            "genre_diversity": len(genre_counts),
            "top_genre_idx": GENRE2IDX.get(top_genre, -1)
        })

    # NOTE: apply() with a custom function is slow for large DataFrames.
    # Optimization: parse genres once, explode, then use vectorized groupby.
    ratings["genre_list"] = ratings["genres"].apply(parse_genres)
    exploded_genres = ratings[["userId", "genre_list"]].explode("genre_list")
    exploded_genres = exploded_genres[exploded_genres["genre_list"].notna()]
    exploded_genres.columns = ["userId", "genre"]

    genre_counts_per_user = (
        exploded_genres.groupby(["userId", "genre"])
        .size()
        .reset_index(name="count")
    )

    genre_diversity = (
        genre_counts_per_user.groupby("userId")["genre"]
        .nunique()
        .reset_index()
    )
    genre_diversity.columns = ["userId", "genre_diversity"]
    genre_diversity["genre_diversity"] = genre_diversity["genre_diversity"].astype(np.int32)

    top_genre = (
        genre_counts_per_user.sort_values("count", ascending=False)
        .drop_duplicates("userId")[["userId", "genre"]]
    )
    top_genre["top_genre_idx"] = top_genre["genre"].map(GENRE2IDX).fillna(-1).astype(np.int32)
    top_genre = top_genre[["userId", "top_genre_idx"]]

    # ── GROUP 7: Session features ──
    print("    [6/7] Session features (slowest step)...")
    print("          (processing per-user session boundaries...)")

    # Compute session features using optimized approach:
    # Sort all ratings by userId + timestamp, then use numpy diff
    ratings_sorted = ratings[["userId", "timestamp"]].sort_values(["userId", "timestamp"])

    user_ids = ratings_sorted["userId"].values
    timestamps = ratings_sorted["timestamp"].values

    # Find user boundaries
    user_change = np.concatenate([[True], user_ids[1:] != user_ids[:-1], [True]])
    user_start_idx = np.where(user_change)[0]
    unique_users = user_ids[user_start_idx[:-1]]

    session_counts = []
    avg_session_lengths = []

    for i in range(len(user_start_idx) - 1):
        start = user_start_idx[i]
        end   = user_start_idx[i + 1]
        ts    = timestamps[start:end]

        if len(ts) <= 1:
            session_counts.append(1)
            avg_session_lengths.append(float(len(ts)))
            continue

        gaps = np.diff(ts)
        n_sessions = int(np.sum(gaps > 3600)) + 1
        session_lengths = np.diff(
            np.concatenate([[-1], np.where(gaps > 3600)[0], [len(ts) - 1]])
        )
        avg_len = float(np.mean(session_lengths))
        session_counts.append(n_sessions)
        avg_session_lengths.append(round(avg_len, 2))

    session_df = pd.DataFrame({
        "userId": unique_users,
        "session_count": np.array(session_counts, dtype=np.int32),
        "avg_session_length": np.array(avg_session_lengths, dtype=np.float32)
    })

    # ── GROUP 8: Popularity bias ──
    print("    [7/7] Popularity bias...")
    ratings_with_pop = ratings[["userId", "movieId"]].copy()
    ratings_with_pop["pop_rank"] = ratings_with_pop["movieId"].map(popularity_rank).fillna(9999)
    pop_bias = ratings_with_pop.groupby("userId")["pop_rank"].mean().reset_index()
    pop_bias.columns = ["userId", "popularity_bias"]
    pop_bias["popularity_bias"] = pop_bias["popularity_bias"].astype(np.float32)

    # ─────────────────────────────────────────────────────────────
    # MERGE ALL FEATURE GROUPS
    # ─────────────────────────────────────────────────────────────
    print("\n  Merging feature groups...")

    df = basic_stats.copy()
    df = df.merge(high_rating,    on="userId", how="left")
    df = df.merge(temporal[["userId", "temporal_spread_days", "recency_days"]],
                  on="userId", how="left")
    df = df.merge(active_days,    on="userId", how="left")
    df = df.merge(genre_diversity, on="userId", how="left")
    df = df.merge(top_genre,       on="userId", how="left")
    df = df.merge(session_df,      on="userId", how="left")
    df = df.merge(pop_bias,        on="userId", how="left")

    # Compute interaction density (requires total_interactions + active_days)
    df["interaction_density"] = (
        df["total_interactions"] / df["active_days"].clip(lower=1)
    ).astype(np.float32)

    # Fill any remaining NaN values
    df["genre_diversity"]    = df["genre_diversity"].fillna(0).astype(np.int32)
    df["top_genre_idx"]      = df["top_genre_idx"].fillna(-1).astype(np.int32)
    df["session_count"]      = df["session_count"].fillna(1).astype(np.int32)
    df["avg_session_length"] = df["avg_session_length"].fillna(0.0).astype(np.float32)
    df["popularity_bias"]    = df["popularity_bias"].fillna(9999.0).astype(np.float32)

    print(f"  User features DataFrame shape: {df.shape}")
    print(f"  Users with features: {len(df):,}")

    return df


# ─────────────────────────────────────────────────────────────
# STEP 5: MAP userId → user_idx (Feast uses integer entity keys)
# ─────────────────────────────────────────────────────────────

def add_feast_columns(df: pd.DataFrame, max_ts: int) -> pd.DataFrame:
    """
    Feast requires:
      1. An entity column matching the entity join key name: 'user_idx'
      2. An 'event_timestamp' column (timezone-aware datetime)

    We map MovieLens userId → user_idx using our item2idx mapping.
    (The user2idx mapping was created during Week 1 data pipeline.)
    """
    # Load user2idx mapping (created in Week 1)
    user2idx_path = PROJECT_ROOT / "data" / "processed" / "user2idx.json"

    if user2idx_path.exists():
        print("  Loading user2idx mapping from Week 1...")
        with open(user2idx_path) as f:
            user2idx = json.load(f)
        # user2idx keys might be strings → convert
        user2idx = {int(k): int(v) for k, v in user2idx.items()}
        df["user_idx"] = df["userId"].map(user2idx)
    else:
        # Fallback: create sequential mapping
        print("  user2idx.json not found — creating sequential mapping...")
        unique_users = sorted(df["userId"].unique())
        user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        df["user_idx"] = df["userId"].map(user2idx)
        # Save for future use
        with open(user2idx_path, "w") as f:
            json.dump({str(k): v for k, v in user2idx.items()}, f)
        print(f"  Saved user2idx to {user2idx_path}")

    # Drop rows where mapping failed
    before = len(df)
    df = df.dropna(subset=["user_idx"])
    df["user_idx"] = df["user_idx"].astype(np.int32)
    after = len(df)
    if before != after:
        print(f"  ⚠️  Dropped {before - after} users without user_idx mapping")

    # Add event_timestamp column (Feast requirement)
    # For offline features: use the global max timestamp (features are "as of" that point)
    # In production, this would be the actual computation timestamp
    feast_timestamp = pd.Timestamp(max_ts, unit="s", tz="UTC")
    df["event_timestamp"] = feast_timestamp

    print(f"  event_timestamp set to: {feast_timestamp}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 6: WRITE TO PARQUET
# ─────────────────────────────────────────────────────────────

def write_parquet(df: pd.DataFrame, output_path: Path):
    """
    Writes the feature DataFrame to Parquet format.
    Feast requires the Parquet file to:
      - Have an entity column (user_idx)
      - Have an event_timestamp column (datetime64[us, UTC])
      - Have one row per entity
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Feast expects specific column ordering: entity + timestamp first
    feature_cols = [c for c in df.columns if c not in ["userId", "user_idx", "event_timestamp"]]
    final_cols = ["user_idx", "event_timestamp"] + feature_cols
    df_final = df[final_cols].copy()

    # Ensure event_timestamp is timezone-aware datetime
    if not hasattr(df_final["event_timestamp"].dtype, "tz"):
        df_final["event_timestamp"] = pd.to_datetime(
            df_final["event_timestamp"], utc=True
        )

    print(f"\n  Writing to: {output_path}")
    print(f"  Shape: {df_final.shape}")
    print(f"  Columns: {list(df_final.columns)}")
    print(f"  Memory usage: {df_final.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Write with compression for efficiency
    table = pa.Table.from_pandas(df_final, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")

    # Verify
    verify = pd.read_parquet(output_path)
    print(f"\n  ✅ Parquet verified:")
    print(f"     Rows: {len(verify):,}")
    print(f"     Columns: {list(verify.columns)}")
    print(f"\n  Feature statistics:")
    print(df_final[feature_cols].describe().to_string())


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  DS19 Week 7 — User Feature Engineering Pipeline")
    print("=" * 65)

    # Step 1: Load raw data
    print("\n[1/6] Loading raw data...")
    ratings = load_raw_data()
    max_ts = int(ratings["timestamp"].max())
    print(f"  Max timestamp: {max_ts} ({pd.to_datetime(max_ts, unit='s')})")

    # Step 2: Compute item popularity
    print("\n[2/6] Computing item popularity ranks...")
    popularity_rank = compute_item_popularity(ratings)
    print(f"  Item popularity computed for {len(popularity_rank):,} items")

    # Step 3: Compute user features
    print("\n[3/6] Computing user features...")
    user_features_df = compute_user_features(ratings, popularity_rank)

    # Step 4: Add Feast columns
    print("\n[4/6] Adding Feast entity and timestamp columns...")
    user_features_df = add_feast_columns(user_features_df, max_ts)

    # Step 5: Write to Parquet
    print("\n[5/6] Writing to Parquet...")
    write_parquet(user_features_df, OUTPUT_PATH)

    # Step 6: Summary
    print("\n[6/6] Summary")
    print("=" * 65)
    print(f"  ✅ User features written to:")
    print(f"     {OUTPUT_PATH}")
    print(f"  ✅ Total users: {len(user_features_df):,}")
    print(f"  ✅ Features per user: {len(user_features_df.columns) - 2}")  # -2 for entity+ts
    print("=" * 65)
    print("\n  Next: python feature_store/pipelines/item_features_pipeline.py")


if __name__ == "__main__":
    main()