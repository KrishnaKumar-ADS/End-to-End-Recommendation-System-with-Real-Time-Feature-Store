import re
import sys
import json
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
RATINGS_PATH   = PROJECT_ROOT / "data" / "raw" / "ratings.csv"
MOVIES_PATH    = PROJECT_ROOT / "data" / "raw" / "movies.csv"
ITEM2IDX_PATH  = PROJECT_ROOT / "data" / "processed" / "item2idx.json"
OUTPUT_PATH    = PROJECT_ROOT / "feature_store" / "data" / "item_features.parquet"

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western", "(no genres listed)"
]
GENRE2IDX = {g: i for i, g in enumerate(ALL_GENRES)}


# ─────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────

def extract_release_year(title: str) -> int:
    """
    Extracts release year from MovieLens title format:
    'Toy Story (1995)' → 1995
    Returns 0 if not found.
    """
    if not isinstance(title, str):
        return 0
    match = re.search(r'\((\d{4})\)\s*$', title.strip())
    if match:
        year = int(match.group(1))
        if 1888 <= year <= 2030:  # sanity check
            return year
    return 0


def parse_genres(genre_str: str) -> list:
    if not isinstance(genre_str, str) or genre_str == "(no genres listed)":
        return []
    return [g.strip() for g in genre_str.split("|") if g.strip() in GENRE2IDX]


# ─────────────────────────────────────────────────────────────
# MAIN FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_item_features() -> pd.DataFrame:
    """
    Computes all 12 item features.
    Returns DataFrame indexed by movieId.
    """

    # ── LOAD DATA ──
    print("  Loading ratings.csv (all ratings, not just implicit)...")
    ratings = pd.read_csv(
        RATINGS_PATH,
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
        usecols=["movieId", "rating", "timestamp"]
    )
    print(f"  Total ratings loaded: {len(ratings):,}")

    print("  Loading movies.csv...")
    movies = pd.read_csv(MOVIES_PATH)
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    print(f"  Total movies: {len(movies):,}")

    # ── FEATURE GROUP 1: Rating statistics (all ratings, not just implicit) ──
    print("\n  [1/5] Computing rating statistics...")
    # Use ALL ratings (not just implicit >=4) for item statistics
    # because items should reflect their full rating distribution
    grp = ratings.groupby("movieId")

    rating_stats = grp["rating"].agg(
        avg_item_rating="mean",
        rating_count="count",
        rating_std="std",
    ).reset_index()
    rating_stats["rating_std"] = rating_stats["rating_std"].fillna(0.0)
    rating_stats.columns = ["movieId", "avg_item_rating", "rating_count", "rating_std"]

    high_ratio = grp.apply(lambda x: (x["rating"] >= 4.0).mean()).reset_index()
    high_ratio.columns = ["movieId", "high_rating_ratio"]

    # ── FEATURE GROUP 2: Popularity features ──
    print("  [2/5] Computing popularity features...")

    # Global popularity = number of ALL interactions (implicit filter too)
    implicit_ratings = ratings[ratings["rating"] >= 4.0]
    global_popularity = (
        implicit_ratings.groupby("movieId").size().reset_index(name="global_popularity")
    )

    # Popularity rank (1 = most popular)
    global_popularity["popularity_rank"] = (
        global_popularity["global_popularity"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Niche score = 1 / log(1 + popularity), so rare items have high niche score
    max_pop = global_popularity["global_popularity"].max()
    global_popularity["niche_score"] = (
        1.0 / np.log1p(global_popularity["global_popularity"])
    ).astype(np.float32)

    # ── FEATURE GROUP 3: Temporal features ──
    print("  [3/5] Computing temporal features...")
    temporal = grp["timestamp"].agg(
        first_rating_ts="min",
        last_rating_ts="max"
    ).reset_index()

    max_ts = int(ratings["timestamp"].max())

    temporal["item_active_days"] = (
        (temporal["last_rating_ts"] - temporal["first_rating_ts"]) / 86400
    ).clip(lower=1).astype(np.float32)

    # ── FEATURE GROUP 4: Velocity ──
    # avg_ratings_per_day = rating_count / item_active_days
    # (computed after merge)

    # ── FEATURE GROUP 5: Content features (from movies.csv) ──
    print("  [4/5] Computing content features from movies.csv...")

    movies["release_year"] = movies["title"].apply(extract_release_year)

    movies["genre_list"] = movies["genres"].apply(parse_genres)

    movies["genre_count"] = movies["genre_list"].apply(len).astype(np.int32)

    movies["primary_genre_idx"] = movies["genre_list"].apply(
        lambda g: GENRE2IDX.get(g[0], -1) if len(g) > 0 else -1
    ).astype(np.int32)

    # Approximate item_age_days using release year and dataset max timestamp
    max_year = pd.to_datetime(max_ts, unit="s").year
    movies["item_age_days"] = (
        (max_year - movies["release_year"]).clip(lower=0) * 365
    ).astype(np.float32)
    movies.loc[movies["release_year"] == 0, "item_age_days"] = 0.0

    # ── MERGE ALL GROUPS ──
    print("  [5/5] Merging feature groups...")

    df = movies[["movieId", "release_year", "genre_count",
                 "primary_genre_idx", "item_age_days"]].copy()

    df = df.merge(rating_stats, on="movieId", how="left")
    df = df.merge(high_ratio, on="movieId", how="left")
    df = df.merge(global_popularity[["movieId", "global_popularity",
                                     "popularity_rank", "niche_score"]],
                  on="movieId", how="left")
    df = df.merge(temporal[["movieId", "item_active_days"]], on="movieId", how="left")

    # Fill items with no ratings
    df["global_popularity"]  = df["global_popularity"].fillna(0).astype(np.int32)
    df["popularity_rank"]    = df["popularity_rank"].fillna(99999).astype(np.int32)
    df["niche_score"]        = df["niche_score"].fillna(1.0).astype(np.float32)
    df["avg_item_rating"]    = df["avg_item_rating"].fillna(0.0).astype(np.float32)
    df["rating_count"]       = df["rating_count"].fillna(0).astype(np.int32)
    df["rating_std"]         = df["rating_std"].fillna(0.0).astype(np.float32)
    df["high_rating_ratio"]  = df["high_rating_ratio"].fillna(0.0).astype(np.float32)
    df["item_active_days"]   = df["item_active_days"].fillna(1.0).astype(np.float32)

    # Velocity: ratings per active day
    df["avg_ratings_per_day"] = (
        df["rating_count"] / df["item_active_days"].clip(lower=1)
    ).astype(np.float32)

    print(f"\n  Item features DataFrame shape: {df.shape}")
    return df, max_ts


def add_feast_columns(df: pd.DataFrame, max_ts: int) -> pd.DataFrame:
    """
    Adds Feast-required columns:
      - item_idx: integer entity key (mapped from movieId)
      - event_timestamp: timezone-aware datetime
    """
    # Load item2idx mapping
    print("  Loading item2idx mapping...")
    with open(ITEM2IDX_PATH) as f:
        item2idx = json.load(f)
    item2idx = {int(k): int(v) for k, v in item2idx.items()}

    df["item_idx"] = df["movieId"].map(item2idx)

    before = len(df)
    df = df.dropna(subset=["item_idx"])
    df["item_idx"] = df["item_idx"].astype(np.int32)
    after = len(df)
    print(f"  Items after idx mapping: {after:,} (dropped {before-after} unmapped)")

    feast_timestamp = pd.Timestamp(max_ts, unit="s", tz="UTC")
    df["event_timestamp"] = feast_timestamp

    return df


def write_parquet(df: pd.DataFrame, output_path: Path, max_ts: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns
                    if c not in ["movieId", "item_idx", "event_timestamp", "genre_list"]]
    final_cols = ["item_idx", "event_timestamp"] + feature_cols
    df_final = df[[c for c in final_cols if c in df.columns]].copy()

    print(f"\n  Writing to: {output_path}")
    print(f"  Shape: {df_final.shape}")

    table = pa.Table.from_pandas(df_final, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")

    verify = pd.read_parquet(output_path)
    print(f"\n  ✅ Parquet verified: {len(verify):,} rows")
    print(f"\n  Feature statistics:")
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    print(df_final[numeric_cols].describe().to_string())


def main():
    print("=" * 65)
    print("  DS19 Week 7 — Item Feature Engineering Pipeline")
    print("=" * 65)

    print("\n[1/4] Computing item features...")
    item_features_df, max_ts = compute_item_features()

    print("\n[2/4] Adding Feast entity and timestamp columns...")
    item_features_df = add_feast_columns(item_features_df, max_ts)

    print("\n[3/4] Writing to Parquet...")
    write_parquet(item_features_df, OUTPUT_PATH, max_ts)

    print("\n[4/4] Summary")
    print("=" * 65)
    print(f"  ✅ Item features written to:")
    print(f"     {OUTPUT_PATH}")
    print(f"  ✅ Total items with features: {len(item_features_df):,}")
    print(f"  ✅ Features per item: 12")
    print("=" * 65)
    print("\n  Next: python feature_store/feature_repo/ feast definitions")


if __name__ == "__main__":
    main()