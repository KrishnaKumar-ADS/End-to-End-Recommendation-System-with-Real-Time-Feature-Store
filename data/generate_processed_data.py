"""
DS19 — Generate Processed Data for Week 5+
Creates all files in data/processed/ from raw MovieLens 25M dataset.
Run: python data/generate_processed_data.py
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Filtering thresholds
RATING_THRESHOLD = 4.0
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_stats(df, label: str):
    print(f"\n  [{label}]")
    print(f"  Rows    : {len(df):,}")
    print(f"  Users   : {df['userId'].nunique():,}")
    print(f"  Items   : {df['movieId'].nunique():,}")

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD RAW DATA
# ─────────────────────────────────────────────────────────────

def load_raw_data():
    print_section("Step 1: Loading Raw MovieLens 25M Data")
    
    ratings_path = RAW_DIR / "ratings.csv"
    movies_path = RAW_DIR / "movies.csv"
    
    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing {ratings_path}. Run download_dataset.py first.")
    
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    print(f"  ✅ Loaded ratings: {len(ratings):,} rows")
    print(f"  ✅ Loaded movies:  {len(movies):,} rows")
    
    return ratings, movies

# ─────────────────────────────────────────────────────────────
# STEP 2: FILTER BY RATING THRESHOLD
# ─────────────────────────────────────────────────────────────

def filter_by_rating(ratings: pd.DataFrame) -> pd.DataFrame:
    print_section("Step 2: Filter by Rating Threshold")
    print(f"  Threshold: rating ≥ {RATING_THRESHOLD}")
    
    before = len(ratings)
    filtered = ratings[ratings['rating'] >= RATING_THRESHOLD].copy()
    after = len(filtered)
    
    print(f"  Before: {before:,} ratings")
    print(f"  After:  {after:,} positive interactions")
    return filtered

# ─────────────────────────────────────────────────────────────
# STEP 3: REMOVE COLD-START USERS & ITEMS
# ─────────────────────────────────────────────────────────────

def remove_cold_start(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Step 3: Remove Cold-Start Users & Items")
    
    iteration = 0
    while True:
        iteration += 1
        n_before = len(df)
        
        user_counts = df.groupby('userId').size()
        valid_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
        df = df[df['userId'].isin(valid_users)]
        
        item_counts = df.groupby('movieId').size()
        valid_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
        df = df[df['movieId'].isin(valid_items)]
        
        n_after = len(df)
        removed = n_before - n_after
        
        if removed == 0:
            break
        if iteration > 10:
            break
    
    print_stats(df, "After Cold-Start Removal")
    return df.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# STEP 4: ENCODE USER & ITEM IDs
# ─────────────────────────────────────────────────────────────

def encode_ids(df: pd.DataFrame):
    print_section("Step 4: Encode User & Item IDs")
    
    unique_users = sorted(df['userId'].unique())
    unique_items = sorted(df['movieId'].unique())
    
    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item2idx = {mid: idx for idx, mid in enumerate(unique_items)}
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}
    
    df['user_idx'] = df['userId'].map(user2idx)
    df['item_idx'] = df['movieId'].map(item2idx)
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    print(f"  Total unique users: {n_users:,}")
    print(f"  Total unique items: {n_items:,}")
    
    return df, user2idx, item2idx, idx2user, idx2item, n_users, n_items

# ─────────────────────────────────────────────────────────────
# STEP 5: SAVE ALL PROCESSED FILES (FIXED FOR WEEK 5)
# ─────────────────────────────────────────────────────────────

def save_processed(df, user2idx, item2idx, idx2user, idx2item, 
                   n_users, n_items, movies_df):
    print_section("Step 5: Saving Processed Files")
    
    # ── 1. ratings_filtered.csv (CRITICAL FOR WEEK 5) ──
    # Keeps original userId, movieId, rating, timestamp
    ratings_filtered = df[['userId', 'movieId', 'rating', 'timestamp']].copy()
    ratings_filtered.to_csv(PROCESSED_DIR / "ratings_filtered.csv", index=False)
    print(f"  ✅ Saved: data/processed/ratings_filtered.csv ({len(ratings_filtered):,} rows)")
    
    # ── 2. interactions.csv (For Week 2+ model training) ──
    interactions = df[['user_idx', 'item_idx', 'timestamp']].copy()
    interactions['interaction'] = 1
    interactions.to_csv(PROCESSED_DIR / "interactions.csv", index=False)
    print(f"  ✅ Saved: data/processed/interactions.csv ({len(interactions):,} rows)")
    
    # ── 3. ID mapping files (FIXED: Cast to native Python int) ──
    with open(PROCESSED_DIR / "user2idx.json", "w") as f:
        json.dump({str(k): int(v) for k, v in user2idx.items()}, f)
    
    with open(PROCESSED_DIR / "item2idx.json", "w") as f:
        json.dump({str(k): int(v) for k, v in item2idx.items()}, f)
    
    with open(PROCESSED_DIR / "idx2user.json", "w") as f:
        json.dump({int(k): int(v) for k, v in idx2user.items()}, f)
    
    with open(PROCESSED_DIR / "idx2item.json", "w") as f:
        json.dump({int(k): int(v) for k, v in idx2item.items()}, f)
    
    print(f"  ✅ Saved: ID mapping JSON files")
    
    # ── 4. movies.csv (Filtered to our vocabulary) ──
    movies_filtered = movies_df[movies_df['movieId'].isin(item2idx.keys())].copy()
    movies_filtered.to_csv(PROCESSED_DIR / "movies.csv", index=False)
    print(f"  ✅ Saved: data/processed/movies.csv ({len(movies_filtered):,} movies)")
    
    # ── 5. dataset_meta.json (FIXED: Includes BOTH key styles) ──
    n_interactions = len(df)
    sparsity = 1 - (n_interactions / (n_users * n_items))
    density = n_interactions / (n_users * n_items)
    
    meta = {
        # Week 1 Style
        "n_users": int(n_users),
        "n_items": int(n_items),
        # Week 5 Style (Aliases)
        "num_users": int(n_users),
        "num_items": int(n_items),
        # Stats
        "n_interactions": int(n_interactions),
        "rating_threshold": float(RATING_THRESHOLD),
        "min_user_interactions": int(MIN_USER_INTERACTIONS),
        "min_item_interactions": int(MIN_ITEM_INTERACTIONS),
        "sparsity": float(sparsity),
        "density": float(density),
        "avg_interactions_user": float(n_interactions / n_users),
        "avg_interactions_item": float(n_interactions / n_items),
        # Mappings (Cast to native types)
        "user2id": {str(k): int(v) for k, v in user2idx.items()},
        "id2user": {str(k): int(v) for k, v in idx2user.items()},
        "item2id": {str(k): int(v) for k, v in item2idx.items()},
        "id2item": {str(k): int(v) for k, v in idx2item.items()},
        "created_at": datetime.now().isoformat(),
    }
    
    with open(PROCESSED_DIR / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ Saved: data/processed/dataset_meta.json")
    
    # ── 6. Copy Sequences for Week 5 Compatibility ──
    # Week 5 expects 'train_sequences.pkl', Week 1 creates 'train_seqs.pkl'
    train_seqs_src = SPLITS_DIR / "train_seqs.pkl"
    train_seqs_dst = SPLITS_DIR / "train_sequences.pkl"
    if train_seqs_src.exists():
        import shutil
        shutil.copy(train_seqs_src, train_seqs_dst)
        print(f"  ✅ Saved: data/splits/train_sequences.pkl (compatibility copy)")
    
    print(f"\n  📊 Final Dataset Summary:")
    print(f"    Users          : {n_users:,}")
    print(f"    Items          : {n_items:,}")
    print(f"    Interactions   : {n_interactions:,}")
    
    return meta

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DS19 — Generate Processed Data")
    print("  From MovieLens 25M → data/processed/")
    print("=" * 60)
    
    ratings, movies = load_raw_data()
    ratings = filter_by_rating(ratings)
    ratings = remove_cold_start(ratings)
    ratings, user2idx, item2idx, idx2user, idx2item, n_users, n_items = encode_ids(ratings)
    meta = save_processed(ratings, user2idx, item2idx, idx2user, idx2item, 
                          n_users, n_items, movies)
    
    print()
    print("=" * 60)
    print("  ✅ Processed Data Generation Complete!")
    print("=" * 60)
    print("\n  Next: python data/features/feature_engineering.py")

if __name__ == "__main__":
    main()