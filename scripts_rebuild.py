# save as: scripts/rebuild_meta.py
import os
import json
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
META_PATH = Path("models/saved/dataset_meta.json")

# ─────────────────────────────────────────────────────────────
# REBUILD METADATA
# ─────────────────────────────────────────────────────────────

def rebuild_metadata():
    print("🔧 Rebuilding dataset_meta.json...")
    
    # Load existing meta if it exists
    if META_PATH.exists():
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        print("  ✅ Loaded existing metadata")
    else:
        meta = {}
        print("  ⚠️  No existing metadata found — creating new")
    
    # ── Load processed data ───────────────────────────────────
    interactions_path = PROCESSED_DIR / "interactions.csv"
    if not interactions_path.exists():
        print("  ❌ interactions.csv not found!")
        print("     Run: python data/week1_problem_formulation.py")
        return False
    
    df = pd.read_csv(interactions_path)
    print(f"  ✅ Loaded {len(df):,} interactions")
    
    # ── Compute core stats ────────────────────────────────────
    n_users = df['user_idx'].nunique()
    n_items = df['item_idx'].nunique()
    n_interactions = len(df)
    
    meta['n_users'] = n_users
    meta['n_items'] = n_items
    meta['n_interactions'] = n_interactions
    meta['n_items_with_pad'] = n_items + 1  # +1 for PAD token
    meta['pad_token'] = 0
    meta['sparsity'] = 1 - (n_interactions / (n_users * n_items))
    meta['density'] = n_interactions / (n_users * n_items)
    meta['avg_interactions_user'] = n_interactions / n_users
    meta['avg_interactions_item'] = n_interactions / n_items
    
    # ── Load mappings from processed dir ──────────────────────
    user2idx_path = PROCESSED_DIR / "user2idx.json"
    item2idx_path = PROCESSED_DIR / "item2idx.json"
    idx2item_path = PROCESSED_DIR / "idx2item.json"
    
    if user2idx_path.exists():
        with open(user2idx_path, "r") as f:
            meta['user2id'] = json.load(f)
        meta['id2user'] = {str(v): k for k, v in meta['user2id'].items()}
        print("  ✅ Loaded user mappings")
    
    if item2idx_path.exists():
        with open(item2idx_path, "r") as f:
            meta['item2id'] = json.load(f)
        meta['id2item'] = {str(v): k for k, v in meta['item2id'].items()}
        print("  ✅ Loaded item mappings")
    
    # ── Load sequence stats ───────────────────────────────────
    train_seqs_path = SPLITS_DIR / "train_seqs.pkl"
    if train_seqs_path.exists():
        import pickle
        with open(train_seqs_path, "rb") as f:
            train_seqs = pickle.load(f)
        meta['n_split_users'] = len(train_seqs)
        meta['max_seq_len'] = 50  # From Week 1 config
        print(f"  ✅ Loaded {len(train_seqs):,} train sequences")
    
    # ── Add Week 4/5 required keys ────────────────────────────
    meta['num_users'] = meta.get('num_users', n_users)
    meta['num_items'] = meta.get('num_items', n_items)
    meta['pad_id'] = meta.get('pad_id', 0)
    meta['item_vocab_size'] = n_items + 1
    meta['user_vocab_size'] = n_users + 1
    
    # ── Save ──────────────────────────────────────────────────
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ Metadata rebuilt successfully!")
    print(f"   File: {META_PATH}")
    print(f"   Users: {meta['n_users']:,}")
    print(f"   Items: {meta['n_items']:,}")
    
    return True

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rebuild_metadata()