"""
DS19 — Week 5: Ranking Dataset Builder (RTX 3050 4GB - Final Version)
Builds training/validation datasets for LightGBM ranking model.

SMART BUILD: Checks if valid datasets exist, only rebuilds if needed.

Run: python models/ranking/dataset_builder.py
"""

import os
import sys
import json
import pickle
import gc
import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
from tqdm import tqdm
from pathlib import Path
import warnings
import psutil

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"

META_PATH = PROCESSED_DIR / "dataset_meta.json"
TWO_TOWER_PATH = MODELS_DIR / "two_tower_best.pt"
FAISS_INDEX_PATH = RETRIEVAL_DIR / "faiss_item.index"
ITEM_EMBEDDINGS_PATH = RETRIEVAL_DIR / "item_embeddings.npy"
ITEM_IDS_PATH = RETRIEVAL_DIR / "item_ids.npy"

# ── CONFIG ──
BATCH_SIZE = 8
USE_AMP = False
DTYPE = torch.float32
NUM_CANDIDATES = 100
MAX_TRAIN_USERS = 50_000
MAX_VAL_USERS = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INTERMEDIATE_SAVE_EVERY = 200

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MEMORY MONITORING
# ─────────────────────────────────────────────────────────────

def log_memory_usage(prefix=""):
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 / 1024
    if DEVICE == "cuda":
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_max = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  {prefix} RAM: {ram_mb:.0f}MB | GPU: {gpu_mb:.0f}MB (max: {gpu_max:.0f}MB)")
    else:
        print(f"  {prefix} RAM: {ram_mb:.0f}MB")

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────

from data.features.feature_engineering import (
    load_features,
    build_retrieval_features,
    build_interaction_features,
)

# ─────────────────────────────────────────────────────────────
# CHECK IF DATASET IS VALID
# ─────────────────────────────────────────────────────────────

def check_dataset_valid(split_name):
    """Check if dataset exists and has required columns."""
    df_path = FEATURES_DIR / f"ranking_{split_name}.parquet"
    groups_path = FEATURES_DIR / f"groups_{split_name}.pkl"
    feat_path = FEATURES_DIR / "feature_names.json"
    
    if not df_path.exists():
        print(f"  ❌ {split_name}: Dataset file missing")
        return False
    
    if not groups_path.exists():
        print(f"  ❌ {split_name}: Groups file missing")
        return False
    
    if not feat_path.exists():
        print(f"  ❌ {split_name}: Feature names file missing")
        return False
    
    try:
        df = pd.read_parquet(df_path)
        required_cols = ["label", "__user_id", "__item_id"]
        missing = [c for c in required_cols if c not in df.columns]
        
        if missing:
            print(f"  ❌ {split_name}: Missing columns: {missing}")
            del df
            return False
        
        if "label" not in df.columns:
            print(f"  ❌ {split_name}: 'label' column missing")
            del df
            return False
        
        label_unique = df["label"].unique()
        if not all(l in [0, 1, 0.0, 1.0] for l in label_unique):
            print(f"  ❌ {split_name}: Invalid label values: {label_unique}")
            del df
            return False
        
        row_count = len(df)
        print(f"  ✅ {split_name}: Valid ({row_count:,} rows, {df['label'].sum():,} positives)")
        del df
        return True
        
    except Exception as e:
        print(f"  ❌ {split_name}: Error reading file: {e}")
        return False


def load_existing_dataset(split_name):
    """Load existing valid dataset."""
    df_path = FEATURES_DIR / f"ranking_{split_name}.parquet"
    groups_path = FEATURES_DIR / f"groups_{split_name}.pkl"
    feat_path = FEATURES_DIR / "feature_names.json"
    
    df = pd.read_parquet(df_path)
    with open(groups_path, "rb") as f:
        groups = pickle.load(f)
    with open(feat_path, "r") as f:
        feature_names = json.load(f)
    
    meta_cols = [c for c in ["__user_id", "__item_id", "label", "group_id"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    leaky_patterns = ["item_last_timestamp", "item_first_timestamp",
                      "user_last_timestamp", "user_first_timestamp"]
    feature_cols = [c for c in feature_cols
                    if not any(p in c for p in leaky_patterns)]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    meta_df = df[meta_cols].copy()
    
    del df
    gc.collect()
    
    return X, y, groups, meta_df, feature_names


# ─────────────────────────────────────────────────────────────
# LOAD TWO-TOWER MODEL
# ─────────────────────────────────────────────────────────────

def load_two_tower_model(meta):
    print("  Loading Two-Tower model...")
    from models.two_tower.model import TwoTowerModel

    n_items = meta.get("num_items", meta.get("n_items", 19935))

    if not TWO_TOWER_PATH.exists():
        print(f"  ⚠️  Two-Tower model not found at {TWO_TOWER_PATH}")
        return None

    try:
        checkpoint = torch.load(TWO_TOWER_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(TWO_TOWER_PATH, map_location="cpu")

    config = checkpoint.get("config", checkpoint.get("hyperparams", {}))
    if not config:
        config = {
            "n_items": n_items, "hidden_dim": 64, "num_heads": 2,
            "num_blocks": 2, "max_seq_len": 50, "dropout": 0.2,
        }

    try:
        model = TwoTowerModel(
            n_items=config.get("n_items", n_items),
            hidden_dim=config.get("hidden_dim", config.get("embed_dim", 64)),
            num_heads=config.get("num_heads", 2),
            num_blocks=config.get("num_blocks", 2),
            max_seq_len=config.get("max_seq_len", config.get("max_len", 50)),
            dropout=config.get("dropout", 0.2),
            pad_token=config.get("pad_token", 0),
        )
    except TypeError:
        model = TwoTowerModel(
            n_items=config.get("n_items", n_items),
            hidden_dim=config.get("hidden_dim", config.get("embed_dim", 64)),
        )

    state_key = None
    for key in ["model_state_dict", "model_state", "state_dict"]:
        if key in checkpoint:
            state_key = key
            break
    
    if state_key:
        model.load_state_dict(checkpoint[state_key])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    if DEVICE == "cuda":
        model = model.to(DEVICE, dtype=DTYPE)
        print(f"  ✅ Two-Tower loaded → device: {DEVICE}, dtype: {DTYPE}")
    else:
        print(f"  ✅ Two-Tower loaded → device: {DEVICE}")
        
    return model


def encode_user_sequences_batch(model, sequences, max_len=50):
    if model is None:
        batch_size = len(sequences)
        return np.random.randn(batch_size, 64).astype(np.float32)

    padded_seqs = []
    for seq in sequences:
        s = seq[-max_len:]
        padded = [0] * (max_len - len(s)) + list(s)
        padded_seqs.append(padded)

    seq_tensor = torch.tensor(padded_seqs, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        if USE_AMP and DEVICE == "cuda":
            with amp.autocast(dtype=DTYPE):
                user_embs = model.encode_user(seq_tensor)
        else:
            user_embs = model.encode_user(seq_tensor)

    result = user_embs.cpu().numpy()
    
    del seq_tensor, user_embs
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return result


# ─────────────────────────────────────────────────────────────
# FAISS RETRIEVAL
# ─────────────────────────────────────────────────────────────

def load_faiss_index():
    import faiss

    if not FAISS_INDEX_PATH.exists():
        print(f"  ⚠️  FAISS index not found at {FAISS_INDEX_PATH}")
        return None, None, None

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    item_embeddings = np.load(str(ITEM_EMBEDDINGS_PATH))

    if ITEM_IDS_PATH.exists():
        item_ids = np.load(str(ITEM_IDS_PATH)).astype(np.int64)
    else:
        # Fallback for older artifacts: assume contiguous 1..N item IDs.
        item_ids = np.arange(1, index.ntotal + 1, dtype=np.int64)
        print("  ⚠️  item_ids.npy missing; using fallback 1..N mapping")

    if len(item_ids) != index.ntotal:
        print("  ⚠️  item_ids length mismatch with FAISS index; truncating to min length")
        n = min(len(item_ids), index.ntotal)
        item_ids = item_ids[:n]

    print(f"  ✅ FAISS index: {index.ntotal:,} items, dim={index.d}")
    return index, item_embeddings, item_ids


def retrieve_candidates_batch(faiss_index, user_embs, top_k=100, item_ids=None):
    if faiss_index is None:
        batch_size = len(user_embs)
        return (
            np.zeros((batch_size, top_k), dtype=np.float32), 
            np.random.randint(1, 20000, (batch_size, top_k), dtype=np.int64)
        )

    queries = user_embs.astype(np.float32, copy=True)
    
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    norms[norms == 0] = 1
    queries = queries / norms

    scores, item_indices = faiss_index.search(queries, top_k)

    if item_ids is not None:
        mapped_ids = np.zeros_like(item_indices, dtype=np.int64)
        valid_mask = item_indices >= 0
        if valid_mask.any():
            mapped_ids[valid_mask] = item_ids[item_indices[valid_mask]]
        item_indices = mapped_ids

    del queries, norms
    
    return scores, item_indices


# ─────────────────────────────────────────────────────────────
# FEATURE ROW BUILDER
# ─────────────────────────────────────────────────────────────

def prepare_feature_lookups(user_features, item_features):
    user_features = user_features.fillna(0)
    item_features = item_features.fillna(0)
    return user_features.to_dict('index'), item_features.to_dict('index')


def build_feature_rows_batch(user_ids, candidate_item_ids_batch, faiss_scores_batch,
                             user_features_dict, item_features_dict, 
                             user_features_df, item_features_df,
                             genre_cols, train_sequences, meta,
                             targets=None):
    batch_size = len(user_ids)
    num_cands = candidate_item_ids_batch.shape[1]
    
    all_rows = []
    
    for i in range(batch_size):
        user_id = user_ids[i]
        cands = candidate_item_ids_batch[i]
        scores = faiss_scores_batch[i]
        target = targets[i] if targets is not None else -1

        u_feats = user_features_dict.get(user_id, {})
        if not u_feats:
            u_feats = {col: 0.0 for col in user_features_df.columns}
        
        ret_feats = build_retrieval_features(scores, cands)

        try:
            int_feats = build_interaction_features(
                user_id, cands,
                user_features_df, item_features_df,
                genre_cols, train_sequences
            )
        except Exception as e:
            print(f"  ⚠️  Interaction features error: {e}")
            int_feats = {k: np.zeros(num_cands) for k in ["genre_overlap", "popularity_score"]}

        for idx, item_id in enumerate(cands):
            row = {}
            
            for k, v in u_feats.items():
                try:
                    row[f"u_{k}"] = float(v) if not np.isnan(float(v)) else 0.0
                except (TypeError, ValueError):
                    row[f"u_{k}"] = 0.0
            
            i_feats = item_features_dict.get(item_id, {})
            if not i_feats:
                i_feats = {col: 0.0 for col in item_features_df.columns}
            
            for k, v in i_feats.items():
                try:
                    row[f"i_{k}"] = float(v) if not np.isnan(float(v)) else 0.0
                except (TypeError, ValueError):
                    row[f"i_{k}"] = 0.0
            
            for feat_name, feat_array in ret_feats.items():
                row[feat_name] = float(feat_array[idx])
            
            for feat_name, feat_array in int_feats.items():
                row[feat_name] = float(feat_array[idx])
            
            row["__user_id"] = user_id
            row["__item_id"] = item_id
            row["label"] = 1 if item_id == target else 0
            
            all_rows.append(row)
            
    return all_rows


# ─────────────────────────────────────────────────────────────
# SAVE INTERMEDIATE RESULTS
# ─────────────────────────────────────────────────────────────

def save_intermediate_results(all_rows, groups, split_name, batch_num):
    if len(all_rows) == 0:
        return
    
    df = pd.DataFrame(all_rows)
    out_path = FEATURES_DIR / f"ranking_{split_name}_part{batch_num}.parquet"
    df.to_parquet(out_path, index=False)
    
    groups_path = FEATURES_DIR / f"groups_{split_name}_part{batch_num}.pkl"
    with open(groups_path, "wb") as f:
        pickle.dump(groups, f)
    
    print(f"  💾 Saved intermediate: {len(all_rows):,} rows (part {batch_num})")
    
    del df
    gc.collect()
    
    return out_path


def merge_intermediate_results(split_name, num_parts):
    all_dfs = []
    all_groups = []
    
    for i in range(num_parts):
        df_path = FEATURES_DIR / f"ranking_{split_name}_part{i}.parquet"
        groups_path = FEATURES_DIR / f"groups_{split_name}_part{i}.pkl"
        
        if df_path.exists():
            df = pd.read_parquet(df_path)
            all_dfs.append(df)
            
            with open(groups_path, "rb") as f:
                groups = pickle.load(f)
                all_groups.extend(groups)
            
            df_path.unlink()
            groups_path.unlink()
    
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        final_path = FEATURES_DIR / f"ranking_{split_name}.parquet"
        merged_df.to_parquet(final_path, index=False)
        
        final_groups_path = FEATURES_DIR / f"groups_{split_name}.pkl"
        with open(final_groups_path, "wb") as f:
            pickle.dump(all_groups, f)
        
        print(f"  ✅ Merged {num_parts} parts → {len(merged_df):,} rows")
        
        del merged_df, all_dfs
        gc.collect()
        
        return final_path, all_groups
    
    return None, []


# ─────────────────────────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────────────────────────

def build_ranking_dataset(sequences, split_name,
                           two_tower_model, faiss_index,
                           user_features, item_features, genre_cols,
                           train_sequences, meta,
                           faiss_item_id_map=None,
                           max_users=None):
    print(f"\n🔨 Building {split_name} ranking dataset...")
    log_memory_usage("Start")

    if isinstance(sequences, dict):
        seq_list = [
            {"user_id": uid, "seq": v["seq"], "target": v["target"]}
            if isinstance(v, dict) else
            {"user_id": uid, "seq": v[:-1], "target": v[-1]}
            for uid, v in sequences.items()
        ]
    else:
        seq_list = sequences

    if max_users and len(seq_list) > max_users:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(seq_list), max_users, replace=False)
        seq_list = [seq_list[i] for i in indices]

    print(f"  Processing {len(seq_list):,} users in batches of {BATCH_SIZE}...")

    user_features_dict, item_features_dict = prepare_feature_lookups(user_features, item_features)

    all_rows = []
    groups = []
    skipped = 0
    intermediate_count = 0
    
    total_batches = (len(seq_list) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in tqdm(range(total_batches), desc=f"Building {split_name}"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(seq_list))
        batch_data = seq_list[start_idx:end_idx]
        
        user_ids = [entry["user_id"] for entry in batch_data]
        sequences_batch = [entry["seq"] for entry in batch_data]
        targets = [entry["target"] for entry in batch_data]

        valid_indices = [i for i, seq in enumerate(sequences_batch) if seq and len(seq) > 0]
        if not valid_indices:
            skipped += len(batch_data)
            del batch_data, user_ids, sequences_batch, targets
            gc.collect()
            continue
            
        valid_user_ids = [user_ids[i] for i in valid_indices]
        valid_sequences = [sequences_batch[i] for i in valid_indices]
        valid_targets = [targets[i] for i in valid_indices]
        skipped += (len(batch_data) - len(valid_indices))
        
        del batch_data, user_ids, sequences_batch, targets

        user_embs = encode_user_sequences_batch(two_tower_model, valid_sequences)
        del valid_sequences

        faiss_scores, faiss_item_ids = retrieve_candidates_batch(
            faiss_index, user_embs, top_k=NUM_CANDIDATES, item_ids=faiss_item_id_map
        )
        del user_embs

        for i in range(len(valid_user_ids)):
            valid_mask = faiss_item_ids[i] > 0
            scores = faiss_scores[i][valid_mask]
            items = faiss_item_ids[i][valid_mask]

            # Ensure one positive candidate exists per user for supervised ranking.
            target_item = int(valid_targets[i])
            if target_item > 0 and target_item not in items:
                if len(items) == 0:
                    items = np.array([target_item], dtype=np.int64)
                    scores = np.array([0.0], dtype=np.float32)
                else:
                    replace_idx = len(items) - 1
                    min_score = float(scores.min()) if len(scores) > 0 else 0.0
                    items[replace_idx] = target_item
                    scores[replace_idx] = min_score - 1e-6

            if len(items) < NUM_CANDIDATES:
                pad_needed = NUM_CANDIDATES - len(items)
                items = np.concatenate([items, np.zeros(pad_needed, dtype=items.dtype)])
                scores = np.concatenate([scores, np.zeros(pad_needed, dtype=scores.dtype)])
            
            faiss_item_ids[i] = items[:NUM_CANDIDATES]
            faiss_scores[i] = scores[:NUM_CANDIDATES]

        rows = build_feature_rows_batch(
            valid_user_ids, faiss_item_ids, faiss_scores,
            user_features_dict, item_features_dict,
            user_features, item_features,
            genre_cols, train_sequences, meta,
            targets=valid_targets
        )
        
        del faiss_scores, faiss_item_ids, valid_targets

        all_rows.extend(rows)
        groups.extend([NUM_CANDIDATES] * len(valid_user_ids))
        
        del rows, valid_user_ids

        if len(all_rows) >= INTERMEDIATE_SAVE_EVERY * BATCH_SIZE * NUM_CANDIDATES:
            intermediate_count += 1
            save_intermediate_results(all_rows, groups, split_name, intermediate_count)
            all_rows = []
            groups = []
            gc.collect()
            log_memory_usage(f"Batch {batch_idx}")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if all_rows:
        intermediate_count += 1
        save_intermediate_results(all_rows, groups, split_name, intermediate_count)

    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} users with empty sequences")

    print(f"  Merging {intermediate_count} intermediate files...")
    final_path, final_groups = merge_intermediate_results(split_name, intermediate_count)
    
    if final_path is None:
        print("  ❌ No data generated!")
        return None, None, None, None, None

    print(f"  Loading final dataset...")
    df = pd.read_parquet(final_path)
    
    log_memory_usage("Before feature extraction")
    
    meta_cols = [c for c in ["__user_id", "__item_id", "label", "group_id"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    leaky_patterns = ["item_last_timestamp", "item_first_timestamp",
                      "user_last_timestamp", "user_first_timestamp"]
    feature_cols = [c for c in feature_cols
                    if not any(p in c for p in leaky_patterns)]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    meta_df = df[meta_cols].copy()
    
    del df
    gc.collect()
    
    log_memory_usage("After feature extraction")

    pos_rate = y.mean()
    print(f"\n  📊 {split_name.upper()} dataset stats:")
    print(f"     Total rows:      {len(X):,}")
    print(f"     Users:           {len(final_groups):,}")
    print(f"     Features:        {X.shape[1]}")
    print(f"     Positive rate:   {pos_rate:.4f}")
    print(f"     Hit rate:        {y.sum() / len(final_groups):.3f}")

    return X, y, final_groups, meta_df, feature_cols


# ─────────────────────────────────────────────────────────────
# SAVE/LOAD DATASET
# ─────────────────────────────────────────────────────────────

def save_ranking_dataset(X, y, groups, meta_df, feature_names, split_name):
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    df["group_id"] = np.repeat(np.arange(len(groups)), NUM_CANDIDATES)
    df["__user_id"] = meta_df["__user_id"].values
    df["__item_id"] = meta_df["__item_id"].values

    out_path = FEATURES_DIR / f"ranking_{split_name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  ✅ Saved {split_name} dataset: {out_path} ({len(df):,} rows)")
    
    del df
    gc.collect()

    with open(FEATURES_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    with open(FEATURES_DIR / f"groups_{split_name}.pkl", "wb") as f:
        pickle.dump(groups, f)

    return out_path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  DS19 — Week 5: Ranking Dataset Builder (RTX 3050 4GB)")
    print("  Smart Build: Checks existing files, rebuilds only if needed")
    print("=" * 65)

    log_memory_usage("Init")

    # ── FIX: Initialize ALL variables correctly (4 values for val) ──
    X_train, y_train, groups_train, meta_train = None, None, None, None
    X_val, y_val, groups_val, meta_val = None, None, None, None  # ✅ FIXED: 4 values now
    feature_names = None

    # ── CHECK IF DATASETS ALREADY EXIST AND ARE VALID ──
    print("\n🔍 Checking for existing valid datasets...")
    
    train_valid = check_dataset_valid("train")
    val_valid = check_dataset_valid("val")
    
    if train_valid and val_valid:
        print("\n✅ Both datasets are valid! Loading existing files...")
        
        X_train, y_train, groups_train, meta_train, feature_names = load_existing_dataset("train")
        X_val, y_val, groups_val, meta_val, _ = load_existing_dataset("val")
        
        print("\n📊 TRAIN dataset stats:")
        print(f"   Total rows:      {len(X_train):,}")
        print(f"   Features:        {X_train.shape[1]}")
        print(f"   Positive rate:   {y_train.mean():.4f}")
        
        print("\n📊 VAL dataset stats:")
        print(f"   Total rows:      {len(X_val):,}")
        print(f"   Features:        {X_val.shape[1]}")
        print(f"   Positive rate:   {y_val.mean():.4f}")
        
        log_memory_usage("Complete")
        print("\n✅ Dataset loading complete! Ready for LightGBM training.")
        return  # Don't return values when running as script
    
    # ── NEED TO REBUILD ──
    print("\n⚠️  Some datasets are missing or invalid. Rebuilding...")
    
    print("\n📂 Loading metadata and sequences...")
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    train_seq_path = SPLITS_DIR / "train_sequences.pkl"
    if not train_seq_path.exists():
        train_seq_path = SPLITS_DIR / "train_seqs.pkl"

    val_seq_path = SPLITS_DIR / "val_sequences.pkl"
    if not val_seq_path.exists():
        val_seq_path = SPLITS_DIR / "val_seqs.pkl"

    with open(train_seq_path, "rb") as f:
        train_sequences = pickle.load(f)
    with open(val_seq_path, "rb") as f:
        val_sequences = pickle.load(f)

    print(f"  Train sequences: {len(train_sequences):,}")
    print(f"  Val sequences:   {len(val_sequences):,}")

    log_memory_usage("After loading sequences")

    print("\n📂 Loading precomputed features...")
    user_features, item_features, genre_cols = load_features()
    if user_features is None:
        print("  ❌ Features not found! Run: python data/features/feature_engineering.py")
        return

    print(f"  User features: {user_features.shape}")
    print(f"  Item features: {item_features.shape}")

    log_memory_usage("After loading features")

    two_tower_model = load_two_tower_model(meta)
    faiss_index, _, faiss_item_ids = load_faiss_index()

    log_memory_usage("After loading models")

    # ── BUILD TRAIN ──
    if not train_valid:
        X_train, y_train, groups_train, meta_train, feature_names = build_ranking_dataset(
            train_sequences, "train",
            two_tower_model, faiss_index,
            user_features, item_features, genre_cols,
            train_sequences, meta,
            faiss_item_id_map=faiss_item_ids,
            max_users=MAX_TRAIN_USERS
        )
        
        if X_train is not None:
            save_ranking_dataset(X_train, y_train, groups_train,
                                 meta_train, feature_names, "train")
            
            del X_train, y_train, groups_train, meta_train
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            log_memory_usage("After train dataset")
    else:
        print("\n✅ Skipping train dataset (already valid)")
        X_train, y_train, groups_train, meta_train, feature_names = load_existing_dataset("train")

    # ── BUILD VAL ──
    if not val_valid:
        X_val, y_val, groups_val, meta_val, _ = build_ranking_dataset(
            val_sequences, "val",
            two_tower_model, faiss_index,
            user_features, item_features, genre_cols,
            train_sequences, meta,
            faiss_item_id_map=faiss_item_ids,
            max_users=MAX_VAL_USERS
        )
        
        if X_val is not None:
            save_ranking_dataset(X_val, y_val, groups_val,
                                 meta_val, feature_names, "val")
    else:
        print("\n✅ Skipping val dataset (already valid)")
        X_val, y_val, groups_val, meta_val, _ = load_existing_dataset("val")

    log_memory_usage("Complete")
    print("\n✅ Dataset construction complete!")
    print(f"   Feature names: {feature_names[:5]}... ({len(feature_names)} total)")

# ─────────────────────────────────────────────────────────────
# LOAD DATASET FROM PARQUET  ← ADD THIS!
# ─────────────────────────────────────────────────────────────

def load_ranking_dataset(split_name):
    """Load a previously saved ranking dataset from parquet."""
    df_path = FEATURES_DIR / f"ranking_{split_name}.parquet"
    groups_path = FEATURES_DIR / f"groups_{split_name}.pkl"
    feat_path = FEATURES_DIR / "feature_names.json"

    if not df_path.exists():
        return None, None, None, None, None

    df = pd.read_parquet(df_path)
    
    with open(groups_path, "rb") as f:
        groups = pickle.load(f)
    
    with open(feat_path, "r") as f:
        feature_names = json.load(f)

    y = df["label"].values.astype(np.float32)
    meta_df = df[["__user_id", "__item_id"]].copy()
    X = df[feature_names].values.astype(np.float32)

    return X, y, groups, meta_df, feature_names
if __name__ == "__main__":
    main()