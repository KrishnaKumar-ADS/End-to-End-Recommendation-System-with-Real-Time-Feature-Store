import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.ranking.model import LightGBMRanker
from models.ranking.dataset import (
    load_two_tower_model,
    load_faiss_index,
    encode_user_sequence,
    retrieve_candidates,
    build_feature_rows,
)
from features.feature_engineering import load_features


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

META_PATH = "models/saved/dataset_meta.json"
SPLITS_DIR = "data/splits"
LGBM_MODEL_PATH = "models/saved/lgbm_ranker.pkl"
TWO_TOWER_PATH = "models/saved/two_tower_best.pt"
FAISS_INDEX_PATH = "retrieval/faiss_item.index"
ITEM_EMBEDDINGS_PATH = "retrieval/item_embeddings.npy"

NUM_CANDIDATES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# RANKING INFERENCE CLASS
# ─────────────────────────────────────────────────────────────

class RankingInference:
    def __init__(self,
                 lgbm_path=LGBM_MODEL_PATH,
                 two_tower_path=TWO_TOWER_PATH,
                 faiss_index_path=FAISS_INDEX_PATH,
                 num_candidates=NUM_CANDIDATES):
        
        self.num_candidates = num_candidates
        self._loaded = False

        print("🔄 Initializing RankingInference pipeline...")

        # ── Load metadata ─────────────────────────────────────────
        print("  Loading metadata...")
        with open(META_PATH, "r") as f:
            self.meta = json.load(f)
        self.id2item = {int(k): v for k, v in self.meta["id2item"].items()}
        self.item2id = {str(k): v for k, v in self.meta["item2id"].items()}
        self.id2user = {int(k): v for k, v in self.meta["id2user"].items()}
        self.user2id = {str(k): v for k, v in self.meta["user2id"].items()}

        # ── Load user sequences ──────────────────────────────────
        print("  Loading user sequences...")
        with open(os.path.join(SPLITS_DIR, "train_sequences.pkl"), "rb") as f:
            self.train_sequences = pickle.load(f)

        # ── Load item metadata ────────────────────────────────────
        print("  Loading movie metadata...")
        try:
            self.movies_df = pd.read_csv("data/processed/movies.csv")
            self.movies_df["movieId_str"] = self.movies_df["movieId"].astype(str)
            self.item_title_map = dict(zip(
                self.movies_df["movieId_str"].map(self.item2id).dropna().astype(int),
                self.movies_df["title"]
            ))
            self.item_genre_map = dict(zip(
                self.movies_df["movieId_str"].map(self.item2id).dropna().astype(int),
                self.movies_df["genres"]
            ))
        except Exception as e:
            print(f"  ⚠️  Could not load movies.csv: {e}")
            self.item_title_map = {}
            self.item_genre_map = {}

        # ── Load Two-Tower model ─────────────────────────────────
        print(f"  Loading Two-Tower model (device={DEVICE})...")
        self.two_tower = load_two_tower_model(self.meta)
        self.two_tower.eval()

        # ── Load FAISS index ─────────────────────────────────────
        print("  Loading FAISS index...")
        self.faiss_index, self.item_embeddings = load_faiss_index()
        
        # ── FAISS Debug Info ─────────────────────────────────────
        if self.faiss_index is not None:
            print(f"  📊 FAISS Index Info:")
            print(f"     Total items: {self.faiss_index.ntotal}")
            print(f"     Dimension: {self.faiss_index.d}")
            if self.item_embeddings is not None:
                print(f"     Embeddings shape: {self.item_embeddings.shape}")
                print(f"     Embeddings norm (mean): {np.linalg.norm(self.item_embeddings, axis=1).mean():.4f}")

        # ── Load LightGBM ranker ─────────────────────────────────
        print("  Loading LightGBM ranker...")
        self.ranker = LightGBMRanker.load(lgbm_path)

        # ── Load precomputed features ────────────────────────────
        print("  Loading precomputed user/item features...")
        self.user_features, self.item_features, self.genre_cols = load_features()
        if self.user_features is None:
            raise RuntimeError("Feature cache not found! Run: python features/feature_engineering.py")

        # ── Load feature names from model (source of truth) ──────
        self.feature_names = getattr(self.ranker, 'feature_names', None)
        if self.feature_names is None:
            feat_path = os.path.join("data/features", "feature_names.json")
            if os.path.exists(feat_path):
                with open(feat_path, "r") as f:
                    self.feature_names = json.load(f)
            else:
                raise RuntimeError("No feature names found in model or feature_names.json!")

        print(f"  ✅ Feature names loaded: {len(self.feature_names)} features")
        self._loaded = True
        print("✅ RankingInference pipeline ready!\n")

    def recommend(self, user_id, top_k=10, exclude_history=True, verbose=False):
        if not self._loaded:
            raise RuntimeError("Pipeline not initialized!")

        timings = {}
        total_start = time.time()
        lgbm_scores = None

        # ── Stage 1: Lookup user history ─────────────────────────
        t0 = time.time()
        if user_id in self.train_sequences:
            user_history = self.train_sequences[user_id]
        else:
            if verbose:
                print(f"  ⚠️  Cold-start user {user_id} — no history available")
            user_history = []
        timings["lookup_ms"] = round((time.time() - t0) * 1000, 2)

        if verbose:
            print(f"  User history: {len(user_history)} items")
            if user_history:
                print(f"  Last 5 items: {user_history[-5:]}")

        # ── Stage 2: Encode user sequence ────────────────────────
        t0 = time.time()
        user_emb = encode_user_sequence(self.two_tower, user_history, max_len=50)
        timings["encode_ms"] = round((time.time() - t0) * 1000, 2)

        # ── DEBUG: Check user embedding ──────────────────────────
        if verbose:
            emb_norm = np.linalg.norm(user_emb)
            print(f"  🔍 User embedding: shape={user_emb.shape}, norm={emb_norm:.4f}")
            print(f"     Min={user_emb.min():.4f}, Max={user_emb.max():.4f}, Mean={user_emb.mean():.4f}")
            if emb_norm < 1e-6:
                print(f"  ⚠️  WARNING: User embedding is near-zero! This will cause FAISS to return nothing.")

        # ── Stage 3: FAISS retrieval ─────────────────────────────
        t0 = time.time()
        
        # 🔧 FIXED: Direct FAISS search with debugging
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print(f"  ⚠️  FAISS index is empty or not loaded!")
            faiss_scores = np.zeros(self.num_candidates, dtype=np.float32)
            faiss_item_ids = np.zeros(self.num_candidates, dtype=np.int64)
        else:
            # Normalize user embedding (FAISS expects normalized vectors for inner product)
            user_emb_normalized = user_emb.copy()
            emb_norm = np.linalg.norm(user_emb_normalized)
            if emb_norm > 1e-6:
                user_emb_normalized = user_emb_normalized / emb_norm
            
            # Search FAISS
            raw_scores, raw_item_ids = self.faiss_index.search(
                user_emb_normalized.reshape(1, -1).astype(np.float32), 
                self.num_candidates
            )
            faiss_scores = raw_scores[0]
            faiss_item_ids = raw_item_ids[0]
            
            if verbose:
                print(f"  🔍 FAISS raw results:")
                print(f"     Top 5 scores: {faiss_scores[:5]}")
                print(f"     Top 5 item IDs: {faiss_item_ids[:5]}")
                print(f"     Zero IDs count: {(faiss_item_ids == 0).sum()}")
        
        # Filter out padding tokens (item_id = 0)
        valid_mask = faiss_item_ids > 0
        num_valid = valid_mask.sum()
        
        if verbose:
            print(f"  FAISS retrieved: {num_valid}/{self.num_candidates} valid candidates")
        
        if num_valid > 0:
            faiss_scores = faiss_scores[valid_mask]
            faiss_item_ids = faiss_item_ids[valid_mask]
        else:
            # No valid candidates - create empty arrays
            faiss_scores = np.array([], dtype=np.float32)
            faiss_item_ids = np.array([], dtype=np.int64)
        
        timings["faiss_ms"] = round((time.time() - t0) * 1000, 2)

        # ── EARLY EXIT: No candidates ────────────────────────────
        if num_valid == 0:
            print(f"  ⚠️  WARNING: No valid FAISS candidates for user {user_id}!")
            print(f"     Possible causes:")
            print(f"     1. User embedding is zero/invalid")
            print(f"     2. FAISS index has no items with ID > 0")
            print(f"     3. Item ID mismatch between FAISS and metadata")
            
            timings["feature_ms"] = 0.0
            timings["lgbm_ms"] = 0.0
            timings["rank_ms"] = 0.0
            timings["total_ms"] = round((time.time() - total_start) * 1000, 2)
            
            return {
                "user_id": user_id,
                "recommendations": [],
                "faiss_candidates": 0,
                "latency_ms": timings,
            }

        # Pad to NUM_CANDIDATES if necessary
        if num_valid < self.num_candidates:
            pad = self.num_candidates - num_valid
            faiss_item_ids = np.concatenate([faiss_item_ids, np.zeros(pad, dtype=faiss_item_ids.dtype)])
            faiss_scores = np.concatenate([faiss_scores, np.zeros(pad)])
        faiss_item_ids = faiss_item_ids[:self.num_candidates]
        faiss_scores = faiss_scores[:self.num_candidates]

        # ── Stage 4: Feature engineering ─────────────────────────
        t0 = time.time()
        rows = build_feature_rows(
            user_id, faiss_item_ids, faiss_scores,
            self.user_features, self.item_features, self.genre_cols,
            self.train_sequences, self.meta
        )

        leaky_patterns = ["item_last_timestamp", "item_first_timestamp",
                          "user_last_timestamp", "user_first_timestamp"]
        meta_cols = {"__user_id", "__item_id", "label"}

        feat_rows = []
        for row in rows:
            feat_row = {
                k: v for k, v in row.items()
                if k not in meta_cols
                and not any(p in k for p in leaky_patterns)
            }
            feat_rows.append(feat_row)

        feat_df = pd.DataFrame(feat_rows)

        # ── ROBUST FEATURE ALIGNMENT ─────────────────────────────
        expected_features = self.feature_names
        for col in expected_features:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        
        X = feat_df[expected_features].values.astype(np.float32)
        
        if X.shape[1] != len(expected_features):
            raise RuntimeError(
                f"Feature alignment failed: expected {len(expected_features)} features, "
                f"got {X.shape[1]}"
            )

        timings["feature_ms"] = round((time.time() - t0) * 1000, 2)

        # ── Stage 5: LightGBM scoring ────────────────────────────
        t0 = time.time()
        try:
            lgbm_scores = self.ranker.predict(X)
        except Exception as e:
            raise RuntimeError(f"LightGBM prediction failed: {e}")
        timings["lgbm_ms"] = round((time.time() - t0) * 1000, 2)

        if verbose:
            print(f"  LGBM scores: min={lgbm_scores.min():.4f}, max={lgbm_scores.max():.4f}")

        # ── Stage 6: Rank and filter ─────────────────────────────
        t0 = time.time()
        history_set = set(user_history) if exclude_history else set()

        sorted_indices = np.argsort(lgbm_scores)[::-1]
        recommendations = []

        for idx in sorted_indices:
            item_id = int(faiss_item_ids[idx])
            if item_id == 0:
                continue
            if item_id in history_set:
                continue

            title = self.item_title_map.get(item_id, f"Unknown (id={item_id})")
            genres = self.item_genre_map.get(item_id, "Unknown")
            score = float(lgbm_scores[idx])

            recommendations.append({
                "item_id": item_id,
                "title": title,
                "genres": genres,
                "lgbm_score": round(score, 5),
                "faiss_rank": int(np.where(faiss_item_ids == item_id)[0][0]) + 1,
            })

            if len(recommendations) >= top_k:
                break

        timings["rank_ms"] = round((time.time() - t0) * 1000, 2)
        timings["total_ms"] = round((time.time() - total_start) * 1000, 2)

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "faiss_candidates": num_valid,
            "latency_ms": timings,
        }

    def batch_recommend(self, user_ids, top_k=10):
        results = {}
        for uid in user_ids:
            try:
                results[uid] = self.recommend(uid, top_k=top_k)
            except Exception as e:
                results[uid] = {"error": str(e)}
        return results

    def benchmark(self, n_users=200):
        print(f"\n⏱️  Running latency benchmark ({n_users} users)...")

        all_user_ids = list(self.train_sequences.keys())
        if len(all_user_ids) > n_users:
            sample_ids = np.random.choice(all_user_ids, n_users, replace=False)
        else:
            sample_ids = all_user_ids

        latencies = []
        stage_latencies = {"lookup_ms": [], "encode_ms": [], "faiss_ms": [],
                          "feature_ms": [], "lgbm_ms": [], "rank_ms": [], "total_ms": []}

        for uid in sample_ids:
            result = self.recommend(uid, top_k=10)
            latencies.append(result["latency_ms"]["total_ms"])
            for stage, val in result["latency_ms"].items():
                if stage in stage_latencies:
                    stage_latencies[stage].append(val)

        latencies = np.array(latencies)

        print(f"\n  📊 Latency Distribution (total):")
        print(f"     P50:  {np.percentile(latencies, 50):.1f} ms")
        print(f"     P95:  {np.percentile(latencies, 95):.1f} ms")
        print(f"     P99:  {np.percentile(latencies, 99):.1f} ms")
        print(f"     Mean: {latencies.mean():.1f} ms")
        print(f"     Max:  {latencies.max():.1f} ms")

        print(f"\n  📊 Per-Stage Breakdown (mean):")
        for stage, vals in stage_latencies.items():
            mean_ms = np.mean(vals)
            pct = mean_ms / latencies.mean() * 100
            print(f"     {stage:<15s} {mean_ms:>6.1f} ms  ({pct:.1f}%)")

        throughput = 1000 / latencies.mean()
        print(f"\n  📊 Throughput: {throughput:.0f} recommendations/second")

        return {
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "mean_ms": float(latencies.mean()),
            "throughput_rps": float(throughput),
            "n_users": n_users,
        }


# ─────────────────────────────────────────────────────────────
# COMMAND LINE INTERFACE
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DS19 Week 5 — Ranking Inference")
    parser.add_argument("--user_id", type=int, default=None,
                        help="Encoded user ID to get recommendations for")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of recommendations to return (default: 10)")
    parser.add_argument("--show_history", action="store_true",
                        help="Show user's interaction history")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run latency benchmark")
    parser.add_argument("--n_users", type=int, default=200,
                        help="Number of users for benchmark (default: 200)")
    args = parser.parse_args()

    pipeline = RankingInference()

    if args.benchmark:
        pipeline.benchmark(n_users=args.n_users)
        return

    if args.user_id is None:
        sample_users = list(pipeline.train_sequences.keys())
        user_id = int(np.random.choice(sample_users))
        print(f"  No user_id specified — using random user: {user_id}")
    else:
        user_id = args.user_id

    print(f"\n🎬 Generating top-{args.top_k} recommendations for user {user_id}...")
    result = pipeline.recommend(user_id, top_k=args.top_k, verbose=args.show_history)

    print(f"\n{'='*65}")
    print(f"  USER {user_id} — TOP {args.top_k} RECOMMENDATIONS")
    print(f"{'='*65}")

    if not result["recommendations"]:
        print("  ❌ No recommendations generated!")
        print("     User may have no history or all candidates filtered.")
        if result["faiss_candidates"] == 0:
            print("     ⚠️  FAISS returned 0 candidates - check embedding/index!")
    else:
        print(f"  {'Rank':<5} {'LGBM Score':>10} {'FAISS Rank':>10}  Title")
        print(f"  {'-'*5} {'-'*10} {'-'*10}  {'-'*40}")
        for rank, rec in enumerate(result["recommendations"], 1):
            title = rec["title"][:45] if len(rec["title"]) > 45 else rec["title"]
            print(f"  {rank:<5} {rec['lgbm_score']:>10.5f} "
                  f"{'#'+str(rec['faiss_rank']):>10}  {title}")
            print(f"  {'':<5} {'':<10} {'':<10}  {rec['genres']}")
            print()

    print(f"\n⏱️  Latency breakdown:")
    for stage, ms in result["latency_ms"].items():
        print(f"     {stage:<15s}: {ms:.1f} ms")

    print(f"\n  FAISS candidates retrieved: {result['faiss_candidates']}")
    print(f"  Total latency: {result['latency_ms']['total_ms']:.1f} ms")


if __name__ == "__main__":
    main()