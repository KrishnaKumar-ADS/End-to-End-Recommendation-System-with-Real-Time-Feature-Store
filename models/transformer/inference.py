import sys
import json
import pickle
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.transformer.model import SASRecModel

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

MODELS_DIR    = Path("models/saved")
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR    = Path("data/splits")
MAX_SEQ_LEN   = 50

# ─────────────────────────────────────────────────────────────
# SASREC RECOMMENDER CLASS
# ─────────────────────────────────────────────────────────────

class SASRecRecommender:
    """
    High-level inference wrapper for SASRec.

    Provides:
      - recommend(user_id, top_k) → list of (item_id, score, title)
      - get_user_history(user_id) → list of (item_id, title)
    """

    def __init__(
        self,
        model_path: Path = MODELS_DIR / "sasrec_best.pt",
        device:     str  = "auto",
    ):
        # ── Device ────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Load metadata ─────────────────────────────────────
        self.meta     = json.load(open(PROCESSED_DIR / "dataset_meta.json"))
        self.n_items  = self.meta.get("n_items_with_pad", self.meta["n_items"] + 1)
        self.idx2item = json.load(open(PROCESSED_DIR / "idx2item.json"))
        self.item2idx = json.load(open(PROCESSED_DIR / "item2idx.json"))

        # ── Load movie titles ──────────────────────────────────
        self.idx2title = {}
        try:
            import pandas as pd
            movies = pd.read_csv("data/raw/movies.csv")
            # Build: item_id (original) → title
            raw2title = dict(zip(movies['movieId'].astype(str), movies['title']))
            # Build: idx → title
            for idx_str, raw_id in self.idx2item.items():
                title = raw2title.get(str(raw_id), f"Item {raw_id}")
                self.idx2title[int(idx_str)] = title
        except Exception:
            # Fallback: just use item IDs
            self.idx2title = {int(k): f"Item_{v}" for k, v in self.idx2item.items()}

        # ── Load sequences ────────────────────────────────────
        self.train_seqs = pickle.load(open(SPLITS_DIR / "train_seqs.pkl", "rb"))
        self.val_seqs   = pickle.load(open(SPLITS_DIR / "val_seqs.pkl",   "rb"))
        self.test_seqs  = pickle.load(open(SPLITS_DIR / "test_seqs.pkl",  "rb"))

        # ── Load model ────────────────────────────────────────
        ckpt = torch.load(model_path, map_location=self.device)
        hp   = ckpt["hyperparams"]
        self.model = SASRecModel(
            n_items    = hp["n_items"],
            hidden_dim = hp["hidden_dim"],
            max_seq_len= hp["max_seq_len"],
            num_blocks = hp["num_blocks"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        print(f"  [SASRecRecommender] Model loaded from {model_path.name}")
        print(f"  [SASRecRecommender] Device: {self.device}")
        print(f"  [SASRecRecommender] Parameters: {self.model.count_parameters():,}")

    def _get_sequence(self, user_id: int) -> List[int]:
        """Return the training sequence for a user."""
        # Use test_seqs (includes val item) for most complete history
        if user_id in self.test_seqs:
            return self.test_seqs[user_id]
        elif user_id in self.val_seqs:
            return self.val_seqs[user_id]
        elif user_id in self.train_seqs:
            return self.train_seqs[user_id]
        else:
            return []

    def _pad_sequence(self, seq: List[int]) -> torch.Tensor:
        """Left-pad sequence to MAX_SEQ_LEN and return as tensor [1, L]."""
        pad_len = MAX_SEQ_LEN - len(seq)
        padded  = [0] * pad_len + seq
        padded  = padded[-MAX_SEQ_LEN:]
        return torch.tensor([padded], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def recommend(
        self,
        user_id:      int,
        top_k:        int  = 10,
        exclude_seen: bool = True,
    ) -> List[Dict]:
        """
        Generate top-K recommendations for a user.

        Returns list of dicts: [{rank, item_idx, title, score}]
        """
        seq = self._get_sequence(user_id)
        if len(seq) == 0:
            print(f"  ⚠️  User {user_id} not found in splits.")
            return []

        # Pad and run forward pass
        seq_t = self._pad_sequence(seq)

        # Score all items
        all_items = torch.arange(self.n_items, device=self.device)
        scores    = self.model.predict_scores(seq_t, all_items).squeeze(0)  # [n_items]

        # Guard against unstable checkpoints producing NaN/Inf scores.
        scores = torch.nan_to_num(scores, nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf"))

        # Exclude PAD token
        scores[0] = float('-inf')

        # Exclude seen items
        if exclude_seen:
            seen = torch.tensor(list(set(seq)), dtype=torch.long, device=self.device)
            scores[seen] = float('-inf')

        # Get top-K
        topk_scores, topk_items = scores.topk(top_k)

        results = []
        for rank, (item_idx, score) in enumerate(
            zip(topk_items.tolist(), topk_scores.tolist()), start=1
        ):
            title = self.idx2title.get(item_idx, f"Item_{item_idx}")
            results.append({
                "rank"    : rank,
                "item_idx": item_idx,
                "title"   : title,
                "score"   : round(score, 4),
            })

        return results

    def get_user_history(self, user_id: int, last_n: int = 10) -> List[Dict]:
        """Return the last N items the user interacted with."""
        seq = self._get_sequence(user_id)
        recent = seq[-last_n:]
        return [
            {
                "position": i + 1,
                "item_idx": item_idx,
                "title"   : self.idx2title.get(item_idx, f"Item_{item_idx}"),
            }
            for i, item_idx in enumerate(recent)
        ]


# ─────────────────────────────────────────────────────────────
# MF COMPARISON (optional)
# ─────────────────────────────────────────────────────────────

def recommend_with_mf(user_id: int, top_k: int = 10) -> List[Dict]:
    """
    Get MF recommendations for comparison.
    Requires models/saved/mf_best.pt and models/matrix_factorization/inference.py
    """
    try:
        from models.matrix_factorization.inference import MFRecommender
        mf = MFRecommender()
        return mf.recommend(user_id, top_k=top_k)
    except Exception as e:
        print(f"  ⚠️  MF comparison unavailable: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SASRec Inference")
    parser.add_argument("--user_id",      type=int, default=1,    help="User ID")
    parser.add_argument("--top_k",        type=int, default=10,   help="Top-K recommendations")
    parser.add_argument("--show_history", action="store_true",    help="Show user's watch history")
    parser.add_argument("--compare_mf",   action="store_true",    help="Compare vs MF (Week 2)")
    parser.add_argument("--model_path",   type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    print("=" * 64)
    print("  DS19 — Week 3 | SASRec Inference")
    print("=" * 64)

    # ── Load model ────────────────────────────────────────────
    model_path = Path(args.model_path) if args.model_path else MODELS_DIR / "sasrec_best.pt"
    rec        = SASRecRecommender(model_path=model_path)

    # ── Show history ──────────────────────────────────────────
    if args.show_history:
        print(f"\n  User {args.user_id} — Recent Watch History (last 10):")
        print(f"  {'─'*60}")
        history = rec.get_user_history(args.user_id, last_n=10)
        if not history:
            print("  (no history found)")
        for h in history:
            print(f"  [{h['position']:2d}] idx={h['item_idx']:6d}  {h['title']}")

    # ── SASRec recommendations ────────────────────────────────
    print(f"\n  SASRec — Top-{args.top_k} Recommendations for User {args.user_id}:")
    print(f"  {'─'*60}")
    recs = rec.recommend(args.user_id, top_k=args.top_k)
    if not recs:
        print("  (no recommendations)")
    for r in recs:
        print(f"  [{r['rank']:2d}] score={r['score']:6.4f}  idx={r['item_idx']:6d}  {r['title']}")

    # ── MF comparison ─────────────────────────────────────────
    if args.compare_mf:
        print(f"\n  MF (Week 2) — Top-{args.top_k} Recommendations for User {args.user_id}:")
        print(f"  {'─'*60}")
        mf_recs = recommend_with_mf(args.user_id, top_k=args.top_k)
        if not mf_recs:
            print("  (MF model not available or user not found)")
        else:
            for r in mf_recs:
                print(f"  [{r['rank']:2d}] score={r.get('score', 0):6.4f}  "
                      f"idx={r.get('item_idx', 0):6d}  {r.get('title', '')}")

            # Overlap analysis
            if recs and mf_recs:
                sr_set = {r["item_idx"] for r in recs}
                mf_set = {r["item_idx"] for r in mf_recs}
                overlap = len(sr_set & mf_set)
                print(f"\n  Overlap between SASRec and MF top-{args.top_k}: {overlap}/{args.top_k}")
                print(f"  (low overlap = SASRec is using sequence context, not just popularity)")

    print()


if __name__ == "__main__":
    main()