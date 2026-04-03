import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────
# CONFIG (must match Week 1 pipeline values)
# ─────────────────────────────────────────────────────────────

MAX_SEQ_LEN  = 50
PAD_TOKEN    = 0
NEG_SAMPLES  = 128   # negatives per positive during training

SPLITS_DIR    = Path("data/splits")
SEQUENCES_DIR = Path("data/sequences")
PROCESSED_DIR = Path("data/processed")

# ─────────────────────────────────────────────────────────────
# TRAINING DATASET
# ─────────────────────────────────────────────────────────────

class SASRecTrainDataset(Dataset):
    """
    Training dataset for SASRec.

    For each user with training sequence [s_1, ..., s_n]:
      Input  (x): [PAD..., s_1, ..., s_{n-1}]   — left-padded to MAX_SEQ_LEN
      Pos    (y): [PAD..., s_2, ..., s_n]        — shifted by 1 (next item)
      Neg    (n): [0...,   neg_samples...]        — K negatives per position

    The idea: at position t, given input[t], predict pos[t] (= input[t+1]).
    PAD positions in pos → label = 0 → excluded from loss.
    """

    def __init__(
        self,
        train_seqs: Dict[int, List[int]],
        n_items: int,                        # total items INCLUDING pad (items start at 1)
        neg_samples: int = NEG_SAMPLES,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.n_items     = n_items
        self.neg_samples = neg_samples
        self.max_seq_len = max_seq_len

        # Convert dict → list of (user_id, sequence)
        # Each sequence is already sorted by time (from Week 1)
        self.users    : List[int]        = []
        self.sequences: List[List[int]]  = []
        self.user_sets: Dict[int, set]   = {}   # for fast negative sampling

        for user_id, seq in train_seqs.items():
            if len(seq) < 2:
                continue   # need at least 1 input + 1 target
            self.users.append(user_id)
            self.sequences.append(seq)
            self.user_sets[user_id] = set(seq)

        print(f"  [SASRecTrainDataset] {len(self.users):,} users loaded")
        print(f"  [SASRecTrainDataset] n_items={n_items:,}, neg_samples={neg_samples}")

    def __len__(self) -> int:
        return len(self.users)

    def _sample_negatives(self, user_idx: int, n: int) -> List[int]:
        """
        Sample n items NOT in this user's history.
        Uses rejection sampling — fast for sparse datasets (ML-25M is ~0.03% dense).
        """
        user_id   = self.users[user_idx]
        positives = self.user_sets[user_id]
        negs      = []
        while len(negs) < n:
            cand = random.randint(1, self.n_items - 1)   # items start at 1
            if cand not in positives:
                negs.append(cand)
        return negs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single training example.

        seq : [MAX_SEQ_LEN]   — input sequence (left-padded)
        pos : [MAX_SEQ_LEN]   — target items (pos[t] = seq[t+1], 0 for PAD)
        neg : [MAX_SEQ_LEN, neg_samples] — negative items per position
        """
        seq = self.sequences[idx]

        # ── Build input (x) and target (y) sequences ──────────
        # Input:  [..., s_1, s_2, ..., s_{n-1}]
        # Target: [...,  0,  s_2, ..., s_n    ]   (0 = ignore PAD)
        #
        # Shift: if seq = [3, 7, 12, 45]
        #   input  = [3, 7, 12]   (all but last)
        #   target = [7, 12, 45]  (all but first)

        n     = len(seq)
        inp   = seq[:-1]   # length n-1
        tgt   = seq[1:]    # length n-1 (positive next items)

        # ── Left-pad to MAX_SEQ_LEN ────────────────────────────
        pad_len = self.max_seq_len - len(inp)

        # seq_pad: left-padded input
        seq_pad = [PAD_TOKEN] * pad_len + inp
        seq_pad = seq_pad[-self.max_seq_len:]   # truncate from left if needed

        # pos_pad: left-padded target (0 for pad positions)
        pos_pad = [PAD_TOKEN] * pad_len + tgt
        pos_pad = pos_pad[-self.max_seq_len:]

        # ── Sample negatives for each position ────────────────
        # neg_pad: [MAX_SEQ_LEN, neg_samples]
        # Only sample negatives for NON-PAD positions in target
        neg_pad = np.zeros((self.max_seq_len, self.neg_samples), dtype=np.int32)
        negs    = self._sample_negatives(idx, self.neg_samples * self.max_seq_len)
        nptr    = 0
        for t in range(self.max_seq_len):
            if pos_pad[t] != 0:   # non-pad position
                neg_pad[t] = negs[nptr : nptr + self.neg_samples]
                nptr += self.neg_samples

        return {
            "seq" : torch.tensor(seq_pad, dtype=torch.long),           # [L]
            "pos" : torch.tensor(pos_pad, dtype=torch.long),           # [L]
            "neg" : torch.tensor(neg_pad, dtype=torch.long),           # [L, K]
        }


# ─────────────────────────────────────────────────────────────
# EVALUATION DATASET
# ─────────────────────────────────────────────────────────────

class SASRecEvalDataset(Dataset):
    """
    Evaluation dataset for SASRec.

    For each user:
      seq   : full training sequence, left-padded to MAX_SEQ_LEN
      label : the single held-out item (val or test)
      mask  : set of all items to exclude from ranking (training history)

    We do NOT include the label in seq — that would be data leakage.
    """

    def __init__(
        self,
        eval_seqs:   Dict[int, List[int]],   # user_id → training sequence
        eval_labels: Dict[int, int],          # user_id → held-out item
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.max_seq_len = max_seq_len

        self.user_ids  : List[int]       = []
        self.sequences : List[List[int]] = []
        self.labels    : List[int]       = []
        self.masks     : List[set]       = []  # items to exclude from ranking

        # Only include users that exist in both eval_seqs and eval_labels
        for user_id in eval_labels:
            if user_id not in eval_seqs:
                continue
            seq   = eval_seqs[user_id]
            label = eval_labels[user_id]
            if label == 0 or len(seq) == 0:
                continue

            self.user_ids.append(user_id)
            self.sequences.append(seq)
            self.labels.append(label)
            self.masks.append(set(seq))   # exclude training items from ranking

        print(f"  [SASRecEvalDataset] {len(self.user_ids):,} eval users loaded")

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single evaluation example.
        seq   : [MAX_SEQ_LEN] — padded input sequence
        label : int           — the correct next item
        user  : int           — user_id (for result tracking)
        """
        seq = self.sequences[idx]

        # Left-pad
        pad_len = self.max_seq_len - len(seq)
        seq_pad = [PAD_TOKEN] * pad_len + seq
        seq_pad = seq_pad[-self.max_seq_len:]

        return {
            "seq"  : torch.tensor(seq_pad, dtype=torch.long),
            "label": torch.tensor(self.labels[idx],  dtype=torch.long),
            "user" : torch.tensor(self.user_ids[idx], dtype=torch.long),
        }

    def get_mask(self, idx: int) -> set:
        """Return the set of training items to exclude from ranking for user idx."""
        return self.masks[idx]


# ─────────────────────────────────────────────────────────────
# DATASET FACTORY
# ─────────────────────────────────────────────────────────────

def load_datasets(
    neg_samples: int = NEG_SAMPLES,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Tuple[SASRecTrainDataset, SASRecEvalDataset, SASRecEvalDataset, dict]:
    """
    Loads all split files and returns train/val/test datasets + metadata.
    Call this from train.py and evaluate.py.
    """
    import json

    meta       = json.load(open(PROCESSED_DIR / "dataset_meta.json"))
    n_items    = meta.get("n_items_with_pad", meta["n_items"] + 1)

    train_seqs  = pickle.load(open(SPLITS_DIR / "train_seqs.pkl",  "rb"))
    val_seqs    = pickle.load(open(SPLITS_DIR / "val_seqs.pkl",    "rb"))
    test_seqs   = pickle.load(open(SPLITS_DIR / "test_seqs.pkl",   "rb"))
    val_labels  = pickle.load(open(SPLITS_DIR / "val_labels.pkl",  "rb"))
    test_labels = pickle.load(open(SPLITS_DIR / "test_labels.pkl", "rb"))

    train_ds = SASRecTrainDataset(train_seqs, n_items, neg_samples, max_seq_len)
    val_ds   = SASRecEvalDataset(val_seqs,   val_labels,  max_seq_len)
    test_ds  = SASRecEvalDataset(test_seqs,  test_labels, max_seq_len)

    return train_ds, val_ds, test_ds, meta


# ─────────────────────────────────────────────────────────────
# SMOKE TEST (run this file directly to verify)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  SASRec Dataset — Smoke Test")
    print("=" * 60)

    meta    = json.load(open(PROCESSED_DIR / "dataset_meta.json"))
    n_items = meta.get("n_items_with_pad", meta["n_items"] + 1)

    train_seqs  = pickle.load(open(SPLITS_DIR / "train_seqs.pkl",  "rb"))
    val_seqs    = pickle.load(open(SPLITS_DIR / "val_seqs.pkl",    "rb"))
    val_labels  = pickle.load(open(SPLITS_DIR / "val_labels.pkl",  "rb"))

    # ── Training dataset ──────────────────────────────────────
    print("\n  Training dataset:")
    train_ds = SASRecTrainDataset(train_seqs, n_items, neg_samples=128)
    loader   = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    batch    = next(iter(loader))

    print(f"  seq  shape : {batch['seq'].shape}   dtype: {batch['seq'].dtype}")
    print(f"  pos  shape : {batch['pos'].shape}   dtype: {batch['pos'].dtype}")
    print(f"  neg  shape : {batch['neg'].shape}   dtype: {batch['neg'].dtype}")
    print(f"\n  First sequence (seq)   : {batch['seq'][0].tolist()}")
    print(f"  First target  (pos)   : {batch['pos'][0].tolist()}")
    print(f"  First negatives (neg[0][:5]): {batch['neg'][0, 0, :5].tolist()}")

    # Verify: pos is seq shifted right by 1
    s = batch['seq'][0].tolist()
    p = batch['pos'][0].tolist()
    # Find first non-pad in seq
    first_item_idx = next(i for i, v in enumerate(s) if v != 0)
    print(f"\n  Shift verification:")
    print(f"    seq[{first_item_idx}] = {s[first_item_idx]}   → pos[{first_item_idx}] = {p[first_item_idx]}")
    if first_item_idx + 1 < len(s):
        print(f"    seq[{first_item_idx+1}] = {s[first_item_idx+1]}   → matches pos[{first_item_idx}] = {p[first_item_idx]}? "
              f"{'✅' if s[first_item_idx+1] == p[first_item_idx] else '❌'}")

    # ── Eval dataset ──────────────────────────────────────────
    print("\n  Evaluation dataset:")
    val_ds   = SASRecEvalDataset(val_seqs, val_labels)
    val_batch= next(iter(DataLoader(val_ds, batch_size=4, num_workers=0)))

    print(f"  seq   shape : {val_batch['seq'].shape}")
    print(f"  label shape : {val_batch['label'].shape}")
    print(f"  labels      : {val_batch['label'].tolist()}")
    print(f"  users       : {val_batch['user'].tolist()}")
    print(f"\n  Mask for user 0 (first 5 items): "
          f"{list(val_ds.get_mask(0))[:5]}")

    print("\n  ✅ Dataset smoke test PASSED")