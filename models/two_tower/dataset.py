import sys
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SPLITS_DIR    = Path("data/splits")
SEQUENCES_DIR = Path("data/sequences")
PROCESSED_DIR = Path("data/processed")

MAX_SEQ_LEN   = 50
PAD_TOKEN     = 0
SEED          = 42

# ─────────────────────────────────────────────────────────────
# TRAINING DATASET
# ─────────────────────────────────────────────────────────────

class TwoTowerTrainDataset(Dataset):
    """
    Returns (user_seq, target_item) pairs for Two-Tower training.

    Automatically adapts to column naming differences:
    - target_item / label / next_item
    """

    def __init__(self, max_seq_len: int = MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self._load_samples()

    def _load_samples(self):
        samples_path = SEQUENCES_DIR / "training_samples.parquet"

        if not samples_path.exists():
            raise FileNotFoundError(f"Dataset not found: {samples_path}")

        df = pd.read_parquet(samples_path)

        print(f"\n  [DEBUG] Columns found: {df.columns.tolist()}")

        # ── Validate required columns ──────────────────────────
        if 'user_idx' not in df.columns:
            raise ValueError(f"'user_idx' column missing. Found: {df.columns.tolist()}")

        # Detect input sequence column
        if 'input_seq' in df.columns:
            seq_col = 'input_seq'
        elif 'sequence' in df.columns:
            seq_col = 'sequence'
        else:
            raise ValueError(f"No sequence column found. Columns: {df.columns.tolist()}")

        # Detect target column
        if 'target_item' in df.columns:
            target_col = 'target_item'
        elif 'label' in df.columns:
            target_col = 'label'
        elif 'next_item' in df.columns:
            target_col = 'next_item'
        elif 'target' in df.columns:   # ✅ ADD THIS LINE
            target_col = 'target'
        else:
            raise ValueError(f"No target column found. Columns: {df.columns.tolist()}")

        print(f"  [INFO] Using sequence column: {seq_col}")
        print(f"  [INFO] Using target column  : {target_col}")

        # ── Convert sequence column ────────────────────────────
        if df[seq_col].dtype == object:
            first_val = df[seq_col].iloc[0]

            # Case 1: stored as string "1,2,3"
            first_val = df[seq_col].iloc[0]

# Case 1: string → "1,2,3"
        if isinstance(first_val, str):
            df[seq_col] = df[seq_col].apply(lambda x: list(map(int, x.split(','))))

# Case 2: numpy array → convert to list
        elif isinstance(first_val, np.ndarray):
            df[seq_col] = df[seq_col].apply(lambda x: x.tolist())

# Case 3: already list → OK
        elif isinstance(first_val, list):
            pass

        else:
            raise ValueError(f"Unsupported sequence format: {type(first_val)}")

        # ── Assign tensors ─────────────────────────────────────
        self.user_indices = df['user_idx'].values.astype(np.int64)
        self.target_items = df[target_col].values.astype(np.int64)
        self.sequences    = df[seq_col].tolist()

        print(f"  [TwoTowerTrainDataset] Loaded {len(self.sequences):,} samples")
        print(f"  [TwoTowerTrainDataset] Max sequence length: {self.max_seq_len}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq         = self.sequences[idx]
        target_item = int(self.target_items[idx])

        # ── Pad / truncate sequence ────────────────────────────
        if len(seq) < self.max_seq_len:
            seq = [PAD_TOKEN] * (self.max_seq_len - len(seq)) + seq
        elif len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]

        seq_tensor = torch.tensor(seq, dtype=torch.long)
        tgt_tensor = torch.tensor(target_item, dtype=torch.long)

        return seq_tensor, tgt_tensor

# ─────────────────────────────────────────────────────────────
# EVALUATION DATASET (UNCHANGED)
# ─────────────────────────────────────────────────────────────

class TwoTowerEvalDataset(Dataset):
    def __init__(self, split: str = "val", max_seq_len: int = MAX_SEQ_LEN):
        assert split in ("val", "test")
        self.max_seq_len = max_seq_len
        self._load(split)

    def _load(self, split: str):
        seq_path   = SPLITS_DIR / f"{split}_seqs.pkl"
        label_path = SPLITS_DIR / f"{split}_labels.pkl"

        with open(seq_path, "rb") as f:
            seqs = pickle.load(f)
        with open(label_path, "rb") as f:
            labels = pickle.load(f)

        user_ids = sorted(set(seqs.keys()) & set(labels.keys()))

        self.user_ids    = user_ids
        self.seqs_dict   = seqs
        self.labels_dict = labels

        print(f"  [TwoTowerEvalDataset({split})] {len(user_ids):,} users")

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int):
        uid   = self.user_ids[idx]
        seq   = self.seqs_dict[uid]
        label = int(self.labels_dict[uid])

        if len(seq) < self.max_seq_len:
            seq = [PAD_TOKEN] * (self.max_seq_len - len(seq)) + seq
        else:
            seq = seq[-self.max_seq_len:]

        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def get_user_history(self, idx: int):
        uid = self.user_ids[idx]
        return set(self.seqs_dict[uid])

# ─────────────────────────────────────────────────────────────
# ITEM DATASET (UNCHANGED)
# ─────────────────────────────────────────────────────────────

class ItemCatalogDataset(Dataset):
    def __init__(self):
        import json
        meta = json.load(open(PROCESSED_DIR / "dataset_meta.json"))
        n_items = meta['n_items']
        self.item_ids = list(range(1, n_items + 1))
        print(f"  [ItemCatalogDataset] {len(self.item_ids):,} items")

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.item_ids[idx], dtype=torch.long)

# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Testing TwoTowerTrainDataset")
    print("=" * 60)

    train_ds = TwoTowerTrainDataset()
    loader   = DataLoader(train_ds, batch_size=4, shuffle=True)
    seqs, targets = next(iter(loader))

    print(f"  seqs.shape    : {seqs.shape}")
    print(f"  targets.shape : {targets.shape}")
    print(f"  sample seq    : {seqs[0].tolist()}")
    print(f"  sample target : {targets.tolist()}")

    assert seqs.shape[1] == MAX_SEQ_LEN
    assert (targets > 0).all()

    print("\n  ✅ Train dataset OK")