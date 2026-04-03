import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR    = Path("data/splits")
SEQUENCES_DIR = Path("data/sequences")

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ─────────────────────────────────────────────────────────────
# INTERACTION LOADER
# ─────────────────────────────────────────────────────────────

def load_interaction_data() -> Tuple[pd.DataFrame, dict, dict]:
    """
    Loads all Week 1 outputs needed for training.

    Returns:
        df       : pd.DataFrame with columns [user_idx, item_idx, timestamp]
        meta     : dict with n_users, n_items, n_items_with_pad
        sequences: dict {user_idx: [item1, item2, ..., itemN]} (chronological)
    """
    import json

    print_section("Loading Interaction Data")

    df   = pd.read_csv(PROCESSED_DIR / "interactions.csv", parse_dates=['timestamp'])
    meta = json.load(open(PROCESSED_DIR / "dataset_meta.json"))

    with open(SPLITS_DIR / "train_seqs.pkl", "rb") as f:
        train_seqs = pickle.load(f)

    with open(SPLITS_DIR / "val_labels.pkl", "rb") as f:
        val_labels = pickle.load(f)

    with open(SPLITS_DIR / "test_labels.pkl", "rb") as f:
        test_labels = pickle.load(f)

    # Align ID space: interactions.csv may still be 0-based while split files are 1-based (PAD=0).
    min_df_item = int(df["item_idx"].min()) if len(df) > 0 else 0
    min_seq_item = min((min(seq) for seq in train_seqs.values() if len(seq) > 0), default=0)
    if min_df_item == 0 and min_seq_item >= 1:
        df = df.copy()
        df["item_idx"] = df["item_idx"] + 1
        print("  ℹ️  Shifted interactions.csv item_idx by +1 to match split ID space")

    print(f"  ✅ interactions.csv : {len(df):,} rows")
    print(f"  ✅ n_users          : {meta['n_users']:,}")
    print(f"  ✅ n_items          : {meta['n_items']:,}")
    print(f"  ✅ train_seqs       : {len(train_seqs):,} users")
    print(f"  ✅ val_labels       : {len(val_labels):,} users")
    print(f"  ✅ test_labels      : {len(test_labels):,} users")

    return df, meta, train_seqs, val_labels, test_labels


# ─────────────────────────────────────────────────────────────
# BUILD USER INTERACTION SETS (FOR NEGATIVE SAMPLING)
# ─────────────────────────────────────────────────────────────

def build_user_item_sets(data) -> Dict[int, set]:
    """
    Builds a dictionary mapping each user_idx to the SET of items they interacted with.

    This is used during negative sampling: when we need a "negative" item,
    we must ensure it is NOT in the user's interaction set.

    Args:
        data: Either
              - DataFrame with columns [user_idx, item_idx], or
              - dict {user_idx: [item_idx, ...]} (preferred: training sequences)

    Returns:
        user_items: {user_idx: {item_idx1, item_idx2, ...}}

    Example:
        user_items[42] = {5, 19, 113, 207, 881, ...}
        If we sample a random item_idx and it's NOT in this set → valid negative.
    """
    print_section("Building User-Item Interaction Sets")

    user_items: Dict[int, set] = {}

    if isinstance(data, dict):
        for user_idx, seq in data.items():
            filtered = {int(i) for i in seq if int(i) != 0}
            user_items[int(user_idx)] = filtered
    else:
        grouped = data.groupby('user_idx')['item_idx'].apply(set)
        for user_idx, item_set in grouped.items():
            user_items[int(user_idx)] = item_set

    # Stats
    set_sizes = [len(s) for s in user_items.values()]
    print(f"  ✅ Built sets for {len(user_items):,} users")
    print(f"  Min items/user  : {min(set_sizes)}")
    print(f"  Max items/user  : {max(set_sizes)}")
    print(f"  Mean items/user : {np.mean(set_sizes):.1f}")
    print(f"  Total item-user pairs covered: {sum(set_sizes):,}")

    return user_items


# ─────────────────────────────────────────────────────────────
# TRAINING DATASET
# ─────────────────────────────────────────────────────────────

class MFTrainDataset(Dataset):
    """
    PyTorch Dataset for Matrix Factorization training.

    For each training sample: returns (user_idx, pos_item_idx, neg_item_idx)
    where:
        user_idx      : the user
        pos_item_idx  : an item the user DID interact with (positive)
        neg_item_idx  : an item the user DID NOT interact with (negative)

    Negative Sampling Strategy:
        - UNIFORM: sample any random item not in user's history
        - POPULARITY: sample from items proportionally to their popularity
          (harder negatives — these are items users are more likely to see)

    We default to UNIFORM for simplicity and speed.

    Args:
        train_seqs     : {user_idx: [item1, item2, ..., itemN]} training sequences
        user_items     : {user_idx: set(all items user interacted with)} — for filtering negatives
        n_items        : total number of items (excluding PAD, PAD = 0)
        neg_per_pos    : number of negative samples per positive interaction (default: 1)
        strategy       : 'uniform' or 'popularity'
        item_pop       : item popularity array (required if strategy='popularity')
    """

    def __init__(
        self,
        train_seqs  : Dict[int, List[int]],
        user_items  : Dict[int, set],
        n_items     : int,
        neg_per_pos : int = 1,
        strategy    : str = 'uniform',
        item_pop    : np.ndarray = None,
    ):
        self.user_items  = user_items
        self.n_items     = n_items
        self.neg_per_pos = neg_per_pos
        self.strategy    = strategy
        self.item_pop    = item_pop

        # Flatten: expand each user's sequence into individual (user, item) pairs
        # Example: user 3 with seq [5, 19, 42] → (3,5), (3,19), (3,42)
        # Each of these is a "positive" pair
        self.samples: List[Tuple[int, int]] = []

        for user_idx, seq in train_seqs.items():
            for item_idx in seq:
                if item_idx == 0:
                    continue  # Skip PAD tokens
                self.samples.append((int(user_idx), int(item_idx)))

        print(f"\n  MFTrainDataset created:")
        print(f"    Positive pairs     : {len(self.samples):,}")
        print(f"    neg_per_pos        : {self.neg_per_pos}")
        print(f"    strategy           : {self.strategy}")
        print(f"    Effective size     : {len(self.samples) * self.neg_per_pos:,} (pos × neg)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns one training triplet: (user, pos_item, neg_item)

        Note: neg_item is sampled ONLINE — different every epoch.
        """
        user_idx, pos_item = self.samples[idx]

        # ── Sample ONE negative item ──────────────────────────
        # Must not be in user's interaction set.
        # Must not be 0 (PAD token).
        neg_item = self._sample_negative(user_idx)

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )

    def _sample_negative(self, user_idx: int) -> int:
        """
        Samples a valid negative item for the given user.

        Strategy:
            UNIFORM: sample from [1, n_items] uniformly at random,
                     reject if it's in the user's interaction set.
                     Expected retries: n_interactions_user / n_items
                     For our dataset: ~20/60000 ≈ 0.03% → almost never retries.

            POPULARITY: sample from items weighted by interaction count.
                        Harder negatives — popular items are easy to mistake for positives.
        """
        user_seen = self.user_items.get(user_idx, set())

        if self.strategy == 'uniform':
            while True:
                # Sample from [1, n_items] — skip 0 (PAD)
                neg = random.randint(1, self.n_items)
                if neg not in user_seen:
                    return neg

        elif self.strategy == 'popularity':
            # Sample proportional to item popularity
            # self.item_pop must be a normalized probability array of length n_items+1
            # index 0 = PAD (weight 0), index 1..n_items = item weights
            while True:
                neg = np.random.choice(len(self.item_pop), p=self.item_pop)
                if neg != 0 and neg not in user_seen:
                    return neg

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. Use 'uniform' or 'popularity'")


# ─────────────────────────────────────────────────────────────
# EVALUATION DATASET
# ─────────────────────────────────────────────────────────────

class MFEvalDataset(Dataset):
    """
    PyTorch Dataset for evaluation (validation or test).

    For each user: returns (user_idx, ground_truth_item)

    During evaluation, we:
    1. Compute scores for ALL items for this user: scores[i] = dot(E_u, E_i)
    2. Rank items by score descending
    3. Check if ground_truth_item is in Top-K ranked items
    4. Compute HR@K and NDCG@K

    NOTE: This does NOT use batching in the traditional sense —
    evaluation is done user-by-user to compute ranking over the full item set.

    Args:
        labels     : {user_idx: ground_truth_item_idx}
        input_seqs : {user_idx: [item1, ..., itemN]} the input sequence for this user
    """

    def __init__(
        self,
        labels     : Dict[int, int],
        input_seqs : Dict[int, List[int]],
    ):
        self.users  = list(labels.keys())
        self.labels = labels
        self.input_seqs = input_seqs

        print(f"\n  MFEvalDataset created: {len(self.users):,} users")

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        user_idx = self.users[idx]
        gt_item  = self.labels[user_idx]

        return (
            torch.tensor(user_idx,  dtype=torch.long),
            torch.tensor(gt_item,   dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────
# POPULARITY DISTRIBUTION (FOR POPULARITY NEGATIVE SAMPLING)
# ─────────────────────────────────────────────────────────────

def build_item_popularity_distribution(df: pd.DataFrame, n_items: int) -> np.ndarray:
    """
    Builds a normalized probability distribution over items based on interaction count.

    Used for popularity-based negative sampling (harder negatives).

    Returns:
        pop_dist: np.ndarray of shape [n_items+1]
                  index 0 = 0 (PAD has 0 probability)
                  index 1..n_items = probability proportional to item popularity
    """
    print_section("Building Item Popularity Distribution")

    item_counts = df.groupby('item_idx').size()

    # Build array indexed by item_idx (items start at 1, 0=PAD)
    pop_arr = np.zeros(n_items + 1, dtype=np.float64)

    for item_idx, count in item_counts.items():
        if 1 <= item_idx <= n_items:
            pop_arr[item_idx] = count

    # Normalize to probability distribution
    total = pop_arr.sum()
    pop_dist = pop_arr / total

    # Sanity checks
    assert abs(pop_dist.sum() - 1.0) < 1e-6, "Distribution must sum to 1"
    assert pop_dist[0] == 0.0, "PAD token must have 0 probability"

    print(f"  ✅ Built popularity distribution over {n_items:,} items")
    print(f"  ✅ Total probability mass (should be 1.0): {pop_dist.sum():.8f}")
    print(f"  ✅ PAD probability (should be 0.0): {pop_dist[0]:.8f}")

    top5_items = np.argsort(pop_dist)[-5:][::-1]
    print(f"  Top 5 most popular item_idxs: {top5_items.tolist()}")

    return pop_dist


# ─────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────

def build_dataloaders(
    train_dataset : MFTrainDataset,
    eval_dataset  : MFEvalDataset,
    batch_size    : int = 1024,
    num_workers   : int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Wraps datasets in PyTorch DataLoaders.

    Args:
        train_dataset : MFTrainDataset
        eval_dataset  : MFEvalDataset
        batch_size    : training batch size (eval uses batch_size=512 for full-ranking)
        num_workers   : CPU workers for data loading (0 = main process, safe for all OS)

    Returns:
        train_loader, eval_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,              # Shuffle every epoch
        num_workers = num_workers,
        pin_memory  = True,              # Faster GPU transfer
        drop_last   = False,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size  = 512,               # Fixed batch size for evaluation
        shuffle     = False,             # Never shuffle eval
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    print(f"\n  DataLoaders created:")
    print(f"    train_loader : {len(train_loader):,} batches  (batch_size={batch_size})")
    print(f"    eval_loader  : {len(eval_loader):,} batches  (batch_size=512)")

    return train_loader, eval_loader


# ─────────────────────────────────────────────────────────────
# SMOKE TEST (run this file directly to verify)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print_section("DS19 — Dataset Smoke Test")

    # Load data
    df, meta, train_seqs, val_labels, test_labels = load_interaction_data()

    n_users = meta['n_users']
    n_items = meta.get('n_items_with_pad', meta['n_items'] + 1) - 1  # exclude PAD

    # Build user-item sets for negative sampling
    user_items = build_user_item_sets(df)

    # Build training dataset
    train_ds = MFTrainDataset(
        train_seqs  = train_seqs,
        user_items  = user_items,
        n_items     = n_items,
        neg_per_pos = 1,
        strategy    = 'uniform',
    )

    # Build eval dataset (using val_seqs as input)
    with open(SPLITS_DIR / "val_seqs.pkl", "rb") as f:
        val_seqs = pickle.load(f)

    eval_ds = MFEvalDataset(
        labels     = val_labels,
        input_seqs = val_seqs,
    )

    # Test one sample
    print_section("Sample Inspection")
    user, pos, neg = train_ds[0]
    print(f"  Train sample[0]:")
    print(f"    user_idx  : {user.item()}")
    print(f"    pos_item  : {pos.item()}")
    print(f"    neg_item  : {neg.item()}")
    print(f"    neg ∉ user history: {pos.item() not in user_items.get(user.item(), set())}")

    user_eval, gt = eval_ds[0]
    print(f"\n  Eval sample[0]:")
    print(f"    user_idx  : {user_eval.item()}")
    print(f"    gt_item   : {gt.item()}")

    # Build dataloaders
    train_loader, eval_loader = build_dataloaders(train_ds, eval_ds, batch_size=1024)

    # Inspect one batch
    print_section("Batch Inspection")
    batch_users, batch_pos, batch_neg = next(iter(train_loader))
    print(f"  Batch shapes:")
    print(f"    users : {batch_users.shape}  dtype={batch_users.dtype}")
    print(f"    pos   : {batch_pos.shape}    dtype={batch_pos.dtype}")
    print(f"    neg   : {batch_neg.shape}    dtype={batch_neg.dtype}")
    print(f"  First 5 users: {batch_users[:5].tolist()}")
    print(f"  First 5 pos  : {batch_pos[:5].tolist()}")
    print(f"  First 5 neg  : {batch_neg[:5].tolist()}")
    print(f"\n  ✅ Dataset smoke test PASSED")