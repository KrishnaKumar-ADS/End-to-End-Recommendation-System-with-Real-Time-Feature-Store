import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
sys.modules['numpy._core'] = np.core
# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.matrix_factorization.model import MatrixFactorization
from models.matrix_factorization.dataset import (
    load_interaction_data,
    build_user_item_sets,
)
from experiments.metrics import compute_metrics, scores_to_rank

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR    = Path("data/splits")
MODELS_DIR    = Path("models/saved")
LOGS_DIR      = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES      = [5, 10, 20, 50]
BATCH_SIZE    = 256   # Users per batch during evaluation

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> MatrixFactorization:
    """
    Loads a trained MatrixFactorization model from a checkpoint file.

    The checkpoint contains both the model weights AND the hyperparameters,
    so we can reconstruct the model without manually specifying them again.

    Args:
        checkpoint_path : path to .pt checkpoint file
        device          : torch.device to load model onto

    Returns:
        model : MatrixFactorization, loaded and in eval mode
    """
    print_section("Loading Model")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    hp = checkpoint['hyperparams']
    model = MatrixFactorization(
        n_users       = hp['n_users'],
        n_items       = hp['n_items'],
        embedding_dim = hp['embedding_dim'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"  ✅ Loaded: {checkpoint_path}")
    print(f"     Epoch    : {checkpoint['epoch']}")
    print(f"     n_users  : {hp['n_users']:,}")
    print(f"     n_items  : {hp['n_items']:,}")
    print(f"     emb_dim  : {hp['embedding_dim']}")

    val_metrics = checkpoint.get('metrics', {})
    if val_metrics:
        print(f"  Checkpoint val metrics:")
        for k, v in val_metrics.items():
            if isinstance(v, float):
                print(f"    {k:<20}: {v:.4f}")

    return model


# ─────────────────────────────────────────────────────────────
# EVALUATE ON ONE SPLIT
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_split(
    model      : MatrixFactorization,
    labels     : Dict[int, int],
    user_items : Dict[int, set],
    device     : torch.device,
    k_values   : List[int],
    split_name : str,
) -> Dict[str, float]:
    """
    Runs full evaluation for all users in a split.

    Pipeline per user:
      1. Get user embedding from model
      2. Compute dot product with ALL item embeddings → [n_items] scores
      3. Mask training history + PAD token → -inf
      4. Argsort descending → ranked item list
      5. Find rank of ground truth item
      6. Compute HR@K, NDCG@K, MRR@K

    Args:
        model      : trained MatrixFactorization (eval mode)
        labels     : {user_idx: ground_truth_item}
        user_items : {user_idx: set of all interacted items} for masking
        device     : torch.device
        k_values   : e.g. [5, 10, 20, 50]
        split_name : 'val' or 'test'

    Returns:
        metrics dict with HR@K, NDCG@K, MRR@K for all K in k_values
    """
    model.eval()

    # Pre-extract ALL item embeddings (avoid redundant lookups per user)
    # Shape: [n_items, embedding_dim]
    all_item_embs = model.item_emb.weight.data   # [n_items, d]

    user_list = list(labels.keys())
    n_users   = len(user_list)
    ranks     = []  # Will collect one rank per user

    print(f"\n  Evaluating {n_users:,} users on [{split_name}] split...")

    # Process users in batches for GPU efficiency
    for batch_start in tqdm(
        range(0, n_users, BATCH_SIZE),
        desc  = f"  [{split_name}] Eval",
        ncols = 90,
        unit  = "batch",
    ):
        batch_users = user_list[batch_start : batch_start + BATCH_SIZE]
        batch_idxs  = torch.tensor(batch_users, dtype=torch.long).to(device)

        # Get user embeddings: [batch, d]
        user_embs = model.user_emb(batch_idxs)

        # Compute scores vs all items: [batch, n_items]
        # (user_embs @ all_item_embs.T)
        scores_batch = (user_embs @ all_item_embs.t()).cpu().numpy()  # [B, n_items]

        # Process each user in the batch
        for i, user_idx in enumerate(batch_users):
            gt_item     = labels[user_idx]
            user_scores = scores_batch[i]  # [n_items]

            # Get items this user has seen (for masking)
            seen = user_items.get(user_idx, set())

            # Compute rank of gt_item
            rank = scores_to_rank(user_scores, gt_item, mask_items=seen)
            ranks.append(rank)

    # Aggregate metrics
    metrics = compute_metrics(ranks, k_values=k_values)

    return metrics, ranks


# ─────────────────────────────────────────────────────────────
# PRINT RESULTS TABLE
# ─────────────────────────────────────────────────────────────

def print_results_table(val_metrics: dict, test_metrics: dict, k_values: List[int]):
    """
    Pretty-prints a comparison table of val vs test metrics.
    """
    print_section("Evaluation Results")

    print()
    print(f"  {'Metric':<15} {'Validation':>12} {'Test':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")

    for k in k_values:
        hr_key   = f'HR@{k}'
        ndcg_key = f'NDCG@{k}'
        mrr_key  = f'MRR@{k}'

        if hr_key in val_metrics:
            print(f"  {hr_key:<15} {val_metrics[hr_key]:>12.4f} {test_metrics.get(hr_key, 0):>12.4f}")
        if ndcg_key in val_metrics:
            print(f"  {ndcg_key:<15} {val_metrics[ndcg_key]:>12.4f} {test_metrics.get(ndcg_key, 0):>12.4f}")
        if mrr_key in val_metrics:
            print(f"  {mrr_key:<15} {val_metrics[mrr_key]:>12.4f} {test_metrics.get(mrr_key, 0):>12.4f}")
        print()

    print(f"  {'mean_rank':<15} {val_metrics.get('mean_rank', 0):>12.1f} {test_metrics.get('mean_rank', 0):>12.1f}")
    print(f"  {'median_rank':<15} {val_metrics.get('median_rank', 0):>12.1f} {test_metrics.get('median_rank', 0):>12.1f}")
    print(f"  {'n_users':<15} {val_metrics.get('n_users', 0):>12,} {test_metrics.get('n_users', 0):>12,}")

    print()
    print(f"  ⭐  KEY METRIC  ⭐")
    print(f"  Test HR@10   = {test_metrics.get('HR@10', 0):.4f}")
    print(f"  Test NDCG@10 = {test_metrics.get('NDCG@10', 0):.4f}")
    print()
    print(f"  Interpretation:")
    hr10 = test_metrics.get('HR@10', 0)
    ndcg10 = test_metrics.get('NDCG@10', 0)
    print(f"    HR@10 = {hr10:.4f} means in {hr10*100:.1f}% of cases,")
    print(f"    the correct item appeared in the top 10 recommendations.")
    print(f"    NDCG@10 = {ndcg10:.4f} reflects the average position quality.")


# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────

def save_results(
    val_metrics  : dict,
    test_metrics : dict,
    val_ranks    : List[int],
    test_ranks   : List[int],
):
    """Saves evaluation results and rank distributions to disk."""
    results = {
        'model'       : 'MatrixFactorization',
        'val_metrics' : val_metrics,
        'test_metrics': test_metrics,
    }

    with open(LOGS_DIR / "mf_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save rank distributions
    pd.DataFrame({'val_rank': val_ranks}).to_csv(
        LOGS_DIR / "mf_val_ranks.csv", index=False
    )
    pd.DataFrame({'test_rank': test_ranks}).to_csv(
        LOGS_DIR / "mf_test_ranks.csv", index=False
    )

    print(f"\n  ✅ Saved: logs/mf_eval_results.json")
    print(f"  ✅ Saved: logs/mf_val_ranks.csv")
    print(f"  ✅ Saved: logs/mf_test_ranks.csv")


# ─────────────────────────────────────────────────────────────
# RANK DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyze_rank_distribution(ranks: List[int], split_name: str):
    """Prints a rank distribution summary — useful for debugging."""
    print_section(f"Rank Distribution Analysis ({split_name})")

    ranks_arr = np.array(ranks)
    total = len(ranks_arr)

    print(f"\n  Total users : {total:,}")
    print(f"  Min rank    : {ranks_arr.min()}")
    print(f"  Max rank    : {ranks_arr.max()}")
    print(f"  Mean rank   : {ranks_arr.mean():.1f}")
    print(f"  Median rank : {np.median(ranks_arr):.1f}")
    print(f"\n  Rank thresholds:")

    for threshold in [1, 3, 5, 10, 20, 50, 100, 200]:
        count = (ranks_arr <= threshold).sum()
        pct   = count / total * 100
        bar   = "█" * int(pct / 2)
        print(f"  ≤{threshold:<5} : {count:>8,}  ({pct:5.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print_section("DS19 — Week 2 | Full Evaluation Pipeline")

    # ── Device ────────────────────────────────────────────────
    device = get_device()
    print(f"  Device: {device}")

    # ── Load Data ─────────────────────────────────────────────
    df, meta, train_seqs, val_labels, test_labels = load_interaction_data()
    user_items = build_user_item_sets(train_seqs)

    # Also load val_seqs and test_seqs (needed to check for seq availability)
    with open(SPLITS_DIR / "val_seqs.pkl", "rb") as f:
        val_seqs = pickle.load(f)
    with open(SPLITS_DIR / "test_seqs.pkl", "rb") as f:
        test_seqs = pickle.load(f)

    # ── Load Model ────────────────────────────────────────────
    checkpoint_path = str(MODELS_DIR / "mf_best.pt")
    model = load_model(checkpoint_path, device)

    # ── Validate Val Labels ───────────────────────────────────
    # Ensure val_labels users are a subset of user_items users
    print_section("Label Validation")
    val_user_set  = set(val_labels.keys())
    test_user_set = set(test_labels.keys())
    ui_user_set   = set(user_items.keys())

    print(f"  val_labels users  : {len(val_user_set):,}")
    print(f"  test_labels users : {len(test_user_set):,}")
    print(f"  user_items users  : {len(ui_user_set):,}")

    # Find users not in user_items (sanity check)
    missing_val  = val_user_set - ui_user_set
    missing_test = test_user_set - ui_user_set
    if missing_val:
        print(f"  ⚠️  {len(missing_val)} val users not in user_items (they'll have no masking)")
    else:
        print(f"  ✅ All val users present in user_items")
    if missing_test:
        print(f"  ⚠️  {len(missing_test)} test users not in user_items")
    else:
        print(f"  ✅ All test users present in user_items")

    # ── Evaluate on Validation Set ────────────────────────────
    val_metrics, val_ranks = evaluate_split(
        model      = model,
        labels     = val_labels,
        user_items = user_items,
        device     = device,
        k_values   = K_VALUES,
        split_name = "val",
    )

    # ── Evaluate on Test Set ──────────────────────────────────
    test_metrics, test_ranks = evaluate_split(
        model      = model,
        labels     = test_labels,
        user_items = user_items,
        device     = device,
        k_values   = K_VALUES,
        split_name = "test",
    )

    # ── Print Results ─────────────────────────────────────────
    print_results_table(val_metrics, test_metrics, K_VALUES)

    # ── Rank Distribution ─────────────────────────────────────
    analyze_rank_distribution(test_ranks, "test")

    # ── Save Results ──────────────────────────────────────────
    save_results(val_metrics, test_metrics, val_ranks, test_ranks)

    print()
    print("=" * 60)
    print("✅  Full Evaluation Complete!")
    print("=" * 60)
    print()
    print("  Results saved to:")
    print("    logs/mf_eval_results.json")
    print("    logs/mf_val_ranks.csv")
    print("    logs/mf_test_ranks.csv")
    print()
    print("  Next: python models/matrix_factorization/inference.py")
    print("=" * 60)


if __name__ == "__main__":
    main()