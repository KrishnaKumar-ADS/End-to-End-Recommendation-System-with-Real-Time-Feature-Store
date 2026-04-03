import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.transformer.dataset import SASRecEvalDataset, load_datasets
from models.transformer.model   import SASRecModel
from experiments.metrics        import hit_rate_at_k, ndcg_at_k, mrr_at_k

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

EVAL_K      = [5, 10, 20]
BATCH_SIZE  = 512              # users per eval batch
MODELS_DIR  = Path("models/saved")
LOGS_DIR    = Path("logs")
PROCESSED_DIR = Path("data/processed")

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)

def load_model(path: Path, device: torch.device) -> SASRecModel:
    """Load a saved SASRec model from checkpoint."""
    # Fix: Use weights_only=True for security (PyTorch 2.0+), fallback for older versions
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only
        ckpt = torch.load(path, map_location=device)
    
    hp = ckpt["hyperparams"]
    model = SASRecModel(
        n_items    = hp["n_items"],
        hidden_dim = hp["hidden_dim"],
        max_seq_len= hp["max_seq_len"],
        num_blocks = hp["num_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# FULL RANKING EVALUATION
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_full_ranking(
    model:      SASRecModel,
    eval_ds:    SASRecEvalDataset,
    device:     torch.device,
    k_list:     List[int] = EVAL_K,
    split_name: str       = "val",
) -> Dict[str, float]:
    """
    Full-item ranking evaluation.

    For each user:
      1. Feed their padded sequence into SASRec → get h_last [D]
      2. Score ALL n_items: scores = h_last @ E_items.T → [n_items]
      3. Set scores of training items (history) to -inf (masking)
      4. Rank the held-out label item among all remaining items
      5. Compute HR@K, NDCG@K, MRR@K
    """
    model.eval()
    meta = json.load(open(PROCESSED_DIR / "dataset_meta.json"))
    n_items = meta.get("n_items_with_pad", meta["n_items"] + 1)

    # ── Pre-load full item embedding matrix ───────────────────
    all_item_ids = torch.arange(n_items, device=device)
    item_embs = model.item_emb(all_item_ids)   # [n_items, D]
    print(f"  Item embedding matrix: {item_embs.shape} → GPU")

    # ── Per-user metrics storage ──────────────────────────────
    ranks = []   # rank of the correct item for each user (1-indexed)

    loader = DataLoader(
        eval_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
    )

    for batch_idx, batch in enumerate(loader):
        seq   = batch["seq"].to(device)     # [B, L]
        label = batch["label"].to(device)   # [B]
        users = batch["user"].tolist()      # list of user IDs
        B     = seq.shape[0]

        # ── Forward pass ──────────────────────────────────────
        hidden = model(seq)         # [B, L, D]
        h_last = hidden[:, -1, :]   # [B, D] — last position

        # ── Score all items ───────────────────────────────────
        scores = h_last @ item_embs.T   # [B, n_items]

        # ── Mask training history items ───────────────────────
        for i in range(B):
            # Fix: Compute user_idx once
            if hasattr(eval_ds, 'user_ids'):
                user_idx = eval_ds.user_ids.index(users[i])
            else:
                user_idx = i
            
            # Get training history mask
            history = eval_ds.get_mask(user_idx)
            history_t = torch.tensor(list(history), dtype=torch.long, device=device)
            scores[i, history_t] = float('-inf')

        # ── Get rank of the true label ─────────────────────────
        label_scores = scores.gather(1, label.unsqueeze(1))   # [B, 1]
        rank_tensor  = (scores >= label_scores).sum(dim=1)    # [B] (1-indexed rank)
        ranks.extend(rank_tensor.cpu().tolist())

        if (batch_idx + 1) % 50 == 0:
            print(f"    Evaluated {(batch_idx+1)*BATCH_SIZE:,} / {len(eval_ds):,} users")

    # ── Compute metrics ───────────────────────────────────────
    ranks_array = np.array(ranks)
    metrics = {}

    for k in k_list:
        hr   = hit_rate_at_k(ranks_array, k)
        ndcg = ndcg_at_k(ranks_array, k)
        mrr  = mrr_at_k(ranks_array, k)
        metrics[f"hr@{k}"]   = round(hr,   4)
        metrics[f"ndcg@{k}"] = round(ndcg, 4)
        metrics[f"mrr@{k}"]  = round(mrr,  4)

    # ── Print results ─────────────────────────────────────────
    print(f"\n  {split_name.upper()} RESULTS (Full Item Ranking):")
    print(f"  {'Metric':<12} {'K=5':>8} {'K=10':>8} {'K=20':>8}")
    print(f"  {'─'*40}")
    for metric in ['hr', 'ndcg', 'mrr']:
        row = f"  {metric.upper():<12}"
        for k in k_list:
            row += f" {metrics[f'{metric}@{k}']:>8.4f}"
        print(row)

    return metrics


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print_section("DS19 — Week 3 | SASRec Full Evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Load datasets ─────────────────────────────────────────
    _, val_ds, test_ds, meta = load_datasets()
    n_items = meta.get("n_items_with_pad", meta["n_items"] + 1)

    # ── Load best model ───────────────────────────────────────
    model_path = MODELS_DIR / "sasrec_best.pt"
    if not model_path.exists():
        print(f"  ❌ sasrec_best.pt not found. Run train.py first.")
        return

    print(f"\n  Loading model: {model_path}")
    model = load_model(model_path, device)
    print(f"  Parameters: {model.count_parameters():,}")

    # ── Validation evaluation ─────────────────────────────────
    print_section("Validation Set — Full Ranking")
    val_metrics = evaluate_full_ranking(model, val_ds, device, split_name="val")

    # ── Test evaluation ───────────────────────────────────────
    print_section("Test Set — Full Ranking")
    test_metrics = evaluate_full_ranking(model, test_ds, device, split_name="test")

    # ── Save results ──────────────────────────────────────────
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model"       : "SASRec",
        "val_metrics" : val_metrics,
        "test_metrics": test_metrics,
        "model_path"  : str(model_path),
    }

    out_path = LOGS_DIR / "sasrec_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print_section("Evaluation Complete")
    print(f"  Results saved: {out_path}")
    print(f"\n  Summary:")
    print(f"    Val  HR@10  : {val_metrics.get('hr@10', 0):.4f}")
    print(f"    Test HR@10  : {test_metrics.get('hr@10', 0):.4f}")
    print(f"    Val  NDCG@10: {val_metrics.get('ndcg@10', 0):.4f}")
    print(f"    Test NDCG@10: {test_metrics.get('ndcg@10', 0):.4f}")

    # ── Compare vs MF baseline ────────────────────────────────
    mf_results_path = LOGS_DIR / "mf_eval_results.json"
    if mf_results_path.exists():
        with open(mf_results_path) as f:
            mf = json.load(f)
        mf_hr10 = mf.get("test_metrics", {}).get("hr@10", 0)
        sr_hr10 = test_metrics.get("hr@10", 0)
        delta   = sr_hr10 - mf_hr10
        print(f"\n  vs MF Baseline (Test HR@10):")
        print(f"    MF (Week 2)  : {mf_hr10:.4f}")
        print(f"    SASRec       : {sr_hr10:.4f}")
        print(f"    Improvement  : {delta:+.4f} ({'✅ better' if delta > 0 else '⚠️ worse'})")


if __name__ == "__main__":
    main()