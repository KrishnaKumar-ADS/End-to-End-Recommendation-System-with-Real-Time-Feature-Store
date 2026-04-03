"""
DS19 — Week 5: Ranking Model Evaluation (FIXED - Feature Alignment)
Computes NDCG@K and HR@K for:
  1. FAISS-only baseline (just use faiss_rank as the score)
  2. LightGBM ranker (use LightGBM scores to re-rank)
  3. Comparison table: how much does LightGBM improve over FAISS baseline?

Saves results to: logs/ranking_eval_results.json

Run:
  python models/ranking/evaluate.py
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ranking.model import LightGBMRanker
from models.ranking.dataset import load_ranking_dataset

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "lgbm_ranker.pkl"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
EVAL_RESULTS_PATH = PROJECT_ROOT / "logs" / "ranking_eval_results.json"
NUM_CANDIDATES = 100
K_VALUES = [1, 5, 10, 20, 50]

os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────

def dcg_at_k(relevances, k):
    relevances = np.array(relevances[:k], dtype=np.float32)
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1)
    gains = (2 ** relevances - 1) / np.log2(positions + 1)
    return float(gains.sum())


def ndcg_at_k(relevances, k):
    actual_dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def hit_rate_at_k(relevances, k):
    return 1.0 if any(r > 0 for r in relevances[:k]) else 0.0


def mrr_at_k(relevances, k):
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


# ─────────────────────────────────────────────────────────────
# EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────

def evaluate_ranker(scores, y, groups, model_name, k_values=K_VALUES):
    print(f"\n  Evaluating: {model_name}")

    all_ndcg = {k: [] for k in k_values}
    all_hr = {k: [] for k in k_values}
    all_mrr = {k: [] for k in k_values}

    start_idx = 0
    for group_size in groups:
        end_idx = start_idx + group_size
        user_scores = scores[start_idx:end_idx]
        user_labels = y[start_idx:end_idx]
        sorted_indices = np.argsort(user_scores)[::-1]
        sorted_labels = user_labels[sorted_indices].tolist()

        for k in k_values:
            all_ndcg[k].append(ndcg_at_k(sorted_labels, k))
            all_hr[k].append(hit_rate_at_k(sorted_labels, k))
            all_mrr[k].append(mrr_at_k(sorted_labels, k))

        start_idx = end_idx

    metrics = {}
    for k in k_values:
        ndcg = np.mean(all_ndcg[k])
        hr = np.mean(all_hr[k])
        mrr = np.mean(all_mrr[k])
        metrics[f"NDCG@{k}"] = round(float(ndcg), 5)
        metrics[f"HR@{k}"] = round(float(hr), 5)
        metrics[f"MRR@{k}"] = round(float(mrr), 5)

    print(f"\n  {'Metric':<12} " + " ".join(f"{'K='+str(k):>10}" for k in k_values))
    print("  " + "-" * (12 + 11 * len(k_values)))

    for metric_prefix in ["NDCG", "HR", "MRR"]:
        row = f"  {metric_prefix:<12} "
        for k in k_values:
            val = metrics[f"{metric_prefix}@{k}"]
            row += f"{val:>10.4f}"
        print(row)

    return metrics


# ─────────────────────────────────────────────────────────────
# FAISS BASELINE SCORES
# ─────────────────────────────────────────────────────────────

def get_faiss_baseline_scores(X, feature_names, groups):
    if "faiss_rank_normalized" in feature_names:
        idx = feature_names.index("faiss_rank_normalized")
        return X[:, idx]
    elif "faiss_score" in feature_names:
        idx = feature_names.index("faiss_score")
        return X[:, idx]
    else:
        scores = np.zeros(len(X))
        start = 0
        for g in groups:
            scores[start:start + g] = np.linspace(1.0, 0.0, g)
            start += g
        return scores


# ─────────────────────────────────────────────────────────────
# ALIGN FEATURES WITH TRAINING
# ─────────────────────────────────────────────────────────────

def align_features(X_eval, eval_feature_names, train_feature_names):
    """
    Align evaluation features with training features.
    Adds missing columns as zeros, reorders to match training order.
    """
    if set(eval_feature_names) == set(train_feature_names):
        # Just reorder to match training order
        col_order = [eval_feature_names.index(f) for f in train_feature_names]
        return X_eval[:, col_order], train_feature_names
    
    # Need to add missing features
    X_aligned = np.zeros((X_eval.shape[0], len(train_feature_names)), dtype=np.float32)
    
    for i, train_feat in enumerate(train_feature_names):
        if train_feat in eval_feature_names:
            eval_idx = eval_feature_names.index(train_feat)
            X_aligned[:, i] = X_eval[:, eval_idx]
        else:
            # Missing feature - fill with zeros
            X_aligned[:, i] = 0.0
            print(f"  ⚠️  Missing feature (filled with 0): {train_feat}")
    
    return X_aligned, train_feature_names


# ─────────────────────────────────────────────────────────────
# FULL EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  DS19 — Week 5: Ranking Model Evaluation")
    print("=" * 65)

    # ── Load dataset ──────────────────────────────────────────────
    print("\n📂 Loading validation dataset...")
    X_val, y_val, groups_val, meta_val, feature_names = load_ranking_dataset("val")

    if X_val is None:
        print("  ❌ Validation dataset not found!")
        print("     Run: python models/ranking/dataset.py")
        return

    print(f"  Val rows:     {len(X_val):,}")
    print(f"  Val users:    {len(groups_val):,}")
    print(f"  Val features: {X_val.shape[1]}")
    print(f"  Positive rate: {y_val.mean():.5f}")

    # ── Compute FAISS baseline scores ─────────────────────────────
    print("\n📊 Computing FAISS baseline (no re-ranking)...")
    faiss_scores = get_faiss_baseline_scores(X_val, feature_names, groups_val)
    faiss_metrics = evaluate_ranker(
        faiss_scores, y_val, groups_val,
        "FAISS Baseline (no re-ranking)"
    )

    # ── Load and evaluate LightGBM ranker ─────────────────────────
    print("\n🤖 Loading LightGBM ranker...")
    if not MODEL_PATH.exists():
        print(f"  ❌ Model not found at {MODEL_PATH}")
        print("     Run: python models/ranking/train.py")
        return

    try:
        ranker = LightGBMRanker.load(str(MODEL_PATH))
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return

    print(f"  ✅ Model loaded: {MODEL_PATH}")
    print(f"     Best iteration: {ranker.best_iteration}")
    print(f"     Training features: {len(ranker.feature_names)}")
    print(f"     Evaluation features: {len(feature_names)}")

    # ── FIX: Align features with training ─────────────────────────
    if len(feature_names) != len(ranker.feature_names):
        print(f"\n  ⚠️  Feature mismatch detected! Aligning...")
        print(f"     Training: {len(ranker.feature_names)} features")
        print(f"     Evaluation: {len(feature_names)} features")
        
        X_val, feature_names = align_features(
            X_val, feature_names, ranker.feature_names
        )
        
        print(f"  ✅ Features aligned: {X_val.shape[1]} features")

    print("\n📊 Computing LightGBM ranking scores...")
    try:
        lgbm_scores = ranker.predict(X_val)
    except Exception as e:
        print(f"  ❌ Prediction failed: {e}")
        print("  💡 Try rebuilding dataset with: python models/ranking/dataset.py --rebuild")
        return
    
    lgbm_metrics = evaluate_ranker(
        lgbm_scores, y_val, groups_val,
        "LightGBM LambdaRank"
    )

    # ── Comparison table ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  COMPARISON: FAISS Baseline vs LightGBM Ranker")
    print("=" * 65)

    print(f"\n  {'Metric':<12} {'FAISS':>10} {'LightGBM':>12} {'Improvement':>14}")
    print("  " + "-" * 50)

    for k in K_VALUES:
        for metric_prefix in ["NDCG", "HR"]:
            key = f"{metric_prefix}@{k}"
            faiss_val = faiss_metrics.get(key, 0)
            lgbm_val = lgbm_metrics.get(key, 0)
            improvement = lgbm_val - faiss_val
            pct_change = (improvement / faiss_val * 100) if faiss_val > 0 else 0

            marker = "🟢" if improvement > 0.005 else ("🟡" if improvement > 0 else "🔴")
            print(f"  {key:<12} {faiss_val:>10.4f} {lgbm_val:>12.4f} "
                  f"{improvement:>+10.4f} ({pct_change:>+5.1f}%)  {marker}")

    # ── Feature importance ─────────────────────────────────────────
    print("\n📊 Top 15 Features by Importance (gain):")
    print(f"  {'Rank':<6} {'Feature':<45} {'Importance %':>12}")
    print("  " + "-" * 65)

    if ranker.feature_names and ranker.feature_importances is not None:
        total_imp = ranker.feature_importances.sum()
        sorted_idx = np.argsort(ranker.feature_importances)[::-1]
        for rank, i in enumerate(sorted_idx[:15], 1):
            fname = ranker.feature_names[i]
            imp_pct = ranker.feature_importances[i] / total_imp * 100 if total_imp > 0 else 0
            print(f"  {rank:<6} {fname:<45} {imp_pct:>11.2f}%")

    # ── Save results ───────────────────────────────────────────────
    eval_results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "val_users": len(groups_val),
        "val_rows": len(X_val),
        "num_features": X_val.shape[1],
        "faiss_baseline": faiss_metrics,
        "lgbm_lambdarank": lgbm_metrics,
        "improvement": {
            k: round(lgbm_metrics.get(k, 0) - faiss_metrics.get(k, 0), 5)
            for k in lgbm_metrics
        },
    }

    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n  ✅ Results saved: {EVAL_RESULTS_PATH}")

    # ── Final interpretation ──────────────────────────────────────
    ndcg10_lgbm = lgbm_metrics.get("NDCG@10", 0)
    ndcg10_faiss = faiss_metrics.get("NDCG@10", 0)
    improvement = ndcg10_lgbm - ndcg10_faiss

    print("\n" + "=" * 65)
    print("  FINAL VERDICT")
    print("=" * 65)

    if improvement > 0.02:
        print(f"  🟢 LightGBM significantly improves ranking quality")
        print(f"     NDCG@10: {ndcg10_faiss:.4f} → {ndcg10_lgbm:.4f} "
              f"(+{improvement:.4f})")
        print(f"     The feature engineering is effective.")
    elif improvement > 0:
        print(f"  🟡 LightGBM provides marginal improvement")
        print(f"     NDCG@10: {ndcg10_faiss:.4f} → {ndcg10_lgbm:.4f} "
              f"(+{improvement:.4f})")
        print(f"     Consider adding more informative features.")
    else:
        print(f"  🔴 LightGBM not improving over FAISS baseline")
        print(f"     Check Troubleshooting in WEEK_5_COMPLETE.md")
        print(f"     Most likely issue: data leakage or label mismatch")

    print("=" * 65)


if __name__ == "__main__":
    main()