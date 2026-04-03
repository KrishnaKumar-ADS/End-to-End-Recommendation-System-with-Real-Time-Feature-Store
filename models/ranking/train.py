"""
DS19 — Week 5: LightGBM LambdaRank Training (RTX 3050 4GB - Final Version)
Trains ranking model on pre-built datasets with memory-efficient loading.

SMART LOAD: Checks if model exists, only retrains if needed or forced.

Run: python models/ranking/train.py
Run: python models/ranking/train.py --rebuild  (force rebuild)
Run: python models/ranking/train.py --tune    (hyperparameter search)
"""

import os
import sys
import json
import pickle
import gc
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
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
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
LOGS_DIR = PROJECT_ROOT / "logs"

MODEL_SAVE_PATH = MODELS_DIR / "lgbm_ranker.pkl"
LOG_PATH = LOGS_DIR / "ranking_training_log.csv"
RESULTS_PATH = LOGS_DIR / "ranking_training_results.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── CONFIG ──
TRAINING_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5, 10],
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 10,
    "min_child_weight": 1e-3,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "lambdarank_truncation_level": 20,
    "num_threads": 4,  # Match your CPU cores
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,  # Memory-efficient for large datasets
}

EARLY_STOPPING_ROUNDS = 20
MAX_TRAIN_USERS = 50_000
MAX_VAL_USERS = 10_000

# ─────────────────────────────────────────────────────────────
# MEMORY MONITORING
# ─────────────────────────────────────────────────────────────

def log_memory_usage(prefix=""):
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 / 1024
    print(f"  {prefix} RAM: {ram_mb:.0f}MB")
    return ram_mb


def clear_memory():
    gc.collect()
    log_memory_usage("After GC")


# ─────────────────────────────────────────────────────────────
# CHECK IF MODEL IS VALID
# ─────────────────────────────────────────────────────────────

def check_model_valid():
    """Check if trained model exists and is valid."""
    if not MODEL_SAVE_PATH.exists():
        print(f"  ❌ Model file missing: {MODEL_SAVE_PATH}")
        return False
    
    try:
        with open(MODEL_SAVE_PATH, "rb") as f:
            model_data = pickle.load(f)
        
        required_keys = ["model", "params", "feature_names", "best_iteration"]
        missing = [k for k in required_keys if k not in model_data]
        
        if missing:
            print(f"  ❌ Model missing keys: {missing}")
            return False
        
        print(f"  ✅ Model valid (best_iteration={model_data['best_iteration']})")
        return True
        
    except Exception as e:
        print(f"  ❌ Model load error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# LOAD DATASET (MEMORY EFFICIENT)
# ─────────────────────────────────────────────────────────────

def load_ranking_dataset(split_name):
    """Load ranking dataset from parquet with memory monitoring."""
    print(f"\n📂 Loading {split_name} dataset...")
    log_memory_usage("Before load")
    
    df_path = FEATURES_DIR / f"ranking_{split_name}.parquet"
    groups_path = FEATURES_DIR / f"groups_{split_name}.pkl"
    feat_path = FEATURES_DIR / "feature_names.json"
    
    if not df_path.exists():
        print(f"  ❌ Dataset not found: {df_path}")
        print(f"  → Run: python models/ranking/dataset_builder.py")
        return None, None, None, None, None
    
    if not groups_path.exists():
        print(f"  ❌ Groups file not found: {groups_path}")
        return None, None, None, None, None
    
    if not feat_path.exists():
        print(f"  ❌ Feature names not found: {feat_path}")
        return None, None, None, None, None
    
    # Load in chunks to monitor memory
    df = pd.read_parquet(df_path)
    
    with open(groups_path, "rb") as f:
        groups = pickle.load(f)
    
    with open(feat_path, "r") as f:
        feature_names = json.load(f)
    
    log_memory_usage("After parquet load")
    
    # Extract features and labels
    meta_cols = [c for c in ["__user_id", "__item_id", "label", "group_id"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    # Remove leaky features
    leaky_patterns = ["item_last_timestamp", "item_first_timestamp",
                      "user_last_timestamp", "user_first_timestamp"]
    feature_cols = [c for c in feature_cols
                    if not any(p in c for p in leaky_patterns)]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    meta_df = df[meta_cols].copy()
    
    # Free memory
    del df
    clear_memory()
    
    print(f"  ✅ {split_name}: {X.shape[0]:,} rows, {X.shape[1]} features, {len(groups)} users")
    print(f"  ✅ Positive rate: {y.mean():.4f}")
    
    return X, y, groups, meta_df, feature_cols


# ─────────────────────────────────────────────────────────────
# CSV LOGGER
# ─────────────────────────────────────────────────────────────

class CSVLogger:
    """Logs training metrics to CSV for visualization."""
    
    def __init__(self, log_path):
        self.log_path = log_path
        self.rows = []
    
    def log(self, step, train_ndcg5, train_ndcg10, val_ndcg5=None, val_ndcg10=None):
        row = {
            "step": step,
            "train_ndcg5": round(train_ndcg5, 6),
            "train_ndcg10": round(train_ndcg10, 6),
            "val_ndcg5": round(val_ndcg5, 6) if val_ndcg5 is not None else None,
            "val_ndcg10": round(val_ndcg10, 6) if val_ndcg10 is not None else None,
        }
        self.rows.append(row)
    
    def save(self):
        df = pd.DataFrame(self.rows)
        df.to_csv(self.log_path, index=False)
        print(f"  ✅ Training log saved: {self.log_path}")


# ─────────────────────────────────────────────────────────────
# MLFLOW LOGGING (OPTIONAL)
# ─────────────────────────────────────────────────────────────

def try_init_mlflow(params):
    """Attempt to initialize MLflow. Silent fail if not available."""
    try:
        import mlflow
        mlflow.set_experiment("ds19_ranking")
        run = mlflow.start_run(
            run_name=f"lgbm_lambdarank_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        mlflow.log_params(params)
        print("  ✅ MLflow tracking enabled")
        return mlflow, run
    except Exception as e:
        print(f"  ⚠️  MLflow not available ({e}) — training without MLflow")
        return None, None


def log_mlflow_metrics(mlflow, history, best_metrics):
    """Log final training metrics to MLflow."""
    if mlflow is None:
        return
    try:
        if "val" in history and "ndcg@10" in history["val"]:
            for step, v in enumerate(history["val"]["ndcg@10"]):
                mlflow.log_metric("val_ndcg10", v, step=step)
        
        mlflow.log_metrics({
            "best_val_ndcg5": best_metrics.get("best_val_ndcg5", 0),
            "best_val_ndcg10": best_metrics.get("best_val_ndcg10", 0),
            "best_iteration": best_metrics.get("best_iteration", 0),
        })
        print("  ✅ MLflow metrics logged")
    except Exception as e:
        print(f"  ⚠️  MLflow metric logging failed: {e}")


# ─────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────

def train(rebuild_dataset=False):  # ✅ FIXED: Match week5_run.py signature
    """
    Full training pipeline with memory monitoring.
    
    Args:
        rebuild_dataset: bool — if True, force rebuild even if model exists
    """
    print("=" * 65)
    print("  DS19 — Week 5: LightGBM LambdaRank Training")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    log_memory_usage("Init")
    
    # ── Check if model already exists ─────────────────────────
    if not rebuild_dataset and check_model_valid():  # ✅ Use correct param name
        print("\n✅ Model already exists and is valid!")
        print(f"   Path: {MODEL_SAVE_PATH}")
        print("\n   To retrain, run: python models/ranking/train.py --rebuild")
        
        # Still load and show stats
        with open(MODEL_SAVE_PATH, "rb") as f:
            model_data = pickle.load(f)
        
        print(f"\n   Best iteration: {model_data['best_iteration']}")
        print(f"   Features: {len(model_data['feature_names'])}")
        
        return None
    
    # ── Load datasets ─────────────────────────────────────────
    print("\n📂 Loading ranking datasets...")
    
    X_train, y_train, groups_train, meta_train, feature_names = load_ranking_dataset("train")
    X_val, y_val, groups_val, meta_val, _ = load_ranking_dataset("val")
    
    if X_train is None or X_val is None:
        print("\n❌ Could not load datasets!")
        print("   Run: python models/ranking/dataset_builder.py")
        return None
    
    log_memory_usage("After dataset load")
    
    print(f"\n  Train: {X_train.shape[0]:,} rows, {X_train.shape[1]} features, {len(groups_train):,} users")
    print(f"  Val:   {X_val.shape[0]:,} rows, {X_val.shape[1]} features, {len(groups_val):,} users")
    print(f"  Positive rate: {y_train.mean():.5f}")
    
    # ── Initialize MLflow ────────────────────────────────────
    mlflow, run = try_init_mlflow(TRAINING_PARAMS)
    
    # ── Create LightGBM datasets ─────────────────────────────
    print(f"\n🔧 Creating LightGBM datasets...")
    
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=groups_train,
        feature_name=feature_names,
        free_raw_data=False,
        params={"max_bin": 63}  # Memory optimization
    )
    
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        group=groups_val,
        feature_name=feature_names,
        reference=train_data,
        free_raw_data=False,
        params={"max_bin": 63}
    )
    
    log_memory_usage("After LGB dataset creation")
    
    # ── Train ────────────────────────────────────────────────
    print(f"\n🚀 Training LightGBM LambdaRank...")
    print(f"   Objective: {TRAINING_PARAMS['objective']}")
    print(f"   Metric:    {TRAINING_PARAMS['metric']} @ {TRAINING_PARAMS['eval_at']}")
    print(f"   Early stopping: {EARLY_STOPPING_ROUNDS} rounds")
    print(f"   Max rounds: {TRAINING_PARAMS['n_estimators']}")
    print()
    
    # Remove n_estimators from params (passed separately)
    params = {k: v for k, v in TRAINING_PARAMS.items() if k != "n_estimators"}
    
    evals_result = {}
    
    start_time = datetime.now()
    
    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=TRAINING_PARAMS["n_estimators"],
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.log_evaluation(period=10),
            lgb.record_evaluation(evals_result),
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
        ],
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n  ⏱️  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    best_iteration = model.best_iteration
    print(f"  ✅ Best iteration: {best_iteration}")
    
    log_memory_usage("After training")
    
    # ── Extract metrics ──────────────────────────────────────
    train_ndcg5 = evals_result.get("train", {}).get("ndcg@5", [0])
    train_ndcg10 = evals_result.get("train", {}).get("ndcg@10", [0])
    val_ndcg5 = evals_result.get("val", {}).get("ndcg@5", [None] * len(train_ndcg10))
    val_ndcg10 = evals_result.get("val", {}).get("ndcg@10", [None] * len(train_ndcg10))
    
    best_val_ndcg5 = max([v for v in val_ndcg5 if v is not None]) if val_ndcg5 else 0
    best_val_ndcg10 = max([v for v in val_ndcg10 if v is not None]) if val_ndcg10 else 0
    
    # ── Save model ───────────────────────────────────────────
    print(f"\n💾 Saving model...")
    
    save_dict = {
        "model": model,
        "params": TRAINING_PARAMS,
        "feature_names": feature_names,
        "feature_importances": model.feature_importance(importance_type="gain"),
        "best_iteration": best_iteration,
        "training_history": evals_result,
        "trained_at": datetime.now().isoformat(),
    }
    
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(save_dict, f)
    
    model_size = MODEL_SAVE_PATH.stat().st_size / 1024
    print(f"  ✅ Model saved: {MODEL_SAVE_PATH} ({model_size:.1f} KB)")
    
    # ── Log to CSV ───────────────────────────────────────────
    logger = CSVLogger(LOG_PATH)
    
    for step in range(len(train_ndcg10)):
        logger.log(
            step=step + 1,
            train_ndcg5=train_ndcg5[step] if step < len(train_ndcg5) else 0,
            train_ndcg10=train_ndcg10[step],
            val_ndcg5=val_ndcg5[step] if step < len(val_ndcg5) else None,
            val_ndcg10=val_ndcg10[step] if step < len(val_ndcg10) else None,
        )
    logger.save()
    
    # ── Save results JSON ────────────────────────────────────
    results = {
        "model": "LightGBM LambdaRank",
        "objective": TRAINING_PARAMS["objective"],
        "best_iteration": best_iteration,
        "training_time_seconds": round(elapsed, 1),
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "num_features": int(X_train.shape[1]),
        "best_val_ndcg5": round(best_val_ndcg5, 5),
        "best_val_ndcg10": round(best_val_ndcg10, 5),
        "train_ndcg5_final": round(train_ndcg5[-1] if train_ndcg5 else 0, 5),
        "train_ndcg10_final": round(train_ndcg10[-1] if train_ndcg10 else 0, 5),
        "timestamp": datetime.now().isoformat(),
        "feature_names": feature_names[:10],
    }
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✅ Results saved: {RESULTS_PATH}")
    
    # ── Log to MLflow ────────────────────────────────────────
    if mlflow:
        log_mlflow_metrics(mlflow, evals_result, {
            "best_val_ndcg5": best_val_ndcg5,
            "best_val_ndcg10": best_val_ndcg10,
            "best_iteration": best_iteration,
        })
        mlflow.log_artifact(str(MODEL_SAVE_PATH))
        mlflow.end_run()
    
    # ── Print feature importance ─────────────────────────────
    print(f"\n📊 Top 15 Features by Importance:")
    print(f"  {'Rank':<6} {'Feature':<45} {'Importance %':>12}")
    print(f"  {'-'*65}")
    
    total_imp = save_dict["feature_importances"].sum()
    sorted_idx = np.argsort(save_dict["feature_importances"])[::-1]
    
    for rank, i in enumerate(sorted_idx[:15], 1):
        fname = feature_names[i]
        imp_pct = save_dict["feature_importances"][i] / total_imp * 100 if total_imp > 0 else 0
        print(f"  {rank:<6} {fname:<45} {imp_pct:>11.2f}%")
    
    # ── Print summary ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRAINING SUMMARY")
    print("=" * 65)
    print(f"  Model:           LightGBM LambdaRank")
    print(f"  Best iteration:  {best_iteration}")
    print(f"  Val NDCG@5:      {best_val_ndcg5:.4f}")
    print(f"  Val NDCG@10:     {best_val_ndcg10:.4f}")
    print(f"  Training time:   {elapsed:.1f}s")
    print(f"  Model saved:     {MODEL_SAVE_PATH}")
    print(f"  Training log:    {LOG_PATH}")
    print()
    
    # ── Interpretation ───────────────────────────────────────
    print("  📊 How to interpret:")
    if best_val_ndcg10 >= 0.85:
        print(f"     NDCG@10 = {best_val_ndcg10:.3f} → EXCELLENT ranking quality 🟢")
        print("     LightGBM is substantially re-ranking FAISS candidates well.")
    elif best_val_ndcg10 >= 0.70:
        print(f"     NDCG@10 = {best_val_ndcg10:.3f} → GOOD ranking quality 🟡")
        print("     Solid performance. May improve with more features or tuning.")
    else:
        print(f"     NDCG@10 = {best_val_ndcg10:.3f} → NEEDS IMPROVEMENT 🔴")
        print("     See Troubleshooting section in WEEK_5_COMPLETE.md")
    
    print("=" * 65)
    
    # ── Cleanup ──────────────────────────────────────────────
    del X_train, y_train, X_val, y_val, train_data, val_data
    clear_memory()
    
    return model


# ─────────────────────────────────────────────────────────────
# QUICK HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────────────────────

def quick_tune():
    """Run quick hyperparameter search over key LambdaRank parameters."""
    print("=" * 65)
    print("  DS19 — Week 5: Quick Hyperparameter Search")
    print("=" * 65)
    
    # Load datasets
    X_train, y_train, groups_train, _, feature_names = load_ranking_dataset("train")
    X_val, y_val, groups_val, _, _ = load_ranking_dataset("val")
    
    if X_train is None:
        print("  ❌ Dataset not found! Run dataset_builder.py first")
        return
    
    # Configurations to try
    configs = [
        {
            "name": "baseline",
            "num_leaves": 64,
            "learning_rate": 0.05,
            "lambdarank_truncation_level": 20,
        },
        {
            "name": "deeper_trees",
            "num_leaves": 128,
            "learning_rate": 0.05,
            "lambdarank_truncation_level": 20,
        },
        {
            "name": "faster_lr",
            "num_leaves": 64,
            "learning_rate": 0.1,
            "lambdarank_truncation_level": 20,
        },
        {
            "name": "higher_truncation",
            "num_leaves": 64,
            "learning_rate": 0.05,
            "lambdarank_truncation_level": 50,
        },
    ]
    
    results = []
    
    for cfg in configs:
        name = cfg.pop("name")
        print(f"\n  🔧 Config: {name} → {cfg}")
        
        params = {**TRAINING_PARAMS, **cfg, "n_estimators": 100, "verbose": -1}
        
        train_data = lgb.Dataset(
            X_train, label=y_train, group=groups_train,
            feature_name=feature_names, free_raw_data=False
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, group=groups_val,
            feature_name=feature_names, reference=train_data, free_raw_data=False
        )
        
        model = lgb.train(
            params={k: v for k, v in params.items() if k != "n_estimators"},
            train_set=train_data,
            num_boost_round=params["n_estimators"],
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
        )
        
        best_ndcg10 = max(model.best_score.get("val", {}).get("ndcg@10", [0]))
        
        results.append({
            "config": name,
            "best_val_ndcg10": round(best_ndcg10, 5),
            "best_iteration": model.best_iteration,
            **cfg,
        })
        
        print(f"     Best NDCG@10: {best_ndcg10:.5f} (iteration {model.best_iteration})")
        
        del train_data, val_data, model
        gc.collect()
    
    # Summary
    print("\n" + "=" * 65)
    print("  HYPERPARAMETER SEARCH RESULTS")
    print("=" * 65)
    results.sort(key=lambda x: x["best_val_ndcg10"], reverse=True)
    
    for r in results:
        print(f"  {r['config']:<20} NDCG@10={r['best_val_ndcg10']:.5f} (iter={r['best_iteration']})")
    
    best = results[0]
    print(f"\n  🏆 Best config: {best['config']} with NDCG@10={best['best_val_ndcg10']:.5f}")
    print("  Retrain with the best config to get the final model.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS19 Week 5 — LightGBM Training")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild model even if it exists"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run quick hyperparameter search"
    )
    parser.add_argument(
        "--rebuild_dataset",
        action="store_true",
        help="Force rebuild dataset and model (alias for --rebuild)"
    )
    args = parser.parse_args()
    
    log_memory_usage("Script start")
    
    # Support both --rebuild and --rebuild_dataset for compatibility
    rebuild_flag = args.rebuild or args.rebuild_dataset
    
    if args.tune:
        quick_tune()
    else:
        train(rebuild_dataset=rebuild_flag)
    
    log_memory_usage("Script complete")