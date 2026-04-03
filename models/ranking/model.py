import os
import json
import pickle
import numpy as np
import lightgbm as lgb


# ─────────────────────────────────────────────────────────────
# LIGHTGBM HYPERPARAMETERS
# These are the parameters for lambdarank training.
# ─────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    # ── Objective and metric ──────────────────────────────────────
    "objective": "lambdarank",        # LambdaRank: directly optimizes NDCG
                                      # Alternative: "rank_xendcg" (newer, sometimes better)
    "metric": "ndcg",                 # Measure NDCG during training
    "eval_at": [5, 10],               # Compute NDCG@5 and NDCG@10

    # ── Tree structure ────────────────────────────────────────────
    "num_leaves": 64,                 # Max leaves per tree.
                                      # Higher = more complex = risk overfitting.
                                      # Rule of thumb: num_leaves < 2^(max_depth)
                                      # For ranking: 64 is a strong default.
    "max_depth": -1,                  # No explicit max depth (controlled by num_leaves)
    "min_child_samples": 10,          # Minimum samples per leaf.
                                      # Prevents overfitting on users with few candidates.
    "min_child_weight": 1e-3,         # Minimum sum of instance weight in a leaf.

    # ── Learning rate and boosting rounds ─────────────────────────
    "learning_rate": 0.05,            # Step size per boosting round.
                                      # Lower → more stable, needs more rounds.
                                      # Higher → faster convergence, may overshoot.
    "n_estimators": 300,              # Maximum boosting rounds (early stopping will
                                      # typically stop this well before 300).

    # ── Regularization ───────────────────────────────────────────
    "reg_alpha": 0.1,                 # L1 regularization on leaf weights.
                                      # Makes leaves sparse → good for noisy features.
    "reg_lambda": 0.1,                # L2 regularization on leaf weights.
                                      # Prevents extreme weights.

    # ── Subsampling (bagging) ─────────────────────────────────────
    "subsample": 0.8,                 # Use 80% of rows per tree.
                                      # Acts as stochastic boosting → reduces overfitting.
    "subsample_freq": 1,              # Subsample every round.
    "colsample_bytree": 0.8,          # Use 80% of features per tree.
                                      # Introduces feature diversity.

    # ── LambdaRank specific ───────────────────────────────────────
    "lambdarank_truncation_level": 20, # Only consider items in top-20 for NDCG computation.
                                       # Focusing on top positions is correct for our use case:
                                       # We only care about NDCG@5 and NDCG@10.
                                       # Higher truncation = considers more positions = slower.

    # ── Computational ─────────────────────────────────────────────
    "num_threads": 4,                 # Number of CPU threads.
                                      # Set to your CPU core count.
    "verbose": -1,                    # Suppress LightGBM output (we handle logging ourselves)

    # ── Reproducibility ───────────────────────────────────────────
    "seed": 42,
    "deterministic": True,            # Deterministic mode (slightly slower, reproducible)
}


class LightGBMRanker:
    """
    Clean wrapper around LightGBM's LambdaRank model.

    Usage:
        ranker = LightGBMRanker()
        ranker.fit(X_train, y_train, groups_train,
                   X_val, y_val, groups_val,
                   feature_names=feature_names)
        scores = ranker.predict(X_test)
        ranker.save("models/saved/lgbm_ranker.pkl")

        ranker2 = LightGBMRanker.load("models/saved/lgbm_ranker.pkl")
        scores = ranker2.predict(X_test)
    """

    def __init__(self, params=None):
        """
        Initialize LightGBMRanker.

        Args:
            params: dict — LightGBM parameters (uses DEFAULT_PARAMS if None)
        """
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model = None            # lgb.Booster — set after fit()
        self.feature_names = None    # list[str] — set after fit()
        self.feature_importances = None  # np.array — set after fit()
        self.best_iteration = None   # int — best round from early stopping
        self.training_history = {}   # dict — validation NDCG per round

    def fit(self, X_train, y_train, groups_train,
            X_val=None, y_val=None, groups_val=None,
            feature_names=None, early_stopping_rounds=20,
            callbacks=None):
        """
        Train the LambdaRank model.

        Args:
            X_train:               np.array [N × F] — training feature matrix
            y_train:               np.array [N] — binary labels (0/1)
            groups_train:          list[int] — group sizes (one per user)
            X_val:                 np.array [M × F] — optional validation features
            y_val:                 np.array [M] — optional validation labels
            groups_val:            list[int] — optional validation groups
            feature_names:         list[str] — feature column names
            early_stopping_rounds: int — stop if NDCG@10 doesn't improve
            callbacks:             list — LightGBM callbacks (e.g., MLflow callback)

        Returns:
            self (for chaining)
        """
        self.feature_names = feature_names

        print(f"  Training config:")
        print(f"    Objective:     {self.params['objective']}")
        print(f"    Metric:        {self.params['metric']} @ {self.params['eval_at']}")
        print(f"    Num leaves:    {self.params['num_leaves']}")
        print(f"    Learning rate: {self.params['learning_rate']}")
        print(f"    Max rounds:    {self.params['n_estimators']}")
        print(f"    Train rows:    {len(X_train):,} ({len(groups_train):,} groups)")

        # ── Build LightGBM Dataset ────────────────────────────────
        # CRITICAL: group= tells LightGBM which rows belong to the same query (user)
        # The rows MUST already be sorted by group (all rows for user 0, then user 1, etc.)
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            group=groups_train,
            feature_name=feature_names,
            free_raw_data=False       # Keep raw data for later inspection
        )

        valid_sets = [train_data]
        valid_names = ["train"]
        evals_result = {}

        if X_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                group=groups_val,
                feature_name=feature_names,
                reference=train_data,  # Reference ensures consistent feature encoding
                free_raw_data=False
            )
            valid_sets.append(val_data)
            valid_names.append("val")
            print(f"    Val rows:      {len(X_val):,} ({len(groups_val):,} groups)")

        # ── Default callbacks ─────────────────────────────────────
        _callbacks = [
            lgb.log_evaluation(period=10),   # Print metrics every 10 rounds
            lgb.record_evaluation(evals_result),  # Record history
        ]

        if X_val is not None:
            _callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds,
                                   first_metric_only=False,
                                   verbose=True)
            )

        if callbacks:
            _callbacks.extend(callbacks)

        # ── Train ─────────────────────────────────────────────────
        params = {k: v for k, v in self.params.items() if k != "n_estimators"}

        self.model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=self.params["n_estimators"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=_callbacks,
        )

        # ── Save training history ─────────────────────────────────
        self.training_history = evals_result
        self.best_iteration = self.model.best_iteration

        # ── Feature importances ───────────────────────────────────
        self.feature_importances = self.model.feature_importance(importance_type="gain")

        print(f"\n  ✅ Training complete!")
        print(f"     Best iteration: {self.best_iteration}")

        # ── Print top-10 most important features ──────────────────
        if feature_names:
            sorted_idx = np.argsort(self.feature_importances)[::-1]
            print(f"\n  📊 Top 10 features by importance (gain):")
            for rank, i in enumerate(sorted_idx[:10], 1):
                feat = feature_names[i]
                imp = self.feature_importances[i]
                print(f"     {rank:2d}. {feat:<40s} {imp:.1f}")

        return self

    def predict(self, X, num_iteration=None):
        """
        Predict relevance scores for candidates.

        Args:
            X:             np.array [N × F] — feature matrix
            num_iteration: int or None — use best iteration if None

        Returns:
            np.array [N] — relevance scores (higher = more relevant)
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet! Call fit() first.")

        itr = num_iteration or self.best_iteration

        scores = self.model.predict(
            X,
            num_iteration=itr
        )
        return scores

    def rank_candidates(self, X, candidate_item_ids, top_k=10, exclude_ids=None):
        """
        Given features for a user's candidates, return top-K ranked item IDs.

        Args:
            X:                  np.array [num_cands × F] — feature matrix
            candidate_item_ids: list[int] or np.array — item IDs corresponding to X rows
            top_k:              int — return top-K items
            exclude_ids:        set or None — item IDs to exclude (user's history)

        Returns:
            top_k_item_ids: list[int] — top-K item IDs, sorted by score (best first)
            top_k_scores:   list[float] — corresponding scores
        """
        scores = self.predict(X)

        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]

        top_k_ids = []
        top_k_scores = []

        for idx in sorted_indices:
            item_id = int(candidate_item_ids[idx])
            if exclude_ids and item_id in exclude_ids:
                continue
            top_k_ids.append(item_id)
            top_k_scores.append(float(scores[idx]))
            if len(top_k_ids) >= top_k:
                break

        return top_k_ids, top_k_scores

    def save(self, path):
        """
        Save the trained model and metadata to disk.

        Saves as a pickle dict containing:
          - model:              the lgb.Booster
          - params:             training parameters
          - feature_names:      list of feature names
          - feature_importances: np.array
          - best_iteration:     int
          - training_history:   dict
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_dict = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "best_iteration": self.best_iteration,
            "training_history": self.training_history,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

        print(f"  ✅ Model saved: {path}  (size: {os.path.getsize(path) / 1024:.1f} KB)")

    @classmethod
    def load(cls, path):
        """
        Load a trained model from disk.

        Args:
            path: str — path to saved pkl file

        Returns:
            LightGBMRanker instance with model loaded
        """
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        ranker = cls(params=save_dict["params"])
        ranker.model = save_dict["model"]
        ranker.feature_names = save_dict["feature_names"]
        ranker.feature_importances = save_dict["feature_importances"]
        ranker.best_iteration = save_dict["best_iteration"]
        ranker.training_history = save_dict["training_history"]

        print(f"  ✅ Model loaded: {path}")
        print(f"     Best iteration: {ranker.best_iteration}")
        print(f"     Features: {len(ranker.feature_names) if ranker.feature_names else 'unknown'}")

        return ranker

    def get_feature_importance_df(self):
        """Return feature importances as a sorted DataFrame."""
        if self.feature_importances is None or self.feature_names is None:
            return None

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance_gain": self.feature_importances,
            "importance_split": self.model.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

        df["importance_gain_pct"] = df["importance_gain"] / df["importance_gain"].sum() * 100

        return df


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST — verify model can be instantiated
# ─────────────────────────────────────────────────────────────

def main():
    import pandas as pd  # needed for get_feature_importance_df()

    print("=" * 60)
    print("  DS19 — Week 5: LightGBM Ranker — Smoke Test")
    print("=" * 60)

    print("\n🔧 Creating synthetic ranking data for smoke test...")

    # ── Create synthetic data ─────────────────────────────────────
    NUM_USERS = 100
    NUM_CANDS = 10   # use 10 for fast test (100 in real training)
    NUM_FEATURES = 20

    np.random.seed(42)
    X = np.random.randn(NUM_USERS * NUM_CANDS, NUM_FEATURES).astype(np.float32)
    y = np.zeros(NUM_USERS * NUM_CANDS, dtype=np.float32)

    # One positive per user (randomly placed)
    for u in range(NUM_USERS):
        pos_idx = u * NUM_CANDS + np.random.randint(0, NUM_CANDS)
        y[pos_idx] = 1.0

    groups = [NUM_CANDS] * NUM_USERS
    feature_names = [f"feature_{i}" for i in range(NUM_FEATURES)]

    # ── Train ─────────────────────────────────────────────────────
    print("\n🚀 Training on synthetic data (fast test)...")
    ranker = LightGBMRanker(params={"n_estimators": 30, "verbose": -1})
    ranker.fit(
        X[:NUM_USERS * NUM_CANDS // 2],
        y[:NUM_USERS * NUM_CANDS // 2],
        groups[:NUM_USERS // 2],
        X[NUM_USERS * NUM_CANDS // 2:],
        y[NUM_USERS * NUM_CANDS // 2:],
        groups[NUM_USERS // 2:],
        feature_names=feature_names,
        early_stopping_rounds=5
    )

    # ── Predict ───────────────────────────────────────────────────
    scores = ranker.predict(X[:NUM_CANDS])
    print(f"\n  Sample scores (first 10 candidates): {scores.round(4)}")

    # ── Rank ──────────────────────────────────────────────────────
    cand_ids = list(range(1, NUM_CANDS + 1))
    top_k_ids, top_k_scores = ranker.rank_candidates(
        X[:NUM_CANDS], cand_ids, top_k=3
    )
    print(f"  Top-3 ranked item IDs: {top_k_ids}")
    print(f"  Top-3 scores:          {[f'{s:.4f}' for s in top_k_scores]}")

    # ── Save / Load ───────────────────────────────────────────────
    test_path = "models/saved/lgbm_ranker_test.pkl"
    ranker.save(test_path)
    ranker2 = LightGBMRanker.load(test_path)
    scores2 = ranker2.predict(X[:NUM_CANDS])
    assert np.allclose(scores, scores2), "Save/Load scores don't match!"
    print(f"\n  ✅ Save/Load verified (scores match)")

    # ── Feature importance ────────────────────────────────────────
    imp_df = ranker.get_feature_importance_df()
    print(f"\n  Top 5 features:\n{imp_df.head(5).to_string()}")

    # ── Cleanup test file ─────────────────────────────────────────
    os.remove(test_path)
    print(f"\n✅ Smoke test passed! LightGBMRanker is working correctly.")


if __name__ == "__main__":
    import pandas as pd
    main()