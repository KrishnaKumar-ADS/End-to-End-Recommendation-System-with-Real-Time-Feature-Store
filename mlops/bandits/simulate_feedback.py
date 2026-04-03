import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlops.bandits.bandit_engine import ThompsonSamplingBandit
from mlops.bandits.bandit_store import BanditStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_item_popularity(interactions_path: str, n_items: int) -> np.ndarray:
    """
    Compute item popularity (interaction count) from interactions.csv.
    Returns: popularity[item_idx] = count of interactions for that item.
    """
    logger.info(f"Loading interactions from {interactions_path}...")
    df = pd.read_csv(interactions_path, usecols=["item_idx"])
    counts = df["item_idx"].value_counts()

    popularity = np.zeros(n_items, dtype=np.float32)
    for item_idx, count in counts.items():
        if 0 <= item_idx < n_items:
            popularity[item_idx] = count

    max_pop = popularity.max()
    if max_pop > 0:
        popularity = popularity / max_pop  # normalize to [0, 1]

    logger.info(
        f"Item popularity computed | "
        f"n_items={n_items} | "
        f"items_with_data={int((popularity > 0).sum())} | "
        f"max_raw_count={int(counts.max())}"
    )
    return popularity


def compute_true_ctr(popularity: np.ndarray) -> np.ndarray:
    """
    Synthetic true CTR model:
      true_ctr[i] = 0.05 + 0.90 * popularity[i]
    
    This maps popularity [0,1] → CTR [0.05, 0.95]
    Popular items: high CTR
    Rare items:    5% base CTR (floor, never zero)
    """
    return 0.05 + 0.90 * popularity


def simulate_user_feedback(
    bandit: ThompsonSamplingBandit,
    store: BanditStore,
    true_ctr: np.ndarray,
    n_users: int = 2000,
    n_recs_per_user: int = 10,
    top_k_pool: int = 100,
    seed: int = 42,
    save_every: int = 500,
) -> Dict:
    """
    Simulate user feedback for the bandit.
    
    For each simulated user:
      1. Sample top_k_pool candidate items randomly (simulating LightGBM output)
      2. Bandit re-ranks to top n_recs_per_user
      3. Simulate clicks: clicked[item] ~ Bernoulli(true_ctr[item])
      4. Update bandit for each impression (whether clicked or not)
    
    Args:
        bandit:            ThompsonSamplingBandit instance
        store:             BanditStore for Redis persistence
        true_ctr:          Synthetic true click-through rate per item
        n_users:           Number of simulated user sessions
        n_recs_per_user:   Items shown per session
        top_k_pool:        LightGBM candidate pool size (100 is standard)
        seed:              Random seed for reproducibility
        save_every:        Persist arms to Redis every N users
    
    Returns:
        Simulation statistics dictionary
    """
    rng = np.random.default_rng(seed)
    n_items = len(true_ctr)

    total_impressions = 0
    total_clicks = 0
    session_ctrs = []

    logger.info(
        f"Starting bandit simulation | "
        f"n_users={n_users} | "
        f"n_recs={n_recs_per_user} | "
        f"pool_size={top_k_pool}"
    )

    for user_idx in range(n_users):
        # Simulate LightGBM candidate pool (sample from catalog)
        # In reality, this would be the output of Two-Tower + FAISS
        candidates = rng.choice(n_items, size=top_k_pool, replace=False).tolist()

        # Simulate LightGBM scores (correlated with popularity, some noise)
        lgbm_scores = [
            float(true_ctr[item] + rng.normal(0, 0.05))
            for item in candidates
        ]

        # Bandit re-ranks candidates
        ranked = bandit.rerank(
            candidate_item_ids=candidates[:n_recs_per_user],  # already top-K
            lgbm_scores=lgbm_scores[:n_recs_per_user],
            blend_weight=0.3,
        )

        # Simulate clicks on shown items
        session_clicks = 0
        for item_idx, _ in ranked:
            # Bernoulli click with item's true CTR
            clicked = int(rng.random() < true_ctr[item_idx])
            bandit.update(item_idx, reward=clicked)
            total_impressions += 1
            total_clicks += clicked
            session_clicks += clicked

        session_ctrs.append(session_clicks / n_recs_per_user)

        # Persist to Redis periodically (not every step — too slow)
        if (user_idx + 1) % save_every == 0:
            saved = store.save_all(bandit)
            logger.info(
                f"  Checkpoint at user {user_idx + 1}/{n_users} | "
                f"CTR so far: {total_clicks/total_impressions:.4f} | "
                f"arms_saved={saved}"
            )

    # Final save
    store.save_all(bandit)

    simulated_ctr = total_clicks / total_impressions if total_impressions > 0 else 0
    stats = {
        "n_users_simulated":    n_users,
        "total_impressions":    total_impressions,
        "total_clicks":         total_clicks,
        "simulated_ctr":        round(simulated_ctr, 4),
        "avg_session_ctr":      round(float(np.mean(session_ctrs)), 4),
        "arms_created":         len(bandit),
        "bandit_stats":         bandit.get_stats(),
    }

    logger.info("=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info(f"  Users simulated:   {n_users}")
    logger.info(f"  Total impressions: {total_impressions}")
    logger.info(f"  Total clicks:      {total_clicks}")
    logger.info(f"  Observed CTR:      {simulated_ctr:.4f}")
    logger.info(f"  Arms created:      {len(bandit)}")
    logger.info("=" * 60)

    return stats


def validate_bandit_learning(
    bandit: ThompsonSamplingBandit,
    true_ctr: np.ndarray,
    top_n: int = 20,
) -> None:
    """
    Validate that the bandit has learned to distinguish good items from bad ones.
    Prints top-N items by expected CTR alongside their true CTR.
    """
    print("\n📊 BANDIT LEARNING VALIDATION")
    print(f"{'Item':>8} {'Expected CTR':>14} {'True CTR':>10} {'Trials':>8} {'Quality':>10}")
    print("-" * 56)

    top_items = bandit.top_items_by_ctr(n=top_n)
    for item_info in top_items:
        item_idx   = item_info["item_idx"]
        exp_ctr    = item_info["expected_ctr"]
        t_ctr      = true_ctr[item_idx] if item_idx < len(true_ctr) else 0.0
        trials     = item_info["total_trials"]
        quality    = "🟢 Good" if abs(exp_ctr - t_ctr) < 0.10 else "🟡 Learning"
        print(
            f"{item_idx:>8} "
            f"{exp_ctr:>14.4f} "
            f"{t_ctr:>10.4f} "
            f"{trials:>8} "
            f"{quality:>10}"
        )

    print(
        "\nNote: Expected CTR should converge toward True CTR as trials increase."
    )
    print("Items with few trials will have high variance (that's the exploration working).")


def main():
    import json
    from pathlib import Path

    # ── Config ──────────────────────────────────────────────────
    INTERACTIONS_PATH = "data/processed/interactions.csv"
    META_PATH         = "data/processed/dataset_meta.json"

    with open(META_PATH) as f:
        meta = json.load(f)
    n_items = meta["n_items"]

    # ── Build components ──────────────────────────────────────────
    bandit = ThompsonSamplingBandit(n_items=n_items)
    store  = BanditStore(host="localhost", port=6379, db=1)
    store.connect()

    # Optionally: flush old simulation data before re-running
    # store.flush_all_arms()

    # Load any existing state first (incremental simulation)
    existing_count = store.load(bandit)
    logger.info(f"Loaded {existing_count} existing arms from Redis")

    # ── Compute synthetic CTR model ───────────────────────────────
    popularity = load_item_popularity(INTERACTIONS_PATH, n_items)
    true_ctr   = compute_true_ctr(popularity)

    # ── Run simulation ────────────────────────────────────────────
    stats = simulate_user_feedback(
        bandit=bandit,
        store=store,
        true_ctr=true_ctr,
        n_users=2000,
        n_recs_per_user=10,
        top_k_pool=100,
        seed=42,
        save_every=500,
    )

    # ── Validate learning ─────────────────────────────────────────
    validate_bandit_learning(bandit, true_ctr, top_n=20)

    # ── Save final stats ──────────────────────────────────────────
    stats_path = Path("mlops/bandits/simulation_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Simulation stats saved to {stats_path}")


if __name__ == "__main__":
    main()