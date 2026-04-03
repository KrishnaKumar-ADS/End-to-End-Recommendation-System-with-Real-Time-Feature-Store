import json
import sys
import time
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlops.ab_testing.ab_router import ABRouter
from mlops.ab_testing.ab_logger import ABLogger
from mlops.ab_testing.ab_analyzer import ABAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def simulate_mf_recommendations(user_id: int, n: int = 10) -> list:
    """
    Simulate MF model recommendations.
    Returns a list of item_ids that MF would recommend.
    (In the real system, this calls the actual MF model.)
    """
    rng = np.random.default_rng(seed=user_id * 100)
    # MF tends to recommend popular items → sample from top-5000
    return rng.choice(5000, size=n, replace=False).tolist()


def simulate_sasrec_recommendations(user_id: int, n: int = 10) -> list:
    """
    Simulate SASRec model recommendations.
    SASRec is more personalized → slightly better long-tail coverage.
    """
    rng = np.random.default_rng(seed=user_id * 200)
    # SASRec has broader coverage → sample from top-8000
    return rng.choice(8000, size=n, replace=False).tolist()


def simulate_click_behavior(
    user_id: int,
    item_ids: list,
    variant: str,
    base_ctr_mf: float = 0.08,
    base_ctr_sasrec: float = 0.11,
    seed_offset: int = 0,
) -> list:
    """
    Simulate which items the user clicks on.
    
    SASRec has a higher base CTR (0.11) vs MF (0.08) to simulate
    that the transformer model is better at personalization.
    This creates a realistic scenario where B (SASRec) wins the A/B test.
    
    Returns list of clicked item_ids.
    """
    rng = np.random.default_rng(seed=user_id + seed_offset)
    base_ctr = base_ctr_sasrec if variant == "sasrec" else base_ctr_mf

    clicked = []
    for item_id in item_ids:
        # Add item-level CTR variation (some items are more clickable)
        item_ctr = base_ctr * (0.5 + rng.random())
        item_ctr = min(item_ctr, 0.95)  # cap at 95%
        if rng.random() < item_ctr:
            clicked.append(item_id)
    return clicked


def run_ab_simulation(
    experiment_id: str = "retrieval_v1",
    n_users: int = 2000,
    n_recs: int = 10,
    seed: int = 42,
    reset_logs: bool = True,
) -> dict:
    """
    Run the full A/B test simulation.
    
    Args:
        experiment_id: Which experiment config to use
        n_users:       Number of simulated user sessions
        n_recs:        Items recommended per session
        seed:          Base random seed
    
    Returns:
        Summary statistics of the simulation
    """
    router  = ABRouter()
    ab_log  = ABLogger()
    rng     = np.random.default_rng(seed)

    # Use all available user_ids (or a random sample)
    user_ids = list(range(1, n_users + 1))

    logger.info(f"Starting A/B simulation: experiment='{experiment_id}' n_users={n_users}")

    if reset_logs:
        # Keep this run statistically clean by clearing prior simulation logs.
        ab_log.exposure_path.write_text("", encoding="utf-8")
        ab_log.conversion_path.write_text("", encoding="utf-8")
        logger.info(
            f"Reset A/B logs: {ab_log.exposure_path} and {ab_log.conversion_path}"
        )

    # Check actual split distribution
    distribution = router.get_variant_distribution(user_ids, experiment_id)
    logger.info(f"Variant distribution: {distribution}")

    summary = {
        "experiment_id":  experiment_id,
        "n_users":        n_users,
        "distribution":   distribution,
        "exposures_by_variant": {},
        "conversions_by_variant": {},
        "ctr_by_variant": {},
    }

    exposures_by_v:   dict = {}
    conversions_by_v: dict = {}

    for user_id in user_ids:
        # Assign variant
        variant = router.assign(user_id, experiment_id)

        # Get recommendations
        if variant == "mf":
            recs = simulate_mf_recommendations(user_id, n=n_recs)
        else:
            recs = simulate_sasrec_recommendations(user_id, n=n_recs)

        # Log exposure
        ab_log.log_exposure(
            user_id=user_id,
            variant=variant,
            experiment_id=experiment_id,
            n_recs=n_recs,
            extra={"simulated": True},
        )
        exposures_by_v[variant] = exposures_by_v.get(variant, 0) + 1

        # Simulate clicks
        clicked_items = simulate_click_behavior(
            user_id=user_id,
            item_ids=recs,
            variant=variant,
            base_ctr_mf=0.08,
            base_ctr_sasrec=0.11,
            seed_offset=seed,
        )

        # Session-level conversion: at most one conversion event per exposure.
        session_clicked = len(clicked_items) > 0
        if session_clicked:
            ab_log.log_conversion(
                user_id=user_id,
                item_id=clicked_items[0],
                variant=variant,
                experiment_id=experiment_id,
                extra={
                    "simulated": True,
                    "clicked_items_count": len(clicked_items),
                },
            )
        conversions_by_v[variant] = (
            conversions_by_v.get(variant, 0) + int(session_clicked)
        )

    # Compute CTR per variant
    for variant in exposures_by_v:
        n_exp  = exposures_by_v[variant]
        n_conv = conversions_by_v.get(variant, 0)
        ctr    = n_conv / n_exp if n_exp > 0 else 0.0
        summary["ctr_by_variant"][variant] = round(ctr, 4)

    summary["exposures_by_variant"]   = exposures_by_v
    summary["conversions_by_variant"] = conversions_by_v

    logger.info("=" * 60)
    logger.info("SIMULATION COMPLETE")
    for variant in exposures_by_v:
        exp  = exposures_by_v[variant]
        conv = conversions_by_v.get(variant, 0)
        ctr  = summary["ctr_by_variant"][variant]
        logger.info(f"  {variant:10s}: {exp:5d} exposures | {conv:4d} conversions | CTR={ctr:.4f}")
    logger.info("=" * 60)

    return summary


def run_and_analyze():
    """Run simulation then immediately analyze and print results."""
    # Step 1: Simulate
    summary = run_ab_simulation(
        experiment_id="retrieval_v1",
        n_users=2000,
        n_recs=10,
        seed=42,
    )

    print(f"\n  Simulation summary: {json.dumps(summary['ctr_by_variant'], indent=4)}")

    # Step 2: Analyze
    analyzer = ABAnalyzer()
    result = analyzer.analyze(
        experiment_id="retrieval_v1",
        control="mf",
        treatment="sasrec",
        confidence_level=0.95,
    )
    result.print_summary()

    # Step 3: Save results
    result_path = analyzer.save_results(result)
    print(f"\n  Results saved: {result_path}")

    return result


if __name__ == "__main__":
    run_and_analyze()