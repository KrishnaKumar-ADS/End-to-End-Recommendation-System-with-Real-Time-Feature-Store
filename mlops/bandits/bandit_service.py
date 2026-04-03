import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from mlops.bandits.bandit_engine import ThompsonSamplingBandit, BanditArm
from mlops.bandits.bandit_store import BanditStore

logger = logging.getLogger(__name__)


class BanditService:
    """
    Facade for the Thompson Sampling bandit system.
    
    Initialize once on FastAPI startup:
      bandit_service = BanditService(n_items=53889)
      bandit_service.startup()   ← connects Redis, loads arms
    
    Use per request:
      ranked = bandit_service.rerank(top_k_item_ids, lgbm_scores)
    
    Use per feedback event:
      bandit_service.record_feedback(item_id, clicked=True)
    """

    def __init__(
        self,
        n_items: int,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 1,
        blend_weight: float = 0.3,
    ):
        """
        Args:
            n_items:      Total catalog size (for coverage tracking)
            redis_host:   Redis hostname
            redis_port:   Redis port
            redis_db:     Redis DB index (use 1 to separate from Feast)
            blend_weight: 0.0 = pure bandit, 1.0 = pure LightGBM, 0.3 = default blend
        """
        self.n_items       = n_items
        self.blend_weight  = blend_weight
        self.bandit        = ThompsonSamplingBandit(n_items=n_items)
        self.store         = BanditStore(
            host=redis_host, port=redis_port, db=redis_db
        )
        self._initialized  = False
        self._feedback_log_path = Path("mlops/ab_testing/bandit_feedback.jsonl")

    def startup(self) -> Dict:
        """
        Initialize the bandit system.
        Call this in FastAPI's lifespan/startup event.
        
        Steps:
          1. Connect to Redis
          2. Load all previously learned arm states from Redis
          3. Report how many arms were restored
        """
        t0 = time.time()
        self.store.connect()
        arms_loaded = self.store.load(self.bandit)
        elapsed = (time.time() - t0) * 1000

        self._initialized = True
        self._feedback_log_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "status": "initialized",
            "arms_loaded_from_redis": arms_loaded,
            "startup_time_ms": round(elapsed, 2),
            "blend_weight": self.blend_weight,
        }
        logger.info(f"BanditService startup: {summary}")
        return summary

    def rerank(
        self,
        candidate_item_ids: List[int],
        lgbm_scores: Optional[List[float]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank top-K candidates using Thompson Sampling.
        
        Args:
            candidate_item_ids: Item indices from LightGBM (ordered best first)
            lgbm_scores:        Corresponding LightGBM predicted scores
        
        Returns:
            List of (item_idx, bandit_score) tuples, sorted descending.
        
        IMPORTANT: This is called in the hot path (every recommendation request).
        Keep it fast. No Redis reads here — arms are in-memory.
        """
        if not self._initialized:
            # Graceful degradation: preserve base ranker scores when bandit is unavailable.
            logger.warning("BanditService not initialized — returning base ranking scores")

            if lgbm_scores is not None and len(lgbm_scores) == len(candidate_item_ids):
                return [
                    (int(item_id), float(score))
                    for item_id, score in zip(candidate_item_ids, lgbm_scores)
                ]

            total = max(len(candidate_item_ids), 1)
            return [
                (int(item_id), float(total - rank_idx) / float(total))
                for rank_idx, item_id in enumerate(candidate_item_ids)
            ]

        return self.bandit.rerank(
            candidate_item_ids=candidate_item_ids,
            lgbm_scores=lgbm_scores,
            blend_weight=self.blend_weight,
        )

    def record_feedback(self, item_idx: int, clicked: bool) -> None:
        """
        Record user feedback and update the corresponding bandit arm.
        
        Called by POST /feedback endpoint.
        
        Steps:
          1. Update in-memory arm (immediate effect on future recommendations)
          2. Persist updated arm to Redis (survives restarts)
          3. Log feedback event to JSONL (for analysis)
        
        Args:
            item_idx: The item index that was interacted with
            clicked:  True if user clicked, False if user ignored
        """
        reward = 1 if clicked else 0
        self.bandit.update(item_idx, reward)

        # Persist to Redis (async in production — sync here for simplicity)
        arm = self.bandit.get_arm(item_idx)
        self.store.persist_arm(arm)

        # Log for offline analysis
        log_entry = {
            "timestamp": time.time(),
            "item_idx": item_idx,
            "reward": reward,
            "arm_alpha": arm.alpha,
            "arm_beta": arm.beta,
            "expected_ctr": round(arm.expected_ctr, 4),
        }
        self._append_log(log_entry)

    def _append_log(self, entry: Dict) -> None:
        """Append a JSON line to the feedback log."""
        try:
            with open(self._feedback_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError as e:
            logger.warning(f"Failed to write bandit feedback log: {e}")

    def get_arm_info(self, item_idx: int) -> Dict:
        """Return info about a specific item's bandit arm. For debugging/API."""
        arm = self.bandit.get_arm(item_idx)
        return arm.to_dict()

    def get_system_stats(self) -> Dict:
        """Return overall bandit system statistics."""
        bandit_stats = self.bandit.get_stats()
        store_health = self.store.health_check()
        return {
            "bandit": bandit_stats,
            "store": store_health,
            "blend_weight": self.blend_weight,
            "initialized": self._initialized,
        }

    def top_items(self, n: int = 20) -> List[Dict]:
        """Return top N items by expected CTR. For the MLflow dashboard."""
        return self.bandit.top_items_by_ctr(n=n)

    def bulk_save(self) -> int:
        """Force-save all arms to Redis. Call on shutdown or periodically."""
        return self.store.save_all(self.bandit)