import redis
import logging
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mlops.bandits.bandit_engine import ThompsonSamplingBandit, BanditArm

from mlops.bandits.bandit_engine import BanditArm

logger = logging.getLogger(__name__)

REDIS_KEY_PREFIX = "bandit:arm:"
REDIS_STATS_KEY  = "bandit:metadata"


class BanditStore:
    """
    Persists and restores bandit arm state in Redis.
    
    Lifecycle:
      1. App startup: BanditStore.load(bandit) → restores all arms
      2. Each /feedback call: BanditStore.persist_arm(arm)
      3. Periodic sync: BanditStore.save_all(bandit) → bulk write
      4. Health check: BanditStore.health_check()
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 1):
        """
        Args:
            host: Redis host
            port: Redis port
            db:   Redis database index (use db=1 to separate from Feast's db=0)
        """
        self.host = host
        self.port = port
        self.db   = db
        self._client: Optional[redis.Redis] = None
        logger.info(f"BanditStore configured | redis={host}:{port} db={db}")

    def connect(self) -> None:
        """Initialize Redis connection. Call once on startup."""
        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True,       # auto-decode bytes → str
            socket_connect_timeout=5,
            socket_timeout=2,
            retry_on_timeout=True,
        )
        self._client.ping()
        logger.info("BanditStore connected to Redis successfully")

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self.connect()
        return self._client

    def _arm_key(self, item_idx: int) -> str:
        return f"{REDIS_KEY_PREFIX}{item_idx}"

    def persist_arm(self, arm: "BanditArm") -> None:
        """
        Persist a single arm to Redis after an update.
        Called on every /feedback event.

        Stored as Redis hash fields:
          alpha -> float
          beta  -> float
        """
        key = self._arm_key(arm.item_idx)
        self.client.hset(
            key,
            mapping={
                "alpha": float(arm.alpha),
                "beta": float(arm.beta),
            },
        )

    def load_arm(self, item_idx: int) -> Optional["BanditArm"]:
        """
        Load a single arm from Redis.
        Returns None if arm not found (new item → start with prior).
        """
        key = self._arm_key(item_idx)

        # Preferred format (Week 9): Redis hash fields alpha/beta.
        hash_value = self.client.hgetall(key)
        if hash_value:
            try:
                return BanditArm(
                    item_idx=item_idx,
                    alpha=float(hash_value.get("alpha", 1.0)),
                    beta=float(hash_value.get("beta", 1.0)),
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Corrupt hash arm data for item={item_idx}: {hash_value} | {e}")

        # Backward-compatible format (Week 8): "alpha:beta" string.
        value = self.client.get(key)
        if value is None:
            return None
        try:
            alpha_str, beta_str = value.split(":")
            return BanditArm(
                item_idx=item_idx,
                alpha=float(alpha_str),
                beta=float(beta_str),
            )
        except (ValueError, AttributeError) as e:
            logger.warning(f"Corrupt arm data for item={item_idx}: {value} | {e}")
            return None

    def load(self, bandit: "ThompsonSamplingBandit") -> int:
        """
        Load ALL persisted arms from Redis into the bandit instance.
        Call on application startup.
        
        Returns:
            Number of arms loaded.
        """
        pattern = f"{REDIS_KEY_PREFIX}*"
        keys = self.client.keys(pattern)

        loaded = 0
        failed = 0
        arms_to_load: Dict[int, BanditArm] = {}

        for key in keys:
            try:
                item_idx = int(key.replace(REDIS_KEY_PREFIX, ""))
                arm = self.load_arm(item_idx)
                if arm is not None:
                    arms_to_load[item_idx] = arm
                    loaded += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse key: {key} | {e}")
                failed += 1

        bandit.load_arms(arms_to_load)
        logger.info(
            f"BanditStore.load complete | loaded={loaded} arms "
            f"| failed={failed} | redis_db={self.db}"
        )
        return loaded

    def save_all(self, bandit: "ThompsonSamplingBandit") -> int:
        """
        Bulk-save all arms from the bandit to Redis.
        Use after batch updates or for periodic checkpointing.
        
        Uses Redis pipeline for efficiency (sends all commands at once).
        
        Returns:
            Number of arms saved.
        """
        pipe = self.client.pipeline(transaction=False)
        count = 0
        for item_idx, arm in bandit._arms.items():
            key = self._arm_key(item_idx)
            pipe.hset(
                key,
                mapping={
                    "alpha": float(arm.alpha),
                    "beta": float(arm.beta),
                },
            )
            count += 1

        if count > 0:
            pipe.execute()
        logger.info(f"BanditStore.save_all complete | saved={count} arms")
        return count

    def delete_arm(self, item_idx: int) -> None:
        """Reset a single arm back to prior Beta(1,1). Useful for testing."""
        self.client.delete(self._arm_key(item_idx))
        logger.info(f"BanditStore: deleted arm for item={item_idx}")

    def flush_all_arms(self) -> None:
        """
        Delete ALL bandit arm data from Redis.
        WARNING: This resets the entire bandit to prior state.
        Only use for testing or complete system reset.
        """
        pattern = f"{REDIS_KEY_PREFIX}*"
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)
        logger.warning(f"BanditStore: flushed {len(keys)} arms from Redis")

    def get_total_arms(self) -> int:
        """Return total number of arms currently persisted in Redis."""
        pattern = f"{REDIS_KEY_PREFIX}*"
        return len(self.client.keys(pattern))

    def health_check(self) -> Dict:
        """
        Check Redis connection and return store health metrics.
        Called by FastAPI /health endpoint.
        """
        try:
            self.client.ping()
            info = self.client.info("memory")
            total_arms = self.get_total_arms()
            return {
                "status": "healthy",
                "redis_host": f"{self.host}:{self.port}",
                "redis_db": self.db,
                "arms_persisted": total_arms,
                "redis_used_memory_mb": round(
                    info.get("used_memory", 0) / 1024 / 1024, 2
                ),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}