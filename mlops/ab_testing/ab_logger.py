import json
import time
import threading
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ABLogger:
    """
    Thread-safe logger for A/B test exposures and conversions.
    
    Usage:
      ab_logger = ABLogger()
      ab_logger.log_exposure(user_id=42, variant="sasrec", experiment_id="retrieval_v1")
      # ... user gets recommendations ...
      ab_logger.log_conversion(user_id=42, item_id=4886, variant="sasrec", experiment_id="retrieval_v1")
    """

    def __init__(
        self,
        exposure_path:   str = "mlops/ab_testing/exposures.jsonl",
        conversion_path: str = "mlops/ab_testing/conversions.jsonl",
    ):
        self.exposure_path   = Path(exposure_path)
        self.conversion_path = Path(conversion_path)
        self._lock = threading.Lock()

        # Ensure directories exist
        self.exposure_path.parent.mkdir(parents=True, exist_ok=True)
        self.conversion_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ABLogger initialized | "
            f"exposures={self.exposure_path} | "
            f"conversions={self.conversion_path}"
        )

    def log_exposure(
        self,
        user_id:       int,
        variant:       str,
        experiment_id: str,
        n_recs:        int = 10,
        extra:         Optional[dict] = None,
    ) -> None:
        """
        Log that a user was shown recommendations from a specific model variant.
        Called every time GET /recommend is served.
        
        Args:
            user_id:       User integer index
            variant:       Variant name ("mf" or "sasrec")
            experiment_id: Which experiment ("retrieval_v1")
            n_recs:        Number of recommendations shown
            extra:         Optional additional fields (latency_ms, etc.)
        """
        entry = {
            "user_id":       user_id,
            "variant":       variant,
            "experiment_id": experiment_id,
            "n_recs":        n_recs,
            "ts":            time.time(),
        }
        if extra:
            entry.update(extra)

        self._append(self.exposure_path, entry)

    def log_conversion(
        self,
        user_id:       int,
        item_id:       int,
        variant:       str,
        experiment_id: str,
        extra:         Optional[dict] = None,
    ) -> None:
        """
        Log that a user clicked on a specific item.
        Called every time POST /feedback is received with clicked=true.
        
        Args:
            user_id:       User integer index
            item_id:       Item that was clicked
            variant:       Variant the user was assigned to
            experiment_id: Which experiment
            extra:         Optional additional fields
        """
        entry = {
            "user_id":       user_id,
            "item_id":       item_id,
            "variant":       variant,
            "experiment_id": experiment_id,
            "ts":            time.time(),
        }
        if extra:
            entry.update(extra)

        self._append(self.conversion_path, entry)

    def _append(self, path: Path, entry: dict) -> None:
        """Thread-safe append of a JSON line to a file."""
        with self._lock:
            try:
                with open(path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except IOError as e:
                logger.error(f"Failed to write A/B log to {path}: {e}")

    def get_exposure_count(self, experiment_id: Optional[str] = None) -> int:
        """Count total exposures logged (optionally filtered by experiment)."""
        return self._count_lines(self.exposure_path, experiment_id)

    def get_conversion_count(self, experiment_id: Optional[str] = None) -> int:
        """Count total conversions logged (optionally filtered by experiment)."""
        return self._count_lines(self.conversion_path, experiment_id)

    def _count_lines(self, path: Path, experiment_id: Optional[str] = None) -> int:
        if not path.exists():
            return 0
        count = 0
        try:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        if experiment_id:
                            entry = json.loads(line)
                            if entry.get("experiment_id") == experiment_id:
                                count += 1
                        else:
                            count += 1
        except IOError:
            pass
        return count