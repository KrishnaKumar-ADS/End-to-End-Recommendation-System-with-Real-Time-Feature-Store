import os
import time
import json
import logging
import shutil
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List

import mlflow
import mlflow.lightgbm
import mlflow.pytorch
import numpy as np
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# GLOBAL MLFLOW CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MLFLOW_DB_ENV = "DS19_MLFLOW_DB_PATH"
DEFAULT_MLFLOW_DB_PATH = PROJECT_ROOT / "mlops" / "experiments" / "mlflow.db"
MLFLOW_DB_PATH = DEFAULT_MLFLOW_DB_PATH
MLFLOW_ARTIFACTS_PATH = PROJECT_ROOT / "mlops" / "experiments" / "artifacts"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

# Experiment names — consistent naming across all training scripts
EXPERIMENTS = {
    "mf":         "ds19_matrix_factorization",
    "sasrec":     "ds19_sasrec_transformer",
    "two_tower":  "ds19_two_tower_retrieval",
    "lgbm":       "ds19_lgbm_ranking",
    "bandits":    "ds19_bandits",
    "ab_testing": "ds19_ab_testing",
    "system":     "ds19_system_metrics",
}

# Model names in the registry — used for staging/production promotion
REGISTRY_MODEL_NAMES = {
    "mf":         "mf_retrieval",
    "sasrec":     "sasrec_retrieval",
    "two_tower":  "two_tower_retrieval",
    "lgbm":       "lgbm_ranker",
}


_MLFLOW_METRIC_NAME_RE = re.compile(r"[^0-9A-Za-z_\-./ ]")


def sanitize_metric_name(metric_name: str) -> str:
    """Convert arbitrary metric names to MLflow-compatible metric keys."""
    normalized = metric_name.replace("@", "_at_")
    normalized = _MLFLOW_METRIC_NAME_RE.sub("_", normalized)
    return normalized


def setup_mlflow() -> None:
    """
    Initialize MLflow for the DS19 project.
    Call this ONCE at the start of any script that uses MLflow.
    
    Creates:
      - mlops/experiments/ directory
      - mlops/experiments/mlflow.db (SQLite backend — enables Model Registry)
      - mlops/experiments/artifacts/ (artifact root)
    """
    global MLFLOW_DB_PATH, MLFLOW_TRACKING_URI

    configured_db_path = Path(
        os.environ.get(MLFLOW_DB_ENV, str(DEFAULT_MLFLOW_DB_PATH))
    )
    MLFLOW_DB_PATH = configured_db_path
    MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

    MLFLOW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow artifact root: {MLFLOW_ARTIFACTS_PATH}")

    # Self-heal stale/incompatible SQLite migration states from older MLflow versions.
    try:
        MlflowClient().search_experiments(max_results=1)
    except Exception as e:
        msg = str(e)
        if "Can't locate revision identified by" in msg:
            ts = int(time.time())
            stale_db_path = MLFLOW_DB_PATH
            backup_path = MLFLOW_DB_PATH.with_name(f"mlflow_stale_{ts}.db")
            moved_stale_db = False
            if MLFLOW_DB_PATH.exists():
                try:
                    shutil.move(str(MLFLOW_DB_PATH), str(backup_path))
                    moved_stale_db = True
                    logger.warning(
                        "Detected incompatible MLflow DB revision. "
                        f"Backed up old DB to: {backup_path}"
                    )
                except PermissionError:
                    fallback_db = MLFLOW_DB_PATH.with_name(
                        f"mlflow_recovered_{ts}.db"
                    )
                    os.environ[MLFLOW_DB_ENV] = str(fallback_db)
                    MLFLOW_DB_PATH = fallback_db
                    MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
                    MLFLOW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                    logger.warning(
                        "Primary MLflow DB is locked by another process. "
                        f"Using fallback DB for this run: {fallback_db}"
                    )

            # Clean possible SQLite sidecar files too.
            if moved_stale_db:
                for suffix in ("-journal", "-wal", "-shm"):
                    sidecar = Path(str(stale_db_path) + suffix)
                    if sidecar.exists():
                        sidecar.unlink()

            # Re-create a clean DB by touching the store once.
            MlflowClient().search_experiments(max_results=1)
            logger.info("Initialized new MLflow database after stale revision recovery")
        else:
            raise


def get_or_create_experiment(component: str) -> str:
    """
    Get or create an MLflow experiment for a given component.
    
    Args:
        component: One of the keys in EXPERIMENTS dict
                   ("mf", "sasrec", "two_tower", "lgbm", etc.)
    
    Returns:
        experiment_id string
    """
    setup_mlflow()
    experiment_name = EXPERIMENTS.get(component, f"ds19_{component}")
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    logger.info(f"Using experiment: '{experiment_name}' (id={experiment.experiment_id})")
    return experiment.experiment_id


@contextmanager
def start_run(component: str, run_name: Optional[str] = None, tags: Optional[Dict] = None):
    """
    Context manager for MLflow runs. Handles setup and cleanup.
    
    Usage:
        with start_run("lgbm", run_name="lgbm_v3_lambdarank") as run:
            mlflow.log_param("n_estimators", 1000)
            mlflow.log_metric("ndcg@10", 0.847)
            mlflow.lightgbm.log_model(lgb_model, "lgbm_ranker")
            print(f"Run ID: {run.info.run_id}")
    """
    get_or_create_experiment(component)
    default_tags = {
        "dataset":    "MovieLens25M",
        "project":    "DS19",
        "component":  component,
        "week":       "8",
    }
    if tags:
        default_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=default_tags) as run:
        logger.info(
            f"MLflow run started | "
            f"experiment={EXPERIMENTS.get(component)} | "
            f"run_id={run.info.run_id[:8]}..."
        )
        yield run
        logger.info(f"MLflow run completed | run_id={run.info.run_id[:8]}...")


def log_params(params: Dict[str, Any]) -> None:
    """Log a dictionary of parameters. Flattens nested dicts."""
    flat_params = {}
    for k, v in params.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_params[f"{k}.{sub_k}"] = sub_v
        else:
            flat_params[k] = v
    mlflow.log_params(flat_params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log a dictionary of metrics."""
    safe_metrics: Dict[str, float] = {}
    for metric_name, metric_val in metrics.items():
        if isinstance(metric_val, (int, float, np.integer, np.floating)):
            if isinstance(metric_val, float) and np.isnan(metric_val):
                continue
            safe_name = sanitize_metric_name(str(metric_name))
            safe_metrics[safe_name] = float(metric_val)

    if safe_metrics:
        mlflow.log_metrics(safe_metrics, step=step)


def log_evaluation_report(metrics: Dict, filepath: str = "evaluation_report.json") -> None:
    """Log evaluation metrics as both MLflow metrics AND a JSON artifact."""
    # Log individual metrics
    numeric_metrics: Dict[str, float] = {}
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            numeric_metrics[key] = float(val)
    log_metrics(numeric_metrics)

    # Log full report as artifact
    tmp_path = Path(f"/tmp/{filepath}")
    with open(tmp_path, "w") as f:
        json.dump(metrics, f, indent=2)
    mlflow.log_artifact(str(tmp_path), "evaluation")
    tmp_path.unlink(missing_ok=True)


def get_best_run(component: str, metric: str = "ndcg@10", mode: str = "max") -> Optional[Dict]:
    """
    Find the best run for a given component by a specific metric.
    
    Args:
        component: e.g., "lgbm"
        metric:    e.g., "ndcg@10"
        mode:      "max" (higher is better) or "min" (lower is better)
    
    Returns:
        Dict with run info, or None if no runs exist.
    """
    setup_mlflow()
    experiment_name = EXPERIMENTS.get(component, f"ds19_{component}")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Experiment '{experiment_name}' not found")
        return None

    safe_metric = sanitize_metric_name(metric)
    order_by = f"metrics.{safe_metric} {'DESC' if mode == 'max' else 'ASC'}"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{safe_metric} > 0",
        order_by=[order_by],
        max_results=1,
    )

    if runs.empty:
        logger.warning(f"No runs found for experiment '{experiment_name}'")
        return None

    best_run = runs.iloc[0]
    return {
        "run_id":    best_run["run_id"],
        "run_name":  best_run.get("tags.mlflow.runName", ""),
        "metric":    metric,
        "value":     best_run.get(f"metrics.{safe_metric}", None),
        "params":    {
            k.replace("params.", ""): v
            for k, v in best_run.items()
            if k.startswith("params.")
        },
    }


def print_experiment_summary(component: str) -> None:
    """Print a formatted summary of all runs for a component."""
    setup_mlflow()
    experiment_name = EXPERIMENTS.get(component, f"ds19_{component}")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"No experiment found for: {component}")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=10,
    )

    if runs.empty:
        print(f"No runs found in experiment: {experiment_name}")
        return

    print(f"\n{'='*70}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Total Runs: {len(runs)}")
    print(f"{'='*70}")

    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    for _, row in runs.iterrows():
        print(f"\n  Run: {row.get('tags.mlflow.runName', row['run_id'][:8])}")
        print(f"    ID:     {row['run_id'][:16]}...")
        print(f"    Status: {row['status']}")
        print(f"    Start:  {row['start_time']}")
        for col in metric_cols[:5]:  # Show top 5 metrics
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                metric_name = col.replace("metrics.", "")
                print(f"    {metric_name:20s}: {val:.4f}")