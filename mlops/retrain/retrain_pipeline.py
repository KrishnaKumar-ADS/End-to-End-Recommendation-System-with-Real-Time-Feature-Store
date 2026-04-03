import os
import sys
import time
import json
import argparse
import mlflow
import mlflow.pytorch
import structlog
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = structlog.get_logger()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
MODELS_DIR = ROOT / "models" / "saved"
DATA_DIR = ROOT / "data" / "processed"

# Quality gate: new model must beat this NDCG@10 to be promoted
NDCG_QUALITY_GATE = 0.80

# Minimum improvement: new model must be at least this much better
# than production to justify a model swap (avoids noisy swaps)
MIN_IMPROVEMENT_THRESHOLD = 0.01   # 1% relative improvement required


def parse_args():
    parser = argparse.ArgumentParser(description="DS19 Retraining Pipeline")
    parser.add_argument(
        "--run-id",
        default="auto",
        help="Run identifier (used in MLflow experiment name)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if quality gate would normally block"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mf", "sasrec", "two_tower", "lgbm"],
        help="Which models to retrain"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline but don't promote to production"
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# STAGE 1: DATA CHECK
# ══════════════════════════════════════════════════════════════════════

def stage_data_check() -> dict:
    """
    Verify that enough new data is available to justify retraining.

    Returns metadata about the current dataset.
    Raises RuntimeError if data is insufficient.
    """
    logger.info("stage_start", stage="data_check")

    interactions_path = DATA_DIR / "interactions.csv"
    if not interactions_path.exists():
        raise RuntimeError(f"interactions.csv not found at {interactions_path}")

    import pandas as pd
    df = pd.read_csv(interactions_path)

    n_interactions = len(df)
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()

    # Check for minimum data volume
    MIN_INTERACTIONS = 100_000
    if n_interactions < MIN_INTERACTIONS:
        raise RuntimeError(
            f"Insufficient data: {n_interactions} interactions "
            f"(minimum required: {MIN_INTERACTIONS})"
        )

    # Check data freshness (if timestamp column exists)
    timestamp_col = None
    for col in ["timestamp", "ts", "time", "date"]:
        if col in df.columns:
            timestamp_col = col
            break

    freshness_info = {}
    if timestamp_col:
        max_ts = pd.to_datetime(df[timestamp_col], unit="s").max()
        days_old = (datetime.now() - max_ts).days
        freshness_info = {
            "latest_interaction": str(max_ts),
            "data_age_days": days_old,
        }
        logger.info("data_freshness", **freshness_info)

    meta = {
        "n_interactions": n_interactions,
        "n_users": n_users,
        "n_items": n_items,
        **freshness_info,
    }

    logger.info("stage_complete", stage="data_check", **meta)
    return meta


# ══════════════════════════════════════════════════════════════════════
# STAGE 2: RETRAIN MODELS
# ══════════════════════════════════════════════════════════════════════

def stage_retrain_model(model_name: str, experiment_name: str) -> str:
    """
    Retrain a single model by calling its training script.
    Returns the MLflow run_id for the new model.

    Architecture decision: Each model has its own training script
    (train_mf.py, train_sasrec.py, etc.) that logs to MLflow.
    The retraining pipeline orchestrates them.
    """
    logger.info("stage_start", stage=f"retrain_{model_name}")

    script_map = {
        "mf":         "models/training/train_mf.py",
        "sasrec":     "models/training/train_sasrec.py",
        "two_tower":  "models/training/train_two_tower.py",
        "lgbm":       "models/training/train_lgbm.py",
    }

    script = ROOT / script_map[model_name]
    if not script.exists():
        raise RuntimeError(f"Training script not found: {script}")

    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "MLFLOW_EXPERIMENT_NAME": experiment_name,
        "RETRAINING_MODE": "1",   # signal to scripts to use retraining config
    }

    start_time = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        logger.error(
            "training_failed",
            model=model_name,
            returncode=result.returncode,
            stderr=result.stderr[-500:],   # last 500 chars of error
        )
        raise RuntimeError(f"{model_name} training failed:\n{result.stderr[-500:]}")

    # Parse the MLflow run_id from the training script's stdout
    # Convention: training scripts print "MLFLOW_RUN_ID: <uuid>" at end
    run_id = None
    for line in result.stdout.splitlines():
        if line.startswith("MLFLOW_RUN_ID:"):
            run_id = line.split(":", 1)[1].strip()
            break

    if not run_id:
        logger.warning(
            "no_run_id_found",
            model=model_name,
            stdout_tail=result.stdout[-200:],
        )
        # Fall back: get the latest run from MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                run_id = runs[0].info.run_id

    logger.info(
        "stage_complete",
        stage=f"retrain_{model_name}",
        run_id=run_id,
        elapsed_seconds=round(elapsed, 1),
    )
    return run_id


# ══════════════════════════════════════════════════════════════════════
# STAGE 3: MODEL VALIDATION
# ══════════════════════════════════════════════════════════════════════

def stage_validate_model(model_name: str, run_id: str) -> dict:
    """
    Fetch evaluation metrics from the MLflow run.
    Returns metrics dict: {ndcg_10, hit_rate_10, mrr}
    """
    logger.info("stage_start", stage=f"validate_{model_name}", run_id=run_id)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics

    # Expected metric keys from training scripts
    ndcg = metrics.get("ndcg_10", metrics.get("ndcg@10", 0.0))
    hr = metrics.get("hit_rate_10", metrics.get("hr@10", 0.0))
    mrr = metrics.get("mrr", 0.0)

    result = {"ndcg_10": ndcg, "hit_rate_10": hr, "mrr": mrr}
    logger.info("validation_metrics", model=model_name, **result)
    return result


def stage_compare_with_production(
    model_name: str,
    new_metrics: dict
) -> dict:
    """
    Compare new model metrics with the current Production model in MLflow registry.
    Returns comparison dict.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Try to get Production model metrics
    try:
        prod_versions = client.get_latest_versions(
            name=f"ds19_{model_name}",
            stages=["Production"],
        )
        if not prod_versions:
            logger.info("no_production_model", model=model_name)
            return {"has_production": False, "new_is_better": True}

        prod_run_id = prod_versions[0].run_id
        prod_run = client.get_run(prod_run_id)
        prod_metrics = prod_run.data.metrics

        prod_ndcg = prod_metrics.get("ndcg_10", prod_metrics.get("ndcg@10", 0.0))
        new_ndcg = new_metrics["ndcg_10"]

        improvement = (new_ndcg - prod_ndcg) / max(prod_ndcg, 1e-6)

        comparison = {
            "has_production": True,
            "production_ndcg": prod_ndcg,
            "new_ndcg": new_ndcg,
            "improvement_pct": round(improvement * 100, 2),
            "new_is_better": (
                new_ndcg >= NDCG_QUALITY_GATE and
                improvement >= MIN_IMPROVEMENT_THRESHOLD
            ),
        }

        logger.info("model_comparison", model=model_name, **comparison)
        return comparison

    except Exception as e:
        logger.warning("comparison_error", model=model_name, error=str(e))
        return {"has_production": False, "new_is_better": True}


# ══════════════════════════════════════════════════════════════════════
# STAGE 4: PROMOTE TO PRODUCTION
# ══════════════════════════════════════════════════════════════════════

def stage_promote_model(model_name: str, run_id: str) -> bool:
    """
    Transition model in MLflow registry to "Production" stage.
    Archives the previous Production model.
    """
    logger.info("stage_start", stage=f"promote_{model_name}", run_id=run_id)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    registry_name = f"ds19_{model_name}"

    # Check if model is registered. If not, register it.
    try:
        client.get_registered_model(registry_name)
    except Exception:
        client.create_registered_model(
            name=registry_name,
            description=f"DS19 {model_name} model — auto-registered by retraining pipeline"
        )

    # Create new version from run_id
    # The model artifact must have been logged via mlflow.log_model() in training script
    artifact_path = f"models/saved/{model_name}_best"
    model_version = client.create_model_version(
        name=registry_name,
        source=f"{MLFLOW_TRACKING_URI}/artifacts/{run_id}/{artifact_path}",
        run_id=run_id,
    )

    # Archive previous Production versions
    prod_versions = client.get_latest_versions(
        name=registry_name, stages=["Production"]
    )
    for old_version in prod_versions:
        client.transition_model_version_stage(
            name=registry_name,
            version=old_version.version,
            stage="Archived",
        )
        logger.info(
            "model_archived",
            model=model_name,
            version=old_version.version,
        )

    # Promote new version to Production
    client.transition_model_version_stage(
        name=registry_name,
        version=model_version.version,
        stage="Production",
    )

    logger.info(
        "stage_complete",
        stage=f"promote_{model_name}",
        version=model_version.version,
    )
    return True


# ══════════════════════════════════════════════════════════════════════
# STAGE 5: RELOAD MODELS IN FASTAPI (HOT RELOAD)
# ══════════════════════════════════════════════════════════════════════

def stage_reload_fastapi() -> bool:
    """
    Signal FastAPI to reload models from MLflow registry.
    Uses the /admin/reload endpoint (you'll add this to main.py).
    No service restart needed — zero-downtime model update.
    """
    import urllib.request, urllib.error

    logger.info("stage_start", stage="reload_fastapi")

    try:
        req = urllib.request.Request(
            f"{FASTAPI_URL}/admin/reload-models",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        response = urllib.request.urlopen(req, timeout=60)
        body = json.loads(response.read())

        logger.info(
            "stage_complete",
            stage="reload_fastapi",
            result=body,
        )
        return True
    except urllib.error.URLError as e:
        logger.warning(
            "fastapi_reload_failed",
            error=str(e),
            note="FastAPI may need a manual restart to pick up new models",
        )
        return False


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    pipeline_start = time.time()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"ds19_retraining_{args.run_id}_{timestamp}"

    logger.info(
        "pipeline_start",
        experiment=experiment_name,
        models=args.models,
        dry_run=args.dry_run,
        force=args.force,
    )

    results = {
        "experiment": experiment_name,
        "timestamp": timestamp,
        "models": {},
        "promoted": [],
        "failed": [],
        "skipped": [],
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # ── STAGE 1: Data Check ─────────────────────────────────────────
    try:
        data_meta = stage_data_check()
        results["data_meta"] = data_meta
    except RuntimeError as e:
        logger.error("data_check_failed", error=str(e))
        notify_slack(
            title="❌ Retraining Failed — Data Check",
            message=str(e),
            color="danger",
        )
        sys.exit(1)

    # ── STAGES 2-5: Per-Model Pipeline ─────────────────────────────
    for model_name in args.models:
        model_result = {"model": model_name, "status": "pending"}

        try:
            # Stage 2: Retrain
            run_id = stage_retrain_model(model_name, experiment_name)
            model_result["run_id"] = run_id

            # Stage 3: Validate
            new_metrics = stage_validate_model(model_name, run_id)
            model_result["metrics"] = new_metrics

            # Quality gate check
            if new_metrics["ndcg_10"] < NDCG_QUALITY_GATE and not args.force:
                logger.warning(
                    "quality_gate_failed",
                    model=model_name,
                    ndcg=new_metrics["ndcg_10"],
                    gate=NDCG_QUALITY_GATE,
                )
                model_result["status"] = "quality_gate_failed"
                results["skipped"].append(model_name)
                continue

            # Compare with production
            comparison = stage_compare_with_production(model_name, new_metrics)
            model_result["comparison"] = comparison

            if not comparison.get("new_is_better") and not args.force:
                logger.info(
                    "model_not_better",
                    model=model_name,
                    improvement_pct=comparison.get("improvement_pct"),
                )
                model_result["status"] = "not_improved"
                results["skipped"].append(model_name)
                continue

            # Stage 4: Promote (unless dry-run)
            if not args.dry_run:
                stage_promote_model(model_name, run_id)
                model_result["status"] = "promoted"
                results["promoted"].append(model_name)
            else:
                model_result["status"] = "dry_run_skip"
                logger.info("dry_run_skip", model=model_name)

        except Exception as e:
            logger.error("model_pipeline_failed", model=model_name, error=str(e))
            model_result["status"] = "error"
            model_result["error"] = str(e)
            results["failed"].append(model_name)

        results["models"][model_name] = model_result

    # ── STAGE 5: Reload FastAPI (if any models promoted) ────────────
    if results["promoted"] and not args.dry_run:
        stage_reload_fastapi()

    # ── FINAL: Save pipeline results ────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    results["total_elapsed_seconds"] = round(total_elapsed, 1)

    results_path = ROOT / "mlops" / "retraining" / f"run_{timestamp}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("pipeline_complete", **{
        k: v for k, v in results.items() if k != "models"
    })

    # ── Slack notification ───────────────────────────────────────────
    if results["failed"]:
        notify_slack(
            title=f"❌ Retraining Partial Failure — {timestamp}",
            message=(
                f"*Promoted:* {', '.join(results['promoted']) or 'none'}\n"
                f"*Failed:* {', '.join(results['failed'])}\n"
                f"*Skipped:* {', '.join(results['skipped']) or 'none'}\n"
                f"*Total time:* {results['total_elapsed_seconds']}s"
            ),
            color="warning",
        )
    else:
        notify_slack(
            title=f"✅ Retraining Complete — {timestamp}",
            message=(
                f"*Promoted:* {', '.join(results['promoted']) or 'none (no improvement)'}\n"
                f"*Skipped:* {', '.join(results['skipped']) or 'none'}\n"
                f"*Total time:* {results['total_elapsed_seconds']}s\n"
                f"*Data:* {data_meta['n_interactions']:,} interactions"
            ),
            color="good",
        )

    return 0 if not results["failed"] else 1


def notify_slack(title: str, message: str, color: str = "good"):
    """Send a notification to Slack via webhook."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        logger.info("slack_notify_skipped", reason="SLACK_WEBHOOK_URL not set")
        return

    import urllib.request
    payload = json.dumps({
        "attachments": [{
            "color": color,
            "title": title,
            "text": message,
            "footer": "DS19 Retraining Pipeline",
            "ts": int(time.time()),
        }]
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        logger.info("slack_notified", title=title)
    except Exception as e:
        logger.warning("slack_notify_failed", error=str(e))


if __name__ == "__main__":
    sys.exit(main())