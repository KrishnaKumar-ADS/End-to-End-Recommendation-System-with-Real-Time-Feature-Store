import pickle
import logging
from pathlib import Path
from typing import Optional, Any, Dict

import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient

from mlops.mlflow_setup.tracking import (
    setup_mlflow,
    REGISTRY_MODEL_NAMES,
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads models from the MLflow Model Registry.
    Handles staging transitions and fallback to local files.
    """

    def __init__(self):
        setup_mlflow()
        self.client = MlflowClient()

    # ──────────────────────────────────────────────────────────────────
    # STAGE TRANSITIONS
    # ──────────────────────────────────────────────────────────────────

    def list_model_versions(self, model_name: str) -> None:
        """Print all registered versions of a model with their stages."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            print(f"\n{'='*60}")
            print(f"  Model: {model_name}")
            print(f"  Versions: {len(versions)}")
            print(f"{'='*60}")
            for v in versions:
                print(
                    f"  v{v.version:>3} | "
                    f"Stage: {v.current_stage:12s} | "
                    f"Run: {v.run_id[:8]}... | "
                    f"Created: {v.creation_timestamp}"
                )
            print()
        except Exception as e:
            print(f"  Error listing versions for {model_name}: {e}")

    def promote_to_staging(self, model_name: str, version: int) -> None:
        """
        Transition a model version to Staging.
        
        Do this after logging a new model run and reviewing its metrics.
        Staging means "ready for testing" — may be used in A/B experiments.
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Staging",
            archive_existing_versions=False,
        )
        logger.info(f"Promoted {model_name} v{version} → Staging")

    def promote_to_production(
        self, model_name: str, version: int, archive_old: bool = True
    ) -> None:
        """
        Transition a model version to Production.
        
        If archive_old=True: the current Production version is moved to Archived.
        This ensures only ONE version is in Production at any time.
        FastAPI's load_lgbm_from_registry() will now return this new version.
        
        Args:
            model_name:  Registry model name (e.g., "lgbm_ranker")
            version:     Version number to promote
            archive_old: Whether to archive the previous Production version
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Production",
            archive_existing_versions=archive_old,
        )
        logger.info(
            f"Promoted {model_name} v{version} → Production "
            f"(archive_old={archive_old})"
        )

    def archive_version(self, model_name: str, version: int) -> None:
        """Manually archive a specific model version."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Archived",
        )
        logger.info(f"Archived {model_name} v{version}")

    # ──────────────────────────────────────────────────────────────────
    # MODEL LOADING
    # ──────────────────────────────────────────────────────────────────

    def load_lgbm_from_registry(
        self, fallback_path: str = "models/saved/lgbm_ranker.pkl"
    ) -> Any:
        """
        Load the Production LightGBM model from the MLflow registry.
        
        Loads the model as a pyfunc flavor — works for prediction.
        If no Production model, falls back to local .pkl file.
        
        Returns:
            Loaded model object (LightGBM Booster or pyfunc wrapper)
        """
        model_name = REGISTRY_MODEL_NAMES["lgbm"]
        model_uri  = f"models:/{model_name}/Production"

        try:
            model = mlflow.lightgbm.load_model(model_uri)
            logger.info(
                f"Loaded LightGBM from registry | "
                f"stage=Production | name={model_name}"
            )
            return model
        except Exception as e:
            logger.warning(
                f"Failed to load from MLflow registry ({e}). "
                f"Falling back to local file: {fallback_path}"
            )
            return self._load_lgbm_from_file(fallback_path)

    def _load_lgbm_from_file(self, path: str) -> Any:
        """Load LightGBM model from local .pkl file (fallback)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"LightGBM model not found: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"LightGBM loaded from local file: {path}")
        return model

    def get_production_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Return metadata about the currently deployed Production model.
        Useful for health checks and logging.
        """
        try:
            versions = self.client.get_latest_versions(
                model_name, stages=["Production"]
            )
            if not versions:
                return None
            v = versions[0]
            run = self.client.get_run(v.run_id)
            return {
                "model_name":    model_name,
                "version":       v.version,
                "stage":         v.current_stage,
                "run_id":        v.run_id,
                "created":       v.creation_timestamp,
                "metrics":       dict(run.data.metrics),
                "params":        dict(run.data.params),
                "description":   v.description,
            }
        except Exception as e:
            logger.warning(f"Could not get production model info: {e}")
            return None

    def setup_all_production_stages(self) -> None:
        """
        One-time setup: promote all registered models to Production.
        Run this once after log_all_models() in model_logger.py.
        
        For each model: takes the latest version → Production.
        """
        for component, model_name in REGISTRY_MODEL_NAMES.items():
            try:
                versions = self.client.search_model_versions(
                    f"name='{model_name}'"
                )
                if not versions:
                    logger.warning(f"No versions found for {model_name} — skipping")
                    continue

                # Get the latest version
                latest = max(versions, key=lambda v: int(v.version))
                logger.info(
                    f"Promoting {model_name} v{latest.version} → Production"
                )
                self.promote_to_production(
                    model_name=model_name,
                    version=int(latest.version),
                    archive_old=True,
                )
            except Exception as e:
                logger.warning(f"Could not promote {model_name}: {e}")

        logger.info("All available models promoted to Production stage.")