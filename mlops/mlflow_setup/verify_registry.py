import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops.mlflow_setup.tracking import setup_mlflow, REGISTRY_MODEL_NAMES, EXPERIMENTS
from mlops.mlflow_setup.model_loader import ModelLoader
import mlflow

def verify_registry():
    setup_mlflow()
    loader = ModelLoader()

    print("\n" + "=" * 70)
    print("  DS19 — MLflow Registry Verification")
    print("=" * 70)

    # 1. Check experiments
    print("\n📋 EXPERIMENTS:")
    for component, exp_name in EXPERIMENTS.items():
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp:
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id], max_results=1
            )
            run_count = len(mlflow.search_runs(experiment_ids=[exp.experiment_id]))
            status = f"✅ {run_count} run(s)"
        else:
            status = "❌ Not found"
        print(f"  {component:15s}: {exp_name:40s} → {status}")

    # 2. Check model registry
    print("\n🏛️  MODEL REGISTRY:")
    for component, model_name in REGISTRY_MODEL_NAMES.items():
        info = loader.get_production_model_info(model_name)
        if info:
            print(f"\n  ✅ {model_name}")
            print(f"     Version:  v{info['version']}")
            print(f"     Stage:    {info['stage']}")
            print(f"     Run ID:   {info['run_id'][:16]}...")
            key_metrics = {
                k: v for k, v in info["metrics"].items()
                if "ndcg" in k or "hit_rate" in k or "recall" in k
            }
            for k, v in key_metrics.items():
                print(f"     {k:20s}: {v:.4f}")
        else:
            print(f"\n  ❌ {model_name}: No Production version found")
            print(f"     Run: python mlops/mlflow_setup/model_logger.py")

    # 3. Test loading Production LightGBM
    print("\n🧪 TEST LOADING PRODUCTION LGBM:")
    try:
        lgbm_model = loader.load_lgbm_from_registry()
        if isinstance(lgbm_model, dict) and "model" in lgbm_model:
            lgbm_model = lgbm_model["model"]
        print(f"  ✅ LightGBM loaded successfully")
        print(f"     Type: {type(lgbm_model).__name__}")
        if hasattr(lgbm_model, "num_trees"):
            print(f"     Num trees: {lgbm_model.num_trees()}")
        else:
            print("     Num trees: n/a (object does not expose num_trees)")
    except Exception as e:
        print(f"  ❌ Failed to load LightGBM: {e}")

    print("\n" + "=" * 70)
    print("  Verification complete.")
    print(f"  MLflow UI: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")
    print("=" * 70)


if __name__ == "__main__":
    verify_registry()