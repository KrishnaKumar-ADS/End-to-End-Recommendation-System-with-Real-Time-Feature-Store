import time

from backend.app.core.config import REDIS_HOST, REDIS_PORT
from backend.app.core.model_loader import ModelLoader as CoreModelLoader
from backend.app.services.cache_service import CacheService
from backend.app.services.feature_service import FeatureService
from backend.app.services.pipeline_service import PipelineService
from backend.app.services.ranking_service import RankingService
from backend.app.services.retrieval_service import RetrievalService
from mlops.ab_testing.ab_logger import ABLogger
from mlops.ab_testing.ab_router import ABRouter as ABTestRouter
from mlops.bandits.bandit_service import BanditService
from mlops.mlflow_setup.model_loader import ModelLoader as MLflowModelLoader


def mark(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


start = time.time()
mark("start")

model_loader = CoreModelLoader()
mark("core model loader created")
model_loader.load_all()
mark(f"core model loader finished in {time.time() - start:.1f}s")

cache_service = CacheService()
mark("cache service ready")

feature_service = FeatureService(model_loader)
mark("feature service ready")

retrieval_service = RetrievalService(model_loader)
mark("retrieval service ready")

ranking_service = RankingService(model_loader)
mark("ranking service ready")

ab_router = ABTestRouter()
mark("ab router ready")

ab_logger = ABLogger()
mark("ab logger ready")

bandit_service = BanditService(
    n_items=model_loader.n_items,
    redis_host=REDIS_HOST,
    redis_port=REDIS_PORT,
    redis_db=1,
    blend_weight=0.3,
)
mark("bandit service created")

try:
    bandit_info = bandit_service.startup()
    mark(f"bandit startup: {bandit_info}")
except Exception as exc:
    mark(f"bandit startup failed: {exc}")

mlflow_loader = MLflowModelLoader()
mark("mlflow loader created")

try:
    info = mlflow_loader.get_production_model_info("lgbm_ranker")
    mark(f"mlflow production info fetched: {bool(info)}")
except Exception as exc:
    mark(f"mlflow info failed: {exc}")

pipeline_service = PipelineService(
    model_loader=model_loader,
    cache_service=cache_service,
    feature_service=feature_service,
    retrieval_service=retrieval_service,
    ranking_service=ranking_service,
)
mark("pipeline service created")

mark("done")
