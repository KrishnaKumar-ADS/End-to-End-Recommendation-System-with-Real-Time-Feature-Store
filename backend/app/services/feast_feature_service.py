import sys
from pathlib import Path

# Add feature_store to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "feature_store"))

from services.feast_feature_service import FeastFeatureService

# Re-export as the same class name expected by pipeline_service.py
# pipeline_service.py does:
#   from backend.app.services.feature_service import FeatureService
# We simply alias:
FeatureService = FeastFeatureService

__all__ = ["FeatureService"]