from feast import FeatureService
from feature_views import user_features_view, item_features_view

# ─────────────────────────────────────────────────────────────
# RANKING FEATURE SERVICE
# ─────────────────────────────────────────────────────────────

ranking_feature_service = FeatureService(
    name="ranking_feature_service",
    features=[
        user_features_view,   # all 15 user features
        item_features_view,   # all 12 item features
    ],
    description=(
        "Feature service for LightGBM ranking model. "
        "Returns all 27 features (15 user + 12 item) needed to "
        "score a (user, item) candidate pair."
    ),
    tags={
        "model": "lgbm_ranker",
        "version": "1.0",
        "owner": "ds19_team"
    }
)

# ─────────────────────────────────────────────────────────────
# USER-ONLY FEATURE SERVICE (for Two-Tower user encoding)
# ─────────────────────────────────────────────────────────────

user_profile_service = FeatureService(
    name="user_profile_service",
    features=[
        user_features_view,   # all 15 user features
    ],
    description=(
        "Feature service for user profile retrieval. "
        "Used for Two-Tower model user encoding and cold-start handling."
    ),
    tags={
        "model": "two_tower",
        "version": "1.0"
    }
)