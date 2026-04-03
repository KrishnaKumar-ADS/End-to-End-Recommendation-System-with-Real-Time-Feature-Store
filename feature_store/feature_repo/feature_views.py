from datetime import timedelta
from pathlib import Path
from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int32
from entities import user_entity, item_entity

# ─────────────────────────────────────────────────────────────
# OFFLINE STORE SOURCES (Parquet files)
# ─────────────────────────────────────────────────────────────

# NOTE: These paths are relative to the feature_repo directory.
# If running feast CLI from feature_store/feature_repo/, they are:
#   ../data/user_features.parquet

USER_FEATURES_PARQUET = str(
    Path(__file__).parent.parent / "data" / "user_features.parquet"
)

ITEM_FEATURES_PARQUET = str(
    Path(__file__).parent.parent / "data" / "item_features.parquet"
)

user_features_source = FileSource(
    path=USER_FEATURES_PARQUET,
    timestamp_field="event_timestamp",
    description="User features computed from MovieLens 25M interaction history"
)

item_features_source = FileSource(
    path=ITEM_FEATURES_PARQUET,
    timestamp_field="event_timestamp",
    description="Item features computed from MovieLens 25M movies and ratings"
)

# ─────────────────────────────────────────────────────────────
# USER FEATURE VIEW
# ─────────────────────────────────────────────────────────────

user_features_view = FeatureView(
    name="user_features_view",
    entities=[user_entity],
    ttl=timedelta(days=30),    # Features stay fresh in Redis for 30 days
    schema=[
        # Basic rating statistics
        Field(name="avg_rating",          dtype=Float32),
        Field(name="total_interactions",  dtype=Int32),
        Field(name="rating_std",          dtype=Float32),
        Field(name="min_rating",          dtype=Float32),
        Field(name="max_rating",          dtype=Float32),

        # Engagement patterns
        Field(name="high_rating_ratio",   dtype=Float32),
        Field(name="active_days",         dtype=Int32),
        Field(name="interaction_density", dtype=Float32),
        Field(name="temporal_spread_days",dtype=Float32),
        Field(name="recency_days",        dtype=Float32),

        # Session behavior
        Field(name="session_count",       dtype=Int32),
        Field(name="avg_session_length",  dtype=Float32),

        # Content preferences
        Field(name="genre_diversity",     dtype=Int32),
        Field(name="top_genre_idx",       dtype=Int32),

        # Popularity behavior
        Field(name="popularity_bias",     dtype=Float32),
    ],
    source=user_features_source,
    online=True,    # Materialize to online store (Redis)
    description="15 user-level features derived from interaction history",
    tags={
        "owner": "ds19_team",
        "version": "1.0",
        "model_usage": "ranking_lightgbm, two_tower"
    }
)

# ─────────────────────────────────────────────────────────────
# ITEM FEATURE VIEW
# ─────────────────────────────────────────────────────────────

item_features_view = FeatureView(
    name="item_features_view",
    entities=[item_entity],
    ttl=timedelta(days=30),
    schema=[
        # Popularity and engagement
        Field(name="global_popularity",    dtype=Int32),
        Field(name="popularity_rank",      dtype=Int32),
        Field(name="niche_score",          dtype=Float32),
        Field(name="avg_ratings_per_day",  dtype=Float32),

        # Rating quality
        Field(name="avg_item_rating",      dtype=Float32),
        Field(name="rating_count",         dtype=Int32),
        Field(name="rating_std",           dtype=Float32),
        Field(name="high_rating_ratio",    dtype=Float32),

        # Content metadata
        Field(name="genre_count",          dtype=Int32),
        Field(name="primary_genre_idx",    dtype=Int32),
        Field(name="release_year",         dtype=Int32),
        Field(name="item_age_days",        dtype=Float32),
    ],
    source=item_features_source,
    online=True,
    description="12 item-level features derived from MovieLens movies and ratings",
    tags={
        "owner": "ds19_team",
        "version": "1.0",
        "model_usage": "ranking_lightgbm, two_tower"
    }
)