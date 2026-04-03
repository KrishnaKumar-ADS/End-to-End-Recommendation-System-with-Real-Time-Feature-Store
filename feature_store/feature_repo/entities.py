from feast import Entity, ValueType

# ─────────────────────────────────────────────────────────────
# USER ENTITY
# ─────────────────────────────────────────────────────────────

user_entity = Entity(
    name="user_entity",
    join_keys=["user_idx"],
    value_type=ValueType.INT32,
    description=(
        "A unique user identified by user_idx (integer). "
        "Mapped from MovieLens userId via the user2idx mapping "
        "created during Week 1 data preprocessing."
    ),
    tags={
        "source": "movielens_25m",
        "created_week": "week7",
        "pipeline": "user_features_pipeline"
    }
)

# ─────────────────────────────────────────────────────────────
# ITEM ENTITY
# ─────────────────────────────────────────────────────────────

item_entity = Entity(
    name="item_entity",
    join_keys=["item_idx"],
    value_type=ValueType.INT32,
    description=(
        "A unique movie/item identified by item_idx (integer). "
        "Mapped from MovieLens movieId via the item2idx mapping "
        "created during Week 1 data preprocessing."
    ),
    tags={
        "source": "movielens_25m",
        "created_week": "week7",
        "pipeline": "item_features_pipeline"
    }
)