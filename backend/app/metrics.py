from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

# ── DO NOT create a custom registry ────────────────────────────────
# prometheus_fastapi_instrumentator uses the default registry.
# We add our custom metrics to the SAME default registry.
# This ensures /metrics exposes BOTH HTTP metrics AND our custom ones.
# ───────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════
# METRIC 1: Recommendation end-to-end latency (Histogram)
# ═══════════════════════════════════════════════════════════════════
RECOMMENDATION_LATENCY = Histogram(
    name="ds19_recommendation_latency_seconds",
    documentation=(
        "End-to-end latency of the full recommendation pipeline: "
        "feature retrieval → FAISS → LightGBM → bandit re-rank"
    ),
    # Buckets: 5ms, 10ms, 25ms, 50ms, 100ms, 200ms, 500ms, 1s, 2s, 5s
    # Covers our targets: ~20ms cold, ~2ms cache.
    # Anything > 200ms indicates a serious problem.
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    labelnames=["variant"],  # "mf" or "sasrec" — from A/B router
)

# ═══════════════════════════════════════════════════════════════════
# METRIC 2: Total recommendation requests (Counter)
# ═══════════════════════════════════════════════════════════════════
RECOMMENDATION_REQUESTS = Counter(
    name="ds19_recommendation_requests_total",
    documentation="Total number of /recommend endpoint calls",
    labelnames=["variant", "status"],  # status: "success" | "error" | "cache_hit"
)

# ═══════════════════════════════════════════════════════════════════
# METRIC 3: Cache hit ratio (Gauge)
# ═══════════════════════════════════════════════════════════════════
CACHE_HIT_RATIO = Gauge(
    name="ds19_cache_hit_ratio",
    documentation=(
        "Rolling cache hit ratio for /recommend endpoint. "
        "Updated on every request. Range: [0.0, 1.0]."
    ),
)

# ═══════════════════════════════════════════════════════════════════
# METRIC 4: Per-model inference latency (Histogram)
# ═══════════════════════════════════════════════════════════════════
MODEL_INFERENCE_LATENCY = Histogram(
    name="ds19_model_inference_seconds",
    documentation="Latency of individual model inference calls",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    labelnames=["model"],  # "two_tower", "lgbm", "mf", "sasrec", "bandit"
)

# ═══════════════════════════════════════════════════════════════════
# METRIC 5: Bandit arm selections (Counter)
# ═══════════════════════════════════════════════════════════════════
BANDIT_ARM_SELECTIONS = Counter(
    name="ds19_bandit_arm_selections_total",
    documentation=(
        "Number of times each item was selected by Thompson Sampling. "
        "High counts = exploitation. Items with low counts are being explored."
    ),
    labelnames=["item_bucket"],  # bucket: "top_100", "top_1000", "tail"
)

# ═══════════════════════════════════════════════════════════════════
# METRIC 6: User feedback events (Counter)
# ═══════════════════════════════════════════════════════════════════
FEEDBACK_EVENTS = Counter(
    name="ds19_feedback_events_total",
    documentation="User feedback clicks received via /feedback",
    labelnames=["variant", "feedback_type"],  # type: "positive" | "negative"
)

# ═══════════════════════════════════════════════════════════════════
# METRIC 7: Active A/B experiments (Gauge)
# ═══════════════════════════════════════════════════════════════════
ACTIVE_AB_EXPERIMENTS = Gauge(
    name="ds19_active_ab_experiments",
    documentation="Number of A/B experiments currently running",
)

# ═══════════════════════════════════════════════════════════════════
# METRIC 8: Feature store retrieval latency (Histogram)
# ═══════════════════════════════════════════════════════════════════
FEATURE_STORE_LATENCY = Histogram(
    name="ds19_feature_store_latency_seconds",
    documentation="Latency of Feast online store feature retrieval",
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1],
    labelnames=["feature_view"],  # "user_features" | "item_features"
)

# ═══════════════════════════════════════════════════════════════════
# METRIC 9: Recommendation diversity (Histogram)
# ═══════════════════════════════════════════════════════════════════
RECOMMENDATION_DIVERSITY = Histogram(
    name="ds19_recommendation_genre_diversity",
    documentation=(
        "Number of unique genres in each recommendation list. "
        "Low diversity → users stuck in filter bubble."
    ),
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    labelnames=["variant"],
)

# ═══════════════════════════════════════════════════════════════════
# HELPER: Context managers for timing
# ═══════════════════════════════════════════════════════════════════
class ModelTimer:
    """
    Context manager for recording model inference latency.

    Usage:
        with ModelTimer("lgbm"):
            scores = lgbm_model.predict(features)
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._timer = None

    def __enter__(self):
        self._timer = MODEL_INFERENCE_LATENCY.labels(
            model=self.model_name
        ).time()
        return self._timer.__enter__()

    def __exit__(self, *args):
        return self._timer.__exit__(*args)


class FeatureStoreTimer:
    """Context manager for feature store latency recording."""
    def __init__(self, feature_view: str):
        self.feature_view = feature_view
        self._timer = None

    def __enter__(self):
        self._timer = FEATURE_STORE_LATENCY.labels(
            feature_view=self.feature_view
        ).time()
        return self._timer.__enter__()

    def __exit__(self, *args):
        return self._timer.__exit__(*args)