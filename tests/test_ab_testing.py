import pytest
import sys
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.ab_testing.ab_router import ABRouter, ExperimentConfig
from mlops.ab_testing.ab_analyzer import ABAnalyzer, VariantStats


class TestABRouter:
    """Unit tests for ABRouter deterministic assignment."""

    def setup_method(self):
        self.router = ABRouter()

    def test_assignment_is_deterministic(self):
        """Same user always gets same variant."""
        v1 = self.router.assign(42, "retrieval_v1")
        v2 = self.router.assign(42, "retrieval_v1")
        assert v1 == v2

    def test_assignment_returns_valid_variant(self):
        """Assigned variant must be one of the registered variants."""
        config = self.router.get_experiment("retrieval_v1")
        valid_variants = config.variants
        for user_id in range(100):
            v = self.router.assign(user_id, "retrieval_v1")
            assert v in valid_variants

    def test_split_is_approximately_50_50(self):
        """With 10000 users, each variant should get ~50% ± 2%."""
        user_ids = list(range(10000))
        dist = self.router.get_variant_distribution(user_ids, "retrieval_v1")
        total = sum(dist.values())
        for variant, count in dist.items():
            fraction = count / total
            assert 0.48 <= fraction <= 0.52, (
                f"Variant {variant} got {fraction:.3f}, expected ~0.50 ± 0.02"
            )

    def test_unknown_experiment_raises(self):
        """Assigning to an unregistered experiment raises KeyError."""
        with pytest.raises(KeyError):
            self.router.assign(42, "nonexistent_experiment_xyz")

    def test_deactivated_experiment_returns_control(self):
        """A deactivated experiment always returns the first (control) variant."""
        self.router.register_experiment(
            experiment_id="test_deact",
            description="Test",
            variants=["control", "treatment"],
            traffic_split=[0.5, 0.5],
            is_active=False,
        )
        for user_id in range(50):
            v = self.router.assign(user_id, "test_deact")
            assert v == "control"

    def test_traffic_split_validation(self):
        """Traffic split that doesn't sum to 1.0 raises AssertionError."""
        with pytest.raises(AssertionError):
            ExperimentConfig(
                experiment_id="bad",
                description="bad",
                variants=["a", "b"],
                traffic_split=[0.3, 0.3],  # sums to 0.6, not 1.0
            )

    def test_different_experiments_can_differ(self):
        """Same user can be in different variants across experiments."""
        self.router.register_experiment(
            experiment_id="exp_A",
            description="Exp A",
            variants=["x", "y"],
            traffic_split=[0.5, 0.5],
        )
        self.router.register_experiment(
            experiment_id="exp_B",
            description="Exp B",
            variants=["p", "q"],
            traffic_split=[0.5, 0.5],
        )
        # It's valid (and expected) for a user to be in different variants
        v_a = self.router.assign(999, "exp_A")
        v_b = self.router.assign(999, "exp_B")
        assert v_a in ["x", "y"]
        assert v_b in ["p", "q"]


class TestABAnalyzer:
    """Unit tests for the statistical significance analyzer."""

    def setup_method(self):
        """Create temporary log files for testing."""
        self.tmp_dir = tempfile.mkdtemp()
        self.exp_path  = Path(self.tmp_dir) / "exposures.jsonl"
        self.conv_path = Path(self.tmp_dir) / "conversions.jsonl"
        self.analyzer  = ABAnalyzer(
            exposure_path=str(self.exp_path),
            conversion_path=str(self.conv_path),
        )

    def _write_exposure(self, user_id, variant, experiment_id="test_exp"):
        entry = {"user_id": user_id, "variant": variant,
                 "experiment_id": experiment_id, "ts": 0.0}
        with open(self.exp_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _write_conversion(self, user_id, item_id, variant, experiment_id="test_exp"):
        entry = {"user_id": user_id, "item_id": item_id,
                 "variant": variant, "experiment_id": experiment_id, "ts": 0.0}
        with open(self.conv_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def test_compute_stats_equal_ctrs(self):
        """When CTRs are equal, test should not be significant."""
        for i in range(500):
            self._write_exposure(i, "A")
            if i % 10 == 0:
                self._write_conversion(i, 1, "A")
        for i in range(500, 1000):
            self._write_exposure(i, "B")
            if i % 10 == 0:
                self._write_conversion(i, 2, "B")

        result = self.analyzer.analyze("test_exp", control="A", treatment="B")
        assert not result.is_significant
        assert abs(result.control_stats.ctr - result.treatment_stats.ctr) < 0.02

    def test_compute_stats_different_ctrs(self):
        """With large enough difference and sample, test should be significant."""
        # A: 5% CTR
        for i in range(1000):
            self._write_exposure(i, "A", "exp2")
            if i % 20 == 0:
                self._write_conversion(i, 1, "A", "exp2")

        # B: 15% CTR (big difference)
        for i in range(1000, 2000):
            self._write_exposure(i, "B", "exp2")
            if i % 7 == 0:
                self._write_conversion(i, 2, "B", "exp2")

        result = self.analyzer.analyze("exp2", control="A", treatment="B")
        assert result.is_significant
        assert result.relative_lift_pct > 0  # B is better

    def test_variant_stats_ctr_calculation(self):
        """VariantStats CTR is conversions/exposures."""
        vs = VariantStats(variant="A", exposures=100, conversions=15)
        assert vs.ctr == 0.15
        assert vs.std_err > 0
        ci_lo, ci_hi = vs.confidence_interval_95
        assert ci_lo < vs.ctr < ci_hi