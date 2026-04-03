import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class VariantStats:
    """Statistics for one variant in an A/B experiment."""
    variant:     str
    exposures:   int
    conversions: int

    @property
    def effective_conversions(self) -> int:
        """Cap conversions to exposures for binomial-proportion calculations."""
        if self.exposures <= 0:
            return 0
        return min(self.conversions, self.exposures)

    @property
    def ctr(self) -> float:
        return (
            self.effective_conversions / self.exposures
            if self.exposures > 0
            else 0.0
        )

    @property
    def std_err(self) -> float:
        """Standard error of the CTR estimate: sqrt(p(1-p)/n)"""
        if self.exposures == 0:
            return 0.0
        p = min(max(self.ctr, 0.0), 1.0)
        return math.sqrt(p * (1 - p) / self.exposures)

    @property
    def confidence_interval_95(self) -> Tuple[float, float]:
        """95% confidence interval: CTR ± 1.96 * SE"""
        se = self.std_err
        z = 1.96
        return (max(0.0, self.ctr - z * se), min(1.0, self.ctr + z * se))

    def __repr__(self) -> str:
        ci_lo, ci_hi = self.confidence_interval_95
        return (
            f"VariantStats({self.variant}: "
            f"CTR={self.ctr:.4f}, "
            f"n={self.exposures}, "
            f"95%CI=[{ci_lo:.4f},{ci_hi:.4f}])"
        )


@dataclass
class ABTestResult:
    """Result of a statistical significance test between two variants."""
    experiment_id:     str
    control_variant:   str
    treatment_variant: str
    control_stats:     VariantStats
    treatment_stats:   VariantStats
    z_statistic:       float
    p_value:           float
    relative_lift_pct: float
    absolute_lift:     float
    is_significant:    bool
    confidence_level:  float  # 0.95 or 0.99
    recommendation:    str

    @staticmethod
    def format_p_value(p_value: float) -> str:
        """Human-readable p-value formatting that preserves very small values."""
        if not math.isfinite(p_value):
            return "N/A"
        if p_value == 0.0:
            return "<1e-16"
        if p_value < 1e-4:
            return f"{p_value:.2e}"
        return f"{p_value:.4f}"

    def to_dict(self) -> Dict:
        ci_c  = self.control_stats.confidence_interval_95
        ci_t  = self.treatment_stats.confidence_interval_95
        return {
            "experiment_id":      self.experiment_id,
            "control_variant":    self.control_variant,
            "treatment_variant":  self.treatment_variant,
            "control": {
                "exposures":    self.control_stats.exposures,
                "conversions":  self.control_stats.conversions,
                "ctr":          round(self.control_stats.ctr, 4),
                "ci_95_lo":     round(ci_c[0], 4),
                "ci_95_hi":     round(ci_c[1], 4),
            },
            "treatment": {
                "exposures":    self.treatment_stats.exposures,
                "conversions":  self.treatment_stats.conversions,
                "ctr":          round(self.treatment_stats.ctr, 4),
                "ci_95_lo":     round(ci_t[0], 4),
                "ci_95_hi":     round(ci_t[1], 4),
            },
            "statistics": {
                "z_statistic":      round(self.z_statistic, 4),
                "p_value":          float(self.p_value),
                "p_value_display":  self.format_p_value(self.p_value),
                "relative_lift_%":  round(self.relative_lift_pct, 2),
                "absolute_lift":    round(self.absolute_lift, 4),
                "is_significant":   self.is_significant,
                "confidence_level": self.confidence_level,
            },
            "recommendation": self.recommendation,
        }

    def print_summary(self) -> None:
        result = self.to_dict()
        print("\n" + "=" * 70)
        print(f"  A/B TEST RESULTS: {self.experiment_id}")
        print("=" * 70)
        print(f"\n  Control  ({self.control_variant:10s}):")
        print(f"    Exposures:   {self.control_stats.exposures:,}")
        print(f"    Conversions: {self.control_stats.conversions:,}")
        print(f"    CTR:         {self.control_stats.ctr:.4f} "
              f"({self.control_stats.ctr*100:.2f}%)")
        ci_c = self.control_stats.confidence_interval_95
        print(f"    95% CI:      [{ci_c[0]:.4f}, {ci_c[1]:.4f}]")

        print(f"\n  Treatment ({self.treatment_variant:10s}):")
        print(f"    Exposures:   {self.treatment_stats.exposures:,}")
        print(f"    Conversions: {self.treatment_stats.conversions:,}")
        print(f"    CTR:         {self.treatment_stats.ctr:.4f} "
              f"({self.treatment_stats.ctr*100:.2f}%)")
        ci_t = self.treatment_stats.confidence_interval_95
        print(f"    95% CI:      [{ci_t[0]:.4f}, {ci_t[1]:.4f}]")

        print(f"\n  Statistical Test:")
        print(f"    Z-statistic:  {self.z_statistic:+.4f}")
        print(f"    P-value:      {self.format_p_value(self.p_value)}")
        print(f"    Significant:  {'YES ✅' if self.is_significant else 'NO ❌'} "
              f"(α = {1 - self.confidence_level:.2f})")
        print(f"    Relative lift: {self.relative_lift_pct:+.2f}%")
        print(f"    Absolute lift: {self.absolute_lift:+.4f}")

        print(f"\n  📋 RECOMMENDATION: {self.recommendation}")
        print("=" * 70)


class ABAnalyzer:
    """
    Analyzes A/B test logs and computes statistical significance.
    
    Usage:
      analyzer = ABAnalyzer()
      result = analyzer.analyze("retrieval_v1", control="mf", treatment="sasrec")
      result.print_summary()
      analyzer.save_results(result)
    """

    def __init__(
        self,
        exposure_path:   str = "mlops/ab_testing/exposures.jsonl",
        conversion_path: str = "mlops/ab_testing/conversions.jsonl",
    ):
        self.exposure_path   = Path(exposure_path)
        self.conversion_path = Path(conversion_path)

    def _load_logs(self, path: Path) -> List[Dict]:
        """Load all JSONL entries from a log file."""
        entries = []
        if not path.exists():
            return entries
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping corrupt log line: {e}")
        return entries

    @staticmethod
    def _format_p_value(p_value: float) -> str:
        """Format p-value text without rounding tiny values to 0.0000."""
        return ABTestResult.format_p_value(p_value)

    def compute_variant_stats(
        self, experiment_id: str
    ) -> Dict[str, VariantStats]:
        """
        Compute exposures and conversions per variant for an experiment.
        
        Returns:
            Dict mapping variant_name → VariantStats
        """
        exposures   = self._load_logs(self.exposure_path)
        conversions = self._load_logs(self.conversion_path)

        # Count exposures per variant
        exp_counts: Dict[str, int] = {}
        for entry in exposures:
            if entry.get("experiment_id") == experiment_id:
                v = entry.get("variant", "unknown")
                exp_counts[v] = exp_counts.get(v, 0) + 1

        # Count conversions per variant
        # Conversions are matched by user_id to get variant attribution
        user_variant: Dict[int, str] = {}
        for entry in exposures:
            if entry.get("experiment_id") == experiment_id:
                user_variant[entry["user_id"]] = entry["variant"]

        conv_counts: Dict[str, int] = {}
        for entry in conversions:
            if entry.get("experiment_id") == experiment_id:
                uid = entry.get("user_id")
                variant = entry.get("variant") or user_variant.get(uid, "unknown")
                conv_counts[variant] = conv_counts.get(variant, 0) + 1

        # Build VariantStats
        all_variants = set(exp_counts.keys()) | set(conv_counts.keys())
        stats_dict = {}
        for v in all_variants:
            stats_dict[v] = VariantStats(
                variant=v,
                exposures=exp_counts.get(v, 0),
                conversions=conv_counts.get(v, 0),
            )
            if stats_dict[v].conversions > stats_dict[v].exposures > 0:
                logger.warning(
                    "Variant '%s' has conversions (%d) > exposures (%d). "
                    "Using capped conversions for proportion math.",
                    v,
                    stats_dict[v].conversions,
                    stats_dict[v].exposures,
                )

        return stats_dict

    def analyze(
        self,
        experiment_id:     str,
        control:           str,
        treatment:         str,
        confidence_level:  float = 0.95,
    ) -> ABTestResult:
        """
        Run the full statistical significance test.
        
        Args:
            experiment_id:    Which experiment to analyze
            control:          Name of the control variant (e.g., "mf")
            treatment:        Name of the treatment variant (e.g., "sasrec")
            confidence_level: 0.95 for 95% confidence, 0.99 for 99%
        
        Returns:
            ABTestResult with full statistical analysis
        """
        variant_stats = self.compute_variant_stats(experiment_id)

        if control not in variant_stats:
            raise ValueError(
                f"Control variant '{control}' not found in logs. "
                f"Available: {list(variant_stats.keys())}"
            )
        if treatment not in variant_stats:
            raise ValueError(
                f"Treatment variant '{treatment}' not found in logs. "
                f"Available: {list(variant_stats.keys())}"
            )

        ctrl  = variant_stats[control]
        treat = variant_stats[treatment]

        # Z-test for difference in proportions
        n_A, k_A = ctrl.exposures,  ctrl.effective_conversions
        n_B, k_B = treat.exposures, treat.effective_conversions

        ctr_A = ctrl.ctr
        ctr_B = treat.ctr

        # Pooled proportion under H0
        if n_A + n_B == 0:
            raise ValueError("No data found for this experiment")

        p_pool = (k_A + k_B) / (n_A + n_B)
        p_pool = min(max(p_pool, 0.0), 1.0)

        # Standard error of difference
        if n_A == 0 or n_B == 0:
            z_stat  = 0.0
            p_value = 1.0
        else:
            se = math.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
            if se < 1e-10:
                z_stat  = 0.0
                p_value = 1.0
            else:
                z_stat  = (ctr_B - ctr_A) / se
                # Two-tailed p-value
                p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        # Lift
        relative_lift_pct = ((ctr_B - ctr_A) / ctr_A * 100) if ctr_A > 0 else 0.0
        absolute_lift     = ctr_B - ctr_A
        is_significant    = p_value < (1 - confidence_level)

        # Decision logic
        alpha = 1 - confidence_level
        p_value_text = self._format_p_value(p_value)
        if not is_significant:
            recommendation = (
                f"Not significant (p={p_value_text} ≥ α={alpha:.2f}). "
                f"Collect more data (need ≥{self._min_sample_size(ctr_A):.0f} "
                f"exposures per variant for 5% lift detection)."
            )
        elif relative_lift_pct > 0:
            recommendation = (
                f"DEPLOY {treatment.upper()} (p={p_value_text} < α={alpha:.2f}). "
                f"Treatment shows {relative_lift_pct:.1f}% relative lift over control."
            )
        else:
            recommendation = (
                f"KEEP {control.upper()} (p={p_value_text} < α={alpha:.2f}). "
                f"Treatment is WORSE by {abs(relative_lift_pct):.1f}% relative to control."
            )

        return ABTestResult(
            experiment_id=experiment_id,
            control_variant=control,
            treatment_variant=treatment,
            control_stats=ctrl,
            treatment_stats=treat,
            z_statistic=z_stat,
            p_value=p_value,
            relative_lift_pct=relative_lift_pct,
            absolute_lift=absolute_lift,
            is_significant=is_significant,
            confidence_level=confidence_level,
            recommendation=recommendation,
        )

    def _min_sample_size(
        self,
        baseline_ctr: float,
        effect_size: float = 0.05,
        power: float = 0.80,
        alpha: float = 0.05,
    ) -> float:
        """
        Minimum sample size per group to detect a given effect size.
        Effect size = absolute lift in CTR.
        Uses the standard formula for two-proportion z-test.
        """
        p1 = baseline_ctr
        p2 = min(baseline_ctr + effect_size, 0.999999)
        p_bar = (p1 + p2) / 2
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta  = stats.norm.ppf(power)
        numerator   = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
                       + z_beta * math.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2
        denominator = (p2 - p1) ** 2
        return numerator / denominator if denominator > 0 else float("inf")

    def save_results(self, result: ABTestResult, output_dir: str = "mlops/reports") -> str:
        """Save analysis results as a JSON file."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        import time
        filename = f"ab_result_{result.experiment_id}_{int(time.time())}.json"
        filepath = Path(output_dir) / filename
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"A/B test result saved to: {filepath}")
        return str(filepath)