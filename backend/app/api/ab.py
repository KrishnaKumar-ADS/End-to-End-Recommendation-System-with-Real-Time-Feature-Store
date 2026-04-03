from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from mlops.ab_testing.ab_analyzer import ABAnalyzer

router = APIRouter()


def _variant_row(variant: str, exposures: int, conversions: int, ci_low: float, ci_high: float) -> Dict[str, Any]:
    return {
        "variant": variant,
        "n_exposures": int(exposures),
        "n_conversions": int(conversions),
        "ctr": float(conversions / exposures) if exposures > 0 else 0.0,
        "confidence_interval": [float(ci_low), float(ci_high)],
    }


@router.get("/ab/report", tags=["monitoring"], summary="Get A/B experiment report")
async def ab_report(
    experiment_id: str = Query(default="retrieval_v1", description="Experiment identifier"),
):
    """Return per-variant CTR stats and significance for the requested experiment."""
    try:
        from backend.app.main import get_ab_router

        ab_router = get_ab_router()
        exp_cfg = ab_router.get_experiment(experiment_id)

        analyzer = ABAnalyzer()
        stats_by_variant = analyzer.compute_variant_stats(experiment_id)

        configured_variants: List[str] = exp_cfg.variants if exp_cfg else []
        observed_variants = sorted(stats_by_variant.keys())

        if configured_variants:
            all_variants = configured_variants
        elif observed_variants:
            all_variants = observed_variants
        else:
            all_variants = ["mf", "sasrec"]

        variants_payload: List[Dict[str, Any]] = []
        for variant in all_variants:
            stats = stats_by_variant.get(variant)
            if stats is None:
                variants_payload.append(_variant_row(variant, 0, 0, 0.0, 0.0))
                continue
            ci_low, ci_high = stats.confidence_interval_95
            variants_payload.append(
                _variant_row(
                    variant=variant,
                    exposures=stats.exposures,
                    conversions=stats.conversions,
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
            )

        winner: Optional[str] = None
        if variants_payload:
            winner = max(variants_payload, key=lambda row: row["ctr"]).get("variant")

        z_score = 0.0
        p_value = 1.0
        is_significant = False

        if len(all_variants) >= 2:
            control = all_variants[0]
            treatment = all_variants[1]
            try:
                result = analyzer.analyze(
                    experiment_id=experiment_id,
                    control=control,
                    treatment=treatment,
                    confidence_level=0.95,
                )
                z_score = float(result.z_statistic)
                p_value = float(result.p_value)
                is_significant = bool(result.is_significant)
                if is_significant:
                    winner = treatment if result.relative_lift_pct > 0 else control
            except Exception as e:
                logger.info(f"A/B significance not available yet for {experiment_id}: {e}")

        return {
            "experiment_name": experiment_id,
            "start_time": datetime.fromtimestamp(
                exp_cfg.created_at, tz=timezone.utc
            ).isoformat() if exp_cfg else datetime.now(timezone.utc).isoformat(),
            "variants": variants_payload,
            "winner": winner,
            "is_significant": is_significant,
            "p_value": p_value,
            "z_score": z_score,
        }

    except Exception as e:
        logger.exception(f"Failed to build A/B report for {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to build A/B report")
