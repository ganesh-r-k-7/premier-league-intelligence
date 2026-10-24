"""Rolling-origin evaluation orchestration for match outcome prediction.

Implements chronological evaluation with strictly prior matchweeks,
refitting predictions at each cutoff, respecting simultaneous-matchweek policy.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from src.features.match_baselines import MatchBaselineConfig, evaluate_match_baselines_at_cutoff
from src.features.match_features import MatchFeatureConfig
from src.models.match_model import MatchModelConfig
from src.evaluation.metrics import (
    compute_match_aggregate_metrics,
    compute_match_metrics_per_cutoff,
)

logger = logging.getLogger("plintel.evaluation.match_rolling")


def run_match_rolling_evaluation(
    fixtures: pd.DataFrame,
    config: MatchBaselineConfig,
    cutoffs: list[int] | None = None,
    season_id: int | None = None
) -> dict[str, pd.DataFrame]:
    """Run full rolling-origin evaluation for match outcome baselines.

    Args:
        fixtures: Full fixtures DataFrame with match_week, season_id, scores, team_ids.
        config: MatchBaselineConfig with parameters.
        cutoffs: List of cutoff matchweeks. Defaults to 1-38.
        season_id: If provided, only evaluate matches from this season.

    Returns:
        Dictionary with keys:
        - 'predictions': concatenated predictions for all cutoffs
        - 'metrics_per_cutoff': per-cutoff metrics
        - 'metrics_aggregate': aggregate metrics across all cutoffs
    """
    if cutoffs is None:
        cutoffs = list(range(1, 39))

    all_predictions = []

    for cw in cutoffs:
        logger.info("Evaluating match outcome cutoff week %d", cw)

        pred = evaluate_match_baselines_at_cutoff(
            fixtures, cw, config, season_id
        )
        if not pred.empty:
            all_predictions.append(pred)

    # Concatenate all predictions
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    # Compute metrics
    metrics_per_cutoff = compute_match_metrics_per_cutoff(predictions) if not predictions.empty else pd.DataFrame()
    metrics_aggregate = compute_match_aggregate_metrics(predictions) if not predictions.empty else pd.DataFrame()

    return {
        "predictions": predictions,
        "metrics_per_cutoff": metrics_per_cutoff,
        "metrics_aggregate": metrics_aggregate,
    }


def run_match_model_evaluation(
    fixtures: pd.DataFrame,
    player_match: pd.DataFrame | None,
    feature_config: MatchFeatureConfig,
    model_config: MatchModelConfig,
    cutoffs: list[int] | None = None,
    season_id: int | None = None
) -> dict[str, pd.DataFrame]:
    """Run rolling-origin evaluation for match feature model.

    Args:
        fixtures: Full fixtures DataFrame.
        player_match: Player match data for xG aggregation.
        feature_config: MatchFeatureConfig.
        model_config: MatchModelConfig.
        cutoffs: List of cutoff matchweeks.
        season_id: Season identifier.

    Returns:
        Dictionary with predictions and metrics.
    """
    from src.models.match_model import run_model_rolling_evaluation

    results = run_model_rolling_evaluation(
        fixtures, player_match, feature_config, model_config, cutoffs, season_id
    )

    # Compute metrics for model predictions
    preds = results.get("predictions", pd.DataFrame())
    model_pred_cols = {
        "feature_model": ("model_home_prob", "model_draw_prob", "model_away_prob"),
    }
    metrics_per_cutoff = compute_match_metrics_per_cutoff(preds, target_col="outcome", pred_cols_map=model_pred_cols) if not preds.empty else pd.DataFrame()
    metrics_aggregate = compute_match_aggregate_metrics(preds, target_col="outcome", pred_cols_map=model_pred_cols) if not preds.empty else pd.DataFrame()

    return {
        "predictions": preds,
        "metrics_per_cutoff": metrics_per_cutoff,
        "metrics_aggregate": metrics_aggregate,
        "fallback_counts": results.get("fallback_counts", 0),
    }


def evaluate_all_match_models(
    fixtures: pd.DataFrame,
    player_match: pd.DataFrame | None,
    baseline_config: MatchBaselineConfig,
    feature_config: MatchFeatureConfig,
    model_config: MatchModelConfig,
    cutoffs: list[int] | None = None,
    season_id: int | None = None
) -> pd.DataFrame:
    """Evaluate all match models (baselines + feature model) on identical observations.

    Args:
        fixtures: Full fixtures DataFrame.
        player_match: Player match data for xG aggregation.
        baseline_config: MatchBaselineConfig for baselines.
        feature_config: MatchFeatureConfig for feature engineering.
        model_config: MatchModelConfig for feature model.
        cutoffs: List of cutoff matchweeks.
        season_id: Season identifier.

    Returns:
        DataFrame with metrics for all models.
    """
    from src.features.match_baselines import evaluate_match_baselines_at_cutoff
    from src.models.match_model import train_model_at_cutoff
    from src.evaluation.metrics import compute_match_metrics_per_cutoff

    if cutoffs is None:
        cutoffs = list(range(1, 39))

    all_baseline_preds = []
    all_model_preds = []

    for cw in cutoffs:
        # Baseline predictions
        base_preds = evaluate_match_baselines_at_cutoff(
            fixtures, cw, baseline_config, season_id
        )
        if not base_preds.empty:
            all_baseline_preds.append(base_preds)

        # Feature model predictions
        model_preds, _ = train_model_at_cutoff(
            fixtures, player_match, cw, season_id, feature_config, model_config
        )
        if not model_preds.empty:
            all_model_preds.append(model_preds)

    if not all_baseline_preds or not all_model_preds:
        return pd.DataFrame()

    base_all = pd.concat(all_baseline_preds, ignore_index=True)
    model_all = pd.concat(all_model_preds, ignore_index=True)

    # Ensure same rows for comparison
    # Merge on match_id
    merged = base_all.merge(
        model_all[["match_id", "model_home_prob", "model_draw_prob", "model_away_prob"]],
        on="match_id",
        how="inner"
    )

    if merged.empty:
        return pd.DataFrame()

    # Compute metrics including all models
    pred_cols_map = {
        "global_frequencies": ("global_home_prob", "global_draw_prob", "global_away_prob"),
        "home_advantage": ("home_advantage_home_prob", "home_advantage_draw_prob", "home_advantage_away_prob"),
        "team_strength": ("team_strength_home_prob", "team_strength_draw_prob", "team_strength_away_prob"),
        "recent_form": ("recent_form_home_prob", "recent_form_draw_prob", "recent_form_away_prob"),
        "feature_model": ("model_home_prob", "model_draw_prob", "model_away_prob"),
    }

    metrics = compute_match_metrics_per_cutoff(
        merged,
        target_col="outcome",
        pred_cols_map=pred_cols_map
    )

    return metrics


def save_match_evaluation_results(
    results: dict[str, pd.DataFrame],
    output_dir: str = "reports",
    prefix: str = "phase4_stage41"
) -> None:
    """Save match outcome evaluation results to parquet files.

    Args:
        results: Dictionary from run_match_rolling_evaluation.
        output_dir: Directory to save files.
        prefix: Prefix for output files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for key, df in results.items():
        if not df.empty:
            path = os.path.join(output_dir, f"{prefix}_{key}.parquet")
            df.to_parquet(path, index=False)
            logger.info("Saved %s to %s (%d rows)", key, path, len(df))


def generate_match_report(
    results: dict[str, pd.DataFrame],
    config: MatchBaselineConfig,
    output_path: str = "reports/phase4_stage41_report.md"
) -> str:
    """Generate human-readable markdown report for match outcome baselines.

    Args:
        results: Dictionary from run_match_rolling_evaluation.
        config: MatchBaselineConfig used.
        output_path: Path to write markdown report.

    Returns:
        Markdown report content as string.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# Phase 4 Stage 4.1 — Match Outcome Baseline Evaluation Report",
        "",
        f"**Configuration:** min_history={config.min_history}, "
        f"smoothing_alpha={config.smoothing_alpha}",
        "",
        "---",
        "",
    ]

    # Aggregate metrics table
    metrics_agg = results.get("metrics_aggregate", pd.DataFrame())
    if not metrics_agg.empty:
        lines.append("## Aggregate Metrics (All Matchweeks Combined)")
        lines.append("")
        lines.append("| Population | Baseline | Log Loss | Brier Score | Accuracy | Sample Size |")
        lines.append("|------------|----------|----------|-------------|----------|-------------|")

        for _, row in metrics_agg.iterrows():
            lines.append(
                f"| {row['population']} | {row['baseline']} | "
                f"{row['log_loss'] if row['log_loss'] is not None else 'N/A':>8} | "
                f"{row['brier_score'] if row['brier_score'] is not None else 'N/A':>11} | "
                f"{row['accuracy'] if row['accuracy'] is not None else 'N/A':>8} | "
                f"{int(row['sample_size'])} |"
            )
        lines.append("")

    # Per-cutoff metrics summary
    metrics_per = results.get("metrics_per_cutoff", pd.DataFrame())
    if not metrics_per.empty:
        lines.append("## Per-Matchweek Metrics Summary")
        lines.append("")

        for pop in sorted(metrics_per["population"].unique()):
            pop_data = metrics_per[metrics_per["population"] == pop]
            if pop_data.empty:
                continue

            lines.append(f"### Population: {pop}")
            lines.append("")
            lines.append("| Cutoff | Baseline | Log Loss | Brier Score | Accuracy | N |")
            lines.append("|--------|----------|----------|-------------|----------|---|")

            for cw in sorted(pop_data["cutoff_week"].unique()):
                cw_data = pop_data[pop_data["cutoff_week"] == cw]
                for _, row in cw_data.iterrows():
                    lines.append(
                        f"| {int(row['cutoff_week'])} | {row['baseline']} | "
                        f"{row['log_loss'] if row['log_loss'] is not None else 'N/A':>8} | "
                        f"{row['brier_score'] if row['brier_score'] is not None else 'N/A':>11} | "
                        f"{row['accuracy'] if row['accuracy'] is not None else 'N/A':>8} | "
                        f"{int(row['sample_size'])} |"
                    )
            lines.append("")

    # Sample sizes
    pred = results.get("predictions", pd.DataFrame())

    lines.append("## Sample Sizes")
    lines.append("")
    lines.append("| Population | Matchweek | Rows |")
    lines.append("|------------|-----------|------|")

    if not pred.empty:
        for pop in sorted(pred["population"].unique()):
            pop_data = pred[pred["population"] == pop]
            for cw in sorted(pop_data["cutoff_week"].unique()):
                n = len(pop_data[pop_data["cutoff_week"] == cw])
                lines.append(f"| {pop} | {cw} | {n} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Known Limitations")
    lines.append("")
    lines.append("1. **Single season only (2015/16):** No cross-season validation possible.")
    lines.append("2. **Matchweek = simultaneous assumption:** Matches within same matchweek treated as concurrent.")
    lines.append("3. **No team strength features beyond historical results:** Baselines use only past scores.")
    lines.append("4. **2003/04 season is partial:** Only 38 matches, not a full season.")
    lines.append("5. **Laplace smoothing:** Applied to handle unseen outcomes in early training windows.")
    lines.append("6. **No injury/suspension/rotation signals:** Not available in current data.")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    logger.info("Match outcome report saved to %s", output_path)
    return report