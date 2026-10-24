"""Rolling-origin evaluation orchestration for match outcome prediction.

Implements chronological evaluation with strictly prior matchweeks,
refitting predictions at each cutoff, respecting simultaneous-matchweek policy.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from src.features.match_baselines import MatchBaselineConfig, evaluate_match_baselines_at_cutoff
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


def save_match_evaluation_results(
    results: dict[str, pd.DataFrame],
    output_dir: str = "reports"
) -> None:
    """Save match outcome evaluation results to parquet files.

    Args:
        results: Dictionary from run_match_rolling_evaluation.
        output_dir: Directory to save files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for key, df in results.items():
        if not df.empty:
            path = os.path.join(output_dir, f"phase4_stage41_{key}.parquet")
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