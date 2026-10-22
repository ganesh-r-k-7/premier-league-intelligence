"""Rolling-origin evaluation orchestration for goals and assists baselines.

Implements chronological evaluation with strictly prior matchweeks,
refitting predictions at each cutoff, respecting simultaneous-matchweek policy.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from src.features.baselines import BaselineConfig, evaluate_baselines_at_cutoff
from src.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_metrics_per_cutoff,
)

logger = logging.getLogger("plintel.evaluation.rolling")


def prepare_data_with_matchweek(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame
) -> pd.DataFrame:
    """Merge match_week into player_match from fixtures.

    Args:
        player_match: Player-match DataFrame without match_week.
        fixtures: Fixtures DataFrame with match_id and match_week.

    Returns:
        player_match with match_week column added.
    """
    pm = player_match.merge(
        fixtures[["match_id", "match_week"]],
        on="match_id",
        how="left",
        validate="m:1"
    )
    if pm["match_week"].isna().any():
        missing = pm[pm["match_week"].isna()]["match_id"].unique()
        raise ValueError(f"Missing match_week for {len(missing)} match_ids: {missing[:10]}")
    return pm


def run_rolling_evaluation(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    config: BaselineConfig,
    cutoffs: list[int] | None = None
) -> dict[str, pd.DataFrame]:
    """Run full rolling-origin evaluation for both populations.

    Args:
        player_match: Full player_match DataFrame (must have or be mergeable to match_week).
        fixtures: Full fixtures DataFrame with match_week.
        config: BaselineConfig with parameters.
        cutoffs: List of cutoff matchweeks. Defaults to 1-38.

    Returns:
        Dictionary with keys:
        - 'predictions_all_lineup': concatenated predictions for Population A
        - 'predictions_played_only': concatenated predictions for Population B
        - 'metrics_all_lineup': per-cutoff + aggregate metrics for Population A
        - 'metrics_played_only': per-cutoff + aggregate metrics for Population B
    """
    if cutoffs is None:
        cutoffs = list(range(1, 39))

    # Ensure match_week is in player_match
    if "match_week" not in player_match.columns:
        player_match = prepare_data_with_matchweek(player_match, fixtures)

    all_predictions_a = []
    all_predictions_b = []

    for cw in cutoffs:
        logger.info("Evaluating cutoff week %d", cw)

        # Population A: all lineup players
        pred_a = evaluate_baselines_at_cutoff(
            player_match, fixtures, cw, config, "all_lineup"
        )
        if not pred_a.empty:
            all_predictions_a.append(pred_a)

        # Population B: players who played (minutes > 0)
        pred_b = evaluate_baselines_at_cutoff(
            player_match, fixtures, cw, config, "played_only"
        )
        if not pred_b.empty:
            all_predictions_b.append(pred_b)

    # Concatenate all predictions
    predictions_a = pd.concat(all_predictions_a, ignore_index=True) if all_predictions_a else pd.DataFrame()
    predictions_b = pd.concat(all_predictions_b, ignore_index=True) if all_predictions_b else pd.DataFrame()

    # Compute metrics
    metrics_a_per_cutoff = compute_metrics_per_cutoff(predictions_a) if not predictions_a.empty else pd.DataFrame()
    metrics_b_per_cutoff = compute_metrics_per_cutoff(predictions_b) if not predictions_b.empty else pd.DataFrame()

    metrics_a_aggregate = compute_aggregate_metrics(predictions_a) if not predictions_a.empty else pd.DataFrame()
    metrics_b_aggregate = compute_aggregate_metrics(predictions_b) if not predictions_b.empty else pd.DataFrame()

    # Combine per-cutoff metrics
    metrics_per_cutoff = pd.concat([metrics_a_per_cutoff, metrics_b_per_cutoff], ignore_index=True)
    metrics_aggregate = pd.concat([metrics_a_aggregate, metrics_b_aggregate], ignore_index=True)

    return {
        "predictions_all_lineup": predictions_a,
        "predictions_played_only": predictions_b,
        "metrics_per_cutoff": metrics_per_cutoff,
        "metrics_aggregate": metrics_aggregate,
    }


def save_evaluation_results(
    results: dict[str, pd.DataFrame],
    output_dir: str = "reports"
) -> None:
    """Save evaluation results to parquet files.

    Args:
        results: Dictionary from run_rolling_evaluation.
        output_dir: Directory to save files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for key, df in results.items():
        if not df.empty:
            path = os.path.join(output_dir, f"phase2_stage3_{key}.parquet")
            df.to_parquet(path, index=False)
            logger.info("Saved %s to %s (%d rows)", key, path, len(df))


def generate_report(
    results: dict[str, pd.DataFrame],
    config: BaselineConfig,
    output_path: str = "reports/phase2_stage3_report.md"
) -> str:
    """Generate human-readable markdown report.

    Args:
        results: Dictionary from run_rolling_evaluation.
        config: BaselineConfig used.
        output_path: Path to write markdown report.

    Returns:
        Markdown report content as string.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# Phase 2 Stage 3 — Baseline Evaluation Report",
        "",
        f"**Configuration:** min_history={config.min_history}, "
        f"recent_window={config.recent_window}, "
        f"expected_minutes_method={config.expected_minutes_method}",
        "",
        "---",
        "",
    ]

    # Aggregate metrics table
    metrics_agg = results.get("metrics_aggregate", pd.DataFrame())
    if not metrics_agg.empty:
        lines.append("## Aggregate Metrics (All Matchweeks Combined)")
        lines.append("")
        lines.append("| Population | Baseline | MAE Goals | RMSE Goals | Poisson Dev Goals | MAE Assists | RMSE Assists | Poisson Dev Assists | Sample Size |")
        lines.append("|------------|----------|-----------|------------|-------------------|-------------|--------------|---------------------|-------------|")

        for _, row in metrics_agg.iterrows():
            lines.append(
                f"| {row['population']} | {row['baseline']} | "
                f"{row['mae_goals']:.4f} | {row['rmse_goals']:.4f} | "
                f"{row['poisson_deviance_goals'] if row['poisson_deviance_goals'] is not None else 'N/A':>17} | "
                f"{row['mae_assists']:.4f} | {row['rmse_assists']:.4f} | "
                f"{row['poisson_deviance_assists'] if row['poisson_deviance_assists'] is not None else 'N/A':>19} | "
                f"{int(row['sample_size'])} |"
            )
        lines.append("")

    # Per-cutoff metrics summary
    metrics_per = results.get("metrics_per_cutoff", pd.DataFrame())
    if not metrics_per.empty:
        lines.append("## Per-Matchweek Metrics Summary")
        lines.append("")
        lines.append("Showing aggregate MAE/RMSE across populations per matchweek.")
        lines.append("")

        # Summary per cutoff
        for pop in ["all_lineup", "played_only"]:
            pop_data = metrics_per[metrics_per["population"] == pop]
            if pop_data.empty:
                continue

            lines.append(f"### Population: {pop}")
            lines.append("")
            lines.append("| Cutoff | Baseline | MAE Goals | RMSE Goals | MAE Assists | RMSE Assists | N |")
            lines.append("|--------|----------|-----------|------------|-------------|--------------|---|")

            for cw in sorted(pop_data["cutoff_week"].unique()):
                cw_data = pop_data[pop_data["cutoff_week"] == cw]
                for _, row in cw_data.iterrows():
                    lines.append(
                        f"| {int(row['cutoff_week'])} | {row['baseline']} | "
                        f"{row['mae_goals']:.4f} | {row['rmse_goals']:.4f} | "
                        f"{row['mae_assists']:.4f} | {row['rmse_assists']:.4f} | "
                        f"{int(row['sample_size'])} |"
                    )
            lines.append("")

    # Sample sizes
    pred_a = results.get("predictions_all_lineup", pd.DataFrame())
    pred_b = results.get("predictions_played_only", pd.DataFrame())

    lines.append("## Sample Sizes")
    lines.append("")
    lines.append("| Population | Matchweek | Rows |")
    lines.append("|------------|-----------|------|")

    for pop_name, pred in [("all_lineup", pred_a), ("played_only", pred_b)]:
        if not pred.empty:
            for cw in sorted(pred["cutoff_week"].unique()):
                n = len(pred[pred["cutoff_week"] == cw])
                lines.append(f"| {pop_name} | {cw} | {n} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Known Limitations (from Implementation Contract)")
    lines.append("")
    lines.append("1. **Single season only (2015/16):** No cross-season validation possible.")
    lines.append("2. **Matchweek = simultaneous assumption:** Matches within same matchweek treated as concurrent.")
    lines.append("3. **No opponent/team strength features:** Baselines are purely historical averages.")
    lines.append("4. **Position from modal event, not lineup role:** Affects position_team_median estimation.")
    lines.append("5. **Stoppage time in minutes:** Match-end minutes include stoppage time (91-101).")
    lines.append("6. **Unused substitutes in Population A:** 0-minute players receive predictions but actuals=0.")
    lines.append("7. **No injury/suspension/rotation signal:** Expected minutes use historical medians only.")
    lines.append("8. **Penalty goals included in goals:** Baselines predict total goals.")
    lines.append("9. **One 0-minute substitute:** Jordi Gómez (Sunderland) appears in Population A with 0 minutes.")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    logger.info("Report saved to %s", output_path)
    return report