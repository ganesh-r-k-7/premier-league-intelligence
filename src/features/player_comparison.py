"""Player comparison module for Phase 5.2.

Provides side-by-side player comparisons with percentile ranks within position bands.
All outputs are pandas DataFrames with stable column names and deterministic sorting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.features.player_scouting import POSITION_BAND_ORDER, ScoutingConfig


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for player comparison."""
    min_minutes: int = 450
    min_appearances: int = 5
    min_xg_samples: int = 3
    percentile_warning: float = 95.0
    include_percentiles: bool = True
    percentile_scope: Literal["position_band", "global"] = "position_band"
    handle_missing: Literal["flag", "exclude", "zero"] = "flag"
    tiebreaker: Literal["points", "goal_diff", "goals_scored"] = "points"


@dataclass(frozen=True)
class ComparisonResult:
    """Result of a player comparison."""
    comparison_df: pd.DataFrame
    summary_df: pd.DataFrame
    config: "ComparisonConfig"


# Default metrics to compare, ordered for display
DEFAULT_COMPARISON_METRICS = [
    ("minutes", "Minutes", True),
    ("appearances", "Appearances", True),
    ("starts", "Starts", True),
    ("starts_rate", "Starts Rate", True),
    ("goals", "Goals", True),
    ("goals_per_90", "Goals per 90", True),
    ("assists", "Assists", True),
    ("assists_per_90", "Assists per 90", True),
    ("xg", "xG", True),
    ("xg_per_90", "xG per 90", True),
    ("npxg", "npxG", True),
    ("npxg_per_90", "npxG per 90", True),
    ("xa", "xA", True),
    ("xa_per_90", "xA per 90", True),
    ("shots", "Shots", True),
    ("shots_per_90", "Shots per 90", True),
    ("shots_on_target", "Shots on Target", True),
    ("shots_on_target_per_90", "Shots on Target per 90", True),
    ("goals_per_shot", "Goals per Shot", True),
    ("goals_per_shot_on_target", "Goals per Shot on Target", True),
    ("goals_over_xg", "Goals over xG", True),
    ("assists_over_xa", "Assists over xA", True),
    ("minutes_per_match", "Minutes per Match", True),
    ("starts_rate", "Starts Rate", True),
    ("goals_per_shot", "Goals per Shot", True),
    ("goals_per_shot_on_target", "Goals per Shot on Target", True),
    ("goals_over_xg", "Goals over xG", True),
    ("assists_over_xa", "Assists over xA", True),
]


class PlayerComparison:
    """Side-by-side player comparison with percentile ranks within position bands."""

    def __init__(
        self,
        profiles_df: pd.DataFrame,
        config: Optional["ComparisonConfig"] = None,
    ):
        """Initialize with player profiles DataFrame.

        Args:
            profiles_df: DataFrame from build_player_profiles() with per-90 rates and percentiles.
            config: ComparisonConfig for thresholds and options.
        """
        self.profiles = profiles_df.copy()
        self.config = config or ComparisonConfig()

    def compare_players(
        self,
        player_ids: list[int],
        metrics: Optional[list[str]] = None,
    ) -> ComparisonResult:
        """Compare multiple players side-by-side with percentile ranks.

        Args:
            player_ids: List of player_ids to compare (2 or more).
            metrics: List of metric column names to compare. If None, uses DEFAULT_COMPARISON_METRICS.

        Returns:
            ComparisonResult with comparison_df (side-by-side) and summary_df (differences).
        """
        if len(player_ids) < 2:
            raise ValueError("At least 2 player_ids required for comparison")

        # Validate players exist
        missing_ids = [pid for pid in player_ids if pid not in self.profiles["player_id"].values]
        if missing_ids:
            raise ValueError(f"Player IDs not found in profiles: {missing_ids}")

        # Filter to requested players
        players_df = self.profiles[self.profiles["player_id"].isin(player_ids)].copy()
        if len(players_df) != len(player_ids):
            found_ids = set(players_df["player_id"].values)
            missing = [pid for pid in player_ids if pid not in found_ids]
            raise ValueError(f"Player IDs not found: {missing}")

        # Determine metrics to use
        if metrics is None:
            metrics_to_use = [m for m, _, _ in DEFAULT_COMPARISON_METRICS if m in self.profiles.columns]
        else:
            metrics_to_use = [m for m in metrics if m in self.profiles.columns]
            if not metrics_to_use:
                raise ValueError("None of the requested metrics are available in profiles")

        # Build comparison DataFrame
        comparison_rows = []
        for _, row in self.profiles[self.profiles["player_id"].isin(player_ids)].iterrows():
            row_data = {
                "player_id": int(row["player_id"]),
                "player_name": row["player_name"],
                "team_name": row.get("team_name", ""),
                "position": row.get("position", ""),
                "position_band": row.get("position_band", ""),
                "minutes": int(row.get("minutes", 0)),
                "appearances": int(row.get("appearances", 0)),
            }
            for metric in metrics_to_use:
                value = row.get(metric)
                row_data[f"{metric}_raw"] = value if pd.notna(value) else None
            comparison_rows.append(row_data)

        comparison_df = pd.DataFrame(comparison_rows)

        # Add percentile ranks if requested
        if self.config.include_percentiles:
            for metric in metrics_to_use:
                raw_col = f"{metric}_raw"
                if raw_col in comparison_df.columns:
                    if self.config.percentile_scope == "position_band":
                        # Group by position_band for percentile
                        comparison_df[f"{metric}_percentile"] = (
                            comparison_df.groupby("position_band")[raw_col]
                            .rank(pct=True, method="min") * 100
                        )
                    else:
                        comparison_df[f"{metric}_percentile"] = (
                            comparison_df[raw_col].rank(pct=True, method="min") * 100
                        )
                    # Round for display
                    comparison_df[f"{metric}_percentile"] = comparison_df[f"{metric}_percentile"].round(1)

        # Add difference columns for 2-player comparisons
        if len(player_ids) == 2:
            p1, p2 = player_ids
            for metric in metrics_to_use:
                raw_col = f"{metric}_raw"
                if raw_col in comparison_df.columns:
                    v1 = comparison_df.loc[comparison_df["player_id"] == p1, raw_col].values[0]
                    v2 = comparison_df.loc[comparison_df["player_id"] == p2, raw_col].values[0]
                    if pd.notna(v1) and pd.notna(v2):
                        diff = v1 - v2
                        pct_diff = (diff / v2 * 100) if v2 != 0 else np.inf
                    else:
                        diff = np.nan
                        pct_diff = np.nan
                    comparison_df[f"{metric}_diff"] = diff
                    comparison_df[f"{metric}_pct_diff"] = pct_diff

        # Sort for deterministic output
        sort_cols = ["position_band_order", "goals_per_90_raw"]
        available_sort = [c for c in sort_cols if c in comparison_df.columns]
        if len(available_sort) == 1:
            comparison_df = comparison_df.sort_values(available_sort, ascending=[True]).reset_index(drop=True)
        elif len(available_sort) >= 2:
            comparison_df = comparison_df.sort_values(available_sort, ascending=[True, False]).reset_index(drop=True)
        else:
            comparison_df = comparison_df.reset_index(drop=True)

        # Build summary DataFrame
        summary_rows = []
        if len(player_ids) == 2:
            p1, p2 = player_ids
            for metric in metrics_to_use:
                raw_col = f"{metric}_raw"
                if raw_col in comparison_df.columns:
                    v1 = comparison_df.loc[comparison_df["player_id"] == p1, raw_col].values[0]
                    v2 = comparison_df.loc[comparison_df["player_id"] == p2, raw_col].values[0]
                    if pd.notna(v1) and pd.notna(v2):
                        diff = v1 - v2
                        pct_diff = (diff / v2 * 100) if v2 != 0 else np.inf
                        higher = p1 if v1 > v2 else (p2 if v2 > v1 else "tied")
                    else:
                        diff = np.nan
                        pct_diff = np.nan
                        higher = "insufficient_data"
                    summary_rows.append({
                        "metric": metric,
                        f"{p1}_value": v1,
                        f"{p2}_value": v2,
                        "difference": diff,
                        "pct_diff": pct_diff,
                        "higher": higher,
                    })

        summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

        # Handle missing values according to config
        if self.config.handle_missing == "flag":
            # Already flagged via _high_warning columns and NaN raw values
            pass
        elif self.config.handle_missing == "zero":
            comparison_df = comparison_df.fillna(0)
        elif self.config.handle_missing == "exclude":
            comparison_df = comparison_df.dropna()

        return ComparisonResult(
            comparison_df=comparison_df,
            summary_df=summary_df,
            config=self.config,
        )

    def get_percentile_ranks(
        self,
        player_ids: list[int],
        metrics: list[str],
        scope: Literal["position_band", "global"] = "position_band",
    ) -> dict[int, dict[str, float]]:
        """Get percentile ranks for specified players and metrics.

        Returns:
            Dict mapping player_id -> {metric: percentile}
        """
        # Filter to players with sufficient minutes
        df = self.profiles[self.profiles["player_id"].isin(player_ids)].copy()

        results = {}
        for pid in player_ids:
            row = self.profiles[self.profiles["player_id"] == pid]
            if row.empty:
                results[pid] = {}
                continue

            row = row.iloc[0]
            band = row["position_band"]

            if scope == "position_band":
                reference = self.profiles[self.profiles["position_band"] == band]
            else:
                reference = self.profiles

            percentiles = {}
            for metric in metrics:
                if metric not in self.profiles.columns:
                    percentiles[metric] = np.nan
                    continue

                value = row.get(metric)
                if pd.isna(value):
                    percentiles[metric] = np.nan
                    continue

                ref_values = reference[metric].dropna()
                if ref_values.empty:
                    percentiles[metric] = np.nan
                else:
                    pct = (ref_values <= value).mean() * 100
                    percentiles[metric] = float(pct)

            results[pid] = percentiles

        return results


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for player comparison."""
    min_minutes: int = 450
    min_appearances: int = 5
    min_xg_samples: int = 3
    percentile_warning: float = 95.0
    include_percentiles: bool = True
    percentile_scope: Literal["position_band", "global"] = "position_band"
    handle_missing: Literal["flag", "exclude", "zero"] = "flag"
    tiebreaker: Literal["points", "goal_diff", "goals_scored"] = "points"


@dataclass(frozen=True)
class ComparisonResult:
    """Result of a player comparison."""
    comparison_df: pd.DataFrame
    summary_df: pd.DataFrame
    config: ComparisonConfig