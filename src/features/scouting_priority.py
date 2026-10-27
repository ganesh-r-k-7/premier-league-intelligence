"""Scouting priority indicators for Phase 5.2.

Computes performance-based scouting priority scores from player profiles.
All outputs are performance-based; no market values, transfer fees, or financial
undervaluation claims are made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.features.player_scouting import ScoutingConfig, compute_similar_players


@dataclass(frozen=True)
class PriorityConfig:
    """Configuration for scouting priority calculation.

    Weights are for component scores. They will be normalized to sum to 1.0
    across components that have positive weight and available data.
    """
    weights: dict[str, float] = field(default_factory=lambda: {
        "production_vs_peers": 0.40,
        "xg_vs_actual": 0.25,
        "playing_time_reliability": 0.20,
        "similarity_to_target": 0.15,
    })
    min_minutes: int = 450
    target_player_id: Optional[int] = None  # For similarity component


def _validate_weights(weights: dict[str, float]) -> dict[str, float]:
    """Validate and normalize weights.

    Args:
        weights: Dictionary of component weights.

    Returns:
        Normalized weights dict (sum = 1.0).

    Raises:
        ValueError: If all weights are zero or negative.
    """
    if not weights:
        raise ValueError("At least one weight must be provided")

    # Check for negative weights
    if any(v < 0 for v in weights.values()):
        raise ValueError("Weights must be non-negative")

    # Filter to non-negative weights
    valid_weights = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(valid_weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be positive")

    return {k: v / total for k, v in valid_weights.items()}


def _percentile_rank(series: pd.Series, value: float) -> float:
    """Compute percentile rank of value within series (0-100)."""
    if series.empty or np.isnan(value):
        return np.nan
    return float((series <= value).mean() * 100)


def _compute_production_vs_peers(
    profiles_df: pd.DataFrame,
    position_band: str,
    min_minutes: int,
) -> pd.Series:
    """Compute production vs peers percentile for each player.

    Combines goals/90, assists/90, xG/90 percentiles within position band.
    """
    band_df = profiles_df[profiles_df["position_band"] == position_band].copy()
    if band_df.empty:
        return pd.Series(np.nan, index=profiles_df.index, name="production_vs_peers")

    # Filter to eligible players
    eligible = band_df[band_df["minutes"] >= min_minutes].copy()
    if eligible.empty:
        return pd.Series(np.nan, index=profiles_df.index, name="production_vs_peers")

    # Compute percentiles for each metric
    metrics = ["goals_per_90", "assists_per_90", "xg_per_90"]
    available_metrics = [m for m in metrics if m in eligible.columns]
    if not available_metrics:
        return pd.Series(np.nan, index=profiles_df.index, name="production_vs_peers")

    percentiles = pd.DataFrame(index=eligible.index)
    for metric in available_metrics:
        percentiles[metric] = eligible[metric].rank(pct=True, method="min") * 100

    # Average percentile across metrics
    production_score = percentiles.mean(axis=1)

    # Map back to full profiles
    result = pd.Series(np.nan, index=profiles_df.index, name="production_vs_peers")
    result.loc[eligible.index] = production_score

    return result


def _compute_xg_vs_actual(
    profiles_df: pd.DataFrame,
) -> pd.Series:
    """Compute xG vs actual goals difference (goals - xG).

    Positive = outperforming xG (overperforming)
    Negative = underperforming xG
    """
    if "goals" not in profiles_df.columns or "xg" not in profiles_df.columns:
        return pd.Series(np.nan, index=profiles_df.index, name="xg_vs_actual")

    diff = profiles_df["goals"] - profiles_df["xg"]
    return diff.rename("xg_vs_actual")


def _compute_playing_time_reliability(
    profiles_df: pd.DataFrame,
    min_minutes: int,
) -> pd.Series:
    """Compute playing time reliability score.

    Based on minutes played relative to max in position band,
    and consistency (starts_rate).
    """
    if "minutes" not in profiles_df.columns:
        return pd.Series(np.nan, index=profiles_df.index, name="playing_time_reliability")

    # Minutes percentile within position band
    minutes_pctl = profiles_df.groupby("position_band")["minutes"].rank(pct=True, method="min") * 100

    # Starts rate contribution (if available)
    if "starts_rate" in profiles_df.columns:
        starts_pctl = profiles_df.groupby("position_band")["starts_rate"].rank(pct=True, method="min") * 100
        reliability = (minutes_pctl + starts_pctl) / 2
    else:
        reliability = minutes_pctl

    return reliability.rename("playing_time_reliability")


def _compute_similarity_to_target(
    profiles_df: pd.DataFrame,
    target_player_id: int,
    config: "ScoutingConfig",
) -> pd.Series:
    """Compute similarity to target player.

    Returns cosine similarity scores for players in same position band.
    """
    if target_player_id is None:
        return pd.Series(np.nan, index=profiles_df.index, name="similarity_to_target")

    from src.features.player_scouting import compute_similar_players, ScoutingConfig

    if target_player_id not in profiles_df["player_id"].values:
        return pd.Series(np.nan, index=profiles_df.index, name="similarity_to_target")

    # Use existing similarity function
    similar = compute_similar_players(
        profiles_df, target_player_id, ScoutingConfig(min_minutes=1)
    )

    if similar is None or similar.empty:
        return pd.Series(np.nan, index=profiles_df.index, name="similarity_to_target")

    # Map similarity scores back to full DataFrame
    sim_series = pd.Series(np.nan, index=profiles_df.index, name="similarity_to_target")
    for _, row in similar.iterrows():
        pid = int(row["player_id"])
        if pid in profiles_df["player_id"].values:
            sim_series.loc[profiles_df["player_id"] == pid] = row["similarity"]

    return sim_series


def _compute_priority_scores(
    profiles_df: pd.DataFrame,
    config: "PriorityConfig",
) -> pd.DataFrame:
    """Compute composite priority scores with component breakdown.

    Args:
        profiles_df: Player profiles DataFrame from build_player_profiles()
        config: PriorityConfig with weights and thresholds

    Returns:
        DataFrame with priority scores and component breakdown
    """
    if config is None:
        config = PriorityConfig()

    # Validate and normalize weights
    weights = _validate_weights(config.weights)

    df = profiles_df.copy()
    n_players = len(df)

    # Filter to minimum minutes
    df = df[df["minutes"] >= config.min_minutes].copy()
    if df.empty:
        return pd.DataFrame()

    # Compute component scores
    components = {}

    # 1. Production vs peers (per position band)
    if weights.get("production_vs_peers", 0) > 0:
        # Compute per position band and combine
        prod_scores = pd.Series(np.nan, index=profiles_df.index, name="production_vs_peers")
        for band in df["position_band"].unique():
            band_df = df[df["position_band"] == band].copy()
            if band_df.empty:
                continue
            eligible = band_df[band_df["minutes"] >= config.min_minutes].copy()
            if eligible.empty:
                continue
            metrics = ["goals_per_90", "assists_per_90", "xg_per_90"]
            available_metrics = [m for m in metrics if m in eligible.columns]
            if available_metrics:
                percentiles = pd.DataFrame(index=eligible.index)
                for metric in available_metrics:
                    percentiles[metric] = eligible[metric].rank(pct=True, method="min") * 100
                prod_score = percentiles.mean(axis=1)
                prod_scores.loc[eligible.index] = prod_score
        components["production_vs_peers"] = prod_scores

    # 2. xG vs actual
    if weights.get("xg_vs_actual", 0) > 0:
        components["xg_vs_actual"] = _compute_xg_vs_actual(df)

    # 3. Playing time reliability
    if weights.get("playing_time_reliability", 0) > 0:
        components["playing_time_reliability"] = _compute_playing_time_reliability(
            df, config.min_minutes
        )

    # 4. Similarity to target (optional)
    if weights.get("similarity_to_target", 0) > 0 and config.target_player_id is not None:
        components["similarity_to_target"] = _compute_similarity_to_target(
            df, config.target_player_id, ScoutingConfig(min_minutes=1)
        )

    # Combine with normalized weights
    available_components = {k: v for k, v in components.items() if v is not None}
    if not available_components:
        return pd.DataFrame()

    # Normalize weights for available components
    available_weights = {k: weights.get(k, 0) for k in available_components}
    total_weight = sum(available_weights.values())
    if total_weight <= 0:
        return pd.DataFrame()

    norm_weights = {k: w / total_weight for k, w in available_weights.items()}

    # Compute weighted score
    df["priority_score"] = 0.0
    df["coverage"] = 0.0  # Fraction of intended score available

    # Store component values and compute weighted score
    for component, weight in norm_weights.items():
        comp_values = available_components[component].fillna(0)
        df[component] = available_components[component]  # Store raw component values
        df["priority_score"] += weight * comp_values
        df["coverage"] += weight * available_components[component].notna().astype(float)

    # Normalize coverage
    df["coverage"] = (df["coverage"] / total_weight * 100).round(1)

    # Scale priority score to 0-100
    max_score = df["priority_score"].max()
    if max_score > 0:
        df["priority_score"] = (df["priority_score"] / max_score * 100).round(1)
    # coverage is already 0-100 from the calculation above

    # Exclude players with zero coverage
    df = df[df["coverage"] > 0].copy()

    # Sort by priority score
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)

    # Add rank
    df["priority_rank"] = range(1, len(df) + 1)

    # Select output columns
    output_cols = [
        "player_id", "player_name", "team_name", "position", "position_band",
        "minutes", "priority_score", "coverage", "priority_rank",
    ]
    for comp in ["production_vs_peers", "xg_vs_actual", "playing_time_reliability", "similarity_to_target"]:
        if comp in df.columns:
            output_cols.append(comp)

    return df[output_cols]


def _validate_weights(weights: dict[str, float]) -> dict[str, float]:
    """Validate and normalize weights.

    Args:
        weights: Dictionary of component weights.

    Returns:
        Normalized weights dict (sum = 1.0).

    Raises:
        ValueError: If all weights are zero or negative.
    """
    if not weights:
        raise ValueError("At least one weight must be provided")

    # Check for negative weights - treat negative as zero
    if any(v < 0 for v in weights.values()):
        # Instead of raising, just treat negative as zero (will be filtered)
        pass

    # Filter to non-negative weights
    valid_weights = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(valid_weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be positive")

    return {k: v / total for k, v in valid_weights.items()}


@dataclass(frozen=True)
class PriorityConfig:
    """Configuration for scouting priority calculation.

    Weights are for component scores. They will be normalized to sum to 1.0
    across components that have positive weight and available data.
    """
    weights: dict[str, float] = field(default_factory=lambda: {
        "production_vs_peers": 0.40,
        "xg_vs_actual": 0.25,
        "playing_time_reliability": 0.20,
        "similarity_to_target": 0.15,
    })
    min_minutes: int = 450
    target_player_id: Optional[int] = None  # For similarity component


def calculate_scouting_priority(
    profiles_df: pd.DataFrame,
    config: Optional[PriorityConfig] = None,
) -> pd.DataFrame:
    """Compute scouting priority scores for all players.

    Args:
        profiles_df: Player profiles DataFrame from build_player_profiles().
        config: PriorityConfig with weights and thresholds.

    Returns:
        DataFrame with priority scores, component scores, and metadata.
    """
    if config is None:
        config = PriorityConfig()

    return _compute_priority_scores(profiles_df, config)


# Export all
__all__ = [
    "PriorityConfig",
    "calculate_scouting_priority",
    "_validate_weights",
    "_compute_production_vs_peers",
    "_compute_xg_vs_actual",
    "_compute_playing_time_reliability",
    "_compute_similarity_to_target",
    "_compute_priority_scores",
    "_validate_weights",
]