"""Player scouting profiles, rankings, and similarity for Phase 5.

Builds player scouting profiles from validated StatsBomb data, with
position-aware rankings and player similarity. All computations use only
validated StatsBomb 2015/16 data. Outputs are explicitly labelled as
heuristics, not predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.features.player_features import POSITION_BANDS, get_position_band


@dataclass(frozen=True)
class ScoutingConfig:
    """Configuration for player scouting profiles and rankings."""
    min_minutes: int = 450              # Minimum minutes for inclusion in rankings
    min_appearances: int = 5            # Minimum appearances for inclusion
    min_xg_samples: int = 3             # Minimum matches with xG for xG metrics
    percentile_warning: float = 95.0    # Percentile above which per-90 metrics flagged as high
    similarity_top_k: int = 5           # Number of similar players to return
    similarity_metric: Literal["euclidean", "cosine"] = "cosine"  # Distance metric
    similarity_features: tuple[str, ...] = (
        "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
        "shots_per_90", "shots_on_target_per_90", "npxg_per_90",
        "goals_per_shot", "goals_per_shot_on_target",
        "minutes_per_match", "starts_rate",
    )  # Features used for similarity


# Position band order for display/sorting
POSITION_BAND_ORDER = ["GK", "DEF", "MID", "WING", "FWD", "UNKNOWN"]


# Metrics that require minimum minutes for per-90 computation
RATE_METRICS = [
    "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
    "npxg_per_90", "shots_per_90", "shots_on_target_per_90",
    "npxg_per_90", "goals_per_shot", "goals_per_shot_on_target",
]


def _compute_per_90(total_stat: float, total_minutes: int) -> float | None:
    """Compute per-90 rate, returning None if minutes == 0."""
    if total_minutes <= 0:
        return None
    return float(total_stat / total_minutes * 90)


def _safe_divide(numerator: float, denominator: float) -> float | None:
    """Safe division returning None for zero denominator."""
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _percentile_rank(series: pd.Series, value: float) -> float:
    """Compute percentile rank of value within series (0-100)."""
    if series.empty or np.isnan(value):
        return np.nan
    return float((series <= value).mean() * 100)


def build_player_profiles(
    player_season_df: pd.DataFrame,
    config: "ScoutingConfig" | None = None,
) -> pd.DataFrame:
    """Build player scouting profiles from validated player_season data.

    Args:
        player_season_df: Validated player_season DataFrame from Phase 2.
        config: ScoutingConfig with thresholds and settings.

    Returns:
        DataFrame with one row per player-season, containing:
        - Identifiers (player_id, player_name, team_id, team_name, position, position_band)
        - Playing time (appearances, starts, minutes)
        - Output (goals, assists, own_goals, penalty_goals)
        - Creation (shots, shots_on_target, xG, npxG, xA)
        - Derived per-90 rates and efficiency metrics
        - Position band and flags for sample size warnings
    """
    if config is None:
        config = ScoutingConfig()

    df = player_season_df.copy()

    # Filter to players with sufficient playing time for meaningful per-90 stats
    # We still keep all players but flag those below thresholds
    df = df.copy()

    # Ensure position_band exists
    if "position_band" not in df.columns:
        df["position_band"] = df["position"].apply(
            lambda x: POSITION_BANDS.get(x, "UNKNOWN") if pd.notna(x) else "UNKNOWN"
        )

    # Total minutes check
    df["total_minutes"] = df["minutes"]

    # Compute per-90 rates (only where minutes > 0)
    for col, new_col in [
        ("goals", "goals_per_90"),
        ("assists", "assists_per_90"),
        ("xg", "xg_per_90"),
        ("xa", "xa_per_90"),
        ("npxg", "npxg_per_90"),
        ("shots", "shots_per_90"),
        ("shots_on_target", "shots_on_target_per_90"),
        ("npxg", "npxg_per_90"),
    ]:
        if col in df.columns:
            df[new_col] = df.apply(
                lambda row: _compute_per_90(row[col], row["minutes"])
                if row["minutes"] > 0 else None,
                axis=1
            )

    # Efficiency metrics (avoid division by zero)
    df["goals_per_shot"] = df.apply(
        lambda row: _safe_divide(row["goals"], row["shots"]) if row.get("shots", 0) > 0 else None,
        axis=1
    )
    df["goals_per_shot_on_target"] = df.apply(
        lambda row: _safe_divide(row["goals"], row["shots_on_target"])
        if row.get("shots_on_target", 0) > 0 else None,
        axis=1
    )

    # xG efficiency
    df["goals_over_xg"] = df.apply(
        lambda row: row["goals"] - row["xg"] if pd.notna(row.get("xg")) else np.nan,
        axis=1
    )
    df["assists_over_xa"] = df.apply(
        lambda row: row["assists"] - row["xa"] if pd.notna(row.get("xa")) else np.nan,
        axis=1
    )

    # npxG (non-penalty xG) is already in data as npxg
    # npxG per 90 already computed above as npxg_per_90

    # Playing time reliability flags
    df["minutes_flag"] = np.where(
        df["minutes"] < config.min_minutes,
        f"below_min_{config.min_minutes}",
        "ok"
    )
    df["appearances_flag"] = np.where(
        df["appearances"] < config.min_appearances,
        f"below_min_{config.min_appearances}",
        "ok"
    )

    # High per-90 warning flags (above percentile threshold)
    rate_cols = [
        "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
        "shots_per_90", "shots_on_target_per_90", "npxg_per_90"
    ]
    for col in rate_cols:
        if col in df.columns:
            pctl = config.percentile_warning
            threshold = df[col].quantile(pctl / 100) if not df[col].isna().all() else np.inf
            df[f"{col}_high_warning"] = df[col].apply(
                lambda v: v > threshold if pd.notna(v) and not np.isinf(threshold) else False
            )

    # Sample size info
    df["matches_with_xg"] = df["xg"].notna().astype(int)  # proxy for xG availability

    # Position band order for sorting
    band_order = {band: i for i, band in enumerate(POSITION_BAND_ORDER)}
    df["position_band_order"] = df["position_band"].map(band_order).fillna(len(POSITION_BAND_ORDER))

    # Sort for consistent output
    df = df.sort_values(["season_id", "position_band_order", "goals_per_90"],
                        ascending=[True, True, False], na_position="last")

    return df


def compute_position_rankings(
    profiles_df: pd.DataFrame,
    config: "ScoutingConfig" | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute position-aware scouting rankings for each metric.

    Args:
        profiles_df: Output from build_player_profiles.
        config: ScoutingConfig with thresholds.

    Returns:
        Dictionary mapping metric name -> ranking DataFrame with columns:
        [player_id, player_name, team_name, position_band, value, rank, percentile, minutes, flag].
        Only includes players meeting minimum thresholds for that metric.
    """
    if config is None:
        config = ScoutingConfig()

    df = profiles_df.copy()

    # Define ranking metrics per position band
    # Each metric: (column_name, display_name, higher_is_better, min_minutes_required)
    ranking_metrics = {
        "goals_per_90": ("Goals per 90", True, config.min_minutes),
        "assists_per_90": ("Assists per 90", True, config.min_minutes),
        "xg_per_90": ("xG per 90", True, config.min_xg_samples * 90),  # approximate
        "xa_per_90": ("xA per 90", True, config.min_xg_samples * 90),
        "npxg_per_90": ("npxG per 90", True, config.min_xg_samples * 90),
        "shots_per_90": ("Shots per 90", True, config.min_minutes),
        "shots_on_target_per_90": ("Shots on Target per 90", True, config.min_minutes),
        "goals_per_shot": ("Goals per Shot", True, config.min_minutes),
        "goals_per_shot_on_target": ("Goals per Shot on Target", True, config.min_minutes),
        "goals_per_90": ("Goals per 90", True, config.min_minutes),
        "assists_per_90": ("Assists per 90", True, config.min_minutes),
        "xg_per_90": ("xG per 90", True, config.min_minutes),
        "xa_per_90": ("xA per 90", True, config.min_minutes),
    }

    # For goalkeepers and defenders, also rank defensive metrics
    gk_def_metrics = {
        "shots_on_target_per_90": ("Shots on Target Faced per 90 (lower better)", False, config.min_minutes),
    }

    rankings = {}

    for band in POSITION_BAND_ORDER:
        if band == "UNKNOWN":
            continue
        band_df = df[df["position_band"] == band].copy()
        if band_df.empty:
            continue

        band_rankings = {}
        metrics_to_use = ranking_metrics.copy()
        if band == "GK":
            metrics_to_use = gk_def_metrics
        elif band == "DEF":
            metrics_to_use = {**ranking_metrics, **gk_def_metrics}

        for metric_col, (display_name, higher_is_better, min_min) in metrics_to_use.items():
            if metric_col not in band_df.columns:
                continue

            # Filter to players with sufficient minutes
            eligible = band_df[band_df["minutes"] >= min_min].copy()
            if eligible.empty:
                continue

            # Drop NaN values for this metric
            eligible = eligible.dropna(subset=[metric_col])
            if eligible.empty:
                continue

            # Compute rank and percentile
            ascending = not higher_is_better
            eligible = eligible.copy()
            eligible["rank"] = eligible[metric_col].rank(ascending=ascending, method="min").astype(int)
            eligible["percentile"] = eligible[metric_col].rank(pct=True, method="min") * 100

            # Select and format output columns
            out_cols = [
                "player_id", "player_name", "team_name", "position",
                metric_col, "rank", "percentile", "minutes",
                "minutes_flag", "appearances_flag"
            ]
            for col in [f"{metric_col}_high_warning"]:
                if col in eligible.columns:
                    out_cols.append(col)

            ranking_df = eligible[out_cols].copy()
            ranking_df = ranking_df.rename(columns={metric_col: "value"})
            ranking_df["metric"] = metric_col
            ranking_df["metric_name"] = display_name
            ranking_df["position_band"] = band

            key = f"{band}_{metric_col}"
            rankings[key] = ranking_df.sort_values("rank")

    return rankings


def compute_similar_players(
    profiles_df: pd.DataFrame,
    target_player_id: int,
    config: "ScoutingConfig" | None = None,
) -> pd.DataFrame | None:
    """Find similar players using cosine/euclidean distance on standardized features.

    Args:
        profiles_df: Output from build_player_profiles.
        target_player_id: Player ID to find similar players for.
        config: ScoutingConfig with similarity settings.

    Returns:
        DataFrame with similar players, or None if player not found / insufficient data.
        Columns: [player_id, player_name, team_name, position_band, similarity,
                  feature_contributions (dict), minutes, position_band]
    """
    if config is None:
        config = ScoutingConfig()

    df = profiles_df.copy()
    target_row = df[df["player_id"] == target_player_id]
    if target_row.empty:
        return None

    target_row = target_row.iloc[0]
    target_band = target_row["position_band"]

    # Filter to same position band (and sufficient minutes)
    candidates = df[
        (df["position_band"] == target_band) &
        (df["minutes"] >= config.min_minutes) &
        (df["player_id"] != target_player_id)
    ].copy()

    print(f"DEBUG: candidates shape: {candidates.shape}, empty: {candidates.empty}")

    if candidates.empty:
        print("DEBUG: Returning None because candidates is empty")
        return None

    # Select similarity features that exist and are not constant
    feature_cols = [f for f in config.similarity_features if f in df.columns]
    if not feature_cols:
        return None

    # Prepare feature matrix
    target_features = target_row[feature_cols].values
    candidate_features = candidates[feature_cols].values

    # Check for NaN and constant columns
    valid_features = []
    for i, col in enumerate(feature_cols):
        col_data = candidates[col]
        if col_data.isna().all() or (col_data.nunique() <= 1 and pd.notna(col_data.iloc[0]) and col_data.iloc[0] == 0):
            continue
        if pd.isna(target_row[col]) or target_row[col] == 0:
            # If target has 0 or NaN, this feature may not be useful
            continue
        valid_features.append(col)

    if len(valid_features) < 2:
        return None

    # Standardize features (fit on candidates, transform both)
    X_cand = candidates[valid_features].fillna(0).values
    X_target = target_row[valid_features].fillna(0).values.reshape(1, -1)

    # Simple standardization
    means = X_cand.mean(axis=0)
    stds = X_cand.std(axis=0)
    stds[stds == 0] = 1  # avoid division by zero
    X_cand_std = (X_cand - means) / stds
    X_target_std = (X_target - means) / stds

    # Compute similarity
    if config.similarity_metric == "cosine":
        # Cosine similarity
        norms_cand = np.linalg.norm(X_cand_std, axis=1)
        norm_target = np.linalg.norm(X_target_std)
        if norm_target == 0 or np.any(norms_cand == 0):
            similarities = np.zeros(len(candidates), dtype=float)
        else:
            similarities = (X_cand_std @ X_target_std.T).flatten() / (norms_cand * norm_target)
            similarities = np.maximum(similarities.astype(float), 0.0)  # Ensure non-negative
    else:
        # Euclidean distance -> similarity = 1 / (1 + distance)
        distances = np.linalg.norm(X_cand_std - X_target_std, axis=1)
        similarities = 1 / (1 + distances)
        similarities = similarities.astype(float)

    candidates = candidates.copy()
    candidates["similarity"] = similarities

    # Feature contributions for top similar players
    # Contribution = (target_val - candidate_val) * weight (weight = 1/std)
    top_k = candidates.nlargest(config.similarity_top_k, "similarity")
    
    result_rows = []
    for _, row in top_k.iterrows():
        contrib = {}
        for col in valid_features:
            if col in row and col in target_row.index:
                target_val = target_row[col]
                cand_val = row[col]
                if pd.notna(target_val) and pd.notna(cand_val):
                    contrib[col] = float(abs(target_val - cand_val))
        contrib_sorted = dict(sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:5])

        result_rows.append({
            "player_id": int(row["player_id"]),
            "player_name": row["player_name"],
            "team_name": row["team_name"],
            "position_band": row["position_band"],
            "similarity": float(row["similarity"]),
            "feature_contributions": contrib_sorted,
            "minutes": int(row["minutes"]),
        })

    return pd.DataFrame(result_rows)


# Configuration dataclass
@dataclass(frozen=True)
class ScoutingConfig:
    """Configuration for player scouting profiles and rankings."""
    min_minutes: int = 450              # Minimum minutes for inclusion in rankings
    min_appearances: int = 5            # Minimum appearances for inclusion
    min_xg_samples: int = 3             # Minimum matches with xG for xG metrics
    percentile_warning: float = 95.0    # Percentile above which per-90 metrics flagged as high
    similarity_top_k: int = 5           # Number of similar players to return
    similarity_metric: Literal["euclidean", "cosine"] = "cosine"  # Distance metric
    similarity_features: tuple[str, ...] = (
        "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
        "shots_per_90", "shots_on_target_per_90", "npxg_per_90",
        "goals_per_shot", "goals_per_shot_on_target",
        "minutes_per_match", "starts_rate",
    )  # Features used for similarity