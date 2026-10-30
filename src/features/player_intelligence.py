"""Multi-season player intelligence analytics for Phase 9.1.

Provides reusable metrics for player performance analysis across multiple seasons,
with identity safeguards and position-specific benchmarking.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.data.player_mapping import load_player_registry
from src.data.team_mapping import load_team_registry
from src.features.player_scouting import (
    POSITION_BAND_ORDER,
    POSITION_BANDS,
    get_position_band,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class PlayerIntelligenceConfig:
    """Configuration for player intelligence analysis."""
    # Minimum thresholds for meaningful analysis
    min_minutes_per_season: int = 450       # Minimum minutes per season for inclusion
    min_appearances_per_season: int = 5     # Minimum appearances per season
    min_seasons_for_trend: int = 2          # Minimum seasons for trend analysis
    min_matches_for_consistency: int = 5    # Minimum matches for consistency metrics
    
    # Rolling window parameters
    rolling_window_matches: int = 5         # Matches for rolling consistency metrics
    recent_form_window: int = 5             # Matches for recent form
    
    # Percentile thresholds
    high_percentile: float = 90.0           # Top percentile for breakout/outlier detection
    low_percentile: float = 10.0            # Bottom percentile for decline detection
    consistency_high: float = 75.0          # Consistency percentile threshold
    consistency_low: float = 25.0           # Inconsistency percentile threshold
    
    # Breakout/decline scoring weights
    breakout_weight_production: float = 0.3
    breakout_weight_minutes: float = 0.2
    breakout_weight_per90: float = 0.3
    breakout_weight_consistency: float = 0.2
    
    decline_weight_production: float = 0.4
    decline_weight_minutes: float = 0.2
    decline_weight_per90: float = 0.2
    decline_weight_consistency: float = 0.2
    
    # Position handling
    include_unknown_position: bool = False


# =============================================================================
# Position Mapping (FPL and StatsBomb)
# =============================================================================

# FPL abbreviated positions
FPL_POSITION_BANDS = {
    "GK": "GK",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
    "AM": "MID",  # Attacking midfielder -> MID
}

# StatsBomb full position names (from player_scouting)
STATSBOMB_POSITION_BANDS = {
    "Goalkeeper": "GK",
    "Left Center Back": "DEF",
    "Right Center Back": "DEF",
    "Center Back": "DEF",
    "Left Back": "DEF",
    "Right Back": "DEF",
    "Left Wing Back": "DEF",
    "Right Wing Back": "DEF",
    "Left Wing Back": "DEF",
    "Right Wing Back": "DEF",
    "Left Defensive Midfield": "MID",
    "Right Defensive Midfield": "MID",
    "Center Defensive Midfield": "MID",
    "Left Center Midfield": "MID",
    "Right Center Midfield": "MID",
    "Center Midfield": "MID",
    "Left Midfield": "MID",
    "Right Midfield": "MID",
    "Left Wing": "WING",
    "Right Wing": "WING",
    "Left Attacking Midfield": "MID",
    "Right Attacking Midfield": "MID",
    "Center Attacking Midfield": "MID",
    "Left Center Forward": "FWD",
    "Right Center Forward": "FWD",
    "Center Forward": "FWD",
}


def get_position_band_fpl(position: str | None) -> str:
    """Map FPL abbreviated position to position band."""
    if position is None or (isinstance(position, float) and np.isnan(position)):
        return "UNKNOWN"
    pos = str(position).strip().upper()
    return FPL_POSITION_BANDS.get(pos, "UNKNOWN")


def get_position_band_statsbomb(position: str | None) -> str:
    """Map StatsBomb full position name to position band."""
    if position is None or (isinstance(position, float) and np.isnan(position)):
        return "UNKNOWN"
    pos = str(position).strip()
    return STATSBOMB_POSITION_BANDS.get(pos, "UNKNOWN")


def get_position_band_auto(position: str | None, source: str = "auto") -> str:
    """Automatically detect position format and map to band.
    
    Args:
        position: Position string
        source: "fpl", "statsbomb", or "auto" to detect
    """
    if position is None or (isinstance(position, float) and np.isnan(position)):
        return "UNKNOWN"
    
    pos = str(position).strip()
    
    if source == "fpl" or (source == "auto" and pos in ["GK", "DEF", "MID", "FWD", "AM"]):
        return get_position_band_fpl(pos)
    elif source == "statsbomb" or (source == "auto" and pos not in ["GK", "DEF", "MID", "FWD", "AM"]):
        return get_position_band_statsbomb(pos)
    else:
        return "UNKNOWN"


# =============================================================================
# Identity and Data Loading Helpers
# =============================================================================

def load_fpl_player_seasons(
    min_minutes: Optional[int] = None,
    seasons: Optional[list[int]] = None,
    include_quality_flag: bool = True,
) -> pd.DataFrame:
    """Load FPL player season data from remediated parquet.
    
    Args:
        min_minutes: Filter to players with at least this many minutes per season
        seasons: Specific seasons to load (default: all)
        include_quality_flag: Whether to include quality_flag column
        
    Returns:
        DataFrame with FPL player season data
    """
    df = pd.read_parquet("data/processed/fpl_player_season_remediated.parquet")
    
    if seasons is not None:
        df = df[df["season_id"].isin(seasons)]
    
    if min_minutes is not None:
        df = df[df["minutes"] >= min_minutes]
    
    if not include_quality_flag and "quality_flag" in df.columns:
        df = df.drop(columns=["quality_flag"])
    
    return df


def load_statsbomb_player_seasons() -> pd.DataFrame:
    """Load StatsBomb player season data."""
    return pd.read_parquet("data/processed/statsbomb/player_season.parquet")


def get_canonical_player_ids(
    source: str,
    source_id: str,
    season: Optional[int] = None,
) -> list[str]:
    """Get canonical identity mappings for a source player-season.
    
    Returns list of canonical IDs linking this source record to others.
    Only returns mappings that have been explicitly confirmed.
    """
    try:
        mapping = pd.read_parquet("data/processed/canonical_identity_mapping.parquet")
    except FileNotFoundError:
        return []
    
    # Filter by source and season
    mask = (
        ((mapping["source_a"] == source) & (mapping["source_a_id"] == str(source_id))) |
        ((mapping["source_b"] == source) & (mapping["source_b_id"] == str(source_id)))
    )
    if season is not None:
        mask = mask & (
            ((mapping["source_a"] == source) & (mapping["source_a_season"] == season)) |
            ((mapping["source_b"] == source) & (mapping["source_b_season"] == season))
        )
    
    return mapping[mask]["canonical_id"].tolist()


def get_same_identity_records(
    fpl_player_id: int,
    season: int,
) -> pd.DataFrame:
    """Get all FPL player-season records for the same canonical identity.
    
    Uses canonical identity mapping to find all season records for the same player.
    Only returns records where identity has been confirmed.
    """
    canonical_ids = get_canonical_player_ids("fpl", str(fpl_player_id), season)
    
    if not canonical_ids:
        return pd.DataFrame()
    
    # Load canonical mapping and find all linked records
    mapping = pd.read_parquet("data/processed/canonical_identity_mapping.parquet")
    linked = mapping[mapping["canonical_id"].isin(canonical_ids)]
    
    # Extract all FPL records from confirmed mappings
    fpl_records = []
    for _, row in linked.iterrows():
        if row["source_a"] == "fpl":
            fpl_records.append({
                "player_id": row["source_a_id"],
                "season": row["source_a_season"],
                "canonical_id": row["canonical_id"],
            })
        if row["source_b"] == "fpl":
            fpl_records.append({
                "player_id": row["source_b_id"],
                "season": row["source_b_season"],
                "canonical_id": row["canonical_id"],
            })
    
    if not fpl_records:
        return pd.DataFrame()
    
    fpl_df = pd.DataFrame(fpl_records).drop_duplicates()
    
    # Join with full player season data
    seasons_df = load_fpl_player_seasons()
    result = fpl_df.merge(
        seasons_df,
        left_on=["player_id", "season"],
        right_on=["player_id", "season_id"],
        how="left",
    )
    
    return result


def check_identity_confirmed(
    source_a: str,
    id_a: str,
    season_a: int,
    source_b: str,
    id_b: str,
    season_b: int,
) -> bool:
    """Check if two source records have a confirmed identity mapping."""
    try:
        mapping = pd.read_parquet("data/processed/canonical_identity_mapping.parquet")
    except FileNotFoundError:
        return False
    
    # Check both directions
    mask1 = (
        (mapping["source_a"] == source_a) &
        (mapping["source_a_id"] == str(id_a)) &
        (mapping["source_a_season"] == season_a) &
        (mapping["source_b"] == source_b) &
        (mapping["source_b_id"] == str(id_b)) &
        (mapping["source_b_season"] == season_b)
    )
    mask2 = (
        (mapping["source_a"] == source_b) &
        (mapping["source_a_id"] == str(id_b)) &
        (mapping["source_a_season"] == season_b) &
        (mapping["source_b"] == source_a) &
        (mapping["source_b_id"] == str(id_a)) &
        (mapping["source_b_season"] == season_a)
    )
    
    return bool((mask1 | mask2).any())


# =============================================================================
# Performance Production Metrics
# =============================================================================

def compute_per_90_rate(
    total_stat: float,
    total_minutes: int,
) -> Optional[float]:
    """Compute per-90 rate, returning None if minutes <= 0."""
    if total_minutes <= 0:
        return None
    return float(total_stat / total_minutes * 90)


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    """Safe division returning None for zero denominator."""
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return None
    return float(numerator / denominator)


def compute_performance_metrics(
    player_season_df: pd.DataFrame,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Compute comprehensive performance production metrics per player-season.
    
    Adds per-90 rates, efficiency metrics, and playing time metrics.
    
    Args:
        player_season_df: DataFrame with player-season data
        config: Optional configuration object
        
    Returns:
        DataFrame with added performance metrics
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    df = player_season_df.copy()
    
    # Basic per-90 rates
    rate_mappings = [
        ("goals", "goals_per_90"),
        ("assists", "assists_per_90"),
        ("expected_goals", "expected_goals_per_90"),
        ("expected_assists", "expected_assists_per_90"),
        ("shots", "shots_per_90"),
        ("shots_on_target", "shots_on_target_per_90"),
        ("clean_sheets", "clean_sheets_per_90"),
        ("saves", "saves_per_90"),
    ]
    
    for stat_col, rate_col in rate_mappings:
        if stat_col in df.columns:
            df[rate_col] = df.apply(
                lambda row: compute_per_90_rate(row[stat_col], row["minutes"])
                if row["minutes"] > 0 else None,
                axis=1
            )
    
    # Efficiency metrics
    if "goals" in df.columns and "shots" in df.columns:
        df["goals_per_shot"] = df.apply(
            lambda row: safe_divide(row["goals"], row["shots"]) if row["shots"] > 0 else None,
            axis=1
        )
    
    if "goals" in df.columns and "shots_on_target" in df.columns:
        df["goals_per_shot_on_target"] = df.apply(
            lambda row: safe_divide(row["goals"], row["shots_on_target"]) if row["shots_on_target"] > 0 else None,
            axis=1
        )
    
    if "goals" in df.columns and "expected_goals" in df.columns:
        df["goals_over_xg"] = df["goals"] - df["expected_goals"]
        df["goals_over_xg_per_90"] = df.apply(
            lambda row: compute_per_90_rate(row["goals_over_xg"], row["minutes"])
            if row["minutes"] > 0 else None,
            axis=1
        )
    
    if "assists" in df.columns and "expected_assists" in df.columns:
        df["assists_over_xa"] = df["assists"] - df["expected_assists"]
        df["assists_over_xa_per_90"] = df.apply(
            lambda row: compute_per_90_rate(row["assists_over_xa"], row["minutes"])
            if row["minutes"] > 0 else None,
            axis=1
        )
    
    # Playing time metrics
    if "starts" in df.columns and "appearances" in df.columns:
        df["starts_rate"] = df.apply(
            lambda row: safe_divide(row["starts"], row["appearances"])
            if row["appearances"] > 0 else None,
            axis=1
        )
    
    if "minutes" in df.columns and "appearances" in df.columns:
        df["minutes_per_appearance"] = df.apply(
            lambda row: safe_divide(row["minutes"], row["appearances"])
            if row["appearances"] > 0 else None,
            axis=1
        )
    
    # FPL points efficiency
    if "total_points" in df.columns and "minutes" in df.columns:
        df["points_per_90"] = df.apply(
            lambda row: compute_per_90_rate(row["total_points"], row["minutes"])
            if row["minutes"] > 0 else None,
            axis=1
        )
        df["points_per_appearance"] = df.apply(
            lambda row: safe_divide(row["total_points"], row["appearances"])
            if row["appearances"] > 0 else None,
            axis=1
        )
    
    # xG/xA availability flags
    if "expected_goals" in df.columns:
        df["has_xg"] = df["expected_goals"].notna() & (df["expected_goals"] != 0)
    if "expected_assists" in df.columns:
        df["has_xa"] = df["expected_assists"].notna() & (df["expected_assists"] != 0)
    if "starts" in df.columns:
        df["has_starts"] = df["starts"].notna() & (df["starts"] != 0)
    
    return df


def compute_position_benchmarks(
    player_season_df: pd.DataFrame,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Compute position-specific benchmarks per season.
    
    Returns DataFrame with median, mean, and percentile values per metric per position band per season.
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    df = player_season_df.copy()
    
    # Ensure position_band exists - use auto-detection for FPL/StatsBomb
    if "position_band" not in df.columns and "position" in df.columns:
        df["position_band"] = df["position"].apply(
            lambda x: get_position_band_auto(x) if pd.notna(x) else "UNKNOWN"
        )
    elif "position_band" not in df.columns:
        df["position_band"] = "UNKNOWN"
    
    # Filter to players meeting minimum thresholds
    min_min = config.min_minutes_per_season
    min_app = config.min_appearances_per_season
    
    eligible = df[
        (df["minutes"] >= min_min) &
        (df["appearances"] >= min_app)
    ].copy()
    
    if eligible.empty:
        return pd.DataFrame()
    
    # Filter out UNKNOWN position if not including unknown
    if not config.include_unknown_position:
        eligible = eligible[eligible["position_band"] != "UNKNOWN"]
    
    if eligible.empty:
        return pd.DataFrame()
    
    # Metrics to benchmark
    metrics = [
        "goals_per_90", "assists_per_90", "expected_goals_per_90",
        "expected_assists_per_90", "points_per_90", "total_points",
        "goals_per_90", "assists_per_90", "shots_per_90",
        "goals_per_shot", "goals_per_shot_on_target",
        "clean_sheets_per_90", "saves_per_90",
        "minutes_per_appearance", "starts_rate",
        "goals_over_xg_per_90", "assists_over_xa_per_90",
    ]
    
    available_metrics = [m for m in metrics if m in eligible.columns]
    
    # Compute benchmarks per season per position band
    benchmarks = []
    for season_id, season_df in eligible.groupby("season_id"):
        for band, band_df in season_df.groupby("position_band"):
            if band_df.empty or band == "UNKNOWN":
                continue
            
            row = {"season_id": season_id, "position_band": band, "n_players": len(band_df)}
            
            for metric in available_metrics:
                values = band_df[metric].dropna()
                if len(values) >= 3:
                    row[f"{metric}_median"] = values.median()
                    row[f"{metric}_mean"] = values.mean()
                    row[f"{metric}_std"] = values.std()
                    row[f"{metric}_q25"] = values.quantile(0.25)
                    row[f"{metric}_q75"] = values.quantile(0.75)
                    row[f"{metric}_p90"] = values.quantile(0.90)
                    row[f"{metric}_p10"] = values.quantile(0.10)
            
            benchmarks.append(row)
    
    return pd.DataFrame(benchmarks)


def merge_with_benchmarks(
    player_season_df: pd.DataFrame,
    benchmarks_df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Merge player-season data with position-season benchmarks.
    
    Adds percentile ranks and z-scores relative to position-season peers.
    """
    df = player_season_df.copy()
    
    if "position_band" not in df.columns:
        df["position_band"] = df["position"].apply(
            lambda x: get_position_band(x) if pd.notna(x) else "UNKNOWN"
        )
    
    if metrics is None:
        metrics = [
            "goals_per_90", "assists_per_90", "expected_goals_per_90",
            "expected_assists_per_90", "points_per_90", "total_points",
            "goals_over_xg_per_90", "assists_over_xa_per_90",
        ]
    
    available_metrics = [m for m in metrics if m in df.columns]
    
    for metric in available_metrics:
        median_col = f"{metric}_median"
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        
        if median_col in benchmarks_df.columns and mean_col in benchmarks_df.columns:
            # Merge benchmarks
            merged = df.merge(
                benchmarks_df[["season_id", "position_band", median_col, mean_col, std_col]],
                on=["season_id", "position_band"],
                how="left",
                suffixes=("", "_bench")
            )
            
            # Percentile rank (approximate using normal distribution)
            if metric in df.columns and std_col in merged.columns:
                z_score = (merged[metric] - merged[mean_col]) / merged[std_col].replace(0, np.nan)
                merged[f"{metric}_zscore"] = z_score
                # Approximate percentile from z-score
                merged[f"{metric}_pctl"] = 50 + 50 * np.tanh(z_score * 0.8)
            
            df = merged
    
    return df


# =============================================================================
# Consistency Analysis
# =============================================================================

def compute_match_consistency(
    player_match_df: pd.DataFrame,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Compute match-to-match consistency metrics per player-season.
    
    Uses rolling windows to measure performance variability.
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    df = player_match_df.copy()
    
    # Ensure we have gameweek ordering
    sort_cols = ["player_id", "season_id", "gameweek"]
    for col in sort_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not in player_match data")
    
    df = df.sort_values(sort_cols)
    
    # Only include appearances (minutes > 0)
    appearances = df[df["minutes"] > 0].copy()
    
    if appearances.empty:
        return pd.DataFrame()
    
    results = []
    
    for (player_id, season_id), player_df in appearances.groupby(["player_id", "season_id"]):
        player_df = player_df.sort_values("gameweek")
        
        if len(player_df) < 3:
            continue
        
        # Rolling metrics
        window = min(config.rolling_window_matches, len(player_df))
        
        metrics_to_roll = [
            "goals", "assists", "minutes", "total_points",
            "expected_goals", "expected_assists",
        ]
        
        for metric in metrics_to_roll:
            if metric not in player_df.columns:
                continue
            
            # Rolling mean and std
            rolling_mean = player_df[metric].rolling(window=window, min_periods=1).mean()
            rolling_std = player_df[metric].rolling(window=window, min_periods=2).std()
            
            # Coefficient of variation (std/mean) for consistency
            rolling_cv = rolling_std / rolling_mean.replace(0, np.nan)
            
            # Store last value (end of season)
            row = {
                "player_id": player_id,
                "season_id": season_id,
                f"{metric}_rolling_mean": rolling_mean.iloc[-1],
                f"{metric}_rolling_std": rolling_std.iloc[-1],
                f"{metric}_rolling_cv": rolling_cv.iloc[-1],
            }
            results.append(row)
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def compute_season_consistency(
    player_season_df: pd.DataFrame,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Compute season-to-season consistency metrics.
    
    Measures how consistently a player performs across seasons.
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    df = player_season_df.copy()
    
    # Ensure we have enough seasons per player
    player_seasons = df.groupby("player_id").size().reset_index(name="n_seasons")
    eligible_players = player_seasons[player_seasons["n_seasons"] >= config.min_seasons_for_trend]["player_id"]
    
    if len(eligible_players) == 0:
        return pd.DataFrame()
    
    df = df[df["player_id"].isin(eligible_players)].copy()
    df = df.sort_values(["player_id", "season_id"])
    
    results = []
    
    metrics = [
        "goals_per_90", "assists_per_90", "points_per_90",
        "total_points", "goals_over_xg_per_90", "assists_over_xa_per_90",
        "minutes_per_appearance", "starts_rate",
    ]
    
    for player_id, player_df in df.groupby("player_id"):
        player_df = player_df.sort_values("season_id")
        
        row = {"player_id": player_id, "n_seasons": len(player_df)}
        
        for metric in ["total_points", "goals_per_90", "assists_per_90", "points_per_90"]:
            if metric not in player_df.columns:
                continue
            
            values = player_df[metric].dropna()
            if len(values) < config.min_seasons_for_trend:
                continue
            
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
            row[f"{metric}_cv"] = safe_divide(values.std(), values.mean()) if values.mean() != 0 else None
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
            row[f"{metric}_range"] = values.max() - values.min()
            
            # Trend: slope of linear regression
            if len(values) >= 2:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                row[f"{metric}_trend"] = slope
                # R-squared for trend reliability
                y_pred = np.polyval([slope, np.polyfit(x, values, 1)[1]], x)
                ss_res = np.sum((values - y_pred) ** 2)
                ss_tot = np.sum((values - values.mean()) ** 2)
                row[f"{metric}_r2"] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_match_variability(
    player_match_df: pd.DataFrame,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Compute match-to-match variability metrics per player-season.
    
    Includes:
    - Game-to-game standard deviation in key metrics
    - Coefficient of variation
    - Streak metrics (consecutive good/poor performances)
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    df = player_match_df.copy()
    df = df.sort_values(["player_id", "season_id", "gameweek"])
    
    # Only appearances
    appearances = df[df["minutes"] > 0].copy()
    
    results = []
    
    for (player_id, season_id), player_df in appearances.groupby(["player_id", "season_id"]):
        if len(player_df) < config.min_matches_for_consistency:
            continue
        
        player_df = player_df.sort_values("gameweek")
        
        row = {"player_id": player_id, "season_id": season_id, "n_matches": len(player_df)}
        
        for metric in ["goals", "assists", "minutes", "total_points", "expected_goals", "expected_assists"]:
            if metric not in player_df.columns:
                continue
            
            values = player_df[metric].dropna()
            if len(values) < 2:
                continue
            
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
            row[f"{metric}_cv"] = safe_divide(values.std(), values.mean()) if values.mean() != 0 else None
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
            row[f"{metric}_range"] = values.max() - values.min()
            
            # Streaks: consecutive above/below median
            median_val = values.median()
            above = (values > median_val).astype(int)
            below = (values < median_val).astype(int)
            
            # Max consecutive above/below median
            row[f"{metric}_max_streak_above"] = _max_consecutive(above)
            row[f"{metric}_max_streak_below"] = _max_consecutive(below)
        
        results.append(row)
    
    return pd.DataFrame(results)


def _max_consecutive(series: pd.Series) -> int:
    """Compute maximum consecutive 1s in a binary series."""
    max_consec = 0
    current = 0
    for val in series:
        if val == 1:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    return max_consec


# =============================================================================
# Player Development & Trends
# =============================================================================

def compute_season_over_season_changes(
    player_season_df: pd.DataFrame,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Compute season-over-season changes for key metrics.
    
    Returns DataFrame with changes between consecutive seasons.
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    df = player_season_df.copy()
    df = df.sort_values(["player_id", "season_id"])
    
    # Only players with multiple seasons
    player_seasons = df.groupby("player_id").size().reset_index(name="n_seasons")
    eligible = player_seasons[player_seasons["n_seasons"] >= 2]["player_id"]
    df = df[df["player_id"].isin(eligible)].copy()
    
    # Metrics to track changes for
    metrics = [
        "total_points", "goals_per_90", "assists_per_90", "points_per_90",
        "goals_over_xg_per_90", "assists_over_xa_per_90",
        "minutes", "appearances", "starts", "minutes_per_appearance",
        "starts_rate", "goals_per_90", "assists_per_90",
        "goals_over_xg_per_90", "assists_over_xa_per_90",
    ]
    
    results = []
    
    for player_id, player_df in df.groupby("player_id"):
        player_df = player_df.sort_values("season_id")
        
        for i in range(1, len(player_df)):
            prev = player_df.iloc[i-1]
            curr = player_df.iloc[i]
            
            if curr["season_id"] != prev["season_id"] + 1:
                # Non-consecutive seasons
                continue
            
            row = {
                "player_id": player_id,
                "prev_season": prev["season_id"],
                "curr_season": curr["season_id"],
                "seasons_gap": curr["season_id"] - prev["season_id"],
            }
            
            for metric in metrics:
                if metric not in prev.index or metric not in curr.index:
                    continue
                
                prev_val = prev[metric]
                curr_val = curr[metric]
                
                if pd.isna(prev_val) or pd.isna(curr_val):
                    continue
                
                row[f"{metric}_change"] = curr_val - prev_val
                row[f"{metric}_pct_change"] = safe_divide(curr_val - prev_val, abs(prev_val)) if prev_val != 0 else None
            
            results.append(row)
    
    return pd.DataFrame(results)


def compute_development_trends(
    player_season_df: pd.DataFrame,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Compute linear trend (improvement/decline) over career.
    
    Returns slope, R-squared, and trend classification for each metric.
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    df = player_season_df.copy()
    df = df.sort_values(["player_id", "season_id"])
    
    # Only players with enough seasons
    player_seasons = df.groupby("player_id").size().reset_index(name="n_seasons")
    eligible = player_seasons[player_seasons["n_seasons"] >= config.min_seasons_for_trend]["player_id"]
    df = df[df["player_id"].isin(eligible)].copy()
    
    metrics = [
        "total_points", "goals_per_90", "assists_per_90", "points_per_90",
        "goals_over_xg_per_90", "assists_over_xa_per_90",
        "minutes", "goals_per_90", "assists_per_90",
    ]
    
    results = []
    
    for player_id, player_df in df.groupby("player_id"):
        player_df = player_df.sort_values("season_id")
        
        row = {"player_id": player_id, "n_seasons": len(player_df)}
        
        for metric in metrics:
            if metric not in player_df.columns:
                continue
            
            values = player_df[metric].dropna()
            if len(values) < config.min_seasons_for_trend:
                continue
            
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            y_pred = slope * x + intercept
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - values.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            row[f"{metric}_slope"] = slope
            row[f"{metric}_intercept"] = intercept
            row[f"{metric}_r2"] = r2
            row[f"{metric}_trend"] = _classify_trend(slope, r2)
        
        results.append(row)
    
    return pd.DataFrame(results)


def _classify_trend(slope: float, r2: float) -> str:
    """Classify trend based on slope and R-squared."""
    if r2 < 0.3:
        return "unclear"
    elif slope > 0.1:
        return "improving"
    elif slope < -0.1:
        return "declining"
    else:
        return "stable"


# =============================================================================
# Breakout and Decline Detection
# =============================================================================

def compute_breakout_score(
    player_season_df: pd.DataFrame,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Compute breakout player score based on measurable improvement.
    
    Combines:
    - Production improvement (goals/assists/points)
    - Playing time increase
    - Per-90 efficiency improvement
    - Consistency improvement
    
    Returns DataFrame with breakout_score and component scores.
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    df = player_season_df.copy()
    df = df.sort_values(["player_id", "season_id"])
    
    # Need at least 2 consecutive seasons
    player_seasons = df.groupby("player_id").size().reset_index(name="n_seasons")
    eligible = player_seasons[player_seasons["n_seasons"] >= 2]["player_id"]
    df = df[df["player_id"].isin(eligible)].copy()
    df = df.sort_values(["player_id", "season_id"])
    
    results = []
    
    for player_id, player_df in df.groupby("player_id"):
        player_df = player_df.sort_values("season_id")
        
        for i in range(1, len(player_df)):
            prev = player_df.iloc[i-1]
            curr = player_df.iloc[i]
            
            if curr["season_id"] != prev["season_id"] + 1:
                continue
            
            row = {
                "player_id": player_id,
                "prev_season": prev["season_id"],
                "curr_season": curr["season_id"],
            }
            
            # Production change
            production_metrics = ["total_points", "goals", "assists", "goals_per_90", "assists_per_90"]
            prod_changes = []
            for metric in production_metrics:
                if metric in prev.index and metric in curr.index:
                    pv, cv = prev[metric], curr[metric]
                    if pd.notna(pv) and pd.notna(cv) and pv != 0:
                        prod_changes.append((cv - pv) / abs(pv))
            
            row["production_improvement"] = np.mean(prod_changes) if prod_changes else 0
            
            # Playing time change
            if "minutes" in prev.index and "minutes" in curr.index:
                pm, cm = prev["minutes"], curr["minutes"]
                if pd.notna(pm) and pd.notna(cm) and pm != 0:
                    row["minutes_increase"] = (cm - pm) / abs(pm)
                else:
                    row["minutes_increase"] = 0
            
            # Per-90 efficiency change
            per90_metrics = ["goals_per_90", "assists_per_90", "points_per_90"]
            per90_changes = []
            for metric in per90_metrics:
                if metric in prev.index and metric in curr.index:
                    pv, cv = prev[metric], curr[metric]
                    if pd.notna(pv) and pd.notna(cv) and pv != 0:
                        per90_changes.append((cv - pv) / abs(pv))
            
            row["per90_improvement"] = np.mean(per90_changes) if per90_changes else 0
            
            # Consistency change (lower CV = more consistent)
            if "minutes" in prev.index and "minutes" in curr.index:
                # Simple consistency proxy: inverse of CV change
                row["consistency_improvement"] = 0  # Placeholder
            
            # Combined breakout score
            row["breakout_score"] = (
                config.breakout_weight_production * row.get("production_improvement", 0) +
                config.breakout_weight_minutes * row.get("minutes_increase", 0) +
                config.breakout_weight_per90 * row.get("per90_improvement", 0) +
                config.breakout_weight_consistency * row.get("consistency_improvement", 0)
            )
            
            # Percentile rank within season
            row["player_name"] = curr.get("player_name", "")
            row["team"] = curr.get("team", curr.get("team_name", ""))
            row["position"] = curr.get("position", "")
            
            results.append(row)
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    
    # Add percentile ranks within current season
    for col in ["breakout_score", "production_improvement", "minutes_increase", "per90_improvement"]:
        if col in result_df.columns:
            result_df[f"{col}_pctl"] = result_df.groupby("curr_season")[col].rank(pct=True) * 100
    
    # Flag breakout candidates
    result_df["is_breakout_candidate"] = result_df["breakout_score_pctl"] >= config.high_percentile
    
    return result_df


def compute_decline_score(
    player_season_df: pd.DataFrame,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Compute decline score based on measurable deterioration.
    
    Similar to breakout but inverted - looks for sustained deterioration.
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    df = player_season_df.copy()
    df = df.sort_values(["player_id", "season_id"])
    
    eligible = df.groupby("player_id").size().reset_index(name="n_seasons")
    eligible = eligible[eligible["n_seasons"] >= 2]["player_id"]
    df = df[df["player_id"].isin(eligible)].copy()
    df = df.sort_values(["player_id", "season_id"])
    
    results = []
    
    for player_id, player_df in df.groupby("player_id"):
        player_df = player_df.sort_values("season_id")
        
        for i in range(1, len(player_df)):
            prev = player_df.iloc[i-1]
            curr = player_df.iloc[i]
            
            if curr["season_id"] != prev["season_id"] + 1:
                continue
            
            row = {
                "player_id": player_id,
                "prev_season": prev["season_id"],
                "curr_season": curr["season_id"],
            }
            
            # Production decline
            production_metrics = ["total_points", "goals", "assists", "goals_per_90", "assists_per_90"]
            prod_changes = []
            for metric in production_metrics:
                if metric in prev.index and metric in curr.index:
                    pv, cv = prev[metric], curr[metric]
                    if pd.notna(pv) and pd.notna(cv) and pv != 0:
                        prod_changes.append((cv - pv) / abs(pv))
            
            row["production_decline"] = -np.mean(prod_changes) if prod_changes else 0  # Negative = decline
            
            # Minutes decline
            if "minutes" in prev.index and "minutes" in curr.index:
                pm, cm = prev["minutes"], curr["minutes"]
                if pd.notna(pm) and pd.notna(cm) and pm != 0:
                    row["minutes_decline"] = -(cm - pm) / abs(pm)  # Negative = decline
                else:
                    row["minutes_decline"] = 0
            
            # Per-90 decline
            per90_metrics = ["goals_per_90", "assists_per_90", "points_per_90"]
            per90_changes = []
            for metric in per90_metrics:
                if metric in prev.index and metric in curr.index:
                    pv, cv = prev[metric], curr[metric]
                    if pd.notna(pv) and pd.notna(cv) and pv != 0:
                        per90_changes.append((cv - pv) / abs(pv))
            
            row["per90_decline"] = -np.mean(per90_changes) if per90_changes else 0
            
            # Combined decline score
            row["decline_score"] = (
                config.decline_weight_production * row.get("production_decline", 0) +
                config.decline_weight_minutes * row.get("minutes_decline", 0) +
                config.decline_weight_per90 * row.get("per90_decline", 0) +
                config.decline_weight_consistency * 0  # Placeholder
            )
            
            row["player_name"] = curr.get("player_name", "")
            row["team"] = curr.get("team", curr.get("team_name", ""))
            row["position"] = curr.get("position", "")
            
            results.append(row)
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    
    # Percentile ranks
    for col in ["decline_score", "production_decline", "minutes_decline", "per90_decline"]:
        if col in result_df.columns:
            result_df[f"{col}_pctl"] = result_df.groupby("curr_season")[col].rank(pct=True) * 100
    
    result_df["is_decline_candidate"] = result_df["decline_score_pctl"] >= config.high_percentile
    
    return result_df


# =============================================================================
# Identity-Aware Player Analysis
# =============================================================================

def get_player_career(
    fpl_player_id: int,
    min_minutes: int = 450,
) -> pd.DataFrame:
    """Get complete career for a confirmed FPL player identity.
    
    Uses canonical identity mappings to link seasons.
    Only returns seasons where identity is confirmed.
    """
    df = get_same_identity_records(fpl_player_id, 0)
    if df.empty:
        return df
    return df[df["minutes"] >= min_minutes].copy()


def get_career_trajectory(
    fpl_player_id: int,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Get complete career trajectory with all computed metrics.
    
    Only uses confirmed identity mappings.
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    career = get_player_career(fpl_player_id, config.min_minutes_per_season)
    
    if career.empty:
        return pd.DataFrame()
    
    # Add performance metrics
    career = compute_performance_metrics(career, config)
    
    # Add development trends
    trends = compute_development_trends(career[["player_id", "season_id", "player_name", "team"] + 
                                                [c for c in career.columns if c not in ["player_id", "season_id", "player_name", "team"]]], config)
    
    if not trends.empty:
        career = career.merge(trends, on="player_id", how="left")
    
    return career


def find_breakout_players(
    player_season_df: pd.DataFrame,
    season: Optional[int] = None,
    min_breakout_score: float = 0,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Find breakout players in a specific season or all seasons.
    
    Only uses confirmed identity continuity.
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    breakout_df = compute_breakout_score(player_season_df, config)
    
    if breakout_df.empty:
        return pd.DataFrame()
    
    if season is not None:
        breakout_df = breakout_df[breakout_df["curr_season"] == season]
    
    if min_breakout_score > 0:
        breakout_df = breakout_df[breakout_df["breakout_score"] >= min_breakout_score]
    
    # Sort by breakout score descending
    breakout_df = breakout_df.sort_values("breakout_score", ascending=False)
    
    return breakout_df


def find_declining_players(
    player_season_df: pd.DataFrame,
    season: Optional[int] = None,
    min_decline_score: float = 0,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> pd.DataFrame:
    """Find declining players in a specific season or all seasons.
    
    Only uses confirmed identity continuity.
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    decline_df = compute_decline_score(player_season_df, config)
    
    if decline_df.empty:
        return pd.DataFrame()
    
    if season is not None:
        decline_df = decline_df[decline_df["curr_season"] == season]
    
    if min_decline_score > 0:
        decline_df = decline_df[decline_df["decline_score"] >= min_decline_score]
    
    decline_df = decline_df.sort_values("decline_score", ascending=False)
    
    return decline_df


# =============================================================================
# Dashboard Integration Helpers
# =============================================================================

def get_top_performers_by_metric(
    player_season_df: pd.DataFrame,
    metric: str,
    season: Optional[int] = None,
    position_band: Optional[str] = None,
    top_n: int = 10,
    min_minutes: int = 450,
) -> pd.DataFrame:
    """Get top N performers for a specific metric."""
    df = player_season_df.copy()
    
    if season is not None:
        df = df[df["season_id"] == season]
    
    if "position_band" not in df.columns:
        df["position_band"] = df["position"].apply(
            lambda x: get_position_band(x) if pd.notna(x) else "UNKNOWN"
        )
    
    if position_band is not None:
        df = df[df["position_band"] == position_band]
    
    df = df[df["minutes"] >= min_minutes]
    df = df.dropna(subset=[metric])
    
    return df.nlargest(top_n, metric)[
        ["player_id", "player_name", "team", "position", "season_id", metric, "minutes"]
    ]


def get_consistency_leaders(
    player_season_df: pd.DataFrame,
    top_n: int = 10,
    metric: str = "total_points",
) -> pd.DataFrame:
    """Get most consistent players (lowest CV) for a metric across all seasons."""
    df = player_season_df.copy()
    
    # Need consistency metrics computed
    consistency_df = compute_season_consistency(df)
    
    if consistency_df.empty:
        return pd.DataFrame()
    
    cv_col = f"{metric}_cv"
    if cv_col not in consistency_df.columns:
        return pd.DataFrame()
    
    consistency_df = consistency_df.dropna(subset=[cv_col])
    consistency_df = consistency_df.nsmallest(top_n, cv_col)
    
    # Merge with player info
    player_info = player_season_df[["player_id", "player_name", "team", "position", "season_id"]].drop_duplicates("player_id")
    result = consistency_df.merge(player_info, on="player_id", how="left")
    
    return result[["player_id", "player_name", "team", "position", cv_col]]


def get_development_summary(
    fpl_player_id: int,
    config: Optional["PlayerIntelligenceConfig"] = None,
) -> dict:
    """Get comprehensive development summary for a player.
    
    Returns dictionary with career trajectory, trends, breakout/decline status.
    """
    if config is None:
        config = PlayerIntelligenceConfig()
    
    career = get_career_trajectory(fpl_player_id, config)
    
    if career.empty:
        return {"error": "No career data found or identity not confirmed"}
    
    summary = {
        "player_id": fpl_player_id,
        "player_name": career["player_name"].iloc[0] if not career.empty else "",
        "team": career["team"].iloc[0] if not career.empty else "",
        "seasons": len(career),
        "total_minutes": int(career["minutes"].sum()),
        "total_goals": int(career["goals"].sum()),
        "total_assists": int(career["assists"].sum()),
        "total_points": int(career["total_points"].sum()),
    }
    
    # Latest season trends
    if "goals_per_90_trend" in career.columns:
        latest = career.iloc[-1]
        for metric in ["goals_per_90", "assists_per_90", "total_points", "goals_over_xg_per_90"]:
            trend_col = f"{metric}_trend"
            slope_col = f"{metric}_slope"
            if trend_col in career.columns:
                summary[f"{metric}_trend"] = latest[trend_col]
            if slope_col in career.columns:
                summary[f"{metric}_slope"] = latest[slope_col]
    
    # Breakout/decline status
    breakout_df = compute_breakout_score(pd.DataFrame([career.iloc[-1]]))
    if not breakout_df.empty:
        summary["is_breakout"] = breakout_df["is_breakout_candidate"].iloc[0]
        summary["breakout_score"] = breakout_df["breakout_score"].iloc[0]
    
    decline_df = compute_decline_score(pd.DataFrame([career.iloc[-1]]))
    if not decline_df.empty:
        summary["is_decline"] = decline_df["is_decline_candidate"].iloc[0]
        summary["decline_score"] = decline_df["decline_score"].iloc[0]
    
    return summary


# =============================================================================
# Main Configuration Class
# =============================================================================

@dataclass(frozen=True)
class PlayerIntelligenceConfig:
    """Configuration for player intelligence analysis."""
    # Minimum thresholds for meaningful analysis
    min_minutes_per_season: int = 450
    min_appearances_per_season: int = 5
    min_seasons_for_trend: int = 2
    min_matches_for_consistency: int = 5
    
    # Rolling window parameters
    rolling_window_matches: int = 5
    recent_form_window: int = 5
    
    # Percentile thresholds
    high_percentile: float = 90.0
    low_percentile: float = 10.0
    consistency_high: float = 75.0
    consistency_low: float = 25.0
    
    # Breakout/decline scoring weights
    breakout_weight_production: float = 0.3
    breakout_weight_minutes: float = 0.2
    breakout_weight_per90: float = 0.3
    breakout_weight_consistency: float = 0.2
    
    decline_weight_production: float = 0.4
    decline_weight_minutes: float = 0.2
    decline_weight_per90: float = 0.2
    decline_weight_consistency: float = 0.2
    
    # Position handling
    include_unknown_position: bool = False


if __name__ == "__main__":
    # Quick test
    print("Player Intelligence module loaded successfully")
    print("Available functions:")
    print("  - compute_performance_metrics")
    print("  - compute_position_benchmarks")
    print("  - compute_season_consistency")
    print("  - compute_match_consistency")
    print("  - compute_season_over_season_changes")
    print("  - compute_development_trends")
    print("  - compute_breakout_score")
    print("  - compute_decline_score")
    print("  - get_same_identity_records")
    print("  - get_career_trajectory")
    print("  - find_breakout_players")
    print("  - find_declining_players")
    print("  - get_top_performers_by_metric")
    print("  - get_consistency_leaders")
    print("  - get_development_summary")