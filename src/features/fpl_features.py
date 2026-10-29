"""FPL-specific feature engineering for multi-season player performance forecasting.

Builds features from FPL player-season and player-match (gameweek) data.
All features computed with strict chronological boundaries to prevent leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FPLFeatureConfig:
    """Configuration for FPL feature engineering."""
    # Historical window for rolling features (in gameweeks)
    rolling_window: int = 5
    # Minimum gameweeks for player-specific features
    min_history: int = 3
    # Whether to include xG/xA features (only available 2022/23+)
    include_xg: bool = True
    # Whether to include starts feature (only available 2022/23+)
    include_starts: bool = True
    # Whether to include position/team features (2020/21+)
    include_position_team: bool = True
    # Gameweek column name
    gameweek_col: str = "gameweek"
    # Season column name
    season_col: str = "season_name"
    # Player identifier column
    player_id_col: str = "player_id"
    # Team identifier column
    team_col: str = "team"
    # Position column
    position_col: str = "position"


# Columns that indicate xG/xA availability (all zeros = not available)
XG_AVAILABILITY_CHECK_COLS = ["expected_goals", "expected_assists"]
STARTS_AVAILABILITY_CHECK_COLS = ["starts"]
POSITION_TEAM_AVAILABILITY_CHECK_COLS = ["position", "team"]


def _check_feature_availability(
    df: pd.DataFrame,
    check_cols: list[str]
) -> bool:
    """Check if feature columns have meaningful (non-zero, non-null) data."""
    if not check_cols:
        return False
    for col in check_cols:
        if col not in df.columns:
            return False
        # Check if column has any non-zero, non-null values
        non_null = df[col].notna().sum()
        non_zero = (df[col] != 0).sum()
        if non_null == 0 or non_zero == 0:
            return False
    return True


def _compute_player_rolling_stats(
    player_history: pd.DataFrame,
    config: FPLFeatureConfig,
    target_gameweek: int,
    stat_cols: list[str]
) -> dict[str, float | None]:
    """Compute rolling statistics for a player up to a target gameweek.

    Args:
        player_history: Player's gameweek history (all seasons).
        config: Feature configuration.
        target_gameweek: Target gameweek (exclusive upper bound).
        stat_cols: List of stat columns to compute rolling means for.

    Returns:
        Dictionary of rolling statistics.
    """
    # Filter to gameweeks before target
    history = player_history[
        (player_history[config.gameweek_col] < target_gameweek) &
        (player_history["minutes"] > 0)  # Only appearances
    ].copy()

    if len(history) == 0:
        return {f"{col}_rolling_{config.rolling_window}": None for col in stat_cols}

    # Sort by season and gameweek
    history = history.sort_values(["season_id", config.gameweek_col])

    # Take most recent `rolling_window` gameweeks
    recent = history.tail(config.rolling_window)

    result = {}
    for col in stat_cols:
        if col in recent.columns:
            vals = recent[col].dropna()
            if len(vals) > 0:
                result[f"{col}_rolling_{config.rolling_window}"] = float(vals.mean())
            else:
                result[f"{col}_rolling_{config.rolling_window}"] = None
    return result


def _compute_player_season_stats(
    player_season_df: pd.DataFrame,
    config: FPLFeatureConfig,
    target_season_id: int,
    stat_cols: list[str]
) -> dict[str, float | None]:
    """Compute prior-season statistics for a player.

    Args:
        player_season_df: Player's season-level history (all seasons).
        config: Feature configuration.
        target_season_id: Target season ID (exclusive upper bound).
        stat_cols: List of stat columns to compute means for.

    Returns:
        Dictionary of prior-season statistics.
    """
    # Filter to seasons before target
    history = player_season_df[player_season_df["season_id"] < target_season_id].copy()

    if len(history) == 0:
        return {f"{col}_season_mean": None for col in stat_cols}

    # Sort by season
    history = history.sort_values("season_id")

    # Take most recent seasons
    recent = history.tail(config.rolling_window)

    result = {}
    for col in stat_cols:
        if col in recent.columns:
            vals = recent[col].dropna()
            if len(vals) > 0:
                result[f"{col}_season_mean"] = float(vals.mean())
            else:
                result[f"{col}_season_mean"] = None
    return result


def _compute_team_strength_features(
    player_match_df: pd.DataFrame,
    config: FPLFeatureConfig,
    target_season_id: int,
    target_gameweek: int,
    team_name: str,
    is_home: bool
) -> dict[str, float | None]:
    """Compute team strength features from match results.

    Args:
        player_match_df: FPL player-match (gameweek) data.
        config: Feature configuration.
        target_season_id: Target season ID.
        target_gameweek: Target gameweek.
        team_name: Team name.
        is_home: Whether computing for home team.

    Returns:
        Dictionary of team strength features.
    """
    # Filter matches for this team in this season before target gameweek
    if is_home:
        # For home team, look at home matches
        venue_col = "was_home"
        score_for = "team_h_score"
        score_against = "team_a_score"
        venue_val = True
    else:
        # For away team, look at away matches
        venue_col = "was_home"
        score_for = "team_a_score"
        score_against = "team_h_score"
        venue_val = False

    team_matches = player_match_df[
        (player_match_df["season_id"] == target_season_id) &
        (player_match_df["team"] == team_name) &
        (player_match_df[venue_col] == venue_val) &
        (player_match_df["gameweek"] < target_gameweek) &
        (player_match_df["minutes"] > 0)
    ].copy()

    if len(team_matches) == 0:
        prefix = "home_" if is_home else "away_"
        return {
            f"{prefix}matches": 0.0,
            f"{prefix}wins": 0.0,
            f"{prefix}draws": 0.0,
            f"{prefix}losses": 0.0,
            f"{prefix}goals_scored": 0.0,
            f"{prefix}goals_conceded": 0.0,
        }

    team_matches = team_matches.sort_values("gameweek")

    wins = 0
    draws = 0
    losses = 0
    goals_scored = 0
    goals_conceded = 0

    for _, row in team_matches.iterrows():
        gf = row[score_for]
        ga = row[score_against]
        goals_scored += gf
        goals_conceded += ga
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    n_matches = len(team_matches)
    prefix = "home_" if is_home else "away_"
    return {
        f"{prefix}matches": float(n_matches),
        f"{prefix}wins": float(wins),
        f"{prefix}draws": float(draws),
        f"{prefix}losses": float(losses),
        f"{prefix}goals_scored": float(goals_scored),
        f"{prefix}goals_conceded": float(goals_conceded),
    }


def build_fpl_player_features(
    player_season_df: pd.DataFrame,
    player_match_df: pd.DataFrame,
    config: FPLFeatureConfig | None = None
) -> pd.DataFrame:
    """Build FPL player features for modeling.

    Creates rolling gameweek features, prior-season statistics, and team strength features
    for each player-gameweek observation.

    Args:
        player_season_df: FPL player-season DataFrame.
        player_match_df: FPL player-match (gameweek) DataFrame.
        config: Feature configuration.

    Returns:
        DataFrame with features for each player-gameweek.
    """
    if config is None:
        config = FPLFeatureConfig()

    # Handle empty DataFrames
    if len(player_season_df) == 0 or len(player_match_df) == 0:
        return pd.DataFrame()

    # Check feature availability per season
    xg_available = _check_feature_availability(
        player_season_df, XG_AVAILABILITY_CHECK_COLS)
    starts_available = _check_feature_availability(
        player_season_df, STARTS_AVAILABILITY_CHECK_COLS)
    position_team_available = _check_feature_availability(
        player_season_df, POSITION_TEAM_AVAILABILITY_CHECK_COLS)

    # Merge player_match with player_season to get season-level info
    # We need to ensure we have team and position for each gameweek
    player_match = player_match_df.copy()

    # Ensure gameweek is numeric
    player_match["gameweek"] = pd.to_numeric(
        player_match[config.gameweek_col], errors="coerce"
    ).astype("Int64")

    # Ensure season_id is numeric
    player_match["season_id"] = pd.to_numeric(
        player_match["season_id"], errors="coerce"
    ).astype("Int64")

    # Build features for each player-gameweek
    features_list = []

    # Group by player to compute rolling features efficiently
    for player_id, player_match_group in player_match.groupby(config.player_id_col):
        player_match_group = player_match_group.sort_values(
            ["season_id", config.gameweek_col]
        )

        # Get player's season data
        player_seasons = player_season_df[
            player_season_df[config.player_id_col] == player_id
        ].copy()

        for _, row in player_match_group.iterrows():
            if row["minutes"] == 0:
                continue  # Skip non-appearances

            season_id = row["season_id"]
            gameweek = row["gameweek"]
            team_name = row["team"]
            is_home = row["was_home"]

            # Build feature row
            feature_row = {
                "player_id": player_id,
                "season_id": season_id,
                "gameweek": gameweek,
                "team": team_name,
                "is_home": is_home,
                "minutes": row["minutes"],
                "goals": row["goals"],
                "assists": row["assists"],
                "xg": row.get("expected_goals", 0.0),
                "xa": row.get("expected_assists", 0.0),
                "starts": row.get("starts", 0),
                "position": row.get("position", None),
                "team": team_name,
                "was_home": is_home,
            }

            # Rolling gameweek features
            stat_cols = ["goals", "assists", "xg", "xa", "minutes", "shots",
                         "shots_on_target", "key_passes", "threat", "creativity"]
            rolling_feats = _compute_player_rolling_stats(
                player_match_group, config, row["gameweek"], stat_cols
            )
            feature_row.update(rolling_feats)

            # Prior-season statistics
            season_stat_cols = ["goals", "assists", "xg", "xa", "minutes", "starts"]
            season_feats = _compute_player_season_stats(
                player_seasons, config, season_id, season_stat_cols
            )
            feature_row.update(season_feats)

            # Team strength features
            team_feats = _compute_team_strength_features(
                player_match_df, config, row["season_id"],
                row["gameweek"], team_name, is_home
            )
            feature_row.update(team_feats)

            # Position/team features if available
            if position_team_available:
                feature_row["position"] = row.get("position")
                feature_row["team"] = row.get("team")

            # xG/xA features if available
            if xg_available:
                feature_row["xg"] = row.get("expected_goals", 0.0)
                feature_row["xa"] = row.get("expected_assists", 0.0)
                feature_row["xg_per_90"] = row.get("expected_goals_per_90", 0.0)
                feature_row["xa_per_90"] = row.get("expected_assists_per_90", 0.0)

            # Starts feature if available
            if starts_available:
                feature_row["starts"] = row.get("starts", 0)
                feature_row["starts_rate"] = (
                    row.get("starts", 0) / row["gameweek"]
                    if row["gameweek"] > 0 else 0
                )

            features_list.append(feature_row)

    return pd.DataFrame(features_list)


def compute_fpl_features_for_cutoff(
    player_match_df: pd.DataFrame,
    player_season_df: pd.DataFrame,
    cutoff_gameweek: int,
    config: FPLFeatureConfig | None = None
) -> pd.DataFrame:
    """Build FPL features up to a specific gameweek cutoff.

    Args:
        player_match_df: FPL player-match (gameweek) DataFrame.
        player_season_df: FPL player-season DataFrame.
        cutoff_gameweek: Gameweek to predict (exclusive upper bound).
        config: Feature configuration.

    Returns:
        DataFrame with features for gameweek == cutoff_gameweek.
    """
    features_df = build_fpl_player_features(
        player_season_df, player_match_df, config
    )
    return features_df[features_df["gameweek"] == cutoff_gameweek].copy()


# For backward compatibility with existing code structure
def build_fpl_features(
    player_season_df: pd.DataFrame,
    player_match_df: pd.DataFrame,
    config: FPLFeatureConfig | None = None
) -> pd.DataFrame:
    """Alias for build_fpl_player_features."""
    return build_fpl_player_features(player_season_df, player_match_df, config)


# Column definitions for feature availability checks
XG_AVAILABILITY_CHECK_COLS = ["expected_goals", "expected_assists"]
STARTS_AVAILABILITY_CHECK_COLS = ["starts"]
POSITION_TEAM_AVAILABILITY_CHECK_COLS = ["position", "team"]


def _check_feature_availability(
    df: pd.DataFrame,
    check_cols: list[str]
) -> bool:
    """Check if feature columns have meaningful (non-zero, non-null) data."""
    if not check_cols:
        return False
    for col in check_cols:
        if col not in df.columns:
            return False
        # Check if column has any non-zero, non-null values
        non_null = df[col].notna().sum()
        non_zero = (df[col] != 0).sum()
        if non_null == 0 or non_zero == 0:
            return False
    return True


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, "/home/ganesh/projects/premier-league-intelligence")
    from src.dashboard.data_loader import load_player_season_fpl, load_player_match_fpl

    ps = load_player_season_fpl()
    pm = load_player_match_fpl()

    features = build_fpl_player_features(ps, pm)
    print(f"Features shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    print(f"Seasons: {features['season_name'].unique()}")
    print(f"Sample: {features.head(2).to_string()}")