"""Player feature engineering for Phase 3.

Pure, offline-testable functions that compute player-level features from historical
match data. All features are computed exclusively from data available before the
target matchweek (match_week < target_matchweek). No future information is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.data import config


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature computation."""
    min_history: int = 3
    recent_window: int = 3
    position_grouping: Literal["raw", "banded"] = "banded"


# Position band mapping for grouping sparse positions
POSITION_BANDS = {
    "Goalkeeper": "GK",
    "Left Center Back": "DEF",
    "Right Center Back": "DEF",
    "Center Back": "DEF",
    "Left Back": "DEF",
    "Right Back": "DEF",
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


def get_position_band(position: str | None) -> str:
    """Map a raw position to its band (GK, DEF, MID, WING, FWD)."""
    if position is None:
        return "UNKNOWN"
    return POSITION_BANDS.get(position, "UNKNOWN")


def _prepare_history_with_matchweek(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    cutoff_week: int
) -> pd.DataFrame:
    """Merge match_week into player_match and filter to history before cutoff_week."""
    # player_match may already have match_date; merge all needed columns from fixtures
    # Using suffixes to handle potential overlap
    merge_cols = ["match_id", "match_week", "match_date", "home_team_id", "away_team_id", "home_score", "away_score"]
    pm = player_match.merge(
        fixtures[merge_cols],
        on="match_id",
        how="left",
        validate="m:1",
        suffixes=("", "_fixture")
    )
    if pm["match_week"].isna().any():
        raise ValueError("Missing match_week for some player_match rows")
    # If both have match_date, use player_match's (no suffix) and drop fixture's
    if "match_date_fixture" in pm.columns:
        pm = pm.drop(columns=["match_date_fixture"])
    # Add position_band for grouping
    pm["position_band"] = pm["position"].apply(get_position_band)
    return pm[pm["match_week"] < cutoff_week].copy()


def _prepare_fixtures_history(fixtures: pd.DataFrame, cutoff_week: int) -> pd.DataFrame:
    """Filter fixtures to completed matches before cutoff_week."""
    return fixtures[fixtures["match_week"] < cutoff_week].copy()


def _get_player_history(history: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """Get player's historical matches sorted by match_week."""
    return history[history["player_id"] == player_id].sort_values("match_week")


def _compute_per_90_rate(total_stat: float, total_minutes: int) -> float | None:
    """Compute per-90 rate, returning None if minutes == 0."""
    if total_minutes <= 0:
        return None
    return float(total_stat / total_minutes * 90)


def _fallback_rate(
    history: pd.DataFrame,
    stat_col: str,
    groupby_col: str | None = None,
    group_value: str | None = None
) -> float:
    """Compute fallback rate (per-90) from group or global."""
    if groupby_col and group_value is not None:
        subset = history[history[groupby_col] == group_value]
        if not subset.empty and subset["minutes"].sum() > 0:
            return float(subset[stat_col].sum() / subset["minutes"].sum() * 90)
    # Global fallback
    if history.empty or history["minutes"].sum() == 0:
        return 0.0
    return float(history[stat_col].sum() / history["minutes"].sum() * 90)


def _fallback_match_mean(
    history: pd.DataFrame,
    stat_col: str,
    groupby_col: str | None = None,
    group_value: str | None = None
) -> float:
    """Compute fallback match-level mean from group or global."""
    if groupby_col and group_value is not None:
        subset = history[history[groupby_col] == group_value]
        if not subset.empty:
            return float(subset[stat_col].mean())
    if history.empty:
        return 0.0
    return float(history[stat_col].mean())


def compute_player_features(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    target_match: pd.Series,
    player_id: int,
    config: FeatureConfig
) -> dict[str, float | str | None]:
    """Compute features for a single player at a target match.

    All features use only historical data (match_week < target matchweek).
    Returns a dictionary of feature_name -> value, with None for unavailable features
    and a 'fallback_info' key documenting which fallbacks were used.
    """
    target_matchweek = int(target_match["match_week"])
    target_team_id = int(target_match["team_id"])
    target_match_id = int(target_match["match_id"])
    target_match_date = target_match["match_date"]

    # Prepare history data
    history = _prepare_history_with_matchweek(player_match, fixtures, target_matchweek)
    fixtures_history = _prepare_fixtures_history(fixtures, target_matchweek)

    player_history = _get_player_history(history, player_id)

    # Track fallback usage
    fallback_info = {}

    # Initialize features
    features = {}

    # --- Player history features ---
    if len(player_history) >= config.min_history and player_history["minutes"].sum() > 0:
        total_minutes = player_history["minutes"].sum()
        features["player_goals_per_90"] = _compute_per_90_rate(player_history["goals"].sum(), total_minutes)
        features["player_assists_per_90"] = _compute_per_90_rate(player_history["assists"].sum(), total_minutes)
        fallback_info["player_rates"] = "player_history"
    else:
        # Fallback: position band league median per-90
        pos_band = None
        if not player_history.empty:
            pos_band = get_position_band(player_history.iloc[-1]["position"])
        elif "position" in target_match and pd.notna(target_match["position"]):
            pos_band = get_position_band(target_match["position"])

        features["player_goals_per_90"] = _fallback_rate(
            history, "goals", "position_band", pos_band
        ) if pos_band and pos_band != "UNKNOWN" else _fallback_rate(history, "goals")
        features["player_assists_per_90"] = _fallback_rate(
            history, "assists", "position_band", pos_band
        ) if pos_band and pos_band != "UNKNOWN" else _fallback_rate(history, "assists")
        fallback_info["player_rates"] = f"position_band_{pos_band}" if pos_band and pos_band != "UNKNOWN" else "global"

    # Recent form (last N matches)
    recent = player_history.tail(config.recent_window)
    if len(recent) >= 1:
        features["player_recent_goals_mean_3"] = float(recent["goals"].mean())
        features["player_recent_assists_mean_3"] = float(recent["assists"].mean())
        fallback_info["recent_form"] = f"player_last_{len(recent)}"
    else:
        # Fallback to player match mean or global
        if len(player_history) >= 1:
            features["player_recent_goals_mean_3"] = float(player_history["goals"].mean())
            features["player_recent_assists_mean_3"] = float(player_history["assists"].mean())
            fallback_info["recent_form"] = "player_history_mean"
        else:
            features["player_recent_goals_mean_3"] = _fallback_match_mean(history, "goals")
            features["player_recent_assists_mean_3"] = _fallback_match_mean(history, "assists")
            fallback_info["recent_form"] = "global"

    # Minutes per match and starts rate
    if len(player_history) >= 1:
        features["player_minutes_per_match"] = float(player_history["minutes"].mean())
        features["player_starts_rate"] = float(player_history["started"].mean())
        fallback_info["availability"] = "player_history"
    else:
        # Fallback to position/team or global
        pos_band = None
        if not player_history.empty:
            pos_band = get_position_band(player_history.iloc[-1]["position"])
        elif "position" in target_match and pd.notna(target_match["position"]):
            pos_band = get_position_band(target_match["position"])

        pos_team_history = history[
            (history["team_id"] == target_team_id) &
            (history["position_band"] == pos_band) if pos_band and pos_band != "UNKNOWN" else (history["team_id"] == target_team_id)
        ]
        if pos_band and pos_band != "UNKNOWN" and "position_band" in history.columns:
            pos_team_history = history[
                (history["team_id"] == target_team_id) &
                (history["position_band"] == pos_band)
            ]

        if not pos_team_history.empty:
            features["player_minutes_per_match"] = float(pos_team_history["minutes"].mean())
            features["player_starts_rate"] = float(pos_team_history["started"].mean())
            fallback_info["availability"] = f"position_team_{pos_band}"
        else:
            team_history = history[history["team_id"] == target_team_id]
            if not team_history.empty:
                features["player_minutes_per_match"] = float(team_history["minutes"].mean())
                features["player_starts_rate"] = float(team_history["started"].mean())
                fallback_info["availability"] = "team"
            else:
                features["player_minutes_per_match"] = float(history["minutes"].mean()) if not history.empty else 90.0
                features["player_starts_rate"] = float(history["started"].mean()) if not history.empty else 1.0
                fallback_info["availability"] = "global"

    # --- Team context features ---
    # Team goals per match (from fixtures history)
    team_fixtures = fixtures_history[
        (fixtures_history["home_team_id"] == target_team_id) | (fixtures_history["away_team_id"] == target_team_id)
    ]
    if not team_fixtures.empty:
        team_goals = []
        for _, fx in team_fixtures.iterrows():
            if fx["home_team_id"] == target_team_id:
                team_goals.append(fx["home_score"])
            else:
                team_goals.append(fx["away_score"])
        features["team_goals_per_match"] = float(np.mean(team_goals))
        fallback_info["team_goals"] = "team_history"
    else:
        features["team_goals_per_match"] = _fallback_match_mean(
            pd.DataFrame({"goals": [fx["home_score"] if fx["home_team_id"] == target_team_id else fx["away_score"]
                                    for _, fx in fixtures_history.iterrows()]}), "goals"
        ) if not fixtures_history.empty else 0.0
        fallback_info["team_goals"] = "global"

    # Team xG per match - only if player_match has xG aggregated per team per match
    # For now, compute from player_match team aggregates (requires team xG per match)
    team_match_xg = history[history["team_id"] == target_team_id].groupby("match_id")["xg"].sum()
    if not team_match_xg.empty:
        features["team_xg_per_match"] = float(team_match_xg.mean())
        fallback_info["team_xg"] = "team_history"
    else:
        features["team_xg_per_match"] = 0.0
        fallback_info["team_xg"] = "unavailable"

    # Opponent goals conceded per match
    opponent_team_id = int(target_match["away_team_id"]) if target_match["home_team_id"] == target_team_id else int(target_match["home_team_id"])
    opponent_fixtures = fixtures_history[
        (fixtures_history["home_team_id"] == opponent_team_id) | (fixtures_history["away_team_id"] == opponent_team_id)
    ]
    if not opponent_fixtures.empty:
        opp_conceded = []
        for _, fx in opponent_fixtures.iterrows():
            if fx["home_team_id"] == opponent_team_id:
                opp_conceded.append(fx["away_score"])
            else:
                opp_conceded.append(fx["home_score"])
        features["opponent_goals_conceded_per_match"] = float(np.mean(opp_conceded))
        fallback_info["opponent_conceded"] = "opponent_history"
    else:
        features["opponent_goals_conceded_per_match"] = _fallback_match_mean(
            pd.DataFrame({"goals_conceded": [fx["away_score"] if fx["home_team_id"] == opponent_team_id else fx["home_score"]
                                                for _, fx in fixtures_history.iterrows()]}), "goals_conceded"
        ) if not fixtures_history.empty else 0.0
        fallback_info["opponent_conceded"] = "global"

    # --- Position context ---
    # League-wide position band goals per 90
    pos_band = None
    if not player_history.empty:
        pos_band = get_position_band(player_history.iloc[-1]["position"])
    elif "position" in target_match and pd.notna(target_match["position"]):
        pos_band = get_position_band(target_match["position"])

    if pos_band and pos_band != "UNKNOWN":
        # Add position_band to history for this computation
        history_with_band = history.copy()
        history_with_band["position_band"] = history_with_band["position"].apply(get_position_band)
        pos_players = history_with_band[history_with_band["position_band"] == pos_band]
        if not pos_players.empty and pos_players["minutes"].sum() > 0:
            features["position_goals_per_90_league"] = float(pos_players["goals"].sum() / pos_players["minutes"].sum() * 90)
            fallback_info["position_rate"] = f"position_band_{pos_band}"
        else:
            features["position_goals_per_90_league"] = _fallback_rate(history, "goals")
            fallback_info["position_rate"] = "global"
    else:
        features["position_goals_per_90_league"] = _fallback_rate(history, "goals")
        fallback_info["position_rate"] = "global"

    # --- Fixture context ---
    # Home/away
    features["is_home"] = 1.0 if target_match["home_team_id"] == target_team_id else 0.0

    # Days rest (days since player's last appearance)
    if len(player_history) >= 1:
        last_match = player_history.iloc[-1]
        last_match_date = last_match["match_date"]
        if pd.notna(last_match_date) and pd.notna(target_match_date):
            days_rest = (target_match_date - last_match_date).days
            features["days_rest"] = float(max(0, days_rest))
            fallback_info["days_rest"] = "player_history"
        else:
            features["days_rest"] = 7.0  # Default week
            fallback_info["days_rest"] = "default"
    else:
        features["days_rest"] = 7.0
        fallback_info["days_rest"] = "default"

    # Add metadata
    features["fallback_info"] = str(fallback_info)
    features["player_history_matches"] = len(player_history)
    features["player_history_minutes"] = int(player_history["minutes"].sum()) if not player_history.empty else 0

    return features


def compute_features_for_cutoff(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    cutoff_week: int,
    config: FeatureConfig
) -> pd.DataFrame:
    """Compute features for all players in a target matchweek.

    Args:
        player_match: Full player_match DataFrame.
        fixtures: Full fixtures DataFrame.
        cutoff_week: Target matchweek (1-38).
        config: FeatureConfig.

    Returns:
        DataFrame with one row per player in target matchweek, containing features.
    """
    # Get target matchweek data - player_match already has match_date
    target_pm = player_match.merge(
        fixtures[["match_id", "match_week", "home_team_id", "away_team_id"]],
        on="match_id",
        how="left",
        validate="m:1"
    )
    target_pm = target_pm[target_pm["match_week"] == cutoff_week].copy()

    if target_pm.empty:
        return pd.DataFrame()

    # Add position_band to target for consistency
    target_pm["position_band"] = target_pm["position"].apply(get_position_band)

    all_features = []
    for _, row in target_pm.iterrows():
        # Ensure match_date is present (from player_match)
        target_row = row.copy()
        if "match_date" not in target_row.index:
            # Should not happen since player_match has match_date
            target_row["match_date"] = pd.NaT
        feats = compute_player_features(player_match, fixtures, target_row, row["player_id"], config)
        feats["player_id"] = row["player_id"]
        feats["player_name"] = row["player_name"]
        feats["match_id"] = row["match_id"]
        feats["match_week"] = cutoff_week
        feats["team_id"] = row["team_id"]
        feats["position"] = row["position"]
        feats["position_band"] = row["position_band"]
        feats["minutes_actual"] = row["minutes"]
        feats["goals_actual"] = row["goals"]
        feats["assists_actual"] = row["assists"]
        all_features.append(feats)

    return pd.DataFrame(all_features)


def add_position_band_column(df: pd.DataFrame, position_col: str = "position") -> pd.DataFrame:
    """Add position_band column to a DataFrame."""
    df = df.copy()
    df["position_band"] = df[position_col].apply(get_position_band)
    return df