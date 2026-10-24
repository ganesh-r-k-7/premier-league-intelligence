"""Match feature engineering for Phase 4.

Pure, offline-testable functions that compute match-level features from historical
match data. All features are computed exclusively from data available before the
target matchweek (match_week < target_matchweek). No future information is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchFeatureConfig:
    """Configuration for match feature computation."""
    rolling_window: int = 5              # Number of previous matches for rolling features
    min_history: int = 3                 # Minimum matches required for team-specific features
    h2h_window: int = 5                  # Number of previous H2H meetings to consider
    include_xg: bool = True              # Whether to include xG/xGA features (if available)
    include_form: bool = True            # Whether to include form features
    include_h2h: bool = True             # Whether to include head-to-head features
    include_rest: bool = True            # Whether to include rest days


def _prepare_history_with_matchweek(
    fixtures: pd.DataFrame,
    cutoff_week: int,
    season_id: int | None = None
) -> pd.DataFrame:
    """Filter fixtures to history before cutoff_week."""
    history = fixtures[fixtures["match_week"] < cutoff_week].copy()
    if season_id is not None:
        history = history[history["season_id"] == season_id].copy()
    return history.sort_values(["match_week", "match_id"])


def _compute_team_match_stats(
    history: pd.DataFrame,
    team_id: int,
    window: int
) -> dict[str, float | None]:
    """Compute rolling stats for a team from their perspective.

    Args:
        history: DataFrame with historical matches (match_week < cutoff).
        team_id: Team identifier.
        window: Number of most recent matches to consider.

    Returns:
        Dictionary of team stats from their perspective.
    """
    # Get team's matches (both home and away)
    team_matches = history[
        (history["home_team_id"] == team_id) | (history["away_team_id"] == team_id)
    ].sort_values("match_week", ascending=False).head(window)

    if len(team_matches) < 1:
        return {}

    wins = 0
    draws = 0
    losses = 0
    goals_scored = 0
    goals_conceded = 0
    xg_for = 0.0
    xg_against = 0.0

    for _, row in team_matches.iterrows():
        if row["home_team_id"] == team_id:
            # Team played at home
            goals_scored += row["home_score"]
            goals_conceded += row["away_score"]
            if row["home_score"] > row["away_score"]:
                wins += 1
            elif row["home_score"] == row["away_score"]:
                draws += 1
            else:
                losses += 1
            # xG from home perspective
            xg_for += row.get("home_xg", 0.0) or 0.0
            xg_against += row.get("away_xg", 0.0) or 0.0
        else:
            # Team played away
            goals_scored += row["away_score"]
            goals_conceded += row["home_score"]
            if row["away_score"] > row["home_score"]:
                wins += 1
            elif row["away_score"] == row["home_score"]:
                draws += 1
            else:
                losses += 1
            # xG from away perspective
            xg_for += row.get("away_xg", 0.0) or 0.0
            xg_against += row.get("home_xg", 0.0) or 0.0

    n_matches = len(team_matches)
    points = wins * 3 + draws

    result = {
        "matches": float(n_matches),
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "points": float(points),
        "points_per_match": points / n_matches if n_matches > 0 else 0.0,
        "goals_scored": float(goals_scored),
        "goals_conceded": float(goals_conceded),
        "goals_scored_per_match": goals_scored / n_matches if n_matches > 0 else 0.0,
        "goals_conceded_per_match": goals_conceded / n_matches if n_matches > 0 else 0.0,
        "goal_diff": float(goals_scored - goals_conceded),
        "goal_diff_per_match": (goals_scored - goals_conceded) / n_matches if n_matches > 0 else 0.0,
    }

    if xg_for > 0 or xg_against > 0:
        result["xg_for"] = float(xg_for)
        result["xg_against"] = float(xg_against)
        result["xg_for_per_match"] = xg_for / n_matches if n_matches > 0 else 0.0
        result["xg_against_per_match"] = xg_against / n_matches if n_matches > 0 else 0.0
        result["xg_diff"] = float(xg_for - xg_against)
        result["xg_diff_per_match"] = (xg_for - xg_against) / n_matches if n_matches > 0 else 0.0

    return result


def _compute_home_away_stats(
    history: pd.DataFrame,
    team_id: int,
    window: int
) -> dict[str, float]:
    """Compute home and away specific stats for a team.

    Args:
        history: DataFrame with historical matches.
        team_id: Team identifier.
        window: Number of most recent matches to consider.

    Returns:
        Dictionary of home/away specific stats.
    """
    # Home matches
    home_matches = history[history["home_team_id"] == team_id].sort_values("match_week", ascending=False).head(window)
    # Away matches
    away_matches = history[history["away_team_id"] == team_id].sort_values("match_week", ascending=False).head(window)

    # Home stats
    home_wins = 0
    home_draws = 0
    home_losses = 0
    home_goals_scored = 0
    home_goals_conceded = 0
    home_xg_for = 0.0
    home_xg_against = 0.0

    for _, row in home_matches.iterrows():
        home_goals_scored += row["home_score"]
        home_goals_conceded += row["away_score"]
        if row["home_score"] > row["away_score"]:
            home_wins += 1
        elif row["home_score"] == row["away_score"]:
            home_draws += 1
        else:
            home_losses += 1
        home_xg_for += row.get("home_xg", 0.0) or 0.0
        home_xg_against += row.get("away_xg", 0.0) or 0.0

    home_n = len(home_matches)
    home_points = home_wins * 3 + home_draws

    # Away stats
    away_wins = 0
    away_draws = 0
    away_losses = 0
    away_goals_scored = 0
    away_goals_conceded = 0
    away_xg_for = 0.0
    away_xg_against = 0.0

    for _, row in away_matches.iterrows():
        away_goals_scored += row["away_score"]
        away_goals_conceded += row["home_score"]
        if row["away_score"] > row["home_score"]:
            away_wins += 1
        elif row["away_score"] == row["home_score"]:
            away_draws += 1
        else:
            away_losses += 1
        away_xg_for += row.get("away_xg", 0.0) or 0.0
        away_xg_against += row.get("home_xg", 0.0) or 0.0

    away_n = len(away_matches)
    away_points = away_wins * 3 + away_draws

    result = {
        "home_matches": float(home_n),
        "home_wins": float(home_wins),
        "home_draws": float(home_draws),
        "home_losses": float(home_losses),
        "home_points": float(home_points),
        "home_points_per_match": home_points / home_n if home_n > 0 else 0.0,
        "home_goals_scored": float(home_goals_scored),
        "home_goals_conceded": float(home_goals_conceded),
        "home_goals_scored_per_match": home_goals_scored / home_n if home_n > 0 else 0.0,
        "home_goals_conceded_per_match": home_goals_conceded / home_n if home_n > 0 else 0.0,
        "away_matches": float(away_n),
        "away_wins": float(away_wins),
        "away_draws": float(away_draws),
        "away_losses": float(away_losses),
        "away_points": float(away_points),
        "away_points_per_match": away_points / away_n if away_n > 0 else 0.0,
        "away_goals_scored": float(away_goals_scored),
        "away_goals_conceded": float(away_goals_conceded),
        "away_goals_scored_per_match": away_goals_scored / away_n if away_n > 0 else 0.0,
        "away_goals_conceded_per_match": away_goals_conceded / away_n if away_n > 0 else 0.0,
    }

    if home_xg_for > 0 or home_xg_against > 0 or away_xg_for > 0 or away_xg_against > 0:
        result["home_xg_for"] = float(home_xg_for)
        result["home_xg_against"] = float(home_xg_against)
        result["home_xg_for_per_match"] = home_xg_for / home_n if home_n > 0 else 0.0
        result["home_xg_against_per_match"] = home_xg_against / home_n if home_n > 0 else 0.0
        result["away_xg_for"] = float(away_xg_for)
        result["away_xg_against"] = float(away_xg_against)
        result["away_xg_for_per_match"] = away_xg_for / away_n if away_n > 0 else 0.0
        result["away_xg_against_per_match"] = away_xg_against / away_n if away_n > 0 else 0.0

    return result


def _compute_h2h_stats(
    history: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    window: int
) -> dict[str, float]:
    """Compute head-to-head stats between two teams.

    Only considers matches where these two teams played each other.

    Args:
        history: DataFrame with historical matches.
        home_team_id: Home team identifier.
        away_team_id: Away team identifier.
        window: Number of most recent H2H meetings to consider.

    Returns:
        Dictionary of H2H stats from home team's perspective.
    """
    h2h_matches = history[
        ((history["home_team_id"] == home_team_id) & (history["away_team_id"] == away_team_id)) |
        ((history["home_team_id"] == away_team_id) & (history["away_team_id"] == home_team_id))
    ].sort_values("match_week", ascending=False).head(window)

    if len(h2h_matches) < 1:
        return {}

    h_wins = 0
    h_draws = 0
    h_losses = 0
    h_goals_scored = 0
    h_goals_conceded = 0
    h_xg_for = 0.0
    h_xg_against = 0.0

    for _, row in h2h_matches.iterrows():
        if row["home_team_id"] == home_team_id:
            # Home team was at home in this H2H match
            h_goals_scored += row["home_score"]
            h_goals_conceded += row["away_score"]
            if row["home_score"] > row["away_score"]:
                h_wins += 1
            elif row["home_score"] == row["away_score"]:
                h_draws += 1
            else:
                h_losses += 1
            h_xg_for += row.get("home_xg", 0.0) or 0.0
            h_xg_against += row.get("away_xg", 0.0) or 0.0
        else:
            # Home team was away in this H2H match
            h_goals_scored += row["away_score"]
            h_goals_conceded += row["home_score"]
            if row["away_score"] > row["home_score"]:
                h_wins += 1
            elif row["away_score"] == row["home_score"]:
                h_draws += 1
            else:
                h_losses += 1
            h_xg_for += row.get("away_xg", 0.0) or 0.0
            h_xg_against += row.get("home_xg", 0.0) or 0.0

    n_matches = len(h2h_matches)
    h_points = h_wins * 3 + h_draws

    result = {
        "h2h_matches": float(n_matches),
        "h2h_wins": float(h_wins),
        "h2h_draws": float(h_draws),
        "h2h_losses": float(h_losses),
        "h2h_points": float(h_points),
        "h2h_points_per_match": h_points / n_matches if n_matches > 0 else 0.0,
        "h2h_goals_scored": float(h_goals_scored),
        "h2h_goals_conceded": float(h_goals_conceded),
        "h2h_goals_scored_per_match": h_goals_scored / n_matches if n_matches > 0 else 0.0,
        "h2h_goals_conceded_per_match": h_goals_conceded / n_matches if n_matches > 0 else 0.0,
        "h2h_goal_diff": float(h_goals_scored - h_goals_conceded),
    }

    if h_xg_for > 0 or h_xg_against > 0:
        result["h2h_xg_for"] = float(h_xg_for)
        result["h2h_xg_against"] = float(h_xg_against)
        result["h2h_xg_for_per_match"] = h_xg_for / n_matches if n_matches > 0 else 0.0
        result["h2h_xg_against_per_match"] = h_xg_against / n_matches if n_matches > 0 else 0.0
        result["h2h_xg_diff"] = float(h_xg_for - h_xg_against)
        result["h2h_xg_diff_per_match"] = (h_xg_for - h_xg_against) / n_matches if n_matches > 0 else 0.0

    return result


def _compute_rest_days(
    history: pd.DataFrame,
    fixtures_history: pd.DataFrame,
    team_id: int,
    target_match_date: pd.Timestamp
) -> float | None:
    """Compute days since team's last match.

    Args:
        history: Player/match history (not used, but kept for API consistency).
        fixtures_history: Fixtures history with match dates.
        team_id: Team identifier.
        target_match_date: Date of the target match.

    Returns:
        Days rest as float, or None if no prior match found.
    """
    team_fixtures = fixtures_history[
        (fixtures_history["home_team_id"] == team_id) | (fixtures_history["away_team_id"] == team_id)
    ].sort_values("match_date", ascending=False)

    if len(team_fixtures) < 1:
        return None

    last_match = team_fixtures.iloc[0]
    last_match_date = last_match["match_date"]

    if pd.isna(last_match_date) or pd.isna(target_match_date):
        return None

    days_rest = (target_match_date - last_match_date).days
    return float(max(0, days_rest))


def _get_global_fallbacks(history: pd.DataFrame) -> dict[str, float]:
    """Compute global fallback values from all historical matches.

    Args:
        history: Historical matches DataFrame.

    Returns:
        Dictionary of global average stats.
    """
    if len(history) < 1:
        return {}

    # Global averages from home team perspective
    global_home_goals = history["home_score"].mean()
    global_away_goals = history["away_score"].mean()
    global_home_xg = history.get("home_xg", pd.Series([0.0])).mean()
    global_away_xg = history.get("away_xg", pd.Series([0.0])).mean()

    # Home win/draw/away win rates
    home_wins = (history["home_score"] > history["away_score"]).mean()
    draws = (history["home_score"] == history["away_score"]).mean()
    away_wins = (history["home_score"] < history["away_score"]).mean()

    return {
        "global_home_goals_per_match": float(global_home_goals),
        "global_away_goals_per_match": float(global_away_goals),
        "global_home_xg_per_match": float(global_home_xg),
        "global_away_xg_per_match": float(global_away_xg),
        "global_home_win_rate": float(home_wins),
        "global_draw_rate": float(draws),
        "global_away_win_rate": float(away_wins),
    }


def compute_match_features(
    fixtures: pd.DataFrame,
    player_match: pd.DataFrame | None = None,
    cutoff_week: int = 1,
    season_id: int | None = None,
    config: MatchFeatureConfig | None = None
) -> pd.DataFrame:
    """Compute features for all matches in a target matchweek.

    Args:
        fixtures: Full fixtures DataFrame.
        player_match: Player match data for xG aggregation (optional).
        cutoff_week: Target matchweek (1-38).
        season_id: If provided, only compute features for this season.
        config: MatchFeatureConfig for feature computation.

    Returns:
        DataFrame with one row per match in target matchweek, containing features.
    """
    if config is None:
        config = MatchFeatureConfig()

    # Aggregate xG/xA from player_match to team-match level if provided
    if player_match is not None and config.include_xg:
        # Drop existing xG/xA columns from fixtures to avoid merge conflicts
        fixtures = fixtures.drop(
            columns=[c for c in ["home_xg", "home_xa", "away_xg", "away_xa"] if c in fixtures.columns],
            errors="ignore"
        )
        
        # Aggregate xG/xA per team per match
        team_xg = player_match.groupby(["match_id", "team_id"]).agg(
            xg_sum=("xg", "sum"),
            xa_sum=("xa", "sum"),
        ).reset_index()

        # Pivot to have home and away columns
        team_xg_home = team_xg.rename(columns={"team_id": "home_team_id", "xg_sum": "home_xg", "xa_sum": "home_xa"})
        team_xg_away = team_xg.rename(columns={"team_id": "away_team_id", "xg_sum": "away_xg", "xa_sum": "away_xa"})

        fixtures = fixtures.merge(
            team_xg_home[["match_id", "home_team_id", "home_xg", "home_xa"]],
            on=["match_id", "home_team_id"],
            how="left"
        ).merge(
            team_xg_away[["match_id", "away_team_id", "away_xg", "away_xa"]],
            on=["match_id", "away_team_id"],
            how="left"
        )
    else:
        # Ensure columns exist even if not computed
        for col in ["home_xg", "home_xa", "away_xg", "away_xa"]:
            if col not in fixtures.columns:
                fixtures[col] = 0.0

    # Get target matches
    if season_id is not None:
        target_fixtures = fixtures[
            (fixtures["match_week"] == cutoff_week) & (fixtures["season_id"] == season_id)
        ].copy()
    else:
        target_fixtures = fixtures[fixtures["match_week"] == cutoff_week].copy()

    if target_fixtures.empty:
        return pd.DataFrame()

    # Get history (matches before cutoff)
    history = _prepare_history_with_matchweek(fixtures, cutoff_week, season_id)

    # Global fallbacks
    global_fallbacks = _get_global_fallbacks(history)

    all_features = []

    for _, match in target_fixtures.iterrows():
        match_id = match["match_id"]
        home_team_id = match["home_team_id"]
        away_team_id = match["away_team_id"]
        match_date = match["match_date"]

        features = {
            "match_id": match_id,
            "match_week": cutoff_week,
            "season_id": match.get("season_id", -1),
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
        }

        # Home team features (full rolling window)
        if config.include_form:
            home_stats = _compute_team_match_stats(history, home_team_id, config.rolling_window)
            for k, v in home_stats.items():
                features[f"home_{k}"] = v

            # Home/away specific for home team (as home)
            home_home_stats = _compute_home_away_stats(history, home_team_id, config.rolling_window)
            for k, v in home_home_stats.items():
                features[f"home_{k}"] = v

        # Away team features (full rolling window)
        if config.include_form:
            away_stats = _compute_team_match_stats(history, away_team_id, config.rolling_window)
            for k, v in away_stats.items():
                features[f"away_{k}"] = v

            # Home/away specific for away team (as away)
            away_away_stats = _compute_home_away_stats(history, away_team_id, config.rolling_window)
            for k, v in away_away_stats.items():
                features[f"away_{k}"] = v

        # Head-to-head features
        if config.include_h2h:
            h2h_stats = _compute_h2h_stats(history, home_team_id, away_team_id, config.h2h_window)
            for k, v in h2h_stats.items():
                features[k] = v

        # Rest days
        if config.include_rest:
            home_rest = _compute_rest_days(history, fixtures, home_team_id, match_date)
            away_rest = _compute_rest_days(history, fixtures, away_team_id, match_date)
            features["home_rest_days"] = home_rest if home_rest is not None else 7.0  # Default 1 week
            features["away_rest_days"] = away_rest if away_rest is not None else 7.0
            features["rest_days_diff"] = features["home_rest_days"] - features["away_rest_days"]

        # Global fallbacks for missing values
        for k, v in global_fallbacks.items():
            # Only use as fallback if not already set
            if k not in features or features.get(k) is None:
                features[k] = v

        all_features.append(features)

    return pd.DataFrame(all_features)


def compute_match_features_for_cutoff(
    fixtures: pd.DataFrame,
    player_match: pd.DataFrame | None = None,
    cutoff_week: int = 1,
    season_id: int | None = None,
    config: MatchFeatureConfig | None = None
) -> pd.DataFrame:
    """Compute features for a specific cutoff week.

    Convenience wrapper that returns features with target labels attached.

    Args:
        fixtures: Full fixtures DataFrame.
        player_match: Player match data for xG aggregation.
        cutoff_week: Target matchweek.
        season_id: Season identifier.
        config: Feature configuration.

    Returns:
        DataFrame with features and target labels (home_score, away_score, outcome).
    """
    features_df = compute_match_features(
        fixtures, player_match, cutoff_week, season_id, config
    )

    if features_df.empty:
        return features_df

    # Merge with target data
    target = fixtures[fixtures["match_week"] == cutoff_week].copy()
    if season_id is not None:
        target = target[target["season_id"] == season_id]

    # Add target labels
    target["outcome"] = np.where(
        target["home_score"] > target["away_score"], 2,  # home win
        np.where(target["home_score"] == target["away_score"], 1, 0)  # draw / away win
    )

    result = features_df.merge(
        target[["match_id", "home_score", "away_score", "outcome"]],
        on="match_id",
        how="left"
    )

    return result