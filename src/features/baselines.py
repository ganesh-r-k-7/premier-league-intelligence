"""Baseline implementations for goals and assists forecasting.

All baselines produce expected match-level goals and assists counts (not per-90 rates).
Rates may be used as intermediate calculations but must be converted to expected counts
using expected minutes. Never use actual target-match minutes for pre-match predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for baseline computations."""
    min_history: int = 3
    recent_window: int = 5
    expected_minutes_method: Literal["position_team_median", "player_median", "fixed_90"] = "position_team_median"


def compute_global_mean(history: pd.DataFrame) -> tuple[float, float]:
    """Return (goals_mean, assists_mean) from history DataFrame.

    Args:
        history: DataFrame with columns 'goals', 'assists' (all players, all positions).

    Returns:
        Tuple of (global_goals_mean, global_assists_mean). Returns (0.0, 0.0) if history is empty.
    """
    if history.empty:
        return (0.0, 0.0)
    goals_mean = history["goals"].mean()
    assists_mean = history["assists"].mean()
    return (float(goals_mean), float(assists_mean))


def compute_player_mean(
    history: pd.DataFrame,
    player_id: int,
    min_history: int,
    global_fallback: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Return (goals_mean, assists_mean) for player; fallback to global if insufficient history.

    Args:
        history: DataFrame with columns 'player_id', 'goals', 'assists'.
        player_id: Player identifier.
        min_history: Minimum number of historical player-match rows required.
        global_fallback: Pre-computed global mean to use as fallback. If None, computed internally.

    Returns:
        Tuple of (player_goals_mean, player_assists_mean) or global fallback.
    """
    player_history = history[history["player_id"] == player_id]
    if len(player_history) >= min_history:
        goals_mean = player_history["goals"].mean()
        assists_mean = player_history["assists"].mean()
        return (float(goals_mean), float(assists_mean))

    if global_fallback is None:
        global_fallback = compute_global_mean(history)
    return global_fallback


def compute_recent_form(
    history: pd.DataFrame,
    player_id: int,
    window: int,
    global_fallback: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Return (goals_mean, assists_mean) from recent window; chained fallback.

    Args:
        history: DataFrame with columns 'player_id', 'match_week', 'goals', 'assists'.
        player_id: Player identifier.
        window: Number of most recent matches to consider.
        global_fallback: Pre-computed global mean for ultimate fallback.

    Returns:
        Tuple of (recent_goals_mean, recent_assists_mean) or chained fallback.
    """
    player_history = history[history["player_id"] == player_id].sort_values("match_week", ascending=False)
    recent = player_history.head(window)

    if len(recent) >= 1:
        goals_mean = recent["goals"].mean()
        assists_mean = recent["assists"].mean()
        return (float(goals_mean), float(assists_mean))

    # Chained fallback: player mean -> global mean
    player_mean = compute_player_mean(history, player_id, min_history=1, global_fallback=global_fallback)
    return player_mean


def _compute_position_team_median_minutes(
    history: pd.DataFrame,
    target_team_id: int,
    target_position: str | None,
    global_median: float
) -> float:
    """Compute median minutes for players at same position on same team in history."""
    if target_position is None:
        team_players = history[history["team_id"] == target_team_id]
        if team_players.empty or team_players["minutes"].sum() == 0:
            return global_median
        return float(team_players[team_players["minutes"] > 0]["minutes"].median())

    pos_team_players = history[
        (history["team_id"] == target_team_id) &
        (history["position"] == target_position)
    ]
    if not pos_team_players.empty and pos_team_players["minutes"].sum() > 0:
        return float(pos_team_players[pos_team_players["minutes"] > 0]["minutes"].median())

    # Fallback: team median across all positions
    team_players = history[history["team_id"] == target_team_id]
    if not team_players.empty and team_players["minutes"].sum() > 0:
        return float(team_players[team_players["minutes"] > 0]["minutes"].median())

    # Fallback: global median for that position
    pos_players = history[history["position"] == target_position]
    if not pos_players.empty and pos_players["minutes"].sum() > 0:
        return float(pos_players[pos_players["minutes"] > 0]["minutes"].median())

    return global_median


def _compute_player_median_minutes(
    history: pd.DataFrame,
    player_id: int,
    min_history: int
) -> float | None:
    """Compute player's own historical median minutes (requires minutes > 0)."""
    player_history = history[
        (history["player_id"] == player_id) &
        (history["minutes"] > 0)
    ]
    if len(player_history) >= min_history:
        return float(player_history["minutes"].median())
    return None


def _global_median_minutes(history: pd.DataFrame) -> float:
    """Compute global median minutes across all players with minutes > 0."""
    positive_minutes = history[history["minutes"] > 0]["minutes"]
    if positive_minutes.empty:
        return 90.0
    return float(positive_minutes.median())


def estimate_expected_minutes(
    history: pd.DataFrame,
    fixtures_history: pd.DataFrame,
    target_match: pd.Series,
    player_id: int,
    method: Literal["position_team_median", "player_median", "fixed_90"],
    min_history: int = 3
) -> float:
    """Estimate expected minutes for player in target_match using only history.

    Never uses target_match minutes. Returns float >= 0.

    Args:
        history: player_match history DataFrame (match_week < cutoff).
        fixtures_history: fixtures history DataFrame (match_week < cutoff).
        target_match: Row from fixtures for the target match (has team_id, etc.).
        player_id: Player identifier.
        method: Expected minutes estimation method.
        min_history: Minimum history for player_median method.

    Returns:
        Estimated expected minutes (float).
    """
    if method == "fixed_90":
        return 90.0

    global_median = _global_median_minutes(history)
    target_team_id = target_match["team_id"]
    target_position = None

    # Get player's position from target match lineup (known pre-match)
    player_history = history[history["player_id"] == player_id]
    if not player_history.empty:
        target_position = player_history.iloc[-1]["position"]

    if method == "position_team_median":
        return _compute_position_team_median_minutes(
            history, target_team_id, target_position, global_median
        )

    if method == "player_median":
        player_median = _compute_player_median_minutes(history, player_id, min_history)
        if player_median is not None:
            return player_median
        return 90.0  # Final fallback

    return 90.0


def _compute_player_rates(
    history: pd.DataFrame,
    player_id: int,
    min_history: int,
    global_rates: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Compute player's per-90 goals and assists rates.

    Returns (goals_per_90, assists_per_90). Falls back to player mean (match-level)
    converted to per-90, then global mean converted to per-90.
    """
    player_history = history[history["player_id"] == player_id]
    total_minutes = player_history["minutes"].sum()

    if len(player_history) >= min_history and total_minutes > 0:
        goals_per_90 = player_history["goals"].sum() / total_minutes * 90
        assists_per_90 = player_history["assists"].sum() / total_minutes * 90
        return (float(goals_per_90), float(assists_per_90))

    # Fallback chain: Player Mean (match-level) -> Global Mean (match-level)
    # Convert match-level means to per-90 using 90 minutes
    player_mean_goals, player_mean_assists = compute_player_mean(
        history, player_id, min_history=1, global_fallback=None
    )

    if player_mean_goals > 0 or player_mean_assists > 0:
        # Convert match-level mean to per-90 (assumes 90 min per match)
        return (player_mean_goals, player_mean_assists)

    # Final fallback: global mean converted to per-90
    if global_rates is None:
        global_goals_mean, global_assists_mean = compute_global_mean(history)
        global_rates = (global_goals_mean, global_assists_mean)

    return global_rates


def compute_minutes_adjusted(
    history: pd.DataFrame,
    fixtures_history: pd.DataFrame,
    target_match: pd.Series,
    player_id: int,
    config: BaselineConfig
) -> tuple[float, float]:
    """Return (expected_goals, expected_assists) using per-90 rate * expected_minutes / 90.

    Chained fallback: player rate -> global rate; expected_minutes -> fixed_90.

    Args:
        history: player_match history DataFrame (match_week < cutoff).
        fixtures_history: fixtures history DataFrame (match_week < cutoff).
        target_match: Row from fixtures for the target match.
        player_id: Player identifier.
        config: BaselineConfig with parameters.

    Returns:
        Tuple of (expected_goals, expected_assists) as match-level counts.
    """
    # Get global rates for fallback
    if history.empty or history["minutes"].sum() == 0:
        global_rates = (0.0, 0.0)
    else:
        global_goals_per_90 = history["goals"].sum() / history["minutes"].sum() * 90
        global_assists_per_90 = history["assists"].sum() / history["minutes"].sum() * 90
        global_rates = (float(global_goals_per_90), float(global_assists_per_90))

    # Player per-90 rates with fallback
    goals_per_90, assists_per_90 = _compute_player_rates(
        history, player_id, config.min_history, global_rates
    )

    # Expected minutes with fallback
    try:
        expected_minutes = estimate_expected_minutes(
            history,
            fixtures_history,
            target_match,
            player_id,
            config.expected_minutes_method,
            config.min_history
        )
    except Exception:
        # Ultimate fallback: fixed 90
        expected_minutes = 90.0

    expected_minutes = max(0.0, float(expected_minutes))

    expected_goals = goals_per_90 * expected_minutes / 90.0
    expected_assists = assists_per_90 * expected_minutes / 90.0

    return (expected_goals, expected_assists)


def evaluate_baselines_at_cutoff(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    cutoff_week: int,
    config: BaselineConfig,
    population: Literal["all_lineup", "played_only"]
) -> pd.DataFrame:
    """Run all 4 baselines for a single cutoff week.

    Args:
        player_match: Full player_match DataFrame with match_week column.
        fixtures: Full fixtures DataFrame with match_week column.
        cutoff_week: Target matchweek (1-38).
        config: BaselineConfig with parameters.
        population: "all_lineup" (Population A) or "played_only" (Population B).

    Returns:
        DataFrame with columns:
        [match_id, player_id, player_name, team_id, position, minutes_actual,
         goals_actual, assists_actual,
         global_goals_pred, global_assists_pred,
         player_mean_goals_pred, player_mean_assists_pred,
         recent_form_goals_pred, recent_form_assists_pred,
         minutes_adj_goals_pred, minutes_adj_assists_pred,
         population, cutoff_week]
    """
    # Split history and target
    history = player_match[player_match["match_week"] < cutoff_week].copy()
    target = player_match[player_match["match_week"] == cutoff_week].copy()
    fixtures_history = fixtures[fixtures["match_week"] < cutoff_week].copy()
    target_fixtures = fixtures[fixtures["match_week"] == cutoff_week].copy()

    if target.empty:
        return pd.DataFrame()

    # Filter population
    if population == "played_only":
        target = target[target["minutes"] > 0].copy()

    if target.empty:
        return pd.DataFrame()

    # Pre-compute global mean for this cutoff
    global_goals_mean, global_assists_mean = compute_global_mean(history)

    # Pre-compute target match lookup for expected minutes
    target_match_lookup = target_fixtures.set_index("match_id")

    rows = []
    for _, row in target.iterrows():
        player_id = row["player_id"]
        match_id = row["match_id"]
        target_match_info = target_match_lookup.loc[match_id]

        # Global mean
        global_goals_pred, global_assists_pred = global_goals_mean, global_assists_mean

        # Player historical mean
        player_mean_goals_pred, player_mean_assists_pred = compute_player_mean(
            history, player_id, config.min_history, (global_goals_mean, global_assists_mean)
        )

        # Recent form
        recent_form_goals_pred, recent_form_assists_pred = compute_recent_form(
            history, player_id, config.recent_window, (global_goals_mean, global_assists_mean)
        )

        # Minutes-adjusted
        minutes_adj_goals_pred, minutes_adj_assists_pred = compute_minutes_adjusted(
            history, fixtures_history, target_match_info, player_id, config
        )

        rows.append({
            "match_id": match_id,
            "player_id": player_id,
            "player_name": row["player_name"],
            "team_id": row["team_id"],
            "position": row["position"],
            "minutes_actual": row["minutes"],
            "goals_actual": row["goals"],
            "assists_actual": row["assists"],
            "global_goals_pred": global_goals_pred,
            "global_assists_pred": global_assists_pred,
            "player_mean_goals_pred": player_mean_goals_pred,
            "player_mean_assists_pred": player_mean_assists_pred,
            "recent_form_goals_pred": recent_form_goals_pred,
            "recent_form_assists_pred": recent_form_assists_pred,
            "minutes_adj_goals_pred": minutes_adj_goals_pred,
            "minutes_adj_assists_pred": minutes_adj_assists_pred,
            "population": population,
            "cutoff_week": cutoff_week,
        })

    return pd.DataFrame(rows)