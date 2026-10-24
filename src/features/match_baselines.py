"""Baseline implementations for match outcome prediction.

Predicts probabilities for three classes: home_win, draw, away_win.
All baselines produce valid probability vectors (sum to 1, all >= 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchBaselineConfig:
    """Configuration for match outcome baseline computations."""
    min_history: int = 5              # Minimum matches for team-specific baselines
    smoothing_alpha: float = 1.0      # Laplace smoothing for probability estimation


def compute_global_outcome_frequencies(
    history: pd.DataFrame,
    smoothing_alpha: float = 1.0
) -> tuple[float, float, float]:
    """Return (home_win_prob, draw_prob, away_win_prob) from match history.

    Args:
        history: DataFrame with columns 'home_score', 'away_score'.
        smoothing_alpha: Laplace smoothing parameter (added to each count).

    Returns:
        Tuple of probabilities summing to 1.0.
    """
    # Filter out matches with missing scores
    valid_history = history.dropna(subset=["home_score", "away_score"])
    
    if valid_history.empty:
        return (1.0 / 3, 1.0 / 3, 1.0 / 3)

    home_wins = (valid_history["home_score"] > valid_history["away_score"]).sum()
    draws = (valid_history["home_score"] == valid_history["away_score"]).sum()
    away_wins = (valid_history["home_score"] < valid_history["away_score"]).sum()

    total = len(valid_history) + 3 * smoothing_alpha
    home_win_prob = (home_wins + smoothing_alpha) / total
    draw_prob = (draws + smoothing_alpha) / total
    away_win_prob = (away_wins + smoothing_alpha) / total

    return (float(home_win_prob), float(draw_prob), float(away_win_prob))


def compute_team_strength_baseline(
    history: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    min_history: int,
    global_probs: tuple[float, float, float],
    smoothing_alpha: float = 1.0
) -> tuple[float, float, float]:
    """Compute outcome probabilities based on team historical performance.

    Uses home team's home record and away team's away record.
    Falls back to global frequencies if insufficient history.

    Args:
        history: DataFrame with match results including team_ids and scores.
        home_team_id: Home team identifier.
        away_team_id: Away team identifier.
        min_history: Minimum matches required for team-specific estimate.
        global_probs: Pre-computed global probabilities for fallback.
        smoothing_alpha: Laplace smoothing parameter.

    Returns:
        Tuple of (home_win_prob, draw_prob, away_win_prob).
    """
    # Home team's home matches
    home_team_home = history[history["home_team_id"] == home_team_id]
    # Away team's away matches
    away_team_away = history[history["away_team_id"] == away_team_id]

    if len(home_team_home) >= min_history and len(away_team_away) >= min_history:
        # Combine perspectives: home team at home vs away team away
        # Home team home record
        h_home_wins = (home_team_home["home_score"] > home_team_home["away_score"]).sum()
        h_draws = (home_team_home["home_score"] == home_team_home["away_score"]).sum()
        h_away_wins = (home_team_home["home_score"] < home_team_home["away_score"]).sum()

        # Away team away record (from their perspective)
        a_home_wins = (away_team_away["home_score"] < away_team_away["away_score"]).sum()  # away team wins
        a_draws = (away_team_away["home_score"] == away_team_away["away_score"]).sum()
        a_away_wins = (away_team_away["home_score"] > away_team_away["away_score"]).sum()  # away team loses

        # Combine: average the two perspectives
        combined_home_wins = (h_home_wins + a_home_wins) / 2
        combined_draws = (h_draws + a_draws) / 2
        combined_away_wins = (h_away_wins + a_away_wins) / 2

        total = combined_home_wins + combined_draws + combined_away_wins + 3 * smoothing_alpha
        home_win_prob = (combined_home_wins + smoothing_alpha) / total
        draw_prob = (combined_draws + smoothing_alpha) / total
        away_win_prob = (combined_away_wins + smoothing_alpha) / total

        return (float(home_win_prob), float(draw_prob), float(away_win_prob))

    # Fallback: use global frequencies
    return global_probs


def compute_recent_form_baseline(
    history: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    window: int,
    global_probs: tuple[float, float, float],
    smoothing_alpha: float = 1.0
) -> tuple[float, float, float]:
    """Compute outcome probabilities based on recent form of both teams.

    Looks at the most recent `window` matches for each team (any venue).
    Falls back to global if insufficient history.

    Args:
        history: DataFrame with match results sorted by match_week.
        home_team_id: Home team identifier.
        away_team_id: Away team identifier.
        window: Number of recent matches to consider per team.
        global_probs: Pre-computed global probabilities for fallback.
        smoothing_alpha: Laplace smoothing parameter.

    Returns:
        Tuple of (home_win_prob, draw_prob, away_win_prob).
    """
    # Filter out matches with missing scores first
    valid_history = history.dropna(subset=["home_score", "away_score"])
    
    # Get recent matches for home team
    home_matches = valid_history[
        (valid_history["home_team_id"] == home_team_id) | (valid_history["away_team_id"] == home_team_id)
    ].sort_values("match_week", ascending=False).head(window)

    # Get recent matches for away team
    away_matches = valid_history[
        (valid_history["home_team_id"] == away_team_id) | (valid_history["away_team_id"] == away_team_id)
    ].sort_values("match_week", ascending=False).head(window)

    if len(home_matches) < 1 or len(away_matches) < 1:
        return global_probs

    # Compute results from each team's perspective
    def get_team_results(matches: pd.DataFrame, team_id: int) -> tuple[int, int, int]:
        """Return (wins, draws, losses) from team's perspective."""
        wins = 0
        draws = 0
        losses = 0
        for _, row in matches.iterrows():
            if row["home_team_id"] == team_id:
                if row["home_score"] > row["away_score"]:
                    wins += 1
                elif row["home_score"] == row["away_score"]:
                    draws += 1
                else:
                    losses += 1
            else:  # team is away
                if row["away_score"] > row["home_score"]:
                    wins += 1
                elif row["away_score"] == row["home_score"]:
                    draws += 1
                else:
                    losses += 1
        return wins, draws, losses

    h_wins, h_draws, h_losses = get_team_results(home_matches, home_team_id)
    a_wins, a_draws, a_losses = get_team_results(away_matches, away_team_id)

    # Combine: home team's wins + away team's losses favor home win
    # home team's losses + away team's wins favor away win
    combined_home_wins = h_wins + a_losses
    combined_draws = h_draws + a_draws
    combined_away_wins = h_losses + a_wins

    total = combined_home_wins + combined_draws + combined_away_wins + 3 * smoothing_alpha
    home_win_prob = (combined_home_wins + smoothing_alpha) / total
    draw_prob = (combined_draws + smoothing_alpha) / total
    away_win_prob = (combined_away_wins + smoothing_alpha) / total

    return (float(home_win_prob), float(draw_prob), float(away_win_prob))


def compute_home_advantage_baseline(
    history: pd.DataFrame,
    global_probs: tuple[float, float, float],
    smoothing_alpha: float = 1.0
) -> tuple[float, float, float]:
    """Compute outcome probabilities using only home advantage effect.

    Estimates P(home win), P(draw), P(away win) from historical home/away split
    without team-specific information.

    Args:
        history: DataFrame with match results.
        global_probs: Global frequencies (unused, kept for API consistency).
        smoothing_alpha: Laplace smoothing parameter.

    Returns:
        Tuple of (home_win_prob, draw_prob, away_win_prob).
    """
    return compute_global_outcome_frequencies(history, smoothing_alpha)


def evaluate_match_baselines_at_cutoff(
    fixtures: pd.DataFrame,
    cutoff_week: int,
    config: MatchBaselineConfig,
    season_id: int | None = None
) -> pd.DataFrame:
    """Run all match outcome baselines for a single cutoff week.

    Args:
        fixtures: Full fixtures DataFrame with match_week, season_id, scores, team_ids.
        cutoff_week: Target matchweek (1-38).
        config: MatchBaselineConfig with parameters.
        season_id: If provided, only evaluate matches from this season.

    Returns:
        DataFrame with columns:
        [match_id, match_week, season_id, home_team_id, away_team_id,
         home_score, away_score, outcome (0=away, 1=draw, 2=home),
         global_home_prob, global_draw_prob, global_away_prob,
         home_advantage_home_prob, home_advantage_draw_prob, home_advantage_away_prob,
         team_strength_home_prob, team_strength_draw_prob, team_strength_away_prob,
         recent_form_home_prob, recent_form_draw_prob, recent_form_away_prob,
         population, cutoff_week]
    """
    # Filter to target season if specified
    if season_id is not None:
        fixtures = fixtures[fixtures["season_id"] == season_id].copy()

    # Split history and target
    history = fixtures[fixtures["match_week"] < cutoff_week].copy()
    target = fixtures[fixtures["match_week"] == cutoff_week].copy()

    if target.empty:
        return pd.DataFrame()

    # Compute target labels
    target = target.copy()
    target["outcome"] = np.where(
        target["home_score"] > target["away_score"], 2,  # home win
        np.where(target["home_score"] == target["away_score"], 1, 0)  # draw / away win
    )

    # Pre-compute global frequencies for this cutoff
    global_probs = compute_global_outcome_frequencies(history, config.smoothing_alpha)

    rows = []
    for _, match in target.iterrows():
        match_id = match["match_id"]
        home_team_id = match["home_team_id"]
        away_team_id = match["away_team_id"]

        # Global frequencies baseline
        global_home, global_draw, global_away = global_probs

        # Home advantage baseline (same as global for this implementation)
        ha_home, ha_draw, ha_away = compute_home_advantage_baseline(
            history, global_probs, config.smoothing_alpha
        )

        # Team strength baseline
        ts_home, ts_draw, ts_away = compute_team_strength_baseline(
            history, home_team_id, away_team_id,
            config.min_history, global_probs, config.smoothing_alpha
        )

        # Recent form baseline
        rf_home, rf_draw, rf_away = compute_recent_form_baseline(
            history, home_team_id, away_team_id,
            config.min_history, global_probs, config.smoothing_alpha
        )

        rows.append({
            "match_id": match_id,
            "match_week": cutoff_week,
            "season_id": match.get("season_id", -1),
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_score": match["home_score"],
            "away_score": match["away_score"],
            "outcome": match["outcome"],
            "global_home_prob": global_home,
            "global_draw_prob": global_draw,
            "global_away_prob": global_away,
            "home_advantage_home_prob": ha_home,
            "home_advantage_draw_prob": ha_draw,
            "home_advantage_away_prob": ha_away,
            "team_strength_home_prob": ts_home,
            "team_strength_draw_prob": ts_draw,
            "team_strength_away_prob": ts_away,
            "recent_form_home_prob": rf_home,
            "recent_form_draw_prob": rf_draw,
            "recent_form_away_prob": rf_away,
            "population": "all_matches",
            "cutoff_week": cutoff_week,
        })

    return pd.DataFrame(rows)