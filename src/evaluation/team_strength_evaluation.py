"""Team strength evaluation module for Phase 9.2A.

Provides evaluation functions for team strength baselines and models.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.features.team_strength import (
    EloRatingSystem,
    EloConfig,
)
from src.features.match_baselines import MatchBaselineConfig

logger = logging.getLogger("plintel.evaluation.team_strength")


def compute_global_outcome_frequencies(
    history: pd.DataFrame,
    smoothing_alpha: float = 1.0
) -> tuple[float, float, float]:
    """Return (home_win_prob, draw_prob, away_win_prob) from match history."""
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
    """Compute outcome probabilities based on team historical performance."""
    # Home team's home matches
    home_team_home = history[history["home_team_id"] == home_team_id]
    # Away team's away matches
    away_team_away = history[history["away_team_id"] == away_team_id]

    if len(home_team_home) >= 5 and len(away_team_away) >= 5:
        h_home_wins = (home_team_home["home_score"] > home_team_home["away_score"]).sum()
        h_draws = (home_team_home["home_score"] == home_team_home["away_score"]).sum()
        h_away_wins = (home_team_home["home_score"] < home_team_home["away_score"]).sum()

        a_home_wins = (away_team_away["home_score"] < away_team_away["away_score"]).sum()
        a_draws = (away_team_away["home_score"] == away_team_away["away_score"]).sum()
        a_away_wins = (away_team_away["home_score"] > away_team_away["away_score"]).sum()

        combined_home_wins = (h_home_wins + a_home_wins) / 2
        combined_draws = (h_draws + a_draws) / 2
        combined_away_wins = (h_away_wins + a_away_wins) / 2

        total = combined_home_wins + combined_draws + combined_away_wins + 3 * 1.0
        home_win_prob = (combined_home_wins + 1.0) / total
        draw_prob = (combined_draws + 1.0) / total
        away_win_prob = (combined_away_wins + 1.0) / total

        return (float(home_win_prob), float(draw_prob), float(away_win_prob))

    return (1.0/3, 1.0/3, 1.0/3)


def compute_recent_form_baseline(
    history: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    window: int,
    global_probs: tuple[float, float, float],
    smoothing_alpha: float = 1.0
) -> tuple[float, float, float]:
    """Compute outcome probabilities based on recent form of both teams."""
    valid_history = history.dropna(subset=["home_score", "away_score"])
    
    home_matches = valid_history[
        (valid_history["home_team_id"] == home_team_id) | (valid_history["away_team_id"] == home_team_id)
    ].sort_values("match_date", ascending=False).head(5)

    away_matches = valid_history[
        (valid_history["home_team_id"] == away_team_id) | (valid_history["away_team_id"] == away_team_id)
    ].sort_values("match_date", ascending=False).head(5)

    if len(home_matches) < 1 or len(away_matches) < 1:
        return (1.0/3, 1.0/3, 1.0/3)

    def get_team_results(matches: pd.DataFrame, team_id: int) -> tuple[int, int, int]:
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
            else:
                if row["away_score"] > row["home_score"]:
                    wins += 1
                elif row["away_score"] == row["home_score"]:
                    draws += 1
                else:
                    losses += 1
        return wins, draws, losses

    h_wins, h_draws, h_losses = get_team_results(home_matches, home_team_id)
    a_wins, a_draws, a_losses = get_team_results(away_matches, away_team_id)

    combined_home_wins = h_wins + a_losses
    combined_draws = h_draws + a_draws
    combined_away_wins = h_losses + a_wins

    total = combined_home_wins + combined_draws + combined_away_wins + 3 * 1.0
    home_win_prob = (combined_home_wins + 1.0) / total
    draw_prob = (combined_draws + 1.0) / total
    away_win_prob = (combined_away_wins + 1.0) / total

    return (float(home_win_prob), float(draw_prob), float(away_win_prob))


class EloRatingSystem:
    """Elo rating system for team strength estimation."""
    
    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 20.0, home_advantage: float = 100.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict[int, float] = {}
        self.match_counts: dict[int, int] = {}
    
    def get_rating(self, team_id: int) -> float:
        return self.ratings.get(team_id, 1500.0)
    
    def expected_score(self, rating_a: float, rating_b: float, home_adv: float = 0.0) -> float:
        diff = (rating_a + home_adv) - rating_b
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))
    
    def update_ratings(self, team_a: int, team_b: int, score_a: int, score_b: int, home_advantage: bool = True) -> None:
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        home_adv = 100.0 if home_advantage else 0.0
        expected_a = self.expected_score(rating_a, rating_b, home_adv)
        expected_b = 1.0 - expected_a
        
        if score_a > score_b:
            actual_a, actual_b = 1.0, 0.0
        elif score_a == score_b:
            actual_a = actual_b = 0.5
        else:
            actual_a, actual_b = 0.0, 1.0
        
        new_rating_a = rating_a + 20.0 * (actual_a - expected_a)
        new_rating_b = rating_b + 20.0 * (actual_b - expected_b)
        
        self.ratings[team_a] = new_rating_a
        self.ratings[team_b] = new_rating_b
    
    def build_from_history(self, history: pd.DataFrame) -> None:
        """Build Elo ratings from match history."""
        valid_history = history.dropna(subset=["home_score", "away_score"]).sort_values("match_date")
        
        for _, match in valid_history.iterrows():
            self.update_ratings(
                team_a=int(match["home_team_id"]),
                team_b=int(match["away_team_id"]),
                score_a=int(match["home_score"]),
                score_b=int(match["away_score"]),
                home_advantage=True
            )
    
    def predict_match(self, home_team_id: int, away_team_id: int) -> tuple[float, float, float]:
        """Predict match outcome probabilities using Elo ratings."""
        home_rating = self.get_rating(home_team_id)
        away_rating = self.get_rating(away_team_id)
        
        home_win = 1.0 / (1.0 + 10 ** (-(home_rating - away_rating + 100) / 400))
        away_win = 1.0 - home_win
        
        # Estimate draw probability
        draw_prob = max(0.05, min(0.35, 0.3 - 0.0002 * abs(home_rating - away_rating)))
        
        # Normalize
        home_win = home_win * (1 - 2 * draw_prob)
        away_win = away_win * (1 - 2 * draw_prob)
        
        total = home_win + 0.25 + away_win
        home_win = home_win / (home_win + 0.25 + away_win)
        away_win = away_win / (home_win + 0.25 + away_win)
        
        return home_win, 0.25, away_win


if __name__ == "__main__":
    print("Team strength evaluation module loaded successfully")