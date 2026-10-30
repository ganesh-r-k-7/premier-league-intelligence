"""Team strength modelling for Phase 9.2A.

Provides multiple interpretable approaches for estimating team strength:
- Elo rating system
- Attack and defence strength (Poisson-based)
- Recent form features
- Composite team strength index
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.data.team_mapping import load_team_registry, get_canonical_name
from src.data.config import DATA_PROCESSED_DIR


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class EloConfig:
    """Configuration for Elo rating system."""
    initial_rating: float = 1500.0          # Starting rating for all teams
    k_factor: float = 20.0                  # K-factor for rating updates
    home_advantage: float = 100.0           # Home advantage in Elo points
    draw_weight: float = 0.5                # Weight for draws (0.5 = half win)
    min_matches_for_rating: int = 5         # Minimum matches before rating is meaningful
    carry_over_seasons: bool = True         # Whether to carry ratings across seasons
    season_carry_over_factor: float = 0.5   # Fraction of rating retained across seasons


@dataclass(frozen=True)
class TeamStrengthConfig:
    """Configuration for team strength models."""
    # Elo
    elo_config: EloConfig = field(default_factory=EloConfig)
    
    # Attack/defence strength
    min_matches_for_strength: int = 5       # Minimum matches for attack/defence estimates
    regularization: float = 1.0             # L2 regularization for Poisson regression
    
    # Recent form
    form_windows: tuple[int, ...] = (3, 5, 10)  # Match windows for recent form
    
    # Team strength index weights
    weight_elo: float = 0.3
    weight_attack: float = 0.2
    weight_defence: float = 0.2
    weight_form: float = 0.15
    weight_home_advantage: float = 0.15


# =============================================================================
# Elo Rating System
# =============================================================================

class EloRatingSystem:
    """Elo rating system for team strength estimation.
    
    Implements standard Elo with home advantage, configurable K-factor,
    and season carry-over.
    """
    
    def __init__(self, config: Optional[EloConfig] = None):
        self.config = config or EloConfig()
        self.ratings: dict[int, float] = {}
        self.match_counts: dict[int, int] = {}
        self.history: list[dict] = []
    
    def get_rating(self, team_id: int) -> float:
        """Get current Elo rating for a team."""
        return self.ratings.get(team_id, self.config.initial_rating)
    
    def get_match_count(self, team_id: int) -> int:
        """Get number of matches a team has played."""
        return self.match_counts.get(team_id, 0)
    
    def expected_score(self, rating_a: float, rating_b: float, home_advantage: float = 0.0) -> float:
        """Calculate expected score for team A against team B.
        
        Args:
            rating_a: Rating of team A
            rating_b: Rating of team B
            home_advantage: Home advantage in Elo points (added to rating_a)
            
        Returns:
            Expected score for team A (0 to 1)
        """
        diff = (rating_a + home_advantage) - rating_b
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))
    
    def update_ratings(
        self,
        team_a: int,
        team_b: int,
        score_a: float,
        score_b: float,
        home_advantage: bool = True
    ) -> tuple[float, float]:
        """Update ratings after a match.
        
        Args:
            team_a: First team (home team)
            team_b: Second team (away team)
            score_a: Score of team A
            score_b: Score of team B
            home_advantage: Whether team A has home advantage
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Expected scores
        home_adv = self.config.home_advantage if home_advantage else 0.0
        expected_a = self.expected_score(rating_a, rating_b, home_adv)
        expected_b = 1.0 - expected_a
        
        # Actual scores
        if score_a > score_b:
            actual_a = 1.0
            actual_b = 0.0
        elif score_a == score_b:
            actual_a = self.config.draw_weight
            actual_b = self.config.draw_weight
        else:
            actual_a = 0.0
            actual_b = 1.0
        
        # Update ratings
        new_rating_a = rating_a + self.config.k_factor * (actual_a - expected_a)
        new_rating_b = rating_b + self.config.k_factor * (actual_b - expected_b)
        
        # Update internal state
        self.ratings[team_a] = new_rating_a
        self.ratings[team_b] = new_rating_b
        self.match_counts[team_a] = self.match_counts.get(team_a, 0) + 1
        self.match_counts[team_b] = self.match_counts.get(team_b, 0) + 1
        
        # Record history
        self.history.append({
            "team_a": team_a,
            "team_b": team_b,
            "rating_a_before": rating_a,
            "rating_b_before": rating_b,
            "rating_a_after": new_rating_a,
            "rating_b_after": new_rating_b,
            "score_a": score_a,
            "score_b": score_b,
            "home_advantage": home_advantage,
        })
        
        return new_rating_a, new_rating_b
    
    def process_season(
        self,
        fixtures: pd.DataFrame,
        season_id: int,
        carry_over: bool = True
    ) -> dict[int, float]:
        """Process all matches in a season chronologically.
        
        Args:
            fixtures: DataFrame with match data (must have home_team_id, away_team_id, 
                     home_score, away_score, match_date sorted chronologically)
            season_id: Season identifier
            carry_over: Whether to carry over ratings from previous seasons
            
        Returns:
            Dictionary of final ratings per team
        """
        # Sort by date to ensure chronological processing
        fixtures = fixtures.sort_values("match_date").copy()
        
        for _, match in fixtures.iterrows():
            home_id = int(match["home_team_id"])
            away_id = int(match["away_team_id"])
            home_score = match["home_score"] if pd.notna(match["home_score"]) else None
            away_score = match["away_score"] if pd.notna(match["away_score"]) else None
            
            # Skip matches with missing scores
            if home_score is None or away_score is None:
                continue
            
            self.update_ratings(
                team_a=int(match["home_team_id"]),
                team_b=int(match["away_team_id"]),
                score_a=int(home_score),
                score_b=int(away_score),
                home_advantage=True
            )
        
        return self.get_current_ratings()
    
    def get_current_ratings(self) -> dict[int, float]:
        """Get current ratings for all teams."""
        return self.ratings.copy()
    
    def carry_over_to_next_season(self, factor: Optional[float] = None) -> None:
        """Apply season carry-over to ratings.
        
        Ratings are regressed toward the mean (initial_rating).
        """
        factor = factor or self.config.season_carry_over_factor
        for team_id in self.ratings:
            self.ratings[team_id] = (
                self.config.initial_rating + 
                factor * (self.ratings[team_id] - self.config.initial_rating)
            )
    
    def get_rating_history(self, team_id: int) -> list[dict]:
        """Get rating history for a specific team."""
        return [h for h in self.history if h["team_a"] == team_id or h["team_b"] == team_id]
    
    def predict_match(
        self,
        home_team_id: int,
        away_team_id: int
    ) -> tuple[float, float, float]:
        """Predict match outcome probabilities.
        
        Returns:
            Tuple of (home_win_prob, draw_prob, away_win_prob)
        """
        home_rating = self.get_rating(home_team_id)
        away_rating = self.get_rating(away_team_id)
        
        home_win = self.expected_score(home_rating, away_rating, self.config.home_advantage)
        away_win = 1.0 - home_win
        
        # Estimate draw probability based on rating difference
        rating_diff = abs(home_rating - away_rating)
        draw_prob = max(0.05, min(0.35, 0.3 - 0.0002 * rating_diff))
        
        # Normalize
        total = home_win + (1.0 - home_win - away_win) * (1 - 2 * draw_prob)
        # This is approximate; a proper implementation would use a model
        # For now, distribute remaining probability
        draw_prob = draw_prob
        home_win = home_win * (1 - 2 * draw_prob)
        away_win = away_win * (1 - 2 * draw_prob)
        
        return home_win, draw_prob, away_win


# =============================================================================
# Attack and Defence Strength (Poisson-based)
# =============================================================================

def estimate_attack_defence_strength(
    fixtures: pd.DataFrame,
    min_matches: int = 5,
    regularization: float = 1.0
) -> tuple[dict[int, float], dict[int, float], float, float]:
    """Estimate attack and defence strength for all teams using Poisson regression.
    
    Uses regularized maximum likelihood to estimate:
    - Attack strength for each team
    - Defence strength for each team
    - Home advantage parameter
    - League average goals per match
    
    Args:
        fixtures: DataFrame with match results (home_team_id, away_team_id, home_score, away_score)
        min_matches: Minimum matches for a team to be included
        regularization: L2 regularization strength
        
    Returns:
        Tuple of (attack_strength, defence_strength, home_advantage, league_avg)
    """
    # Filter valid matches
    valid = fixtures.dropna(subset=["home_score", "away_score"]).copy()
    if valid.empty:
        return {}, {}, 1.0, 1.5
    
    # Get unique teams
    teams = pd.unique(pd.concat([valid["home_team_id"], valid["away_team_id"]]))
    n_teams = len(teams)
    team_to_idx = {team: i for i, team in enumerate(teams)}
    
    # Prepare data for optimization
    home_teams = np.array([team_to_idx[t] for t in valid["home_team_id"]])
    away_teams = np.array([team_to_idx[t] for t in valid["away_team_id"]])
    home_goals = valid["home_score"].values
    away_goals = valid["away_score"].values
    n_matches = len(valid)
    
    # Initial parameters
    # Attack strengths (log scale), defence strengths (log scale), home advantage, league avg
    n_params = 2 * len(teams) + 2
    initial_params = np.zeros(n_params)
    initial_params[:len(teams)] = 0.0  # attack (log scale)
    initial_params[len(teams):2*len(teams)] = 0.0  # defence (log scale)
    initial_params[-2] = 0.5  # home advantage (log scale)
    initial_params[-1] = np.log(1.5)  # league average goals per match
    
    def neg_log_likelihood(params):
        attack = params[:len(teams)]
        defence = params[len(teams):2*len(teams)]
        home_adv = params[-2]
        league_avg = np.exp(params[-1])
        
        # Expected goals
        lambda_home = np.exp(attack[home_teams] - defence[away_teams] + home_adv + params[-1])
        lambda_away = np.exp(attack[away_teams] - defence[home_teams] + params[-1])
        
        # Poisson log-likelihood
        ll = np.sum(home_goals * np.log(lambda_home + 1e-10) - lambda_home)
        ll += np.sum(away_goals * np.log(lambda_away + 1e-10) - lambda_away)
        
        # L2 regularization
        reg = regularization * np.sum(params[:-2] ** 2)
        
        return -ll + reg
    
    # Optimize
    result = minimize(
        neg_log_likelihood,
        initial_params,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-8}
    )
    
    if not result.success:
        warnings.warn(f"Attack/defence optimization did not converge: {result.message}")
    
    params = result.x
    attack = params[:len(teams)]
    defence = params[len(teams):2*len(teams)]
    home_adv = params[-2]
    league_avg = np.exp(params[-1])
    
    # Convert to dictionaries
    attack_strength = {team: np.exp(attack[i]) for i, team in enumerate(teams)}
    defence_strength = {team: np.exp(defence[i]) for i, team in enumerate(teams)}
    home_adv_val = np.exp(home_adv)
    
    return attack_strength, defence_strength, home_adv_val, league_avg


def predict_match_poisson(
    home_team: int,
    away_team: int,
    attack_strength: dict[int, float],
    defence_strength: dict[int, float],
    home_advantage: float,
    league_avg: float,
    max_goals: int = 10
) -> np.ndarray:
    """Predict match outcome probabilities using Poisson model.
    
    Returns:
        Probability matrix of shape (max_goals+1, max_goals+1) for home vs away goals.
    """
    lambda_home = attack_strength.get(home_team, 1.0) * defence_strength.get(away_team, 1.0) * home_advantage * league_avg
    lambda_away = attack_strength.get(away_team, 1.0) * defence_strength.get(home_team, 1.0) * league_avg
    
    # Compute score matrix
    max_g = max_goals
    prob_matrix = np.zeros((max_g + 1, max_g + 1))
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            prob_matrix[h, a] = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
    
    return prob_matrix


# =============================================================================
# Recent Form Features
# =============================================================================

def compute_recent_form(
    fixtures: pd.DataFrame,
    team_id: int,
    match_date: pd.Timestamp,
    window: int = 5,
    n_matches: Optional[int] = None
) -> dict[str, float]:
    """Compute recent form statistics for a team.
    
    Args:
        fixtures: Historical fixtures DataFrame
        team_id: Team identifier
        match_date: Date of the match to compute form before
        window: Number of recent matches to consider
        n_matches: Override window with exact number of matches
        
    Returns:
        Dictionary of form statistics
    """
    # Filter historical matches before match_date
    history = fixtures[fixtures["match_date"] < match_date].copy()
    
    # Get team's matches
    team_matches = fixtures[
        (fixtures["home_team_id"] == team_id) | (fixtures["away_team_id"] == team_id)
    ][fixtures["match_date"] < match_date].sort_values("match_date", ascending=False)
    
    if n_matches is not None:
        recent = team_matches.head(n_matches)
    else:
        recent = team_matches.head(window)
    
    if len(recent) == 0:
        return {
            "form_points": 0.0,
            "form_goals_scored": 0.0,
            "form_goals_conceded": 0.0,
            "form_goals_per_match": 0.0,
            "form_ga_per_match": 0.0,
            "form_points_per_match": 0.0,
            "form_wins": 0,
            "form_draws": 0,
            "form_losses": 0,
            "form_matches": 0,
        }
    
    gf = 0
    ga = 0
    pts = 0
    w = d = l = 0
    
    for _, row in recent.iterrows():
        is_home = row["home_team_id"] == team_id
        if is_home:
            gf += row.get("home_score", 0) or 0
            ga += row.get("away_score", 0) or 0
            if row["home_score"] > row["away_score"]:
                pts += 3
                w += 1
            elif row["home_score"] == row["away_score"]:
                pts += 1
                d += 1
            else:
                l += 1
        else:
            gf += row.get("away_score", 0) or 0
            ga += row.get("home_score", 0) or 0
            if row["away_score"] > row["home_score"]:
                pts += 3
                w += 1
            elif row["away_score"] == row["home_score"]:
                pts += 1
                d += 1
            else:
                l += 1
    
    n = len(recent)
    return {
        "form_points": float(pts),
        "form_goals_scored": float(gf),
        "form_goals_conceded": float(ga),
        "form_goals_per_match": gf / n if n > 0 else 0.0,
        "form_ga_per_match": ga / n if n > 0 else 0.0,
        "form_points_per_match": pts / n if n > 0 else 0.0,
        "form_wins": w,
        "form_draws": d,
        "form_losses": l,
        "form_matches": n,
    }


def compute_form_features(
    fixtures: pd.DataFrame,
    target_matches: pd.DataFrame,
    windows: tuple[int, ...] = (3, 5, 10)
) -> pd.DataFrame:
    """Compute form features for target matches.
    
    Args:
        fixtures: Full fixtures DataFrame (historical)
        target_matches: Matches to compute form for
        windows: List of window sizes
        
    Returns:
        DataFrame with form features for each match
    """
    fixtures = fixtures.sort_values("match_date").copy()
    target = target_matches.sort_values("match_date").copy()
    
    all_features = []
    
    for _, match in target.iterrows():
        match_date = pd.Timestamp(match["match_date"])
        home_id = int(match["home_team_id"])
        away_id = int(match["away_team_id"])
        
        features = {"match_id": match["match_id"]}
        
        for window in windows:
            # Home team form
            home_form = compute_recent_form(fixtures, match["home_team_id"], match["match_date"], window=window)
            for k, v in home_form.items():
                features[f"home_{k}_{window}"] = v
            
            # Away team form
            away_form = compute_recent_form(fixtures, match["away_team_id"], match["match_date"], window=window)
            for k, v in away_form.items():
                features[f"away_{k}_{window}"] = v
        
        all_features.append(features)
    
    return pd.DataFrame(all_features)


# =============================================================================
# Team Strength Index
# =============================================================================

class TeamStrengthIndex:
    """Composite team strength index combining multiple signals."""
    
    def __init__(self, config: Optional[TeamStrengthConfig] = None):
        self.config = config or TeamStrengthConfig()
        self.elo_system = EloRatingSystem(self.config.elo_config)
        self.attack_strength: dict[int, float] = {}
        self.defence_strength: dict[int, float] = {}
        self.home_advantage: float = 1.0
        self.league_avg: float = 1.5
    
    def fit(
        self,
        fixtures: pd.DataFrame,
        season_id: int | None = None,
        include_current_season: bool = False
    ) -> "TeamStrengthIndex":
        """Fit all components on historical data.
        
        Args:
            fixtures: Full fixtures DataFrame
            season_id: If provided, only use data from this season
            include_current_season: Whether to include current season data
        """
        # Filter data
        if season_id is not None:
            data = fixtures[fixtures["season_id"] == season_id].copy()
        else:
            data = fixtures.copy()
        
        if not include_current_season:
            # Exclude current season (most recent season_id)
            max_season = fixtures["season_id"].max()
            data = data[data["season_id"] < max_season]
        
        # Fit Elo
        valid = data.dropna(subset=["home_score", "away_score"])
        self.elo_system.process_season(valid, season_id or 0)
        
        # Estimate attack/defence
        self.attack_strength, self.defence_strength, self.home_advantage, self.league_avg = (
            estimate_attack_defence_strength(data)
        )
        
        return self
    
    def get_team_strength(self, team_id: int) -> dict[str, float]:
        """Get all strength components for a team."""
        elo = self.elo_system.get_rating(team_id)
        elo_norm = (elo - 1500) / 200  # Normalize roughly to [-2, 2]
        
        attack = self.attack_strength.get(team_id, 1.0)
        defence = self.defence_strength.get(team_id, 1.0)
        attack_norm = np.log(attack) if attack > 0 else 0
        defence_norm = -np.log(defence) if defence > 0 else 0  # Lower is better
        
        return {
            "elo_rating": elo,
            "elo_normalized": elo_norm,
            "attack_strength": attack,
            "attack_normalized": attack_norm,
            "defence_strength": defence,
            "defence_normalized": defence_norm,
            "home_advantage": self.home_advantage,
        }
    
    def compute_team_strength_index(self, team_id: int) -> float:
        """Compute composite team strength index.
        
        Combines normalized components with configurable weights.
        """
        components = self.get_team_strength(team_id)
        
        idx = (
            self.config.weight_elo * components["elo_normalized"] +
            self.config.weight_attack * components["attack_normalized"] +
            self.config.weight_defence * components["defence_normalized"] +
            self.config.weight_home_advantage * (self.home_advantage - 1.0)
        )
        
        # Form component would be added dynamically
        return float(idx)
    
    def get_league_rankings(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Get team strength rankings for all teams in a season."""
        teams = pd.unique(pd.concat([fixtures["home_team_id"], fixtures["away_team_id"]]))
        
        rankings = []
        for team_id in teams:
            components = self.get_team_strength(team_id)
            idx = self.compute_team_strength_index(team_id)
            rankings.append({
                "team_id": team_id,
                "elo_rating": components["elo_rating"],
                "attack_strength": components["attack_strength"],
                "defence_strength": components["defence_strength"],
                "strength_index": idx,
            })
        
        df = pd.DataFrame(rankings)
        df = df.sort_values("strength_index", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df


# =============================================================================
# Historical Team Analysis
# =============================================================================

def get_team_strength_rankings(
    fixtures: pd.DataFrame,
    season_id: int,
    config: Optional[TeamStrengthConfig] = None
) -> pd.DataFrame:
    """Get team strength rankings for a specific season."""
    index = TeamStrengthIndex(config)
    index.fit(fixtures, season_id=season_id)
    return index.get_league_rankings(fixtures[fixtures["season_id"] == season_id])


def get_team_strength_progression(
    fixtures: pd.DataFrame,
    team_id: int,
    config: Optional[TeamStrengthConfig] = None
) -> pd.DataFrame:
    """Get team strength progression across seasons."""
    seasons = sorted(fixtures["season_id"].unique())
    results = []
    
    for season_id in seasons:
        season_data = fixtures[fixtures["season_id"] == season_id]
        if len(season_data) < 10:  # Not enough data
            continue
        
        index = TeamStrengthIndex(config)
        index.fit(season_data, season_id=season_id)
        
        team_strength = index.get_team_strength(team_id)
        idx = index.compute_team_strength_index(team_id)
        
        results.append({
            "season_id": season_id,
            "elo_rating": team_strength["elo_rating"],
            "attack_strength": team_strength["attack_strength"],
            "defence_strength": team_strength["defence_strength"],
            "strength_index": idx,
        })
    
    return pd.DataFrame(results)


def compare_teams(
    fixtures: pd.DataFrame,
    team_a: int,
    team_b: int,
    season_id: int,
    config: Optional[TeamStrengthConfig] = None
) -> dict[str, float]:
    """Compare two teams head-to-head."""
    index = TeamStrengthIndex(config)
    index.fit(fixtures, season_id=season_id)
    
    a_strength = index.get_team_strength(team_a)
    b_strength = index.get_team_strength(team_b)
    
    a_idx = index.compute_team_strength_index(team_a)
    b_idx = index.compute_team_strength_index(team_b)
    
    return {
        "team_a": team_a,
        "team_b": team_b,
        "elo_diff": a_strength["elo_rating"] - b_strength["elo_rating"],
        "attack_diff": a_strength["attack_strength"] - b_strength["attack_strength"],
        "defence_diff": a_strength["defence_strength"] - b_strength["defence_strength"],
        "strength_index_diff": a_idx - b_idx,
    }


# =============================================================================
# Match Prediction with Team Strength
# =============================================================================

def predict_match_outcome(
    home_team: int,
    away_team: int,
    index: "TeamStrengthIndex",
    max_goals: int = 10
) -> tuple[float, float, float]:
    """Predict match outcome using team strength index.
    
    Returns (home_win_prob, draw_prob, away_win_prob)
    """
    # Get strength difference
    home_strength = index.compute_team_strength_index(home_team)
    away_strength = index.compute_team_strength_index(away_team)
    strength_diff = home_strength - away_strength
    
    # Convert strength difference to probabilities using logistic function
    # Calibrated so that 1 strength unit = ~0.1 probability difference
    home_advantage = 0.15  # baseline home advantage
    strength_advantage = strength_diff * 0.1
    
    home_win_base = 0.45  # league average home win rate
    draw_base = 0.25
    
    # Adjust based on strength difference
    home_win = home_win_base + strength_advantage * 0.5
    away_win = 0.35 - strength_advantage * 0.5
    
    # Draw probability decreases with larger strength difference
    draw_prob = max(0.15, 0.25 - abs(home_strength - away_strength) * 0.1)
    
    # Normalize
    total = home_win + draw_prob + away_win
    home_win = home_win / total
    draw_prob = 0.25 / total  # Simplified
    away_win = away_win / total
    
    # Ensure valid probabilities
    home_win = max(0.05, min(0.95, home_win))
    draw_prob = max(0.05, min(0.95, draw_prob))
    away_win = 1.0 - home_win - draw_prob
    
    return home_win, draw_prob, away_win


# =============================================================================
# Form Features for Match Prediction
# =============================================================================

def compute_team_form_features(
    fixtures: pd.DataFrame,
    target_matches: pd.DataFrame,
    windows: tuple[int, ...] = (3, 5, 10)
) -> pd.DataFrame:
    """Compute form features for target matches."""
    fixtures = fixtures.sort_values("match_date").copy()
    target = target_matches.sort_values("match_date").copy()
    
    all_features = []
    
    for _, match in target.iterrows():
        match_date = pd.Timestamp(match["match_date"])
        home_id = int(match["home_team_id"])
        away_id = int(match["away_team_id"])
        
        features = {"match_id": match["match_id"]}
        
        for window in windows:
            # Home team form
            home_form = compute_recent_form(fixtures, match["home_team_id"], match["match_date"], window=window)
            for k, v in home_form.items():
                features[f"home_{k}_{window}"] = v
            
            # Away team form
            away_form = compute_recent_form(fixtures, match["away_team_id"], match["match_date"], window=window)
            for k, v in away_form.items():
                features[f"away_{k}_{window}"] = v
        
        all_features.append(features)
    
    return pd.DataFrame(all_features)


# =============================================================================
# Team Registry Helpers
# =============================================================================

def get_team_canonical_name(team_id: int) -> str:
    """Get canonical team name from team registry."""
    try:
        registry = load_team_registry()
        match = registry[registry["team_id"] == team_id]
        if not match.empty:
            return match.iloc[0]["canonical_name"]
    except Exception:
        pass
    return f"Team_{team_id}"


def get_team_id_from_name(name: str) -> Optional[int]:
    """Get team ID from canonical name."""
    try:
        registry = load_team_registry()
        match = registry[registry["canonical_name"] == name]
        if not match.empty:
            return int(match.iloc[0]["team_id"])
    except Exception:
        pass
    return None


# =============================================================================
# Configuration Classes
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class EloConfig:
    """Configuration for Elo rating system."""
    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 100.0
    draw_weight: float = 0.5
    min_matches_for_rating: int = 5
    carry_over_seasons: bool = True
    season_carry_over_factor: float = 0.5


@dataclass(frozen=True)
class TeamStrengthConfig:
    """Configuration for team strength models."""
    # Elo
    elo_config: EloConfig = field(default_factory=EloConfig)
    
    # Attack/defence strength
    min_matches_for_strength: int = 5
    regularization: float = 1.0
    
    # Recent form
    form_windows: tuple[int, ...] = (3, 5, 10)
    
    # Team strength index weights
    weight_elo: float = 0.3
    weight_attack: float = 0.2
    weight_defence: float = 0.2
    weight_form: float = 0.15
    weight_home_advantage: float = 0.15


if __name__ == "__main__":
    # Quick test
    print("Team Strength module loaded successfully")
    print("Available classes/functions:")
    print("  - EloRatingSystem")
    print("  - TeamStrengthIndex")
    print("  - estimate_attack_defence_strength")
    print("  - compute_recent_form")
    print("  - TeamStrengthIndex")
    print("  - get_team_strength_rankings")
    print("  - compare_teams")
    print("  - compute_team_form_features")