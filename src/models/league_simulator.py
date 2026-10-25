"""Monte Carlo league simulator for Premier League standings.

Simulates remaining fixtures using match outcome probabilities to estimate
final standings, title probabilities, top-four probabilities, and
relegation probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.stats import poisson


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration for league simulator."""
    n_simulations: int = 10_000           # Number of Monte Carlo simulations
    random_seed: int | None = 42          # Random seed for reproducibility
    points_win: int = 3                   # Points for a win
    points_draw: int = 1                  # Points for a draw
    points_loss: int = 0                  # Points for a loss
    n_relegation_places: int = 3          # Number of relegation places
    n_top_places: int = 4                 # Number of top places (e.g., Champions League)
    tiebreaker: Literal["points_only", "goal_diff_then_goals"] = "points_only"
                                        # Tiebreaker method for standings
    simulation_mode: Literal["outcome", "scoreline"] = "outcome"
                                        # "outcome" = W/D/L only, "scoreline" = simulate goals


# Type for probability provider function
# Takes (fixtures_row, history) and returns (p_home, p_draw, p_away)
ProbabilityProvider = Callable[
    [pd.Series, pd.DataFrame],
    tuple[float, float, float]
]


def default_probability_provider(
    match_row: pd.Series,
    history: pd.DataFrame,
    config: 'MatchBaselineConfig' | None = None
) -> tuple[float, float, float]:
    """Default probability provider using global outcome frequencies.
    
    This is a simple provider that uses Laplace-smoothed global frequencies.
    In practice, you would inject a more sophisticated provider.
    
    Args:
        match_row: Row from fixtures DataFrame for the match to predict.
        history: Historical matches (match_week < current match_week).
        config: Optional configuration for baseline.
    
    Returns:
        Tuple of (p_home, p_draw, p_away).
    """
    from src.features.match_baselines import compute_global_outcome_frequencies, MatchBaselineConfig
    
    if config is None:
        config = MatchBaselineConfig()
    
    return compute_global_outcome_frequencies(history, config.smoothing_alpha)


def validate_probabilities(
    p_home: float,
    p_draw: float,
    p_away: float,
    match_id: int | str = "unknown"
) -> None:
    """Validate that probabilities are valid.
    
    Args:
        p_home: Home win probability.
        p_draw: Draw probability.
        p_away: Away win probability.
        match_id: Match identifier for error messages.
    
    Raises:
        ValueError: If probabilities are invalid.
    """
    probs = np.array([p_home, p_draw, p_away])
    
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"Match {match_id}: Non-finite probabilities")
    
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError(f"Match {match_id}: Probabilities outside [0, 1]")
    
    total = p_home + p_draw + p_away
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(
            f"Match {match_id}: Probabilities sum to {total:.10f}, not 1.0"
        )


def validate_fixtures(
    fixtures: pd.DataFrame,
    required_columns: list[str] | None = None
) -> None:
    """Validate fixtures DataFrame.
    
    Args:
        fixtures: Fixtures DataFrame to validate.
        required_columns: List of required column names.
    
    Raises:
        ValueError: If validation fails.
    """
    if required_columns is None:
        required_columns = [
            "match_id", "match_week", "home_team_id", "away_team_id",
            "home_score", "away_score"
        ]
    
    missing = set(required_columns) - set(fixtures.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for duplicate match_ids
    if fixtures["match_id"].duplicated().any():
        dupes = fixtures[fixtures["match_id"].duplicated()]["match_id"].unique()
        raise ValueError(f"Duplicate match_ids found: {dupes}")
    
    # Check for missing teams
    if fixtures["home_team_id"].isna().any() or fixtures["away_team_id"].isna().any():
        raise ValueError("Missing team IDs in fixtures")
    
    # Check for matches where home and away team are the same
    same_team = fixtures[fixtures["home_team_id"] == fixtures["away_team_id"]]
    if len(same_team) > 0:
        raise ValueError("Fixtures contain matches where home_team_id == away_team_id")


def compute_standings(
    results: pd.DataFrame,
    teams: list[int] | None = None,
    config: SimulatorConfig | None = None,
    existing_points: dict[int, int] | None = None,
    existing_gd: dict[int, int] | None = None,
    existing_gs: dict[int, int] | None = None
) -> pd.DataFrame:
    """Compute league standings from match results.
    
    Args:
        results: DataFrame with columns [home_team_id, away_team_id, home_score, away_score].
        teams: List of team IDs. If None, inferred from results.
        config: SimulatorConfig for tiebreaker rules.
        existing_points: Dict of team_id -> points from already-played matches.
        existing_gd: Dict of team_id -> goal difference from already-played matches.
        existing_gs: Dict of team_id -> goals scored from already-played matches.
    
    Returns:
        DataFrame with standings (team_id, points, position, etc.).
    """
    if config is None:
        config = SimulatorConfig()
    
    if teams is None:
        home_teams = set(results["home_team_id"].unique())
        away_teams = set(results["away_team_id"].unique())
        teams = sorted(home_teams | away_teams)
    
    # Initialize standings
    standings = []
    for team_id in teams:
        pts = existing_points.get(team_id, 0) if existing_points else 0
        gd = existing_gd.get(team_id, 0) if existing_gd else 0
        gs = existing_gs.get(team_id, 0) if existing_gs else 0
        standings.append({
            "team_id": team_id,
            "points": pts,
            "goal_diff": gd,
            "goals_scored": gs,
            "played": 0,
            "won": 0,
            "drawn": 0,
            "lost": 0,
        })
    
    standings_df = pd.DataFrame(standings).set_index("team_id")
    
    # Process results
    for _, row in results.iterrows():
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        
        # Update played
        standings_df.loc[home_id, "played"] += 1
        standings_df.loc[away_id, "played"] += 1
        
        # Update goals
        standings_df.loc[home_id, "goals_scored"] += home_score
        standings_df.loc[away_id, "goals_scored"] += away_score
        
        if home_score > away_score:
            # Home win
            standings_df.loc[home_id, "points"] += 3
            standings_df.loc[home_id, "won"] += 1
            standings_df.loc[away_id, "lost"] += 1
        elif home_score == away_score:
            # Draw
            standings_df.loc[home_id, "points"] += 1
            standings_df.loc[away_id, "points"] += 1
            standings_df.loc[home_id, "drawn"] += 1
            standings_df.loc[away_id, "drawn"] += 1
        else:
            # Away win
            standings_df.loc[away_id, "points"] += 3
            standings_df.loc[away_id, "won"] += 1
            standings_df.loc[home_id, "lost"] += 1
        
        # Goal difference
        standings_df.loc[home_id, "goal_diff"] += home_score - away_score
        standings_df.loc[away_id, "goal_diff"] += away_score - home_score
    
    # Sort by points, then tiebreaker
    if config and config.tiebreaker == "goal_diff_then_goals":
        standings_df = standings_df.sort_values(
            ["points", "goal_diff", "goals_scored"],
            ascending=[False, False, False]
        )
    else:
        standings_df = standings_df.sort_values("points", ascending=False)
    
    # Assign positions (1-indexed)
    standings_df = standings_df.reset_index()
    standings_df["position"] = range(1, len(standings_df) + 1)
    
    return standings_df


def simulate_match(
    p_home: float,
    p_draw: float,
    p_away: float,
    rng: np.random.Generator
) -> int:
    """Simulate a single match outcome.
    
    Args:
        p_home: Home win probability.
        p_draw: Draw probability.
        p_away: Away win probability.
        rng: NumPy random generator.
    
    Returns:
        2 for home win, 1 for draw, 0 for away win.
    """
    validate_probabilities(p_home, p_draw, p_away)
    r = rng.random()
    if r < p_home:
        return 2  # Home win
    elif r < p_home + p_draw:
        return 1  # Draw
    else:
        return 0  # Away win


def simulate_match_scoreline(
    home_goal_rate: float,
    away_goal_rate: float,
    rng: np.random.Generator
) -> tuple[int, int]:
    """Simulate a match scoreline from independent Poisson distributions.
    
    Args:
        home_goal_rate: Home team's expected goals (lambda for Poisson).
        away_goal_rate: Away team's expected goals (lambda for Poisson).
        rng: NumPy random generator.
    
    Returns:
        Tuple of (home_goals, away_goals).
    """
    home_goals = rng.poisson(home_goal_rate)
    away_goals = rng.poisson(away_goal_rate)
    return int(home_goals), int(away_goals)


def outcome_from_scoreline(home_goals: int, away_goals: int) -> int:
    """Convert scoreline to outcome code.
    
    Args:
        home_goals: Home team goals.
        away_goals: Away team goals.
    
    Returns:
        2 for home win, 1 for draw, 0 for away win.
    """
    if home_goals > away_goals:
        return 2  # Home win
    elif home_goals == away_goals:
        return 1  # Draw
    else:
        return 0  # Away win


def simulate_season(
    fixtures: pd.DataFrame,
    probability_provider: ProbabilityProvider,
    config: SimulatorConfig,
    history_provider: Callable[[int], pd.DataFrame] | None = None,
    existing_results: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None
) -> tuple[pd.DataFrame, dict]:
    """Simulate a full season of matches.
    
    Args:
        fixtures: DataFrame of all fixtures for the season.
        probability_provider: Function(match_row, history) -> (p_home, p_draw, p_away).
        config: SimulatorConfig.
        history_provider: Function(cutoff_week) -> history DataFrame.
            If None, uses fixtures[fixtures["match_week"] < cw].
        existing_results: Already-played matches with known outcomes.
        rng: Random number generator (created from config if None).
    
    Returns:
        Tuple of (final_standings, simulation_info).
    """
    if rng is None:
        rng = np.random.default_rng(config.random_seed)
    
    # Validate fixtures
    validate_fixtures(fixtures)
    
    # Get teams
    home_teams = set(fixtures["home_team_id"].unique())
    away_teams = set(fixtures["away_team_id"].unique())
    teams = sorted(home_teams | away_teams)
    
    # Separate played and unplayed matches
    if existing_results is not None:
        played = existing_results.copy()
        unplayed = fixtures[~fixtures["match_id"].isin(played["match_id"])].copy()
    else:
        # Determine played from non-null scores
        played = fixtures[fixtures["home_score"].notna() & fixtures["away_score"].notna()].copy()
        unplayed = fixtures[fixtures["home_score"].isna() | fixtures["away_score"].isna()].copy()
    
    # If no unplayed matches, just compute standings
    if unplayed.empty:
        if existing_results is not None:
            standings = compute_standings(existing_results, config=config)
        else:
            standings = compute_standings(played, config=config)
        return standings, {"simulations": 1, "note": "No unplayed matches"}
    
    # Prepare existing points, GD, GS from played matches
    existing_points = {}
    existing_gd = {}
    existing_gs = {}
    for _, row in played.iterrows():
        for team_id, score_for, score_against in [
            (row["home_team_id"], row["home_score"], row["away_score"]),
            (row["away_team_id"], row["away_score"], row["home_score"])
        ]:
            pts = existing_points.get(team_id, 0)
            gd = existing_gd.get(team_id, 0)
            gs = existing_gs.get(team_id, 0)
            
            if score_for > score_against:
                pts += 3
            elif score_for == score_against:
                pts += 1
            
            existing_points[team_id] = pts
            existing_gd[team_id] = gd + (score_for - score_against)
            existing_gs[team_id] = gs + score_for
    
    # Sort unplayed by match_week for chronological simulation
    if "match_week" in unplayed.columns:
        unplayed = unplayed.sort_values("match_week")
    
    # Pre-compute history for each matchweek if no custom provider
    if history_provider is None:
        history_cache = {}
        for cw in unplayed["match_week"].unique():
            history_cache[cw] = fixtures[fixtures["match_week"] < cw].copy()
    
    # Run simulations
    simulation_results = []
    
    for sim in range(config.n_simulations):
        # Initialize simulation state
        sim_points = existing_points.copy()
        sim_gd = existing_gd.copy()
        sim_gs = existing_gs.copy()
        
        # Simulate each unplayed match
        for _, match in unplayed.iterrows():
            match_id = match["match_id"]
            
            # Get history for this match
            if history_provider is not None:
                history = history_provider(match.get("match_week", 0))
            else:
                cw = match.get("match_week", 0)
                history = history_cache.get(cw, pd.DataFrame())
            
            # Add already simulated matches to history
            # (In a more sophisticated version, we'd update this dynamically)
            
# Get probabilities
            p_home, p_draw, p_away = probability_provider(match, history)
            validate_probabilities(p_home, p_draw, p_away, match_id)
            
            if config.simulation_mode == "scoreline":
                # Scoreline simulation mode: estimate goal rates from probabilities
                p_total = p_home + p_draw + p_away
                if p_total > 0:
                    # Simple heuristic: outcome probabilities -> expected goals
                    # Home win prob -> higher home goals, away win prob -> higher away goals
                    home_goal_rate = 1.8 * p_home + 0.9 * p_draw + 0.3 * p_away
                    away_goal_rate = 1.6 * p_away + 0.9 * p_draw + 0.2 * p_home
                else:
                    home_goal_rate = 1.4
                    away_goal_rate = 1.2
                
                home_goals, away_goals = simulate_match_scoreline(home_goal_rate, away_goal_rate, rng)
                outcome = outcome_from_scoreline(home_goals, away_goals)
            else:
                # Standard outcome simulation
                outcome = simulate_match(p_home, p_draw, p_away, rng)
            
            # Update simulated standings
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]
            
            if config.simulation_mode == "scoreline":
                # Scoreline simulation mode: we have actual goals
                if 'home_goals' in locals() and 'away_goals' in locals():
                    home_goals_val = home_goals
                    away_goals_val = away_goals
                else:
                    # Fallback if scoreline not available
                    home_goals_val = 2 if outcome == 2 else (1 if outcome == 1 else 0)
                    away_goals_val = 2 if outcome == 0 else (1 if outcome == 1 else 0)
                
                if outcome == 2:  # Home win
                    sim_points[home_id] = sim_points.get(home_id, 0) + 3
                    sim_gd[home_id] = sim_gd.get(home_id, 0) + (home_goals_val - away_goals_val)
                    sim_gd[away_id] = sim_gd.get(away_id, 0) + (away_goals_val - home_goals_val)
                    sim_gs[home_id] = sim_gs.get(home_id, 0) + home_goals_val
                    sim_gs[away_id] = sim_gs.get(away_id, 0) + away_goals_val
                elif outcome == 1:  # Draw
                    sim_points[home_id] = sim_points.get(home_id, 0) + 1
                    sim_points[away_id] = sim_points.get(away_id, 0) + 1
                    sim_gd[home_id] = sim_gd.get(home_id, 0) + (home_goals_val - away_goals_val)
                    sim_gd[away_id] = sim_gd.get(away_id, 0) + (away_goals_val - home_goals_val)
                    sim_gs[home_id] = sim_gs.get(home_id, 0) + home_goals_val
                    sim_gs[away_id] = sim_gs.get(away_id, 0) + away_goals_val
                else:  # Away win
                    sim_points[away_id] = sim_points.get(away_id, 0) + 3
                    sim_gd[away_id] = sim_gd.get(away_id, 0) + (away_goals_val - home_goals_val)
                    sim_gd[home_id] = sim_gd.get(home_id, 0) + (home_goals_val - away_goals_val)
                    sim_gs[away_id] = sim_gs.get(away_id, 0) + away_goals_val
                    sim_gs[home_id] = sim_gs.get(home_id, 0) + home_goals_val
            else:
                # Standard outcome simulation
                if outcome == 2:  # Home win
                    sim_points[home_id] = sim_points.get(home_id, 0) + 3
                    sim_gd[home_id] = sim_gd.get(home_id, 0) + 1
                    sim_gd[away_id] = sim_gd.get(away_id, 0) - 1
                    sim_gs[home_id] = sim_gs.get(home_id, 0) + 2
                    sim_gs[away_id] = sim_gs.get(away_id, 0) + 0
                elif outcome == 1:  # Draw
                    sim_points[home_id] = sim_points.get(home_id, 0) + 1
                    sim_points[away_id] = sim_points.get(away_id, 0) + 1
                    sim_gs[home_id] = sim_gs.get(home_id, 0) + 1
                    sim_gs[away_id] = sim_gs.get(away_id, 0) + 1
                else:  # Away win
                    sim_points[away_id] = sim_points.get(away_id, 0) + 3
                    sim_gd[away_id] = sim_gd.get(away_id, 0) + 1
                    sim_gd[home_id] = sim_gd.get(home_id, 0) - 1
                    sim_gs[away_id] = sim_gs.get(away_id, 0) + 2
                    sim_gs[home_id] = sim_gs.get(home_id, 0) + 0
        
        # Compute final standings for this simulation
        standings = []
        for team_id in teams:
            standings.append({
                "team_id": team_id,
                "points": sim_points.get(team_id, 0),
                "goal_diff": sim_gd.get(team_id, 0),
                "goals_scored": sim_gs.get(team_id, 0),
            })
        
        standings_df = pd.DataFrame(standings)
        
        # Sort by points, then GD, then GS
        standings_df = standings_df.sort_values(
            ["points", "goal_diff", "goals_scored"],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        
        standings_df["position"] = range(1, len(standings_df) + 1)
        standings_df["simulation"] = sim
        
        simulation_results.append(standings_df)
    
    # Combine all simulations
    all_standings = pd.concat(simulation_results, ignore_index=True)
    
    return all_standings, {"n_simulations": config.n_simulations}


def compute_simulation_statistics(
    all_standings: pd.DataFrame,
    n_teams: int,
    config: SimulatorConfig | None = None
) -> dict[str, pd.DataFrame]:
    """Compute summary statistics from simulation results.
    
    Args:
        all_standings: Combined standings from all simulations.
        n_teams: Number of teams in the league.
        config: SimulatorConfig for top/relegation places.
    
    Returns:
        Dictionary with statistics DataFrames.
    """
    if config is None:
        config = SimulatorConfig()
    
    # Position counts: team_id x position -> count
    position_counts = all_standings.groupby(["team_id", "position"]).size().unstack(fill_value=0)
    position_counts = position_counts.reindex(columns=range(1, n_teams + 1), fill_value=0)
    
    # Title probability (position 1)
    title_probs = position_counts[1] / position_counts.sum(axis=1)
    title_probs = title_probs.sort_values(ascending=False)
    
    # Top-N probabilities
    if config.n_top_places <= all_standings["position"].max():
        top_cols = list(range(1, config.n_top_places + 1))
        top_probs = position_counts[top_cols].sum(axis=1) / position_counts.sum(axis=1)
        top_probs = top_probs.sort_values(ascending=False)
    else:
        top_probs = pd.Series(0.0, index=position_counts.index)
    
    # Relegation probabilities
    releg_cols = list(range(n_teams - config.n_relegation_places + 1, n_teams + 1))
    releg_probs = position_counts[releg_cols].sum(axis=1) / position_counts.sum(axis=1)
    releg_probs = releg_probs.sort_values(ascending=False)
    
    # Expected points
    expected_points = all_standings.groupby("team_id")["points"].mean().sort_values(ascending=False)
    
    # Expected position
    expected_position = all_standings.groupby("team_id")["position"].mean().sort_values()
    
    # Position distribution (for each team, distribution of final positions)
    position_distribution = position_counts.div(position_counts.sum(axis=1), axis=0)
    
    return {
        "position_counts": position_counts,
        "title_probabilities": title_probs,
        "top_n_probabilities": top_probs,
        "relegation_probabilities": releg_probs,
        "expected_points": expected_points,
        "expected_position": expected_position,
        "position_distribution": position_distribution,
    }


def run_league_simulation(
    fixtures: pd.DataFrame,
    probability_provider: ProbabilityProvider,
    config: SimulatorConfig | None = None,
    history_provider: Callable[[int], pd.DataFrame] | None = None,
    existing_results: pd.DataFrame | None = None,
    seed: int | None = None
) -> dict:
    """High-level function to run a complete league simulation.
    
    Args:
        fixtures: All fixtures for the season.
        probability_provider: Function to get match outcome probabilities.
        config: Simulation configuration.
        history_provider: Function to get history for a given matchweek.
        existing_results: Already completed matches.
        seed: Random seed (overrides config).
    
    Returns:
        Dictionary with simulation results and statistics.
    """
    if config is None:
        config = SimulatorConfig()
    if seed is not None:
        config = SimulatorConfig(
            n_simulations=config.n_simulations,
            random_seed=seed,
            points_win=config.points_win,
            points_draw=config.points_draw,
            points_loss=config.points_loss,
            n_relegation_places=config.n_relegation_places,
            n_top_places=config.n_top_places,
            tiebreaker=config.tiebreaker,
        )
    
    # Validate
    validate_fixtures(fixtures)
    
    # Get teams
    home_teams = set(fixtures["home_team_id"].unique())
    away_teams = set(fixtures["away_team_id"].unique())
    teams = sorted(home_teams | away_teams)
    n_teams = len(teams)
    
    # Run simulation
    all_standings, sim_info = simulate_season(
        fixtures=fixtures,
        probability_provider=probability_provider,
        config=config,
        history_provider=history_provider,
        existing_results=existing_results,
    )
    
    # Compute statistics
    stats = compute_simulation_statistics(all_standings, n_teams, config)
    
    return {
        "standings": all_standings,
        "statistics": stats,
        "simulation_info": sim_info,
        "teams": teams,
        "config": config,
    }


# Convenience function for simple probability provider using global frequencies
def make_scoreline_poisson_provider(
    fixtures: pd.DataFrame,
    smoothing_alpha: float = 1.0,
    home_advantage: float = 0.0,
    min_history: int = 5
) -> ProbabilityProvider:
    """Create a probability provider using independent Poisson models for home and away goals.
    
    This provider estimates home and away goal rates from historical data and samples
    from independent Poisson distributions to generate scorelines.
    
    Assumptions:
    - Home and away goals are independent (a known limitation; real matches have dependence)
    - Goal rates are estimated from historical data with Laplace smoothing
    - Home advantage is modeled as an additive boost to home goal rate
    
    Args:
        fixtures: Historical fixtures to compute goal rates from.
        smoothing_alpha: Laplace smoothing parameter for goal rate estimation.
        home_advantage: Additive boost to home team's goal rate (default 0.0).
        min_history: Minimum matches required for team-specific rates.
    
    Returns:
        ProbabilityProvider function returning (p_home, p_draw, p_away) derived from scorelines.
    """
    from src.features.match_baselines import compute_global_outcome_frequencies, MatchBaselineConfig
    
    # Pre-compute global average goals per match for fallback
    valid_fixtures = fixtures.dropna(subset=["home_score", "away_score"])
    if len(valid_fixtures) > 0:
        global_home_goals = valid_fixtures["home_score"].mean()
        global_away_goals = valid_fixtures["away_score"].mean()
    else:
        global_home_goals = 1.4
        global_away_goals = 1.2
    
    # Pre-compute team-specific goal rates for each team (as home and as away)
    team_home_rates = {}
    team_away_rates = {}
    
    all_teams = set(fixtures["home_team_id"].unique()) | set(fixtures["away_team_id"].unique())
    for team_id in all_teams:
        # Home matches for this team
        home_matches = fixtures[fixtures["home_team_id"] == team_id].dropna(subset=["home_score", "away_score"])
        if len(home_matches) >= min_history:
            team_home_rates[team_id] = home_matches["home_score"].mean()
        else:
            team_home_rates[team_id] = None
            
        # Away matches for this team
        away_matches = fixtures[fixtures["away_team_id"] == team_id].dropna(subset=["home_score", "away_score"])
        if len(away_matches) >= min_history:
            team_away_rates[team_id] = away_matches["away_score"].mean()
        else:
            team_away_rates[team_id] = None
    
    config = MatchBaselineConfig(min_history=5, smoothing_alpha=1.0)
    
    def provider(match_row: pd.Series, history: pd.DataFrame) -> tuple[float, float, float]:
        home_team_id = match_row["home_team_id"]
        away_team_id = match_row["away_team_id"]
        
        # Get goal rates for home team (as home)
        if home_team_id in team_home_rates and team_home_rates[home_team_id] is not None:
            home_goal_rate = team_home_rates[home_team_id] + home_advantage
        elif not history.empty:
            # Fallback to history-specific average
            valid_hist = history.dropna(subset=["home_score", "away_score"])
            if len(valid_hist) > 0:
                home_goal_rate = valid_hist["home_score"].mean() + home_advantage
            else:
                home_goal_rate = global_home_goals + home_advantage
        else:
            home_goal_rate = global_home_goals + home_advantage
        
        # Get goal rates for away team (as away)
        if away_team_id in team_away_rates and team_away_rates[away_team_id] is not None:
            away_goal_rate = team_away_rates[away_team_id]
        elif not history.empty:
            valid_hist = history.dropna(subset=["home_score", "away_score"])
            if len(valid_hist) > 0:
                away_goal_rate = valid_hist["away_score"].mean()
            else:
                away_goal_rate = global_away_goals
        else:
            away_goal_rate = global_away_goals
        
        # Ensure non-negative rates
        home_goal_rate = max(0.01, home_goal_rate)
        away_goal_rate = max(0.01, away_goal_rate)
        
        # Compute scoreline probabilities from independent Poissons
        # We need to compute P(home win), P(draw), P(away win) from Poisson distributions
        # P(home win) = sum_{h=0}^{inf} sum_{a=0}^{h-1} Pois(h; lambda_h) * Pois(a; lambda_a)
        # P(draw) = sum_{k=0}^{inf} Pois(k; lambda_h) * Pois(k; lambda_a)
        # P(away win) = sum_{a=0}^{inf} sum_{h=0}^{a-1} Pois(h; lambda_h) * Pois(a; lambda_a)
        
        # Compute up to reasonable max goals (e.g., 10)
        max_goals = 10
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        
        for h in range(max_goals + 1):
            p_h = poisson.pmf(h, home_goal_rate)
            for a in range(max_goals + 1):
                p_a = poisson.pmf(a, away_goal_rate)
                prob = p_h * p_a
                if h > a:
                    p_home += prob
                elif h == a:
                    p_draw += prob
                else:
                    p_away += prob
        
        # Renormalize (in case of truncation)
        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total
        else:
            # Fallback to uniform
            p_home, p_draw, p_away = 1/3, 1/3, 1/3
        
        return (float(p_home), float(p_draw), float(p_away))
    
    return provider


# =============================================================================
# Scoreline simulation functions
# =============================================================================

def simulate_scoreline(
    home_goal_rate: float,
    away_goal_rate: float,
    rng: np.random.Generator,
    max_goals: int = 10
) -> tuple[int, int]:
    """Simulate a single match scoreline from independent Poisson distributions.
    
    Args:
        home_goal_rate: Home team's expected goals (lambda for Poisson).
        away_goal_rate: Away team's expected goals (lambda for Poisson).
        rng: NumPy random generator.
        max_goals: Maximum goals to consider (for validation).
    
    Returns:
        Tuple of (home_goals, away_goals).
    """
    home_goals = rng.poisson(home_goal_rate)
    away_goals = rng.poisson(away_goal_rate)
    return int(home_goals), int(away_goals)


def outcome_from_scoreline(home_goals: int, away_goals: int) -> int:
    """Convert scoreline to outcome code.
    
    Args:
        home_goals: Home team goals.
        away_goals: Away team goals.
    
    Returns:
        2 for home win, 1 for draw, 0 for away win.
    """
    if home_goals > away_goals:
        return 2  # Home win
    elif home_goals == away_goals:
        return 1  # Draw
    else:
        return 0  # Away win
# =============================================================================
# Convenience functions for probability providers
# =============================================================================

def make_global_frequency_provider(
    fixtures: pd.DataFrame,
    smoothing_alpha: float = 1.0
) -> ProbabilityProvider:
    """Create a probability provider using global outcome frequencies.
    
    This provider uses only the global home/draw/away frequencies from
    the provided fixtures data, with Laplace smoothing.
    
    Args:
        fixtures: Historical fixtures to compute frequencies from.
        smoothing_alpha: Laplace smoothing parameter.
    
    Returns:
        ProbabilityProvider function.
    """
    from src.features.match_baselines import compute_global_outcome_frequencies
    
    # Pre-compute global frequencies
    global_probs = compute_global_outcome_frequencies(fixtures, smoothing_alpha)
    
    def provider(match_row: pd.Series, history: pd.DataFrame) -> tuple[float, float, float]:
        return global_probs
    
    return provider


def make_recent_form_provider(
    fixtures: pd.DataFrame,
    window: int = 5,
    smoothing_alpha: float = 1.0
) -> ProbabilityProvider:
    """Create a probability provider using recent form.
    
    Args:
        fixtures: Historical fixtures.
        window: Number of recent matches to consider.
        smoothing_alpha: Laplace smoothing parameter.
    
    Returns:
        ProbabilityProvider function.
    """
    from src.features.match_baselines import compute_recent_form_baseline, compute_global_outcome_frequencies, MatchBaselineConfig
    
    config = MatchBaselineConfig(min_history=1, smoothing_alpha=smoothing_alpha)
    
    def provider(match_row: pd.Series, history: pd.DataFrame) -> tuple[float, float, float]:
        if history.empty:
            return compute_global_outcome_frequencies(fixtures, smoothing_alpha)
        
        global_probs = compute_global_outcome_frequencies(history, smoothing_alpha)
        return compute_recent_form_baseline(
            history,
            match_row["home_team_id"],
            match_row["away_team_id"],
            window,
            global_probs,
            smoothing_alpha
        )
    
    return provider
