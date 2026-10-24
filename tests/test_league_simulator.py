"""Tests for league simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.league_simulator import (
    SimulatorConfig,
    validate_probabilities,
    validate_fixtures,
    compute_standings,
    simulate_match,
    make_global_frequency_provider,
    make_recent_form_provider,
    run_league_simulation,
)


def _make_test_fixtures(
    num_matchweeks: int = 5,
    num_teams: int = 4,
    include_scores: bool = True
) -> pd.DataFrame:
    """Create test fixtures DataFrame with known outcomes."""
    fixtures_rows = []
    match_id = 100
    base_date = pd.Timestamp("2015-08-10")

    team_ids = list(range(1, num_teams + 1))

    for mw in range(1, num_matchweeks + 1):
        match_date = base_date + pd.Timedelta(days=(mw - 1) * 7)
        for i in range(0, num_teams, 2):
            if i + 1 >= num_teams:
                break
            home_id = team_ids[i]
            away_id = team_ids[i + 1]

            if include_scores:
                home_score = (home_id + mw) % 3
                away_score = (away_id + mw) % 3
            else:
                home_score = None
                away_score = None

            fixtures_rows.append({
                "match_id": match_id,
                "match_week": mw,
                "season_id": 27,
                "season_name": "2015/2016",
                "match_date": match_date,
                "home_team_id": home_id,
                "home_team_name": f"Team{home_id}",
                "away_team_id": away_id,
                "away_team_name": f"Team{away_id}",
                "home_score": home_score,
                "away_score": away_score,
                "match_status": "available",
                "stadium": "Stadium",
                "source": "test",
                "source_url": "test",
                "ingested_at": "2024-01-01",
            })
            match_id += 1

    return pd.DataFrame(fixtures_rows)


class TestValidateProbabilities:
    """Tests for validate_probabilities."""

    def test_valid_probabilities(self):
        """Valid probabilities should not raise."""
        validate_probabilities(0.4, 0.3, 0.3)
        validate_probabilities(1.0, 0.0, 0.0)
        validate_probabilities(0.0, 1.0, 0.0)
        validate_probabilities(0.0, 0.0, 1.0)
        validate_probabilities(1/3, 1/3, 1/3)

    def test_invalid_negative(self):
        """Negative probabilities should raise."""
        with pytest.raises(ValueError):
            validate_probabilities(-0.1, 0.5, 0.6)
        with pytest.raises(ValueError):
            validate_probabilities(0.5, -0.1, 0.6)
        with pytest.raises(ValueError):
            validate_probabilities(0.5, 0.5, -0.1)

    def test_invalid_greater_than_one(self):
        """Probabilities > 1 should raise."""
        with pytest.raises(ValueError):
            validate_probabilities(1.1, 0.0, 0.0)

    def test_invalid_sum(self):
        """Probabilities not summing to 1 should raise."""
        with pytest.raises(ValueError):
            validate_probabilities(0.5, 0.5, 0.5)  # sum = 1.5
        with pytest.raises(ValueError):
            validate_probabilities(0.2, 0.2, 0.2)  # sum = 0.6

    def test_non_finite(self):
        """Non-finite probabilities should raise."""
        with pytest.raises(ValueError):
            validate_probabilities(np.nan, 0.5, 0.5)
        with pytest.raises(ValueError):
            validate_probabilities(np.inf, 0.5, 0.5)


class TestValidateFixtures:
    """Tests for validate_fixtures."""

    def test_valid_fixtures(self):
        """Valid fixtures should not raise."""
        fixtures = _make_test_fixtures(3, 4)
        validate_fixtures(fixtures)

    def test_missing_columns(self):
        """Missing columns should raise."""
        fixtures = _make_test_fixtures(3, 4)
        bad = fixtures.drop(columns=["home_team_id"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_fixtures(bad)

    def test_duplicate_match_ids(self):
        """Duplicate match_ids should raise."""
        fixtures = _make_test_fixtures(3, 4)
        dup = pd.concat([fixtures, fixtures.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate match_ids"):
            validate_fixtures(dup)

    def test_same_team(self):
        """Home and away team same should raise."""
        fixtures = _make_test_fixtures(3, 4)
        bad = fixtures.copy()
        bad.loc[0, "away_team_id"] = bad.loc[0, "home_team_id"]
        with pytest.raises(ValueError, match="home_team_id == away_team_id"):
            validate_fixtures(bad)

    def test_missing_team_ids(self):
        """Missing team IDs should raise."""
        fixtures = _make_test_fixtures(3, 4)
        bad = fixtures.copy()
        bad.loc[0, "home_team_id"] = np.nan
        with pytest.raises(ValueError, match="Missing team IDs"):
            validate_fixtures(bad)


class TestComputeStandings:
    """Tests for compute_standings."""

    def test_simple_standings(self):
        """Basic standings computation."""
        results = pd.DataFrame({
            "home_team_id": [1, 2],
            "away_team_id": [2, 1],
            "home_score": [2, 1],
            "away_score": [1, 0],
        })
        standings = compute_standings(results, teams=[1, 2])
        
        assert len(standings) == 2
        assert standings.iloc[0]["position"] == 1
        assert standings.iloc[1]["position"] == 2
        # Match 1: Team 1 (home) 2-1 Team 2 → Team 1 wins (3 pts)
        # Match 2: Team 2 (home) 1-0 Team 1 → Team 2 wins (3 pts)
        # Team 1: 3 pts, Team 2: 3 pts
        team1_pts = standings[standings["team_id"] == 1].iloc[0]["points"]
        team2_pts = standings[standings["team_id"] == 2].iloc[0]["points"]
        assert team1_pts == 3
        assert team2_pts == 3

    def test_tiebreaker_goal_diff(self):
        """Goal difference tiebreaker."""
        results = pd.DataFrame({
            "home_team_id": [1, 2],
            "away_team_id": [2, 1],
            "home_score": [2, 0],
            "away_score": [1, 1],
        })
        # Team 1: win (3 pts) + draw (1 pt) = 4 pts, GD +1
        # Team 2: loss (0 pts) + draw (1 pt) = 1 pt, GD -1
        standings = compute_standings(results, teams=[1, 2], config=SimulatorConfig(tiebreaker="goal_diff_then_goals"))
        assert standings.iloc[0]["team_id"] == 1
        assert standings.iloc[1]["team_id"] == 2

    def test_existing_points(self):
        """Standings with existing points."""
        results = pd.DataFrame({
            "home_team_id": [1, 2],
            "away_team_id": [2, 1],
            "home_score": [2, 1],
            "away_score": [1, 0],
        })
        # Match 1: Team 1 wins 2-1 (3 pts)
        # Match 2: Team 2 wins 1-0 (3 pts)
        existing = {1: 10, 2: 5}
        standings = compute_standings(results, teams=[1, 2], existing_points=existing)
        assert standings[standings["team_id"] == 1].iloc[0]["points"] == 13  # 10 + 3
        assert standings[standings["team_id"] == 2].iloc[0]["points"] == 8   # 5 + 3

    def test_existing_gd_gs(self):
        """Standings with existing GD and GS."""
        results = pd.DataFrame({
            "home_team_id": [1],
            "away_team_id": [2],
            "home_score": [2],
            "away_score": [1],
        })
        existing_gd = {1: 5, 2: -3}
        existing_gs = {1: 10, 2: 8}
        standings = compute_standings(
            results, teams=[1, 2], 
            existing_gd=existing_gd, existing_gs=existing_gs
        )
        assert standings[standings["team_id"] == 1].iloc[0]["goal_diff"] == 6  # 5 + 1
        assert standings[standings["team_id"] == 2].iloc[0]["goal_diff"] == -4  # -3 - 1


class TestSimulateMatch:
    """Tests for simulate_match."""

    def test_distribution(self):
        """Simulated matches should follow probability distribution."""
        rng = np.random.default_rng(42)
        outcomes = [simulate_match(0.5, 0.3, 0.2, rng) for _ in range(10000)]
        counts = pd.Series(outcomes).value_counts().sort_index()
        
        # Check approximate proportions
        assert abs(counts.get(2, 0) / 10000 - 0.5) < 0.02
        assert abs(counts.get(1, 0) / 10000 - 0.3) < 0.02
        assert abs(counts.get(0, 0) / 10000 - 0.2) < 0.02

    def test_deterministic(self):
        """Same seed should give same results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        o1 = [simulate_match(0.4, 0.3, 0.3, rng1) for _ in range(100)]
        o2 = [simulate_match(0.4, 0.3, 0.3, rng2) for _ in range(100)]
        assert o1 == o2


class TestGlobalFrequencyProvider:
    """Tests for make_global_frequency_provider."""

    def test_provider_returns_valid_probs(self):
        """Provider should return valid probability vectors."""
        fixtures = _make_test_fixtures(5, 4)
        provider = make_global_frequency_provider(fixtures)
        
        history = fixtures[fixtures["match_week"] < 3].copy()
        match_row = fixtures[fixtures["match_week"] == 3].iloc[0]
        
        probs = provider(match_row, history)
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-9
        assert all(p >= 0 for p in probs)

    def test_provider_consistent(self):
        """Provider should return same probs for all matches."""
        fixtures = _make_test_fixtures(5, 4)
        provider = make_global_frequency_provider(fixtures)
        
        history = fixtures[fixtures["match_week"] < 3].copy()
        probs1 = provider(fixtures.iloc[0], history)
        probs2 = provider(fixtures.iloc[1], history)
        assert probs1 == probs2


class TestRecentFormProvider:
    """Tests for make_recent_form_provider."""

    def test_provider_with_history(self):
        """Provider should use recent form when history available."""
        fixtures = _make_test_fixtures(5, 4)
        provider = make_recent_form_provider(fixtures, window=3)
        
        history = fixtures[fixtures["match_week"] < 4].copy()
        match_row = fixtures[fixtures["match_week"] == 4].iloc[0]
        
        probs = provider(match_row, history)
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_provider_empty_history(self):
        """Provider should fall back to global when history empty."""
        fixtures = _make_test_fixtures(5, 4)
        provider = make_recent_form_provider(fixtures, window=3)
        
        history = pd.DataFrame(columns=fixtures.columns)
        match_row = fixtures.iloc[0]
        
        probs = provider(match_row, history)
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_provider_handles_none_scores(self):
        """Provider should handle None scores in history."""
        fixtures = _make_test_fixtures(5, 4)
        # Add a match with None scores
        fixtures.loc[len(fixtures)] = {
            "match_id": 999, "match_week": 1, "season_id": 27,
            "match_date": pd.Timestamp("2015-08-10"),
            "home_team_id": 1, "away_team_id": 2,
            "home_score": None, "away_score": None,
            "home_team_name": "Team1", "away_team_name": "Team2",
            "match_status": "available", "stadium": "Stadium",
            "source": "test", "source_url": "test", "ingested_at": "2024-01-01"
        }
        
        provider = make_recent_form_provider(fixtures, window=3)
        history = fixtures[fixtures["match_week"] < 2].copy()
        match_row = fixtures[fixtures["match_week"] == 2].iloc[0]
        
        probs = provider(match_row, history)
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-9


class TestRunLeagueSimulation:
    """Tests for run_league_simulation."""

    def test_full_season_no_unplayed(self):
        """Simulation with no unplayed matches should return actual standings."""
        fixtures = _make_test_fixtures(5, 4, include_scores=True)
        provider = make_global_frequency_provider(fixtures)
        config = SimulatorConfig(n_simulations=1, random_seed=42)
        
        results = run_league_simulation(fixtures, provider, config)
        
        assert "standings" in results
        assert "statistics" in results
        assert len(results["teams"]) == 4

    def test_partial_season(self):
        """Simulation from partial season."""
        fixtures = _make_test_fixtures(5, 4, include_scores=True)
        # Set matchweeks 4-5 as unplayed
        fixtures.loc[fixtures["match_week"] >= 4, "home_score"] = None
        fixtures.loc[fixtures["match_week"] >= 4, "away_score"] = None
        
        provider = make_global_frequency_provider(fixtures)
        config = SimulatorConfig(n_simulations=10, random_seed=42)
        
        results = run_league_simulation(fixtures, provider, config)
        
        assert "standings" in results
        assert len(results["standings"]) > 0
        # Should have 10 simulations * 4 teams = 40 rows
        assert len(results["standings"]) == 10 * 4

    def test_start_of_season(self):
        """Simulation from start of season with no scores."""
        fixtures = _make_test_fixtures(5, 4, include_scores=False)
        provider = make_global_frequency_provider(fixtures)
        config = SimulatorConfig(n_simulations=10, random_seed=42)
        
        results = run_league_simulation(fixtures, provider, config)
        
        assert len(results["standings"]) == 10 * 4

    def test_reproducible(self):
        """Same seed should give identical results."""
        fixtures = _make_test_fixtures(5, 4, include_scores=False)
        provider = make_global_frequency_provider(fixtures)
        config = SimulatorConfig(n_simulations=10, random_seed=42)
        
        results1 = run_league_simulation(fixtures, provider, config)
        config2 = SimulatorConfig(n_simulations=10, random_seed=42)
        results2 = run_league_simulation(fixtures, provider, config2)
        
        pd.testing.assert_frame_equal(
            results1["standings"].sort_values(["simulation", "position"]).reset_index(drop=True),
            results2["standings"].sort_values(["simulation", "position"]).reset_index(drop=True)
        )

    def test_statistics_structure(self):
        """Statistics should have all required keys."""
        fixtures = _make_test_fixtures(5, 4, include_scores=False)
        provider = make_global_frequency_provider(fixtures)
        config = SimulatorConfig(n_simulations=10, random_seed=42)
        
        results = run_league_simulation(fixtures, provider, config)
        stats = results["statistics"]
        
        required_keys = [
            "position_counts", "title_probabilities", "top_n_probabilities",
            "relegation_probabilities", "expected_points", "expected_position",
            "position_distribution"
        ]
        for key in required_keys:
            assert key in stats

    def test_probabilities_sum_to_valid_range(self):
        """All probabilities should be in [0, 1]."""
        fixtures = _make_test_fixtures(5, 4, include_scores=False)
        provider = make_global_frequency_provider(fixtures)
        config = SimulatorConfig(n_simulations=10, random_seed=42)
        
        results = run_league_simulation(fixtures, provider, config)
        stats = results["statistics"]
        
        for key in ["title_probabilities", "top_n_probabilities", "relegation_probabilities"]:
            probs = stats[key]
            assert (probs >= 0).all()
            assert (probs <= 1).all()

    def test_simulation_with_recent_form_provider(self):
        """Simulation should work with recent form provider."""
        # Use a setup with some played and some unplayed matches
        fixtures = _make_test_fixtures(5, 4, include_scores=True)
        # Set matchweeks 4-5 as unplayed
        fixtures.loc[fixtures["match_week"] >= 4, "home_score"] = None
        fixtures.loc[fixtures["match_week"] >= 4, "away_score"] = None
        
        provider = make_recent_form_provider(fixtures, window=3)
        config = SimulatorConfig(n_simulations=10, random_seed=42)
        
        results = run_league_simulation(fixtures, provider, config)
        assert "standings" in results
        # 10 simulations * 4 teams = 40 rows
        assert len(results["standings"]) == 10 * 4


class TestSimulatorConfig:
    """Tests for SimulatorConfig."""

    def test_defaults(self):
        """Default config should have sensible values."""
        config = SimulatorConfig()
        assert config.n_simulations == 10_000
        assert config.random_seed == 42
        assert config.points_win == 3
        assert config.points_draw == 1
        assert config.points_loss == 0
        assert config.n_relegation_places == 3
        assert config.n_top_places == 4

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = SimulatorConfig(
            n_simulations=1000,
            random_seed=123,
            points_win=2,
            n_relegation_places=4,
            n_top_places=6,
            tiebreaker="goal_diff_then_goals"
        )
        assert config.n_simulations == 1000
        assert config.random_seed == 123
        assert config.points_win == 2
        assert config.n_relegation_places == 4
        assert config.n_top_places == 6
        assert config.tiebreaker == "goal_diff_then_goals"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_team(self):
        """Single team league should work."""
        fixtures = pd.DataFrame({
            "match_id": [100],
            "match_week": [1],
            "season_id": [27],
            "match_date": [pd.Timestamp("2015-08-10")],
            "home_team_id": [1],
            "away_team_id": [1],  # Same team (edge case, but validation will catch)
        })
        # This should fail validation
        with pytest.raises(ValueError):
            validate_fixtures(fixtures)

    def test_two_teams(self):
        """Two-team league should work."""
        base_date = pd.Timestamp("2015-08-10")
        fixtures = pd.DataFrame({
            "match_id": [100, 101],
            "match_week": [1, 2],
            "season_id": [27, 27],
            "match_date": [base_date, base_date + pd.Timedelta(days=7)],
            "home_team_id": [1, 2],
            "away_team_id": [2, 1],
            "home_score": [None, None],
            "away_score": [None, None],
            "home_team_name": ["Team1", "Team2"],
            "away_team_name": ["Team2", "Team1"],
            "match_status": "available",
            "stadium": "Stadium",
            "source": "test",
            "source_url": "test",
            "ingested_at": "2024-01-01",
        })
        provider = make_global_frequency_provider(fixtures)
        # Use 1 relegation place for 2-team league
        config = SimulatorConfig(n_simulations=10, random_seed=42, n_relegation_places=1)
        
        results = run_league_simulation(fixtures, provider, config)
        assert len(results["standings"]) == 10 * 2

    def test_duplicate_fixtures_rejected(self):
        """Duplicate fixtures should be rejected."""
        fixtures = _make_test_fixtures(3, 4)
        dup = pd.concat([fixtures, fixtures.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate match_ids"):
            validate_fixtures(dup)

    def test_missing_scores_handled(self):
        """Missing scores should be handled gracefully."""
        fixtures = _make_test_fixtures(5, 4, include_scores=False)
        provider = make_global_frequency_provider(fixtures)
        config = SimulatorConfig(n_simulations=5, random_seed=42)
        
        results = run_league_simulation(fixtures, provider, config)
        assert len(results["standings"]) == 5 * 4


class TestIntegrationWithRealData:
    """Integration tests with real StatsBomb data."""

    def test_real_data_simulation(self):
        """Simulation with real StatsBomb data."""
        fixtures = pd.read_parquet('data/processed/statsbomb/fixtures.parquet')
        fixtures_2015 = fixtures[fixtures['season_id'] == 27].copy()
        
        # Only run if data exists
        if len(fixtures_2015) > 0:
            provider = make_global_frequency_provider(fixtures_2015)
            config = SimulatorConfig(n_simulations=50, random_seed=42)
            
            results = run_league_simulation(
                fixtures=fixtures_2015,
                probability_provider=provider,
                config=config
            )
            
            assert "standings" in results
            assert "statistics" in results
            assert len(results["teams"]) == 20  # 20 teams in 2015/16 PL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])