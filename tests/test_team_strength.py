"""Tests for Phase 9.2A team strength modelling."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.team_strength import (
    EloConfig,
    EloRatingSystem,
    TeamStrengthConfig,
    TeamStrengthIndex,
    predict_match_outcome,
    compute_recent_form,
    estimate_attack_defence_strength,
)
from src.features.player_intelligence import (
    get_position_band_fpl,
    get_position_band_statsbomb,
    get_position_band_auto,
)
from src.evaluation.team_strength_evaluation import (
    compute_global_outcome_frequencies,
    compute_team_strength_baseline,
    compute_recent_form_baseline,
    EloRatingSystem as EloRatingSystemEval,
)


class TestEloConfig:
    """Tests for EloConfig."""

    def test_default_config(self):
        config = EloConfig()
        assert config.initial_rating == 1500.0
        assert config.k_factor == 20.0
        assert config.home_advantage == 100.0
        assert config.draw_weight == 0.5
        assert config.min_matches_for_rating == 5
        assert config.carry_over_seasons is True
        assert config.season_carry_over_factor == 0.5


class TestEloRatingSystem:
    """Tests for EloRatingSystem."""

    def test_initial_rating(self):
        elo = EloRatingSystem()
        assert elo.get_rating(999) == 1500.0

    def test_expected_score(self):
        elo = EloRatingSystem()
        # Equal ratings
        assert elo.expected_score(1500, 1500) == 0.5
        # 200 point advantage
        exp = elo.expected_score(1700, 1500)
        assert 0.7 < exp < 0.8
        # Large difference
        exp = elo.expected_score(2000, 1000)
        assert exp > 0.99

    def test_update_ratings(self):
        elo = EloRatingSystem()
        # Team 1 beats team 2
        elo.update_ratings(1, 2, 2, 1, home_advantage=True)
        assert elo.get_rating(1) > 1500.0
        assert elo.get_rating(2) < 1500.0
        # Sum of changes should be zero
        assert abs(elo.get_rating(1) - 1500.0 + 1500.0 - elo.get_rating(2)) < 1e-10

    def test_draw(self):
        elo = EloRatingSystem()
        # Draw
        elo.update_ratings(1, 2, 1, 1, True)
        # Both should move slightly towards 1500
        assert elo.get_rating(1) < 1500.0
        assert elo.get_rating(2) > 1500.0

    def test_predict_match(self):
        elo = EloRatingSystem()
        # Team 1 much stronger
        elo.ratings[1] = 1800
        elo.ratings[2] = 1400
        hw, d, aw = elo.predict_match(1, 2)
        assert hw > 0.5
        assert aw < 0.5
        # Sum should be ~1
        assert abs(hw + d + aw - 1.0) < 0.01


class TestEloConfig:
    def test_default_config(self):
        config = EloConfig()
        assert config.initial_rating == 1500.0
        assert config.k_factor == 20.0
        assert config.home_advantage == 100.0
        assert config.draw_weight == 0.5


class TestTeamStrengthConfig:
    def test_default_config(self):
        config = TeamStrengthConfig()
        assert config.weight_elo == 0.3
        assert config.weight_attack == 0.2
        assert config.weight_defence == 0.2


class TestGlobalOutcomeFrequencies:
    def test_compute_global_outcome_frequencies(self):
        history = pd.DataFrame({
            "home_score": [2, 1, 3, 0, 2],
            "away_score": [1, 1, 0, 2, 1],
        })
        hw, d, aw = compute_global_outcome_frequencies(history, 1.0)
        assert abs(hw + d + aw - 1.0) < 1e-10

    def test_empty_history(self):
        history = pd.DataFrame({"home_score": [], "away_score": []})
        hw, d, aw = compute_global_outcome_frequencies(history, 1.0)
        assert hw == d == aw == 1/3


class TestEloRatingSystem:
    def test_expected_score(self):
        elo = EloRatingSystem()
        # Equal ratings
        assert elo.expected_score(1500, 1500) == 0.5
        # 200 point advantage
        exp = elo.expected_score(1700, 1500)
        assert 0.7 < exp < 0.8
        # Large difference
        exp = elo.expected_score(2000, 1000)
        assert exp > 0.99

    def test_update_ratings(self):
        elo = EloRatingSystem()
        # Team 1 beats team 2
        elo.update_ratings(1, 2, 2, 1, home_advantage=True)
        assert elo.get_rating(1) > 1500.0
        assert elo.get_rating(2) < 1500.0

    def test_draw(self):
        elo = EloRatingSystem()
        # Draw
        elo.update_ratings(1, 2, 1, 1, True)
        # Both should move slightly towards 1500
        assert elo.get_rating(1) < 1500.0
        assert elo.get_rating(2) > 1500.0

    def test_predict_match(self):
        elo = EloRatingSystem()
        # Team 1 much stronger
        elo.ratings[1] = 1800
        elo.ratings[2] = 1400
        hw, d, aw = elo.predict_match(1, 2)
        assert hw > 0.5
        assert aw < 0.5
        # Probabilities should be positive
        assert hw > 0 and d > 0 and aw > 0


class TestTeamStrengthConfig:
    def test_default_config(self):
        config = TeamStrengthConfig()
        assert config.weight_elo == 0.3
        assert config.weight_attack == 0.2
        assert config.weight_defence == 0.2


class TestTeamStrengthIndex:
    def test_elo_integration(self):
        from src.features.team_strength import TeamStrengthIndex, EloRatingSystem, EloConfig
        
        config = TeamStrengthConfig()
        index = TeamStrengthIndex(config)
        
        # Create minimal valid data
        df = pd.DataFrame({
            "match_id": [1, 2],
            "season_id": [2023, 2023],
            "match_date": ["2023-08-12", "2023-08-19"],
            "home_team_id": [1, 2],
            "away_team_id": [2, 3],
            "home_score": [2, 1],
            "away_score": [1, 1],
            "home_team_name": ["Team A", "Team B"],
            "away_team_name": ["Team B", "Team C"],
        })
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])