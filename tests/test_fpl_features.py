"""Tests for FPL feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.fpl_features import (
    FPLFeatureConfig,
    _check_feature_availability,
    _compute_player_rolling_stats,
    _compute_player_season_stats,
    _compute_team_strength_features,
    build_fpl_player_features,
    compute_fpl_features_for_cutoff,
    build_fpl_features,
)

from src.dashboard.data_loader import load_player_season_fpl, load_player_match_fpl


class TestFPLFeatureConfig:
    """Tests for FPLFeatureConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FPLFeatureConfig()
        assert config.rolling_window == 5
        assert config.min_history == 3
        assert config.include_xg is True
        assert config.include_starts is True
        assert config.include_position_team is True
        assert config.gameweek_col == "gameweek"
        assert config.season_col == "season_name"
        assert config.player_id_col == "player_id"
        assert config.team_col == "team"
        assert config.position_col == "position"


class TestFeatureAvailabilityChecks:
    """Tests for feature availability detection."""

    def test_check_feature_availability_true(self):
        """Test detection of available features."""
        df = pd.DataFrame({
            "expected_goals": [1.0, 0.5, 0.0],
            "expected_assists": [0.5, 0.3, 0.1],
        })
        assert _check_feature_availability(df, ["expected_goals", "expected_assists"]) is True

    def test_check_feature_availability_false_missing_col(self):
        """Test detection of missing column."""
        df = pd.DataFrame({"expected_goals": [1.0, 0.5]})
        assert _check_feature_availability(df, ["expected_goals", "expected_assists"]) is False

    def test_check_feature_availability_false_all_zero(self):
        """Test detection when all values are zero."""
        df = pd.DataFrame({
            "expected_goals": [0.0, 0.0, 0.0],
            "expected_assists": [0.0, 0.0, 0.0],
        })
        assert _check_feature_availability(df, ["expected_goals", "expected_assists"]) is False

    def test_check_feature_availability_false_all_nan(self):
        """Test detection when all values are NaN."""
        df = pd.DataFrame({
            "expected_goals": [np.nan, np.nan],
            "expected_assists": [np.nan, np.nan],
        })
        assert _check_feature_availability(df, ["expected_goals", "expected_assists"]) is False


class TestPlayerRollingStats:
    """Tests for player rolling statistics computation."""

    def test_rolling_stats_basic(self):
        """Test basic rolling statistics computation."""
        config = FPLFeatureConfig(rolling_window=3)
        
        # Create player history with 5 gameweeks
        history = pd.DataFrame({
            "gameweek": [1, 2, 3, 4, 5],
            "season_id": [2023, 2023, 2023, 2023, 2023],
            "minutes": [90, 90, 90, 90, 90],
            "goals": [1, 0, 1, 0, 1],
            "assists": [0, 1, 0, 1, 0],
            "xg": [0.5, 0.3, 0.7, 0.2, 0.6],
            "xa": [0.1, 0.4, 0.2, 0.3, 0.1],
        })

        stat_cols = ["goals", "assists", "xg", "xa"]
        result = _compute_player_rolling_stats(
            history, FPLFeatureConfig(rolling_window=3), target_gameweek=5, stat_cols=stat_cols
        )

        # Should compute rolling mean over last 3 gameweeks (2,3,4)
        assert "goals_rolling_3" in result
        assert "assists_rolling_3" in result
        assert "xg_rolling_3" in result
        assert "xa_rolling_3" in result
        
        # Goals in GW 2,3,4: 0, 1, 0 -> mean = 1/3
        assert abs(result["goals_rolling_3"] - 1/3) < 0.01
        # Assists in GW 2,3,4: 1, 0, 1 -> mean = 2/3
        assert abs(result["assists_rolling_3"] - 2/3) < 0.01

    def test_rolling_stats_insufficient_history(self):
        """Test rolling stats with no history before target."""
        config = FPLFeatureConfig(rolling_window=5)
        
        history = pd.DataFrame({
            "gameweek": [1, 2],
            "season_id": [2023, 2023],
            "minutes": [90, 90],
            "goals": [1, 0],
            "assists": [0, 1],
        })

        # Target gameweek 1 - no gameweeks before it
        result = _compute_player_rolling_stats(
            history, config, target_gameweek=1, stat_cols=["goals", "assists"]
        )

        # Should return None for all stats (no gameweeks < target)
        assert result["goals_rolling_5"] is None
        assert result["assists_rolling_5"] is None

    def test_rolling_stats_excludes_zero_minutes(self):
        """Test that zero-minute gameweeks are excluded."""
        config = FPLFeatureConfig(rolling_window=3)
        
        history = pd.DataFrame({
            "gameweek": [1, 2, 3, 4],
            "season_id": [2023, 2023, 2023, 2023],
            "minutes": [90, 0, 90, 90],  # GW2 has 0 minutes
            "goals": [1, 0, 1, 0],
            "assists": [0, 0, 0, 1],
        })

        result = _compute_player_rolling_stats(
            history, FPLFeatureConfig(rolling_window=3), target_gameweek=4, stat_cols=["goals"]
        )

        # Should only use GW 1 and 3 (GW2 has 0 minutes, GW4 is target)
        # GW1: 1 goal, GW3: 1 goal -> mean = 1.0
        assert result["goals_rolling_3"] == 1.0


class TestPlayerSeasonStats:
    """Tests for player season statistics computation."""

    def test_season_stats_basic(self):
        """Test prior-season statistics computation."""
        config = FPLFeatureConfig(rolling_window=3)
        
        player_seasons = pd.DataFrame({
            "season_id": [2016, 2017, 2018, 2019, 2020],
            "goals": [10, 15, 12, 8, 20],
            "assists": [5, 8, 6, 4, 10],
            "xg": [12.5, 14.2, 11.8, 7.5, 22.0],
        })

        stat_cols = ["goals", "assists", "xg"]
        result = _compute_player_season_stats(
            player_seasons, config, target_season_id=2021, stat_cols=stat_cols
        )

        # Should use last 3 seasons (2018, 2019, 2020)
        # goals: (12 + 8 + 20) / 3 = 40/3 ≈ 13.33
        assert "goals_season_mean" in result
        assert abs(result["goals_season_mean"] - 40/3) < 0.01
        assert abs(result["assists_season_mean"] - 20/3) < 0.01
        assert abs(result["xg_season_mean"] - (11.8 + 7.5 + 22.0)/3) < 0.01

    def test_season_stats_no_history(self):
        """Test with no prior seasons."""
        config = FPLFeatureConfig(rolling_window=3)
        
        player_seasons = pd.DataFrame({
            "season_id": [2020],
            "goals": [10],
            "assists": [5],
        })
        
        # Target season is 2020, so no prior seasons (2020 is not < 2020)
        result = _compute_player_season_stats(
            player_seasons, config, target_season_id=2020, stat_cols=["goals", "assists"]
        )
        
        assert result["goals_season_mean"] is None
        assert result["assists_season_mean"] is None


class TestTeamStrengthFeatures:
    """Tests for team strength feature computation."""

    def _make_test_match_df(self) -> pd.DataFrame:
        """Create test match data."""
        return pd.DataFrame({
            "season_id": [2023, 2023, 2023, 2023, 2023, 2023],
            "gameweek": [1, 2, 3, 4, 5, 6],
            "team": ["Arsenal", "Arsenal", "Arsenal", "Arsenal", "Chelsea", "Chelsea"],
            "was_home": [True, True, False, False, True, False],
            "team_h_score": [2, 1, 0, 2, 1, 0],
            "team_a_score": [1, 1, 2, 1, 2, 3],
            "minutes": [90, 90, 90, 90, 90, 90],
        })

    def test_home_team_strength(self):
        """Test home team strength features."""
        match_df = self._make_test_match_df()
        config = FPLFeatureConfig()

        result = _compute_team_strength_features(
            match_df, config, target_season_id=2023, target_gameweek=5,
            team_name="Arsenal", is_home=True
        )

        # Arsenal home matches before GW5: GW1 (home vs X, 2-1 win), GW2 (home vs X, 1-1 draw)
        # Should have 2 home matches, 1 win, 1 draw, 0 losses, 3 goals scored, 2 conceded
        assert result["home_matches"] == 2.0
        assert result["home_wins"] == 1.0
        assert result["home_draws"] == 1.0
        assert result["home_losses"] == 0.0
        assert result["home_goals_scored"] == 3.0
        assert result["home_goals_conceded"] == 2.0

    def test_away_team_strength(self):
        """Test away team strength features."""
        config = FPLFeatureConfig()

        result = _compute_team_strength_features(
            self._make_test_match_df(), config, 
            target_season_id=2023, target_gameweek=5,
            team_name="Arsenal", is_home=False
        )

        # Arsenal away matches before GW5: 
        # GW3: was_home=False, team_h=0, team_a=2 -> Arsenal away scored 2
        # GW4: was_home=False, team_h=2, team_a=1 -> Arsenal away scored 1
        assert result["away_matches"] == 2.0
        assert result["away_wins"] == 1.0
        assert result["away_draws"] == 0.0
        assert result["away_losses"] == 1.0
        assert result["away_goals_scored"] == 3.0
        assert result["away_goals_conceded"] == 2.0

    def test_no_matches(self):
        """Test with no matching matches."""
        config = FPLFeatureConfig()

        result = _compute_team_strength_features(
            self._make_test_match_df(), config,
            target_season_id=2023, target_gameweek=5,
            team_name="Liverpool", is_home=True
        )

        assert result["home_matches"] == 0.0
        assert result["home_wins"] == 0.0


class TestBuildFPLFeatures:
    """Integration tests for build_fpl_player_features."""

    def test_build_features_basic(self):
        """Test basic feature building with synthetic data."""
        from src.features.fpl_features import FPLFeatureConfig, build_fpl_player_features
        import pandas as pd
        import numpy as np

        # Create minimal synthetic test data
        np.random.seed(42)
        n_players = 3
        n_gameweeks = 5
        
        # Create player season data
        ps_rows = []
        for p in range(n_players):
            ps_rows.append({
                "player_id": p + 1,
                "season_id": 2023,
                "season_name": "2023/24",
                "minutes": np.random.randint(500, 3000),
                "goals": np.random.randint(0, 20),
                "assists": np.random.randint(0, 15),
                "expected_goals": np.random.uniform(0, 20),
                "expected_assists": np.random.uniform(0, 10),
                "starts": np.random.randint(10, 38),
                "position": np.random.choice(["GK", "DEF", "MID", "FWD"]),
                "team": np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"]),
            })
        ps = pd.DataFrame(ps_rows)

        # Create player match data
        pm_rows = []
        for p in range(n_players):
            for gw in range(1, 4):  # Just 3 gameweeks
                home_team = np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"])
                away_team = np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"])
                while away_team == home_team:
                    away_team = np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"])
                
                is_home = np.random.choice([True, False])
                team = home_team if is_home else away_team
                
                home_score = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
                away_score = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
                
                pm_rows.append({
                    "player_id": p + 1,
                    "season_id": 2023,
                    "season_name": "2023/24",
                    "gameweek": gw,
                    "team": team,
                    "was_home": is_home,
                    "minutes": np.random.choice([0, 90], p=[0.1, 0.9]),
                    "goals": np.random.choice([0, 1], p=[0.9, 0.1]),
                    "assists": np.random.choice([0, 1], p=[0.95, 0.05]),
                    "expected_goals": np.random.uniform(0, 1),
                    "expected_assists": np.random.uniform(0, 0.5),
                    "starts": np.random.choice([0, 1]),
                    "was_home": is_home,
                    "team": team,
                    "team_h_score": home_score,
                    "team_a_score": away_score,
                })
        pm = pd.DataFrame(pm_rows)

        config = FPLFeatureConfig()
        features = build_fpl_player_features(ps, pm, config)

        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert "player_id" in features.columns
        assert "season_id" in features.columns
        assert "gameweek" in features.columns
        assert "goals" in features.columns
        assert "assists" in features.columns

    def test_compute_fpl_features_for_cutoff(self):
        """Test computing features for a specific cutoff gameweek."""
        from src.features.fpl_features import FPLFeatureConfig, compute_fpl_features_for_cutoff
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        n_players = 3
        
        # Create synthetic data
        ps_rows = []
        for p in range(n_players):
            ps_rows.append({
                "player_id": p + 1,
                "season_id": 2023,
                "season_name": "2023/24",
                "minutes": np.random.randint(500, 3000),
                "goals": np.random.randint(0, 20),
                "assists": np.random.randint(0, 15),
                "expected_goals": np.random.uniform(0, 20),
                "expected_assists": np.random.uniform(0, 10),
                "starts": np.random.randint(10, 38),
                "position": np.random.choice(["GK", "DEF", "MID", "FWD"]),
                "team": np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"]),
            })
        ps = pd.DataFrame(ps_rows)

        pm_rows = []
        for p in range(n_players):
            for gw in range(1, 4):
                home_team = np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"])
                away_team = np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"])
                while away_team == home_team:
                    away_team = np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"])
                
                is_home = np.random.choice([True, False])
                team = home_team if is_home else away_team
                
                home_score = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
                away_score = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
                
                pm_rows.append({
                    "player_id": p + 1,
                    "season_id": 2023,
                    "season_name": "2023/24",
                    "gameweek": gw,
                    "team": home_team if is_home else away_team,
                    "was_home": is_home,
                    "minutes": np.random.choice([0, 90], p=[0.1, 0.9]),
                    "goals": np.random.choice([0, 1], p=[0.9, 0.1]),
                    "assists": np.random.choice([0, 1], p=[0.95, 0.05]),
                    "expected_goals": np.random.uniform(0, 1),
                    "expected_assists": np.random.uniform(0, 0.5),
                    "starts": np.random.choice([0, 1]),
                    "was_home": is_home,
                    "team": home_team if is_home else away_team,
                    "team_h_score": home_score,
                    "team_a_score": away_score,
                })
        pm = pd.DataFrame(pm_rows)

        features = compute_fpl_features_for_cutoff(pm, ps, cutoff_gameweek=2)

        if len(features) > 0:
            assert all(features["gameweek"] == 2)
            assert all(features["season_id"] == 2023)

    def test_build_fpl_features_alias(self):
        """Test build_fpl_features alias."""
        from src.features.fpl_features import FPLFeatureConfig, build_fpl_features
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        n_players = 3
        
        ps_rows = []
        for p in range(n_players):
            ps_rows.append({
                "player_id": p + 1,
                "season_id": 2023,
                "season_name": "2023/24",
                "minutes": np.random.randint(500, 3000),
                "goals": np.random.randint(0, 20),
                "assists": np.random.randint(0, 15),
                "expected_goals": np.random.uniform(0, 20),
                "expected_assists": np.random.uniform(0, 10),
                "starts": np.random.randint(10, 38),
                "position": np.random.choice(["GK", "DEF", "MID", "FWD"]),
                "team": np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"]),
            })
        ps = pd.DataFrame(ps_rows)

        pm_rows = []
        for p in range(n_players):
            for gw in range(1, 4):
                home_team = np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"])
                away_team = np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"])
                while away_team == home_team:
                    away_team = np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City"])
                
                is_home = np.random.choice([True, False])
                team = home_team if is_home else away_team
                
                home_score = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
                away_score = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
                
                pm_rows.append({
                    "player_id": p + 1,
                    "season_id": 2023,
                    "season_name": "2023/24",
                    "gameweek": gw,
                    "team": home_team if is_home else away_team,
                    "was_home": is_home,
                    "minutes": np.random.choice([0, 90], p=[0.1, 0.9]),
                    "goals": np.random.choice([0, 1], p=[0.9, 0.1]),
                    "assists": np.random.choice([0, 1], p=[0.95, 0.05]),
                    "expected_goals": np.random.uniform(0, 1),
                    "expected_assists": np.random.uniform(0, 0.5),
                    "starts": np.random.choice([0, 1]),
                    "was_home": is_home,
                    "team": home_team if is_home else away_team,
                    "team_h_score": home_score,
                    "team_a_score": away_score,
                })
        pm = pd.DataFrame(pm_rows)

        features = build_fpl_features(ps, pm)
        assert isinstance(features, pd.DataFrame)


class TestFPLFeatureEdgeCases:
    """Tests for edge cases in FPL feature engineering."""

    def test_empty_dataframe(self):
        """Test with empty DataFrames."""
        config = FPLFeatureConfig()
        ps = pd.DataFrame()
        pm = pd.DataFrame()

        # Should handle gracefully
        features = build_fpl_player_features(ps, pm, config)
        assert isinstance(features, pd.DataFrame)

    def test_missing_columns_handled(self):
        """Test handling of missing columns."""
        config = FPLFeatureConfig()
        
        # Create minimal DataFrames missing some columns
        ps = pd.DataFrame({
            "player_id": [1, 2],
            "season_id": [2023, 2023],
            "season_name": ["2023/24", "2023/24"],
            "minutes": [1000, 500],
            "goals": [5, 3],
            "assists": [2, 1],
            "expected_goals": [5.0, 2.5],
            "expected_assists": [2.0, 1.0],
            "starts": [30, 15],
            "position": ["FWD", "MID"],
            "team": ["Arsenal", "Chelsea"],
        })
        
        pm = pd.DataFrame({
            "player_id": [1, 2],
            "season_id": [2023, 2023],
            "gameweek": [1, 1],
            "team": ["Arsenal", "Chelsea"],
            "was_home": [True, False],
            "minutes": [90, 90],
            "goals": [1, 0],
            "assists": [0, 1],
            "expected_goals": [0.5, 0.2],
            "expected_assists": [0.1, 0.3],
            "starts": [1, 1],
        })

        # Should not crash
        features = build_fpl_player_features(ps, pm)
        assert isinstance(features, pd.DataFrame)


class TestComputeFPLFeaturesForCutoff:
    """Tests for compute_fpl_features_for_cutoff."""

    def test_cutoff_filtering(self):
        """Test that cutoff correctly filters gameweeks."""
        from src.dashboard.data_loader import load_player_season_fpl, load_player_match_fpl

        ps = load_player_season_fpl()
        pm = load_player_match_fpl()

        features = compute_fpl_features_for_cutoff(pm, ps, cutoff_gameweek=20)

        if len(features) > 0:
            assert all(features["gameweek"] == 20)
            assert all(features["season_id"] == 2023)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])