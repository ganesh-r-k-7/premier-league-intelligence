"""Tests for player scouting profiles, rankings, and similarity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.player_scouting import (
    ScoutingConfig,
    build_player_profiles,
    compute_position_rankings,
    compute_similar_players,
    POSITION_BAND_ORDER,
    build_player_profiles,
)


def _make_test_player_season() -> pd.DataFrame:
    """Create test player_season data with varied positions and stats."""
    np.random.seed(42)
    rows = []
    player_id = 100

    # Forwards
    for i in range(4):
        minutes = 2000 + i * 300
        goals = 15 + i * 3
        assists = 5 + i
        xg = 14 + i * 2
        xa = 4 + i
        npxg = 12 + i
        xa_val = 4 + i
        shots = 80 + i * 10
        sot = 35 + i * 5
        minutes = minutes
        rows.append({
            "player_id": player_id,
            "player_name": f"Forward{i+1}",
            "season_id": 27,
            "season_name": "2015/2016",
            "competition_id": 2,
            "team_id": 1 + (i % 5),
            "team_name": f"Team{1 + (i % 5)}",
            "position": "Center Forward",
            "appearances": 35 + i,
            "starts": 32 + i,
            "minutes": minutes,
            "goals": goals,
            "own_goals": 0,
            "assists": assists,
            "shots": shots,
            "shots_on_target": sot,
            "xg": float(xg),
            "npxg": float(npxg),
            "xa": float(xa_val),
            "penalty_goals": 2 + i,
            "penalty_attempts": 3 + i,
            "source": "test",
            "source_url": "test",
            "ingested_at": "2024-01-01",
        })
        player_id += 1

    # Midfielders
    for i in range(4):
        minutes = 2500 + i * 200
        goals = 5 + i * 2
        assists = 8 + i
        xg = 6 + i
        xa = 8 + i
        npxg = 5 + i
        xa_val = 8 + i
        shots = 50 + i * 5
        sot = 18 + i * 2
        minutes = minutes
        rows.append({
            "player_id": player_id,
            "player_name": f"Midfielder{i+1}",
            "season_id": 27,
            "season_name": "2015/2016",
            "competition_id": 2,
            "team_id": 1 + (i % 5),
            "team_name": f"Team{1 + (i % 5)}",
            "position": "Center Midfield",
            "appearances": 34 + i,
            "starts": 30 + i,
            "minutes": minutes,
            "goals": goals,
            "own_goals": 0,
            "assists": assists,
            "shots": shots,
            "shots_on_target": sot,
            "xg": float(xg),
            "npxg": float(npxg),
            "xa": float(xa_val),
            "penalty_goals": 0,
            "penalty_attempts": 0,
            "source": "test",
            "source_url": "test",
            "ingested_at": "2024-01-01",
        })
        player_id += 1

    # Defenders
    for i in range(4):
        minutes = 3000 + i * 100
        goals = 1 + i
        assists = 2 + i
        xg = 1 + i * 0.5
        xa = 1 + i
        npxg = 1 + i * 0.3
        xa_val = 1 + i
        shots = 10 + i * 2
        sot = 3 + i
        minutes = minutes
        rows.append({
            "player_id": player_id,
            "player_name": f"Defender{i+1}",
            "season_id": 27,
            "season_name": "2015/2016",
            "competition_id": 2,
            "team_id": 1 + (i % 5),
            "team_name": f"Team{1 + (i % 5)}",
            "position": "Center Back",
            "appearances": 36 + i,
            "starts": 34 + i,
            "minutes": minutes,
            "goals": goals,
            "own_goals": 0,
            "assists": assists,
            "shots": shots,
            "shots_on_target": sot,
            "xg": float(xg),
            "npxg": float(npxg),
            "xa": float(xa_val),
            "penalty_goals": 0,
            "penalty_attempts": 0,
            "source": "test",
            "source_url": "test",
            "ingested_at": "2024-01-01",
        })
        player_id += 1

    # Goalkeepers
    for i in range(2):
        minutes = 3400 + i * 20
        goals = 0
        assists = 0
        xg = 0.0
        xa = 0.0
        npxg = 0.0
        xa_val = 0.0
        shots = 0
        sot = 0
        minutes = minutes
        rows.append({
            "player_id": player_id,
            "player_name": f"Goalkeeper{i+1}",
            "season_id": 27,
            "season_name": "2015/2016",
            "competition_id": 2,
            "team_id": 1 + (i % 5),
            "team_name": f"Team{1 + (i % 5)}",
            "position": "Goalkeeper",
            "appearances": 38,
            "starts": 38,
            "minutes": minutes,
            "goals": goals,
            "own_goals": 0,
            "assists": assists,
            "shots": shots,
            "shots_on_target": sot,
            "xg": float(xg),
            "npxg": float(npxg),
            "xa": float(xa_val),
            "penalty_goals": 0,
            "penalty_attempts": 0,
            "source": "test",
            "source_url": "test",
            "ingested_at": "2024-01-01",
        })
        player_id += 1

    # Low-minutes player (should be flagged)
    rows.append({
        "player_id": 999,
        "player_name": "Benchwarmer",
        "season_id": 27,
        "season_name": "2015/2016",
        "competition_id": 2,
        "team_id": 1,
        "team_name": "Team1",
        "position": "Center Forward",
        "appearances": 3,
        "starts": 1,
        "minutes": 150,
        "goals": 1,
        "own_goals": 0,
        "assists": 0,
        "shots": 5,
        "shots_on_target": 2,
        "xg": 0.5,
        "npxg": 0.5,
        "xa": 0.1,
        "penalty_goals": 0,
        "penalty_attempts": 0,
        "source": "test",
        "source_url": "test",
        "ingested_at": "2024-01-01",
    })

    df = pd.DataFrame(rows)
    return df


class TestScoutingConfig:
    """Tests for ScoutingConfig."""

    def test_defaults(self):
        config = ScoutingConfig()
        assert config.min_minutes == 450
        assert config.min_appearances == 5
        assert config.min_xg_samples == 3
        assert config.percentile_warning == 95.0
        assert config.similarity_top_k == 5
        assert config.similarity_metric == "cosine"

    def test_custom_config(self):
        config = ScoutingConfig(
            min_minutes=900,
            min_appearances=10,
            min_xg_samples=5,
            percentile_warning=90.0,
            similarity_top_k=10,
            similarity_metric="euclidean"
        )
        assert config.min_minutes == 900
        assert config.min_appearances == 10
        assert config.min_xg_samples == 5
        assert config.percentile_warning == 90.0
        assert config.similarity_top_k == 10
        assert config.similarity_metric == "euclidean"


class TestBuildPlayerProfiles:
    """Tests for build_player_profiles."""

    def test_basic_profiles(self):
        df = _make_test_player_season()
        config = ScoutingConfig(min_minutes=450)
        profiles = build_player_profiles(df, config)

        assert not profiles.empty
        assert "goals_per_90" in profiles.columns
        assert "assists_per_90" in profiles.columns
        assert "xg_per_90" in profiles.columns
        assert "xa_per_90" in profiles.columns
        assert "position_band" in profiles.columns
        assert "minutes_flag" in profiles.columns

    def test_per_90_computation(self):
        df = _make_test_player_season()
        config = ScoutingConfig(min_minutes=1)
        profiles = build_player_profiles(df)

        # Check per-90 computation for Forward1 (15 goals, 2000 minutes)
        fwd = profiles[profiles["player_name"] == "Forward1"].iloc[0]
        expected_g90 = 15 / 2000 * 90
        assert np.isclose(fwd["goals_per_90"], expected_g90, atol=0.01)

    def test_zero_minutes_handling(self):
        df = _make_test_player_season()
        # Add a player with 0 minutes
        df.loc[len(df)] = {
            "player_id": 888, "player_name": "ZeroMin", "season_id": 27,
            "season_name": "2015/2016", "competition_id": 2,
            "team_id": 1, "team_name": "Team1", "position": "Center Forward",
            "appearances": 0, "starts": 0, "minutes": 0,
            "goals": 0, "own_goals": 0, "assists": 0,
            "shots": 0, "shots_on_target": 0,
            "xg": 0.0, "npxg": 0.0, "xa": 0.0,
            "penalty_goals": 0, "penalty_attempts": 0,
            "source": "test", "source_url": "test", "ingested_at": "2024-01-01"
        }
        config = ScoutingConfig(min_minutes=1)
        profiles = build_player_profiles(df, config)
        zero_min = profiles[profiles["player_name"] == "ZeroMin"].iloc[0]
        assert zero_min["goals_per_90"] is None or np.isnan(zero_min["goals_per_90"])
        assert zero_min["minutes_flag"] == "below_min_1"

    def test_efficiency_metrics(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df)
        fwd = profiles[profiles["player_name"] == "Forward1"].iloc[0]
        # goals_per_shot = 15/80 = 0.1875
        assert np.isclose(fwd["goals_per_shot"], 15/80, atol=0.01)
        # goals_per_shot_on_target = 15/35
        assert np.isclose(fwd["goals_per_shot_on_target"], 15/35, atol=0.01)

    def test_goals_over_xg(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df)
        fwd = profiles[profiles["player_name"] == "Forward1"].iloc[0]
        assert np.isclose(fwd["goals_over_xg"], 15 - 14, atol=0.01)

    def test_minutes_flag(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=900))
        low_min = profiles[profiles["player_name"] == "Benchwarmer"].iloc[0]
        assert low_min["minutes_flag"] == "below_min_900"

    def test_high_per_90_warning(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(percentile_warning=50))
        # Forward4 has the highest goals_per_90 (24 goals, 2900 min = 0.745)
        fwd = profiles[profiles["player_name"] == "Forward4"].iloc[0]
        assert fwd["goals_per_90_high_warning"] == True


class TestPositionRankings:
    """Tests for compute_position_rankings."""

    def test_basic_rankings(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        rankings = compute_position_rankings(profiles)

        assert "FWD_goals_per_90" in rankings
        assert "MID_xg_per_90" in rankings
        assert "DEF_goals_per_90" in rankings
        assert "GK_shots_on_target_per_90" in rankings

        # Check columns
        fwd_rank = rankings["FWD_goals_per_90"]
        assert "player_id" in fwd_rank.columns
        assert "value" in fwd_rank.columns
        assert "rank" in fwd_rank.columns
        assert "percentile" in fwd_rank.columns
        assert "metric" in fwd_rank.columns
        assert "position_band" in fwd_rank.columns

    def test_ranking_order(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        rankings = compute_position_rankings(profiles)

        fwd_rank = rankings["FWD_goals_per_90"]
        # Forward4 has most goals (24) in 2900 min -> should be rank 1
        top = fwd_rank.iloc[0]
        expected_g90 = 24 / 2900 * 90
        assert np.isclose(top["value"], expected_g90, atol=0.01)
        assert top["rank"] == 1

    def test_minutes_filter(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=2000))
        rankings = compute_position_rankings(profiles)

        fwd_rank = rankings["FWD_goals_per_90"]
        # Benchwarmer (150 min) should be excluded
        assert (fwd_rank["player_name"] == "Benchwarmer").sum() == 0


class TestSimilarPlayers:
    """Tests for compute_similar_players."""

    def test_basic_similarity(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        similar = compute_similar_players(profiles, 101)  # Forward1

        assert similar is not None
        assert len(similar) <= 5
        assert "similarity" in similar.columns
        assert "feature_contributions" in similar.columns
        assert (similar["similarity"] >= 0).all()
        assert (similar["similarity"] <= 1).all()
        # Should not include self
        assert (similar["player_id"] == 101).sum() == 0

    def test_self_excluded(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        similar = compute_similar_players(profiles, 101)
        assert (similar["player_id"] == 101).sum() == 0

    def test_insufficient_features(self):
        df = _make_test_player_season()
        # Use very high min_minutes so no other forwards have enough minutes
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=3000))
        # No other forwards have 3000+ minutes (max is Forward4 with 2900)
        similar = compute_similar_players(profiles, 100, ScoutingConfig(min_minutes=3000))
        assert similar is None or len(similar) == 0

    def test_player_not_found(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        similar = compute_similar_players(profiles, 99999)
        assert similar is None

    def test_same_position_band_only(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        similar = compute_similar_players(profiles, 101)  # Forward1 (FWD)
        # Should only include other FWDs
        assert (similar["position_band"] == "FWD").all()

    def test_deterministic(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        sim1 = compute_similar_players(profiles, 101)
        sim2 = compute_similar_players(profiles, 101)
        pd.testing.assert_frame_equal(sim1, sim2)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=[
            "player_id", "player_name", "season_id", "season_name",
            "competition_id", "team_id", "team_name", "position",
            "appearances", "starts", "minutes", "goals", "own_goals",
            "assists", "shots", "shots_on_target", "xg", "npxg",
            "xa", "penalty_goals", "penalty_attempts",
            "source", "source_url", "ingested_at"
        ])
        profiles = build_player_profiles(empty_df)
        assert profiles.empty

    def test_all_draws(self):
        """Test with all draws (edge case for efficiency metrics)."""
        df = pd.DataFrame([{
            "player_id": i, "player_name": f"Player{i}",
            "season_id": 27, "season_name": "2015/2016",
            "competition_id": 2, "team_id": 1, "team_name": "Team1",
            "position": "Center Forward", "appearances": 10, "starts": 10,
            "minutes": 900, "goals": 0, "own_goals": 0, "assists": 0,
            "shots": 10, "shots_on_target": 5, "xg": 2.0, "npxg": 2.0,
            "xa": 1.0, "penalty_goals": 0, "penalty_attempts": 0,
            "source": "test", "source_url": "test", "ingested_at": "2024-01-01"
        } for i in range(4)])
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        assert not profiles.empty
        for _, row in profiles.iterrows():
            assert row["goals_per_shot"] == 0.0
            assert row["goals_per_shot_on_target"] == 0.0


class TestIntegrationWithRealData:
    """Integration tests with real StatsBomb data."""

    def test_real_data_profiles(self):
        """Test with actual StatsBomb 2015/16 data if available."""
        try:
            player_season = pd.read_parquet("data/processed/statsbomb/player_season.parquet")
            fixtures = pd.read_parquet("data/processed/statsbomb/fixtures.parquet")
            fixtures_2015 = fixtures[fixtures["season_id"] == 27].copy()
            player_season_2015 = player_season[player_season["season_id"] == 27].copy()
        except FileNotFoundError:
            pytest.skip("Real data not available")

        config = ScoutingConfig(min_minutes=450)
        profiles = build_player_profiles(player_season_2015, config)

        assert len(profiles) > 100  # Should have many players
        assert "goals_per_90" in profiles.columns
        assert "position_band" in profiles.columns

        # Test rankings
        rankings = compute_position_rankings(profiles)
        assert "FWD_goals_per_90" in rankings

    def test_real_data_similarity(self):
        """Test similarity with real data."""
        try:
            player_season = pd.read_parquet("data/processed/statsbomb/player_season.parquet")
            player_season_2015 = player_season[player_season["season_id"] == 27].copy()
        except FileNotFoundError:
            pytest.skip("Real data not available")

        profiles = build_player_profiles(player_season_2015, ScoutingConfig(min_minutes=450))

        # Pick a forward with good minutes
        fwds = profiles[(profiles["position_band"] == "FWD") & (profiles["minutes"] >= 900)]
        if not fwds.empty:
            target_id = fwds.iloc[0]["player_id"]
            similar = compute_similar_players(
                profiles, target_id, ScoutingConfig(min_minutes=450)
            )
            assert similar is not None
            assert len(similar) <= 5


class TestLeakagePrevention:
    """Tests for temporal leakage prevention."""

    def test_no_future_data_in_profiles(self):
        """Ensure profiles only use data from the correct season."""
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        # All data should be from season 27
        assert (profiles["season_id"] == 27).all()

    def test_deterministic_output(self):
        """Same input should produce identical output."""
        df = _make_test_player_season()
        config = ScoutingConfig(min_minutes=1)
        p1 = build_player_profiles(df, config)
        p2 = build_player_profiles(df, config)
        pd.testing.assert_frame_equal(p1.sort_values("player_id").reset_index(drop=True),
                                       p2.sort_values("player_id").reset_index(drop=True))

    def test_rankings_deterministic(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        r1 = compute_position_rankings(profiles)
        r2 = compute_position_rankings(profiles)
        for k in r1:
            pd.testing.assert_frame_equal(r1[k].sort_values("player_id").reset_index(drop=True),
                                           r2[k].sort_values("player_id").reset_index(drop=True))


class TestPositionBands:
    """Tests for position band handling."""

    def test_all_bands_present(self):
        df = _make_test_player_season()
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        bands = set(profiles["position_band"].unique())
        assert "FWD" in bands
        assert "MID" in bands
        assert "DEF" in bands
        assert "GK" in bands

    def test_unknown_position(self):
        df = pd.DataFrame([{
            "player_id": 999, "player_name": "UnknownPos",
            "season_id": 27, "season_name": "2015/2016",
            "competition_id": 2, "team_id": 1, "team_name": "Team1",
            "position": "UnknownPosition", "appearances": 10, "starts": 10,
            "minutes": 900, "goals": 5, "own_goals": 0, "assists": 2,
            "shots": 20, "shots_on_target": 10, "xg": 4.0, "npxg": 3.5,
            "xa": 2.0, "penalty_goals": 0, "penalty_attempts": 0,
            "source": "test", "source_url": "test", "ingested_at": "2024-01-01"
        }])
        profiles = build_player_profiles(df, ScoutingConfig(min_minutes=1))
        assert profiles.iloc[0]["position_band"] == "UNKNOWN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])