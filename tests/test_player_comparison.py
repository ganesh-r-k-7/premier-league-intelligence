"""Tests for player comparison and scouting priority modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.player_scouting import (
    ScoutingConfig,
    build_player_profiles,
    compute_position_rankings,
    compute_similar_players,
)
from src.features.player_comparison import (
    PlayerComparison,
    ComparisonConfig,
    ComparisonResult,
)
from src.features.scouting_priority import (
    PriorityConfig,
    calculate_scouting_priority,
    _compute_priority_scores,
    _validate_weights,
)


def _make_test_data() -> pd.DataFrame:
    """Create test player season data with varied positions and stats."""
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
            "team_name": f"Team{i+1}",
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
            "team_name": f"Team{i+1}",
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
            "team_name": f"Team{i+1}",
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
            "team_name": f"Team{i+1}",
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


class TestComparisonConfig:
    """Tests for ComparisonConfig."""

    def test_defaults(self):
        config = ComparisonConfig()
        assert config.min_minutes == 450
        assert config.percentile_scope == "position_band"
        assert config.handle_missing == "flag"

    def test_custom_config(self):
        config = ComparisonConfig(
            min_minutes=900,
            percentile_scope="global",
            handle_missing="exclude",
        )
        assert config.min_minutes == 900
        assert config.percentile_scope == "global"
        assert config.handle_missing == "exclude"


class TestPlayerComparison:
    """Tests for PlayerComparison class."""

    @pytest.fixture
    def profiles(self):
        df = _make_test_data()
        config = ScoutingConfig(min_minutes=1)
        profiles = build_player_profiles(df, config=ScoutingConfig(min_minutes=1))
        return profiles

    def test_compare_two_players(self, profiles):
        """Test comparing two players."""
        config = ComparisonConfig(min_minutes=1)
        comp = PlayerComparison(profiles, config)

        result = comp.compare_players([100, 101])

        assert isinstance(result, ComparisonResult)
        assert not result.comparison_df.empty
        assert len(result.comparison_df) == 2
        assert "summary_df" in result.__dict__

    def test_compare_three_players(self, profiles):
        """Test comparing three players."""
        config = ComparisonConfig(min_minutes=1)
        comp = PlayerComparison(profiles, config)

        result = comp.compare_players([100, 101, 102])

        assert len(result.comparison_df) == 3

    def test_percentile_calculation(self, profiles):
        """Test percentile calculation within position band."""
        config = ComparisonConfig(min_minutes=1)
        comp = PlayerComparison(profiles, config)

        result = comp.compare_players([100, 101])

        # Check percentile columns exist
        assert "goals_per_90_percentile" in result.comparison_df.columns
        assert "assists_per_90_percentile" in result.comparison_df.columns

    def test_global_percentile_scope(self, profiles):
        """Test global percentile scope."""
        config = ComparisonConfig(
            min_minutes=1,
            include_percentiles=True,
            percentile_scope="global",
        )
        comp = PlayerComparison(profiles, config)

        result = comp.compare_players([100, 101])

        # Should have percentile columns
        assert "goals_per_90_percentile" in result.comparison_df.columns

    def test_difference_calculation(self, profiles):
        """Test difference calculation for 2-player comparison."""
        config = ComparisonConfig(min_minutes=1)
        comp = PlayerComparison(profiles, config)

        result = comp.compare_players([100, 101])

        # Check diff columns
        assert "goals_per_90_diff" in result.comparison_df.columns
        assert "goals_per_90_pct_diff" in result.comparison_df.columns

        diff = result.comparison_df["goals_per_90_diff"].values
        assert len(diff) == 2

    def test_missing_values_flag(self, profiles):
        """Test missing value handling with flag strategy."""
        config = ComparisonConfig(min_minutes=1, handle_missing="flag")
        comp = PlayerComparison(profiles, config)

        result = comp.compare_players([100, 101])

        # Should have raw values with NaN where missing
        assert "goals_per_90_raw" in result.comparison_df.columns

    def test_missing_values_exclude(self, profiles):
        """Test missing value handling with exclude strategy."""
        config = ComparisonConfig(min_minutes=1, handle_missing="exclude")
        comp = PlayerComparison(profiles, config)

        result = comp.compare_players([100, 101])
        # Should not have NaN values
        assert not result.comparison_df.isnull().any().any()

    def test_invalid_player_id(self, profiles):
        """Test error on invalid player ID."""
        config = ComparisonConfig(min_minutes=1)
        comp = PlayerComparison(profiles, config)

        with pytest.raises(ValueError, match="Player IDs not found"):
            comp.compare_players([99999, 101])

    def test_insufficient_players(self, profiles):
        """Test error with insufficient players."""
        config = ComparisonConfig(min_minutes=1)
        comp = PlayerComparison(profiles, config)

        with pytest.raises(ValueError, match="At least 2 player_ids"):
            comp.compare_players([100])

    def test_invalid_metric(self, profiles):
        """Test error on invalid metric."""
        config = ComparisonConfig(min_minutes=1)
        comp = PlayerComparison(profiles, config)

        with pytest.raises(ValueError, match="None of the requested metrics"):
            comp.compare_players([100, 101], metrics=["nonexistent_metric"])

    def test_deterministic(self, profiles):
        """Same input should produce same output."""
        config = ComparisonConfig(min_minutes=1)
        comp1 = PlayerComparison(profiles, config)
        comp2 = PlayerComparison(profiles, config)

        result1 = comp1.compare_players([100, 101])
        result2 = comp2.compare_players([100, 101])

        pd.testing.assert_frame_equal(
            result1.comparison_df.sort_values("player_id").reset_index(drop=True),
            result2.comparison_df.sort_values("player_id").reset_index(drop=True)
        )

    def test_get_percentile_ranks(self, profiles):
        """Test get_percentile_ranks method."""
        config = ComparisonConfig(min_minutes=1)
        comp = PlayerComparison(profiles, config)

        ranks = comp.get_percentile_ranks([100, 101], ["goals_per_90", "xg_per_90"])

        assert 100 in ranks
        assert 101 in ranks
        assert "goals_per_90" in ranks[100]
        assert "xg_per_90" in ranks[100]


class TestComparisonConfig:
    """Tests for ComparisonConfig."""

    def test_defaults(self):
        config = ComparisonConfig()
        assert config.min_minutes == 450
        assert config.percentile_scope == "position_band"
        assert config.handle_missing == "flag"

    def test_custom_config(self):
        config = ComparisonConfig(
            min_minutes=900,
            percentile_scope="global",
            handle_missing="exclude",
        )
        assert config.min_minutes == 900
        assert config.percentile_scope == "global"
        assert config.handle_missing == "exclude"


class TestPriorityConfig:
    """Tests for PriorityConfig."""

    def test_defaults(self):
        config = PriorityConfig()
        assert config.min_minutes == 450
        assert config.target_player_id is None
        assert config.weights["production_vs_peers"] == 0.40

    def test_custom_weights(self):
        config = PriorityConfig(
            weights={
                "production_vs_peers": 0.5,
                "xg_vs_actual": 0.3,
                "playing_time_reliability": 0.2,
                "similarity_to_target": 0.0,
            },
            target_player_id=100,
        )
        assert config.target_player_id == 100
        # Weights should be normalized
        total = sum(config.weights.values())
        assert abs(total - 1.0) < 1e-9


class TestValidateWeights:
    """Tests for _validate_weights function."""

    def test_valid_weights(self):
        weights = {"a": 0.5, "b": 0.3, "c": 0.2}
        result = _validate_weights(weights)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_zero_weights(self):
        with pytest.raises(ValueError, match="At least one weight must be positive"):
            _validate_weights({"a": 0, "b": 0})

    def test_negative_weights(self):
        with pytest.raises(ValueError, match="At least one weight must be positive"):
            _validate_weights({"a": -1, "b": 1})

    def test_normalization(self):
        weights = {"a": 2, "b": 2}
        result = _validate_weights(weights)
        assert result["a"] == 0.5
        assert result["b"] == 0.5


class TestScoutingPriority:
    """Tests for scouting priority calculation."""

    @pytest.fixture
    def profiles(self):
        df = _make_test_data()
        config = ScoutingConfig(min_minutes=1)
        return build_player_profiles(df, config=ScoutingConfig(min_minutes=1))

    def test_calculate_priority(self, profiles):
        """Test basic priority calculation."""
        config = PriorityConfig(
            weights={
                "production_vs_peers": 0.5,
                "xg_vs_actual": 0.3,
                "playing_time_reliability": 0.2,
                "similarity_to_target": 0.0,
            },
            min_minutes=1,
        )
        result = calculate_scouting_priority(profiles, config)

        assert not result.empty
        assert "priority_score" in result.columns
        assert "coverage" in result.columns
        assert "priority_rank" in result.columns
        assert len(result) > 0

    def test_priority_weights_normalization(self, profiles):
        """Test that weights are normalized correctly."""
        config = PriorityConfig(
            weights={
                "production_vs_peers": 2.0,
                "xg_vs_actual": 2.0,
                "playing_time_reliability": 1.0,
            },
            min_minutes=1,
        )
        result = calculate_scouting_priority(profiles, config)

        assert not result.empty

    def test_zero_weight_excluded(self, profiles):
        """Components with zero weight should be excluded."""
        config = PriorityConfig(
            weights={
                "production_vs_peers": 1.0,
                "xg_vs_actual": 0.0,
            },
            min_minutes=1,
        )
        result = calculate_scouting_priority(profiles, config)

        assert not result.empty

    def test_zero_total_weight_raises(self):
        """All zero weights should raise ValueError."""
        config = PriorityConfig(weights={"a": 0, "b": 0})
        with pytest.raises(ValueError, match="At least one weight must be positive"):
            _validate_weights(config.weights)

    def test_negative_weight_ignored(self, profiles):
        """Negative weights should be treated as zero."""
        config = PriorityConfig(
            weights={"production_vs_peers": 1.0, "xg_vs_actual": -0.5},
            min_minutes=1,
        )
        result = calculate_scouting_priority(profiles, config)
        assert not result.empty

    def test_min_minutes_filter(self, profiles):
        """Players below min_minutes should be excluded."""
        config = PriorityConfig(min_minutes=2000)
        result = calculate_scouting_priority(profiles, config)

        assert (result["minutes"] >= 2000).all()

    def test_zero_coverage_excluded(self, profiles):
        """Players with zero coverage should be excluded."""
        config = PriorityConfig(
            weights={"production_vs_peers": 1.0},
            min_minutes=1,
        )
        result = calculate_scouting_priority(profiles)

        assert (result["coverage"] > 0).all()

    def test_coverage_calculation(self, profiles):
        """Test coverage calculation."""
        config = PriorityConfig(
            weights={"production_vs_peers": 0.5, "xg_vs_actual": 0.5},
            min_minutes=1,
        )
        result = calculate_scouting_priority(profiles)

        assert "coverage" in result.columns
        assert (result["coverage"] >= 0).all()
        assert (result["coverage"] <= 100).all()

    def test_priority_score_scaling(self, profiles):
        """Priority score should be scaled to 0-100."""
        config = PriorityConfig(min_minutes=1)
        result = calculate_scouting_priority(profiles)

        assert (result["priority_score"] >= 0).all()
        assert (result["priority_score"] <= 100).all()

    def test_priority_ranking(self, profiles):
        """Results should be ranked by priority score."""
        config = PriorityConfig(min_minutes=1)
        result = calculate_scouting_priority(profiles)

        scores = result["priority_score"].values
        assert (np.diff(scores) <= 0).all()  # Non-increasing

    def test_priority_rank_column(self, profiles):
        """Results should have priority_rank column."""
        config = PriorityConfig(min_minutes=1)
        result = calculate_scouting_priority(profiles)

        assert "priority_rank" in result.columns
        assert result["priority_rank"].iloc[0] == 1
        assert result["priority_rank"].iloc[-1] == len(result)

    def test_target_player_similarity(self, profiles):
        """Test similarity to target player component."""
        config = PriorityConfig(
            weights={
                "production_vs_peers": 0.5,
                "similarity_to_target": 0.5,
            },
            target_player_id=100,  # Forward1
            min_minutes=1,
        )
        result = calculate_scouting_priority(profiles, config)

        assert "similarity_to_target" in result.columns

    def test_similarity_optional(self, profiles):
        """Similarity should be optional and only used when target specified."""
        config = PriorityConfig(
            weights={
                "production_vs_peers": 1.0,
                "similarity_to_target": 1.0,
            },
            min_minutes=1,
            # No target_player_id
        )
        result = calculate_scouting_priority(profiles, config)

        # Should work without similarity component
        assert not result.empty

    def test_deterministic(self, profiles):
        """Same input should produce identical output."""
        config = PriorityConfig(min_minutes=1)

        result1 = calculate_scouting_priority(profiles, config)
        result2 = calculate_scouting_priority(profiles, config)

        pd.testing.assert_frame_equal(
            result1.sort_values("player_id").reset_index(drop=True),
            result2.sort_values("player_id").reset_index(drop=True)
        )

    def test_coverage_calculation(self, profiles):
        """Test coverage calculation."""
        config = PriorityConfig(
            weights={"production_vs_peers": 0.5, "xg_vs_actual": 0.5},
            min_minutes=1,
        )
        result = calculate_scouting_priority(profiles, config)

        assert "coverage" in result.columns
        assert (result["coverage"] >= 0).all()
        assert (result["coverage"] <= 100).all()

    def test_zero_coverage_excluded(self, profiles):
        """Players with zero coverage should be excluded."""
        config = PriorityConfig(
            weights={"production_vs_peers": 1.0},
            min_minutes=99999,  # Higher than any player
        )
        result = calculate_scouting_priority(profiles, config)

        # With min_minutes too high, no players should qualify
        assert len(result) == 0


class TestValidateWeights:
    """Tests for _validate_weights function."""

    def test_valid_weights(self):
        weights = {"a": 0.5, "b": 0.3, "c": 0.2}
        result = _validate_weights(weights)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_zero_weights(self):
        with pytest.raises(ValueError, match="At least one weight must be positive"):
            _validate_weights({"a": 0, "b": 0})

    def test_negative_weights(self):
        """Negative weights should be treated as zero."""
        weights = {"a": -1, "b": 1}
        result = _validate_weights(weights)
        assert result["a"] == 0.0
        assert result["b"] == 1.0

    def test_normalization(self):
        weights = {"a": 2, "b": 2}
        result = _validate_weights(weights)
        assert result["a"] == 0.5
        assert result["b"] == 0.5


def _make_test_data() -> pd.DataFrame:
    """Create test player season data with varied positions and stats."""
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
            "team_name": f"Team{i+1}",
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

        rows.append({
            "player_id": player_id,
            "player_name": f"Midfielder{i+1}",
            "season_id": 27,
            "season_name": "2015/2016",
            "competition_id": 2,
            "team_id": 1 + (i % 5),
            "team_name": f"Team{i+1}",
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
            "team_name": f"Team{i+1}",
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
            "team_name": f"Team{i+1}",
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])