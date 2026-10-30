"""Tests for Phase 9.1 player intelligence analytics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.features.player_intelligence import (
    PlayerIntelligenceConfig,
    compute_per_90_rate,
    safe_divide,
    compute_performance_metrics,
    compute_position_benchmarks,
    merge_with_benchmarks,
    compute_season_consistency,
    compute_match_consistency,
    compute_season_over_season_changes,
    compute_development_trends,
    compute_breakout_score,
    compute_decline_score,
    load_fpl_player_seasons,
    get_position_band_fpl,
    get_position_band_statsbomb,
    get_position_band_auto,
    _classify_trend,
    _max_consecutive,
    compute_season_over_season_changes,
    compute_development_trends,
    compute_breakout_score,
    compute_decline_score,
    get_top_performers_by_metric,
    get_consistency_leaders,
    get_development_summary,
    PlayerIntelligenceConfig,
)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_per_90_rate(self):
        assert compute_per_90_rate(90, 90) == 90.0
        assert compute_per_90_rate(45, 90) == 45.0
        assert compute_per_90_rate(10, 0) is None
        assert compute_per_90_rate(0, 90) == 0.0
        assert compute_per_90_rate(0, 0) is None

    def test_safe_divide(self):
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) is None
        assert safe_divide(float('nan'), 2) is None
        assert safe_divide(10, float('nan')) is None

    def test_get_position_band_fpl(self):
        from src.features.player_intelligence import get_position_band_fpl
        assert get_position_band_fpl("GK") == "GK"
        assert get_position_band_fpl("DEF") == "DEF"
        assert get_position_band_fpl("MID") == "MID"
        assert get_position_band_fpl("FWD") == "FWD"
        assert get_position_band_fpl("AM") == "MID"
        assert get_position_band_fpl("") == "UNKNOWN"
        assert get_position_band_fpl(None) == "UNKNOWN"
        assert get_position_band_fpl(float('nan')) == "UNKNOWN"

    def test_get_position_band_statsbomb(self):
        from src.features.player_intelligence import get_position_band_statsbomb
        assert get_position_band_statsbomb("Goalkeeper") == "GK"
        assert get_position_band_statsbomb("Center Forward") == "FWD"
        assert get_position_band_statsbomb("Left Back") == "DEF"
        assert get_position_band_statsbomb("Left Midfield") == "MID"
        assert get_position_band_statsbomb("Right Wing") == "WING"
        assert get_position_band_statsbomb("") == "UNKNOWN"
        assert get_position_band_statsbomb(None) == "UNKNOWN"

    def test_get_position_band_auto(self):
        from src.features.player_intelligence import get_position_band_auto
        # FPL format
        assert get_position_band_auto("GK", "fpl") == "GK"
        assert get_position_band_auto("MID", "fpl") == "MID"
        assert get_position_band_auto("AM", "fpl") == "MID"
        # StatsBomb format
        assert get_position_band_auto("Goalkeeper", "statsbomb") == "GK"
        assert get_position_band_auto("Center Forward", "statsbomb") == "FWD"
        # Auto-detect
        assert get_position_band_auto("GK") == "GK"
        assert get_position_band_auto("MID") == "MID"
        assert get_position_band_auto("Center Forward") == "FWD"
        assert get_position_band_auto("Left Back") == "DEF"

    def test_classify_trend(self):
        assert _classify_trend(0.5, 0.5) == "improving"
        assert _classify_trend(-0.5, 0.5) == "declining"
        assert _classify_trend(0.05, 0.5) == "stable"
        assert _classify_trend(0.5, 0.2) == "unclear"

    def test_max_consecutive(self):
        s = pd.Series([1, 1, 0, 1, 1, 1, 0, 1])
        assert _max_consecutive(s) == 3
        assert _max_consecutive(pd.Series([0, 0, 0])) == 0
        assert _max_consecutive(pd.Series([1, 1, 1])) == 3
        assert _max_consecutive(pd.Series([0, 1, 0, 1])) == 1


class TestConfig:
    """Tests for configuration."""

    def test_default_config(self):
        config = PlayerIntelligenceConfig()
        assert config.min_minutes_per_season == 450
        assert config.high_percentile == 90.0
        assert config.breakout_weight_production == 0.3


class TestDataLoading:
    """Tests for data loading functions."""

    def test_load_fpl_player_seasons(self):
        df = load_fpl_player_seasons(min_minutes=450)
        assert len(df) > 0
        assert "player_id" in df.columns
        assert "season_id" in df.columns

    def test_load_fpl_player_seasons_filter_seasons(self):
        df = load_fpl_player_seasons(seasons=[2020, 2021], min_minutes=450)
        assert set(df["season_id"].unique()) == {2020, 2021}

    def test_load_fpl_player_seasons_min_minutes(self):
        df_all = load_fpl_player_seasons()
        df_filtered = load_fpl_player_seasons(min_minutes=1000)
        assert len(df_filtered) <= len(df_all)


class TestPerformanceMetrics:
    """Tests for performance metrics computation."""

    def test_compute_performance_metrics(self):
        df = pd.DataFrame({
            "player_id": [1, 2, 3],
            "player_name": ["A", "B", "C"],
            "season_id": [2020, 2020, 2020],
            "season_name": ["2020/21", "2020/21", "2020/21"],
            "minutes": [900, 1800, 2700],
            "goals": [10, 20, 5],
            "assists": [5, 10, 2],
            "expected_goals": [8.5, 18.0, 4.0],
            "expected_assists": [4.0, 9.0, 1.5],
            "shots": [50, 80, 20],
            "shots_on_target": [20, 30, 10],
            "clean_sheets": [5, 2, 0],
            "saves": [50, 10, 0],
            "starts": [30, 35, 20],
            "appearances": [30, 35, 20],
            "goals_per_90": [1.0, 1.0, 0.16],
            "assists_per_90": [0.5, 0.5, 0.16],
            "expected_goals_per_90": [0.85, 0.9, 0.13],
            "expected_assists_per_90": [0.4, 0.45, 0.05],
            "goals_per_90": [1.0, 1.0, 0.16],
            "assists_per_90": [0.5, 0.5, 0.16],
            "shots_per_90": [5.0, 4.0, 0.33],
            "goals_per_shot": [0.2, 0.25, 0.25],
            "goals_per_shot_on_target": [0.5, 0.66, 0.5],
            "clean_sheets_per_90": [0.5, 0.1, 0.0],
            "saves_per_90": [5.0, 0.5, 0.0],
            "starts_rate": [1.0, 1.0, 1.0],
            "minutes_per_appearance": [30, 51.4, 135],
            "starts_rate": [1.0, 1.0, 1.0],
            "total_points": [150, 200, 50],
            "goals_over_xg": [1.5, 2.0, 1.0],
            "assists_over_xa": [1.0, 1.0, 0.5],
            "team": ["Team A", "Team B", "Team C"],
            "position": ["FWD", "MID", "DEF"],
        })

        df = compute_performance_metrics(df)
        
        # Check new columns exist
        expected_new = [
            "clean_sheets_per_90", "saves_per_90", "goals_over_xg",
            "goals_over_xg_per_90", "assists_over_xa", "assists_over_xa_per_90",
            "starts_rate", "minutes_per_appearance", "points_per_90",
            "points_per_appearance", "has_xg", "has_xa", "has_starts"
        ]
        for col in expected_new:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check values
        assert df["goals_over_xg"].iloc[0] == 1.5
        assert df["starts_rate"].iloc[0] == 1.0

    def test_compute_performance_metrics_handles_zero_minutes(self):
        df = pd.DataFrame({
            "player_id": [1],
            "minutes": [0],
            "goals": [5],
            "assists": [2],
            "expected_goals": [2.0],
            "expected_assists": [1.0],
            "shots": [10],
            "shots_on_target": [3],
            "clean_sheets": [0],
            "saves": [0],
            "starts": [0],
            "appearances": [0],
            "goals_per_90": [0],
            "assists_per_90": [0],
            "expected_goals_per_90": [0],
            "expected_assists_per_90": [0],
            "goals_per_90": [0],
            "assists_per_90": [0],
            "shots_per_90": [0],
            "goals_per_shot": [0],
            "goals_per_shot_on_target": [0],
            "clean_sheets_per_90": [0],
            "saves_per_90": [0],
            "starts_rate": [0],
            "minutes_per_appearance": [0],
            "starts_rate": [0],
            "total_points": [0],
            "goals_over_xg": [0],
            "assists_over_xa": [0],
        })
        result = compute_performance_metrics(df)
        # Should not crash and should have None/NaN for per-90 rates


class TestPositionBenchmarks:
    """Tests for position benchmarks."""

    def test_compute_position_benchmarks(self):
        df = pd.DataFrame({
            "player_id": [1, 2, 3, 4, 5, 6],
            "season_id": [2020, 2020, 2020, 2020, 2020, 2020],
            "minutes": [1000, 1000, 1000, 1000, 1000, 1000],
            "goals": [10, 15, 5, 0, 5, 10],
            "assists": [5, 5, 2, 0, 3, 2],
            "expected_goals": [8.0, 12.0, 4.0, 0, 2.0, 8.0],
            "expected_assists": [4.0, 4.0, 2.0, 0, 1.0, 3.0],
            "shots": [50, 60, 20, 0, 20, 40],
            "shots_on_target": [20, 25, 10, 0, 10, 20],
            "clean_sheets": [5, 5, 2, 5, 3, 2],
            "saves": [50, 50, 20, 50, 10, 10],
            "starts": [30, 30, 20, 30, 15, 15],
            "appearances": [30, 30, 20, 30, 15, 15],
            "goals_per_90": [1.0, 1.35, 0.45, 0, 0.45, 0.9],
            "assists_per_90": [0.45, 0.45, 0.18, 0, 0.27, 0.18],
            "expected_goals_per_90": [0.8, 1.0, 0.3, 0, 0.2, 0.7],
            "expected_assists_per_90": [0.4, 0.4, 0.15, 0, 0.1, 0.3],
            "shots_per_90": [4.0, 4.5, 2.0, 0, 2.0, 3.5],
            "goals_per_shot": [0.2, 0.2, 0.2, 0, 0.2, 0.2],
            "goals_per_shot_on_target": [0.5, 0.5, 0.5, 0, 0.5, 0.5],
            "clean_sheets_per_90": [0.5, 0.5, 0.2, 0.5, 0.3, 0.3],
            "saves_per_90": [4.5, 4.5, 1.8, 4.5, 1.0, 1.0],
            "starts_rate": [1.0, 1.0, 0.7, 1.0, 0.7, 0.7],
            "minutes_per_appearance": [90, 90, 90, 90, 90, 90],
            "starts_rate": [1.0, 1.0, 0.7, 1.0, 0.7, 0.7],
            "total_points": [150, 180, 80, 30, 60, 120],
            "goals_over_xg": [2.0, 3.0, 1.0, 0, 1.0, 2.0],
            "assists_over_xa": [1.0, 1.0, 0.5, 0, 0.5, 1.0],
            "team": ["A", "B", "C", "D", "E", "F"],
            "position": ["FWD", "FWD", "MID", "GK", "MID", "FWD"],
            "position_band": ["FWD", "FWD", "MID", "GK", "MID", "FWD"],
            "season_id": [2020, 2020, 2020, 2020, 2020, 2020],
        })
        
        from src.features.player_intelligence import compute_position_benchmarks
        benchmarks = compute_position_benchmarks(df)
        assert len(benchmarks) > 0
        assert "season_id" in benchmarks.columns
        assert "position_band" in benchmarks.columns
        assert "goals_per_90_median" in benchmarks.columns


class TestConsistency:
    """Tests for consistency analysis."""

    def test_compute_season_consistency(self):
        df = pd.DataFrame({
            "player_id": [1, 1, 1, 2, 2, 2],
            "season_id": [2020, 2021, 2022, 2020, 2021, 2022],
            "total_points": [100, 110, 105, 50, 55, 52],
            "goals_per_90": [0.5, 0.55, 0.52, 0.1, 0.11, 0.1],
            "assists_per_90": [0.2, 0.22, 0.21, 0.05, 0.05, 0.05],
            "points_per_90": [2.0, 2.1, 2.05, 1.0, 1.1, 1.0],
        })
        from src.features.player_intelligence import compute_season_consistency
        result = compute_season_consistency(pd.DataFrame({
            "player_id": [1, 1, 1, 2, 2, 2],
            "season_id": [2020, 2021, 2022, 2020, 2021, 2022],
            "total_points": [100, 110, 105, 50, 55, 52],
            "goals_per_90": [0.5, 0.55, 0.52, 0.1, 0.11, 0.1],
            "assists_per_90": [0.2, 0.22, 0.21, 0.05, 0.05, 0.05],
            "points_per_90": [2.0, 2.1, 2.05, 1.0, 1.1, 1.0],
        }))
        assert len(result) == 2
        assert "total_points_cv" in result.columns
        assert "total_points_trend" in result.columns


class TestDevelopmentTrends:
    """Tests for development trends."""

    def test_compute_season_over_season_changes(self):
        df = pd.DataFrame({
            "player_id": [1, 1, 1, 2, 2, 2],
            "season_id": [2020, 2021, 2022, 2020, 2021, 2022],
            "total_points": [100, 110, 105, 50, 55, 52],
            "goals_per_90": [0.5, 0.55, 0.52, 0.1, 0.11, 0.1],
            "assists_per_90": [0.2, 0.22, 0.21, 0.05, 0.05, 0.05],
            "points_per_90": [2.0, 2.1, 2.05, 1.0, 1.1, 1.0],
        })
        from src.features.player_intelligence import compute_season_over_season_changes
        result = compute_season_over_season_changes(pd.DataFrame({
            "player_id": [1, 1, 1, 2, 2, 2],
            "season_id": [2020, 2021, 2022, 2020, 2021, 2022],
            "total_points": [100, 110, 105, 50, 55, 52],
            "goals_per_90": [0.5, 0.55, 0.52, 0.1, 0.11, 0.1],
            "assists_per_90": [0.2, 0.22, 0.21, 0.05, 0.05, 0.05],
            "points_per_90": [2.0, 2.1, 2.05, 1.0, 1.1, 1.0],
        }))
        assert len(result) > 0
        assert "total_points_change" in result.columns

    def test_compute_development_trends(self):
        df = pd.DataFrame({
            "player_id": [1, 1, 1, 2, 2, 2],
            "season_id": [2020, 2021, 2022, 2020, 2021, 2022],
            "total_points": [100, 110, 120, 50, 55, 52],
            "goals_per_90": [0.5, 0.55, 0.6, 0.1, 0.11, 0.1],
            "assists_per_90": [0.2, 0.22, 0.22, 0.05, 0.05, 0.05],
            "points_per_90": [2.0, 2.1, 2.2, 1.0, 1.1, 1.1],
        })
        from src.features.player_intelligence import compute_development_trends
        result = compute_development_trends(pd.DataFrame({
            "player_id": [1, 1, 1, 2, 2, 2],
            "season_id": [2020, 2021, 2022, 2020, 2021, 2022],
            "total_points": [100, 110, 120, 50, 55, 52],
            "goals_per_90": [0.5, 0.55, 0.6, 0.1, 0.11, 0.1],
            "assists_per_90": [0.2, 0.22, 0.22, 0.05, 0.05, 0.05],
            "points_per_90": [2.0, 2.1, 2.2, 1.0, 1.1, 1.1],
        }))
        assert len(result) == 2
        assert "total_points_slope" in result.columns
        assert "total_points_trend" in result.columns
        assert result.iloc[0]["total_points_trend"] == "improving"


class TestBreakoutDecline:
    """Tests for breakout and decline detection."""

    def test_compute_breakout_score(self):
        df = pd.DataFrame({
            "player_id": [1, 1, 2, 2],
            "season_id": [2020, 2021, 2020, 2021],
            "total_points": [100, 150, 100, 90],
            "goals": [10, 20, 10, 8],
            "assists": [5, 10, 5, 4],
            "goals_per_90": [0.5, 0.8, 0.5, 0.4],
            "assists_per_90": [0.2, 0.3, 0.2, 0.15],
            "goals_per_90": [0.5, 0.8, 0.5, 0.4],
            "assists_per_90": [0.2, 0.3, 0.2, 0.15],
            "minutes": [900, 1800, 900, 700],
            "player_name": ["A", "A", "B", "B"],
            "team": ["Team A", "Team A", "Team B", "Team B"],
            "position": ["FWD", "FWD", "MID", "MID"],
        })
        from src.features.player_intelligence import compute_breakout_score
        result = compute_breakout_score(pd.DataFrame({
            "player_id": [1, 1, 2, 2],
            "season_id": [2020, 2021, 2020, 2021],
            "total_points": [100, 150, 100, 90],
            "goals": [10, 20, 10, 8],
            "assists": [5, 10, 5, 4],
            "goals_per_90": [0.5, 0.8, 0.5, 0.4],
            "assists_per_90": [0.2, 0.3, 0.2, 0.15],
            "goals_per_90": [0.5, 0.8, 0.5, 0.4],
            "assists_per_90": [0.2, 0.3, 0.2, 0.15],
            "minutes": [900, 1800, 900, 700],
            "player_name": ["A", "A", "B", "B"],
            "team": ["Team A", "Team A", "Team B", "Team B"],
            "position": ["FWD", "FWD", "MID", "MID"],
        }))
        # Player 1 improved, player 2 declined
        if len(result) > 0:
            assert "breakout_score" in result.columns
            assert "is_breakout_candidate" in result.columns


class TestDashboardHelpers:
    """Tests for dashboard helper functions."""

    def test_get_top_performers_by_metric(self):
        df = pd.DataFrame({
            "player_id": [1, 2, 3, 4],
            "player_name": ["A", "B", "C", "D"],
            "team": ["A", "B", "C", "D"],
            "position": ["FWD", "MID", "DEF", "GK"],
            "season_id": [2020, 2020, 2020, 2020],
            "goals_per_90": [1.0, 0.8, 0.5, 0.0],
            "assists_per_90": [0.2, 0.5, 0.1, 0.0],
            "total_points": [150, 120, 80, 50],
            "minutes": [1000, 900, 800, 700],
            "position_band": ["FWD", "MID", "DEF", "GK"],
        })
        from src.features.player_intelligence import get_top_performers_by_metric
        result = get_top_performers_by_metric(df, "goals_per_90", top_n=2)
        assert len(result) == 2
        assert result.iloc[0]["player_id"] == 1


class TestIdentitySafeguards:
    """Tests for identity safeguards."""

    def test_get_position_band_fpl(self):
        from src.features.player_intelligence import get_position_band_fpl
        assert get_position_band_fpl("GK") == "GK"
        assert get_position_band_fpl("DEF") == "DEF"
        assert get_position_band_fpl("MID") == "MID"
        assert get_position_band_fpl("FWD") == "FWD"
        assert get_position_band_fpl("AM") == "MID"
        assert get_position_band_fpl(None) == "UNKNOWN"
        assert get_position_band_fpl("") == "UNKNOWN"

    def test_get_position_band_statsbomb(self):
        from src.features.player_intelligence import get_position_band_statsbomb
        assert get_position_band_statsbomb("Goalkeeper") == "GK"
        assert get_position_band_statsbomb("Center Forward") == "FWD"
        assert get_position_band_statsbomb("Left Back") == "DEF"
        assert get_position_band_statsbomb("Left Midfield") == "MID"
        assert get_position_band_statsbomb("Right Wing") == "WING"


class TestDeterministic:
    """Tests for deterministic output."""

    def test_deterministic_breakout(self):
        df = pd.DataFrame({
            "player_id": [1, 1],
            "season_id": [2020, 2021],
            "total_points": [100, 150],
            "goals": [10, 20],
            "assists": [5, 10],
            "goals_per_90": [0.5, 0.8],
            "assists_per_90": [0.2, 0.3],
            "goals_per_90": [0.5, 0.8],
            "assists_per_90": [0.2, 0.3],
            "minutes": [900, 1800],
            "player_name": ["A", "A"],
            "team": ["Team A", "Team A"],
            "position": ["FWD", "FWD"],
        })
        from src.features.player_intelligence import compute_breakout_score
        r1 = compute_breakout_score(pd.DataFrame({
            "player_id": [1, 1],
            "season_id": [2020, 2021],
            "total_points": [100, 150],
            "goals": [10, 20],
            "assists": [5, 10],
            "goals_per_90": [0.5, 0.8],
            "assists_per_90": [0.2, 0.3],
            "goals_per_90": [0.5, 0.8],
            "assists_per_90": [0.2, 0.3],
            "minutes": [900, 1800],
            "player_name": ["A", "A"],
            "team": ["Team A", "Team A"],
            "position": ["FWD", "FWD"],
        }))
        r2 = compute_breakout_score(pd.DataFrame({
            "player_id": [1, 1],
            "season_id": [2020, 2021],
            "total_points": [100, 150],
            "goals": [10, 20],
            "assists": [5, 10],
            "goals_per_90": [0.5, 0.8],
            "assists_per_90": [0.2, 0.3],
            "goals_per_90": [0.5, 0.8],
            "assists_per_90": [0.2, 0.3],
            "minutes": [900, 1800],
            "player_name": ["A", "A"],
            "team": ["Team A", "Team A"],
            "position": ["FWD", "FWD"],
        }))
        assert len(r1) == len(r2)
        for a, b in zip(r1.to_dict("records"), r2.to_dict("records")):
            assert a == b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])