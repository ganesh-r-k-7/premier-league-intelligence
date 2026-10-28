"""Tests for dashboard data loading and utility functions."""

from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard.data_loader import (
    load_player_season_statsbomb,
    load_player_match_statsbomb,
    load_fixtures_statsbomb,
    load_fixtures_openfootball,
    load_player_season_fpl,
    load_player_match_fpl,
    load_matches_openfootball,
    load_player_profiles,
    get_season_info,
    get_team_names,
    get_position_bands,
    get_players_for_team,
    get_players_by_position,
    search_players,
    get_comparison_metrics,
    _check_data_exists,
    DataLoadError,
    get_available_sources,
    get_available_seasons,
    get_season_coverage_summary,
    get_fpl_metric_availability,
    get_openfootball_season_quality,
)
from src.dashboard.utils import (
    format_number,
    format_percentile,
    get_player_display_name,
)


class TestDataLoader:
    """Tests for dashboard data loading functions."""

    def test_check_data_exists(self):
        """Test data existence check."""
        exists, missing = _check_data_exists()
        assert exists, f"Missing data files: {missing}"

    def test_load_player_season(self):
        """Test loading player season data (StatsBomb 2015/16)."""
        df = load_player_season_statsbomb()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 550
        assert "player_id" in df.columns
        assert "player_name" in df.columns
        assert "team_name" in df.columns
        assert "position" in df.columns
        assert "minutes" in df.columns
        assert "goals" in df.columns
        assert "assists" in df.columns
        assert "xg" in df.columns

    def test_load_player_match(self):
        """Test loading player match data (StatsBomb 2015/16)."""
        df = load_player_match_statsbomb()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10469
        assert "match_id" in df.columns
        assert "player_id" in df.columns
        assert "minutes" in df.columns

    def test_load_fixtures(self):
        """Test loading fixtures data (StatsBomb 2015/16)."""
        df = load_fixtures_statsbomb()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 380  # 2015/16 Premier League season only
        assert "match_id" in df.columns
        assert "home_team_id" in df.columns
        assert "away_team_id" in df.columns
        assert "home_score" in df.columns
        assert "away_score" in df.columns
        assert "match_week" in df.columns

    def test_load_fixtures_openfootball(self):
        """Test loading OpenFootball fixtures."""
        df = load_fixtures_openfootball("2015/16")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 380
        assert "match_id" in df.columns
        assert "home_score_ft" in df.columns
        assert "away_score_ft" in df.columns

    def test_load_matches_openfootball(self):
        """Test loading all OpenFootball matches."""
        df = load_matches_openfootball()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6080  # 16 seasons * 380
        assert "season_name" in df.columns
        assert df["season_name"].nunique() == 16

    def test_load_player_season_fpl(self):
        """Test loading FPL player season data."""
        df = load_player_season_fpl()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7364  # 10 seasons
        assert "season_name" in df.columns
        assert df["season_name"].nunique() == 10
        assert "expected_goals" in df.columns  # FPL xG

    def test_load_player_match_fpl(self):
        """Test loading FPL player match data."""
        df = load_player_match_fpl()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 253900
        assert "season_name" in df.columns
        assert df["season_name"].nunique() == 10

    def test_load_player_profiles(self):
        """Test loading player profiles with defaults (StatsBomb)."""
        df = load_player_profiles()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 550
        # Check derived columns exist
        assert "goals_per_90" in df.columns
        assert "assists_per_90" in df.columns
        assert "xg_per_90" in df.columns
        assert "position_band" in df.columns
        assert "minutes_flag" in df.columns

    def test_load_player_profiles_custom_config(self):
        """Test loading player profiles with custom config."""
        df = load_player_profiles(min_minutes=900, min_appearances=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 550  # Same rows, just different flags

    def test_load_player_profiles_fpl(self):
        """Test loading FPL player profiles."""
        df = load_player_profiles(source="FPL", season_name="2023/24")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "xg" in df.columns  # FPL expected_goals renamed to xg

    def test_get_season_info(self):
        """Test season info retrieval (StatsBomb default)."""
        info = get_season_info()
        assert info["season_name"] == "2015/16"
        assert info["season_id"] == 2015  # Season ID is the starting year
        assert info["n_matches"] == 380  # 2015/16 Premier League season
        assert info["n_teams"] == 20  # 20 PL teams
        assert info["n_players"] == 550
        assert info["competition"] == "Premier League"
        assert info["source"] == "StatsBomb"

    def test_get_season_info_openfootball(self):
        """Test season info for OpenFootball."""
        info = get_season_info("OpenFootball", "2015/16")
        assert info["season_name"] == "2015/16"
        assert info["source"] == "OpenFootball"
        assert info["n_matches"] == 380

    def test_get_season_info_fpl(self):
        """Test season info for FPL."""
        info = get_season_info("FPL", "2023/24")
        assert info["season_name"] == "2023/24"
        assert info["source"] == "FPL"
        assert info["n_matches"] == 380

    def test_get_team_names(self):
        """Test team names retrieval."""
        teams = get_team_names()
        assert isinstance(teams, list)
        assert len(teams) == 20  # 2015/16 Premier League teams
        assert "Arsenal" in teams
        assert "Chelsea" in teams

    def test_get_team_names_openfootball(self):
        """Test team names for OpenFootball."""
        teams = get_team_names("OpenFootball", "2015/16")
        assert isinstance(teams, list)
        assert len(teams) == 20

    def test_get_position_bands(self):
        """Test position bands retrieval."""
        bands = get_position_bands()
        assert isinstance(bands, list)
        assert "GK" in bands
        assert "DEF" in bands
        assert "MID" in bands
        assert "WING" in bands
        assert "FWD" in bands
        assert "UNKNOWN" not in bands

    def test_get_players_for_team(self):
        """Test filtering players by team."""
        df = get_players_for_team("Arsenal")
        assert isinstance(df, pd.DataFrame)
        assert all(df["team_name"] == "Arsenal")
        assert len(df) > 0

    def test_get_players_by_position(self):
        """Test filtering players by position band."""
        df = get_players_by_position("FWD")
        assert isinstance(df, pd.DataFrame)
        assert all(df["position_band"] == "FWD")
        assert len(df) > 0

    def test_search_players(self):
        """Test player search."""
        # Exact match
        df = search_players("Kane")
        assert len(df) > 0
        assert all("Kane" in name for name in df["player_name"])

        # Case insensitive
        df = search_players("kane")
        assert len(df) > 0

        # Partial match
        df = search_players("Agüero")
        assert len(df) > 0

        # No match
        df = search_players("NonExistentPlayer123")
        assert len(df) == 0

    def test_get_comparison_metrics(self):
        """Test comparison metrics retrieval."""
        metrics = get_comparison_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        for col, name in metrics:
            assert isinstance(col, str)
            assert isinstance(name, str)
            assert len(col) > 0
            assert len(name) > 0

    def test_get_available_sources(self):
        """Test getting available sources."""
        sources = get_available_sources()
        assert "StatsBomb" in sources
        assert "OpenFootball" in sources
        assert "FPL" in sources
        for src in sources.values():
            assert "description" in src

    def test_get_available_seasons(self):
        """Test getting available seasons."""
        all_seasons = get_available_seasons()
        assert "2015/16" in all_seasons
        assert "2023/24" in all_seasons
        assert "2010/11" in all_seasons

        sb_seasons = get_available_seasons("StatsBomb")
        assert "2015/16" in sb_seasons

        fpl_seasons = get_available_seasons("FPL")
        assert "2023/24" in fpl_seasons

    def test_get_season_coverage_summary(self):
        """Test season coverage summary."""
        df = get_season_coverage_summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Season" in df.columns
        assert "Source" in df.columns
        assert "Matches" in df.columns
        assert "xG" in df.columns
        assert "xA" in df.columns

    def test_get_fpl_metric_availability(self):
        """Test FPL metric availability by season."""
        avail = get_fpl_metric_availability()
        assert "2016/17" in avail
        assert "2023/24" in avail
        assert avail["2016/17"]["xG"] is False
        assert avail["2023/24"]["xG"] is True

    def test_get_openfootball_season_quality(self):
        """Test OpenFootball season quality info."""
        quality = get_openfootball_season_quality()
        assert "2015/16" in quality
        assert quality["2015/16"]["complete"] is True
        assert quality["2025/26"]["complete"] is False


class TestDataLoaderEdgeCases:
    """Tests for edge cases in dashboard data loading."""

    def test_load_fixtures_openfootball_invalid_season(self):
        """Test loading OpenFootball fixtures for invalid season."""
        df = load_fixtures_openfootball("1999/00")
        assert len(df) == 0

    def test_load_player_profiles_invalid_source(self):
        """Test loading player profiles with invalid source."""
        with pytest.raises(ValueError):
            load_player_profiles(source="InvalidSource")

    def test_get_season_info_invalid_source(self):
        """Test get_season_info with invalid source."""
        with pytest.raises(ValueError):
            get_season_info("InvalidSource")


class TestUtils:
    """Tests for dashboard utility functions."""

    def test_format_number_int(self):
        """Test formatting integers."""
        assert format_number(42) == "42"
        assert format_number(1000) == "1,000"
        assert format_number(0) == "0"

    def test_format_number_float(self):
        """Test formatting floats."""
        assert format_number(1.5) == "1.5"
        assert format_number(1.23456) == "1.2"
        assert format_number(100.5) == "100.5"
        assert format_number(0.001) == "0.0"
        assert format_number(1234.56) == "1,234.6"

    def test_format_number_nan(self):
        """Test formatting NaN."""
        import numpy as np
        assert format_number(np.nan) == "—"
        assert format_number(float("nan")) == "—"

    def test_format_percentile(self):
        """Test formatting percentiles."""
        assert format_percentile(50.0) == "50th"
        assert format_percentile(99.9) == "100th"
        assert format_percentile(0.1) == "0th"

        import numpy as np
        assert format_percentile(np.nan) == "—"

    def test_get_player_display_name(self):
        """Test player display name formatting."""
        row = pd.Series({
            "player_name": "Harry Kane",
            "team_name": "Tottenham Hotspur",
            "position_band": "FWD",
        })
        name = get_player_display_name(row)
        assert "Harry Kane" in name
        assert "Tottenham Hotspur" in name
        assert "FWD" in name

        # Missing optional fields
        row = pd.Series({"player_name": "Test Player"})
        name = get_player_display_name(row)
        assert name == "Test Player"


class TestDataLoadError:
    """Tests for DataLoadError handling."""

    def test_dataload_error_message(self):
        """Test DataLoadError contains useful message."""
        try:
            raise DataLoadError("Test error message")
        except DataLoadError as e:
            assert "Test error message" in str(e)


class TestDashboardEdgeCases:
    """Tests for edge cases in dashboard functions."""

    def test_empty_search_results(self):
        """Test search with no results returns empty DataFrame with correct columns."""
        df = search_players("XYZNonExistent")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "player_name" in df.columns

    def test_empty_team_filter(self):
        """Test filtering for non-existent team."""
        df = get_players_for_team("NonExistentTeam")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_empty_position_filter(self):
        """Test filtering for non-existent position."""
        df = get_players_by_position("INVALID")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestDataLoaderEdgeCases:
    """Tests for edge cases in dashboard data loading."""

    def test_load_fixtures_openfootball_invalid_season(self):
        """Test loading OpenFootball fixtures for invalid season."""
        df = load_fixtures_openfootball("1999/00")
        assert len(df) == 0

    def test_load_player_profiles_invalid_source(self):
        """Test loading player profiles with invalid source."""
        with pytest.raises(ValueError):
            load_player_profiles(source="InvalidSource")

    def test_get_season_info_invalid_source(self):
        """Test get_season_info with invalid source."""
        with pytest.raises(ValueError):
            get_season_info("InvalidSource")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])