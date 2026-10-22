"""Tests for player feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.player_features import (
    FeatureConfig,
    compute_features_for_cutoff,
    compute_player_features,
    get_position_band,
    _prepare_history_with_matchweek,
)


def _make_test_player_match() -> pd.DataFrame:
    """Create a test player_match DataFrame WITHOUT match_week (like production data)."""
    rows = []
    match_id = 100

    # 3 matchweeks, 2 teams, 3 players per team
    for mw in [1, 2, 3]:
        for team_id, team_name in [(1, "TeamA"), (2, "TeamB")]:
            # Player 1: starter, scores in mw 2
            player_id = team_id * 10 + 1
            rows.append({
                "match_id": match_id,
                "player_id": player_id,
                "player_name": f"Player{player_id}",
                "team_id": team_id,
                "team_name": team_name,
                "position": "Center Forward",
                "started": True,
                "minutes": 90,
                "goals": 1 if mw == 2 else 0,
                "assists": 0,
                "shots": 2,
                "shots_on_target": 1,
                "xg": 0.5,
                "npxg": 0.5,
                "xa": 0.0,
                "penalty_goals": 0,
                "penalty_attempts": 0,
            })
            # Player 2: starter, assists in mw 3
            player_id = team_id * 10 + 2
            rows.append({
                "match_id": match_id,
                "player_id": player_id,
                "player_name": f"Player{player_id}",
                "team_id": team_id,
                "team_name": team_name,
                "position": "Center Attacking Midfield",
                "started": True,
                "minutes": 90,
                "goals": 0,
                "assists": 1 if mw == 3 else 0,
                "shots": 1,
                "shots_on_target": 0,
                "xg": 0.1,
                "npxg": 0.1,
                "xa": 0.3,
                "penalty_goals": 0,
                "penalty_attempts": 0,
            })
            # Player 3: unused sub
            player_id = team_id * 10 + 3
            rows.append({
                "match_id": match_id,
                "player_id": player_id,
                "player_name": f"Player{player_id}",
                "team_id": team_id,
                "team_name": team_name,
                "position": "Left Back",
                "started": False,
                "minutes": 0,
                "goals": 0,
                "assists": 0,
                "shots": 0,
                "shots_on_target": 0,
                "xg": 0.0,
                "npxg": 0.0,
                "xa": 0.0,
                "penalty_goals": 0,
                "penalty_attempts": 0,
            })
            match_id += 1

    return pd.DataFrame(rows)


def _make_test_fixtures() -> pd.DataFrame:
    """Create test fixtures DataFrame with proper weekly spacing."""
    rows = []
    match_id = 100
    base_date = pd.Timestamp("2015-08-10")
    for mw in [1, 2, 3]:
        # TeamA home vs TeamB away (Saturday)
        match_date = base_date + pd.Timedelta(days=(mw-1)*7)
        rows.append({
            "match_id": match_id,
            "match_week": mw,
            "match_date": match_date,
            "home_team_id": 1,
            "home_team_name": "TeamA",
            "away_team_id": 2,
            "away_team_name": "TeamB",
            "home_score": 1 if mw == 2 else 0,
            "away_score": 0,
        })
        match_id += 1
        # TeamB home vs TeamA away (Sunday)
        rows.append({
            "match_id": match_id,
            "match_week": mw,
            "match_date": match_date + pd.Timedelta(days=1),
            "home_team_id": 2,
            "home_team_name": "TeamB",
            "away_team_id": 1,
            "away_team_name": "TeamA",
            "home_score": 0,
            "away_score": 1 if mw == 2 else 0,
        })
        match_id += 1
    return pd.DataFrame(rows)


class TestPositionBands:
    """Tests for position band mapping."""

    def test_known_positions(self):
        """Known positions map to correct bands."""
        assert get_position_band("Goalkeeper") == "GK"
        assert get_position_band("Left Center Back") == "DEF"
        assert get_position_band("Center Forward") == "FWD"
        assert get_position_band("Left Wing") == "WING"
        assert get_position_band("Center Attacking Midfield") == "MID"

    def test_unknown_position(self):
        """Unknown positions map to UNKNOWN."""
        assert get_position_band("Fake Position") == "UNKNOWN"
        assert get_position_band(None) == "UNKNOWN"


class TestPrepareHistory:
    """Tests for history preparation."""

    def test_history_excludes_target_matchweek(self):
        """History only includes match_week < cutoff_week."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()

        history = _prepare_history_with_matchweek(pm, fixtures, 3)
        assert set(history["match_week"].unique()) == {1, 2}

    def test_empty_history_at_cutoff_1(self):
        """Cutoff week 1 has empty history."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()

        history = _prepare_history_with_matchweek(pm, fixtures, 1)
        assert len(history) == 0

    def test_matchweek_merged_correctly(self):
        """match_week is correctly merged from fixtures."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()

        history = _prepare_history_with_matchweek(pm, fixtures, 2)
        assert "match_week" in history.columns
        assert "match_date" in history.columns
        assert "home_team_id" in history.columns
        assert "away_team_id" in history.columns
        assert "home_score" in history.columns
        assert "away_score" in history.columns


class TestPlayerFeatures:
    """Tests for compute_player_features."""

    def test_correct_goals_per_90_with_sufficient_history(self):
        """Goals per 90 computed correctly from player history."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=2, recent_window=3)

        # Player 11 (TeamA, CF) at cutoff 3: history mw 1,2
        # mw1: 0 goals, 90 min; mw2: 1 goal, 90 min
        # Total: 1 goal, 180 min -> 0.5 per 90
        target_match = pd.Series({
            "match_id": 104,
            "match_week": 3,
            "match_date": pd.Timestamp("2015-08-25"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Center Forward",
        })

        features = compute_player_features(pm, fixtures, target_match, 11, config)

        assert np.isclose(features["player_goals_per_90"], 0.5)
        assert np.isclose(features["player_assists_per_90"], 0.0)
        assert "player_history" in features["fallback_info"]

    def test_correct_assists_per_90_with_sufficient_history(self):
        """Assists per 90 computed correctly from player history."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=2, recent_window=3)

        # Player 12 (TeamA, CAM) at cutoff 3: history mw 1,2 (0 assists each)
        target_match = pd.Series({
            "match_id": 104,
            "match_week": 3,
            "match_date": pd.Timestamp("2015-08-25"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Center Attacking Midfield",
        })

        features = compute_player_features(pm, fixtures, target_match, 12, config)

        assert np.isclose(features["player_assists_per_90"], 0.0)

    def test_recent_form_calculation_and_ordering(self):
        """Recent form uses most recent matches in correct order."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=3, recent_window=2)

        # Player 11 at cutoff 3: history mw 1,2
        # Recent 2: mw1 (0 goals), mw2 (1 goal) -> mean 0.5
        target_match = pd.Series({
            "match_id": 104,
            "match_week": 3,
            "match_date": pd.Timestamp("2015-08-25"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Center Forward",
        })

        features = compute_player_features(pm, fixtures, target_match, 11, config)

        # Recent 2 matches: 0 and 1 -> mean 0.5
        assert np.isclose(features["player_recent_goals_mean_3"], 0.5)
        assert "player_last_2" in features["fallback_info"]

    def test_sparse_history_fallback(self):
        """Player with < min_history falls back to position band rate."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=3, recent_window=3)

        # Player 11 at cutoff 2: only mw1 in history (< min_history=3)
        target_match = pd.Series({
            "match_id": 102,
            "match_week": 2,
            "match_date": pd.Timestamp("2015-08-20"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Center Forward",
        })

        features = compute_player_features(pm, fixtures, target_match, 11, config)

        # Should fall back to position band (FWD) league median
        assert "position_band_FWD" in features["fallback_info"] or "global" in features["fallback_info"]
        assert features["player_goals_per_90"] >= 0  # Non-negative

    def test_empty_history_at_earliest_cutoff(self):
        """Cutoff week 1 returns global/position fallbacks."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=3, recent_window=3)

        target_match = pd.Series({
            "match_id": 100,
            "match_week": 1,
            "match_date": pd.Timestamp("2015-08-15"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Center Forward",
        })

        features = compute_player_features(pm, fixtures, target_match, 11, config)

        # No history -> global fallback
        assert features["player_history_matches"] == 0
        assert features["player_goals_per_90"] >= 0
        assert features["player_assists_per_90"] >= 0
        assert features["player_recent_goals_mean_3"] >= 0
        assert "global" in features["fallback_info"]

    def test_zero_minute_player_handling(self):
        """Zero-minute player gets fallback rates, not division by zero."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=2, recent_window=3)

        # Player 13 (TeamA, LB) has 0 minutes in all matches
        target_match = pd.Series({
            "match_id": 104,
            "match_week": 3,
            "match_date": pd.Timestamp("2015-08-25"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Left Back",
        })

        features = compute_player_features(pm, fixtures, target_match, 13, config)

        # Should not crash, should use fallback
        assert features["player_goals_per_90"] >= 0
        assert features["player_assists_per_90"] >= 0
        assert features["player_minutes_per_match"] == 0.0

    def test_missing_position_handling(self):
        """Missing position falls back to global."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=3, recent_window=3)

        target_match = pd.Series({
            "match_id": 104,
            "match_week": 3,
            "match_date": pd.Timestamp("2015-08-25"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": None,  # Missing position
        })

        features = compute_player_features(pm, fixtures, target_match, 999, config)

        assert "UNKNOWN" in features["fallback_info"] or "global" in features["fallback_info"]
        assert features["position_goals_per_90_league"] >= 0

    def test_team_and_opponent_stats_use_prior_fixtures_only(self):
        """Team and opponent stats only use fixtures before cutoff."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=2, recent_window=3)

        # At cutoff 2, only mw1 fixtures available
        # TeamA scored 0 in mw1 (away at TeamB)
        target_match = pd.Series({
            "match_id": 102,
            "match_week": 2,
            "match_date": pd.Timestamp("2015-08-20"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Center Forward",
        })

        features = compute_player_features(pm, fixtures, target_match, 11, config)

        # TeamA goals per match from mw1 only: 0 (away at TeamB, lost 0-1)
        assert np.isclose(features["team_goals_per_match"], 0.0)
        # Opponent (TeamB) conceded 0 in mw1 (home vs TeamA, won 1-0)
        assert np.isclose(features["opponent_goals_conceded_per_match"], 0.0)

    def test_position_stats_exclude_future_observations(self):
        """Position-level stats only use historical data."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=2, recent_window=3)

        # At cutoff 2, only mw1 data for position FWD
        target_match = pd.Series({
            "match_id": 102,
            "match_week": 2,
            "match_date": pd.Timestamp("2015-08-20"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Center Forward",
        })

        features = compute_player_features(pm, fixtures, target_match, 11, config)

        # Position FWD goals per 90 from mw1 only (both teams' CFs scored 0)
        assert features["position_goals_per_90_league"] >= 0

    def test_rest_days_use_only_prior_appearances(self):
        """Rest days calculated from most recent prior appearance."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=2, recent_window=3)

        # Player 11 at cutoff 3: last appearance mw2 (2015-08-17), target mw3 (2015-08-24)
        # match_id 104 = TeamA home in matchweek 3 (2015-08-24)
        target_match = pd.Series({
            "match_id": 104,
            "match_week": 3,
            "match_date": pd.Timestamp("2015-08-24"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Center Forward",
        })

        features = compute_player_features(pm, fixtures, target_match, 11, config)

        # 7 days rest (weekly matches)
        assert np.isclose(features["days_rest"], 7.0)

    def test_target_match_stats_do_not_affect_features(self):
        """Target match goals/assists/minutes don't leak into features."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=2, recent_window=3)

        # Player 11 at cutoff 3: target match (mw3) has goals=0 in data
        # But history (mw1,2) has 1 goal total
        target_match = pd.Series({
            "match_id": 104,
            "match_week": 3,
            "match_date": pd.Timestamp("2015-08-25"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Center Forward",
        })

        features = compute_player_features(pm, fixtures, target_match, 11, config)

        # Features should be based on history (1 goal in 180 min = 0.5 per 90)
        # NOT on target match (0 goals)
        assert np.isclose(features["player_goals_per_90"], 0.5)
        # Recent form from mw1,2: 0 and 1 -> 0.5
        assert np.isclose(features["player_recent_goals_mean_3"], 0.5)


class TestLeakage:
    """Leakage tests: future data must not affect earlier-cutoff features."""

    def test_future_data_injection_no_leakage(self):
        """Appending future matchweek data does not change earlier-cutoff features."""
        pm_base = _make_test_player_match()
        fixtures_base = _make_test_fixtures()
        config = FeatureConfig(min_history=2, recent_window=3)

        # Features at cutoff 2 using base data
        target_match = pd.Series({
            "match_id": 102,
            "match_week": 2,
            "match_date": pd.Timestamp("2015-08-20"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Center Forward",
        })

        features_original = compute_player_features(pm_base, fixtures_base, target_match, 11, config)

        # Inject fake future data for matchweek 3: Player 11 scores 5 goals
        # Note: player_match does NOT have match_week - it's merged from fixtures
        future_row = pd.DataFrame([{
            "match_id": 999,
            "player_id": 11,
            "player_name": "Player11",
            "team_id": 1,
            "team_name": "TeamA",
            "position": "Center Forward",
            "started": True,
            "minutes": 90,
            "goals": 5,
            "assists": 3,
            "shots": 10,
            "shots_on_target": 5,
            "xg": 3.0,
            "npxg": 3.0,
            "xa": 1.0,
            "penalty_goals": 0,
            "penalty_attempts": 0,
        }])

        pm_with_future = pd.concat([pm_base, future_row], ignore_index=True)

        # Also need to add the fixture for the future match
        future_fixture = pd.DataFrame([{
            "match_id": 999,
            "match_week": 3,
            "match_date": pd.Timestamp("2015-08-31"),
            "home_team_id": 1,
            "home_team_name": "TeamA",
            "away_team_id": 2,
            "away_team_name": "TeamB",
            "home_score": 5,
            "away_score": 0,
        }])
        fixtures_with_future = pd.concat([fixtures_base, future_fixture], ignore_index=True)

        features_with_future = compute_player_features(pm_with_future, fixtures_with_future, target_match, 11, config)

        # All features must be identical
        for key in features_original:
            if key not in ["fallback_info"]:  # fallback_info may differ in string representation
                orig = features_original[key]
                fut = features_with_future[key]
                if isinstance(orig, float) and isinstance(fut, float):
                    assert np.isclose(orig, fut), f"Leakage in {key}: {orig} vs {fut}"
                else:
                    assert orig == fut, f"Leakage in {key}: {orig} vs {fut}"

    def test_future_fixture_injection_no_leakage(self):
        """Adding future fixtures does not affect team/opponent stats at earlier cutoff."""
        pm_base = _make_test_player_match()
        fixtures_base = _make_test_fixtures()
        config = FeatureConfig(min_history=2, recent_window=3)

        target_match = pd.Series({
            "match_id": 102,
            "match_week": 2,
            "match_date": pd.Timestamp("2015-08-20"),
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
            "position": "Center Forward",
        })

        features_original = compute_player_features(pm_base, fixtures_base, target_match, 11, config)

        # Inject future fixture (matchweek 3) with high scores
        future_fixture = pd.DataFrame([{
            "match_id": 999,
            "match_week": 3,
            "match_date": pd.Timestamp("2015-08-25"),
            "home_team_id": 1,
            "home_team_name": "TeamA",
            "away_team_id": 2,
            "away_team_name": "TeamB",
            "home_score": 5,
            "away_score": 0,
        }])

        fixtures_with_future = pd.concat([fixtures_base, future_fixture], ignore_index=True)

        features_with_future = compute_player_features(pm_base, fixtures_with_future, target_match, 11, config)

        # Team and opponent stats must not change
        assert np.isclose(features_original["team_goals_per_match"], features_with_future["team_goals_per_match"])
        assert np.isclose(features_original["opponent_goals_conceded_per_match"], features_with_future["opponent_goals_conceded_per_match"])


class TestComputeFeaturesForCutoff:
    """Tests for batch feature computation."""

    def test_batch_computation_returns_all_players(self):
        """Batch computation returns features for all players in target matchweek."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig(min_history=2, recent_window=3)

        features_df = compute_features_for_cutoff(pm, fixtures, 2, config)

        # 2 teams * 3 players = 6 players in matchweek 2
        assert len(features_df) == 6
        assert set(features_df["player_id"]) == {11, 12, 13, 21, 22, 23}
        assert all(c in features_df.columns for c in [
            "player_goals_per_90", "player_assists_per_90",
            "player_recent_goals_mean_3", "player_recent_assists_mean_3",
            "team_goals_per_match", "opponent_goals_conceded_per_match",
            "is_home", "days_rest"
        ])

    def test_empty_cutoff_returns_empty_dataframe(self):
        """Cutoff with no matches returns empty DataFrame."""
        pm = _make_test_player_match()
        fixtures = _make_test_fixtures()
        config = FeatureConfig()

        features_df = compute_features_for_cutoff(pm, fixtures, 99, config)
        assert len(features_df) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])