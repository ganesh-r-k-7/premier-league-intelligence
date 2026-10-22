"""Unit tests for baseline implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.baselines import (
    BaselineConfig,
    compute_global_mean,
    compute_minutes_adjusted,
    compute_player_mean,
    compute_recent_form,
    estimate_expected_minutes,
    evaluate_baselines_at_cutoff,
)


def _make_history(
    player_ids: list[int],
    match_weeks: list[int],
    goals: list[int],
    assists: list[int],
    minutes: list[int],
    teams: list[int] | None = None,
    positions: list[str] | None = None,
    match_ids: list[int] | None = None,
    player_names: list[str] | None = None
) -> pd.DataFrame:
    """Create a test history DataFrame."""
    n = len(player_ids)
    if match_ids is None:
        match_ids = list(range(100, 100 + n))
    if player_names is None:
        player_names = [f"Player{pid}" for pid in player_ids]
    return pd.DataFrame({
        "player_id": player_ids,
        "player_name": player_names,
        "match_id": match_ids,
        "match_week": match_weeks,
        "goals": goals,
        "assists": assists,
        "minutes": minutes,
        "team_id": teams if teams else [1] * n,
        "position": positions if positions else ["Forward"] * n,
    })


def _make_target_match(match_id: int = 999, team_id: int = 1) -> pd.Series:
    """Create a target match Series."""
    return pd.Series({
        "match_id": match_id,
        "team_id": team_id,
        "match_week": 5,
    })


class TestGlobalMean:
    """Tests for compute_global_mean."""

    def test_global_mean_empty_history(self):
        """cw=1 returns (0.0, 0.0)."""
        history = pd.DataFrame(columns=["goals", "assists"])
        result = compute_global_mean(history)
        assert result == (0.0, 0.0)

    def test_global_mean_with_data(self):
        """Global mean computed correctly."""
        history = pd.DataFrame({
            "goals": [0, 1, 2, 0, 1],
            "assists": [0, 0, 1, 1, 0],
        })
        result = compute_global_mean(history)
        assert result == (0.8, 0.4)


class TestPlayerMean:
    """Tests for compute_player_mean."""

    def test_player_mean_sufficient_history(self):
        """Player with >= min_history returns player mean."""
        history = _make_history(
            player_ids=[1, 1, 1, 2, 2],
            match_weeks=[1, 2, 3, 1, 2],
            goals=[1, 0, 2, 0, 1],
            assists=[0, 1, 0, 0, 0],
            minutes=[90, 90, 90, 90, 90]
        )
        result = compute_player_mean(history, player_id=1, min_history=3)
        assert result == (1.0, 1/3)

    def test_player_mean_fallback(self):
        """Player with < min_history falls back to global mean."""
        history = _make_history(
            player_ids=[1, 1, 2, 2, 2],
            match_weeks=[1, 2, 1, 2, 3],
            goals=[1, 0, 0, 1, 2],
            assists=[0, 1, 0, 0, 0],
            minutes=[90, 90, 90, 90, 90]
        )
        global_fallback = (0.8, 0.2)
        result = compute_player_mean(history, player_id=1, min_history=3, global_fallback=global_fallback)
        assert result == global_fallback

    def test_player_mean_no_history(self):
        """Player with no history falls back to global mean."""
        history = _make_history(
            player_ids=[2, 2, 2],
            match_weeks=[1, 2, 3],
            goals=[0, 1, 2],
            assists=[0, 0, 1],
            minutes=[90, 90, 90]
        )
        global_fallback = (1.0, 0.33)
        result = compute_player_mean(history, player_id=999, min_history=3, global_fallback=global_fallback)
        assert result == global_fallback


class TestRecentForm:
    """Tests for compute_recent_form."""

    def test_recent_form_sufficient_recent(self):
        """Player with >= 1 recent match returns recent mean."""
        history = _make_history(
            player_ids=[1, 1, 1, 1, 1],
            match_weeks=[1, 2, 3, 4, 5],
            goals=[0, 0, 1, 2, 1],
            assists=[0, 1, 0, 0, 1],
            minutes=[90, 90, 90, 90, 90]
        )
        result = compute_recent_form(history, player_id=1, window=3, global_fallback=(0.0, 0.0))
        # Most recent 3: match_weeks 3,4,5 -> goals=[1,2,1], assists=[0,0,1]
        assert result == (4/3, 1/3)

    def test_recent_form_chained_fallback(self):
        """Player with 0 recent matches falls back through player mean to global."""
        history = _make_history(
            player_ids=[2, 2, 2],
            match_weeks=[1, 2, 3],
            goals=[1, 2, 0],
            assists=[0, 1, 0],
            minutes=[90, 90, 90]
        )
        # Player 1 has no matches at all
        result = compute_recent_form(history, player_id=1, window=5, global_fallback=(0.5, 0.2))
        # Should fallback to player mean (which has no data) -> global
        assert result == (0.5, 0.2)

    def test_recent_form_empty_history(self):
        """Empty history returns global fallback."""
        history = pd.DataFrame(columns=["player_id", "match_week", "goals", "assists", "minutes"])
        result = compute_recent_form(history, player_id=1, window=5, global_fallback=(0.3, 0.1))
        assert result == (0.3, 0.1)


class TestExpectedMinutes:
    """Tests for estimate_expected_minutes."""

    def test_fixed_90(self):
        """fixed_90 method always returns 90."""
        history = _make_history([1], [1], [0], [0], [90])
        fixtures_history = pd.DataFrame()
        target = _make_target_match()
        result = estimate_expected_minutes(history, fixtures_history, target, 1, "fixed_90")
        assert result == 90.0

    def test_position_team_median_with_data(self):
        """position_team_median returns median of same position/team."""
        history = _make_history(
            player_ids=[1, 2, 3, 4],
            match_weeks=[1, 1, 1, 1],
            goals=[0, 0, 0, 0],
            assists=[0, 0, 0, 0],
            minutes=[90, 45, 90, 60],
            teams=[1, 1, 1, 2],
            positions=["Forward", "Forward", "Midfielder", "Forward"]
        )
        fixtures_history = pd.DataFrame()
        target = _make_target_match(team_id=1)
        # Player 1 is Forward on team 1 -> median of [90, 45] = 67.5
        result = estimate_expected_minutes(history, fixtures_history, target, 1, "position_team_median")
        assert result == 67.5

    def test_position_team_median_fallback_chain(self):
        """position_team_median falls back: pos+team -> team -> position -> global -> 90."""
        # No players in history
        history = pd.DataFrame(columns=["player_id", "match_week", "goals", "assists", "minutes", "team_id", "position"])
        fixtures_history = pd.DataFrame()
        target = _make_target_match(team_id=999)
        result = estimate_expected_minutes(history, fixtures_history, target, 1, "position_team_median")
        assert result == 90.0

    def test_player_median_with_sufficient_history(self):
        """player_median returns player's own median minutes."""
        history = _make_history(
            player_ids=[1, 1, 1, 1],
            match_weeks=[1, 2, 3, 4],
            goals=[0, 1, 0, 1],
            assists=[0, 0, 1, 0],
            minutes=[90, 45, 90, 60]
        )
        fixtures_history = pd.DataFrame()
        target = _make_target_match()
        result = estimate_expected_minutes(history, fixtures_history, target, 1, "player_median", min_history=3)
        # Median of [90, 45, 90, 60] = 75.0
        assert result == 75.0

    def test_player_median_fallback(self):
        """player_median falls back to 90 if insufficient history with minutes > 0."""
        history = _make_history(
            player_ids=[1, 1],
            match_weeks=[1, 2],
            goals=[0, 1],
            assists=[0, 0],
            minutes=[90, 0]  # Only 1 match with minutes > 0
        )
        fixtures_history = pd.DataFrame()
        target = _make_target_match()
        result = estimate_expected_minutes(history, fixtures_history, target, 1, "player_median", min_history=3)
        assert result == 90.0


class TestMinutesAdjusted:
    """Tests for compute_minutes_adjusted."""

    def test_minutes_adjusted_rate_fallback(self):
        """Player with 0 historical minutes falls back to player mean rate."""
        history = _make_history(
            player_ids=[1, 1, 1],
            match_weeks=[1, 2, 3],
            goals=[1, 0, 1],
            assists=[0, 1, 0],
            minutes=[0, 0, 0]  # Zero minutes
        )
        fixtures_history = pd.DataFrame()
        target = _make_target_match()
        config = BaselineConfig(min_history=3, expected_minutes_method="fixed_90")

        # Player has 3 matches but 0 minutes -> rate fallback to player mean (which falls back to global)
        result = compute_minutes_adjusted(history, fixtures_history, target, 1, config)
        # Global goals = 2/3, global assists = 1/3, expected_minutes = 90
        # Expected goals = (2/3) * 90/90 = 2/3
        assert np.isclose(result[0], 2/3)
        assert np.isclose(result[1], 1/3)

    def test_minutes_adjusted_expected_minutes_fallback(self):
        """position_team_median falls back through team -> position -> 90."""
        # Empty history
        history = pd.DataFrame(columns=["player_id", "match_week", "goals", "assists", "minutes", "team_id", "position"])
        fixtures_history = pd.DataFrame()
        target = _make_target_match(team_id=999)
        config = BaselineConfig(min_history=3, expected_minutes_method="position_team_median")

        result = compute_minutes_adjusted(history, fixtures_history, target, 1, config)
        # Global rates = 0, expected_minutes = 90
        assert result == (0.0, 0.0)

    def test_minutes_adjusted_normal_case(self):
        """Normal case: player has history with minutes, rate and expected minutes computed."""
        history = _make_history(
            player_ids=[1, 1, 1, 2, 2],
            match_weeks=[1, 2, 3, 1, 2],
            goals=[1, 0, 1, 0, 0],
            assists=[0, 1, 0, 0, 0],
            minutes=[90, 90, 90, 90, 90],
            teams=[1, 1, 1, 1, 1],
            positions=["Forward", "Forward", "Forward", "Midfielder", "Midfielder"]
        )
        fixtures_history = pd.DataFrame()
        target = _make_target_match(team_id=1)
        config = BaselineConfig(min_history=3, expected_minutes_method="position_team_median")

        result = compute_minutes_adjusted(history, fixtures_history, target, 1, config)
        # Player 1: goals=2/270*90=2/3 per 90, assists=1/270*90=1/3 per 90
        # Expected minutes for Forward on team 1: median of [90, 90, 90] = 90
        # Expected goals = (2/3) * 90/90 = 2/3
        assert np.isclose(result[0], 2/3)
        assert np.isclose(result[1], 1/3)


class TestEvaluateBaselinesAtCutoff:
    """Tests for evaluate_baselines_at_cutoff (integration-style)."""

    def test_cutoff_matchweek_boundary(self):
        """match_week < cw strictly; same-week matches never in history."""
        pm = _make_history(
            player_ids=[1, 1, 1, 1],
            match_weeks=[1, 2, 3, 4],
            goals=[0, 1, 0, 1],
            assists=[0, 0, 1, 0],
            minutes=[90, 90, 90, 90],
            match_ids=[10, 20, 30, 40]
        )
        fixtures = pd.DataFrame({
            "match_id": [10, 20, 30, 40],
            "match_week": [1, 2, 3, 4],
            "team_id": [1, 1, 1, 1],
        })
        config = BaselineConfig()

        # Cutoff week 3: history should be match_weeks 1,2; target is match_week 3
        result = evaluate_baselines_at_cutoff(pm, fixtures, 3, config, "all_lineup")
        assert len(result) == 1
        assert result.iloc[0]["cutoff_week"] == 3
        # Player mean should use match_weeks 1,2 only (goals: 0,1 -> mean=0.5)
        assert np.isclose(result.iloc[0]["player_mean_goals_pred"], 0.5)

    def test_matchweek_1_no_history(self):
        """All baselines produce valid outputs for cw=1."""
        pm = _make_history(
            player_ids=[1, 2],
            match_weeks=[1, 1],
            goals=[1, 0],
            assists=[0, 1],
            minutes=[90, 90],
            match_ids=[10, 11]
        )
        fixtures = pd.DataFrame({
            "match_id": [10, 11],
            "match_week": [1, 1],
            "team_id": [1, 2],
        })
        config = BaselineConfig()

        result = evaluate_baselines_at_cutoff(pm, fixtures, 1, config, "all_lineup")
        assert len(result) == 2
        # All baselines should be (0, 0) or global fallback (which is also 0 with empty history)
        for col in ["global_goals_pred", "player_mean_goals_pred", "recent_form_goals_pred", "minutes_adj_goals_pred"]:
            assert (result[col] == 0.0).all()

    def test_population_a_includes_unused_subs(self):
        """Population A includes 0-minute players; Population B excludes them."""
        pm = _make_history(
            player_ids=[1, 2],
            match_weeks=[2, 2],
            goals=[0, 0],
            assists=[0, 0],
            minutes=[90, 0],  # Player 2 is unused sub
            match_ids=[20, 21]
        )
        fixtures = pd.DataFrame({
            "match_id": [20, 21],
            "match_week": [2, 2],
            "team_id": [1, 1],
        })
        config = BaselineConfig()

        result_a = evaluate_baselines_at_cutoff(pm, fixtures, 2, config, "all_lineup")
        result_b = evaluate_baselines_at_cutoff(pm, fixtures, 2, config, "played_only")

        assert len(result_a) == 2
        assert len(result_b) == 1
        assert result_b.iloc[0]["player_id"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])