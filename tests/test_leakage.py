"""Leakage test: ensures future data does not affect predictions at earlier cutoffs.

This is a mandatory test per the implementation contract. It injects fake future data
and verifies that predictions at an earlier cutoff remain identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.baselines import BaselineConfig, evaluate_baselines_at_cutoff


def _make_base_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create base player_match and fixtures DataFrames for testing."""
    # 3 matchweeks, 2 teams, 2 players per team per match
    player_match_rows = []
    match_id = 100

    for mw in [1, 2, 3]:
        for team_id, team_name in [(1, "TeamA"), (2, "TeamB")]:
            for player_num in [1, 2]:
                player_id = team_id * 10 + player_num
                player_match_rows.append({
                    "match_id": match_id,
                    "match_week": mw,
                    "player_id": player_id,
                    "player_name": f"Player{player_id}",
                    "team_id": team_id,
                    "team_name": team_name,
                    "position": "Forward",
                    "minutes": 90,
                    "goals": 0,
                    "assists": 0,
                    "source": "test",
                    "source_url": "test",
                    "ingested_at": "2024-01-01",
                })
            match_id += 1

    player_match = pd.DataFrame(player_match_rows)

    fixtures_rows = []
    match_id = 100
    for mw in [1, 2, 3]:
        fixtures_rows.append({
            "match_id": match_id,
            "match_week": mw,
            "team_id": 1,
            "home_team_id": 1,
            "away_team_id": 2,
        })
        match_id += 1
        fixtures_rows.append({
            "match_id": match_id,
            "match_week": mw,
            "team_id": 2,
            "home_team_id": 2,
            "away_team_id": 1,
        })
        match_id += 1

    fixtures = pd.DataFrame(fixtures_rows)
    return player_match, fixtures


def test_no_leakage_future_data():
    """Injecting future matchweek data does not change predictions at earlier cutoff.

    This test:
    1. Takes the full player_match dataset
    2. Artificially appends/modifies rows for a future matchweek (e.g., matchweek 3)
    3. Runs the prediction pipeline for cutoff cw=2
    4. Asserts that predictions for cw=2 are identical with and without the injected future data
    """
    pm_base, fixtures_base = _make_base_data()
    config = BaselineConfig()

    # Get predictions at cutoff cw=2 WITHOUT future data
    # (only matchweeks 1,2 exist in base data)
    pred_original = evaluate_baselines_at_cutoff(pm_base, fixtures_base, 2, config, "all_lineup")

    # Now INJECT fake future data for matchweek 3
    # Add high-scoring matches for player 11 in matchweek 3
    future_rows = pd.DataFrame([{
        "match_id": 999,
        "match_week": 3,
        "player_id": 11,
        "player_name": "Player11",
        "team_id": 1,
        "team_name": "TeamA",
        "position": "Forward",
        "minutes": 90,
        "goals": 5,  # Artificially high
        "assists": 3,
        "source": "test",
        "source_url": "test",
        "ingested_at": "2024-01-01",
    }])

    # Also need to add the matchweek 3 fixture for team 1 if not present
    # (it is present in base, but let's ensure)
    fixtures_with_future = fixtures_base.copy()

    pm_with_future = pd.concat([pm_base, future_rows], ignore_index=True)

    # Get predictions at cutoff cw=2 WITH future data injected
    pred_with_future = evaluate_baselines_at_cutoff(pm_with_future, fixtures_with_future, 2, config, "all_lineup")

    # Assertions: predictions at cw=2 must be IDENTICAL
    # Sort by player_id for consistent comparison
    pred_original = pred_original.sort_values("player_id").reset_index(drop=True)
    pred_with_future = pred_with_future.sort_values("player_id").reset_index(drop=True)

    # Check same players
    assert len(pred_original) == len(pred_with_future)
    assert list(pred_original["player_id"]) == list(pred_with_future["player_id"])

    # Check all prediction columns are identical (bitwise or within floating-point tolerance)
    pred_cols = [
        "global_goals_pred", "global_assists_pred",
        "player_mean_goals_pred", "player_mean_assists_pred",
        "recent_form_goals_pred", "recent_form_assists_pred",
        "minutes_adj_goals_pred", "minutes_adj_assists_pred",
    ]

    for col in pred_cols:
        orig_vals = pred_original[col].values
        future_vals = pred_with_future[col].values
        np.testing.assert_allclose(
            orig_vals, future_vals, rtol=1e-10, atol=1e-12,
            err_msg=f"Leakage detected: {col} differs with future data injection"
        )

    # Also verify the actuals are the same (they should be, since target is cw=2)
    assert list(pred_original["goals_actual"]) == list(pred_with_future["goals_actual"])
    assert list(pred_original["assists_actual"]) == list(pred_with_future["assists_actual"])


def test_leakage_player_mean_specific():
    """Specific test for player_mean baseline leakage."""
    pm_base, fixtures_base = _make_base_data()
    config = BaselineConfig(min_history=1)

    # In base data, all goals=0. Modify player 11 in matchweek 1 to have 1 goal.
    pm_base_modified = pm_base.copy()
    mask = (pm_base_modified["player_id"] == 11) & (pm_base_modified["match_week"] == 1)
    pm_base_modified.loc[mask, "goals"] = 1

    # Also add a matchweek 2 entry for player 11 with 0 goals (target)
    # This is already in the base data

    # Future injection: 10 goals in matchweek 3
    future_row = pd.DataFrame([{
        "match_id": 999,
        "match_week": 3,
        "player_id": 11,
        "player_name": "Player11",
        "team_id": 1,
        "team_name": "TeamA",
        "position": "Forward",
        "minutes": 90,
        "goals": 10,
        "assists": 0,
        "source": "test",
        "source_url": "test",
        "ingested_at": "2024-01-01",
    }])

    pm_with_future = pd.concat([pm_base_modified, future_row], ignore_index=True)

    pred_original = evaluate_baselines_at_cutoff(pm_base_modified, fixtures_base, 2, config, "all_lineup")
    pred_with_future = evaluate_baselines_at_cutoff(pm_with_future, fixtures_base, 2, config, "all_lineup")

    # Player 11's player_mean_goals_pred at cw=2 should be based on matchweeks 1 only (history)
    # Not affected by matchweek 3 injection
    p11_orig = pred_original[pred_original["player_id"] == 11].iloc[0]
    p11_future = pred_with_future[pred_with_future["player_id"] == 11].iloc[0]

    # At cutoff=2, history is match_week < 2 (only matchweek 1 with 1 goal)
    # Target is matchweek 2 (0 goals)
    # Player mean = 1.0 (from matchweek 1 only)
    assert np.isclose(p11_orig["player_mean_goals_pred"], 1.0)
    assert np.isclose(p11_future["player_mean_goals_pred"], 1.0)
    assert np.isclose(p11_orig["player_mean_goals_pred"], p11_future["player_mean_goals_pred"])


def test_leakage_minutes_adjusted():
    """Test that minutes_adjusted baseline doesn't leak future minutes."""
    pm_base, fixtures_base = _make_base_data()
    config = BaselineConfig(min_history=1, expected_minutes_method="player_median")

    # Player 11 plays 90 mins in matchweek 1, 90 in matchweek 2
    # Future injection: plays 10 minutes in matchweek 3 (injury)
    future_row = pd.DataFrame([{
        "match_id": 999,
        "match_week": 3,
        "player_id": 11,
        "player_name": "Player11",
        "team_id": 1,
        "team_name": "TeamA",
        "position": "Forward",
        "minutes": 10,  # Low minutes due to injury
        "goals": 0,
        "assists": 0,
        "source": "test",
        "source_url": "test",
        "ingested_at": "2024-01-01",
    }])

    pm_with_future = pd.concat([pm_base, future_row], ignore_index=True)

    pred_original = evaluate_baselines_at_cutoff(pm_base, fixtures_base, 2, config, "all_lineup")
    pred_with_future = evaluate_baselines_at_cutoff(pm_with_future, fixtures_base, 2, config, "all_lineup")

    p11_orig = pred_original[pred_original["player_id"] == 11].iloc[0]
    p11_future = pred_with_future[pred_with_future["player_id"] == 11].iloc[0]

    # Expected minutes should be median of [90, 90] = 90, NOT affected by future 10-min match
    assert np.isclose(p11_orig["minutes_adj_goals_pred"], p11_future["minutes_adj_goals_pred"])
    assert np.isclose(p11_orig["minutes_adj_assists_pred"], p11_future["minutes_adj_assists_pred"])


def test_leakage_recent_form():
    """Test that recent_form baseline doesn't leak future matches."""
    pm_base, fixtures_base = _make_base_data()
    config = BaselineConfig(recent_window=2)

    # Player 11: matchweek 1=0 goals, matchweek 2=0 goals
    # Future injection: matchweek 3=5 goals
    future_row = pd.DataFrame([{
        "match_id": 999,
        "match_week": 3,
        "player_id": 11,
        "player_name": "Player11",
        "team_id": 1,
        "team_name": "TeamA",
        "position": "Forward",
        "minutes": 90,
        "goals": 5,
        "assists": 0,
        "source": "test",
        "source_url": "test",
        "ingested_at": "2024-01-01",
    }])

    pm_with_future = pd.concat([pm_base, future_row], ignore_index=True)

    pred_original = evaluate_baselines_at_cutoff(pm_base, fixtures_base, 2, config, "all_lineup")
    pred_with_future = evaluate_baselines_at_cutoff(pm_with_future, fixtures_base, 2, config, "all_lineup")

    p11_orig = pred_original[pred_original["player_id"] == 11].iloc[0]
    p11_future = pred_with_future[pred_with_future["player_id"] == 11].iloc[0]

    # Recent form at cw=2 should use matchweeks 1,2 only (both 0 goals)
    # NOT matchweek 3
    assert np.isclose(p11_orig["recent_form_goals_pred"], 0.0)
    assert np.isclose(p11_future["recent_form_goals_pred"], 0.0)
    assert np.isclose(p11_orig["recent_form_goals_pred"], p11_future["recent_form_goals_pred"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])