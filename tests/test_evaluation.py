"""End-to-end evaluation tests for rolling-origin evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.rolling import run_rolling_evaluation
from src.features.baselines import BaselineConfig


def _make_test_data(num_matchweeks: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create test player_match and fixtures DataFrames."""
    player_match_rows = []
    fixtures_rows = []
    match_id = 100

    for mw in range(1, num_matchweeks + 1):
        for team_id, team_name in [(1, "TeamA"), (2, "TeamB")]:
            # 3 players per team per match: 2 starters, 1 unused sub
            for player_num, minutes in [(1, 90), (2, 90), (3, 0)]:
                player_id = team_id * 100 + mw * 10 + player_num
                goals = 1 if (player_num == 1 and mw % 2 == 0) else 0
                assists = 1 if (player_num == 2 and mw % 3 == 0) else 0

                player_match_rows.append({
                    "match_id": match_id,
                    "match_week": mw,
                    "player_id": player_id,
                    "player_name": f"Player{player_id}",
                    "team_id": team_id,
                    "team_name": team_name,
                    "position": "Forward" if player_num == 1 else "Midfielder",
                    "minutes": minutes,
                    "goals": goals,
                    "assists": assists,
                    "source": "test",
                    "source_url": "test",
                    "ingested_at": "2024-01-01",
                })

            fixtures_rows.append({
                "match_id": match_id,
                "match_week": mw,
                "team_id": team_id,
                "home_team_id": 1,
                "away_team_id": 2,
            })
            match_id += 1

    player_match = pd.DataFrame(player_match_rows)
    fixtures = pd.DataFrame(fixtures_rows)
    return player_match, fixtures


class TestRollingEvaluation:
    """End-to-end tests for run_rolling_evaluation."""

    def test_identical_rows_across_baselines(self):
        """Within each population/cutoff, all baselines evaluated on same rows."""
        pm, fixtures = _make_test_data(5)
        config = BaselineConfig()

        results = run_rolling_evaluation(pm, fixtures, config, cutoffs=[2, 3, 4])

        for pop_name in ["predictions_all_lineup", "predictions_played_only"]:
            pred = results[pop_name]
            if pred.empty:
                continue

            # For each cutoff, check all baselines have same number of rows
            for cw in pred["cutoff_week"].unique():
                cw_data = pred[pred["cutoff_week"] == cw]
                # All prediction columns should have no NaN (same rows)
                pred_cols = [
                    "global_goals_pred", "global_assists_pred",
                    "player_mean_goals_pred", "player_mean_assists_pred",
                    "recent_form_goals_pred", "recent_form_assists_pred",
                    "minutes_adj_goals_pred", "minutes_adj_assists_pred",
                ]
                for col in pred_cols:
                    assert cw_data[col].notna().all(), f"NaN in {col} at cutoff {cw}"

    def test_cutoff_matchweek_boundary(self):
        """match_week < cw strictly; same-week matches never in history."""
        pm, fixtures = _make_test_data(5)
        config = BaselineConfig(min_history=1)

        results = run_rolling_evaluation(pm, fixtures, config, cutoffs=[3])

        pred = results["predictions_all_lineup"]
        cw3_data = pred[pred["cutoff_week"] == 3]

        # At cutoff 3, history is matchweeks 1,2
        # Player_mean for a player should be based on matchweeks 1,2 only
        # Let's check a specific player
        player_101 = cw3_data[cw3_data["player_id"] == 111]  # TeamA, starter in mw3
        if not player_101.empty:
            # This player's history in mw1,2: goals in mw2 (since mw%2==0), 0 in mw1
            # mean = 0.5
            assert np.isclose(player_101.iloc[0]["player_mean_goals_pred"], 0.5)

    def test_matchweek_1_no_history(self):
        """All baselines produce valid outputs for cw=1."""
        pm, fixtures = _make_test_data(5)
        config = BaselineConfig()

        results = run_rolling_evaluation(pm, fixtures, config, cutoffs=[1])

        pred_a = results["predictions_all_lineup"]
        pred_b = results["predictions_played_only"]

        assert not pred_a.empty
        assert len(pred_a) == 6  # 2 teams * 3 players per team (1 match per team per week)

        # All predictions should be 0 or global mean (which is 0 with empty history)
        pred_cols = [
            "global_goals_pred", "global_assists_pred",
            "player_mean_goals_pred", "player_mean_assists_pred",
            "recent_form_goals_pred", "recent_form_assists_pred",
            "minutes_adj_goals_pred", "minutes_adj_assists_pred",
        ]
        for col in pred_cols:
            assert (pred_a[col] == 0.0).all()

    def test_population_a_includes_unused_subs(self):
        """Population A includes 0-minute players; Population B excludes them."""
        pm, fixtures = _make_test_data(3)
        config = BaselineConfig()

        results = run_rolling_evaluation(pm, fixtures, config, cutoffs=[2])

        pred_a = results["predictions_all_lineup"]
        pred_b = results["predictions_played_only"]

        cw2_a = pred_a[pred_a["cutoff_week"] == 2]
        cw2_b = pred_b[pred_b["cutoff_week"] == 2]

        # Population A: 2 teams * 3 players * 1 match = 6 per team? Wait, each team has 1 match per week
        # 2 teams * 3 players = 6 rows
        # Population B: 2 teams * 2 players (starters) = 4 rows
        assert len(cw2_a) == 6
        assert len(cw2_b) == 4

        # Verify 0-minute players in A but not B
        assert (cw2_a["minutes_actual"] == 0).sum() == 2  # 1 unused sub per team
        assert (cw2_b["minutes_actual"] == 0).sum() == 0

    def test_population_b_is_primary(self):
        """Population B (played_only) is smaller and excludes unused subs."""
        pm, fixtures = _make_test_data(4)
        config = BaselineConfig()

        results = run_rolling_evaluation(pm, fixtures, config, cutoffs=[2, 3])

        pred_a = results["predictions_all_lineup"]
        pred_b = results["predictions_played_only"]

        for cw in [2, 3]:
            a_cw = pred_a[pred_a["cutoff_week"] == cw]
            b_cw = pred_b[pred_b["cutoff_week"] == cw]
            assert len(b_cw) < len(a_cw)  # B excludes unused subs

    def test_metrics_computed(self):
        """Metrics are computed for all baselines and populations."""
        pm, fixtures = _make_test_data(5)
        config = BaselineConfig()

        results = run_rolling_evaluation(pm, fixtures, config, cutoffs=[2, 3])

        metrics_per = results["metrics_per_cutoff"]
        metrics_agg = results["metrics_aggregate"]

        assert not metrics_per.empty
        assert not metrics_agg.empty

        # Check required columns
        required_cols = [
            "mae_goals", "rmse_goals", "poisson_deviance_goals",
            "mae_assists", "rmse_assists", "poisson_deviance_assists",
            "sample_size", "cutoff_week", "population", "baseline"
        ]
        for col in required_cols:
            assert col in metrics_per.columns

        # Check 4 baselines * 2 populations * 2 cutoffs = 16 rows
        assert len(metrics_per) == 16

        # Aggregate: 4 baselines * 2 populations = 8 rows
        assert len(metrics_agg) == 8

    def test_sample_sizes_reported(self):
        """Sample sizes per cutoff and in aggregate are reported."""
        pm, fixtures = _make_test_data(4)
        config = BaselineConfig()

        results = run_rolling_evaluation(pm, fixtures, config, cutoffs=[2, 3])

        metrics_per = results["metrics_per_cutoff"]

        # Each row should have sample_size
        assert (metrics_per["sample_size"] > 0).all()

        # Sample sizes should be consistent within population/cutoff
        for (pop, cw), group in metrics_per.groupby(["population", "cutoff_week"]):
            sizes = group["sample_size"].unique()
            assert len(sizes) == 1, f"Inconsistent sample sizes for {pop} cw={cw}"

    def test_refresh_every_cutoff(self):
        """Baselines are recomputed from scratch at each cutoff (stateless)."""
        pm, fixtures = _make_test_data(5)
        config = BaselineConfig()

        # Run evaluation for cutoffs 2 and 3
        results = run_rolling_evaluation(pm, fixtures, config, cutoffs=[2, 3])

        pred = results["predictions_all_lineup"]

        # Global mean at cw=2 should be based on matchweek 1 only
        # Global mean at cw=3 should be based on matchweeks 1,2
        cw2_global = pred[pred["cutoff_week"] == 2]["global_goals_pred"].iloc[0]
        cw3_global = pred[pred["cutoff_week"] == 3]["global_goals_pred"].iloc[0]

        # They should be different (refit at each cutoff)
        # cw=2 history: matchweek 1 goals
        # cw=3 history: matchweek 1 + 2 goals
        assert cw2_global != cw3_global


class TestDataIntegrity:
    """Tests for data integrity in evaluation."""

    def test_all_players_have_match_week(self):
        """All player_match rows have match_week after merge."""
        pm, fixtures = _make_test_data(5)
        # Remove match_week from pm
        pm_no_week = pm.drop(columns=["match_week"])
        fixtures_with_week = fixtures

        # This should not raise
        from src.evaluation.rolling import prepare_data_with_matchweek
        pm_with_week = prepare_data_with_matchweek(pm_no_week, fixtures_with_week)

        assert pm_with_week["match_week"].notna().all()

    def test_no_duplicate_predictions(self):
        """No duplicate (match_id, player_id) in predictions."""
        pm, fixtures = _make_test_data(4)
        config = BaselineConfig()

        results = run_rolling_evaluation(pm, fixtures, config, cutoffs=[2, 3])

        for pop_name in ["predictions_all_lineup", "predictions_played_only"]:
            pred = results[pop_name]
            if pred.empty:
                continue
            # Check no duplicates
            dupes = pred.duplicated(subset=["match_id", "player_id"]).sum()
            assert dupes == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])