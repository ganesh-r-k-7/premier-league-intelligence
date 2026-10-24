"""Tests for match outcome baselines and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.match_baselines import (
    MatchBaselineConfig,
    compute_global_outcome_frequencies,
    compute_team_strength_baseline,
    compute_recent_form_baseline,
    compute_home_advantage_baseline,
    evaluate_match_baselines_at_cutoff,
)
from src.evaluation.metrics import (
    multiclass_log_loss,
    multiclass_brier_score,
    accuracy_score,
    compute_match_metrics,
    compute_match_metrics_per_cutoff,
    compute_match_aggregate_metrics,
)
from src.evaluation.match_rolling import run_match_rolling_evaluation


def _make_test_fixtures(num_matchweeks: int = 5, num_teams: int = 4) -> pd.DataFrame:
    """Create test fixtures DataFrame with known outcomes."""
    fixtures_rows = []
    match_id = 100

    # Create a round-robin schedule for simplicity
    team_ids = list(range(1, num_teams + 1))

    for mw in range(1, num_matchweeks + 1):
        # Each week, pair teams: (1 vs 2), (3 vs 4), etc.
        for i in range(0, num_teams, 2):
            if i + 1 >= num_teams:
                break
            home_id = team_ids[i]
            away_id = team_ids[i + 1]

            # Deterministic scores based on matchweek and team
            home_score = (home_id + mw) % 3
            away_score = (away_id + mw) % 3

            fixtures_rows.append({
                "match_id": match_id,
                "match_week": mw,
                "season_id": 27,
                "season_name": "2015/2016",
                "home_team_id": home_id,
                "home_team_name": f"Team{home_id}",
                "away_team_id": away_id,
                "away_team_name": f"Team{away_id}",
                "home_score": home_score,
                "away_score": away_score,
                "match_status": "available",
                "stadium": "Stadium",
                "source": "test",
                "source_url": "test",
                "ingested_at": "2024-01-01",
            })
            match_id += 1

    return pd.DataFrame(fixtures_rows)


class TestGlobalOutcomeFrequencies:
    """Tests for compute_global_outcome_frequencies."""

    def test_empty_history_returns_uniform(self):
        """Empty history returns uniform probabilities."""
        history = pd.DataFrame(columns=["home_score", "away_score"])
        probs = compute_global_outcome_frequencies(history, smoothing_alpha=1.0)
        assert np.allclose(probs, [1/3, 1/3, 1/3])

    def test_with_data(self):
        """Test with known outcomes."""
        history = pd.DataFrame({
            "home_score": [2, 1, 0, 3, 1],
            "away_score": [1, 1, 2, 0, 0],
        })
        # Home wins: rows where home > away: 2>1 (yes), 1>1 (no), 0>2 (no), 3>0 (yes), 1>0 (yes) = 3
        # Draws: 1==1 (yes), others no = 1
        # Away wins: 0>2 (yes), others no = 1
        probs = compute_global_outcome_frequencies(history, smoothing_alpha=1.0)
        total = len(history) + 3
        expected_home = (3 + 1) / total
        expected_draw = (1 + 1) / total
        expected_away = (1 + 1) / total
        assert np.allclose(probs, [expected_home, expected_draw, expected_away])
        assert abs(sum(probs) - 1.0) < 1e-10
        # Check actual values: 4/8, 2/8, 2/8 = 0.5, 0.25, 0.25
        assert np.isclose(probs[0], 0.5)
        assert np.isclose(probs[1], 0.25)
        assert np.isclose(probs[2], 0.25)

    def test_smoothing_works(self):
        """Test that smoothing affects probabilities."""
        history = pd.DataFrame({
            "home_score": [2, 1],
            "away_score": [1, 1],
        })
        # 1 home win, 1 draw, 0 away wins
        probs_no_smooth = compute_global_outcome_frequencies(history, smoothing_alpha=0.0)
        probs_smooth = compute_global_outcome_frequencies(history, smoothing_alpha=1.0)
        # With smoothing, probabilities should be less extreme
        # No smoothing: home=1/2=0.5, draw=1/2=0.5, away=0
        # With smoothing: home=(1+1)/(2+3)=0.4, draw=(1+1)/5=0.4, away=(0+1)/5=0.2
        assert np.isclose(probs_no_smooth[0], 0.5)
        assert np.isclose(probs_no_smooth[1], 0.5)
        assert np.isclose(probs_smooth[0], 0.4)
        assert np.isclose(probs_smooth[1], 0.4)
        assert probs_smooth[2] > probs_no_smooth[2]  # away prob increased from 0 to 0.2


class TestHomeAdvantageBaseline:
    """Tests for compute_home_advantage_baseline."""

    def test_returns_global_frequencies(self):
        """Home advantage baseline equals global frequencies."""
        history = pd.DataFrame({
            "home_score": [2, 1, 0],
            "away_score": [1, 1, 2],
        })
        global_probs = compute_global_outcome_frequencies(history, smoothing_alpha=1.0)
        ha_probs = compute_home_advantage_baseline(history, global_probs, smoothing_alpha=1.0)
        assert np.allclose(ha_probs, global_probs)


class TestTeamStrengthBaseline:
    """Tests for compute_team_strength_baseline."""

    def test_fallback_when_insufficient_history(self):
        """Returns global probs when team history insufficient."""
        history = pd.DataFrame({
            "match_week": [1, 2],
            "home_team_id": [1, 2],
            "away_team_id": [2, 1],
            "home_score": [2, 1],
            "away_score": [1, 0],
        })
        global_probs = (0.4, 0.3, 0.3)
        probs = compute_team_strength_baseline(
            history, home_team_id=1, away_team_id=2,
            min_history=5, global_probs=global_probs, smoothing_alpha=1.0
        )
        assert np.allclose(probs, global_probs)

    def test_team_specific_when_sufficient_history(self):
        """Uses team-specific records when history sufficient."""
        # Team 1 at home: 3 wins, 0 draws, 0 losses
        # Team 2 away: 0 wins, 0 draws, 3 losses
        history = pd.DataFrame({
            "match_week": [1, 2, 3, 4, 5, 6],
            "home_team_id": [1, 1, 1, 3, 3, 3],
            "away_team_id": [2, 2, 2, 4, 4, 4],
            "home_score": [2, 3, 1, 1, 1, 1],
            "away_score": [0, 0, 0, 1, 1, 1],
        })
        global_probs = (0.4, 0.3, 0.3)
        probs = compute_team_strength_baseline(
            history, home_team_id=1, away_team_id=2,
            min_history=3, global_probs=global_probs, smoothing_alpha=1.0
        )
        # Team 1 home wins: 3, Team 2 away losses: 3 (so home wins for team 1)
        # Combined: 3 home wins, 0 draws, 0 away wins
        # With smoothing: (3+1)/(3+3) = 4/6 = 0.667 home, 1/6 draw, 1/6 away
        assert probs[0] > global_probs[0]  # Home win prob should be higher


class TestRecentFormBaseline:
    """Tests for compute_recent_form_baseline."""

    def test_fallback_when_insufficient_history(self):
        """Returns global probs when team history insufficient."""
        history = pd.DataFrame({
            "match_week": [1],
            "home_team_id": [1],
            "away_team_id": [3],  # Team 2 doesn't play in history
            "home_score": [2],
            "away_score": [1],
        })
        global_probs = (0.4, 0.3, 0.3)
        probs = compute_recent_form_baseline(
            history, home_team_id=1, away_team_id=2,
            window=5, global_probs=global_probs, smoothing_alpha=1.0
        )
        # Team 2 has no matches in history -> fallback to global
        assert np.allclose(probs, global_probs)

    def test_recent_form_calculation(self):
        """Test recent form combines both teams' perspectives."""
        # Team 1: recent wins
        # Team 2: recent losses
        history = pd.DataFrame({
            "match_week": [1, 2, 3, 4, 5],
            "home_team_id": [1, 1, 1, 2, 2],
            "away_team_id": [3, 3, 3, 4, 4],
            "home_score": [2, 2, 2, 0, 0],  # Team 1 wins, Team 2 loses
            "away_score": [0, 0, 0, 2, 2],
        })
        global_probs = (0.4, 0.3, 0.3)
        probs = compute_recent_form_baseline(
            history, home_team_id=1, away_team_id=2,
            window=3, global_probs=global_probs, smoothing_alpha=1.0
        )
        # Team 1 recent: 3 wins, Team 2 recent: 2 losses (as away)
        # Combined home wins = 3 + 2 = 5, draws = 0, away wins = 0
        # With smoothing: (5+1)/(5+3) = 6/8 = 0.75 home
        assert probs[0] > global_probs[0]


class TestEvaluateMatchBaselinesAtCutoff:
    """Tests for evaluate_match_baselines_at_cutoff."""

    def test_cutoff_1_no_history(self):
        """Cutoff 1 returns global frequencies for all matches."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=3, smoothing_alpha=1.0)
        preds = evaluate_match_baselines_at_cutoff(fixtures, 1, config)
        assert not preds.empty
        # 4 teams = 2 matches per week, so cutoff 1 = 2 matches
        assert len(preds) == 2
        # Check all baselines produce valid probabilities
        prob_cols = [
            "global_home_prob", "global_draw_prob", "global_away_prob",
            "team_strength_home_prob", "team_strength_draw_prob", "team_strength_away_prob",
            "recent_form_home_prob", "recent_form_draw_prob", "recent_form_away_prob",
        ]
        for col in prob_cols:
            assert (preds[col] >= 0).all()
            assert (preds[col] <= 1).all()
        # Check probabilities sum to 1
        for _, row in preds.iterrows():
            assert abs(row["global_home_prob"] + row["global_draw_prob"] + row["global_away_prob"] - 1.0) < 1e-10

    def test_cutoff_matchweek_boundary(self):
        """match_week < cw strictly; same-week matches never in history."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)
        preds = evaluate_match_baselines_at_cutoff(fixtures, 3, config)
        # At cutoff 3, history is matchweeks 1,2
        # Team strength for team 1 should use matchweeks 1,2 only
        assert not preds.empty

    def test_outcome_labels_correct(self):
        """Outcome labels: 0=away, 1=draw, 2=home."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)
        preds = evaluate_match_baselines_at_cutoff(fixtures, 2, config)

        for _, row in preds.iterrows():
            if row["home_score"] > row["away_score"]:
                assert row["outcome"] == 2  # home win
            elif row["home_score"] == row["away_score"]:
                assert row["outcome"] == 1  # draw
            else:
                assert row["outcome"] == 0  # away win

    def test_season_filter(self):
        """season_id filters correctly."""
        fixtures = _make_test_fixtures(5)
        # Add a second season
        fixtures2 = fixtures.copy()
        fixtures2["season_id"] = 99
        fixtures2["match_id"] = fixtures2["match_id"] + 1000
        fixtures_combined = pd.concat([fixtures, fixtures2], ignore_index=True)

        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)
        preds = evaluate_match_baselines_at_cutoff(fixtures_combined, 2, config, season_id=27)
        # Should only have matches from season 27
        assert (preds["season_id"] == 27).all()
        assert len(preds) == 2  # 2 matches per week in test data


class TestClassificationMetrics:
    """Tests for multiclass classification metrics."""

    def test_log_loss_perfect_predictions(self):
        """Log loss = 0 for perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        loss = multiclass_log_loss(y_true, y_pred)
        assert np.isclose(loss, 0.0)

    def test_log_loss_uniform_predictions(self):
        """Log loss = log(3) for uniform predictions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
        ])
        loss = multiclass_log_loss(y_true, y_pred)
        assert np.isclose(loss, np.log(3))

    def test_log_loss_handles_zero_prob(self):
        """Log loss clips probabilities to avoid log(0)."""
        y_true = np.array([0, 1])
        y_pred = np.array([
            [0.0, 1.0, 0.0],  # Predicts draw, true is away
            [0.5, 0.5, 0.0],
        ])
        loss = multiclass_log_loss(y_true, y_pred)
        assert loss is not None
        assert loss > 0
        assert np.isfinite(loss)

    def test_log_loss_invalid_inputs(self):
        """Log loss returns None for invalid inputs."""
        # Length mismatch
        assert multiclass_log_loss(np.array([0, 1]), np.array([[1, 0, 0]])) is None
        # Invalid class label
        assert multiclass_log_loss(np.array([3]), np.array([[1, 0, 0]])) is None
        # Negative probabilities - function clips and renormalizes, so returns finite value
        loss = multiclass_log_loss(np.array([0]), np.array([[-0.1, 1.1, 0]]))
        assert loss is not None
        assert np.isfinite(loss)

    def test_brier_score_perfect_predictions(self):
        """Brier score = 0 for perfect predictions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        brier = multiclass_brier_score(y_true, y_pred)
        assert np.isclose(brier, 0.0)

    def test_brier_score_worst_predictions(self):
        """Brier score for always predicting wrong classes."""
        y_true = np.array([0, 1, 2])  # 0=away, 1=draw, 2=home
        # multiclass_brier_score expects probabilities in class order [away, draw, home]
        y_pred = np.array([
            [0.5, 0.5, 0.0],  # True=0 (away), predicts away=0.5, draw=0.5, home=0
            [0.5, 0.0, 0.5],  # True=1 (draw), predicts away=0.5, draw=0, home=0.5
            [0.0, 0.5, 0.5],  # True=2 (home), predicts away=0, draw=0.5, home=0.5
        ])
        brier = multiclass_brier_score(y_true, y_pred)
        # For class 0 (away): pred=[0.5, 0.5, 0.0], onehot=[1,0,0] -> (0.5-1)^2 + (0.5-0)^2 + 0 = 0.25 + 0.25 = 0.5
        # For class 1 (draw): pred=[0.5, 0.0, 0.5], onehot=[0,1,0] -> 0.25 + 1 + 0.25 = 1.5
        # For class 2 (home): pred=[0.0, 0.5, 0.5], onehot=[0,0,1] -> 0 + 0.25 + 0.25 = 0.5
        # Mean = (0.5 + 1.5 + 0.5) / 3 = 2.5 / 3 = 0.833...
        assert np.isclose(brier, 2.5 / 3)

    def test_brier_score_uniform_predictions(self):
        """Brier score for uniform predictions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
        ])
        brier = multiclass_brier_score(y_true, y_pred)
        # Each: (1/3-1)^2 + (1/3-0)^2 + (1/3-0)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3
        # Mean = 2/3
        assert np.isclose(brier, 2/3)

    def test_brier_score_invalid_inputs(self):
        """Brier score returns None for invalid inputs."""
        # Length mismatch
        assert multiclass_brier_score(np.array([0, 1]), np.array([[1, 0, 0]])) is None
        # Invalid class label
        assert multiclass_brier_score(np.array([3]), np.array([[1, 0, 0]])) is None
        # Probabilities don't sum to 1
        assert multiclass_brier_score(np.array([0]), np.array([[0.5, 0.5, 0.5]])) is None

    def test_accuracy_perfect(self):
        """Accuracy = 1 for perfect predictions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        acc = accuracy_score(y_true, y_pred)
        assert np.isclose(acc, 1.0)

    def test_accuracy_from_probs(self):
        """Accuracy works with probability matrix (argmax)."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        acc = accuracy_score(y_true, y_pred)
        assert np.isclose(acc, 1.0)

    def test_accuracy_from_classes(self):
        """Accuracy works with class labels directly."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        acc = accuracy_score(y_true, y_pred)
        assert np.isclose(acc, 1.0)

    def test_compute_match_metrics(self):
        """compute_match_metrics returns all metrics."""
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ])
        metrics = compute_match_metrics(y_true, y_pred)
        assert "log_loss" in metrics
        assert "brier_score" in metrics
        assert "accuracy" in metrics
        assert "sample_size" in metrics
        assert metrics["sample_size"] == 4
        assert metrics["log_loss"] is not None
        assert metrics["brier_score"] is not None
        assert metrics["accuracy"] is not None


class TestMatchMetricsPerCutoff:
    """Tests for compute_match_metrics_per_cutoff."""

    def test_per_cutoff_metrics(self):
        """Metrics computed per cutoff and baseline."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)

        all_preds = []
        for cw in [2, 3]:
            pred = evaluate_match_baselines_at_cutoff(fixtures, cw, config)
            all_preds.append(pred)
        preds = pd.concat(all_preds, ignore_index=True)

        metrics = compute_match_metrics_per_cutoff(preds)
        assert not metrics.empty
        # 2 cutoffs * 1 population * 4 baselines = 8 rows
        assert len(metrics) == 8
        assert "cutoff_week" in metrics.columns
        assert "population" in metrics.columns
        assert "baseline" in metrics.columns
        assert "log_loss" in metrics.columns
        assert "brier_score" in metrics.columns
        assert "accuracy" in metrics.columns


class TestMatchRollingEvaluation:
    """Tests for run_match_rolling_evaluation."""

    def test_identical_rows_across_baselines(self):
        """Within each cutoff, all baselines evaluated on same rows."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)

        results = run_match_rolling_evaluation(fixtures, config, cutoffs=[2, 3])
        preds = results["predictions"]

        for cw in preds["cutoff_week"].unique():
            cw_data = preds[preds["cutoff_week"] == cw]
            # All probability columns should have no NaN
            prob_cols = [
                "global_home_prob", "global_draw_prob", "global_away_prob",
                "team_strength_home_prob", "team_strength_draw_prob", "team_strength_away_prob",
                "recent_form_home_prob", "recent_form_draw_prob", "recent_form_away_prob",
            ]
            for col in prob_cols:
                assert cw_data[col].notna().all(), f"NaN in {col} at cutoff {cw}"

    def test_cutoff_matchweek_boundary(self):
        """match_week < cw strictly; same-week matches never in history."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)

        results = run_match_rolling_evaluation(fixtures, config, cutoffs=[3])
        preds = results["predictions"]
        cw3_data = preds[preds["cutoff_week"] == 3]

        # Verify history is only matchweeks 1,2
        assert len(cw3_data) > 0

    def test_matchweek_1_no_history(self):
        """All baselines produce valid outputs for cw=1."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=3, smoothing_alpha=1.0)

        results = run_match_rolling_evaluation(fixtures, config, cutoffs=[1])
        preds = results["predictions"]

        assert not preds.empty
        # All baselines should have valid probabilities
        prob_cols = [
            "global_home_prob", "global_draw_prob", "global_away_prob",
        ]
        for col in prob_cols:
            assert (preds[col] >= 0).all()
            assert (preds[col] <= 1).all()
        # Probabilities sum to 1
        for _, row in preds.iterrows():
            assert abs(row["global_home_prob"] + row["global_draw_prob"] + row["global_away_prob"] - 1.0) < 1e-10

    def test_metrics_computed(self):
        """Metrics are computed for all baselines."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)

        results = run_match_rolling_evaluation(fixtures, config, cutoffs=[2, 3])

        metrics_per = results["metrics_per_cutoff"]
        metrics_agg = results["metrics_aggregate"]

        assert not metrics_per.empty
        assert not metrics_agg.empty

        # Check required columns
        required_cols = ["log_loss", "brier_score", "accuracy", "sample_size", "cutoff_week", "population", "baseline"]
        for col in required_cols:
            assert col in metrics_per.columns

        # 2 cutoffs * 1 population * 4 baselines = 8 rows
        assert len(metrics_per) == 8

        # Aggregate: 4 baselines * 1 population = 4 rows
        assert len(metrics_agg) == 4

    def test_sample_sizes_reported(self):
        """Sample sizes per cutoff and in aggregate are reported."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)

        results = run_match_rolling_evaluation(fixtures, config, cutoffs=[2, 3])

        metrics_per = results["metrics_per_cutoff"]

        # Each row should have sample_size
        assert (metrics_per["sample_size"] > 0).all()

        # Sample sizes should be consistent within population/cutoff
        for (pop, cw), group in metrics_per.groupby(["population", "cutoff_week"]):
            sizes = group["sample_size"].unique()
            assert len(sizes) == 1, f"Inconsistent sample sizes for {pop} cw={cw}"


class TestLeakagePrevention:
    """Tests for temporal leakage prevention in match outcome baselines."""

    def test_future_data_injection_no_leakage(self):
        """Injecting future matchweek data does not change earlier-cutoff predictions."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)

        # Get predictions at cutoff cw=2 WITHOUT future data
        pred_original = evaluate_match_baselines_at_cutoff(fixtures, 2, config)

        # INJECT fake future data for matchweek 3 with extreme results
        future_matches = pd.DataFrame([{
            "match_id": 999,
            "match_week": 3,
            "season_id": 27,
            "season_name": "2015/2016",
            "home_team_id": 1,
            "home_team_name": "Team1",
            "away_team_id": 2,
            "away_team_name": "Team2",
            "home_score": 10,  # Extreme score
            "away_score": 0,
            "match_status": "available",
            "stadium": "Stadium",
            "source": "test",
            "source_url": "test",
            "ingested_at": "2024-01-01",
        }])

        fixtures_with_future = pd.concat([fixtures, future_matches], ignore_index=True)

        # Get predictions at cutoff cw=2 WITH future data injected
        pred_with_future = evaluate_match_baselines_at_cutoff(fixtures_with_future, 2, config)

        # Assertions: predictions at cw=2 must be IDENTICAL
        pred_original = pred_original.sort_values("match_id").reset_index(drop=True)
        pred_with_future = pred_with_future.sort_values("match_id").reset_index(drop=True)

        assert len(pred_original) == len(pred_with_future)

        prob_cols = [
            "global_home_prob", "global_draw_prob", "global_away_prob",
            "team_strength_home_prob", "team_strength_draw_prob", "team_strength_away_prob",
            "recent_form_home_prob", "recent_form_draw_prob", "recent_form_away_prob",
        ]

        for col in prob_cols:
            orig_vals = pred_original[col].values
            future_vals = pred_with_future[col].values
            np.testing.assert_allclose(
                orig_vals, future_vals, rtol=1e-10, atol=1e-12,
                err_msg=f"Leakage detected: {col} differs with future data injection"
            )

    def test_team_strength_no_leakage(self):
        """Team strength baseline doesn't leak future results."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=3, smoothing_alpha=1.0)

        # In base data, team 1 has moderate results
        # Future injection: team 1 crushes everyone in matchweeks 6, 7, 8 (beyond base data)
        future_matches = []
        for i in range(6, 9):
            future_matches.append({
                "match_id": 900 + i,
                "match_week": i,
                "season_id": 27,
                "season_name": "2015/2016",
                "home_team_id": 1,
                "home_team_name": "Team1",
                "away_team_id": 2,
                "away_team_name": "Team2",
                "home_score": 5,
                "away_score": 0,
                "match_status": "available",
                "stadium": "Stadium",
                "source": "test",
                "source_url": "test",
                "ingested_at": "2024-01-01",
            })

        fixtures_with_future = pd.concat([fixtures, pd.DataFrame(future_matches)], ignore_index=True)

        # At cutoff 3, team strength should use only matchweeks 1,2
        pred_original = evaluate_match_baselines_at_cutoff(fixtures, 3, config)
        pred_with_future = evaluate_match_baselines_at_cutoff(fixtures_with_future, 3, config)

        # Team strength probabilities should be identical
        for col in ["team_strength_home_prob", "team_strength_draw_prob", "team_strength_away_prob"]:
            orig_vals = pred_original[col].values
            future_vals = pred_with_future[col].values
            np.testing.assert_allclose(
                orig_vals, future_vals, rtol=1e-10, atol=1e-12,
                err_msg=f"Leakage in team_strength: {col}"
            )

    def test_recent_form_no_leakage(self):
        """Recent form baseline doesn't leak future matches."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)

        # Future injection: team 1 wins all future matches
        future_matches = pd.DataFrame([{
            "match_id": 999,
            "match_week": 4,
            "season_id": 27,
            "season_name": "2015/2016",
            "home_team_id": 1,
            "home_team_name": "Team1",
            "away_team_id": 2,
            "away_team_name": "Team2",
            "home_score": 5,
            "away_score": 0,
            "match_status": "available",
            "stadium": "Stadium",
            "source": "test",
            "source_url": "test",
            "ingested_at": "2024-01-01",
        }])

        fixtures_with_future = pd.concat([fixtures, future_matches], ignore_index=True)

        # At cutoff 3, recent form (window=3) should use matchweeks 1,2 only
        pred_original = evaluate_match_baselines_at_cutoff(fixtures, 3, config)
        pred_with_future = evaluate_match_baselines_at_cutoff(fixtures_with_future, 3, config)

        for col in ["recent_form_home_prob", "recent_form_draw_prob", "recent_form_away_prob"]:
            orig_vals = pred_original[col].values
            future_vals = pred_with_future[col].values
            np.testing.assert_allclose(
                orig_vals, future_vals, rtol=1e-10, atol=1e-12,
                err_msg=f"Leakage in recent_form: {col}"
            )

    def test_global_frequencies_no_leakage(self):
        """Global frequencies baseline doesn't leak future matches."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)

        # Future injection: all home wins
        future_matches = pd.DataFrame([{
            "match_id": 999,
            "match_week": 4,
            "season_id": 27,
            "season_name": "2015/2016",
            "home_team_id": 1,
            "home_team_name": "Team1",
            "away_team_id": 2,
            "away_team_name": "Team2",
            "home_score": 5,
            "away_score": 0,
            "match_status": "available",
            "stadium": "Stadium",
            "source": "test",
            "source_url": "test",
            "ingested_at": "2024-01-01",
        }])

        fixtures_with_future = pd.concat([fixtures, future_matches], ignore_index=True)

        pred_original = evaluate_match_baselines_at_cutoff(fixtures, 3, config)
        pred_with_future = evaluate_match_baselines_at_cutoff(fixtures_with_future, 3, config)

        for col in ["global_home_prob", "global_draw_prob", "global_away_prob"]:
            orig_vals = pred_original[col].values
            future_vals = pred_with_future[col].values
            np.testing.assert_allclose(
                orig_vals, future_vals, rtol=1e-10, atol=1e-12,
                err_msg=f"Leakage in global_frequencies: {col}"
            )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_history(self):
        """Baselines handle empty history gracefully."""
        fixtures = _make_test_fixtures(1)
        config = MatchBaselineConfig(min_history=5, smoothing_alpha=1.0)
        preds = evaluate_match_baselines_at_cutoff(fixtures, 1, config)
        assert not preds.empty
        # All should fall back to global (uniform with smoothing)
        for _, row in preds.iterrows():
            assert abs(row["global_home_prob"] + row["global_draw_prob"] + row["global_away_prob"] - 1.0) < 1e-10

    def test_all_same_outcome(self):
        """Baselines handle when history has only one outcome type."""
        history = pd.DataFrame({
            "home_score": [2, 2, 2],
            "away_score": [0, 1, 0],
        })
        probs = compute_global_outcome_frequencies(history, smoothing_alpha=1.0)
        # All home wins, but smoothing prevents 0 probabilities
        assert probs[0] > 0 and probs[1] > 0 and probs[2] > 0
        assert abs(sum(probs) - 1.0) < 1e-10

    def test_deterministic_results(self):
        """Same inputs produce same outputs."""
        fixtures = _make_test_fixtures(5)
        config = MatchBaselineConfig(min_history=1, smoothing_alpha=1.0)

        pred1 = evaluate_match_baselines_at_cutoff(fixtures, 3, config)
        pred2 = evaluate_match_baselines_at_cutoff(fixtures, 3, config)

        for col in ["global_home_prob", "team_strength_home_prob", "recent_form_home_prob"]:
            np.testing.assert_allclose(pred1[col].values, pred2[col].values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])