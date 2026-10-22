"""Tests for metric computation (MAE, RMSE, Poisson deviance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_metrics,
    compute_metrics_per_cutoff,
    mean_absolute_error,
    poisson_deviance,
    root_mean_squared_error,
)


class TestMAE:
    """Tests for mean_absolute_error."""

    def test_mae_basic(self):
        """Basic MAE calculation."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0.1, 0.9, 1.8, 0.2, 1.1])
        result = mean_absolute_error(y_true, y_pred)
        expected = np.mean([0.1, 0.1, 0.2, 0.2, 0.1])
        assert np.isclose(result, expected)

    def test_mae_perfect(self):
        """MAE is 0 for perfect predictions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0.0, 1.0, 2.0])
        assert mean_absolute_error(y_true, y_pred) == 0.0

    def test_mae_empty_returns_nan(self):
        """Empty arrays return NaN (numpy behavior)."""
        result = mean_absolute_error(np.array([]), np.array([]))
        assert np.isnan(result)


class TestRMSE:
    """Tests for root_mean_squared_error."""

    def test_rmse_basic(self):
        """Basic RMSE calculation."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0.1, 0.9, 1.8, 0.2, 1.1])
        result = root_mean_squared_error(y_true, y_pred)
        expected = np.sqrt(np.mean([0.01, 0.01, 0.04, 0.04, 0.01]))
        assert np.isclose(result, expected)

    def test_rmse_perfect(self):
        """RMSE is 0 for perfect predictions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0.0, 1.0, 2.0])
        assert root_mean_squared_error(y_true, y_pred) == 0.0


class TestPoissonDeviance:
    """Tests for poisson_deviance."""

    def test_poisson_deviance_basic(self):
        """Basic Poisson deviance calculation."""
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0.1, 1.0, 2.0, 0.5])
        result = poisson_deviance(y_true, y_pred)
        # Manual: 2 * mean(y_pred - y_true + y_true * log(y_true/y_pred))
        # y=0, y_pred=0.1: 2*(0.1 - 0 + 0) = 0.2
        # y=1, y_pred=1.0: 2*(1.0 - 1 + 1*log(1)) = 0
        # y=2, y_pred=2.0: 2*(2.0 - 2 + 2*log(1)) = 0
        # y=0, y_pred=0.5: 2*(0.5 - 0 + 0) = 1.0
        # Mean = (0.2 + 0 + 0 + 1.0) / 4 = 0.3
        assert np.isclose(result, 0.3)

    def test_poisson_deviance_zero_predictions(self):
        """Poisson deviance handles y_pred=0 correctly with 0*log(0)=0 convention."""
        y_true = np.array([0, 0, 1, 2])
        y_pred = np.array([0.0, 0.0, 1.0, 2.0])
        # y=0, y_pred=0: term = 0 - 0 + 0 = 0
        # y=0, y_pred=0: term = 0
        # y=1, y_pred=1: term = 0
        # y=2, y_pred=2: term = 0
        result = poisson_deviance(y_true, y_pred)
        assert result == 0.0

    def test_poisson_deviance_zero_actual_with_positive_pred(self):
        """y_true=0, y_pred>0 gives y_pred term."""
        y_true = np.array([0, 0])
        y_pred = np.array([0.5, 1.0])
        # Each term: 2 * (y_pred - 0 + 0) = 2 * y_pred
        # Mean = (1.0 + 2.0) / 2 = 1.5
        result = poisson_deviance(y_true, y_pred)
        assert np.isclose(result, 1.5)

    def test_poisson_deviance_negative_prediction_returns_none(self):
        """Negative predictions return None and log warning."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0.1, -0.5, 2.0])
        result = poisson_deviance(y_true, y_pred, warn_on_invalid=False)
        assert result is None

    def test_poisson_deviance_nan_prediction_returns_none(self):
        """NaN predictions return None."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0.1, np.nan, 2.0])
        result = poisson_deviance(y_true, y_pred, warn_on_invalid=False)
        assert result is None

    def test_poisson_deviance_inf_prediction_returns_none(self):
        """Inf predictions return None."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0.1, np.inf, 2.0])
        result = poisson_deviance(y_true, y_pred, warn_on_invalid=False)
        assert result is None

    def test_poisson_deviance_empty_returns_none(self):
        """Empty arrays return None."""
        result = poisson_deviance(np.array([]), np.array([]), warn_on_invalid=False)
        assert result is None

    def test_poisson_deviance_length_mismatch_raises(self):
        """Length mismatch raises ValueError."""
        with pytest.raises(ValueError):
            poisson_deviance(np.array([0, 1]), np.array([0.1]))


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_compute_metrics_basic(self):
        """Basic metrics computation."""
        y_true_g = np.array([0, 1, 0, 2])
        y_pred_g = np.array([0.1, 0.9, 0.2, 1.8])
        y_true_a = np.array([0, 0, 1, 0])
        y_pred_a = np.array([0.0, 0.1, 1.1, 0.0])

        result = compute_metrics(y_true_g, y_pred_g, y_true_a, y_pred_a)

        assert "mae_goals" in result
        assert "rmse_goals" in result
        assert "poisson_deviance_goals" in result
        assert "mae_assists" in result
        assert "rmse_assists" in result
        assert "poisson_deviance_assists" in result
        assert "sample_size" in result
        assert result["sample_size"] == 4

    def test_compute_metrics_with_negative_preds_skips_poisson(self):
        """Negative predictions cause Poisson deviance to be None."""
        y_true_g = np.array([0, 1])
        y_pred_g = np.array([-0.1, 0.9])  # Negative!
        y_true_a = np.array([0, 0])
        y_pred_a = np.array([0.0, 0.1])

        result = compute_metrics(y_true_g, y_pred_g, y_true_a, y_pred_a)

        assert result["poisson_deviance_goals"] is None
        assert result["poisson_deviance_assists"] is not None


class TestComputeMetricsPerCutoff:
    """Tests for compute_metrics_per_cutoff."""

    def test_compute_metrics_per_cutoff(self):
        """Metrics computed per cutoff and baseline."""
        pred_data = pd.DataFrame({
            "cutoff_week": [1, 1, 2, 2],
            "population": ["all_lineup", "all_lineup", "all_lineup", "all_lineup"],
            "goals_actual": [0, 1, 0, 1],
            "assists_actual": [0, 0, 1, 0],
            "global_goals_pred": [0.1, 0.1, 0.1, 0.1],
            "global_assists_pred": [0.0, 0.0, 0.0, 0.0],
            "player_mean_goals_pred": [0.0, 1.0, 0.0, 1.0],
            "player_mean_assists_pred": [0.0, 0.0, 1.0, 0.0],
            "recent_form_goals_pred": [0.0, 0.0, 0.0, 0.0],
            "recent_form_assists_pred": [0.0, 0.0, 0.0, 0.0],
            "minutes_adj_goals_pred": [0.0, 0.0, 0.0, 0.0],
            "minutes_adj_assists_pred": [0.0, 0.0, 0.0, 0.0],
        })

        result = compute_metrics_per_cutoff(pred_data)

        assert len(result) == 8  # 2 cutoffs * 2 populations * 4 baselines (but only 1 pop here)
        assert list(result.columns) == [
            "mae_goals", "rmse_goals", "poisson_deviance_goals",
            "mae_assists", "rmse_assists", "poisson_deviance_assists",
            "sample_size", "cutoff_week", "population", "baseline"
        ]


class TestComputeAggregateMetrics:
    """Tests for compute_aggregate_metrics."""

    def test_compute_aggregate_metrics(self):
        """Aggregate metrics across all cutoffs."""
        pred_data = pd.DataFrame({
            "cutoff_week": [1, 1, 2, 2],
            "population": ["all_lineup", "played_only", "all_lineup", "played_only"],
            "goals_actual": [0, 1, 0, 1],
            "assists_actual": [0, 0, 1, 0],
            "global_goals_pred": [0.1, 0.1, 0.1, 0.1],
            "global_assists_pred": [0.0, 0.0, 0.0, 0.0],
            "player_mean_goals_pred": [0.0, 0.0, 0.0, 0.0],
            "player_mean_assists_pred": [0.0, 0.0, 0.0, 0.0],
            "recent_form_goals_pred": [0.0, 0.0, 0.0, 0.0],
            "recent_form_assists_pred": [0.0, 0.0, 0.0, 0.0],
            "minutes_adj_goals_pred": [0.0, 0.0, 0.0, 0.0],
            "minutes_adj_assists_pred": [0.0, 0.0, 0.0, 0.0],
        })

        result = compute_aggregate_metrics(pred_data)

        assert len(result) == 8  # 2 populations * 4 baselines
        assert "population" in result.columns
        assert "baseline" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])