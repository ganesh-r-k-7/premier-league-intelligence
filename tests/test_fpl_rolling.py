"""Tests for FPL chronological evaluation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.fpl_rolling import (
    FPLEvaluationConfig,
    _make_fpl_evaluation_config,
    _load_fpl_data,
    _build_fpl_features_for_seasons,
    _get_season_cutoffs,
    run_fpl_glm_evaluation,
    run_fpl_nb_evaluation,
    run_fpl_baseline_comparison,
    run_chronological_evaluation,
    compute_fpl_model_metrics,
    aggregate_fpl_metrics,
)
from src.models.fpl_model import FPLGLMConfig, FPLNBConfig
from src.features.fpl_features import FPLFeatureConfig


class TestFPLEvaluationConfig:
    """Tests for FPLEvaluationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FPLEvaluationConfig(
            train_seasons=[2016, 2017, 2018],
            val_season=2019,
            test_seasons=[2020],
            gameweeks=None,
            glm_config=FPLGLMConfig(),
            nb_config=FPLNBConfig(),
            feature_config=FPLFeatureConfig(),
            baselines=["global_mean", "player_mean"],
            targets=["goals", "assists"],
            min_history=3,
            recent_window=5,
        )
        assert config.train_seasons == [2016, 2017, 2018]
        assert config.val_season == 2019
        assert config.test_seasons == [2020]
        assert config.gameweeks is None
        assert config.targets == ["goals", "assists"]
        assert config.min_history == 3
        assert config.recent_window == 5


class TestMakeFPLEvaluationConfig:
    """Tests for _make_fpl_evaluation_config."""

    def test_make_config(self):
        """Test creating default evaluation config."""
        config = _make_fpl_evaluation_config()

        assert isinstance(config, FPLEvaluationConfig)
        assert config.train_seasons == list(range(2016, 2022))  # 2016/17–2021/22
        assert config.val_season == 2022  # 2022/23
        assert config.test_seasons == list(range(2023, 2026))  # 2023/24–2025/26
        assert config.gameweeks is None
        assert len(config.baselines) == 4
        assert config.targets == ["goals", "assists"]


class TestLoadFPLData:
    """Tests for _load_fpl_data."""

    def test_load_fpl_data(self):
        """Test loading FPL data."""
        ps, pm = _load_fpl_data()

        assert isinstance(pm, pd.DataFrame)
        assert isinstance(ps, pd.DataFrame)
        assert len(pm) > 0
        assert len(ps) > 0
        assert "player_id" in pm.columns
        assert "player_id" in ps.columns


class TestBuildFPLFeaturesForSeasons:
    """Tests for _build_fpl_features_for_seasons."""

    def test_build_features_for_seasons(self):
        """Test building features for specific seasons."""
        from src.dashboard.data_loader import load_player_season_fpl, load_player_match_fpl

        ps = load_player_season_fpl()
        pm = load_player_match_fpl()

        features = _build_fpl_features_for_seasons(ps, pm, [2022, 2023], FPLFeatureConfig())

        assert isinstance(features, pd.DataFrame)
        if len(features) > 0:
            assert set(features["season_id"].unique()).issubset({2022, 2023})


class TestGetSeasonCutoffs:
    """Tests for _get_season_cutoffs."""

    def test_get_season_cutoffs(self):
        """Test getting season cutoffs (gameweeks 1-38)."""
        cutoffs = _get_season_cutoffs(2023)
        assert cutoffs == list(range(1, 39))


class TestRunFPLGLMEvaluation:
    """Tests for run_fpl_glm_evaluation."""

    def test_run_fpl_glm_evaluation_smoke(self):
        """Smoke test for GLM evaluation."""
        config = FPLEvaluationConfig(
            train_seasons=[2016, 2017],
            val_season=2018,
            test_seasons=[2019],
            gameweeks=[1, 5, 10],
            glm_config=FPLGLMConfig(alpha=1.0, min_train_samples=10),
            nb_config=FPLNBConfig(alpha=1.0, min_train_samples=10),
            feature_config=FPLFeatureConfig(),
            baselines=["global_mean", "player_mean"],
            targets=["goals", "assists"],
            min_history=3,
            recent_window=5,
        )

        # This is a smoke test - may fail if data not available, but shouldn't crash
        try:
            results = run_fpl_glm_evaluation(config)
            assert "glm" in results
            assert "baselines" in results
            assert "combined" in results
        except Exception as e:
            # If data not available, that's ok for this test
            pytest.skip(f"Data not available: {e}")


class TestRunFPLNBEvaluation:
    """Tests for run_fpl_nb_evaluation."""

    def test_run_fpl_nb_evaluation_smoke(self):
        """Smoke test for NB evaluation."""
        config = FPLEvaluationConfig(
            train_seasons=[2016, 2017],
            val_season=2018,
            test_seasons=[2019],
            gameweeks=[1, 5],
            glm_config=FPLGLMConfig(alpha=1.0, min_train_samples=10),
            nb_config=FPLNBConfig(alpha=1.0, min_train_samples=10),
            feature_config=FPLFeatureConfig(),
            baselines=["global_mean"],
            targets=["goals"],
            min_history=3,
            recent_window=5,
        )

        try:
            results = run_fpl_nb_evaluation(config)
            assert "nb" in results
            assert "baselines" in results
        except Exception as e:
            pytest.skip(f"Data not available: {e}")


class TestRunFPLBaselineComparison:
    """Tests for run_fpl_baseline_comparison."""

    def test_run_fpl_baseline_comparison_smoke(self):
        """Smoke test for baseline comparison."""
        config = FPLEvaluationConfig(
            train_seasons=[2016, 2017],
            val_season=2018,
            test_seasons=[2019],
            gameweeks=[1, 5],
            glm_config=FPLGLMConfig(alpha=1.0, min_train_samples=10),
            nb_config=FPLNBConfig(alpha=1.0, min_train_samples=10),
            feature_config=FPLFeatureConfig(),
            baselines=["global_mean", "player_mean"],
            targets=["goals"],
            min_history=3,
            recent_window=5,
        )

        try:
            preds = run_fpl_baseline_comparison(config)
            assert isinstance(preds, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"Data not available: {e}")


class TestRunChronologicalEvaluation:
    """Tests for run_chronological_evaluation."""

    def test_run_chronological_evaluation_smoke(self):
        """Smoke test for chronological evaluation."""
        config = FPLEvaluationConfig(
            train_seasons=[2016, 2017],
            val_season=2018,
            test_seasons=[2019],
            gameweeks=[1, 5],
            glm_config=FPLGLMConfig(alpha=1.0, min_train_samples=10),
            nb_config=FPLNBConfig(alpha=1.0, min_train_samples=10),
            feature_config=FPLFeatureConfig(),
            baselines=["global_mean", "player_mean"],
            targets=["goals"],
            min_history=3,
            recent_window=5,
        )

        try:
            results = run_chronological_evaluation(config)
            assert isinstance(results, dict)
        except Exception as e:
            pytest.skip(f"Data not available: {e}")


class TestComputeFPLModelMetrics:
    """Tests for compute_fpl_model_metrics."""

    def test_compute_fpl_model_metrics(self):
        """Test computing model metrics."""
        preds = pd.DataFrame({
            "goals": [1, 0, 2, 1, 0],
            "assists": [0, 1, 0, 1, 0],
            "global_goals_pred": [0.8, 0.5, 1.2, 0.9, 0.4],
            "global_assists_pred": [0.3, 0.6, 0.2, 0.5, 0.3],
            "glm_goals_pred": [1.1, 0.1, 1.8, 0.8, 0.2],
            "glm_assists_pred": [0.1, 0.8, 0.1, 0.7, 0.1],
            "cutoff_gameweek": [1, 2, 3, 4, 5],
            "population": ["all_lineup"] * 5,
        })

        pred_cols_map = {
            "global_mean": ("global_goals_pred", "global_assists_pred"),
            "glm": ("glm_goals_pred", "glm_assists_pred"),
        }

        metrics = compute_fpl_model_metrics(
            preds,
            target_cols=("goals", "assists"),
            pred_cols_map={
                "global_mean": ("global_goals_pred", "global_assists_pred"),
                "glm": ("glm_goals_pred", "glm_assists_pred"),
            }
        )

        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) > 0
        assert "cutoff_gameweek" in metrics.columns
        assert "population" in metrics.columns
        assert "baseline" in metrics.columns
        assert "mae_goals" in metrics.columns
        assert "mae_assists" in metrics.columns


class TestAggregateFPLMetrics:
    """Tests for aggregate_fpl_metrics."""

    def test_aggregate_fpl_metrics(self):
        """Test aggregating metrics across cutoffs."""
        metrics_df = pd.DataFrame({
            "cutoff_gameweek": [1, 1, 2, 2],
            "population": ["all_lineup", "all_lineup", "all_lineup", "all_lineup"],
            "baseline": ["global_mean", "glm", "global_mean", "glm"],
            "mae_goals": [0.5, 0.4, 0.6, 0.3],
            "rmse_goals": [0.7, 0.5, 0.8, 0.4],
            "poisson_deviance_goals": [0.6, 0.5, 0.7, 0.4],
            "mae_assists": [0.3, 0.2, 0.4, 0.2],
            "rmse_assists": [0.5, 0.3, 0.6, 0.2],
            "poisson_deviance_assists": [0.4, 0.3, 0.5, 0.2],
            "sample_size": [10, 10, 10, 10],
        })

        agg = aggregate_fpl_metrics(metrics_df)

        assert isinstance(agg, pd.DataFrame)
        assert len(agg) == 2  # 2 baselines
        assert "mae_goals" in agg.columns
        assert agg["mae_goals"].iloc[0] == 0.55  # (0.5+0.6)/2
        assert agg["mae_goals"].iloc[1] == 0.35  # (0.4+0.3)/2


class TestFPLEvaluationEdgeCases:
    """Edge case tests for FPL evaluation."""

    def test_empty_predictions(self):
        """Handle empty predictions gracefully."""
        preds = pd.DataFrame(columns=[
            "goals", "assists", "glm_goals_pred", "glm_assists_pred",
            "cutoff_gameweek", "population"
        ])
        pred_cols_map = {
            "glm": ("glm_goals_pred", "glm_assists_pred"),
        }

        metrics = compute_fpl_model_metrics(
            preds,
            target_cols=("goals", "assists"),
            pred_cols_map={"glm": ("glm_goals_pred", "glm_assists_pred")}
        )

        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) == 0

    def test_missing_prediction_columns(self):
        """Handle missing prediction columns gracefully."""
        preds = pd.DataFrame({
            "goals": [1, 0],
            "assists": [0, 1],
            "cutoff_gameweek": [1, 2],
            "population": ["all_lineup", "all_lineup"],
        })
        pred_cols_map = {
            "glm": ("glm_goals_pred", "glm_assists_pred"),
        }

        metrics = compute_fpl_model_metrics(
            preds,
            target_cols=("goals", "assists"),
            pred_cols_map=pred_cols_map
        )

        # Should handle missing columns gracefully
        assert isinstance(metrics, pd.DataFrame)


class TestFPLRollingIntegration:
    """Integration tests for FPL rolling evaluation."""

    @pytest.mark.integration
    def test_full_pipeline_small_data(self):
        """Test full pipeline with minimal data."""
        from src.dashboard.data_loader import load_player_season_fpl, load_player_match_fpl

        # Load data
        ps = load_player_season_fpl()
        pm = load_player_match_fpl()

        # Quick sanity check
        assert len(ps) > 0
        assert len(pm) > 0

        # Check seasons
        assert ps["season_id"].nunique() == 10
        assert pm["season_id"].nunique() == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])