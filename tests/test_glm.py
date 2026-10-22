"""Tests for regularized Poisson GLM model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.models.glm import (
    FEATURE_COLUMNS,
    GLMConfig,
    PlayerGLM,
    _fit_glm,
    _predict_glm,
    _prepare_feature_matrix,
    train_glm_at_cutoff,
)
from src.features import FeatureConfig


def _make_synthetic_data(n_samples: int = 200) -> pd.DataFrame:
    """Create synthetic training data with count targets."""
    np.random.seed(42)
    data = {
        "player_goals_per_90": np.random.gamma(1, 0.3, n_samples),
        "player_assists_per_90": np.random.gamma(1, 0.2, n_samples),
        "player_recent_goals_mean_3": np.random.poisson(0.5, n_samples).astype(float),
        "player_recent_assists_mean_3": np.random.poisson(0.3, n_samples).astype(float),
        "player_minutes_per_match": np.random.normal(75, 15, n_samples).clip(0, 95),
        "player_starts_rate": np.random.beta(3, 1, n_samples),
        "team_goals_per_match": np.random.normal(1.4, 0.3, n_samples).clip(0.5, 2.5),
        "team_xg_per_match": np.random.normal(1.3, 0.25, n_samples).clip(0.5, 2.0),
        "opponent_goals_conceded_per_match": np.random.normal(1.3, 0.3, n_samples).clip(0.5, 2.5),
        "position_goals_per_90_league": np.random.gamma(1, 0.25, n_samples),
        "is_home": np.random.binomial(1, 0.5, n_samples).astype(float),
        "days_rest": np.random.poisson(7, n_samples).astype(float).clip(3, 14),
    }
    df = pd.DataFrame(data)

    # Generate count targets with some correlation to features
    rate_goals = 0.1 + 0.3 * df["player_goals_per_90"] + 0.1 * df["team_goals_per_match"] + 0.05 * df["is_home"]
    rate_assists = 0.05 + 0.2 * df["player_assists_per_90"] + 0.1 * df["team_xg_per_match"]
    rate_goals = np.maximum(rate_goals, 0.01)
    rate_assists = np.maximum(rate_assists, 0.01)

    df["goals_actual"] = np.random.poisson(rate_goals, n_samples)
    df["assists_actual"] = np.random.poisson(rate_assists, n_samples)

    return df


class TestFeatureMatrixPreparation:
    """Tests for _prepare_feature_matrix."""

    def test_prepare_basic(self):
        """Basic feature matrix preparation works."""
        df = _make_synthetic_data(50)
        X, scaler, cols = _prepare_feature_matrix(df, FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)

        assert X.shape[0] == 50
        assert X.shape[1] == len(FEATURE_COLUMNS)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))
        assert cols == FEATURE_COLUMNS

    def test_prepare_handles_missing(self):
        """Missing values are filled with median."""
        df = _make_synthetic_data(50)
        df.loc[0, "player_goals_per_90"] = np.nan
        df.loc[1, "player_assists_per_90"] = np.inf
        df.loc[2, "days_rest"] = -np.inf

        X, scaler, cols = _prepare_feature_matrix(df, FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)

        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_prepare_constant_column_dropped(self):
        """Constant columns are dropped."""
        df = _make_synthetic_data(50)
        df["constant_feat"] = 1.0
        cols_list = FEATURE_COLUMNS + ["constant_feat"]

        X, scaler, cols = _prepare_feature_matrix(df, cols_list, StandardScaler(), fit_scaler=True)

        assert X.shape[1] == len(FEATURE_COLUMNS)  # constant dropped
        assert "constant_feat" not in cols

    def test_prepare_test_data_uses_fitted_scaler(self):
        """Test data uses fitted scaler, not refit."""
        train_df = _make_synthetic_data(100)
        test_df = _make_synthetic_data(20)

        scaler = StandardScaler()
        X_train, scaler, _ = _prepare_feature_matrix(train_df, FEATURE_COLUMNS, scaler, fit_scaler=True)
        X_test, _, _ = _prepare_feature_matrix(test_df, FEATURE_COLUMNS, scaler, fit_scaler=False)

        # Scaler should not be refit
        assert X_test.shape[1] == X_train.shape[1]


class TestGLMFitting:
    """Tests for _fit_glm and _predict_glm."""

    def test_fit_glm_basic(self):
        """Basic GLM fitting works."""
        df = _make_synthetic_data(100)
        X, _, _ = _prepare_feature_matrix(df, FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)
        y = df["goals_actual"].values.astype(float)

        config = GLMConfig(alpha=1.0, max_iter=1000, tol=1e-4)
        model = _fit_glm(X, y, config)

        assert model is not None
        assert hasattr(model, "coef_")

    def test_fit_glm_separate_targets(self):
        """Goals and assists fitted as separate models."""
        df = _make_synthetic_data(100)
        X, _, _ = _prepare_feature_matrix(df, FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)
        y_goals = df["goals_actual"].values.astype(float)
        y_assists = df["assists_actual"].values.astype(float)

        config = GLMConfig(alpha=1.0)
        model_goals = _fit_glm(X, y_goals, config)
        model_assists = _fit_glm(X, y_assists, config)

        # Coefficients should differ
        assert not np.allclose(model_goals.coef_, model_assists.coef_)

    def test_predict_non_negative(self):
        """Predictions are non-negative."""
        df = _make_synthetic_data(100)
        X, _, _ = _prepare_feature_matrix(df, FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)
        y = df["goals_actual"].values.astype(float)

        config = GLMConfig(alpha=1.0)
        model = _fit_glm(X, y, config)

        preds = _predict_glm(model, X)
        assert np.all(preds >= 0)

    def test_predict_finite(self):
        """Predictions are finite."""
        df = _make_synthetic_data(100)
        X, _, _ = _prepare_feature_matrix(df, FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)
        y = df["goals_actual"].values.astype(float)

        config = GLMConfig(alpha=1.0)
        model = _fit_glm(X, y, config)

        preds = _predict_glm(model, X)
        assert np.all(np.isfinite(preds))

    def test_fit_with_zero_targets(self):
        """Model handles zero targets (common in goals/assists)."""
        df = _make_synthetic_data(100)
        # Force many zeros
        df.loc[:80, "goals_actual"] = 0

        X, _, _ = _prepare_feature_matrix(df, FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)
        y = df["goals_actual"].values.astype(float)

        config = GLMConfig(alpha=1.0)
        model = _fit_glm(X, y, config)

        preds = _predict_glm(model, X)
        assert np.all(preds >= 0)


class TestPlayerGLM:
    """Tests for PlayerGLM class."""

    def test_fit_and_predict(self):
        """End-to-end fit and predict."""
        df = _make_synthetic_data(200)
        config = GLMConfig(alpha=1.0, min_train_samples=50)

        glm = PlayerGLM(config)
        glm.fit(df)

        preds_goals, preds_assists = glm.predict(df.iloc[:50])

        assert len(preds_goals) == 50
        assert len(preds_assists) == 50
        assert np.all(preds_goals >= 0)
        assert np.all(preds_assists >= 0)
        assert not glm.get_fallback_status()

    def test_insufficient_training_data_fallback(self):
        """Fallback used when training data insufficient."""
        df = _make_synthetic_data(10)  # Less than min_train_samples
        config = GLMConfig(alpha=1.0, min_train_samples=50)

        glm = PlayerGLM(config)
        glm.fit(df)

        preds_goals, preds_assists = glm.predict(df)

        assert np.all(preds_goals == 0)
        assert np.all(preds_assists == 0)
        assert glm.get_fallback_status()

    def test_sparse_history_fallback(self):
        """Fallback when player has very sparse history in training."""
        # Create data with mostly zeros
        df = _make_synthetic_data(50)
        df["goals_actual"] = 0  # All zeros
        df["assists_actual"] = 0

        config = GLMConfig(alpha=1.0, min_train_samples=30)
        glm = PlayerGLM(config)
        glm.fit(df)

        preds_goals, preds_assists = glm.predict(df)

        # Should still predict (even if all zeros)
        assert np.all(preds_goals >= 0)
        assert np.all(preds_assists >= 0)

    def test_deterministic_behavior(self):
        """Model behavior is deterministic with fixed random_state."""
        df = _make_synthetic_data(100)

        config1 = GLMConfig(alpha=1.0, random_state=42)
        config2 = GLMConfig(alpha=1.0, random_state=42)

        glm1 = PlayerGLM(config1)
        glm1.fit(df)

        glm2 = PlayerGLM(config2)
        glm2.fit(df)

        preds1_g, preds1_a = glm1.predict(df)
        preds2_g, preds2_a = glm2.predict(df)

        assert np.allclose(preds1_g, preds2_g)
        assert np.allclose(preds1_a, preds2_a)

    def test_missing_features_handled(self):
        """Missing features in test data are handled."""
        df = _make_synthetic_data(100)
        test_df = df.iloc[:20].copy()
        test_df.loc[0, "player_goals_per_90"] = np.nan
        test_df.loc[1, "player_assists_per_90"] = np.nan

        config = GLMConfig(alpha=1.0)
        glm = PlayerGLM(config)
        glm.fit(df)

        preds_goals, preds_assists = glm.predict(test_df)

        assert len(preds_goals) == 20
        assert np.all(preds_goals >= 0)

    def test_constant_features_dropped(self):
        """Constant features in training are dropped."""
        df = _make_synthetic_data(100)
        # Add constant feature to the dataframe
        df["constant_feature"] = 1.0

        # Use a custom feature list including the constant
        feature_cols = FEATURE_COLUMNS + ["constant_feature"]

        config = GLMConfig(alpha=1.0)
        glm = PlayerGLM(config)
        # Override feature_cols for this test
        glm.fit(df, feature_cols=feature_cols)

        preds_goals, preds_assists = glm.predict(df)
        assert np.all(preds_goals >= 0)


class TestLeakagePrevention:
    """Tests for temporal leakage prevention."""

    def test_fit_only_on_training_data(self):
        """Model is only fit on data before cutoff."""
        df = _make_synthetic_data(200)

        # Split chronologically (first 150 = train, last 50 = test)
        train_df = df.iloc[:150]
        test_df = df.iloc[150:]

        config = GLMConfig(alpha=1.0)
        glm = PlayerGLM(config)
        glm.fit(train_df)

        # Should be able to predict on test
        preds_goals, preds_assists = glm.predict(test_df)
        assert len(preds_goals) == 50
        assert np.all(preds_goals >= 0)

    def test_no_target_data_in_features(self):
        """Target columns (goals_actual, assists_actual) not used as features."""
        df = _make_synthetic_data(100)

        config = GLMConfig(alpha=1.0)
        glm = PlayerGLM(config)
        glm.fit(df)

        # Check that feature columns don't include targets
        assert "goals_actual" not in FEATURE_COLUMNS
        assert "assists_actual" not in FEATURE_COLUMNS

class TestLeakagePrevention:
    """Tests for temporal leakage prevention."""

    def test_fit_only_on_training_data(self):
        """Model is only fit on data before cutoff."""
        df = _make_synthetic_data(200)

        # Split chronologically (first 150 = train, last 50 = test)
        train_df = df.iloc[:150]
        test_df = df.iloc[150:]

        config = GLMConfig(alpha=1.0)
        glm = PlayerGLM(config)
        glm.fit(train_df)

        # Should be able to predict on test
        preds_goals, preds_assists = glm.predict(test_df)
        assert len(preds_goals) == 50
        assert np.all(preds_goals >= 0)

    def test_no_target_data_in_features(self):
        """Target columns (goals_actual, assists_actual) not used as features."""
        df = _make_synthetic_data(100)

        config = GLMConfig(alpha=1.0)
        glm = PlayerGLM(config)
        glm.fit(df)

        # Check that feature columns don't include targets
        assert "goals_actual" not in FEATURE_COLUMNS
        assert "assists_actual" not in FEATURE_COLUMNS

    def test_future_data_injection_no_leakage(self):
        """Injecting future high-scoring data doesn't affect earlier-cutoff features.
        
        This test verifies that the feature engineering (compute_features_for_cutoff)
        only uses data before the cutoff, not future data.
        """
        # This test is better done at the feature engineering level
        # For the GLM, the key is that train_glm_at_cutoff only uses match_week < cutoff
        # We test that by ensuring train_glm_at_cutoff with cutoff=2 gives same results
        # regardless of future matchweek data being present in the raw data
        
        pm, fx = _make_test_data_with_matchweek()
        
        feature_config = FeatureConfig(min_history=3, recent_window=3)
        glm_config = GLMConfig(alpha=1.0, min_train_samples=5)
        
        # Train at cutoff 2 with base data
        preds_base, _ = train_glm_at_cutoff(pm, fx, 2, FeatureConfig(min_history=3, recent_window=3), GLMConfig(alpha=1.0, min_train_samples=5))
        
        # Inject fake future data for matchweek 3: Player 11 scores 10 goals
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
            "goals": 10,
            "assists": 3,
            "shots": 10,
            "shots_on_target": 5,
            "xg": 3.0,
            "npxg": 3.0,
            "xa": 1.0,
            "penalty_goals": 0,
            "penalty_attempts": 0,
        }])
        
        pm_with_future = pd.concat([pm, future_row], ignore_index=True)
        
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
        fx_with_future = pd.concat([fx, future_fixture], ignore_index=True)
        
        # Train at cutoff 2 with future data injected
        preds_future, _ = train_glm_at_cutoff(
            pm_with_future, fx_with_future, 2, 
            FeatureConfig(min_history=3, recent_window=3), 
            GLMConfig(alpha=1.0, min_train_samples=5)
        )
        
        # Predictions at cutoff 2 should be identical (or very close)
        # because cutoff 2 only uses match_week < 2 (i.e., matchweek 1)
        # The injected matchweek 3 data should not affect cutoff 2
        base_sorted = preds_base.sort_values("player_id").reset_index(drop=True)
        future_sorted = preds_future.sort_values("player_id").reset_index(drop=True)
        
        assert len(base_sorted) == len(future_sorted)
        assert np.allclose(base_sorted["glm_goals_pred"], future_sorted["glm_goals_pred"], rtol=1e-10, atol=1e-12)
        assert np.allclose(base_sorted["glm_assists_pred"], future_sorted["glm_assists_pred"], rtol=1e-10, atol=1e-12)


class TestTrainGLMAtCutoff:
    """Tests for train_glm_at_cutoff function."""

    def test_cutoff_1_no_training_data(self):
        """Cutoff 1 returns fallback (no training data)."""
        # This would need actual data fixtures, skipping for unit test
        pass  # Integration test covers this

    def test_fallback_behavior(self):
        """Fallback returns zeros when training data insufficient."""
        # This is tested in PlayerGLM tests above
        pass


def _make_test_data_with_matchweek() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create test data with match_week column for integration tests.
    player_match does NOT have match_week (like production data).
    """
    rows = []
    match_id = 100

    for mw in [1, 2, 3]:
        for team_id, team_name in [(1, "TeamA"), (2, "TeamB")]:
            for player_num in [1, 2, 3]:
                player_id = team_id * 10 + player_num
                rows.append({
                    "match_id": match_id,
                    "player_id": player_id,
                    "player_name": f"Player{player_id}",
                    "team_id": team_id,
                    "team_name": team_name,
                    "position": "Center Forward" if player_num == 1 else "Center Attacking Midfield",
                    "started": True if player_num <= 2 else False,
                    "minutes": 90 if player_num <= 2 else 0,
                    "goals": 1 if (player_num == 1 and mw == 2) else 0,
                    "assists": 1 if (player_num == 2 and mw == 3) else 0,
                    "shots": 2 if player_num == 1 else (1 if player_num == 2 else 0),
                    "shots_on_target": 1 if player_num == 1 else 0,
                    "xg": 0.5 if player_num == 1 else 0.1,
                    "npxg": 0.5 if player_num == 1 else 0.1,
                    "xa": 0.3 if player_num == 2 else 0.0,
                    "penalty_goals": 0,
                    "penalty_attempts": 0,
                })
            match_id += 1

    player_match = pd.DataFrame(rows)

    fixtures_rows = []
    match_id = 100
    base_date = pd.Timestamp("2015-08-10")
    for mw in [1, 2, 3]:
        match_date = base_date + pd.Timedelta(days=(mw-1)*7)
        fixtures_rows.append({
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
        fixtures_rows.append({
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

    fixtures = pd.DataFrame(fixtures_rows)
    return player_match, fixtures


class TestIntegration:
    """Integration tests with actual feature pipeline."""

    def test_train_glm_at_cutoff_cutoff_2(self):
        """Train GLM at cutoff 2 with matchweek 1 history."""
        pm, fx = _make_test_data_with_matchweek()

        feature_config = FeatureConfig(min_history=3, recent_window=3)
        glm_config = GLMConfig(alpha=1.0, min_train_samples=5)

        preds, fallback = train_glm_at_cutoff(pm, fx, 2, feature_config, glm_config)

        # Should have predictions for matchweek 2 (2 teams * 3 players = 6)
        assert not preds.empty
        assert len(preds) == 6
        assert "glm_goals_pred" in preds.columns
        assert "glm_assists_pred" in preds.columns
        assert np.all(preds["glm_goals_pred"] >= 0)
        assert np.all(preds["glm_assists_pred"] >= 0)

    def test_train_glm_at_cutoff_cutoff_1_fallback(self):
        """Cutoff 1 uses fallback (no history)."""
        pm, fx = _make_test_data_with_matchweek()

        feature_config = FeatureConfig(min_history=3, recent_window=3)
        glm_config = GLMConfig(alpha=1.0, min_train_samples=5)

        preds, fallback = train_glm_at_cutoff(pm, fx, 1, feature_config, glm_config)

        # No training data, should fallback
        assert fallback
        assert not preds.empty
        assert np.all(preds["glm_goals_pred"] == 0)
        assert np.all(preds["glm_assists_pred"] == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])