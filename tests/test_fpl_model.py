"""Tests for FPL model module (src/models/fpl_model.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.models.fpl_model import (
    FPLGLMConfig,
    FPLNBConfig,
    FPLPlayerGLM,
    FPLPlayerNB,
    FPL_FEATURE_COLUMNS,
    _fit_glm,
    _fit_nb,
    _predict_glm,
    _predict_nb,
    _prepare_feature_matrix,
    _global_mean_baseline,
    _player_mean_baseline,
    _recent_form_baseline,
    _minutes_adjusted_baseline,
    run_fpl_baselines_at_cutoff,
    run_fpl_baselines_rolling,
)
from src.features.fpl_features import FPLFeatureConfig


def _make_synthetic_fpl_data(n_samples: int = 200) -> pd.DataFrame:
    """Create synthetic FPL training data with count targets."""
    np.random.seed(42)
    data = {
        "player_id": np.arange(n_samples) % 20 + 1,  # 20 unique players
        "gameweek": np.random.randint(1, 39, n_samples),  # Gameweek 1-38
        "goals_rolling_5": np.random.gamma(1, 0.3, n_samples),
        "assists_rolling_5": np.random.gamma(1, 0.2, n_samples),
        "xg_rolling_5": np.random.gamma(1, 0.15, n_samples),
        "xa_rolling_5": np.random.gamma(1, 0.1, n_samples),
        "minutes_rolling_5": np.random.normal(75, 15, n_samples).clip(0, 95),
        "shots_rolling_5": np.random.gamma(2, 0.5, n_samples),
        "shots_on_target_rolling_5": np.random.gamma(1, 0.5, n_samples),
        "key_passes_rolling_5": np.random.gamma(1, 0.3, n_samples),
        "threat_rolling_5": np.random.gamma(1, 0.2, n_samples),
        "creativity_rolling_5": np.random.gamma(1, 0.15, n_samples),
        "goals_season_mean": np.random.gamma(2, 5, n_samples),
        "assists_season_mean": np.random.gamma(1, 3, n_samples),
        "xg_season_mean": np.random.gamma(2, 3, n_samples),
        "xa_season_mean": np.random.gamma(1, 2, n_samples),
        "minutes_season_mean": np.random.normal(2000, 500, n_samples).clip(0, 3420),
        "starts_season_mean": np.random.normal(25, 10, n_samples).clip(0, 38),
        "home_matches": np.random.poisson(3, n_samples).astype(float),
        "home_wins": np.random.binomial(3, 0.5, n_samples).astype(float),
        "home_draws": np.random.binomial(3, 0.25, n_samples).astype(float),
        "home_losses": np.random.binomial(3, 0.25, n_samples).astype(float),
        "home_goals_scored": np.random.poisson(5, n_samples).astype(float),
        "home_goals_conceded": np.random.poisson(4, n_samples).astype(float),
        "away_matches": np.random.poisson(3, n_samples).astype(float),
        "away_wins": np.random.binomial(3, 0.3, n_samples).astype(float),
        "away_draws": np.random.binomial(3, 0.25, n_samples).astype(float),
        "away_losses": np.random.binomial(3, 0.45, n_samples).astype(float),
        "away_goals_scored": np.random.poisson(4, n_samples).astype(float),
        "away_goals_conceded": np.random.poisson(5, n_samples).astype(float),
        "is_home": np.random.binomial(1, 0.5, n_samples).astype(float),
        "starts": np.random.poisson(15, n_samples).astype(float),
        "starts_rate": np.random.beta(3, 1, n_samples),
        "position": np.random.choice(["GK", "DEF", "MID", "FWD"], n_samples),
        "team": np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd", "Spurs"], n_samples),
    }
    df = pd.DataFrame(data)

    # Generate count targets with some correlation to features
    rate_goals = 0.05 + 0.3 * df["goals_rolling_5"] + 0.1 * df["home_goals_scored"] + 0.05 * df["is_home"]
    rate_assists = 0.03 + 0.2 * df["assists_rolling_5"] + 0.1 * df["away_goals_scored"]
    rate_goals = np.maximum(rate_goals, 0.01)
    rate_assists = np.maximum(rate_assists, 0.01)

    df["goals"] = np.random.poisson(rate_goals, n_samples)
    df["assists"] = np.random.poisson(rate_assists, n_samples)

    # Add required columns for baseline functions
    df["gameweek"] = np.random.randint(1, 39, n_samples)
    df["minutes"] = np.random.normal(75, 15, n_samples).clip(0, 95).astype(int)
    df["is_home"] = np.random.binomial(1, 0.5, n_samples).astype(float)
    df["starts"] = np.random.poisson(15, n_samples).astype(float)
    df["starts_rate"] = np.random.beta(3, 1, n_samples)
    df["position"] = np.random.choice(["GK", "DEF", "MID", "FWD"], n_samples)
    df["team"] = np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd", "Spurs"], n_samples)

    # Generate count targets with some correlation to features
    rate_goals = 0.05 + 0.3 * df["goals_rolling_5"] + 0.1 * df["home_goals_scored"] + 0.05 * df["is_home"]
    rate_assists = 0.03 + 0.2 * df["assists_rolling_5"] + 0.1 * df["away_goals_scored"]
    rate_goals = np.maximum(rate_goals, 0.01)
    rate_assists = np.maximum(rate_assists, 0.01)

    df["goals"] = np.random.poisson(rate_goals, n_samples)
    df["assists"] = np.random.poisson(rate_assists, n_samples)

    return df


class TestPrepareFeatureMatrix:
    """Tests for _prepare_feature_matrix."""

    def test_prepare_basic(self):
        """Basic feature matrix preparation works."""
        df = _make_synthetic_fpl_data(50)
        X, scaler, cols = _prepare_feature_matrix(df, FPL_FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)

        assert X.shape[0] == 50
        # Non-numeric columns (position, team) are dropped, so 31 numeric features remain
        assert X.shape[1] == 31
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))
        assert cols == FPL_FEATURE_COLUMNS[:31]  # Only numeric columns remain

    def test_prepare_handles_missing(self):
        """Missing values are filled with median."""
        df = _make_synthetic_fpl_data(50)
        df.loc[0, "goals_rolling_5"] = np.nan
        df.loc[1, "assists_rolling_5"] = np.inf
        df.loc[2, "minutes_rolling_5"] = -np.inf

        X, scaler, cols = _prepare_feature_matrix(df, FPL_FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)

        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_prepare_constant_column_dropped(self):
        """Constant columns are dropped."""
        df = _make_synthetic_fpl_data(50)
        df["constant_feat"] = 1.0
        cols_list = FPL_FEATURE_COLUMNS + ["constant_feat"]

        X, scaler, cols = _prepare_feature_matrix(df, cols_list, StandardScaler(), fit_scaler=True)

        assert X.shape[1] == 31  # constant dropped, non-numeric cols (position, team) also dropped
        assert "constant_feat" not in cols

    def test_prepare_test_data_uses_fitted_scaler(self):
        """Test data uses fitted scaler, not refit."""
        train_df = _make_synthetic_fpl_data(100)
        test_df = _make_synthetic_fpl_data(20)

        scaler = StandardScaler()
        X_train, scaler, _ = _prepare_feature_matrix(train_df, FPL_FEATURE_COLUMNS, scaler, fit_scaler=True)
        X_test, _, _ = _prepare_feature_matrix(test_df, FPL_FEATURE_COLUMNS, scaler, fit_scaler=False)

        # Scaler should not be refit
        assert X_test.shape[1] == X_train.shape[1]


class TestGLMFitting:
    """Tests for _fit_glm and _predict_glm."""

    def test_fit_glm_basic(self):
        """Basic GLM fitting works."""
        df = _make_synthetic_fpl_data(100)
        X, _, _ = _prepare_feature_matrix(df, FPL_FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)
        y = df["goals"].values.astype(float)

        config = FPLGLMConfig(alpha=1.0, max_iter=1000, tol=1e-4)
        model = _fit_glm(X, y, config)

        assert model is not None
        assert hasattr(model, "coef_")

    def test_fit_glm_separate_targets(self):
        """Goals and assists fitted as separate models."""
        df = _make_synthetic_fpl_data(100)
        X, _, _ = _prepare_feature_matrix(df, FPL_FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)
        y_goals = df["goals"].values.astype(float)
        y_assists = df["assists"].values.astype(float)

        config = FPLGLMConfig(alpha=1.0)
        model_goals = _fit_glm(X, y_goals, config)
        model_assists = _fit_glm(X, y_assists, config)

        # Coefficients should differ
        assert not np.allclose(model_goals.coef_, model_assists.coef_)

    def test_predict_non_negative(self):
        """Predictions are non-negative."""
        df = _make_synthetic_fpl_data(100)
        X, _, _ = _prepare_feature_matrix(df, FPL_FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)
        y = df["goals"].values.astype(float)

        config = FPLGLMConfig(alpha=1.0)
        model = _fit_glm(X, y, config)

        preds = _predict_glm(model, X)
        assert np.all(preds >= 0)

    def test_predict_finite(self):
        """Predictions are finite."""
        df = _make_synthetic_fpl_data(100)
        X, _, _ = _prepare_feature_matrix(df, FPL_FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)
        y = df["goals"].values.astype(float)

        config = FPLGLMConfig(alpha=1.0)
        model = _fit_glm(X, y, config)

        preds = _predict_glm(model, X)
        assert np.all(np.isfinite(preds))

    def test_fit_with_zero_targets(self):
        """Model handles zero targets (common in goals/assists)."""
        df = _make_synthetic_fpl_data(100)
        df.loc[:80, "goals"] = 0  # Force many zeros

        X, _, _ = _prepare_feature_matrix(df, FPL_FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)
        y = df["goals"].values.astype(float)

        config = FPLGLMConfig(alpha=1.0)
        model = _fit_glm(X, y, config)

        preds = _predict_glm(model, X)
        assert np.all(preds >= 0)


class TestFPLPlayerGLM:
    """Tests for FPLPlayerGLM class."""

    def test_fit_and_predict(self):
        """End-to-end fit and predict."""
        df = _make_synthetic_fpl_data(200)
        config = FPLGLMConfig(alpha=1.0, min_train_samples=50)

        glm = FPLPlayerGLM(config)
        glm.fit(df)

        preds_goals, preds_assists = glm.predict(df.iloc[:50])

        assert len(preds_goals) == 50
        assert len(preds_assists) == 50
        assert np.all(preds_goals >= 0)
        assert np.all(preds_assists >= 0)
        assert not glm.get_fallback_status()

    def test_insufficient_training_data_fallback(self):
        """Fallback used when training data insufficient."""
        df = _make_synthetic_fpl_data(10)  # Less than min_train_samples
        config = FPLGLMConfig(alpha=1.0, min_train_samples=50)

        glm = FPLPlayerGLM(config)
        glm.fit(df)

        preds_goals, preds_assists = glm.predict(df)

        assert np.all(preds_goals == 0)
        assert np.all(preds_assists == 0)
        assert glm.get_fallback_status()

    def test_sparse_history_fallback(self):
        """Fallback when player has very sparse history in training."""
        df = _make_synthetic_fpl_data(50)
        df["goals"] = 0  # All zeros
        df["assists"] = 0

        config = FPLGLMConfig(alpha=1.0, min_train_samples=30)
        glm = FPLPlayerGLM(config)
        glm.fit(df)

        preds_goals, preds_assists = glm.predict(df)

        assert np.all(preds_goals >= 0)
        assert np.all(preds_assists >= 0)

    def test_deterministic_behavior(self):
        """Model behavior is deterministic with fixed random_state."""
        df = _make_synthetic_fpl_data(100)

        config1 = FPLGLMConfig(alpha=1.0, random_state=42)
        config2 = FPLGLMConfig(alpha=1.0, random_state=42)

        glm1 = FPLPlayerGLM(config1)
        glm1.fit(df)

        glm2 = FPLPlayerGLM(config2)
        glm2.fit(df)

        preds1_g, preds1_a = glm1.predict(df)
        preds2_g, preds2_a = glm2.predict(df)

        assert np.allclose(preds1_g, preds2_g)
        assert np.allclose(preds1_a, preds2_a)

    def test_missing_features_handled(self):
        """Missing features in test data are handled."""
        df = _make_synthetic_fpl_data(100)
        test_df = df.iloc[:20].copy()
        test_df.loc[0, "goals_rolling_5"] = np.nan
        test_df.loc[1, "assists_rolling_5"] = np.nan

        config = FPLGLMConfig(alpha=1.0)
        glm = FPLPlayerGLM(config)
        glm.fit(df)

        preds_goals, preds_assists = glm.predict(test_df)

        assert len(preds_goals) == 20
        assert np.all(preds_goals >= 0)

    def test_constant_features_dropped(self):
        """Constant features in training are dropped."""
        df = _make_synthetic_fpl_data(100)
        df["constant_feature"] = 1.0

        feature_cols = FPL_FEATURE_COLUMNS + ["constant_feature"]

        config = FPLGLMConfig(alpha=1.0)
        glm = FPLPlayerGLM(config)
        # Override feature_cols for this test
        glm.fit(df, feature_cols=feature_cols)

        preds_goals, preds_assists = glm.predict(df)
        assert np.all(preds_goals >= 0)


class TestFPLPlayerNB:
    """Tests for FPLPlayerNB (Negative Binomial) class."""

    def test_fit_and_predict(self):
        """End-to-end fit and predict for NB."""
        df = _make_synthetic_fpl_data(200)
        config = FPLNBConfig(alpha=1.0, min_train_samples=50)

        nb = FPLPlayerNB(config)
        nb.fit(df)

        preds_goals, preds_assists = nb.predict(df.iloc[:50])

        assert len(preds_goals) == 50
        assert len(preds_assists) == 50
        assert np.all(preds_goals >= 0)
        assert np.all(preds_assists >= 0)
        assert not nb.get_fallback_status()

    def test_insufficient_training_data_fallback(self):
        """Fallback used when training data insufficient."""
        df = _make_synthetic_fpl_data(10)
        config = FPLNBConfig(alpha=1.0, min_train_samples=50)

        nb = FPLPlayerNB(config)
        nb.fit(df)

        preds_goals, preds_assists = nb.predict(df)

        assert np.all(preds_goals == 0)
        assert np.all(preds_assists == 0)
        assert nb.get_fallback_status()

    def test_sparse_history_fallback(self):
        """Fallback when data has mostly zeros."""
        df = _make_synthetic_fpl_data(50)
        df["goals"] = 0
        df["assists"] = 0

        config = FPLNBConfig(alpha=1.0, min_train_samples=30)
        nb = FPLPlayerNB(config)
        nb.fit(df)

        preds_goals, preds_assists = nb.predict(df)

        assert np.all(preds_goals >= 0)
        assert np.all(preds_assists >= 0)

    def test_deterministic_behavior(self):
        """Model behavior is deterministic with fixed random_state."""
        df = _make_synthetic_fpl_data(100)

        config1 = FPLNBConfig(alpha=1.0, random_state=42)
        config2 = FPLNBConfig(alpha=1.0, random_state=42)

        nb1 = FPLPlayerNB(config1)
        nb1.fit(df)

        nb2 = FPLPlayerNB(config2)
        nb2.fit(df)

        preds1_g, preds1_a = nb1.predict(df)
        preds2_g, preds2_a = nb2.predict(df)

        assert np.allclose(preds1_g, preds2_g)
        assert np.allclose(preds1_a, preds2_a)

    def test_missing_features_handled(self):
        """Missing features in test data are handled."""
        df = _make_synthetic_fpl_data(100)
        test_df = df.iloc[:20].copy()
        test_df.loc[0, "goals_rolling_5"] = np.nan
        test_df.loc[1, "assists_rolling_5"] = np.nan

        config = FPLNBConfig(alpha=1.0)
        nb = FPLPlayerNB(config)
        nb.fit(df)

        preds_goals, preds_assists = nb.predict(test_df)

        assert len(preds_goals) == 20
        assert np.all(preds_goals >= 0)

    def test_get_dispersion(self):
        """Dispersion parameters are returned."""
        df = _make_synthetic_fpl_data(200)
        config = FPLNBConfig(alpha=1.0, min_train_samples=50)

        nb = FPLPlayerNB(config)
        nb.fit(df)

        disp_g, disp_a = nb.get_dispersion()
        assert isinstance(disp_g, float)
        assert isinstance(disp_a, float)
        assert disp_g >= 0
        assert disp_a >= 0


class TestBaselineModels:
    """Tests for baseline prediction models."""

    def test_global_mean_baseline(self):
        """Global mean baseline returns correct means."""
        df = _make_synthetic_fpl_data(100)
        mean_goals, mean_assists = _global_mean_baseline(df)

        assert abs(mean_goals - df["goals"].mean()) < 1e-10
        assert abs(mean_assists - df["assists"].mean()) < 1e-10

    def test_player_mean_baseline(self):
        """Player historical mean baseline."""
        train_df = _make_synthetic_fpl_data(100)
        test_df = _make_synthetic_fpl_data(20)

        # Ensure same player IDs in test
        test_df["player_id"] = train_df["player_id"].iloc[:20].values

        pred_goals, pred_assists = _player_mean_baseline(train_df, test_df)

        assert len(pred_goals) == 20
        assert len(pred_assists) == 20
        assert np.all(np.isfinite(pred_goals))
        assert np.all(np.isfinite(pred_assists))

    def test_recent_form_baseline(self):
        """Recent form baseline."""
        train_df = _make_synthetic_fpl_data(100)
        test_df = _make_synthetic_fpl_data(20)
        test_df["player_id"] = train_df["player_id"].iloc[:20].values

        pred_goals, pred_assists = _recent_form_baseline(train_df, test_df)

        assert len(pred_goals) == 20
        assert np.all(np.isfinite(pred_goals))

    def test_minutes_adjusted_baseline(self):
        """Minutes-adjusted baseline."""
        train_df = _make_synthetic_fpl_data(100)
        test_df = _make_synthetic_fpl_data(20)
        test_df["player_id"] = train_df["player_id"].iloc[:20].values
        test_df["position"] = train_df["position"].iloc[:20].values
        test_df["team"] = train_df["team"].iloc[:20].values

        pred_goals, pred_assists = _minutes_adjusted_baseline(train_df, test_df)

        assert len(pred_goals) == 20
        assert np.all(np.isfinite(pred_goals))


class TestRunFPLBaselinesAtCutoff:
    """Tests for run_fpl_baselines_at_cutoff."""

    def test_cutoff_with_no_training_data(self):
        """Cutoff with no training data returns global mean fallback."""
        from src.features.fpl_features import FPLFeatureConfig

        # Create minimal synthetic data for testing
        pm = pd.DataFrame({
            'player_id': [1, 2, 3],
            'season_id': [2023, 2023, 2023],
            'season_name': ['2023/24'] * 3,
            'gameweek': [1, 1, 1],
            'team': ['Arsenal', 'Chelsea', 'Liverpool'],
            'was_home': [True, False, True],
            'minutes': [90, 90, 90],
            'goals': [1, 0, 0],
            'assists': [0, 1, 0],
            'expected_goals': [0.5, 0.2, 0.1],
            'expected_assists': [0.1, 0.2, 0.05],
            'starts': [1, 1, 0],
            'was_home': [True, False, True],
            'team': ['Arsenal', 'Chelsea', 'Liverpool'],
            'team_h_score': [2, 1, 3],
            'team_a_score': [1, 2, 0],
            'goals': [1, 0, 0],
            'assists': [0, 1, 0],
        })

        ps = pd.DataFrame({
            'player_id': [1, 2, 3],
            'season_id': [2023, 2023, 2023],
            'season_name': ['2023/24'] * 3,
            'minutes': [900, 800, 700],
            'goals': [5, 3, 2],
            'assists': [2, 1, 0],
            'expected_goals': [5.5, 2.5, 1.0],
            'expected_assists': [2.0, 1.5, 0.5],
            'starts': [10, 8, 5],
            'position': ['FWD', 'MID', 'DEF'],
            'team': ['Arsenal', 'Chelsea', 'Liverpool'],
        })

        # Cutoff 1 has no training data (gameweek < 1), should fallback to global mean
        preds = run_fpl_baselines_at_cutoff(pm, ps, 1, 2023, FPLFeatureConfig())

        # Cutoff 1 has no training data, should fallback to global mean
        assert not preds.empty
        assert "global_goals_pred" in preds.columns
        assert "global_assists_pred" in preds.columns
        assert "population" in preds.columns

def test_cutoff_with_training_data():
        """Cutoff with training data computes all baselines."""
        import pandas as pd
        import numpy as np
    
        pm = pd.DataFrame({
            'player_id': [1, 2, 3, 1, 2, 3],
            'season_id': [2023, 2023, 2023, 2023, 2023, 2023],
            'season_name': ['2023/24'] * 6,
            'gameweek': [1, 1, 1, 20, 20, 20],
            'team': ['Arsenal', 'Chelsea', 'Liverpool'] * 2,
            'was_home': [True, False, True, True, False, True],
            'minutes': [90, 90, 90, 90, 90, 90],
            'goals': [1, 0, 0, 0, 1, 0],
            'assists': [0, 1, 0, 1, 0, 0],
            'expected_goals': [0.5, 0.2, 0.1, 0.3, 0.4, 0.2],
            'expected_assists': [0.1, 0.2, 0.05, 0.1, 0.3, 0.1],
            'starts': [1, 1, 0, 1, 1, 0],
            'was_home': [True, False, True, True, False, True],
            'team': ['Arsenal', 'Chelsea', 'Liverpool'] * 2,
            'team_h_score': [2, 1, 3, 1, 2, 2],
            'team_a_score': [1, 2, 0, 1, 1, 1],
            'goals': [1, 0, 0, 0, 1, 0],
            'assists': [0, 1, 0, 1, 0, 0],
        })
    
        ps = pd.DataFrame({
            'player_id': [1, 2, 3],
            'season_id': [2023, 2023, 2023],
            'season_name': ['2023/24'] * 3,
            'minutes': [900, 800, 700],
            'goals': [5, 3, 2],
            'assists': [2, 1, 0],
            'expected_goals': [5.5, 2.5, 1.0],
            'expected_assists': [2.0, 1.5, 0.5],
            'starts': [10, 8, 5],
            'position': ['FWD', 'MID', 'DEF'],
            'team': ['Arsenal', 'Chelsea', 'Liverpool'],
        })
    
        from src.features.fpl_features import FPLFeatureConfig
        preds = run_fpl_baselines_at_cutoff(pm, ps, 20, 2023, FPLFeatureConfig())

        assert not preds.empty
        assert "global_goals_pred" in preds.columns
        assert "player_mean_goals_pred" in preds.columns
        assert "recent_form_goals_pred" in preds.columns
        assert "minutes_adj_goals_pred" in preds.columns
        assert "cutoff_gameweek" in preds.columns
        assert "population" in preds.columns


class TestRunFPLBaselinesRolling:
    """Tests for run_fpl_baselines_rolling."""

    def test_multiple_cutoffs(self):
        """Rolling baselines across multiple cutoffs."""
        import pandas as pd
        import numpy as np

        pm = pd.DataFrame({
            'player_id': [1, 2, 3] * 4,
            'season_id': [2023] * 12,
            'season_name': ['2023/24'] * 12,
            'gameweek': [1, 1, 1, 5, 5, 5, 10, 10, 10, 20, 20, 20],
            'team': ['Arsenal', 'Chelsea', 'Liverpool'] * 4,
            'was_home': [True, False, True] * 4,
            'minutes': [90] * 12,
            'goals': [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            'assists': [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            'expected_goals': [0.5, 0.2, 0.1] * 4,
            'expected_assists': [0.1, 0.2, 0.05] * 4,
            'starts': [1, 1, 0] * 4,
            'was_home': [True, False, True] * 4,
            'team': ['Arsenal', 'Chelsea', 'Liverpool'] * 4,
            'team_h_score': [2, 1, 3] * 4,
            'team_a_score': [1, 2, 0] * 4,
            'goals': [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            'assists': [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        })

        ps = pd.DataFrame({
            'player_id': [1, 2, 3],
            'season_id': [2023, 2023, 2023],
            'season_name': ['2023/24'] * 3,
            'minutes': [900, 800, 700],
            'goals': [5, 3, 2],
            'assists': [2, 1, 0],
            'expected_goals': [5.5, 2.5, 1.0],
            'expected_assists': [2.0, 1.5, 0.5],
            'starts': [10, 8, 5],
            'position': ['FWD', 'MID', 'DEF'],
            'team': ['Arsenal', 'Chelsea', 'Liverpool'],
        })

        from src.features.fpl_features import FPLFeatureConfig
        preds = run_fpl_baselines_rolling(pm, ps, cutoffs=[1, 5, 10, 20], season_id=2023, feature_config=FPLFeatureConfig())

        assert not preds.empty
        assert preds["cutoff_gameweek"].nunique() == 4
        assert all(cw in preds["cutoff_gameweek"].values for cw in [1, 5, 10, 20])


class TestFPLBaselineEdgeCases:
    """Tests for edge cases in baseline computation."""

    def test_empty_training_data(self):
        """Handle empty training data gracefully."""
        df = pd.DataFrame(columns=["player_id", "goals", "assists", "minutes", "gameweek", "position", "team"])
        test_df = pd.DataFrame({"player_id": [1], "minutes": [90], "gameweek": [1], "position": ["FWD"], "team": ["Arsenal"]})

        pred_goals, pred_assists = _player_mean_baseline(df, test_df)

        # Should fall back to global mean (which will be NaN/0 for empty df)
        assert len(pred_goals) == 1

    def test_missing_player_in_training(self):
        """Test player not seen in training data."""
        train_df = pd.DataFrame({
            "player_id": [1, 2],
            "goals": [5, 3],
            "assists": [2, 1],
        })
        test_df = pd.DataFrame({"player_id": [3], "minutes": [90]})

        pred_goals, pred_assists = _player_mean_baseline(train_df, test_df)

        # Should fall back to global mean
        assert len(pred_goals) == 1
        assert pred_goals[0] == 4.0  # (5+3)/2
        assert pred_assists[0] == 1.5  # (2+1)/2


class TestFPLModelEdgeCases:
    """Edge case tests for FPL models."""

    def test_fpl_glm_with_all_zero_targets(self):
        """GLM handles all-zero targets."""
        df = _make_synthetic_fpl_data(100)
        df["goals"] = 0
        df["assists"] = 0

        config = FPLGLMConfig(alpha=1.0)
        X, _, _ = _prepare_feature_matrix(df, FPL_FEATURE_COLUMNS, StandardScaler(), fit_scaler=True)
        y = df["goals"].values.astype(float)

        model = _fit_glm(X, y, config)
        preds = _predict_glm(model, X)

        assert np.all(preds >= 0)

    def test_fpl_nb_with_all_zero_targets(self):
        """NB handles all-zero targets."""
        df = _make_synthetic_fpl_data(100)
        df["goals"] = 0
        df["assists"] = 0

        config = FPLNBConfig(alpha=1.0, min_train_samples=50)

        nb = FPLPlayerNB(config)
        nb.fit(df)

        preds_goals, preds_assists = nb.predict(df)

        assert np.all(preds_goals >= 0)
        assert np.all(preds_assists >= 0)

    def test_fpl_glm_constant_features(self):
        """GLM handles constant features correctly."""
        df = _make_synthetic_fpl_data(100)
        df["constant_feature"] = 1.0

        feature_cols = FPL_FEATURE_COLUMNS + ["constant_feature"]

        config = FPLGLMConfig(alpha=1.0)
        glm = FPLPlayerGLM(config)
        glm.fit(df, feature_cols=feature_cols)

        preds_goals, preds_assists = glm.predict(df)
        assert np.all(preds_goals >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])