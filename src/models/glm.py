"""Regularized Poisson GLM for player goal and assist prediction.

Uses scikit-learn's PoissonRegressor with L2 regularization.
Separate models for goals and assists.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GLMConfig:
    """Configuration for GLM model."""
    alpha: float = 1.0              # L2 regularization strength
    max_iter: int = 1000            # Maximum iterations for solver
    tol: float = 1e-4               # Convergence tolerance
    min_train_samples: int = 50     # Minimum training samples required
    standardize: bool = True        # Whether to standardize features
    random_state: int | None = 42   # Random state for reproducibility


# Features used for prediction (from Stage 3.1 feature engineering)
FEATURE_COLUMNS = [
    "player_goals_per_90",
    "player_assists_per_90",
    "player_recent_goals_mean_3",
    "player_recent_assists_mean_3",
    "player_minutes_per_match",
    "player_starts_rate",
    "team_goals_per_match",
    "team_xg_per_match",
    "opponent_goals_conceded_per_match",
    "position_goals_per_90_league",
    "is_home",
    "days_rest",
]


TARGET_COLUMNS = ["goals_actual", "assists_actual"]


def _prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
    fitted_feature_cols: list[str] | None = None
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Prepare feature matrix, handling missing values and constant columns.

    Returns:
        Tuple of (X_scaled, scaler, actual_feature_cols_used)
    """
    X = df[feature_cols].copy()

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median (computed from training data only if fit_scaler)
    if fit_scaler:
        medians = X.median()
        X = X.fillna(medians)
    else:
        # For test data, we'll fill with 0 for any remaining NaN
        X = X.fillna(0)

    # Handle constant columns (zero variance)
    actual_feature_cols = list(X.columns)
    if fit_scaler:
        variances = X.var()
        constant_cols = variances[variances == 0].index.tolist()
        if constant_cols:
            warnings.warn(f"Constant feature columns dropped: {constant_cols}")
            X = X.drop(columns=constant_cols)
            actual_feature_cols = [c for c in actual_feature_cols if c not in constant_cols]

    # If all features dropped, keep at least one to avoid empty matrix
    if len(X.columns) == 0:
        warnings.warn("All features were constant; keeping first feature as fallback")
        # Revert to original and keep first feature
        X = df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        if fit_scaler:
            medians = X.median()
            X = X.fillna(medians)
        else:
            X = X.fillna(0)
        actual_feature_cols = [feature_cols[0]]
        X = X[[feature_cols[0]]]

    if scaler is not None:
        if fit_scaler:
            X_scaled = scaler.fit_transform(X)
        else:
            # For prediction, ensure we only use the columns the scaler was fitted on
            if fitted_feature_cols is not None:
                X = X[fitted_feature_cols]
            X_scaled = scaler.transform(X)
        return X_scaled, scaler, actual_feature_cols

    return X.values, StandardScaler(), actual_feature_cols


def _fit_glm(
    X: np.ndarray,
    y: np.ndarray,
    config: GLMConfig
) -> PoissonRegressor:
    """Fit a regularized Poisson regression model."""
    # Ensure non-negative targets (Poisson requirement)
    y = np.maximum(y, 0)

    model = PoissonRegressor(
        alpha=config.alpha,
        max_iter=config.max_iter,
        tol=config.tol,
        solver="newton-cholesky",  # Handles L2 regularization
        warm_start=False,
        verbose=0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(X, y)

    return model


def _predict_glm(
    model: PoissonRegressor,
    X: np.ndarray
) -> np.ndarray:
    """Generate predictions, ensuring non-negative outputs."""
    preds = model.predict(X)
    return np.maximum(preds, 0.0)


class PlayerGLM:
    """Regularized Poisson GLM for player goal and assist prediction.

    Fits separate models for goals and assists.
    """

    def __init__(self, config: GLMConfig):
        self.config = config
        self.goals_model: PoissonRegressor | None = None
        self.assists_model: PoissonRegressor | None = None
        self.scaler: StandardScaler | None = None
        self.feature_cols: list[str] = []
        self.fitted_feature_cols: list[str] = []  # Columns actually used after preprocessing
        self._is_fitted = False
        self._fallback_used = False

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str] | None = None
    ) -> "PlayerGLM":
        """Fit GLM models for goals and assists on training data.

        Args:
            train_df: DataFrame with features and targets (goals_actual, assists_actual).
            feature_cols: List of feature column names to use.

        Returns:
            Self for chaining.
        """
        self.feature_cols = feature_cols if feature_cols is not None else FEATURE_COLUMNS
        n_samples = len(train_df)

        # Check minimum training samples
        if n_samples < self.config.min_train_samples:
            warnings.warn(
                f"Insufficient training samples ({n_samples} < {self.config.min_train_samples}). "
                "Fallback will be used."
            )
            self._fallback_used = True
            self._is_fitted = True
            return self

        # Prepare features and targets
        X, self.scaler, self.fitted_feature_cols = _prepare_feature_matrix(
            train_df, self.feature_cols, StandardScaler(with_mean=True, with_std=True), fit_scaler=True
        )

        # Check for constant features after scaling
        if X.shape[1] == 0:
            warnings.warn("No valid features after preprocessing. Fallback will be used.")
            self._fallback_used = True
            self._is_fitted = True
            return self

        y_goals = train_df["goals_actual"].values.astype(float)
        y_assists = train_df["assists_actual"].values.astype(float)

        # Fit goals model
        self.goals_model = _fit_glm(X, y_goals, self.config)

        # Fit assists model
        self.assists_model = _fit_glm(X, y_assists, self.config)

        self._is_fitted = True
        return self

    def predict(self, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict expected goals and assists for test data.

        Args:
            test_df: DataFrame with feature columns.

        Returns:
            Tuple of (goals_predictions, assists_predictions) as non-negative floats.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # If fallback was used, return zeros
        if self._fallback_used or self.goals_model is None or self.assists_model is None:
            n = len(test_df)
            return np.zeros(n), np.zeros(n)

        # Prepare features using fitted scaler and fitted feature columns
        X, _, _ = _prepare_feature_matrix(
            test_df, self.feature_cols, self.scaler, fit_scaler=False,
            fitted_feature_cols=self.fitted_feature_cols
        )

        goals_pred = _predict_glm(self.goals_model, X)
        assists_pred = _predict_glm(self.assists_model, X)

        return goals_pred, assists_pred

    def get_fallback_status(self) -> bool:
        """Return whether fallback was used during fitting."""
        return self._fallback_used


def train_glm_at_cutoff(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    cutoff_week: int,
    feature_config: FeatureConfig,
    glm_config: GLMConfig
) -> tuple[pd.DataFrame, bool]:
    """Train GLM at a specific cutoff week and generate predictions.

    Args:
        player_match: Full player_match DataFrame.
        fixtures: Full fixtures DataFrame.
        cutoff_week: Target matchweek.
        feature_config: FeatureConfig for feature engineering.
        glm_config: GLMConfig for model training.

    Returns:
        Tuple of (predictions DataFrame, fallback_used boolean).
    """
    from src.features import compute_features_for_cutoff

    # Get training data: all matchweeks before cutoff
    train_features = []
    for cw in range(1, cutoff_week):
        feats = compute_features_for_cutoff(player_match, fixtures, cw, feature_config)
        if not feats.empty:
            train_features.append(feats)

    if not train_features:
        # No training data (cutoff_week=1)
        test_features = compute_features_for_cutoff(player_match, fixtures, cutoff_week, feature_config)
        if test_features.empty:
            return pd.DataFrame(), True
        n = len(test_features)
        fallback_preds = test_features[["match_id", "player_id", "player_name", "team_id", "position",
                                         "minutes_actual", "goals_actual", "assists_actual"]].copy()
        fallback_preds["glm_goals_pred"] = 0.0
        fallback_preds["glm_assists_pred"] = 0.0
        fallback_preds["cutoff_week"] = cutoff_week
        fallback_preds["population"] = "all_lineup"
        return fallback_preds, True

    train_df = pd.concat(train_features, ignore_index=True)

    # Get test data for this cutoff
    test_df = compute_features_for_cutoff(player_match, fixtures, cutoff_week, feature_config)
    if test_df.empty:
        return pd.DataFrame(), False

    # Fit GLM
    glm = PlayerGLM(glm_config)
    glm.fit(train_df)

    # Predict
    goals_pred, assists_pred = glm.predict(test_df)

    # Build predictions DataFrame
    preds = test_df[["match_id", "player_id", "player_name", "team_id", "position",
                      "minutes_actual", "goals_actual", "assists_actual"]].copy()
    preds["glm_goals_pred"] = goals_pred
    preds["glm_assists_pred"] = assists_pred
    preds["cutoff_week"] = cutoff_week
    preds["population"] = "all_lineup"

    fallback_used = glm.get_fallback_status()
    return preds, fallback_used


def run_glm_rolling_evaluation(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    feature_config: FeatureConfig,
    glm_config: GLMConfig,
    cutoffs: list[int] | None = None
) -> dict[str, pd.DataFrame]:
    """Run rolling-origin evaluation for GLM model.

    Args:
        player_match: Full player_match DataFrame.
        fixtures: Full fixtures DataFrame.
        feature_config: FeatureConfig.
        glm_config: GLMConfig.
        cutoffs: List of cutoff matchweeks.

    Returns:
        Dictionary with GLM predictions and metrics.
    """
    if cutoffs is None:
        cutoffs = list(range(1, 39))

    all_predictions = []
    fallback_counts = 0

    for cw in cutoffs:
        preds, fallback = train_glm_at_cutoff(
            player_match, fixtures, cw, feature_config, glm_config
        )
        if not preds.empty:
            all_predictions.append(preds)
        if fallback:
            fallback_counts += 1

    if not all_predictions:
        return {"predictions": pd.DataFrame(), "fallback_counts": 0}

    predictions = pd.concat(all_predictions, ignore_index=True)

    return {
        "predictions": predictions,
        "fallback_counts": fallback_counts,
    }


def evaluate_glm_vs_baselines(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    feature_config: FeatureConfig,
    baseline_config: BaselineConfig,
    glm_config: GLMConfig,
    cutoffs: list[int] | None = None
) -> pd.DataFrame:
    """Evaluate GLM against Phase 2 baselines on identical observations.

    Args:
        player_match: Full player_match DataFrame.
        fixtures: Full fixtures DataFrame.
        feature_config: FeatureConfig for feature engineering.
        baseline_config: BaselineConfig for baselines.
        cutoffs: List of cutoff matchweeks.

    Returns:
        DataFrame with metrics for GLM and all baselines.
    """
    from src.features.baselines import BaselineConfig, evaluate_baselines_at_cutoff
    from src.evaluation.metrics import compute_metrics_per_cutoff

    if cutoffs is None:
        cutoffs = list(range(1, 39))

    # Merge match_week into player_match for baselines
    pm_with_week = player_match.merge(
        fixtures[["match_id", "match_week"]],
        on="match_id",
        how="left",
        validate="m:1"
    )

    all_glm_preds = []
    all_baseline_preds = []

    for cw in cutoffs:
        # GLM predictions
        glm_preds, _ = train_glm_at_cutoff(
            player_match, fixtures, cw, feature_config, glm_config
        )
        if not glm_preds.empty:
            all_glm_preds.append(glm_preds)

        # Baseline predictions (Population A)
        base_preds = evaluate_baselines_at_cutoff(
            pm_with_week, fixtures, cw, baseline_config, "all_lineup"
        )
        if not base_preds.empty:
            all_baseline_preds.append(base_preds)

    if not all_glm_preds or not all_baseline_preds:
        return pd.DataFrame()

    glm_all = pd.concat(all_glm_preds, ignore_index=True)
    base_all = pd.concat(all_baseline_preds, ignore_index=True)

    # Ensure same rows for comparison
    # Merge on match_id, player_id
    merged = base_all.merge(
        glm_all[["match_id", "player_id", "glm_goals_pred", "glm_assists_pred"]],
        on=["match_id", "player_id"],
        how="inner"
    )

    if merged.empty:
        return pd.DataFrame()

    # Add GLM to the predictions DataFrame for metrics computation
    merged["glm_goals_pred_col"] = merged["glm_goals_pred"]
    merged["glm_assists_pred_col"] = merged["glm_assists_pred"]

    # Compute metrics including GLM
    from src.evaluation.metrics import compute_metrics_per_cutoff

    pred_cols_map = {
        "global_mean": ("global_goals_pred", "global_assists_pred"),
        "player_mean": ("player_mean_goals_pred", "player_mean_assists_pred"),
        "recent_form": ("recent_form_goals_pred", "recent_form_assists_pred"),
        "minutes_adjusted": ("minutes_adj_goals_pred", "minutes_adj_assists_pred"),
        "glm": ("glm_goals_pred", "glm_assists_pred"),
    }

    metrics = compute_metrics_per_cutoff(
        merged,
        target_cols=("goals_actual", "assists_actual"),
        pred_cols_map=pred_cols_map
    )

    return metrics