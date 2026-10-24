"""Regularized Multinomial Logistic Regression for match outcome prediction.

Uses scikit-learn's LogisticRegression with multinomial loss and L2 regularization.
Separate models for different feature sets if needed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MatchModelConfig:
    """Configuration for match outcome model."""
    C: float = 1.0                    # Inverse of regularization strength (1/alpha)
    max_iter: int = 1000              # Maximum iterations for solver
    tol: float = 1e-4                 # Convergence tolerance
    min_train_samples: int = 50       # Minimum training samples required
    standardize: bool = True          # Whether to standardize features
    random_state: int | None = 42     # Random state for reproducibility
    class_weight: Literal["balanced", None] = "balanced"  # Handle class imbalance
    solver: Literal["lbfgs", "newton-cg", "sag", "saga"] = "lbfgs"  # Solver for multinomial


# Default feature columns to use (will be filtered based on availability)
DEFAULT_FEATURE_COLUMNS = [
    # Home team rolling features
    "home_matches",
    "home_wins",
    "home_draws",
    "home_losses",
    "home_points",
    "home_points_per_match",
    "home_goals_scored",
    "home_goals_conceded",
    "home_goals_scored_per_match",
    "home_goals_conceded_per_match",
    "home_goal_diff",
    "home_goal_diff_per_match",
    # Away team rolling features
    "away_matches",
    "away_wins",
    "away_draws",
    "away_losses",
    "away_points",
    "away_points_per_match",
    "away_goals_scored",
    "away_goals_conceded",
    "away_goals_scored_per_match",
    "away_goals_conceded_per_match",
    "away_goal_diff",
    "away_goal_diff_per_match",
    # Home team home/away specific
    "home_home_matches",
    "home_home_wins",
    "home_home_draws",
    "home_home_losses",
    "home_home_points",
    "home_home_points_per_match",
    "home_home_goals_scored",
    "home_home_goals_conceded",
    "home_home_goals_scored_per_match",
    "home_home_goals_conceded_per_match",
    "home_away_matches",
    "home_away_wins",
    "home_away_draws",
    "home_away_losses",
    "home_away_points",
    "home_away_points_per_match",
    "home_away_goals_scored",
    "home_away_goals_conceded",
    "home_away_goals_scored_per_match",
    "home_away_goals_conceded_per_match",
    # Away team home/away specific
    "away_home_matches",
    "away_home_wins",
    "away_home_draws",
    "away_home_losses",
    "away_home_points",
    "away_home_points_per_match",
    "away_home_goals_scored",
    "away_home_goals_conceded",
    "away_home_goals_scored_per_match",
    "away_home_goals_conceded_per_match",
    "away_away_matches",
    "away_away_wins",
    "away_away_draws",
    "away_away_losses",
    "away_away_points",
    "away_away_points_per_match",
    "away_away_goals_scored",
    "away_away_goals_conceded",
    "away_away_goals_scored_per_match",
    "away_away_goals_conceded_per_match",
    # Head-to-head
    "h2h_matches",
    "h2h_wins",
    "h2h_draws",
    "h2h_losses",
    "h2h_points",
    "h2h_points_per_match",
    "h2h_goals_scored",
    "h2h_goals_conceded",
    "h2h_goals_scored_per_match",
    "h2h_goals_conceded_per_match",
    "h2h_goal_diff",
    # Rest days
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    # Global fallbacks
    "global_home_goals_per_match",
    "global_away_goals_per_match",
    "global_home_xg_per_match",
    "global_away_xg_per_match",
    "global_home_win_rate",
    "global_draw_rate",
    "global_away_win_rate",
]


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
    # Only use feature columns that actually exist in the DataFrame
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        warnings.warn(f"Feature columns not found in data (will be skipped): {missing_cols}")

    if not available_cols:
        warnings.warn("No feature columns available in data")
        # Return dummy matrix with single column of zeros
        n = len(df)
        dummy = np.zeros((n, 1))
        return dummy, StandardScaler(), ["dummy"]

    X = df[available_cols].copy()

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median (computed from training data only if fit_scaler)
    if fit_scaler:
        medians = X.median()
        X = X.fillna(medians)
    else:
        # For test data, fill with 0 for any remaining NaN
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
        X = df[available_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        if fit_scaler:
            medians = X.median()
            X = X.fillna(medians)
        else:
            X = X.fillna(0)
        actual_feature_cols = [available_cols[0]]
        X = X[[available_cols[0]]]

    if scaler is not None:
        if fit_scaler:
            X_scaled = scaler.fit_transform(X)
        else:
            # For prediction, ensure we only use the columns the scaler was fitted on
            if fitted_feature_cols is not None:
                # Add missing columns with zeros
                for col in fitted_feature_cols:
                    if col not in X.columns:
                        X[col] = 0.0
                X = X[fitted_feature_cols]
            X_scaled = scaler.transform(X)
        return X_scaled, scaler, actual_feature_cols

    return X.values, StandardScaler(), actual_feature_cols


def _fit_multinomial_logistic(
    X: np.ndarray,
    y: np.ndarray,
    config: MatchModelConfig
) -> LogisticRegression:
    """Fit a regularized multinomial logistic regression model.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target labels (0=away, 1=draw, 2=home).
        config: MatchModelConfig.

    Returns:
        Fitted LogisticRegression model.
    """
    model = LogisticRegression(
        C=config.C,
        max_iter=config.max_iter,
        tol=config.tol,
        solver=config.solver,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=1,
        verbose=0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model.fit(X, y)

    return model


def _predict_multinomial_logistic(
    model: LogisticRegression,
    X: np.ndarray
) -> np.ndarray:
    """Generate probability predictions from fitted model.

    Args:
        model: Fitted LogisticRegression model.
        X: Feature matrix.

    Returns:
        Predicted probabilities of shape (n_samples, 3) for classes [0, 1, 2].
    """
    try:
        probs = model.predict_proba(X)
        # Get the classes the model was trained on
        classes = model.classes_
        
        # Ensure non-negative and handle any numerical issues
        probs = np.maximum(probs, 0.0)
        # Renormalize to sum to 1
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.maximum(row_sums, 1e-12)
        
        # Expand to 3 classes if model only learned 2 classes
        n_samples = len(X)
        full_probs = np.zeros((n_samples, 3))
        for i, cls in enumerate(classes):
            if cls < 3:  # Only map valid classes (0, 1, 2)
                full_probs[:, cls] = probs[:, i]
        
        # If any class is missing, distribute its probability mass evenly among present classes
        # or use uniform for missing classes
        missing_classes = [c for c in [0, 1, 2] if c not in classes]
        if missing_classes:
            # For missing classes, assign uniform probability among them
            missing_prob = 1.0 / (len(missing_classes) + len(classes))
            for cls in missing_classes:
                full_probs[:, cls] = missing_prob
            # Renormalize existing classes
            present_sum = full_probs[:, classes].sum(axis=1, keepdims=True)
            if np.any(present_sum > 0):
                full_probs[:, classes] = full_probs[:, classes] / np.maximum(present_sum, 1e-12) * (1 - len(missing_classes) * missing_prob)
        
        return full_probs
    except Exception:
        # Fallback: uniform probabilities
        n = len(X)
        return np.full((n, 3), 1.0 / 3)


class MatchModel:
    """Regularized Multinomial Logistic Regression for match outcome prediction.

    Fits a single model for three-class outcome prediction (away win, draw, home win).
    """

    def __init__(self, config: MatchModelConfig):
        self.config = config
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None
        self.feature_cols: list[str] = []
        self.fitted_feature_cols: list[str] = []
        self._is_fitted = False
        self._fallback_used = False
        self._feature_importance: np.ndarray | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_col: str = "outcome"
    ) -> "MatchModel":
        """Fit the model on training data.

        Args:
            train_df: DataFrame with features and target.
            feature_cols: List of feature column names to use.
            target_col: Target column name (default: "outcome").

        Returns:
            Self for chaining.
        """
        self.feature_cols = feature_cols if feature_cols is not None else DEFAULT_FEATURE_COLUMNS
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

        y = train_df[target_col].values.astype(int)

        # Check if we have at least 2 classes
        n_classes = len(np.unique(y))
        if n_classes < 2:
            warnings.warn(f"Only {n_classes} class(es) in training data. Fallback will be used.")
            self._fallback_used = True
            self._is_fitted = True
            return self

        # Fit model
        self.model = _fit_multinomial_logistic(X, y, self.config)

        # Store feature importance (coefficients)
        if hasattr(self.model, "coef_"):
            self._feature_importance = self.model.coef_

        self._is_fitted = True
        return self

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict outcome probabilities for test data.

        Args:
            test_df: DataFrame with feature columns.

        Returns:
            Predicted probabilities of shape (n_samples, 3) for classes [0=away, 1=draw, 2=home].
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # If fallback was used, return uniform probabilities
        if self._fallback_used or self.model is None:
            n = len(test_df)
            return np.full((n, 3), 1.0 / 3)

        # Prepare features using fitted scaler and fitted feature columns
        X, _, _ = _prepare_feature_matrix(
            test_df, self.feature_cols, self.scaler, fit_scaler=False,
            fitted_feature_cols=self.fitted_feature_cols
        )

        return _predict_multinomial_logistic(self.model, X)

    def get_fallback_status(self) -> bool:
        """Return whether fallback was used during fitting."""
        return self._fallback_used

    def get_feature_importance(self) -> np.ndarray | None:
        """Return feature importance (coefficients) if available."""
        return self._feature_importance

    def get_feature_names(self) -> list[str]:
        """Return names of features used in the fitted model."""
        return self.fitted_feature_cols


def train_model_at_cutoff(
    fixtures: pd.DataFrame,
    player_match: pd.DataFrame | None,
    cutoff_week: int,
    season_id: int | None,
    feature_config: MatchFeatureConfig,
    model_config: MatchModelConfig
) -> tuple[pd.DataFrame, bool]:
    """Train match model at a specific cutoff week and generate predictions.

    Args:
        fixtures: Full fixtures DataFrame.
        player_match: Player match data for xG aggregation.
        cutoff_week: Target matchweek.
        season_id: Season identifier.
        feature_config: MatchFeatureConfig for feature engineering.
        model_config: MatchModelConfig for model training.

    Returns:
        Tuple of (predictions DataFrame, fallback_used boolean).
    """
    from src.features.match_features import compute_match_features_for_cutoff

    # Get training data: all matchweeks before cutoff
    train_features = []
    for cw in range(1, cutoff_week):
        feats = compute_match_features_for_cutoff(
            fixtures, player_match, cw, season_id, feature_config
        )
        if not feats.empty:
            train_features.append(feats)

    if not train_features:
        # No training data
        test_features = compute_match_features_for_cutoff(
            fixtures, player_match, cutoff_week, season_id, feature_config
        )
        if test_features.empty:
            return pd.DataFrame(), True
        n = len(test_features)
        fallback_preds = test_features[["match_id", "home_team_id", "away_team_id",
                                         "home_score", "away_score", "outcome"]].copy()
        fallback_preds["model_home_prob"] = 1.0 / 3
        fallback_preds["model_draw_prob"] = 1.0 / 3
        fallback_preds["model_away_prob"] = 1.0 / 3
        fallback_preds["cutoff_week"] = cutoff_week
        fallback_preds["population"] = "all_matches"
        return fallback_preds, True

    train_df = pd.concat(train_features, ignore_index=True)

    # Get test data for this cutoff
    test_df = compute_match_features_for_cutoff(
        fixtures, player_match, cutoff_week, season_id, feature_config
    )
    if test_df.empty:
        return pd.DataFrame(), False

    # Fit model
    model = MatchModel(model_config)
    model.fit(train_df)

    # Predict
    probs = model.predict_proba(test_df)

    # Build predictions DataFrame
    preds = test_df[["match_id", "home_team_id", "away_team_id",
                      "home_score", "away_score", "outcome"]].copy()
    preds["model_home_prob"] = probs[:, 2]  # class 2 = home win
    preds["model_draw_prob"] = probs[:, 1]  # class 1 = draw
    preds["model_away_prob"] = probs[:, 0]  # class 0 = away win
    preds["cutoff_week"] = cutoff_week
    preds["population"] = "all_matches"

    fallback_used = model.get_fallback_status()
    return preds, fallback_used


def run_model_rolling_evaluation(
    fixtures: pd.DataFrame,
    player_match: pd.DataFrame | None,
    feature_config: MatchFeatureConfig,
    model_config: MatchModelConfig,
    cutoffs: list[int] | None = None,
    season_id: int | None = None
) -> dict[str, pd.DataFrame]:
    """Run rolling-origin evaluation for match model.

    Args:
        fixtures: Full fixtures DataFrame.
        player_match: Player match data for xG aggregation.
        feature_config: MatchFeatureConfig.
        model_config: MatchModelConfig.
        cutoffs: List of cutoff matchweeks.
        season_id: Season identifier.

    Returns:
        Dictionary with predictions and metrics.
    """
    if cutoffs is None:
        cutoffs = list(range(1, 39))

    all_predictions = []
    fallback_counts = 0

    for cw in cutoffs:
        preds, fallback = train_model_at_cutoff(
            fixtures, player_match, cw, season_id, feature_config, model_config
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