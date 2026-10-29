"""FPL-specific models for multi-season player performance forecasting.

Adapts the existing GLM infrastructure for FPL player-match (gameweek) data.
Implements Poisson GLM, Negative Binomial GLM, and baseline models for FPL data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm = None


@dataclass(frozen=True)
class FPLGLMConfig:
    """Configuration for FPL GLM model."""
    alpha: float = 1.0              # L2 regularization strength
    max_iter: int = 1000            # Maximum iterations for solver
    tol: float = 1e-4               # Convergence tolerance
    min_train_samples: int = 50     # Minimum training samples required
    standardize: bool = True        # Whether to standardize features
    random_state: int | None = 42   # Random state for reproducibility


@dataclass(frozen=True)
class FPLNBConfig:
    """Configuration for FPL Negative Binomial GLM model."""
    alpha: float = 1.0              # L2 regularization strength
    max_iter: int = 1000            # Maximum iterations for solver
    gtol: float = 1e-4              # Convergence tolerance (gradient norm for BFGS)
    min_train_samples: int = 50     # Minimum training samples required
    standardize: bool = True        # Whether to standardize features
    random_state: int | None = 42   # Random state for reproducibility
    dispersion_method: Literal["ml", "pearson"] = "ml"
    alpha_dispersion: float | None = None


# FPL-specific feature columns (adapted from StatsBomb features)
FPL_FEATURE_COLUMNS = [
    # Player rolling form (gameweek-level)
    "goals_rolling_5",
    "assists_rolling_5",
    "xg_rolling_5",
    "xa_rolling_5",
    "minutes_rolling_5",
    "shots_rolling_5",
    "shots_on_target_rolling_5",
    "key_passes_rolling_5",
    "threat_rolling_5",
    "creativity_rolling_5",
    # Prior-season statistics
    "goals_season_mean",
    "assists_season_mean",
    "xg_season_mean",
    "xa_season_mean",
    "minutes_season_mean",
    "starts_season_mean",
    # Team strength features
    "home_matches",
    "home_wins",
    "home_draws",
    "home_losses",
    "home_goals_scored",
    "home_goals_conceded",
    "away_matches",
    "away_wins",
    "away_draws",
    "away_losses",
    "away_goals_scored",
    "away_goals_conceded",
    # Context features
    "is_home",
    "starts",
    "starts_rate",
    # Position/team (when available)
    "position",
    "team",
]

# Target columns
FPL_TARGET_COLUMNS = ["goals", "assists"]


def _prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
    fitted_feature_cols: list[str] | None = None
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Prepare feature matrix, handling missing values and constant columns.

    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        scaler: StandardScaler instance.
        fit_scaler: Whether to fit the scaler.
        fitted_feature_cols: Feature columns from fitting.

    Returns:
        Tuple of (X_scaled, scaler, actual_feature_cols_used).
    """
    # Only use feature columns that actually exist in the DataFrame
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        warnings.warn(f"Feature columns not found in data (will be skipped): {missing_cols}")

    if not available_cols:
        warnings.warn("No feature columns available in data")
        n = len(df)
        dummy = np.zeros((n, 1))
        return dummy, StandardScaler(), ["dummy"]

    X = df[available_cols].copy()

    # Only keep numeric columns for feature matrix
    X = X.select_dtypes(include=[np.number])

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
        X = X.select_dtypes(include=[np.number])
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


def _fit_glm(
    X: np.ndarray,
    y: np.ndarray,
    config: "FPLGLMConfig"
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


def _fit_nb(
    X: np.ndarray,
    y: np.ndarray,
    config: "FPLNBConfig"
) -> tuple[object, float]:
    """Fit a Negative Binomial regression model using statsmodels."""
    if not STATSMODELS_AVAILABLE:
        raise RuntimeError("statsmodels is required for Negative Binomial GLM")

    # Ensure non-negative targets
    y = np.maximum(y, 0)

    # Add constant for intercept
    X_with_const = sm.add_constant(X, has_constant='add')

    # Fit Negative Binomial model
    model = sm.NegativeBinomial(y, X_with_const)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            fitted = model.fit(
                maxiter=config.max_iter,
                method='bfgs',
                gtol=config.gtol,
                disp=False
            )
        except Exception as e:
            warnings.warn(f"NB fit failed: {e}, falling back to Poisson")
            poisson_model = PoissonRegressor(
                alpha=config.alpha,
                max_iter=config.max_iter,
                tol=config.gtol,
                solver="newton-cholesky",
                warm_start=False,
                verbose=0
            )
            poisson_model.fit(X, y)
            return poisson_model, 0.0

    # Extract dispersion parameter (alpha in NB2 parameterization)
    dispersion = getattr(fitted, 'alpha', 0.0)
    if dispersion is None or np.isnan(dispersion) or dispersion <= 0:
        dispersion = 0.0

    return fitted, float(dispersion)


def _predict_nb(
    model: object,
    X: np.ndarray
) -> np.ndarray:
    """Generate predictions from fitted Negative Binomial model."""
    if not STATSMODELS_AVAILABLE:
        return np.zeros(len(X))

    X_with_const = sm.add_constant(X, has_constant='add')
    try:
        preds = model.predict(X_with_const)
        return np.maximum(np.asarray(preds, dtype=float), 0.0)
    except Exception:
        return np.zeros(len(X))


@dataclass(frozen=True)
class FPLGLMConfig:
    """Configuration for FPL GLM model."""
    alpha: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-4
    min_train_samples: int = 50
    standardize: bool = True
    random_state: int | None = 42


@dataclass(frozen=True)
class FPLNBConfig:
    """Configuration for FPL Negative Binomial GLM model."""
    alpha: float = 1.0
    max_iter: int = 1000
    gtol: float = 1e-4
    min_train_samples: int = 50
    standardize: bool = True
    random_state: int | None = 42
    dispersion_method: Literal["ml", "pearson"] = "ml"
    alpha_dispersion: float | None = None


class FPLPlayerGLM:
    """Regularized Poisson GLM for FPL player goal and assist prediction."""

    def __init__(self, config: "FPLGLMConfig"):
        self.config = config
        self.goals_model: PoissonRegressor | None = None
        self.assists_model: PoissonRegressor | None = None
        self.scaler: StandardScaler | None = None
        self.feature_cols: list[str] = []
        self.fitted_feature_cols: list[str] = []
        self._is_fitted = False
        self._fallback_used = False

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_goals: str = "goals",
        target_assists: str = "assists"
    ) -> "FPLPlayerGLM":
        """Fit GLM models for goals and assists on training data."""
        self.feature_cols = feature_cols if feature_cols is not None else FPL_FEATURE_COLUMNS
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

        y_goals = train_df["goals"].values.astype(float)
        y_assists = train_df["assists"].values.astype(float)

        # Fit goals model
        self.goals_model = _fit_glm(X, y_goals, self.config)

        # Fit assists model
        self.assists_model = _fit_glm(X, y_assists, self.config)

        self._is_fitted = True
        return self

    def predict(self, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict expected goals and assists for test data."""
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


class FPLPlayerNB:
    """Regularized Negative Binomial GLM for FPL player goal and assist prediction."""

    def __init__(self, config: "FPLNBConfig"):
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels is required for Negative Binomial GLM")
        self.config = config
        self.goals_model: object | None = None
        self.assists_model: object | None = None
        self.goals_dispersion: float = 0.0
        self.assists_dispersion: float = 0.0
        self.scaler: StandardScaler | None = None
        self.feature_cols: list[str] = []
        self.fitted_feature_cols: list[str] = []
        self._is_fitted = False
        self._fallback_used = False

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_goals: str = "goals",
        target_assists: str = "assists"
    ) -> "FPLPlayerNB":
        """Fit NB models for goals and assists on training data."""
        self.feature_cols = feature_cols if feature_cols is not None else FPL_FEATURE_COLUMNS
        n_samples = len(train_df)

        if n_samples < self.config.min_train_samples:
            warnings.warn(
                f"Insufficient training samples ({n_samples} < {self.config.min_train_samples}). "
                "Fallback will be used."
            )
            self._fallback_used = True
            self._is_fitted = True
            return self

        X, self.scaler, self.fitted_feature_cols = _prepare_feature_matrix(
            train_df, self.feature_cols, StandardScaler(with_mean=True, with_std=True), fit_scaler=True
        )

        if X.shape[1] == 0:
            warnings.warn("No valid features after preprocessing. Fallback will be used.")
            self._fallback_used = True
            self._is_fitted = True
            return self

        y_goals = train_df["goals"].values.astype(float)
        y_assists = train_df["assists"].values.astype(float)

        # Fit goals model
        self.goals_model, self.goals_dispersion = _fit_nb(X, y_goals, self.config)

        # Fit assists model
        self.assists_model, self.assists_dispersion = _fit_nb(X, y_assists, self.config)

        self._is_fitted = True
        return self

    def predict(self, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict expected goals and assists for test data."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self._fallback_used or self.goals_model is None or self.assists_model is None:
            n = len(test_df)
            return np.zeros(n), np.zeros(n)

        X, _, _ = _prepare_feature_matrix(
            test_df, self.feature_cols, self.scaler, fit_scaler=False,
            fitted_feature_cols=self.fitted_feature_cols
        )

        goals_pred = _predict_nb(self.goals_model, X)
        assists_pred = _predict_nb(self.assists_model, X)

        return goals_pred, assists_pred

    def get_fallback_status(self) -> bool:
        """Return whether fallback was used during fitting."""
        return self._fallback_used

    def get_dispersion(self) -> tuple[float, float]:
        """Return estimated dispersion parameters (goals, assists)."""
        return self.goals_dispersion, self.assists_dispersion


# Baseline models for FPL
def _global_mean_baseline(train_df: pd.DataFrame) -> tuple[float, float]:
    """Global mean baseline for goals and assists."""
    return train_df["goals"].mean(), train_df["assists"].mean()


def _player_mean_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_history: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Player historical mean baseline."""
    player_means = train_df.groupby("player_id")[["goals", "assists"]].mean()

    test_goals = test_df["player_id"].map(player_means["goals"]).fillna(train_df["goals"].mean()).values
    test_assists = test_df["player_id"].map(player_means["assists"]).fillna(train_df["assists"].mean()).values

    return test_goals, test_assists


def _recent_form_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Recent form baseline (last N gameweeks)."""
    # Compute recent form per player
    recent_form = (
        train_df.sort_values(["player_id", "gameweek"])
        .groupby("player_id")
        .tail(window)
        .groupby("player_id")[["goals", "assists"]]
        .mean()
    )

    test_goals = test_df["player_id"].map(recent_form["goals"]).fillna(train_df["goals"].mean()).values
    test_assists = test_df["player_id"].map(recent_form["assists"]).fillna(train_df["assists"].mean()).values

    return test_goals, test_assists


def _minutes_adjusted_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Minutes-adjusted baseline (per-90 rate × expected minutes)."""
    # Compute per-90 rates
    train_df = train_df.copy()
    train_df["goals_per_90"] = train_df["goals"] / train_df["minutes"] * 90
    train_df["assists_per_90"] = train_df["assists"] / train_df["minutes"] * 90
    train_df.loc[train_df["minutes"] == 0, ["goals_per_90", "assists_per_90"]] = 0

    player_rates = train_df.groupby("player_id")[["goals_per_90", "assists_per_90"]].mean()

    # Expected minutes from position-team median
    expected_mins = train_df.groupby(["position", "team"])["minutes"].median()

    test_df = test_df.copy()
    test_df["expected_mins"] = test_df.apply(
        lambda r: expected_mins.get((r["position"], r["team"]), train_df["minutes"].median())
        if pd.notna(r["position"]) and pd.notna(r["team"]) else train_df["minutes"].median(),
        axis=1
    )

    test_goals = test_df.apply(
        lambda r: player_rates["goals_per_90"].get(r["player_id"], train_df["goals_per_90"].mean())
        * r["expected_mins"] / 90, axis=1
    ).values

    test_assists = test_df.apply(
        lambda r: player_rates["assists_per_90"].get(r["player_id"], train_df["assists_per_90"].mean())
        * r["expected_mins"] / 90, axis=1
    ).values

    return test_goals, test_assists


def run_fpl_baselines_at_cutoff(
    player_match_df: pd.DataFrame,
    player_season_df: pd.DataFrame,
    cutoff_gameweek: int,
    season_id: int,
    feature_config: "FPLFeatureConfig"
) -> pd.DataFrame:
    """Run all baselines for FPL at a specific gameweek cutoff."""
    from src.features.fpl_features import build_fpl_player_features
    from src.features.fpl_features import FPLFeatureConfig

    config = FPLFeatureConfig()
    features = build_fpl_player_features(
        player_season_df, player_match_df, config
    )

    # Filter to cutoff gameweek
    test_df = features[
        (features["season_id"] == season_id) &
        (features["gameweek"] == cutoff_gameweek)
    ].copy()

    if test_df.empty:
        return pd.DataFrame()

    # Training data: all gameweeks before cutoff
    train_df = features[
        (features["season_id"] == season_id) &
        (features["gameweek"] < cutoff_gameweek)
    ].copy()

    if train_df.empty:
        # No training data - return global mean fallback
        global_goals = player_match_df["goals"].mean()
        global_assists = player_match_df["assists"].mean()
        test_df["global_goals_pred"] = global_goals
        test_df["global_assists_pred"] = global_assists
        test_df["population"] = "all_lineup"
        test_df["cutoff_gameweek"] = cutoff_gameweek
        return test_df

    # Global mean baseline
    global_goals, global_assists = _global_mean_baseline(train_df)
    test_df["global_goals_pred"] = global_goals
    test_df["global_assists_pred"] = global_assists

    # Player mean baseline
    player_goals, player_assists = _player_mean_baseline(train_df, test_df)
    test_df["player_mean_goals_pred"] = player_goals
    test_df["player_mean_assists_pred"] = player_assists

    # Recent form baseline
    recent_goals, recent_assists = _recent_form_baseline(train_df, test_df)
    test_df["recent_form_goals_pred"] = recent_goals
    test_df["recent_form_assists_pred"] = recent_assists

    # Minutes-adjusted baseline
    ma_goals, ma_assists = _minutes_adjusted_baseline(train_df, test_df)
    test_df["minutes_adj_goals_pred"] = ma_goals
    test_df["minutes_adj_assists_pred"] = ma_assists

    # Add metadata
    test_df["cutoff_gameweek"] = cutoff_gameweek
    test_df["population"] = "all_lineup"
    test_df["season_id"] = test_df["season_id"].iloc[0] if len(test_df) > 0 else None

    return test_df


def run_fpl_baselines_rolling(
    player_match_df: pd.DataFrame,
    player_season_df: pd.DataFrame,
    cutoffs: list[int] | None = None,
    season_id: int | None = None,
    feature_config: "FPLFeatureConfig" | None = None
) -> pd.DataFrame:
    """Run baselines across multiple cutoffs for a season."""
    if cutoffs is None:
        cutoffs = list(range(1, 39))

    if feature_config is None:
        feature_config = FPLFeatureConfig()

    all_preds = []
    for cw in cutoffs:
        preds = run_fpl_baselines_at_cutoff(
            player_match_df, player_season_df, cw, season_id, feature_config
        )
        if not preds.empty:
            all_preds.append(preds)

    if not all_preds:
        return pd.DataFrame()

    return pd.concat(all_preds, ignore_index=True)


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, "/home/ganesh/projects/premier-league-intelligence")
    from src.dashboard.data_loader import load_player_season_fpl, load_player_match_fpl

    ps = load_player_season_fpl()
    pm = load_player_match_fpl()

    # Test baselines for 2023/24 season, gameweek 20
    preds = run_fpl_baselines_at_cutoff(pm, ps, 20, 2023)
    print(f"Predictions shape: {preds.shape}")
    print(f"Columns: {list(preds.columns)}")
    print(preds.head())