"""Regularized Poisson and Negative Binomial GLM for player goal and assist prediction.

Uses scikit-learn's PoissonRegressor with L2 regularization for Poisson GLM.
Uses statsmodels NegativeBinomial for Negative Binomial GLM.
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

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm = None


@dataclass(frozen=True)
class GLMConfig:
    """Configuration for GLM model."""
    alpha: float = 1.0              # L2 regularization strength
    max_iter: int = 1000            # Maximum iterations for solver
    tol: float = 1e-4               # Convergence tolerance
    min_train_samples: int = 50     # Minimum training samples required
    standardize: bool = True        # Whether to standardize features
    random_state: int | None = 42   # Random state for reproducibility


@dataclass(frozen=True)
class NBConfig:
    """Configuration for Negative Binomial GLM model.
    
    The Negative Binomial distribution has an additional dispersion parameter (alpha)
    that accounts for overdispersion (variance > mean).
    """
    alpha: float = 1.0              # L2 regularization strength (on coefficients)
    max_iter: int = 1000            # Maximum iterations for solver
    tol: float = 1e-4               # Convergence tolerance
    min_train_samples: int = 50     # Minimum training samples required
    standardize: bool = True        # Whether to standardize features
    random_state: int | None = 42   # Random state for reproducibility
    dispersion_method: Literal["ml", "pearson"] = "ml"  # Method for dispersion estimation
    alpha_dispersion: float | None = None  # Fixed dispersion parameter (None = estimate from data)


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


def compute_dispersion_diagnostic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "pearson"
) -> float | None:
    """Compute dispersion diagnostic to check for overdispersion.
    
    For Poisson: variance = mean. Overdispersion = variance > mean.
    Pearson dispersion = sum((y - mu)^2 / mu) / (n - p)
    Deviance dispersion = deviance / (n - p)
    
    Args:
        y_true: Actual counts
        y_pred: Predicted means (from Poisson GLM)
        method: "pearson" or "deviance"
    
    Returns:
        Dispersion estimate (1.0 = equidispersed, >1 = overdispersed)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    if len(y_true) == 0:
        return None
    
    # Ensure positive predictions
    y_pred = np.maximum(y_pred, 1e-10)
    
    if method == "pearson":
        # Pearson chi-squared dispersion
        pearson_residuals = (y_true - y_pred) / np.sqrt(y_pred)
        n_params = 13  # approximate: 12 features + intercept
        dispersion = np.sum(pearson_residuals ** 2) / max(len(y_true) - n_params, 1)
    elif method == "deviance":
        # Deviance-based dispersion
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(y_true, y_pred, out=np.zeros_like(y_true), where=y_pred > 0)
            log_ratio = np.log(ratio, out=np.zeros_like(ratio), where=ratio > 0)
            ylogy_term = y_true * log_ratio
            deviance = 2 * np.sum(y_pred - y_true + ylogy_term)
        n_params = 13
        dispersion = deviance / max(len(y_true) - n_params, 1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(dispersion)


def tune_regularization_chronological(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_type: Literal["poisson", "negbin"] = "poisson",
    alpha_grid: list[float] | None = None,
    n_folds: int = 3,
    config: GLMConfig | NBConfig | None = None
) -> tuple[float, dict[float, float]]:
    """Tune regularization strength using chronological cross-validation.
    
    Splits training data chronologically into folds (earlier folds = older data).
    For each alpha, trains on earlier folds and validates on the next fold.
    
    Args:
        train_df: Training data with features and target.
        feature_cols: List of feature column names.
        target_col: Target column name ("goals_actual" or "assists_actual").
        model_type: "poisson" or "negbin".
        alpha_grid: List of alpha values to try. Default: [0.01, 0.1, 1.0, 10.0].
        n_folds: Number of chronological folds.
        config: GLMConfig or NBConfig (if None, uses default).
    
    Returns:
        Tuple of (best_alpha, cv_scores_dict).
    """
    if alpha_grid is None:
        alpha_grid = [0.01, 0.1, 1.0, 10.0]
    
    if config is None:
        if model_type == "poisson":
            config = GLMConfig()
        else:
            config = NBConfig()
    
    # Sort by match_week if available, otherwise assume already sorted
    if "match_week" in train_df.columns:
        train_df = train_df.sort_values("match_week")
    
    n = len(train_df)
    fold_size = n // (n_folds + 1)
    if fold_size < 10:
        # Not enough data for CV, return default alpha
        return config.alpha, {config.alpha: 0.0}
    
    cv_scores = {}
    
    for alpha in alpha_grid:
        fold_scores = []
        
        for fold in range(n_folds):
            # Train on earlier folds, validate on next fold
            train_end = (fold + 1) * fold_size
            val_start = train_end
            val_end = min(train_end + fold_size, n)
            
            if val_start >= val_end:
                continue
            
            train_fold = train_df.iloc[:train_end]
            val_fold = train_df.iloc[val_start:val_end]
            
            if len(train_fold) < 20 or len(val_fold) < 5:
                continue
            
            # Prepare features
            X_train, scaler, fitted_cols = _prepare_feature_matrix(
                train_fold, feature_cols, StandardScaler(), fit_scaler=True
            )
            X_val, _, _ = _prepare_feature_matrix(
                val_fold, feature_cols, scaler, fit_scaler=False,
                fitted_feature_cols=fitted_cols
            )
            
            y_train = train_fold[target_col].values.astype(float)
            y_val = val_fold[target_col].values.astype(float)
            
            if model_type == "poisson":
                model_config = GLMConfig(alpha=alpha, max_iter=config.max_iter, tol=config.tol)
                model = _fit_glm(X_train, y_train, model_config)
                preds = _predict_glm(model, X_val)
            else:
                if not STATSMODELS_AVAILABLE:
                    continue
                nb_config = NBConfig(alpha=alpha, max_iter=config.max_iter, tol=config.tol)
                model, _ = _fit_nb(X_train, y_train, nb_config)
                preds = _predict_nb(model, X_val)
            
            # Use Poisson deviance as validation metric (lower is better)
            from src.models.glm import compute_dispersion_diagnostic
            score = compute_dispersion_diagnostic(y_val, preds, method="pearson")
            if score is not None:
                fold_scores.append(score)
        
        if fold_scores:
            cv_scores[alpha] = float(np.mean(fold_scores))
    
    if not cv_scores:
        return config.alpha, {config.alpha: 0.0}
    
    best_alpha = min(cv_scores, key=cv_scores.get)
    return best_alpha, cv_scores


def _fit_nb(
    X: np.ndarray,
    y: np.ndarray,
    config: NBConfig
) -> tuple[object, float]:
    """Fit a Negative Binomial regression model using statsmodels.
    
    Returns:
        Tuple of (fitted_model, dispersion_parameter)
    """
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
                tol=config.tol,
                disp=False
            )
        except Exception as e:
            warnings.warn(f"NB fit failed: {e}, falling back to Poisson")
            poisson_model = PoissonRegressor(
                alpha=config.alpha,
                max_iter=config.max_iter,
                tol=config.tol,
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


class PlayerNB:
    """Regularized Negative Binomial GLM for player goal and assist prediction.
    
    Fits separate models for goals and assists using statsmodels NegativeBinomial.
    Handles overdispersion in count data (variance > mean).
    """

    def __init__(self, config: NBConfig):
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
        feature_cols: list[str] | None = None
    ) -> "PlayerNB":
        """Fit NB models for goals and assists on training data."""
        self.feature_cols = feature_cols if feature_cols is not None else FEATURE_COLUMNS
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

        y_goals = train_df["goals_actual"].values.astype(float)
        y_assists = train_df["assists_actual"].values.astype(float)

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


def train_nb_at_cutoff(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    cutoff_week: int,
    feature_config: FeatureConfig,
    nb_config: NBConfig
) -> tuple[pd.DataFrame, bool]:
    """Train Negative Binomial GLM at a specific cutoff week and generate predictions.
    
    Args:
        player_match: Full player_match DataFrame.
        fixtures: Full fixtures DataFrame.
        cutoff_week: Target matchweek.
        feature_config: FeatureConfig for feature engineering.
        nb_config: NBConfig for model training.
    
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
        fallback_preds["nb_goals_pred"] = 0.0
        fallback_preds["nb_assists_pred"] = 0.0
        fallback_preds["cutoff_week"] = cutoff_week
        fallback_preds["population"] = "all_lineup"
        return fallback_preds, True

    train_df = pd.concat(train_features, ignore_index=True)

    # Get test data for this cutoff
    test_df = compute_features_for_cutoff(player_match, fixtures, cutoff_week, feature_config)
    if test_df.empty:
        return pd.DataFrame(), False

    # Fit NB
    nb = PlayerNB(nb_config)
    nb.fit(train_df)

    # Predict
    goals_pred, assists_pred = nb.predict(test_df)

    # Build predictions DataFrame
    preds = test_df[["match_id", "player_id", "player_name", "team_id", "position",
                      "minutes_actual", "goals_actual", "assists_actual"]].copy()
    preds["nb_goals_pred"] = goals_pred
    preds["nb_assists_pred"] = assists_pred
    preds["cutoff_week"] = cutoff_week
    preds["population"] = "all_lineup"

    fallback_used = nb.get_fallback_status()
    return preds, fallback_used


def run_nb_rolling_evaluation(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    feature_config: FeatureConfig,
    nb_config: NBConfig,
    cutoffs: list[int] | None = None
) -> dict[str, pd.DataFrame]:
    """Run rolling-origin evaluation for NB model.
    
    Args:
        player_match: Full player_match DataFrame.
        fixtures: Full fixtures DataFrame.
        feature_config: FeatureConfig.
        nb_config: NBConfig.
        cutoffs: List of cutoff matchweeks.
    
    Returns:
        Dictionary with NB predictions and metrics.
    """
    if cutoffs is None:
        cutoffs = list(range(1, 39))

    all_predictions = []
    fallback_counts = 0

    for cw in cutoffs:
        preds, fallback = train_nb_at_cutoff(
            player_match, fixtures, cw, feature_config, nb_config
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


def evaluate_all_models(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    feature_config: FeatureConfig,
    baseline_config: BaselineConfig,
    glm_config: GLMConfig,
    nb_config: NBConfig,
    cutoffs: list[int] | None = None
) -> pd.DataFrame:
    """Evaluate all models (baselines, Poisson GLM, Negative Binomial GLM) on identical observations.
    
    Args:
        player_match: Full player_match DataFrame.
        fixtures: Full fixtures DataFrame.
        feature_config: FeatureConfig for feature engineering.
        baseline_config: BaselineConfig for baselines.
        glm_config: GLMConfig for Poisson GLM.
        nb_config: NBConfig for Negative Binomial GLM.
        cutoffs: List of cutoff matchweeks.
    
    Returns:
        DataFrame with metrics for all models.
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
    all_nb_preds = []
    all_baseline_preds = []

    for cw in cutoffs:
        # Poisson GLM predictions
        glm_preds, _ = train_glm_at_cutoff(
            player_match, fixtures, cw, feature_config, glm_config
        )
        if not glm_preds.empty:
            all_glm_preds.append(glm_preds)

        # Negative Binomial predictions
        nb_preds, _ = train_nb_at_cutoff(
            player_match, fixtures, cw, feature_config, nb_config
        )
        if not nb_preds.empty:
            all_nb_preds.append(nb_preds)

        # Baseline predictions (Population A)
        base_preds = evaluate_baselines_at_cutoff(
            pm_with_week, fixtures, cw, baseline_config, "all_lineup"
        )
        if not base_preds.empty:
            all_baseline_preds.append(base_preds)

    if not all_glm_preds or not all_nb_preds or not all_baseline_preds:
        return pd.DataFrame()

    glm_all = pd.concat(all_glm_preds, ignore_index=True)
    nb_all = pd.concat(all_nb_preds, ignore_index=True)
    base_all = pd.concat(all_baseline_preds, ignore_index=True)

    # Ensure same rows for comparison
    # Merge on match_id, player_id
    merged = base_all.merge(
        glm_all[["match_id", "player_id", "glm_goals_pred", "glm_assists_pred"]],
        on=["match_id", "player_id"],
        how="inner"
    ).merge(
        nb_all[["match_id", "player_id", "nb_goals_pred", "nb_assists_pred"]],
        on=["match_id", "player_id"],
        how="inner"
    )

    if merged.empty:
        return pd.DataFrame()

    # Compute metrics including all models
    from src.evaluation.metrics import compute_metrics_per_cutoff

    pred_cols_map = {
        "global_mean": ("global_goals_pred", "global_assists_pred"),
        "player_mean": ("player_mean_goals_pred", "player_mean_assists_pred"),
        "recent_form": ("recent_form_goals_pred", "recent_form_assists_pred"),
        "minutes_adjusted": ("minutes_adj_goals_pred", "minutes_adj_assists_pred"),
        "poisson_glm": ("glm_goals_pred", "glm_assists_pred"),
        "negbin_glm": ("nb_goals_pred", "nb_assists_pred"),
    }

    metrics = compute_metrics_per_cutoff(
        merged,
        target_cols=("goals_actual", "assists_actual"),
        pred_cols_map=pred_cols_map
    )

    return metrics