"""Evaluation metrics for goals/assists forecasting and match outcome prediction.

Implements MAE, RMSE, Poisson deviance, and classification metrics
(log loss, Brier score, accuracy) with explicit handling of
invalid inputs per the implementation contract.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger("plintel.evaluation.metrics")


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def poisson_deviance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    warn_on_invalid: bool = True
) -> float | None:
    """Compute Poisson deviance with explicit zero-handling.

    Formula: 2 * mean(y_pred - y_true + y_true * log(y_true / y_pred))
    Convention: 0 * log(0) = 0

    Returns None if any y_pred < 0 (invalid for Poisson) or if computation
    would produce NaN/inf. Logs warning in such cases.

    Args:
        y_true: Actual counts (non-negative integers).
        y_pred: Predicted expected counts (must be >= 0 for valid Poisson).
        warn_on_invalid: Whether to log warning when returning None.

    Returns:
        Poisson deviance as float, or None if invalid inputs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    if len(y_true) == 0:
        return None

    # Check for invalid predictions (negative values not valid for Poisson)
    if np.any(y_pred < 0):
        if warn_on_invalid:
            logger.warning(
                "Poisson deviance not computed: %d negative predictions found",
                np.sum(y_pred < 0)
            )
            warnings.warn(
                f"Poisson deviance skipped: {np.sum(y_pred < 0)} negative predictions",
                UserWarning
            )
        return None

    # Check for NaN or inf in predictions
    if np.any(~np.isfinite(y_pred)):
        if warn_on_invalid:
            logger.warning("Poisson deviance not computed: non-finite predictions found")
            warnings.warn("Poisson deviance skipped: non-finite predictions", UserWarning)
        return None

    # Poisson deviance with 0 * log(0) = 0 convention
    # For y_true = 0: term = y_pred - 0 + 0 * log(0/y_pred) = y_pred
    # For y_true > 0: term = y_pred - y_true + y_true * log(y_true / y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute y_true * log(y_true / y_pred) with convention 0 * log(0) = 0
        ratio = np.divide(y_true, y_pred, out=np.zeros_like(y_true), where=y_pred > 0)
        log_ratio = np.log(ratio, out=np.zeros_like(ratio), where=ratio > 0)
        ylogy_term = y_true * log_ratio

        deviance_terms = 2 * (y_pred - y_true + ylogy_term)

    if not np.all(np.isfinite(deviance_terms)):
        if warn_on_invalid:
            logger.warning("Poisson deviance produced non-finite values")
            warnings.warn("Poisson deviance produced non-finite values", UserWarning)
        return None

    return float(np.mean(deviance_terms))


def compute_metrics(
    y_true_goals: np.ndarray,
    y_pred_goals: np.ndarray,
    y_true_assists: np.ndarray,
    y_pred_assists: np.ndarray
) -> dict[str, float | None]:
    """Compute all metrics for goals and assists.

    Args:
        y_true_goals: Actual goals.
        y_pred_goals: Predicted expected goals.
        y_true_assists: Actual assists.
        y_pred_assists: Predicted expected assists.

    Returns:
        Dictionary with MAE, RMSE, and Poisson deviance for goals and assists.
    """
    return {
        "mae_goals": mean_absolute_error(y_true_goals, y_pred_goals),
        "rmse_goals": root_mean_squared_error(y_true_goals, y_pred_goals),
        "poisson_deviance_goals": poisson_deviance(y_true_goals, y_pred_goals),
        "mae_assists": mean_absolute_error(y_true_assists, y_pred_assists),
        "rmse_assists": root_mean_squared_error(y_true_assists, y_pred_assists),
        "poisson_deviance_assists": poisson_deviance(y_true_assists, y_pred_assists),
        "sample_size": len(y_true_goals),
    }


def compute_metrics_per_cutoff(
    predictions: pd.DataFrame,
    target_cols: tuple[str, str] = ("goals_actual", "assists_actual"),
    pred_cols_map: dict[str, tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Compute metrics per cutoff week and baseline.

    Args:
        predictions: DataFrame from evaluate_baselines_at_cutoff concatenated across cutoffs.
        target_cols: (goals_actual_col, assists_actual_col).
        pred_cols_map: Mapping from baseline name to (goals_pred_col, assists_pred_col).
                       Defaults to the four baselines in the contract.

    Returns:
        DataFrame with metrics per cutoff_week, population, and baseline.
    """
    if pred_cols_map is None:
        pred_cols_map = {
            "global_mean": ("global_goals_pred", "global_assists_pred"),
            "player_mean": ("player_mean_goals_pred", "player_mean_assists_pred"),
            "recent_form": ("recent_form_goals_pred", "recent_form_assists_pred"),
            "minutes_adjusted": ("minutes_adj_goals_pred", "minutes_adj_assists_pred"),
        }

    goal_col, assist_col = target_cols
    results = []

    for (cutoff_week, population), group in predictions.groupby(["cutoff_week", "population"]):
        y_true_g = group[goal_col].values
        y_true_a = group[assist_col].values

        for baseline_name, (g_pred_col, a_pred_col) in pred_cols_map.items():
            y_pred_g = group[g_pred_col].values
            y_pred_a = group[a_pred_col].values

            metrics = compute_metrics(y_true_g, y_pred_g, y_true_a, y_pred_a)
            metrics.update({
                "cutoff_week": cutoff_week,
                "population": population,
                "baseline": baseline_name,
            })
            results.append(metrics)

    return pd.DataFrame(results)


def compute_aggregate_metrics(
    predictions: pd.DataFrame,
    target_cols: tuple[str, str] = ("goals_actual", "assists_actual"),
    pred_cols_map: dict[str, tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Compute aggregate metrics weighted by sample size across all cutoffs.

    Args:
        predictions: DataFrame from evaluate_baselines_at_cutoff concatenated across cutoffs.
        target_cols: (goals_actual_col, assists_actual_col).
        pred_cols_map: Mapping from baseline name to (goals_pred_col, assists_pred_col).

    Returns:
        DataFrame with aggregate metrics per population and baseline.
    """
    if pred_cols_map is None:
        pred_cols_map = {
            "global_mean": ("global_goals_pred", "global_assists_pred"),
            "player_mean": ("player_mean_goals_pred", "player_mean_assists_pred"),
            "recent_form": ("recent_form_goals_pred", "recent_form_assists_pred"),
            "minutes_adjusted": ("minutes_adj_goals_pred", "minutes_adj_assists_pred"),
        }

    goal_col, assist_col = target_cols
    results = []

    for population, group in predictions.groupby("population"):
        y_true_g = group[goal_col].values
        y_true_a = group[assist_col].values

        for baseline_name, (g_pred_col, a_pred_col) in pred_cols_map.items():
            y_pred_g = group[g_pred_col].values
            y_pred_a = group[a_pred_col].values

            metrics = compute_metrics(y_true_g, y_pred_g, y_true_a, y_pred_a)
            metrics.update({
                "population": population,
                "baseline": baseline_name,
            })
            results.append(metrics)

    return pd.DataFrame(results)


# =============================================================================
# Classification metrics for match outcome prediction
# =============================================================================

def multiclass_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
    warn_on_invalid: bool = True
) -> float | None:
    """Compute multiclass log loss (cross-entropy).

    Args:
        y_true: True class labels as integers (0, 1, 2 for away/draw/home).
        y_pred: Predicted probabilities, shape (n_samples, 3) or (n_samples,)
                If 1D, assumed to be probability of positive class (binary).
        eps: Clipping threshold to avoid log(0).
        warn_on_invalid: Whether to log warning when returning None.

    Returns:
        Log loss as float, or None if invalid inputs.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        if warn_on_invalid:
            logger.warning("Log loss: y_true and y_pred length mismatch")
        return None

    if len(y_true) == 0:
        return None

    # Handle both 1D (binary) and 2D (multiclass) predictions
    if y_pred.ndim == 1:
        # Binary case: y_pred is probability of class 1
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # Convert to 2D for uniform handling
        y_pred_2d = np.column_stack([1 - y_pred, y_pred])
        # Convert y_true to one-hot
        n_classes = 2
    else:
        # Multiclass case
        if y_pred.shape[1] < 2:
            if warn_on_invalid:
                logger.warning("Log loss: y_pred must have at least 2 columns for multiclass")
            return None
        y_pred_2d = np.clip(y_pred, eps, 1 - eps)
        # Renormalize rows to sum to 1
        row_sums = y_pred_2d.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            if warn_on_invalid:
                logger.warning("Log loss: predicted probabilities sum to zero or negative")
            return None
        y_pred_2d = y_pred_2d / row_sums
        n_classes = y_pred_2d.shape[1]

    # Check for valid class labels
    if np.any(y_true < 0) or np.any(y_true >= n_classes):
        if warn_on_invalid:
            logger.warning("Log loss: y_true contains invalid class labels")
        return None

    # Compute log loss: -1/n * sum(log(p_true))
    # where p_true is the predicted probability of the true class
    n_samples = len(y_true)
    log_probs = np.log(y_pred_2d[np.arange(n_samples), y_true])

    if not np.all(np.isfinite(log_probs)):
        if warn_on_invalid:
            logger.warning("Log loss: non-finite log probabilities")
        return None

    return float(-np.mean(log_probs))


def multiclass_brier_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    warn_on_invalid: bool = True
) -> float | None:
    """Compute multiclass Brier score.

    Brier score = mean(sum((y_pred - y_true_onehot)^2)) over classes.
    For 3 classes: 1/n * sum((p_home - I[y=home])^2 + (p_draw - I[y=draw])^2 + (p_away - I[y=away])^2)

    Args:
        y_true: True class labels as integers (0, 1, 2).
        y_pred: Predicted probabilities, shape (n_samples, 3).
        warn_on_invalid: Whether to log warning when returning None.

    Returns:
        Brier score as float (lower is better, 0 is perfect), or None if invalid.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        if warn_on_invalid:
            logger.warning("Brier score: y_true and y_pred length mismatch")
        return None

    if len(y_true) == 0:
        return None

    if y_pred.ndim != 2 or y_pred.shape[1] < 2:
        if warn_on_invalid:
            logger.warning("Brier score: y_pred must be 2D with at least 2 columns")
        return None

    n_classes = y_pred.shape[1]

    # Check for valid class labels
    if np.any(y_true < 0) or np.any(y_true >= n_classes):
        if warn_on_invalid:
            logger.warning("Brier score: y_true contains invalid class labels")
        return None

    # Check for valid probabilities (non-negative, rows sum to ~1)
    if np.any(y_pred < -1e-10):
        if warn_on_invalid:
            logger.warning("Brier score: negative predicted probabilities")
        return None

    row_sums = y_pred.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        if warn_on_invalid:
            logger.warning("Brier score: predicted probabilities do not sum to 1")
        return None

    if np.any(~np.isfinite(y_pred)):
        if warn_on_invalid:
            logger.warning("Brier score: non-finite predicted probabilities")
        return None

    # Create one-hot encoding of true labels
    n_samples = len(y_true)
    y_true_oh = np.zeros((n_samples, n_classes))
    y_true_oh[np.arange(n_samples), y_true] = 1.0

    # Brier score: mean squared error between predicted probs and one-hot targets
    squared_errors = (y_pred - y_true_oh) ** 2
    brier = float(np.mean(np.sum(squared_errors, axis=1)))

    return brier


def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    warn_on_invalid: bool = True
) -> float | None:
    """Compute accuracy (fraction of correct predictions).

    Args:
        y_true: True class labels as integers.
        y_pred: Predicted class labels as integers, OR probability matrix (n_samples, n_classes).
        warn_on_invalid: Whether to log warning when returning None.

    Returns:
        Accuracy as float in [0, 1], or None if invalid.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        if warn_on_invalid:
            logger.warning("Accuracy: y_true and y_pred length mismatch")
        return None

    if len(y_true) == 0:
        return None

    # If y_pred is probability matrix, take argmax
    if y_pred.ndim == 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = np.asarray(y_pred, dtype=int)

    if np.any(y_pred_classes < 0):
        if warn_on_invalid:
            logger.warning("Accuracy: negative predicted classes")
        return None

    return float(np.mean(y_true == y_pred_classes))


# =============================================================================
# Goal-level / scoreline metrics for match outcome prediction
# =============================================================================

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def poisson_log_likelihood(
    y_true_home: np.ndarray,
    y_true_away: np.ndarray,
    y_pred_home_rate: np.ndarray,
    y_pred_away_rate: np.ndarray,
    warn_on_invalid: bool = True
) -> float | None:
    """Compute Poisson log-likelihood for scoreline predictions.
    
    For independent Poisson models: log P(h, a) = log Pois(h; λ_h) + log Pois(a; λ_a)
    
    Args:
        y_true_home: Actual home goals.
        y_true_away: Actual away goals.
        y_pred_home_rate: Predicted home goal rate (lambda).
        y_pred_away_rate: Predicted away goal rate (lambda).
        warn_on_invalid: Whether to log warning when returning None.
    
    Returns:
        Log-likelihood (higher is better), or None if invalid.
    """
    y_true_home = np.asarray(y_true_home, dtype=float)
    y_true_away = np.asarray(y_true_away, dtype=float)
    y_pred_home_rate = np.asarray(y_pred_home_rate, dtype=float)
    y_pred_away_rate = np.asarray(y_pred_away_rate, dtype=float)
    
    if len(y_true_home) != len(y_pred_home_rate):
        if warn_on_invalid:
            logger.warning("Poisson LL: length mismatch")
        return None
    
    if len(y_true_home) == 0:
        return None
    
    # Check for valid rates
    if np.any(y_pred_home_rate <= 0) or np.any(y_pred_away_rate <= 0):
        if warn_on_invalid:
            logger.warning("Poisson LL: non-positive rates")
        return None
    
    if np.any(~np.isfinite(y_pred_home_rate)) or np.any(~np.isfinite(y_pred_away_rate)):
        if warn_on_invalid:
            logger.warning("Poisson LL: non-finite rates")
        return None
    
    # Poisson log-likelihood: log P(x; λ) = x*log(λ) - λ - log(x!)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_prob_home = y_true_home * np.log(y_pred_home_rate) - y_pred_home_rate
        log_prob_away = y_true_away * np.log(y_pred_away_rate) - y_pred_away_rate
        
        # Subtract log factorial (log(x!))
        log_fact_home = np.sum([np.log(np.math.factorial(int(x))) for x in y_true_home.astype(int)])
        log_fact_away = np.sum([np.log(np.math.factorial(int(x))) for x in y_true_away.astype(int)])
    
    if not np.all(np.isfinite(log_prob_home)) or not np.all(np.isfinite(log_prob_away)):
        if warn_on_invalid:
            logger.warning("Poisson LL: non-finite log probs")
        return None
    
    log_likelihood = float(np.mean(log_prob_home + log_prob_away) - (log_fact_home + log_fact_away) / len(y_true_home))
    return log_likelihood


def compute_scoreline_metrics(
    y_true_home: np.ndarray,
    y_true_away: np.ndarray,
    y_pred_home_rate: np.ndarray,
    y_pred_away_rate: np.ndarray,
    y_pred_scoreline_probs: np.ndarray | None = None
) -> dict[str, float | None]:
    """Compute all scoreline/goal-level metrics.
    
    Args:
        y_true_home: Actual home goals.
        y_true_away: Actual away goals.
        y_pred_home_rate: Predicted home goal rate (lambda).
        y_pred_away_rate: Predicted away goal rate (lambda).
        y_pred_scoreline_probs: Optional predicted scoreline probabilities (not used yet).
    
    Returns:
        Dictionary with MAE, RMSE for goals, Poisson log-likelihood, etc.
    """
    return {
        "mae_home_goals": mean_absolute_error(y_true_home, y_pred_home_rate),
        "rmse_home_goals": root_mean_squared_error(y_true_home, y_pred_home_rate),
        "mae_away_goals": mean_absolute_error(y_true_away, y_pred_away_rate),
        "rmse_away_goals": root_mean_squared_error(y_true_away, y_pred_away_rate),
        "poisson_log_likelihood": poisson_log_likelihood(
            y_true_home, y_true_away, y_pred_home_rate, y_pred_away_rate
        ),
        "sample_size": len(y_true_home),
    }


def compute_scoreline_metrics_per_cutoff(
    predictions: pd.DataFrame,
    target_home_col: str = "home_score",
    target_away_col: str = "away_score",
    pred_cols_map: dict[str, tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Compute scoreline/goal metrics per cutoff week and model.
    
    Args:
        predictions: DataFrame with true scores and predicted goal rates.
        target_home_col: Column name for true home goals.
        target_away_col: Column name for true away goals.
        pred_cols_map: Mapping from model name to (home_rate_col, away_rate_col).
    
    Returns:
        DataFrame with metrics per cutoff_week, population, and model.
    """
    if pred_cols_map is None:
        pred_cols_map = {
            "global_frequencies": ("global_home_rate", "global_away_rate"),
            "recent_form": ("recent_form_home_rate", "recent_form_away_rate"),
            "feature_model": ("model_home_rate", "model_away_rate"),
        }
    
    results = []
    
    for (cutoff_week, population), group in predictions.groupby(["cutoff_week", "population"]):
        y_true_home = group[target_home_col].values
        y_true_away = group[target_away_col].values
        
        for model_name, (home_col, away_col) in pred_cols_map.items():
            if home_col not in group.columns or away_col not in group.columns:
                continue
            
            y_pred_home = group[home_col].values
            y_pred_away = group[away_col].values
            
            metrics = compute_scoreline_metrics(
                y_true_home, y_true_away, y_pred_home, y_pred_away
            )
            metrics.update({
                "cutoff_week": cutoff_week,
                "population": population,
                "model": model_name,
            })
            results.append(metrics)
    
    return pd.DataFrame(results)


def compute_scoreline_aggregate_metrics(
    predictions: pd.DataFrame,
    target_home_col: str = "home_score",
    target_away_col: str = "away_score",
    pred_cols_map: dict[str, tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Compute aggregate scoreline metrics across all cutoffs.
    
    Args:
        predictions: DataFrame with true scores and predicted goal rates.
        target_home_col: Column name for true home goals.
        target_away_col: Column name for true away goals.
        pred_cols_map: Mapping from model name to (home_rate_col, away_rate_col).
    
    Returns:
        DataFrame with aggregate metrics per population and model.
    """
    if pred_cols_map is None:
        pred_cols_map = {
            "global_frequencies": ("global_home_rate", "global_away_rate"),
            "recent_form": ("recent_form_home_rate", "recent_form_away_rate"),
            "feature_model": ("model_home_rate", "model_away_rate"),
        }
    
    results = []
    
    for population, group in predictions.groupby("population"):
        y_true_home = group[target_home_col].values
        y_true_away = group[target_away_col].values
        
        for model_name, (home_col, away_col) in pred_cols_map.items():
            if home_col not in group.columns or away_col not in group.columns:
                continue
            
            y_pred_home = group[home_col].values
            y_pred_away = group[away_col].values
            
            metrics = compute_scoreline_metrics(
                y_true_home, y_true_away, y_pred_home, y_pred_away
            )
            metrics.update({
                "population": population,
                "model": model_name,
            })
            results.append(metrics)
    
    return pd.DataFrame(results)


def compute_match_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray
) -> dict[str, float | None]:
    """Compute all match outcome metrics.

    Args:
        y_true: True outcomes (0=away, 1=draw, 2=home).
        y_pred_probs: Predicted probabilities, shape (n_samples, 3) for [away, draw, home].

    Returns:
        Dictionary with log_loss, brier_score, accuracy, sample_size.
    """
    # Ensure probability columns are in correct order [away, draw, home]
    # Our baselines output (home, draw, away) so we need to reorder
    if y_pred_probs.shape[1] == 3:
        # Reorder from [home, draw, away] to [away, draw, home] for metrics
        y_pred_reordered = y_pred_probs[:, [2, 1, 0]]
    else:
        y_pred_reordered = y_pred_probs

    return {
        "log_loss": multiclass_log_loss(y_true, y_pred_reordered),
        "brier_score": multiclass_brier_score(y_true, y_pred_reordered),
        "accuracy": accuracy_score(y_true, y_pred_reordered),
        "sample_size": len(y_true),
    }


def compute_match_metrics_per_cutoff(
    predictions: pd.DataFrame,
    target_col: str = "outcome",
    pred_cols_map: dict[str, tuple[str, str, str]] | None = None
) -> pd.DataFrame:
    """Compute match outcome metrics per cutoff week and baseline.

    Args:
        predictions: DataFrame from evaluate_match_baselines_at_cutoff concatenated across cutoffs.
        target_col: Column name for true outcome (0=away, 1=draw, 2=home).
        pred_cols_map: Mapping from baseline name to (home_prob_col, draw_prob_col, away_prob_col).

    Returns:
        DataFrame with metrics per cutoff_week, population, and baseline.
    """
    if pred_cols_map is None:
        pred_cols_map = {
            "global_frequencies": ("global_home_prob", "global_draw_prob", "global_away_prob"),
            "home_advantage": ("home_advantage_home_prob", "home_advantage_draw_prob", "home_advantage_away_prob"),
            "team_strength": ("team_strength_home_prob", "team_strength_draw_prob", "team_strength_away_prob"),
            "recent_form": ("recent_form_home_prob", "recent_form_draw_prob", "recent_form_away_prob"),
        }

    results = []

    for (cutoff_week, population), group in predictions.groupby(["cutoff_week", "population"]):
        y_true = group[target_col].values

        for baseline_name, (home_col, draw_col, away_col) in pred_cols_map.items():
            y_pred = np.column_stack([
                group[home_col].values,
                group[draw_col].values,
                group[away_col].values,
            ])

            metrics = compute_match_metrics(y_true, y_pred)
            metrics.update({
                "cutoff_week": cutoff_week,
                "population": population,
                "baseline": baseline_name,
            })
            results.append(metrics)

    return pd.DataFrame(results)


def compute_match_aggregate_metrics(
    predictions: pd.DataFrame,
    target_col: str = "outcome",
    pred_cols_map: dict[str, tuple[str, str, str]] | None = None
) -> pd.DataFrame:
    """Compute aggregate match outcome metrics across all cutoffs.

    Args:
        predictions: DataFrame from evaluate_match_baselines_at_cutoff concatenated across cutoffs.
        target_col: Column name for true outcome.
        pred_cols_map: Mapping from baseline name to (home_prob_col, draw_prob_col, away_prob_col).

    Returns:
        DataFrame with aggregate metrics per population and baseline.
    """
    if pred_cols_map is None:
        pred_cols_map = {
            "global_frequencies": ("global_home_prob", "global_draw_prob", "global_away_prob"),
            "home_advantage": ("home_advantage_home_prob", "home_advantage_draw_prob", "home_advantage_away_prob"),
            "team_strength": ("team_strength_home_prob", "team_strength_draw_prob", "team_strength_away_prob"),
            "recent_form": ("recent_form_home_prob", "recent_form_draw_prob", "recent_form_away_prob"),
        }

    results = []

    for population, group in predictions.groupby("population"):
        y_true = group[target_col].values

        for baseline_name, (home_col, draw_col, away_col) in pred_cols_map.items():
            y_pred = np.column_stack([
                group[home_col].values,
                group[draw_col].values,
                group[away_col].values,
            ])

            metrics = compute_match_metrics(y_true, y_pred)
            metrics.update({
                "population": population,
                "baseline": baseline_name,
            })
            results.append(metrics)

    return pd.DataFrame(results)