"""Evaluation metrics for goals and assists forecasting.

Implements MAE, RMSE, and Poisson deviance with explicit handling of
zero predictions and invalid inputs per the implementation contract.
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