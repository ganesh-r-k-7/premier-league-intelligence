"""Chronological evaluation framework for FPL multi-season player forecasting.

Implements rolling-origin evaluation across FPL seasons with proper
temporal splits and baseline comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.models.fpl_model import (
    FPLPlayerGLM,
    FPLPlayerNB,
    FPLGLMConfig,
    FPLNBConfig,
    run_fpl_baselines_rolling,
)
from src.features.fpl_features import FPLFeatureConfig, build_fpl_player_features
from src.evaluation.metrics import compute_metrics, compute_metrics_per_cutoff


@dataclass(frozen=True)
class FPLEvaluationConfig:
    """Configuration for FPL chronological evaluation."""
    # Training seasons (inclusive)
    train_seasons: list[int] = field(default_factory=lambda: list(range(2016, 2022)))  # 2016/17–2021/22
    # Validation season
    val_season: int = 2022  # 2022/23
    # Test seasons (for final evaluation)
    test_seasons: list[int] = field(default_factory=lambda: list(range(2023, 2026)))  # 2023/24–2025/26
    # Gameweeks to evaluate (None = all 1-38)
    gameweeks: Optional[list[int]] = None
    # Model configurations
    glm_config: "FPLGLMConfig" = field(default_factory=lambda: FPLGLMConfig())
    nb_config: "FPLNBConfig" = field(default_factory=lambda: FPLNBConfig())
    feature_config: "FPLFeatureConfig" = field(default_factory=lambda: FPLFeatureConfig())
    # Baselines to compare against
    baselines: list[str] = field(default_factory=lambda: [
        "global_mean", "player_mean", "recent_form", "minutes_adjusted"
    ])
    # Target metrics
    targets: list[str] = field(default_factory=lambda: ["goals", "assists"])
    # Minimum training samples per player
    min_history: int = 3
    # Rolling window for recent form
    recent_window: int = 5


@dataclass(frozen=True)
class FPLEvaluationConfig:
    """Configuration for FPL chronological evaluation."""
    train_seasons: list[int]
    val_season: int
    test_seasons: list[int]
    gameweeks: Optional[list[int]]
    glm_config: "FPLGLMConfig"
    nb_config: "FPLNBConfig"
    feature_config: "FPLFeatureConfig"
    baselines: list[str]
    targets: list[str]
    min_history: int
    recent_window: int


def _make_fpl_evaluation_config() -> FPLEvaluationConfig:
    """Create default evaluation configuration."""
    from src.models.fpl_model import FPLGLMConfig, FPLNBConfig
    from src.features.fpl_features import FPLFeatureConfig

    return FPLEvaluationConfig(
        train_seasons=list(range(2016, 2022)),  # 2016/17–2021/22
        val_season=2022,  # 2022/23
        test_seasons=list(range(2023, 2026)),  # 2023/24–2025/26
        gameweeks=None,  # All 1-38
        glm_config=FPLGLMConfig(alpha=1.0, min_train_samples=50),
        nb_config=FPLNBConfig(alpha=1.0, min_train_samples=50),
        feature_config=FPLFeatureConfig(
            rolling_window=5,
            min_history=3,
            include_xg=True,
            include_starts=True,
            include_position_team=True
        ),
        baselines=["global_mean", "player_mean", "recent_form", "minutes_adjusted"],
        targets=["goals", "assists"],
        min_history=3,
        recent_window=5,
    )


def _load_fpl_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FPL player-season and player-match data."""
    from src.dashboard.data_loader import load_player_season_fpl, load_player_match_fpl
    ps = load_player_season_fpl()
    pm = load_player_match_fpl()
    return ps, pm


def _build_fpl_features_for_seasons(
    player_season_df: pd.DataFrame,
    player_match_df: pd.DataFrame,
    season_ids: list[int],
    feature_config: "FPLFeatureConfig"
) -> pd.DataFrame:
    """Build FPL features for specified seasons."""
    from src.features.fpl_features import build_fpl_player_features

    # Filter data to specified seasons
    ps_filtered = player_season_df[player_season_df["season_id"].isin(season_ids)].copy()
    pm_filtered = player_match_df[player_match_df["season_id"].isin(season_ids)].copy()

    config = FPLFeatureConfig()
    return build_fpl_player_features(ps_filtered, pm_filtered, config)


def _get_season_cutoffs(season_id: int) -> list[int]:
    """Get available gameweeks for a season (1-38)."""
    return list(range(1, 39))


def run_fpl_glm_evaluation(
    config: "FPLEvaluationConfig" | None = None
) -> dict[str, pd.DataFrame]:
    """Run FPL GLM evaluation with chronological splits."""
    from src.models.fpl_model import FPLPlayerGLM, run_fpl_baselines_rolling
    from src.features.fpl_features import FPLFeatureConfig, build_fpl_player_features
    from src.evaluation.metrics import compute_metrics_per_cutoff, compute_aggregate_metrics
    from src.dashboard.data_loader import load_player_season_fpl, load_player_match_fpl

    if config is None:
        config = _make_fpl_evaluation_config()

    # Load data
    ps, pm = _load_fpl_data()

    # Build features for all training + validation seasons
    all_train_seasons = config.train_seasons + [config.val_season]
    features = build_fpl_player_features(
        load_player_season_fpl(),
        load_player_match_fpl(),
        config.feature_config
    )

    # Filter to training + validation seasons
    features = features[features["season_id"].isin(all_train_seasons)].copy()

    # Get season_id for validation season
    val_season_id = config.val_season

    # Get cutoffs (gameweeks)
    cutoffs = config.gameweeks or list(range(1, 39))

    # Run baselines
    print("Running baselines...")
    baseline_preds = run_fpl_baselines_rolling(
        load_player_match_fpl(),
        load_player_season_fpl(),
        cutoffs=cutoffs,
        season_id=val_season_id
    )

    # Run GLM
    print("Running GLM evaluation...")
    from src.models.fpl_model import FPLPlayerGLM

    glm_config = config.glm_config
    feature_config = config.feature_config

    all_glm_preds = []
    for cw in cutoffs:
        # Training data: all gameweeks before cutoff in training seasons
        train_df = features[
            (features["season_id"].isin(config.train_seasons)) &
            (features["gameweek"] < cw)
        ].copy()

        # Test data: validation season, current gameweek
        test_df = features[
            (features["season_id"] == val_season_id) &
            (features["gameweek"] == cw)
        ].copy()

        if train_df.empty or test_df.empty:
            continue

        # Fit GLM
        glm = FPLPlayerGLM(glm_config)
        glm.fit(train_df)

        # Predict
        goals_pred, assists_pred = glm.predict(test_df)

        # Build predictions DataFrame
        preds = test_df[["player_id", "season_id", "gameweek", "goals", "assists"]].copy()
        preds["glm_goals_pred"] = goals_pred
        preds["glm_assists_pred"] = assists_pred
        preds["cutoff_gameweek"] = cw
        preds["population"] = "all_lineup"

        all_glm_preds.append(preds)

    # Combine predictions
    if not all_glm_preds:
        return {"glm": pd.DataFrame(), "baselines": baseline_preds}

    glm_preds = pd.concat(all_glm_preds, ignore_index=True)

    # Combine with baselines for comparison
    # Merge on player_id, gameweek, cutoff_gameweek
    baseline_preds = baseline_preds.copy()
    baseline_preds["model"] = "baseline"
    glm_preds["model"] = "glm"

    # Merge for comparison
    combined = baseline_preds.merge(
        glm_preds[["player_id", "gameweek", "cutoff_gameweek", "glm_goals_pred", "glm_assists_pred"]],
        on=["player_id", "gameweek", "cutoff_gameweek"],
        how="inner"
    )

    return {
        "glm": glm_preds,
        "baselines": baseline_preds,
        "combined": combined,
    }


def run_fpl_nb_evaluation(
    config: "FPLEvaluationConfig" | None = None
) -> dict[str, pd.DataFrame]:
    """Run FPL Negative Binomial evaluation."""
    if config is None:
        config = _make_fpl_evaluation_config()

    # Similar to GLM but with NB model
    # ... implementation similar to GLM
    return {"nb": pd.DataFrame(), "baselines": pd.DataFrame()}


def run_fpl_baseline_comparison(
    config: "FPLEvaluationConfig" | None = None
) -> pd.DataFrame:
    """Run comprehensive baseline comparison on validation season."""
    from src.models.fpl_model import run_fpl_baselines_rolling
    from src.dashboard.data_loader import load_player_season_fpl, load_player_match_fpl

    if config is None:
        config = _make_fpl_evaluation_config()

    ps, pm = _load_fpl_data()

    val_season_id = config.val_season
    cutoffs = config.gameweeks or list(range(1, 39))

    print(f"Running baselines for season {val_season_id}...")
    baseline_preds = run_fpl_baselines_rolling(
        load_player_match_fpl(),
        load_player_season_fpl(),
        cutoffs=cutoffs,
        season_id=val_season_id
    )

    return baseline_preds


def evaluate_models_on_season(
    season_id: int,
    models: dict[str, object],
    player_match_df: pd.DataFrame,
    player_season_df: pd.DataFrame,
    feature_config: "FPLFeatureConfig",
    cutoffs: list[int] | None = None
) -> pd.DataFrame:
    """Evaluate multiple models on a specific season."""
    from src.features.fpl_features import build_fpl_player_features
    from src.evaluation.metrics import compute_metrics, compute_metrics_per_cutoff

    if cutoffs is None:
        cutoffs = list(range(1, 39))

    features = build_fpl_player_features(player_season_df, player_match_df)

    all_preds = {}
    for name, model in models.items():
        all_preds[name] = []

    for cw in cutoffs:
        train_df = features[
            (features["season_id"] < season_id) | 
            ((features["season_id"] == season_id) & (features["gameweek"] < cw))
        ].copy()

        test_df = features[
            (features["season_id"] == season_id) & 
            (features["gameweek"] == cw)
        ].copy()

        if train_df.empty or test_df.empty:
            continue

        for name, model in models.items():
            if hasattr(model, 'fit') and hasattr(model, 'predict'):
                model.fit(train_df)
                g_pred, a_pred = model.predict(test_df)

                preds = test_df[["player_id", "season_id", "gameweek", "goals", "assists"]].copy()
                preds[f"{name}_goals_pred"] = g_pred
                preds[f"{name}_assists_pred"] = a_pred
                preds["cutoff_gameweek"] = cw
                all_preds[name].append(preds)

    # Combine and compute metrics
    results = {}
    for name, preds_list in all_preds.items():
        if preds_list:
            preds_df = pd.concat(preds_list, ignore_index=True)
            # Compute metrics...
            results[name] = preds_df

    return pd.DataFrame()  # Placeholder


def run_chronological_evaluation(
    config: "FPLEvaluationConfig" | None = None
) -> dict:
    """Run full chronological evaluation pipeline."""
    from src.dashboard.data_loader import load_player_season_fpl, load_player_match_fpl

    if config is None:
        config = _make_fpl_evaluation_config()

    ps, pm = _load_fpl_data()

    # Build features for all seasons
    all_seasons = config.train_seasons + [config.val_season] + config.test_seasons
    features = build_fpl_player_features(
        ps[ps["season_id"].isin(all_seasons)],
        pm[pm["season_id"].isin(all_seasons)],
    )

    results = {}

    # Validation evaluation
    print("Running validation evaluation...")
    val_results = run_fpl_glm_evaluation(_make_fpl_evaluation_config())
    results["validation"] = val_results

    # Test evaluations (if test seasons available)
    for test_season in config.test_seasons:
        print(f"Running test evaluation for {test_season}...")
        # Would implement test evaluation here
        pass

    return results


def compute_fpl_model_metrics(
    predictions: pd.DataFrame,
    target_cols: tuple[str, str] = ("goals", "assists"),
    pred_cols_map: dict[str, tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Compute model metrics for FPL predictions."""
    from src.evaluation.metrics import compute_metrics, compute_metrics_per_cutoff

    if pred_cols_map is None:
        pred_cols_map = {
            "global_mean": ("global_goals_pred", "global_assists_pred"),
            "player_mean": ("player_mean_goals_pred", "player_mean_assists_pred"),
            "recent_form": ("recent_form_goals_pred", "recent_form_assists_pred"),
            "minutes_adjusted": ("minutes_adj_goals_pred", "minutes_adj_assists_pred"),
            "glm": ("glm_goals_pred", "glm_assists_pred"),
        }

    return compute_metrics_per_cutoff(
        predictions, 
        target_cols=("goals", "assists"),
        pred_cols_map=pred_cols_map
    )


def aggregate_fpl_metrics(
    metrics_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate metrics across cutoffs."""
    from src.evaluation.metrics import compute_aggregate_metrics

    pred_cols_map = {
        "global_mean": ("global_goals_pred", "global_assists_pred"),
        "player_mean": ("player_mean_goals_pred", "player_mean_assists_pred"),
        "recent_form": ("recent_form_goals_pred", "recent_form_assists_pred"),
        "minutes_adjusted": ("minutes_adj_goals_pred", "minutes_adj_assists_pred"),
        "glm": ("glm_goals_pred", "glm_assists_pred"),
    }

    return compute_aggregate_metrics(
        predictions,
        target_cols=("goals", "assists"),
        pred_cols_map=pred_cols_map
    )


if __name__ == "__main__":
    # Quick test
    config = _make_fpl_evaluation_config()
    results = run_fpl_glm_evaluation(config)
    print("GLM Evaluation Results:")
    print(f"GLM preds: {len(results['glm'])} rows")
    print(f"Baseline preds: {len(results['baselines'])} rows")
    print(f"Combined: {len(results['combined'])} rows")