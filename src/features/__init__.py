"""Feature engineering modules for Premier League Intelligence."""

from src.features.player_aggregation import (
    aggregate_match_players,
    build_player_match_frame,
    build_player_season_frame,
    compute_minutes,
    dedupe_events,
    lineup_lookup,
    match_end_minute,
    starting_players,
)

from src.features.player_features import (
    FeatureConfig,
    compute_features_for_cutoff,
    compute_player_features,
    get_position_band,
)

from src.features.baselines import (
    BaselineConfig,
    compute_global_mean,
    compute_minutes_adjusted,
    compute_player_mean,
    compute_recent_form,
    estimate_expected_minutes,
    evaluate_baselines_at_cutoff,
)

__all__ = [
    # player_aggregation
    "aggregate_match_players",
    "build_player_match_frame",
    "build_player_season_frame",
    "compute_minutes",
    "dedupe_events",
    "lineup_lookup",
    "match_end_minute",
    "starting_players",
    # player_features
    "FeatureConfig",
    "compute_features_for_cutoff",
    "compute_player_features",
    "get_position_band",
    # baselines
    "BaselineConfig",
    "compute_global_mean",
    "compute_minutes_adjusted",
    "compute_player_mean",
    "compute_recent_form",
    "estimate_expected_minutes",
    "evaluate_baselines_at_cutoff",
]