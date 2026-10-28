"""Data loading utilities for the Premier League Intelligence Dashboard.

Provides cached data loading for all dashboard pages.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.config import (
    PLAYER_MATCH_PARQUET,
    PLAYER_SEASON_PARQUET,
    STATSBOMB_PROCESSED_DIR,
    TARGET_SEASON_ID,
    TARGET_SEASON_NAME,
)
from src.features.player_scouting import build_player_profiles, ScoutingConfig


class DataLoadError(Exception):
    """Raised when data files cannot be loaded."""

    pass


def _check_data_exists() -> tuple[bool, list[str]]:
    """Check if required data files exist.

    Returns:
        Tuple of (all_exist, missing_files_list).
    """
    required = [
        PLAYER_SEASON_PARQUET,
        PLAYER_MATCH_PARQUET,
        STATSBOMB_PROCESSED_DIR / "fixtures.parquet",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return len(missing) == 0, missing


@lru_cache(maxsize=1)
def load_player_season() -> pd.DataFrame:
    """Load player season data with provenance.

    Returns:
        DataFrame with one row per player-season.

    Raises:
        DataLoadError: If the file doesn't exist.
    """
    if not PLAYER_SEASON_PARQUET.exists():
        raise DataLoadError(
            f"Player season data not found at {PLAYER_SEASON_PARQUET}. "
            "Run the ingestion pipeline first (see README)."
        )
    return pd.read_parquet(PLAYER_SEASON_PARQUET)


@lru_cache(maxsize=1)
def load_player_match() -> pd.DataFrame:
    """Load player match data.

    Returns:
        DataFrame with one row per player-match.

    Raises:
        DataLoadError: If the file doesn't exist.
    """
    if not (STATSBOMB_PROCESSED_DIR / "fixtures.parquet").exists():
        raise DataLoadError(
            f"Fixtures data not found. Run the ingestion pipeline first."
        )
    return pd.read_parquet(PLAYER_MATCH_PARQUET)


@lru_cache(maxsize=1)
def load_fixtures(season_id: int | None = None) -> pd.DataFrame:
    """Load fixtures data.

    Args:
        season_id: If provided, filter to this season. Defaults to TARGET_SEASON_ID.

    Returns:
        DataFrame with fixtures for the specified season (default: 2015/2016).

    Raises:
        DataLoadError: If the file doesn't exist.
    """
    fixtures_path = STATSBOMB_PROCESSED_DIR / "fixtures.parquet"
    if not fixtures_path.exists():
        raise DataLoadError(
            f"Fixtures data not found at {fixtures_path}. "
            "Run the ingestion pipeline first (see README)."
        )
    fixtures = pd.read_parquet(fixtures_path)
    if season_id is None:
        season_id = TARGET_SEASON_ID
    return fixtures[fixtures["season_id"] == season_id].copy()


@lru_cache(maxsize=1)
def load_player_profiles(
    min_minutes: int = 450,
    min_appearances: int = 5,
    min_xg_samples: int = 3,
) -> pd.DataFrame:
    """Load and build player scouting profiles.

    Args:
        min_minutes: Minimum minutes for inclusion in rankings.
        min_appearances: Minimum appearances for inclusion.
        min_xg_samples: Minimum matches with xG for xG metrics.

    Returns:
        DataFrame with player profiles including per-90 rates and flags.
    """
    player_season = load_player_season()
    config = ScoutingConfig(
        min_minutes=min_minutes,
        min_appearances=min_appearances,
        min_xg_samples=min_xg_samples,
    )
    return build_player_profiles(player_season, config)


def get_season_info() -> dict:
    """Get season metadata for display.

    Returns:
        Dictionary with season information.
    """
    fixtures = load_fixtures()
    player_season = load_player_season()

    return {
        "season_name": TARGET_SEASON_NAME,
        "season_id": TARGET_SEASON_ID,
        "n_matches": len(fixtures),
        "n_teams": fixtures["home_team_id"].nunique(),
        "n_players": player_season["player_id"].nunique(),
        "competition": "Premier League",
        "source": "StatsBomb Open Data",
    }


def get_team_names(season_id: int | None = None) -> list[str]:
    """Get sorted list of team names.

    Args:
        season_id: If provided, filter to this season. Defaults to TARGET_SEASON_ID.

    Returns:
        List of team names.
    """
    fixtures = load_fixtures(season_id)
    home = fixtures["home_team_name"].unique()
    away = fixtures["away_team_name"].unique()
    return sorted(set(home) | set(away))


def get_position_bands() -> list[str]:
    """Get list of position bands in display order.

    Returns:
        List of position bands.
    """
    from src.features.player_scouting import POSITION_BAND_ORDER
    return [b for b in POSITION_BAND_ORDER if b != "UNKNOWN"]


def get_players_for_team(team_name: str) -> pd.DataFrame:
    """Get players for a specific team.

    Args:
        team_name: Name of the team.

    Returns:
        DataFrame of players for that team.
    """
    profiles = load_player_profiles()
    return profiles[profiles["team_name"] == team_name]


def get_players_by_position(band: str) -> pd.DataFrame:
    """Get players by position band.

    Args:
        band: Position band (GK, DEF, MID, WING, FWD).

    Returns:
        DataFrame of players in that position band.
    """
    profiles = load_player_profiles()
    return profiles[profiles["position_band"] == band]


def search_players(query: str) -> pd.DataFrame:
    """Search players by name (case-insensitive partial match).

    Args:
        query: Search string.

    Returns:
        DataFrame of matching players.
    """
    profiles = load_player_profiles()
    mask = profiles["player_name"].str.contains(query, case=False, na=False)
    return profiles[mask]


def get_comparison_metrics() -> list[tuple[str, str]]:
    """Get available comparison metrics with display names.

    Returns:
        List of (column_name, display_name) tuples.
    """
    profiles = load_player_profiles()
    available = set(profiles.columns)

    metric_options = [
        ("minutes", "Minutes"),
        ("appearances", "Appearances"),
        ("starts", "Starts"),
        ("starts_rate", "Starts Rate"),
        ("goals", "Goals"),
        ("goals_per_90", "Goals per 90"),
        ("assists", "Assists"),
        ("assists_per_90", "Assists per 90"),
        ("xg", "xG"),
        ("xg_per_90", "xG per 90"),
        ("npxg", "npxG"),
        ("npxg_per_90", "npxG per 90"),
        ("xa", "xA"),
        ("xa_per_90", "xA per 90"),
        ("shots", "Shots"),
        ("shots_per_90", "Shots per 90"),
        ("shots_on_target", "Shots on Target"),
        ("shots_on_target_per_90", "Shots on Target per 90"),
        ("goals_per_shot", "Goals per Shot"),
        ("goals_per_shot_on_target", "Goals per Shot on Target"),
        ("goals_over_xg", "Goals over xG"),
        ("assists_over_xa", "Assists over xA"),
        ("minutes_per_match", "Minutes per Match"),
    ]

    return [(col, name) for col, name in metric_options if col in available]


def load_report_files() -> dict[str, Path]:
    """Load available report files.

    Returns:
        Dictionary mapping report name to path.
    """
    reports_dir = STATSBOMB_PROCESSED_DIR.parent.parent / "reports"
    if not reports_dir.exists():
        return {}

    report_files = {}
    for f in reports_dir.glob("*.md"):
        report_files[f.stem] = f
    for f in reports_dir.glob("*.parquet"):
        if f.stem not in report_files:
            report_files[f.stem] = f
    return report_files