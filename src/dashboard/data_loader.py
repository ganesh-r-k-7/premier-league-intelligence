"""Data loading utilities for the Premier League Intelligence Dashboard.

Provides cached data loading for all dashboard pages with multi-season support.
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


# Season ID mapping for season name to season_id
SEASON_ID_MAP = {
    "2010/11": 2010,
    "2011/12": 2011,
    "2012/13": 2012,
    "2013/14": 2013,
    "2014/15": 2014,
    "2015/16": 2015,
    "2016/17": 2016,
    "2017/18": 2017,
    "2018/19": 2018,
    "2019/20": 2019,
    "2020/21": 2020,
    "2021/22": 2021,
    "2022/23": 2022,
    "2023/24": 2023,
    "2024/25": 2024,
    "2025/26": 2025,
    "2026/27": 2026,
}

# Available data sources and their seasons
DATA_SOURCES = {
    "StatsBomb": {
        "player_season": ["2015/16"],
        "matches": ["2015/16", "2003/04"],  # StatsBomb fixtures
        "description": "Event-derived player and match data (2015/16 complete)",
    },
    "OpenFootball": {
        "matches": [f"20{i:02d}/{i+1:02d}" for i in range(10, 26)] + ["2025/26"],
        "description": "Match results from openfootball/football.json (2010/11–2025/26)",
    },
    "FPL": {
        "player_season": [f"20{i:02d}/{i+1:02d}" for i in range(16, 26)],
        "player_match": [f"20{i:02d}/{i+1:02d}" for i in range(16, 26)],
        "description": "FPL gameweek/player statistics (2016/17–2025/26)",
    },
}


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
def load_player_season_statsbomb() -> pd.DataFrame:
    """Load StatsBomb player season data (2015/16 only)."""
    if not PLAYER_SEASON_PARQUET.exists():
        raise DataLoadError(
            f"Player season data not found at {PLAYER_SEASON_PARQUET}. "
            "Run the ingestion pipeline first (see README)."
        )
    return pd.read_parquet(PLAYER_SEASON_PARQUET)


@lru_cache(maxsize=1)
def load_player_match_statsbomb() -> pd.DataFrame:
    """Load StatsBomb player match data (2015/16 only)."""
    if not (STATSBOMB_PROCESSED_DIR / "fixtures.parquet").exists():
        raise DataLoadError(
            f"Fixtures data not found. Run the ingestion pipeline first."
        )
    return pd.read_parquet(PLAYER_MATCH_PARQUET)


@lru_cache(maxsize=1)
def load_fixtures_statsbomb(season_id: int | None = None) -> pd.DataFrame:
    """Load StatsBomb fixtures data."""
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
def load_matches_openfootball() -> pd.DataFrame:
    """Load OpenFootball match results (2010/11–2025/26)."""
    matches_path = STATSBOMB_PROCESSED_DIR.parent / "matches.parquet"
    if not matches_path.exists():
        raise DataLoadError(
            f"OpenFootball matches data not found at {matches_path}. "
            "Run the multi-season ingestion pipeline first."
        )
    return pd.read_parquet(matches_path)


@lru_cache(maxsize=1)
def load_player_season_fpl() -> pd.DataFrame:
    """Load FPL player season data (2016/17–2025/26)."""
    fpl_path = STATSBOMB_PROCESSED_DIR.parent / "fpl_player_season.parquet"
    if not fpl_path.exists():
        raise DataLoadError(
            f"FPL player season data not found at {fpl_path}. "
            "Run the multi-season ingestion pipeline first."
        )
    return pd.read_parquet(fpl_path)


@lru_cache(maxsize=1)
def load_player_match_fpl() -> pd.DataFrame:
    """Load FPL player match (gameweek) data."""
    fpl_path = STATSBOMB_PROCESSED_DIR.parent / "fpl_player_match.parquet"
    if not fpl_path.exists():
        raise DataLoadError(
            f"FPL player match data not found at {fpl_path}. "
            "Run the multi-season ingestion pipeline first."
        )
    return pd.read_parquet(fpl_path)


@lru_cache(maxsize=1)
def load_fixtures_openfootball(season_name: str | None = None) -> pd.DataFrame:
    """Load OpenFootball fixtures for a specific season."""
    df = load_matches_openfootball()
    if season_name is not None:
        return df[df["season_name"] == season_name].copy()
    return df


@lru_cache(maxsize=1)
def load_player_season_fpl_filtered(season_name: str | None = None) -> pd.DataFrame:
    """Load FPL player season data, optionally filtered by season."""
    df = load_player_season_fpl()
    if season_name is not None:
        return df[df["season_name"] == season_name].copy()
    return df


@lru_cache(maxsize=1)
def load_player_match_fpl_filtered(season_name: str | None = None) -> pd.DataFrame:
    """Load FPL player match data, optionally filtered by season."""
    df = load_player_match_fpl()
    if season_name is not None:
        return df[df["season_name"] == season_name].copy()
    return df


@lru_cache(maxsize=1)
def load_player_profiles(
    min_minutes: int = 450,
    min_appearances: int = 5,
    min_xg_samples: int = 3,
    source: str = "StatsBomb",
    season_name: str | None = None,
) -> pd.DataFrame:
    """Load and build player scouting profiles from specified source and season.

    Args:
        min_minutes: Minimum minutes for inclusion in rankings.
        min_appearances: Minimum appearances for inclusion.
        min_xg_samples: Minimum matches with xG for xG metrics.
        source: Data source ("StatsBomb" or "FPL").
        season_name: Specific season to load (e.g., "2015/16").

    Returns:
        DataFrame with player profiles including per-90 rates and flags.
    """
    if source == "StatsBomb":
        player_season = load_player_season_statsbomb()
        if season_name is not None:
            player_season = player_season[player_season["season_name"] == season_name]
    elif source == "FPL":
        player_season = load_player_season_fpl()
        if season_name is not None:
            player_season = player_season[player_season["season_name"] == season_name]
    else:
        raise ValueError(f"Unknown source: {source}")

    # Normalize FPL column names to match StatsBomb column names for build_player_profiles
    if source == "FPL":
        player_season = player_season.rename(columns={
            "expected_goals": "xg",
            "expected_assists": "xa",
            "expected_goals_per_90": "xg_per_90",
            "expected_assists_per_90": "xa_per_90",
        })

    config = ScoutingConfig(
        min_minutes=min_minutes,
        min_appearances=min_appearances,
        min_xg_samples=min_xg_samples,
    )
    return build_player_profiles(player_season, config)


def get_available_sources() -> dict[str, dict]:
    """Get available data sources and their seasons."""
    return DATA_SOURCES


def get_available_seasons(source: str | None = None) -> list[str]:
    """Get available seasons, optionally filtered by source."""
    if source is None:
        all_seasons = set()
        for src_info in DATA_SOURCES.values():
            for season_list in src_info.values():
                if isinstance(season_list, list):
                    all_seasons.update(season_list)
        return sorted(all_seasons, reverse=True)
    else:
        seasons = set()
        for season_list in DATA_SOURCES.get(source, {}).values():
            if isinstance(season_list, list):
                seasons.update(season_list)
        return sorted(seasons, reverse=True)


def get_season_info(source: str = "StatsBomb", season_name: str = "2015/16") -> dict:
    """Get season metadata for display.

    Args:
        source: Data source ("StatsBomb", "OpenFootball", "FPL").
        season_name: Season name (e.g., "2015/16").

    Returns:
        Dictionary with season information.
    """
    if source == "StatsBomb":
        fixtures = load_fixtures_statsbomb()
        player_season = load_player_season_statsbomb()
    elif source == "OpenFootball":
        fixtures = load_fixtures_openfootball(season_name)
        player_season = None
    elif source == "FPL":
        fixtures = load_fixtures_openfootball(season_name)
        player_season = load_player_season_fpl_filtered(season_name)
    else:
        raise ValueError(f"Unknown source: {source}")

    info = {
        "season_name": season_name,
        "season_id": SEASON_ID_MAP.get(season_name, 0),
        "source": source,
        "competition": "Premier League",
    }

    if fixtures is not None and len(fixtures) > 0:
        info["n_matches"] = len(fixtures)
        info["n_teams"] = fixtures["home_team_id"].nunique()
    else:
        info["n_matches"] = 0
        info["n_teams"] = 0

    if player_season is not None and len(player_season) > 0:
        info["n_players"] = player_season["player_id"].nunique()
    else:
        info["n_players"] = 0

    return info


def get_team_names(source: str = "StatsBomb", season_name: str | None = None) -> list[str]:
    """Get sorted list of team names for a source and season."""
    if source == "StatsBomb":
        fixtures = load_fixtures_statsbomb()
    elif source == "OpenFootball":
        fixtures = load_fixtures_openfootball(season_name)
    elif source == "FPL":
        fixtures = load_fixtures_openfootball(season_name)
    else:
        raise ValueError(f"Unknown source: {source}")

    if fixtures is None or len(fixtures) == 0:
        return []

    home = fixtures["home_team_name"].unique()
    away = fixtures["away_team_name"].unique()
    return sorted(set(home) | set(away))


def get_position_bands() -> list[str]:
    """Get list of position bands in display order."""
    from src.features.player_scouting import POSITION_BAND_ORDER
    return [b for b in POSITION_BAND_ORDER if b != "UNKNOWN"]


def get_players_for_team(team_name: str, source: str = "StatsBomb", season_name: str | None = None) -> pd.DataFrame:
    """Get players for a specific team from specified source and season."""
    profiles = load_player_profiles(source=source, season_name=season_name)
    return profiles[profiles["team_name"] == team_name]


def get_players_by_position(band: str, source: str = "StatsBomb", season_name: str | None = None) -> pd.DataFrame:
    """Get players by position band from specified source and season."""
    profiles = load_player_profiles(source=source, season_name=season_name)
    return profiles[profiles["position_band"] == band]


def search_players(query: str, source: str = "StatsBomb", season_name: str | None = None) -> pd.DataFrame:
    """Search players by name (case-insensitive partial match)."""
    profiles = load_player_profiles(source=source, season_name=season_name)
    mask = profiles["player_name"].str.contains(query, case=False, na=False)
    return profiles[mask]


def get_comparison_metrics(source: str = "StatsBomb", season_name: str | None = None) -> list[tuple[str, str]]:
    """Get available comparison metrics with display names for a source/season."""
    profiles = load_player_profiles(source=source, season_name=season_name)
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
        ("starts_rate", "Starts Rate"),
        ("expected_goals", "xG (FPL)"),
        ("expected_assists", "xA (FPL)"),
        ("expected_goals_per_90", "xG per 90 (FPL)"),
        ("expected_assists_per_90", "xA per 90 (FPL)"),
        ("goals_per_90", "Goals per 90"),
        ("assists_per_90", "Assists per 90"),
        ("goals_per_shot", "Goals per Shot"),
        ("goals_per_shot_on_target", "Goals per Shot on Target"),
    ]

    return [(col, name) for col, name in metric_options if col in available]


def get_fpl_metric_availability() -> dict[str, dict]:
    """Get FPL metric availability by season."""
    return {
        "2016/17": {"xG": False, "xA": False, "starts": False, "position": False, "team": False, "note": "Legacy schema"},
        "2017/18": {"xG": False, "xA": False, "starts": False, "position": False, "team": False, "note": "Legacy schema"},
        "2018/19": {"xG": False, "xA": False, "starts": False, "position": False, "team": False, "note": "Legacy schema"},
        "2019/20": {"xG": False, "xA": False, "starts": False, "position": True, "team": True, "note": "Transitional"},
        "2020/21": {"xG": False, "xA": False, "starts": False, "position": True, "team": True, "note": "Minimal schema"},
        "2021/22": {"xG": False, "xA": False, "starts": False, "position": True, "team": True, "note": "Minimal schema"},
        "2022/23": {"xG": True, "xA": True, "starts": True, "position": True, "team": True, "note": "Full schema"},
        "2023/24": {"xG": True, "xA": True, "starts": True, "position": True, "team": True, "note": "Full schema"},
        "2024/25": {"xG": True, "xA": True, "starts": True, "position": True, "team": True, "note": "Full schema"},
        "2025/26": {"xG": True, "xA": True, "starts": True, "position": True, "team": True, "note": "Full schema (partial)"},
    }


def get_openfootball_season_quality() -> dict[str, dict]:
    """Get OpenFootball season data quality info."""
    return {
        "2010/11": {"complete": True, "matches": 380, "note": "Complete"},
        "2011/12": {"complete": True, "matches": 380, "note": "Complete"},
        "2012/13": {"complete": True, "matches": 380, "note": "Complete"},
        "2013/14": {"complete": True, "matches": 380, "note": "Complete"},
        "2014/15": {"complete": True, "matches": 380, "note": "Complete"},
        "2015/16": {"complete": True, "matches": 380, "note": "Complete"},
        "2016/17": {"complete": True, "matches": 380, "note": "Complete"},
        "2017/18": {"complete": True, "matches": 380, "note": "Complete"},
        "2018/19": {"complete": True, "matches": 380, "note": "Complete"},
        "2019/20": {"complete": True, "matches": 380, "note": "Complete"},
        "2020/21": {"complete": True, "matches": 380, "note": "Complete"},
        "2021/22": {"complete": True, "matches": 380, "note": "Complete"},
        "2022/23": {"complete": True, "matches": 380, "note": "Complete"},
        "2023/24": {"complete": True, "matches": 380, "note": "Complete"},
        "2024/25": {"complete": True, "matches": 380, "note": "Complete"},
        "2025/26": {"complete": False, "matches": 380, "note": "Partial - some malformed scores"},
    }


def load_report_files() -> dict[str, Path]:
    """Load available report files."""
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


def get_season_coverage_summary() -> pd.DataFrame:
    """Get a summary of data coverage by season and source."""
    rows = []

    # StatsBomb
    rows.append({
        "Season": "2015/16",
        "Source": "StatsBomb",
        "Matches": 380,
        "Teams": 20,
        "Players": 550,
        "Player Match Rows": 10469,
        "Player Season Rows": 550,
        "xG": True,
        "xA": True,
        "Note": "Complete event data",
    })

    # OpenFootball
    of_quality = get_openfootball_season_quality()
    for season in sorted(of_quality.keys(), reverse=True):
        q = of_quality[season]
        rows.append({
            "Season": season,
            "Source": "OpenFootball",
            "Matches": q["matches"],
            "Teams": 20,
            "Players": 0,
            "Player Match Rows": 0,
            "Player Season Rows": 0,
            "xG": False,
            "xA": False,
            "Note": q["note"],
        })

    # FPL
    fpl_avail = get_fpl_metric_availability()
    for season in sorted(fpl_avail.keys(), reverse=True):
        f = fpl_avail[season]
        rows.append({
            "Season": season,
            "Source": "FPL",
            "Matches": 380,
            "Teams": 20,
            "Players": 0,  # Will be filled
            "Player Match Rows": 0,  # Will be filled
            "Player Season Rows": 0,  # Will be filled
            "xG": f["xG"],
            "xA": f["xA"],
            "Note": f["note"],
        })

    df = pd.DataFrame(rows)

    # Fill in FPL player counts
    try:
        fpl_ps = load_player_season_fpl()
        fpl_pm = load_player_match_fpl()
        for season in fpl_avail.keys():
            mask_ps = fpl_ps["season_name"] == season
            mask_pm = fpl_pm["season_name"] == season
            df.loc[(df["Season"] == season) & (df["Source"] == "FPL"), "Players"] = fpl_ps[mask_ps]["player_id"].nunique()
            df.loc[(df["Season"] == season) & (df["Source"] == "FPL"), "Player Match Rows"] = mask_pm.sum()
            df.loc[(df["Season"] == season) & (df["Source"] == "FPL"), "Player Season Rows"] = mask_ps.sum()
    except Exception:
        pass

    return df