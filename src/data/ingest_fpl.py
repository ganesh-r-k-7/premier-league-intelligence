"""Fantasy Premier League (FPL) historical data ingestion for Premier League Intelligence (Phase 8.2).

Downloads player-match (gameweek) and player-season data from vaastav/Fantasy-Premier-League repository,
stores immutable raw CSV snapshots under data/raw/fpl/,
and writes normalized, provenance-carrying tables under data/processed/.

Source: vaastav/Fantasy-Premier-League (https://github.com/vaastav/Fantasy-Premier-League)
License: MIT — permissive, redistribution allowed with attribution
Coverage: Premier League 2016/17 through 2025/26 (10 complete seasons)
Note: 2026/27 is incomplete (early season) and excluded by default.
Schema varies by season:
  - 2016/17–2018/19: legacy schema (no xG/xA, no starts)
  - 2019/20: transitional schema
  - 2020/21–2021/22: minimal schema
  - 2022/23+: full schema with xG/xA and starts
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.data import config

logger = logging.getLogger("plintel.ingest.fpl")

SOURCE_NAME = "vaastav/Fantasy-Premier-League"
SOURCE_URL = "https://github.com/vaastav/Fantasy-Premier-League"
SOURCE_LICENSE = "MIT"
DEFAULT_TIMEOUT = 60
USER_AGENT = "premier-league-intelligence/0.3 (+https://github.com/vaastav/Fantasy-Premier-League)"

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Seasons with complete PL data (380 matches = 38 gameweeks × 10 matches)
COMPLETE_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24",
    "2024-25", "2025-26",
]

# Schema evolution tracking
LEGACY_SCHEMA_SEASONS = {"2016-17", "2017-18", "2018-19"}
TRANSITIONAL_SCHEMA_SEASONS = {"2019-20"}
MINIMAL_SCHEMA_SEASONS = {"2020-21", "2021-22"}
FULL_SCHEMA_SEASONS = {"2022-23", "2023-24", "2024-25", "2025-26"}


class IngestError(RuntimeError):
    """Raised when ingestion cannot complete safely."""


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _sha256_of_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> requests.Response:
    """Perform a single HTTP GET, raising IngestError on any failure."""
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        raise IngestError(f"network request to {url} failed: {exc}") from exc
    if response.status_code != 200:
        raise IngestError(f"unexpected HTTP {response.status_code} for {url}")
    return response


def download_csv_to_file(url: str, dest: Path, timeout: int = DEFAULT_TIMEOUT) -> tuple[pd.DataFrame, Path]:
    """Download CSV and return DataFrame + path. Writes raw bytes atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = http_get(url, timeout=timeout)
    content = response.content

    # Parse to validate with encoding fallback
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to latin-1 for files with special characters
        df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
    except Exception as exc:
        raise IngestError(f"response from {url} was not valid CSV: {exc}") from exc

    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(dest)
    return df, dest


def raw_merged_gw_path(season_dir: str) -> Path:
    """Return path for raw merged_gw.csv for a season."""
    return config.DATA_RAW_DIR / "fpl" / season_dir / "merged_gw.csv"


def raw_cleaned_merged_path() -> Path:
    """Return path for raw cleaned_merged_seasons.csv."""
    return config.DATA_RAW_DIR / "fpl" / "cleaned_merged_seasons.csv"


def raw_master_team_list_path() -> Path:
    """Return path for raw master_team_list.csv."""
    return config.DATA_RAW_DIR / "fpl" / "master_team_list.csv"


def download_fpl_reference_files(timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Download FPL reference files (master_team_list.csv, cleaned_merged_seasons.csv)."""
    files = {
        "master_team_list.csv": f"{BASE_URL}/master_team_list.csv",
        "cleaned_merged_seasons.csv": f"{BASE_URL}/cleaned_merged_seasons.csv",
    }
    results = {}
    for fname, url in files.items():
        path = config.DATA_RAW_DIR / "fpl" / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = http_get(url, timeout=DEFAULT_TIMEOUT)
            content = response.content
            path.write_bytes(content)
            logger.info("downloaded %s -> %s", url, path)
            results[fname] = "success"
        except Exception as exc:
            logger.error("failed to download %s: %s", url, exc)
            results[fname] = f"failed: {exc}"
    return results


def processed_fpl_player_match_path() -> Path:
    return config.DATA_PROCESSED_DIR / "fpl_player_match.parquet"


def processed_fpl_player_season_path() -> Path:
    return config.DATA_PROCESSED_DIR / "fpl_player_season.parquet"


def _load_csv_source(url: str, path: Path, timeout: int, offline: bool) -> tuple[pd.DataFrame, str]:
    """Load CSV from cached raw file or download it. Returns (df, raw_sha256)."""
    path = Path(path)
    if offline:
        if not path.exists():
            raise IngestError(f"offline mode and no cached raw file at {path}")
        logger.info("offline: using cached raw file %s", path)
        content = path.read_bytes()
        # Try UTF-8 first, fall back to latin-1 for older files with special chars
        try:
            return pd.read_csv(io.BytesIO(content), encoding="utf-8", on_bad_lines="skip"), _sha256_of_bytes(content)
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(content), encoding="latin-1", on_bad_lines="skip"), _sha256_of_bytes(content)

    try:
        df, _ = download_csv_to_file(url, path, timeout=timeout)
        content = path.read_bytes()
        logger.info("downloaded %s -> %s", url, path)
        return df, _sha256_of_bytes(content)
    except IngestError as exc:
        if path.exists():
            logger.warning("download failed (%s); using cached snapshot %s", exc, path)
            content = path.read_bytes()
            try:
                return pd.read_csv(io.BytesIO(content), encoding="utf-8", on_bad_lines="skip"), _sha256_of_bytes(content)
            except UnicodeDecodeError:
                return pd.read_csv(io.BytesIO(content), encoding="latin-1", on_bad_lines="skip"), _sha256_of_bytes(content)
        raise


def parse_season_dir(season_dir: str) -> tuple[int, str]:
    """Parse season directory name (e.g., '2016-17') into (season_id, season_name)."""
    parts = season_dir.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid season directory format: {season_dir}")
    start_year = int(parts[0])
    season_name = f"{start_year}/{parts[1]}"
    return start_year, season_name


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching across sources."""
    if pd.isna(name):
        return ""
    return str(name).strip().replace(" FC", "").replace(" AFC", "").replace(" United", " Utd").replace(" City", "")


def validate_merged_gw(df: pd.DataFrame, season_dir: str) -> list[str]:
    """Validate merged_gw DataFrame for a season. Returns list of issues."""
    issues = []
    required_cols = ["name", "element", "fixture", "minutes", "round", "total_points", "was_home", "team_a_score", "team_h_score"]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"{season_dir}: missing required column '{col}'")

    if "round" in df.columns:
        max_gw = df["round"].max()
        if max_gw < 38:
            issues.append(f"{season_dir}: only {max_gw} gameweeks found (expected 38)")

    if "element" in df.columns:
        n_players = df["element"].nunique()
        if n_players < 500:
            issues.append(f"{season_dir}: only {n_players} unique players (expected ~550+)")

    return issues


def standardize_merged_gw(df: pd.DataFrame, season_dir: str, season_id: int, season_name: str, raw_sha256: str) -> pd.DataFrame:
    """Standardize merged_gw columns across schema versions to a common schema."""
    df = df.copy()
    df["season_dir"] = season_dir
    df["season_id"] = season_id
    df["season_name"] = f"{season_id}/{str(season_id+1)[-2:]}"
    df["source"] = SOURCE_NAME
    df["source_url"] = SOURCE_URL
    df["source_license"] = SOURCE_LICENSE
    df["raw_sha256"] = raw_sha256
    df["ingested_at"] = _utcnow()

    # Determine gameweek column first (before renames to avoid conflicts)
    if "gameweek" not in df.columns:
        if "round" in df.columns:
            df["gameweek"] = df["round"]
        elif "GW" in df.columns:
            df["gameweek"] = df["GW"]
        else:
            df["gameweek"] = pd.NA

    # Column mapping for different schemas
    # Standardize column names (excluding round/GW since we handled gameweek)
    rename_map = {
        "name": "player_name",
        "element": "player_id",  # FPL element ID
        "fixture": "fixture_id",
        # "round": "gameweek",  # handled above
        # "GW": "gameweek",  # handled above
        "minutes": "minutes",
        "goals_scored": "goals",
        "assists": "assists",
        "clean_sheets": "clean_sheets",
        "goals_conceded": "goals_conceded",
        "own_goals": "own_goals",
        "penalties_missed": "penalties_missed",
        "penalties_saved": "penalties_saved",
        "red_cards": "red_cards",
        "yellow_cards": "yellow_cards",
        "saves": "saves",
        "bonus": "bonus",
        "bps": "bps",
        "influence": "influence",
        "creativity": "creativity",
        "threat": "threat",
        "ict_index": "ict_index",
        "total_points": "total_points",
        "transfers_balance": "transfers_balance",
        "transfers_in": "transfers_in",
        "transfers_out": "transfers_out",
        "value": "value",
        "was_home": "was_home",
        "kickoff_time": "kickoff_time",
        "opponent_team": "opponent_team_id",
        "team_a_score": "team_a_score",
        "team_h_score": "team_h_score",
        "selected": "selected",
    }

    # Apply renames for columns that exist
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    # Handle schema-specific columns
    # xG/xA (full schema only)
    for col in ["expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded"]:
        if col not in df.columns:
            df[col] = pd.NA

    # starts (full schema only)
    if "starts" not in df.columns:
        df["starts"] = pd.NA

    # position and team (full/transitional schema)
    if "position" not in df.columns:
        df["position"] = pd.NA
    if "team" not in df.columns:
        df["team"] = pd.NA

    # Determine if home/away based on was_home
    df["is_home"] = df["was_home"].astype(bool) if "was_home" in df.columns else pd.NA

    # Standardize types
    if "kickoff_time" in df.columns:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], errors="coerce")

    # Ensure numeric columns
    numeric_cols = ["minutes", "goals", "assists", "clean_sheets", "goals_conceded",
                    "own_goals", "penalties_missed", "penalties_saved", "red_cards",
                    "yellow_cards", "saves", "bonus", "bps", "influence", "creativity",
                    "threat", "ict_index", "total_points", "transfers_balance",
                    "transfers_in", "transfers_out", "value", "selected",
                    "expected_goals", "expected_assists", "expected_goal_involvements",
                    "expected_goals_conceded"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "starts" in df.columns:
        df["starts"] = pd.to_numeric(df["starts"], errors="coerce").astype("Int64")

    # Ensure gameweek is numeric
    if "gameweek" in df.columns:
        df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce").astype("Int64")

    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

    if "fixture_id" in df.columns:
        df["fixture_id"] = pd.to_numeric(df["fixture_id"], errors="coerce").astype("Int64")

    return df


def build_player_season_from_match(player_match_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player_match (gameweek) data to player_season level."""
    # Group by season, player, team
    group_cols = ["season_id", "season_name", "player_id", "player_name"]

    # We need team info - for now aggregate by player only, team will be most frequent
    agg_dict = {
        "minutes": "sum",
        "goals": "sum",
        "assists": "sum",
        "clean_sheets": "sum",
        "goals_conceded": "sum",
        "own_goals": "sum",
        "penalties_missed": "sum",
        "penalties_saved": "sum",
        "red_cards": "sum",
        "yellow_cards": "sum",
        "saves": "sum",
        "bonus": "sum",
        "bps": "sum",
        "influence": "mean",
        "creativity": "mean",
        "threat": "mean",
        "ict_index": "mean",
        "total_points": "sum",
        "expected_goals": "sum",
        "expected_assists": "sum",
        "expected_goal_involvements": "sum",
        "expected_goals_conceded": "sum",
        "gameweek": "count",  # appearances
    }

    # Only include columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in player_match_df.columns}

    # Group and aggregate
    player_season = player_match_df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Rename gameweek count to appearances
    if "gameweek" in player_season.columns:
        player_season = player_season.rename(columns={"gameweek": "appearances"})

    # Add starts if available
    if "starts" in player_match_df.columns:
        starts_sum = player_match_df.groupby(group_cols)["starts"].sum().reset_index()
        player_season = player_season.merge(starts_sum, on=group_cols, how="left")

    # Add position and team (most frequent per player-season)
    for col in ["position", "team"]:
        if col in player_match_df.columns:
            mode_series = player_match_df.groupby(group_cols)[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA).reset_index()
            player_season = player_season.merge(mode_series, on=group_cols, how="left")

    # Add per-90 metrics
    if "minutes" in player_season.columns:
        for col in ["goals", "assists", "expected_goals", "expected_assists"]:
            if col in player_season.columns:
                player_season[f"{col}_per_90"] = player_season.apply(
                    lambda r: (r[col] / r["minutes"] * 90) if r["minutes"] > 0 else pd.NA, axis=1
                )

    player_season["source"] = SOURCE_NAME
    player_season["source_url"] = SOURCE_URL
    player_season["source_license"] = SOURCE_LICENSE
    player_season["ingested_at"] = _utcnow()

    return player_season


def run(
    offline: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    seasons: Optional[list[str]] = None,
    skip_incomplete: bool = True,
) -> dict:
    """Run the FPL historical data ingestion.

    Args:
        offline: Use cached raw files only; never access the network.
        timeout: HTTP timeout in seconds.
        seasons: Specific season directories to process (e.g., ["2016-17", "2017-18"]).
                 Defaults to COMPLETE_SEASONS.
        skip_incomplete: Skip seasons not in COMPLETE_SEASONS (e.g., 2026-27).

    Returns:
        Summary dict with counts and status.
    """
    if not offline:
        download_fpl_reference_files(timeout=timeout)

    season_dirs = seasons or COMPLETE_SEASONS

    if skip_incomplete:
        season_dirs = [s for s in season_dirs if s in COMPLETE_SEASONS]

    all_player_match = []
    summary = {
        "seasons_requested": len(season_dirs),
        "seasons_processed": 0,
        "seasons_failed": 0,
        "player_match_rows": 0,
        "player_season_rows": 0,
        "players_by_season": {},
        "failures": [],
    }

    for season_dir in season_dirs:
        season_id, season_name = parse_season_dir(season_dir)
        label = f"{season_name} ({season_dir})"

        try:
            # Load merged_gw.csv
            url = f"{BASE_URL}/{season_dir}/gws/merged_gw.csv"
            path = raw_merged_gw_path(season_dir)
            df, raw_sha256 = _load_csv_source(url, path, timeout, offline)

            issues = validate_merged_gw(df, season_dir)
            if issues:
                logger.warning("validation issues for %s: %s", label, "; ".join(issues))

            # Standardize
            std_df = standardize_merged_gw(df, season_dir, season_id, season_name, raw_sha256)
            all_player_match.append(std_df)

            summary["players_by_season"][season_dir] = std_df["player_id"].nunique()
            summary["seasons_processed"] += 1

        except IngestError as exc:
            logger.error("failed to process %s: %s", label, exc)
            summary["failures"].append({"season": season_dir, "error": str(exc)})
            summary["seasons_failed"] += 1
            continue

    if not all_player_match:
        raise IngestError("no validated player-match data available; nothing written")

    # Combine all seasons
    player_match_df = pd.concat(all_player_match, ignore_index=True)

    # Build player_season
    player_season_df = build_player_season_from_match(player_match_df)

    # Write outputs
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pm_path = processed_fpl_player_match_path()
    player_match_df.to_parquet(pm_path, index=False)
    logger.info("wrote %d player-match rows to %s", len(player_match_df), pm_path)

    ps_path = processed_fpl_player_season_path()
    player_season_df.to_parquet(ps_path, index=False)
    logger.info("wrote %d player-season rows to %s", len(player_season_df), ps_path)

    summary["player_match_rows"] = len(player_match_df)
    summary["player_season_rows"] = len(player_season_df)

    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest FPL historical data.")
    parser.add_argument("--offline", action="store_true", help="use cached raw files only")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--season", action="append", dest="seasons", help="specific season(s) to process (e.g., 2016-17)")
    parser.add_argument("--include-incomplete", action="store_true", help="include incomplete seasons like 2026-27")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        summary = run(
            offline=args.offline,
            timeout=args.timeout,
            seasons=args.seasons,
            skip_incomplete=not args.include_incomplete,
        )
    except IngestError as exc:
        logger.error("ingestion failed: %s", exc)
        return 1

    logger.info("ingestion complete: %d seasons, %d player-match rows, %d player-season rows",
                summary["seasons_processed"], summary["player_match_rows"], summary["player_season_rows"])
    if summary["failures"]:
        logger.warning("failures: %s", summary["failures"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())