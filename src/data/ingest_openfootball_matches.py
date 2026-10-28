"""OpenFootball match results ingestion for Premier League Intelligence (Phase 8.2).

Downloads match results from openfootball/football.json repository,
stores immutable raw JSON snapshots under data/raw/openfootball/matches/,
and writes normalized, provenance-carrying matches table under data/processed/.

Source: openfootball/football.json (https://github.com/openfootball/football.json)
License: CC0 (Public Domain) — no restrictions
Coverage: Premier League 2010/11 through 2026/27 (17 seasons, 380 matches each)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.data import config

logger = logging.getLogger("plintel.ingest.openfootball_matches")

SOURCE_NAME = "openfootball/football.json"
SOURCE_URL = "https://github.com/openfootball/football.json"
SOURCE_LICENSE = "CC0-1.0"
DEFAULT_TIMEOUT = 30
USER_AGENT = "premier-league-intelligence/0.3 (+https://github.com/openfootball/football.json)"

BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"


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


def download_json_to_file(url: str, dest: Path, timeout: int = DEFAULT_TIMEOUT) -> tuple[dict, Path]:
    """Download URL as JSON and atomically write raw bytes to dest.

    Returns (payload, dest_path). Writes raw bytes only after the response
    has been received and parsed, so a partial download never replaces a good snapshot.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = http_get(url, timeout=timeout)
    try:
        payload = response.json()
    except ValueError as exc:
        raise IngestError(f"response from {url} was not valid JSON: {exc}") from exc

    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(response.content)
    tmp.replace(dest)
    return payload, dest


def raw_match_path(season_dir: str) -> Path:
    """Return path for raw match JSON for a season directory (e.g., '2015-16')."""
    return config.DATA_RAW_DIR / "openfootball" / "matches" / f"{season_dir}.json"


def processed_matches_path() -> Path:
    """Return path for processed matches parquet."""
    return config.DATA_PROCESSED_DIR / "matches.parquet"


def processed_teams_path() -> Path:
    """Return path for processed teams parquet."""
    return config.DATA_PROCESSED_DIR / "teams.parquet"


def processed_seasons_path() -> Path:
    """Return path for processed seasons parquet."""
    return config.DATA_PROCESSED_DIR / "seasons.parquet"


def list_available_seasons() -> list[str]:
    """Return list of available season directories from the football.json repo.

    In practice we know the directory names from the GitHub API, but this
    provides a discoverable interface.
    """
    # Known Premier League seasons in football.json (2010/11 through 2026/27)
    return [
        "2010-11", "2011-12", "2012-13", "2013-14", "2014-15",
        "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
        "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
        "2025-26", "2026-27",
    ]


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching across sources."""
    return name.strip().replace(" FC", "").replace(" AFC", "").replace(" United", " Utd").replace(" City", "")


def parse_season_dir(season_dir: str) -> tuple[int, str]:
    """Parse season directory name (e.g., '2015-16') into (season_id, season_name).

    season_id is the starting year (e.g., 2015 for 2015-16).
    """
    parts = season_dir.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid season directory format: {season_dir}")
    start_year = int(parts[0])
    season_name = f"{start_year}/{parts[1]}"
    return start_year, season_name


def validate_match_payload(payload: dict, label: str) -> list[str]:
    """Validate a football.json match payload.

    Returns list of issues (empty if valid).
    """
    issues = []
    if not isinstance(payload, dict):
        issues.append(f"{label}: payload is not a dict")
        return issues
    if "name" not in payload:
        issues.append(f"{label}: missing 'name' field")
    if "matches" not in payload:
        issues.append(f"{label}: missing 'matches' field")
        return issues
    if not isinstance(payload["matches"], list):
        issues.append(f"{label}: 'matches' is not a list")
        return issues

    for i, match in enumerate(payload["matches"]):
        if not isinstance(match, dict):
            issues.append(f"{label}: match {i} is not a dict")
            continue
        required = ["round", "date", "team1", "team2", "score"]
        for field in required:
            if field not in match:
                issues.append(f"{label}: match {i} missing '{field}'")
        if "score" in match:
            score = match["score"]
            if not isinstance(score, dict):
                # Allow non-dict scores for unplayed matches (e.g., null or string)
                pass
            else:
                # Allow missing ft/ht for unplayed matches - just warn
                if "ft" not in score or "ht" not in score:
                    pass  # Allow missing scores for unplayed matches
                else:
                    ft = score.get("ft")
                    ht = score.get("ht")
                    if ft is not None and not (isinstance(ft, list) and len(ft) == 2 and all(isinstance(x, (int, float)) for x in ft)):
                        issues.append(f"{label}: match {i} score.ft invalid")
                    if ht is not None and not (isinstance(ht, list) and len(ht) == 2 and all(isinstance(x, (int, float)) for x in ht)):
                        issues.append(f"{label}: match {i} score.ht invalid")
    return issues


def matches_to_records(
    payload: dict,
    season_id: int,
    season_name: str,
    season_dir: str,
    ingested_at: str,
    raw_sha256: str,
) -> list[dict]:
    """Flatten a football.json matches payload into provenance-carrying records."""
    records = []
    for match in payload["matches"]:
        score = match.get("score", {})
        ft = score.get("ft") if isinstance(score, dict) else None
        ht = score.get("ht") if isinstance(score, dict) else None

        # Handle missing scores for unplayed matches
        home_score_ft = ft[0] if isinstance(ft, list) and len(ft) == 2 else None
        away_score_ft = ft[1] if isinstance(ft, list) and len(ft) == 2 else None
        home_score_ht = ht[0] if isinstance(ht, list) and len(ht) == 2 else None
        away_score_ht = ht[1] if isinstance(ht, list) and len(ht) == 2 else None

        records.append(
            {
                "source": SOURCE_NAME,
                "source_url": SOURCE_URL,
                "source_license": SOURCE_LICENSE,
                "raw_file": f"openfootball/matches/{season_dir}.json",
                "raw_sha256": raw_sha256,
                "season_id": season_id,
                "season_name": season_name,
                "competition": "Premier League",
                "match_date": match["date"],
                "match_time": match.get("time"),
                "match_week": None,  # football.json doesn't provide match week
                "round": match.get("round"),
                "home_team_name": match["team1"],
                "away_team_name": match["team2"],
                "home_score_ft": home_score_ft,
                "away_score_ft": away_score_ft,
                "home_score_ht": home_score_ht,
                "away_score_ht": away_score_ht,
                "ingested_at": ingested_at,
            }
        )
    return records


def build_teams_dataframe(all_records: list[dict]) -> pd.DataFrame:
    """Build teams reference table from match records."""
    team_names = set()
    for r in all_records:
        team_names.add(r["home_team_name"])
        team_names.add(r["away_team_name"])

    # Create stable team_id based on normalized name
    teams = []
    for name in sorted(team_names):
        norm = normalize_team_name(name)
        # Create a stable hash-based ID
        team_id = int(hashlib.md5(norm.lower().encode()).hexdigest()[:8], 16)
        teams.append(
            {
                "team_id": team_id,
                "team_name": name,
                "normalized_name": norm,
                "source": SOURCE_NAME,
            }
        )
    return pd.DataFrame(teams)


def build_seasons_dataframe(seasons: list[tuple[int, str, str]]) -> pd.DataFrame:
    """Build seasons reference table.

    Args:
        seasons: List of (season_id, season_name, season_dir)
    """
    rows = []
    for season_id, season_name, season_dir in seasons:
        rows.append(
            {
                "season_id": season_id,
                "season_name": season_name,
                "season_dir": season_dir,
                "competition": "Premier League",
                "source": SOURCE_NAME,
            }
        )
    return pd.DataFrame(rows)


def _load_json_source(url: str, path: Path, timeout: int, offline: bool) -> tuple[dict, str]:
    """Load JSON from cached raw file or download it.

    Returns (payload, raw_sha256).
    """
    path = Path(path)
    if offline:
        if not path.exists():
            raise IngestError(f"offline mode and no cached raw file at {path}")
        logger.info("offline: using cached raw file %s", path)
        content = path.read_bytes()
        return json.loads(content), _sha256_of_bytes(content)

    try:
        payload, _ = download_json_to_file(url, path, timeout=timeout)
        content = path.read_bytes()
        logger.info("downloaded %s -> %s", url, path)
        return payload, _sha256_of_bytes(content)
    except IngestError as exc:
        if path.exists():
            logger.warning("download failed (%s); using cached snapshot %s", exc, path)
            content = path.read_bytes()
            return json.loads(content), _sha256_of_bytes(content)
        raise


def run(
    offline: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    season_limit: Optional[int] = None,
    seasons: Optional[list[str]] = None,
) -> dict:
    """Run the OpenFootball match results ingestion.

    Args:
        offline: Use cached raw files only; never access the network.
        timeout: HTTP timeout in seconds.
        season_limit: If set, only process the first N seasons.
        seasons: If set, only process these specific season directories.

    Returns:
        Summary dict with counts and status.
    """
    season_dirs = seasons or list_available_seasons()
    if season_limit is not None:
        season_dirs = season_dirs[:season_limit]

    all_records = []
    season_metadata = []  # (season_id, season_name, season_dir)
    summary = {
        "seasons_requested": len(season_dirs),
        "seasons_processed": 0,
        "seasons_failed": 0,
        "matches_total": 0,
        "matches_by_season": {},
        "failures": [],
    }

    for season_dir in season_dirs:
        season_id, season_name = parse_season_dir(season_dir)
        label = f"{season_name} ({season_dir})"
        url = f"{BASE_URL}/{season_dir}/en.1.json"
        path = raw_match_path(season_dir)

        try:
            payload, raw_sha256 = _load_json_source(url, path, timeout, offline)
            issues = validate_match_payload(payload, label)
            if issues:
                raise IngestError(f"validation failed for {label}: {'; '.join(issues)}")

            logger.info("validated matches for %s: %d rows", label, len(payload["matches"]))

            records = matches_to_records(
                payload, season_id, season_name, season_dir, _utcnow(), raw_sha256
            )
            all_records.extend(records)
            season_metadata.append((season_id, season_name, season_dir))
            summary["matches_by_season"][season_dir] = len(payload["matches"])
            summary["matches_total"] += len(payload["matches"])
            summary["seasons_processed"] += 1

        except IngestError as exc:
            logger.error("failed to process %s: %s", label, exc)
            summary["failures"].append({"season": season_dir, "error": str(exc)})
            summary["seasons_failed"] += 1
            continue

    if not all_records:
        raise IngestError("no validated match data available; nothing written")

    # Build DataFrames
    matches_df = pd.DataFrame.from_records(all_records)
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"], errors="coerce")

    # Map team names to team_ids
    teams_df = build_teams_dataframe(all_records)
    team_name_to_id = dict(zip(teams_df["team_name"], teams_df["team_id"]))
    matches_df["home_team_id"] = matches_df["home_team_name"].map(team_name_to_id)
    matches_df["away_team_id"] = matches_df["away_team_name"].map(team_name_to_id)

    # Add match_id (hash of season + round + teams)
    matches_df["match_id"] = matches_df.apply(
        lambda r: int(hashlib.md5(f"{r['season_id']}|{r['round']}|{r['home_team_name']}|{r['away_team_name']}".encode()).hexdigest()[:12], 16),
        axis=1,
    )

    # Reorder columns
    match_cols = [
        "match_id", "season_id", "season_name", "competition",
        "match_date", "match_time", "match_week", "round",
        "home_team_id", "home_team_name", "away_team_id", "away_team_name",
        "home_score_ft", "away_score_ft", "home_score_ht", "away_score_ht",
        "source", "source_url", "source_license", "raw_file", "raw_sha256", "ingested_at",
    ]
    matches_df = matches_df[match_cols]

    seasons_df = build_seasons_dataframe(season_metadata)

    # Write outputs
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matches_path = processed_matches_path()
    matches_df.to_parquet(matches_path, index=False)
    logger.info("wrote %d matches to %s", len(matches_df), matches_path)

    teams_path = processed_teams_path()
    teams_df.to_parquet(teams_path, index=False)
    logger.info("wrote %d teams to %s", len(teams_df), teams_path)

    seasons_path = processed_seasons_path()
    seasons_df.to_parquet(seasons_path, index=False)
    logger.info("wrote %d seasons to %s", len(seasons_df), seasons_path)

    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest OpenFootball match results.")
    parser.add_argument("--offline", action="store_true", help="use cached raw files only")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--season-limit", type=int, default=None)
    parser.add_argument("--season", action="append", dest="seasons", help="specific season dir(s) to process (e.g., 2015-16)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        summary = run(
            offline=args.offline,
            timeout=args.timeout,
            season_limit=args.season_limit,
            seasons=args.seasons,
        )
    except IngestError as exc:
        logger.error("ingestion failed: %s", exc)
        return 1

    logger.info("ingestion complete: %d seasons, %d matches", summary["seasons_processed"], summary["matches_total"])
    if summary["failures"]:
        logger.warning("failures: %s", summary["failures"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())