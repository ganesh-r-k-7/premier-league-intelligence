"""StatsBomb Open Data ingestion for Premier League Intelligence (Phase 1).

This module retrieves permitted StatsBomb Open Data, stores immutable raw
snapshots under ``data/raw/`` and writes a validated, provenance-carrying
fixtures table under ``data/processed/``. Processed data is written only after
validation succeeds.

Data source: StatsBomb Open Data (https://github.com/statsbomb/open-data).
Attribution to StatsBomb is required by the source terms.

Design notes
------------
- Raw and processed data are separated: raw JSON is written once and never
  modified; processed parquet is derived from validated raw payloads.
- Every processed row carries provenance columns (source, source_url, season,
  ingested_at).
- Network access is wrapped with timeouts and explicit error handling. When
  downloads fail but a cached raw snapshot exists, the cached copy is used and
  the failure is reported; when nothing is available an ``IngestError`` is
  raised.
- Transformation and validation helpers are pure functions so the test suite
  can exercise them offline with synthetic fixtures.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

import pandas as pd
import requests

from src.data import config, validate

logger = logging.getLogger("plintel.ingest")

SOURCE_NAME = config.STATSBOMB_SOURCE_NAME
SOURCE_URL = config.STATSBOMB_SOURCE_URL
DEFAULT_TIMEOUT = 30
USER_AGENT = "premier-league-intelligence/0.1 (+https://github.com/statsbomb/open-data)"


class IngestError(RuntimeError):
    """Raised when ingestion cannot complete safely."""


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def http_get(url, timeout=DEFAULT_TIMEOUT):
    """Perform a single HTTP GET, raising ``IngestError`` on any failure."""
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        raise IngestError("network request to %s failed: %s" % (url, exc)) from exc
    if response.status_code != 200:
        raise IngestError("unexpected HTTP %d for %s" % (response.status_code, url))
    return response


def download_json(url, dest, timeout=DEFAULT_TIMEOUT):
    """Download ``url`` as JSON and atomically write the raw bytes to ``dest``.

    Returns ``(payload, dest_path)``. Writes raw bytes only after the response
    has been received and parsed, so a partial download never replaces a good
    snapshot.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = http_get(url, timeout=timeout)
    try:
        payload = response.json()
    except ValueError as exc:
        raise IngestError("response from %s was not valid JSON" % url) from exc
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(response.content)
    tmp.replace(dest)
    return payload, dest


def competitions_raw_path():
    return config.STATSBOMB_RAW_DIR / "competitions.json"


def matches_raw_path(competition_id, season_id):
    return config.STATSBOMB_RAW_DIR / "matches" / ("%d_%d.json" % (competition_id, season_id))


def fixtures_processed_path():
    return config.STATSBOMB_PROCESSED_DIR / "fixtures.parquet"


def target_seasons(competitions_payload):
    """Return the seasons of the target competition that have matches available.

    Only seasons with ``match_available`` set are returned, so ingestion never
    asserts the presence of data that does not exist.
    """
    seasons = []
    for row in competitions_payload:
        if not isinstance(row, dict):
            continue
        if row.get("competition_id") != config.TARGET_COMPETITION_ID:
            continue
        if row.get("match_available") in (None, "", False):
            continue
        season_id = row.get("season_id")
        if not isinstance(season_id, int):
            continue
        seasons.append({"season_id": season_id, "season_name": row.get("season_name")})
    seasons.sort(key=lambda item: item["season_id"])
    return seasons


def matches_to_records(matches_payload, season_id, season_name, ingested_at):
    """Flatten a StatsBomb matches payload into provenance-carrying records.

    Missing values are preserved as ``None``; nothing is imputed.
    """
    records = []
    for match in matches_payload:
        if not isinstance(match, dict):
            continue
        competition = match.get("competition") or {}
        season = match.get("season") or {}
        home = match.get("home_team") or {}
        away = match.get("away_team") or {}
        stadium = match.get("stadium")
        records.append(
            {
                "match_id": match.get("match_id"),
                "match_date": match.get("match_date"),
                "kick_off": match.get("kick_off"),
                "match_week": match.get("match_week"),
                "competition_id": competition.get("competition_id"),
                "competition_name": competition.get("competition_name"),
                "season_id": season.get("season_id", season_id),
                "season_name": season.get("season_name", season_name),
                "home_team_id": home.get("home_team_id"),
                "home_team_name": home.get("home_team_name"),
                "away_team_id": away.get("away_team_id"),
                "away_team_name": away.get("away_team_name"),
                "home_score": match.get("home_score"),
                "away_score": match.get("away_score"),
                "match_status": match.get("match_status"),
                "stadium": stadium.get("name") if isinstance(stadium, dict) else stadium,
                "source": SOURCE_NAME,
                "source_url": SOURCE_URL,
                "ingested_at": ingested_at,
            }
        )
    return records


def build_fixtures_dataframe(season_payloads, ingested_at=None):
    """Build the processed fixtures DataFrame from validated season payloads.

    ``season_payloads`` is a list of dicts with ``season_id``, ``season_name``
    and ``payload`` (the raw list of matches).
    """
    ingested_at = ingested_at or _utcnow()
    records = []
    for entry in season_payloads:
        records.extend(
            matches_to_records(
                entry["payload"],
                entry["season_id"],
                entry["season_name"],
                ingested_at,
            )
        )
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    return frame


def validate_competitions(payload):
    issues = validate.check_competitions(payload)
    if issues:
        raise IngestError("competitions validation failed: %s" % "; ".join(issues))
    return payload


def validate_matches(payload, label):
    issues = validate.check_matches(payload)
    if issues:
        raise IngestError("matches validation failed for %s: %s" % (label, "; ".join(issues)))
    return payload


def write_processed_fixtures(frame, dest=None):
    dest = Path(dest) if dest is not None else fixtures_processed_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    return dest


def _load_json_source(url, path, timeout, offline):
    """Load JSON from a cached raw file or, when online, download it first.

    Falls back to the cached snapshot if a download fails but one exists.
    """
    path = Path(path)
    if offline:
        if not path.exists():
            raise IngestError("offline mode and no cached raw file at %s" % path)
        logger.info("offline: using cached raw file %s", path)
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        payload, _ = download_json(url, path, timeout=timeout)
        logger.info("downloaded %s -> %s", url, path)
        return payload
    except IngestError as exc:
        if path.exists():
            logger.warning("download failed (%s); using cached snapshot %s", exc, path)
            return json.loads(path.read_text(encoding="utf-8"))
        raise


def run(offline=False, timeout=DEFAULT_TIMEOUT, season_limit=None):
    """Run the full Phase 1 ingestion.

    Downloads (or loads cached) competitions and match lists for the target
    competition, validates them, and writes the processed fixtures table. No
    processed data is written unless every used payload validates.
    """
    competitions_payload = _load_json_source(
        config.STATSBOMB_COMPETITIONS_URL,
        competitions_raw_path(),
        timeout,
        offline,
    )
    validate_competitions(competitions_payload)

    seasons = target_seasons(competitions_payload)
    if season_limit is not None:
        seasons = seasons[:season_limit]
    if not seasons:
        raise IngestError("no target-competition seasons with matches available")

    season_payloads = []
    for season in seasons:
        season_id = season["season_id"]
        label = "%s (%s)" % (season["season_name"], season_id)
        path = matches_raw_path(config.TARGET_COMPETITION_ID, season_id)
        url = config.statsbomb_matches_url(config.TARGET_COMPETITION_ID, season_id)
        payload = _load_json_source(url, path, timeout, offline)
        validate_matches(payload, label)
        logger.info("validated matches for %s: %d rows", label, len(payload))
        season_payloads.append(
            {"season_id": season_id, "season_name": season["season_name"], "payload": payload}
        )

    if not season_payloads:
        raise IngestError("no validated match data available; nothing written")

    frame = build_fixtures_dataframe(season_payloads)
    dest = write_processed_fixtures(frame)
    logger.info("wrote %d processed fixtures to %s", len(frame), dest)
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ingest StatsBomb Open Data fixtures.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use cached raw snapshots only; never access the network",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--season-limit", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        dest = run(offline=args.offline, timeout=args.timeout, season_limit=args.season_limit)
    except IngestError as exc:
        logger.error("ingestion failed: %s", exc)
        return 1
    logger.info("ingestion complete: %s", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
