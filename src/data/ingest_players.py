"""Phase 2 player-data ingestion (StatsBomb events and lineups).

Downloads the complete Premier League 2015/16 season (competition 2, season 27)
event and lineup files, preserving immutable raw JSON snapshots under
``data/raw/statsbomb/events/`` and ``data/raw/statsbomb/lineups/``. Raw files are
validated before use and valid cached files are reused, so an interrupted run
resumes cheaply.

Design:
- Network access uses timeouts, retries with exponential backoff, explicit HTTP
  status handling and clear error messages.
- Offline mode never touches the network and uses only valid cached files.
- A 404 is treated as a missing match (recorded, not fatal); other failures are
  collected and reported.
- Building the player datasets is delegated to
  ``src/features/player_aggregation.py``; this module only ingests and orchestrates.

Source: StatsBomb Open Data (attribution required).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.data import config, validate_players
from src.features import player_aggregation

logger = logging.getLogger("plintel.ingest.players")

USER_AGENT = "premier-league-intelligence/0.2 (+https://github.com/statsbomb/open-data)"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.0
DEFAULT_PAUSE = 0.0
RETRY_STATUS = {429, 500, 502, 503, 504}


class IngestError(RuntimeError):
    """Raised when ingestion cannot complete for a resource."""


class NotFound(IngestError):
    """Raised for HTTP 404 (resource genuinely absent at the source)."""


def raw_event_path(match_id):
    return config.STATSBOMB_EVENTS_RAW_DIR / ("%d.json" % match_id)


def raw_lineup_path(match_id):
    return config.STATSBOMB_LINEUPS_RAW_DIR / ("%d.json" % match_id)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_valid_json_file(path):
    """A cached file is valid only if it parses and is a non-empty list."""
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return False
    return isinstance(payload, list) and len(payload) > 0


def fetch_json_to_file(
    url,
    dest,
    timeout=DEFAULT_TIMEOUT,
    retries=DEFAULT_RETRIES,
    backoff=DEFAULT_BACKOFF,
    get=None,
):
    """Download ``url`` as a non-empty JSON list into ``dest`` with retries.

    Writes atomically (temp file then rename) so a failed download never leaves
    a corrupt cache entry.
    """
    get = get or requests.get
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error = "unknown error"
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(backoff * (2 ** (attempt - 1)))
        try:
            response = get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            last_error = "request error: %s" % exc
            logger.warning("attempt %d/%d failed for %s (%s)", attempt + 1, retries + 1, url, exc)
            continue

        status = response.status_code
        if status == 404:
            raise NotFound("HTTP 404 not found: %s" % url)
        if status in RETRY_STATUS:
            last_error = "HTTP %d" % status
            logger.warning("attempt %d/%d got HTTP %d for %s", attempt + 1, retries + 1, status, url)
            continue
        if status != 200:
            raise IngestError("unexpected HTTP %d for %s" % (status, url))

        try:
            payload = response.json()
        except ValueError as exc:
            raise IngestError("response from %s was not valid JSON: %s" % (url, exc))
        if not isinstance(payload, list) or not payload:
            raise IngestError("response from %s was not a non-empty JSON list" % url)

        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(response.content)
        tmp.replace(dest)
        return payload
    raise IngestError("failed after %d attempts: %s (%s)" % (retries + 1, url, last_error))


def load_match_records(season_id=None):
    """Load match metadata for the target season from the Phase 1 fixtures."""
    season_id = config.TARGET_SEASON_ID if season_id is None else season_id
    fixtures = pd.read_parquet(config.STATSBOMB_PROCESSED_DIR / "fixtures.parquet")
    subset = fixtures[fixtures["season_id"] == season_id]
    columns = [
        "match_id",
        "match_date",
        "season_id",
        "season_name",
        "competition_id",
        "home_team_name",
        "away_team_name",
        "home_score",
        "away_score",
    ]
    return subset[columns].to_dict(orient="records")


def ingest_season(
    offline=False,
    force=False,
    timeout=DEFAULT_TIMEOUT,
    retries=DEFAULT_RETRIES,
    backoff=DEFAULT_BACKOFF,
    pause=DEFAULT_PAUSE,
    get=None,
):
    """Download events and lineups for every match of the target season.

    Returns a summary dict. Valid cached files are skipped unless ``force`` is
    set. Offline mode records uncached resources as failures instead of
    downloading them.
    """
    records = load_match_records()
    summary = {
        "matches": len(records),
        "downloaded": {"events": 0, "lineups": 0},
        "skipped": {"events": 0, "lineups": 0},
        "missing": {"events": [], "lineups": []},
        "failed": [],
    }
    resources = [
        ("events", raw_event_path, config.statsbomb_events_url),
        ("lineups", raw_lineup_path, config.statsbomb_lineups_url),
    ]
    for record in records:
        match_id = record["match_id"]
        for kind, path_fn, url_fn in resources:
            path = path_fn(match_id)
            if not force and is_valid_json_file(path):
                summary["skipped"][kind] += 1
                continue
            if offline:
                summary["failed"].append(
                    {"match_id": match_id, "kind": kind, "error": "offline and no valid cached file"}
                )
                continue
            try:
                fetch_json_to_file(
                    url_fn(match_id),
                    path,
                    timeout=timeout,
                    retries=retries,
                    backoff=backoff,
                    get=get,
                )
                summary["downloaded"][kind] += 1
                logger.info("downloaded %s for match %d", kind, match_id)
            except NotFound:
                summary["missing"][kind].append(match_id)
                logger.warning("%s for match %d is missing at the source (404)", kind, match_id)
            except IngestError as exc:
                summary["failed"].append({"match_id": match_id, "kind": kind, "error": str(exc)})
                logger.error("%s for match %d failed: %s", kind, match_id, exc)
            if pause:
                time.sleep(pause)
    return summary


def build_player_datasets(ingested_at=None):
    """Build player_match and player_season parquet from cached raw payloads."""
    records = load_match_records()
    columns = [
        "match_id",
        "match_date",
        "season_id",
        "season_name",
        "competition_id",
    ]
    rows = []
    warnings = []
    built = []
    skipped = []
    for record in records:
        match_id = record["match_id"]
        event_path = raw_event_path(match_id)
        lineup_path = raw_lineup_path(match_id)
        if not (is_valid_json_file(event_path) and is_valid_json_file(lineup_path)):
            skipped.append(match_id)
            continue
        events = read_json(event_path)
        lineups = read_json(lineup_path)
        match = {column: record[column] for column in columns}
        match_rows, match_warnings = player_aggregation.aggregate_match_players(
            events, lineups, match, ingested_at=ingested_at
        )
        rows.extend(match_rows)
        warnings.extend({"match_id": match_id, "warning": warning} for warning in match_warnings)
        built.append(match_id)

    player_match = player_aggregation.build_player_match_frame(rows)
    player_season = player_aggregation.build_player_season_frame(player_match)

    config.STATSBOMB_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    player_match.to_parquet(config.PLAYER_MATCH_PARQUET, index=False)
    player_season.to_parquet(config.PLAYER_SEASON_PARQUET, index=False)

    return {
        "matches": len(records),
        "built": built,
        "skipped": skipped,
        "warnings": warnings,
        "player_match_rows": len(player_match),
        "player_season_rows": len(player_season),
    }


def read_events(match_id):
    return read_json(raw_event_path(match_id))


def audit_and_report(report_path=None):
    """Audit the built player datasets against raw source events."""
    player_match = pd.read_parquet(config.PLAYER_MATCH_PARQUET)
    player_season = pd.read_parquet(config.PLAYER_SEASON_PARQUET)
    records = load_match_records()
    report = validate_players.audit_season(
        player_match, player_season, records, read_events, config.TARGET_SEASON_ID
    )
    report_path = Path(report_path) if report_path else config.REPORTS_DIR / "phase2_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(validate_players.render_audit_markdown(report), encoding="utf-8")
    return report, report_path


def _print_summary(summary):
    print("matches:", summary["matches"])
    print("downloaded:", summary["downloaded"])
    print("skipped:", summary["skipped"])
    print("missing events:", len(summary["missing"]["events"]))
    print("missing lineups:", len(summary["missing"]["lineups"]))
    print("failures:", len(summary["failed"]))
    for failure in summary["failed"][:10]:
        print("  failure:", failure)


def run(offline=False, force=False, skip_build=False, **kwargs):
    ingest_summary = ingest_season(offline=offline, force=force, **kwargs)
    _print_summary(ingest_summary)
    if not skip_build:
        build_summary = build_player_datasets()
        print(
            "built player_match=%d rows, player_season=%d rows from %d matches; skipped %d"
            % (
                build_summary["player_match_rows"],
                build_summary["player_season_rows"],
                len(build_summary["built"]),
                len(build_summary["skipped"]),
            )
        )
        report, report_path = audit_and_report()
        print(
            "audit: %d matches, %d score mismatches, fixture goals %d vs event goals %d; report %s"
            % (
                report["matches_checked"],
                len(report["score_mismatches"]),
                report["fixture_goals"],
                report["event_goals"],
                report_path,
            )
        )
    return ingest_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ingest StatsBomb events and lineups for Phase 2.")
    parser.add_argument("--offline", action="store_true", help="use cached raw files only")
    parser.add_argument("--force", action="store_true", help="re-download even valid cached files")
    parser.add_argument("--skip-build", action="store_true", help="download only; do not build datasets")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(
        offline=args.offline,
        force=args.force,
        skip_build=args.skip_build,
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
