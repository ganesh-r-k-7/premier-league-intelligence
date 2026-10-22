from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"

STATSBOMB_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master"
STATSBOMB_COMPETITIONS_URL = STATSBOMB_BASE_URL + "/data/competitions.json"
STATSBOMB_MATCHES_URL = STATSBOMB_BASE_URL + "/data/matches"
STATSBOMB_EVENTS_URL = STATSBOMB_BASE_URL + "/data/events"
STATSBOMB_LINEUPS_URL = STATSBOMB_BASE_URL + "/data/lineups"
STATSBOMB_RAW_DIR = DATA_RAW_DIR / "statsbomb"
STATSBOMB_EVENTS_RAW_DIR = STATSBOMB_RAW_DIR / "events"
STATSBOMB_LINEUPS_RAW_DIR = STATSBOMB_RAW_DIR / "lineups"
STATSBOMB_PROCESSED_DIR = DATA_PROCESSED_DIR / "statsbomb"
PLAYER_MATCH_PARQUET = STATSBOMB_PROCESSED_DIR / "player_match.parquet"
PLAYER_SEASON_PARQUET = STATSBOMB_PROCESSED_DIR / "player_season.parquet"
TARGET_COMPETITION_ID = 2
TARGET_SEASON_ID = 27
TARGET_SEASON_NAME = "2015/2016"

# StatsBomb Open Data requires attribution. It is the only source currently
# ingested; other candidate sources remain unverified (see PROJECT_PLAN.md).
STATSBOMB_SOURCE_NAME = "StatsBomb Open Data"
STATSBOMB_SOURCE_URL = "https://github.com/statsbomb/open-data"


def statsbomb_matches_url(competition_id, season_id):
    return "%s/%d/%d.json" % (STATSBOMB_MATCHES_URL, competition_id, season_id)


def statsbomb_events_url(match_id):
    return "%s/%d.json" % (STATSBOMB_EVENTS_URL, match_id)


def statsbomb_lineups_url(match_id):
    return "%s/%d.json" % (STATSBOMB_LINEUPS_URL, match_id)
