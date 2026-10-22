import json

import pandas as pd
import pytest
import requests

from src.data import config, ingest_players


def _response(status=200, payload=None, content=None, json_error=False):
    class FakeResponse:
        status_code = status

        @property
        def content(self):
            if content is not None:
                return content
            return json.dumps(payload).encode("utf-8")

        def json(self):
            if json_error:
                raise ValueError("not json")
            return payload

    return FakeResponse()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- fetch_json_to_file ---------------------------------------------------


def test_fetch_json_success(tmp_path):
    dest = tmp_path / "events" / "1.json"
    payload = [{"id": "a"}]
    result = ingest_players.fetch_json_to_file(
        "http://x/1.json", dest, get=lambda *a, **k: _response(200, payload)
    )
    assert result == payload
    assert json.loads(dest.read_text()) == payload
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_fetch_json_retries_then_succeeds(tmp_path):
    calls = {"n": 0}

    def get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return _response(503, None)
        return _response(200, [{"id": "a"}])

    result = ingest_players.fetch_json_to_file(
        "http://x/1.json", tmp_path / "1.json", retries=3, backoff=0, get=get
    )
    assert result == [{"id": "a"}]
    assert calls["n"] == 3


def test_fetch_json_request_exception_retries(tmp_path):
    calls = {"n": 0}

    def get(*args, **kwargs):
        calls["n"] += 1
        raise requests.ConnectionError("no internet")

    with pytest.raises(ingest_players.IngestError):
        ingest_players.fetch_json_to_file(
            "http://x/1.json", tmp_path / "1.json", retries=2, backoff=0, get=get
        )
    assert calls["n"] == 3


def test_fetch_json_persistent_failure_raises(tmp_path):
    get = lambda *a, **k: _response(500, None)
    with pytest.raises(ingest_players.IngestError):
        ingest_players.fetch_json_to_file("http://x/1.json", tmp_path / "1.json", retries=1, backoff=0, get=get)


def test_fetch_json_404_is_not_found(tmp_path):
    get = lambda *a, **k: _response(404, None)
    with pytest.raises(ingest_players.NotFound):
        ingest_players.fetch_json_to_file("http://x/1.json", tmp_path / "1.json", retries=1, backoff=0, get=get)


def test_fetch_json_invalid_json_raises(tmp_path):
    get = lambda *a, **k: _response(200, None, content=b"<html>", json_error=True)
    with pytest.raises(ingest_players.IngestError):
        ingest_players.fetch_json_to_file("http://x/1.json", tmp_path / "1.json", get=get)


def test_fetch_json_empty_list_raises(tmp_path):
    get = lambda *a, **k: _response(200, [])
    with pytest.raises(ingest_players.IngestError):
        ingest_players.fetch_json_to_file("http://x/1.json", tmp_path / "1.json", get=get)


# --- is_valid_json_file ----------------------------------------------------


def test_is_valid_json_file(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps([{"id": 1}]))
    assert ingest_players.is_valid_json_file(valid) is True

    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    assert ingest_players.is_valid_json_file(empty) is False

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert ingest_players.is_valid_json_file(broken) is False

    assert ingest_players.is_valid_json_file(tmp_path / "missing.json") is False


# --- ingest_season ---------------------------------------------------------


@pytest.fixture
def patched_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "STATSBOMB_EVENTS_RAW_DIR", tmp_path / "events")
    monkeypatch.setattr(config, "STATSBOMB_LINEUPS_RAW_DIR", tmp_path / "lineups")
    monkeypatch.setattr(config, "STATSBOMB_PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "PLAYER_MATCH_PARQUET", tmp_path / "processed" / "player_match.parquet")
    monkeypatch.setattr(config, "PLAYER_SEASON_PARQUET", tmp_path / "processed" / "player_season.parquet")
    return tmp_path


def test_ingest_season_skips_valid_cache(patched_dirs, monkeypatch):
    records = [{"match_id": 1}, {"match_id": 2}]
    monkeypatch.setattr(ingest_players, "load_match_records", lambda *a, **k: records)
    for match_id in (1, 2):
        _write_json(ingest_players.raw_event_path(match_id), [{"id": match_id}])
        _write_json(ingest_players.raw_lineup_path(match_id), [{"team_id": match_id}])

    summary = ingest_players.ingest_season(offline=True)
    assert summary["skipped"] == {"events": 2, "lineups": 2}
    assert summary["downloaded"] == {"events": 0, "lineups": 0}
    assert summary["failed"] == []


def test_ingest_season_offline_without_cache_fails(patched_dirs, monkeypatch):
    monkeypatch.setattr(ingest_players, "load_match_records", lambda *a, **k: [{"match_id": 1}])
    summary = ingest_players.ingest_season(offline=True)
    assert len(summary["failed"]) == 2
    assert summary["downloaded"] == {"events": 0, "lineups": 0}


def test_ingest_season_redownloads_invalid_cache(patched_dirs, monkeypatch):
    monkeypatch.setattr(ingest_players, "load_match_records", lambda *a, **k: [{"match_id": 1}])
    _write_json(ingest_players.raw_event_path(1), [])  # invalid: empty
    _write_json(ingest_players.raw_lineup_path(1), [{"team_id": 1}])  # valid

    summary = ingest_players.ingest_season(
        offline=False, get=lambda *a, **k: _response(200, [{"id": "fresh"}])
    )
    assert summary["downloaded"]["events"] == 1
    assert summary["skipped"]["lineups"] == 1
    assert json.loads(ingest_players.raw_event_path(1).read_text()) == [{"id": "fresh"}]


def test_ingest_season_missing_source_recorded(patched_dirs, monkeypatch):
    monkeypatch.setattr(ingest_players, "load_match_records", lambda *a, **k: [{"match_id": 7}])
    summary = ingest_players.ingest_season(offline=False, get=lambda *a, **k: _response(404, None))
    assert summary["missing"]["events"] == [7]
    assert summary["missing"]["lineups"] == [7]
    assert summary["failed"] == []


def test_ingest_season_resume_partial(patched_dirs, monkeypatch):
    monkeypatch.setattr(ingest_players, "load_match_records", lambda *a, **k: [{"match_id": 1}, {"match_id": 2}])
    _write_json(ingest_players.raw_event_path(1), [{"id": 1}])
    _write_json(ingest_players.raw_lineup_path(1), [{"team_id": 1}])

    summary = ingest_players.ingest_season(
        offline=False, get=lambda *a, **k: _response(200, [{"id": "new"}])
    )
    assert summary["skipped"] == {"events": 1, "lineups": 1}
    assert summary["downloaded"] == {"events": 1, "lineups": 1}


# --- build_player_datasets -------------------------------------------------


def test_build_player_datasets(patched_dirs, monkeypatch, events_synthetic, lineups_synthetic):
    record = {
        "match_id": 1000,
        "match_date": "2015-08-08",
        "season_id": 27,
        "season_name": "2015/2016",
        "competition_id": 2,
        "home_team_name": "Blue FC",
        "away_team_name": "Red FC",
        "home_score": 3,
        "away_score": 0,
    }
    monkeypatch.setattr(ingest_players, "load_match_records", lambda *a, **k: [record])
    _write_json(ingest_players.raw_event_path(1000), events_synthetic)
    _write_json(ingest_players.raw_lineup_path(1000), lineups_synthetic)

    summary = ingest_players.build_player_datasets(ingested_at="2020-01-01T00:00:00Z")
    assert summary["player_match_rows"] == 25
    assert summary["player_season_rows"] == 25
    assert config.PLAYER_MATCH_PARQUET.exists()
    assert config.PLAYER_SEASON_PARQUET.exists()

    player_match = pd.read_parquet(config.PLAYER_MATCH_PARQUET)
    striker = player_match[player_match["player_id"] == 9].iloc[0]
    assert striker["goals"] == 2
    assert striker["assists"] == 0


def test_build_player_datasets_skips_missing_raw(patched_dirs, monkeypatch):
    monkeypatch.setattr(
        ingest_players,
        "load_match_records",
        lambda *a, **k: [{"match_id": 999, "season_id": 27, "season_name": "2015/2016", "competition_id": 2, "match_date": "x"}],
    )
    summary = ingest_players.build_player_datasets()
    assert summary["built"] == []
    assert summary["skipped"] == [999]
    assert summary["player_match_rows"] == 0
