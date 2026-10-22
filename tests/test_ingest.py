import json

import pandas as pd
import pytest
import requests

from src.data import config, ingest


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_target_seasons_filters_to_available_target(competitions_fixture):
    seasons = ingest.target_seasons(competitions_fixture)
    assert seasons == [{"season_id": 99, "season_name": "2018/2019"}]


def test_matches_to_records_adds_provenance(matches_fixture):
    records = ingest.matches_to_records(
        matches_fixture, 99, "2018/2019", "2020-01-01T00:00:00Z"
    )
    assert len(records) == 2
    first = records[0]
    assert first["match_id"] == 1001
    assert first["home_team_name"] == "Alpha FC"
    assert first["away_team_name"] == "Beta FC"
    assert first["season_id"] == 99
    assert first["source"] == config.STATSBOMB_SOURCE_NAME
    assert first["source_url"] == config.STATSBOMB_SOURCE_URL
    assert first["ingested_at"] == "2020-01-01T00:00:00Z"


def test_matches_to_records_preserves_missing_values():
    payload = [{"match_id": 5, "home_team": {}, "away_team": {}, "home_score": None, "away_score": None}]
    records = ingest.matches_to_records(payload, 1, "x", "t")
    assert records[0]["home_team_name"] is None
    assert records[0]["home_score"] is None


def test_build_fixtures_dataframe_shape(matches_fixture):
    frame = ingest.build_fixtures_dataframe(
        [{"season_id": 99, "season_name": "2018/2019", "payload": matches_fixture}],
        ingested_at="2020-01-01T00:00:00Z",
    )
    assert len(frame) == 2
    expected = {"match_id", "home_team_name", "away_team_name", "season_id", "source", "ingested_at"}
    assert expected <= set(frame.columns)
    assert frame["match_date"].notna().all()


def test_download_json_writes_raw_file(monkeypatch, tmp_path):
    class FakeResponse:
        status_code = 200
        content = b'{"ok": true}'

        def json(self):
            return {"ok": True}

    monkeypatch.setattr(ingest, "http_get", lambda *args, **kwargs: FakeResponse())
    dest = tmp_path / "raw" / "competitions.json"
    payload, path = ingest.download_json("http://example.test/competitions.json", dest)
    assert payload == {"ok": True}
    assert path == dest
    assert dest.exists()
    assert json.loads(dest.read_text(encoding="utf-8")) == {"ok": True}
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_download_json_invalid_json_raises(monkeypatch, tmp_path):
    class FakeResponse:
        status_code = 200
        content = b"<html>not json</html>"

        def json(self):
            raise ValueError("no json")

    monkeypatch.setattr(ingest, "http_get", lambda *args, **kwargs: FakeResponse())
    with pytest.raises(ingest.IngestError):
        ingest.download_json("http://example.test/x.json", tmp_path / "x.json")


def test_http_get_network_failure_raises(monkeypatch):
    def fake_get(url, timeout=30, headers=None):
        raise requests.ConnectionError("no internet")

    monkeypatch.setattr(ingest.requests, "get", fake_get)
    with pytest.raises(ingest.IngestError):
        ingest.http_get("http://example.test/x.json")


def test_http_get_non_200_raises(monkeypatch):
    class FakeResponse:
        status_code = 404
        content = b"Not Found"

    monkeypatch.setattr(ingest.requests, "get", lambda *args, **kwargs: FakeResponse())
    with pytest.raises(ingest.IngestError):
        ingest.http_get("http://example.test/missing.json")


def test_run_offline_writes_processed(monkeypatch, tmp_path, competitions_fixture, matches_fixture):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    monkeypatch.setattr(config, "STATSBOMB_RAW_DIR", raw)
    monkeypatch.setattr(config, "STATSBOMB_PROCESSED_DIR", processed)
    _write_json(raw / "competitions.json", competitions_fixture)
    _write_json(raw / "matches" / "2_99.json", matches_fixture)

    dest = ingest.run(offline=True)
    assert dest == processed / "fixtures.parquet"
    assert dest.exists()
    frame = pd.read_parquet(dest)
    assert len(frame) == 2
    assert set(frame["season_id"]) == {99}
    assert set(frame["source"]) == {config.STATSBOMB_SOURCE_NAME}


def test_run_offline_without_cache_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "STATSBOMB_RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "STATSBOMB_PROCESSED_DIR", tmp_path / "processed")
    with pytest.raises(ingest.IngestError):
        ingest.run(offline=True)


def test_run_offline_invalid_competitions_does_not_write(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    monkeypatch.setattr(config, "STATSBOMB_RAW_DIR", raw)
    monkeypatch.setattr(config, "STATSBOMB_PROCESSED_DIR", tmp_path / "processed")
    _write_json(raw / "competitions.json", {"not": "a list"})
    with pytest.raises(ingest.IngestError):
        ingest.run(offline=True)
    assert not (tmp_path / "processed" / "statsbomb" / "fixtures.parquet").exists()


def test_run_offline_invalid_matches_does_not_write(
    monkeypatch, tmp_path, competitions_fixture, matches_fixture
):
    raw = tmp_path / "raw"
    monkeypatch.setattr(config, "STATSBOMB_RAW_DIR", raw)
    monkeypatch.setattr(config, "STATSBOMB_PROCESSED_DIR", tmp_path / "processed")
    _write_json(raw / "competitions.json", competitions_fixture)
    _write_json(raw / "matches" / "2_99.json", {"not": "a list"})

    with pytest.raises(ingest.IngestError):
        ingest.run(offline=True)
    assert not (tmp_path / "processed" / "statsbomb" / "fixtures.parquet").exists()


def test_run_offline_missing_season_cache_raises(monkeypatch, tmp_path, competitions_fixture):
    raw = tmp_path / "raw"
    monkeypatch.setattr(config, "STATSBOMB_RAW_DIR", raw)
    monkeypatch.setattr(config, "STATSBOMB_PROCESSED_DIR", tmp_path / "processed")
    _write_json(raw / "competitions.json", competitions_fixture)
    with pytest.raises(ingest.IngestError):
        ingest.run(offline=True)


def test_download_failure_falls_back_to_cache(monkeypatch, tmp_path, competitions_fixture):
    raw = tmp_path / "raw"
    monkeypatch.setattr(config, "STATSBOMB_RAW_DIR", raw)
    monkeypatch.setattr(config, "STATSBOMB_PROCESSED_DIR", tmp_path / "processed")
    _write_json(raw / "competitions.json", competitions_fixture)

    def failing_download(url, dest, timeout=30):
        raise ingest.IngestError("simulated network failure")

    monkeypatch.setattr(ingest, "download_json", failing_download)
    payload = ingest._load_json_source(
        "http://example.test/competitions.json", raw / "competitions.json", 30, offline=False
    )
    assert payload == competitions_fixture
