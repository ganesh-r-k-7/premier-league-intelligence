from src.features import player_aggregation as pa

MATCH = {
    "match_id": 1000,
    "match_date": "2015-08-08",
    "season_id": 27,
    "season_name": "2015/2016",
    "competition_id": 2,
}


def _by_player(rows):
    return {row["player_id"]: row for row in rows}


def test_dedupe_events_drops_duplicate_ids(events_synthetic):
    assert len(events_synthetic) == 17
    deduped = pa.dedupe_events(events_synthetic)
    assert len(deduped) == 16


def test_match_end_minute(events_synthetic):
    assert pa.match_end_minute(pa.dedupe_events(events_synthetic)) == 90


def test_starting_players(events_synthetic):
    starters = pa.starting_players(events_synthetic)
    assert len(starters) == 22
    assert 9 in starters and 25 in starters and 33 not in starters


def test_goals_penalties_and_xg(events_synthetic, lineups_synthetic):
    rows, _ = pa.aggregate_match_players(events_synthetic, lineups_synthetic, MATCH)
    p = _by_player(rows)
    assert p[9]["goals"] == 2
    assert p[9]["penalty_goals"] == 1
    assert p[9]["penalty_attempts"] == 1
    assert p[9]["shots"] == 2
    assert round(p[9]["xg"], 2) == 1.16
    assert round(p[9]["npxg"], 2) == 0.40


def test_assists_and_xa(events_synthetic, lineups_synthetic):
    rows, _ = pa.aggregate_match_players(events_synthetic, lineups_synthetic, MATCH)
    p = _by_player(rows)
    assert p[7]["assists"] == 1
    assert round(p[7]["xa"], 2) == 0.40


def test_own_goal_not_credited_as_goal(events_synthetic, lineups_synthetic):
    rows, _ = pa.aggregate_match_players(events_synthetic, lineups_synthetic, MATCH)
    p = _by_player(rows)
    assert p[25]["own_goals"] == 1
    assert p[25]["goals"] == 0
    assert p[25]["minutes"] == 90


def test_minutes_from_substitutions(events_synthetic, lineups_synthetic):
    rows, _ = pa.aggregate_match_players(events_synthetic, lineups_synthetic, MATCH)
    p = _by_player(rows)
    assert p[26]["minutes"] == 60
    assert p[26]["started"] is True
    assert p[32]["minutes"] == 30
    assert p[32]["started"] is False
    assert p[8]["minutes"] == 70
    assert p[12]["minutes"] == 20


def test_minutes_shortened_by_red_card(events_synthetic, lineups_synthetic):
    rows, _ = pa.aggregate_match_players(events_synthetic, lineups_synthetic, MATCH)
    p = _by_player(rows)
    assert p[27]["minutes"] == 80


def test_unused_substitute_has_no_row(events_synthetic, lineups_synthetic):
    rows, _ = pa.aggregate_match_players(events_synthetic, lineups_synthetic, MATCH)
    assert 33 not in _by_player(rows)


def test_shots_on_target(events_synthetic, lineups_synthetic):
    rows, _ = pa.aggregate_match_players(events_synthetic, lineups_synthetic, MATCH)
    p = _by_player(rows)
    assert p[10]["shots_on_target"] == 1
    assert p[99]["shots_on_target"] == 0


def test_unknown_player_reported_not_dropped(events_synthetic, lineups_synthetic):
    rows, warnings = pa.aggregate_match_players(events_synthetic, lineups_synthetic, MATCH)
    p = _by_player(rows)
    assert 99 in p
    assert p[99]["minutes"] == 0
    assert p[99]["shots"] == 1
    assert any("99" in warning for warning in warnings)


def test_events_without_player_are_ignored(events_synthetic, lineups_synthetic):
    rows, _ = pa.aggregate_match_players(events_synthetic, lineups_synthetic, MATCH)
    assert all(row["player_id"] is not None for row in rows)


def test_provenance_columns(events_synthetic, lineups_synthetic):
    rows, _ = pa.aggregate_match_players(
        events_synthetic, lineups_synthetic, MATCH, ingested_at="2020-01-01T00:00:00Z"
    )
    for row in rows:
        assert row["source"] == "StatsBomb Open Data"
        assert row["source_url"] == "https://github.com/statsbomb/open-data"
        assert row["ingested_at"] == "2020-01-01T00:00:00Z"
        assert row["match_id"] == 1000
        assert row["season_id"] == 27


def test_player_season_aggregation(events_synthetic, lineups_synthetic):
    first, _ = pa.aggregate_match_players(events_synthetic, lineups_synthetic, {**MATCH, "match_id": 1})
    second, _ = pa.aggregate_match_players(events_synthetic, lineups_synthetic, {**MATCH, "match_id": 2})
    frame = pa.build_player_match_frame(first + second)
    season = pa.build_player_season_frame(frame)
    striker = season[season["player_id"] == 9].iloc[0]
    assert striker["appearances"] == 2
    assert striker["starts"] == 2
    assert striker["goals"] == 4
    assert striker["minutes"] == 180
    assert round(striker["xg"], 2) == 2.32


def test_empty_frames_have_expected_columns():
    empty = pa.build_player_match_frame([])
    assert list(empty.columns) == pa.PLAYER_MATCH_COLUMNS
    season = pa.build_player_season_frame(empty)
    assert season.empty
    assert list(season.columns) == pa.PLAYER_SEASON_COLUMNS


def test_zero_minute_substitute_at_match_end(events_synthetic, lineups_synthetic):
    """A substitute who enters at match end gets 0 minutes but still appears."""
    events = list(events_synthetic)
    events.append({
        "id": "sub_late",
        "index": 18,
        "period": 2,
        "timestamp": "00:45:00.000",
        "minute": 95,
        "type": {"id": 19, "name": "Substitution"},
        "team": {"id": 200, "name": "Red FC"},
        "player": {"id": 29, "name": "Red Forward"},
        "substitution": {"replacement": {"id": 33, "name": "Red Unused"}},
    })
    events.append({
        "id": "halfend_stoppage",
        "index": 19,
        "period": 2,
        "timestamp": "00:46:00.000",
        "minute": 95,
        "type": {"id": 33, "name": "Half End"},
        "team": {"id": 100, "name": "Blue FC"},
    })
    lineups = list(lineups_synthetic)
    rows, _ = pa.aggregate_match_players(events, lineups, MATCH)
    p = _by_player(rows)
    assert 33 in p, "late substitute should appear even with 0 minutes"
    assert p[33]["minutes"] == 0
    assert p[33]["started"] is False


def test_stoppage_time_match_end_minute(events_synthetic):
    """Match end uses the actual final minute including stoppage time."""
    events = list(events_synthetic)
    events.append({
        "id": "halfend_stoppage",
        "index": 18,
        "period": 2,
        "timestamp": "00:46:00.000",
        "minute": 95,
        "type": {"id": 33, "name": "Half End"},
        "team": {"id": 100, "name": "Blue FC"},
    })
    assert pa.match_end_minute(pa.dedupe_events(events)) == 95


def test_second_yellow_shortens_minutes(events_synthetic, lineups_synthetic):
    """A Second Yellow card shortens a player's minutes like a red card."""
    events = list(events_synthetic)
    events.append({
        "id": "second_yellow",
        "index": 18,
        "period": 2,
        "timestamp": "00:35:00.000",
        "minute": 65,
        "type": {"id": 22, "name": "Bad Behaviour"},
        "team": {"id": 200, "name": "Red FC"},
        "player": {"id": 21, "name": "Red Keeper"},
        "bad_behaviour": {"card": {"id": 6, "name": "Second Yellow"}},
    })
    rows, _ = pa.aggregate_match_players(events, lineups_synthetic, MATCH)
    p = _by_player(rows)
    assert p[21]["minutes"] == 65


def test_penalty_miss_not_a_goal(events_synthetic, lineups_synthetic):
    """A penalty shot that is saved is a penalty attempt but not a goal."""
    events = list(events_synthetic)
    events.append({
        "id": "S_penalty_miss",
        "index": 18,
        "period": 2,
        "timestamp": "00:05:00.000",
        "minute": 55,
        "type": {"id": 16, "name": "Shot"},
        "team": {"id": 100, "name": "Blue FC"},
        "player": {"id": 9, "name": "Blue Striker"},
        "shot": {"statsbomb_xg": 0.76, "key_pass_id": None, "type": {"id": 88, "name": "Penalty"}, "outcome": {"id": 98, "name": "Saved"}},
    })
    rows, _ = pa.aggregate_match_players(events, lineups_synthetic, MATCH)
    p = _by_player(rows)
    assert p[9]["penalty_attempts"] == 2
    assert p[9]["penalty_goals"] == 1
    assert p[9]["goals"] == 2
    assert round(p[9]["npxg"], 2) == 0.40
