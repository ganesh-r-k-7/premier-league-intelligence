import pandas as pd

from src.data import validate_players as vp


def _shot(team, outcome, key_pass_id=None, shot_type="Open Play"):
    shot = {"outcome": {"name": outcome}, "type": {"name": shot_type}}
    if key_pass_id:
        shot["key_pass_id"] = key_pass_id
    return {"id": "s-%s-%s" % (team, outcome), "type": {"name": "Shot"}, "team": {"name": team}, "shot": shot}


def test_team_goals_counts_shots_and_own_goals():
    events = [
        _shot("Blue FC", "Goal"),
        _shot("Red FC", "Goal"),
        _shot("Blue FC", "Saved"),
        {"id": "ogf", "type": {"name": "Own Goal For"}, "team": {"name": "Blue FC"}, "player": None},
        {"id": "oga", "type": {"name": "Own Goal Against"}, "team": {"name": "Red FC"}, "player": {"id": 1}},
    ]
    assert vp.team_goals_from_events(events) == {"Blue FC": 2, "Red FC": 1}


def test_assist_linkage_consistent():
    events = [
        {"id": "P1", "type": {"name": "Pass"}, "team": {"name": "Blue FC"}, "pass": {"goal_assist": True}},
        {
            "id": "S1",
            "type": {"name": "Shot"},
            "team": {"name": "Blue FC"},
            "shot": {"outcome": {"name": "Goal"}, "key_pass_id": "P1", "type": {"name": "Open Play"}},
        },
    ]
    report = vp.assist_linkage(events)
    assert report["goal_assist_passes"] == 1
    assert report["goals_with_key_pass"] == 1
    assert report["goal_assist_ids_not_referenced_by_goal"] == 0
    assert report["goal_key_passes_not_flagged_goal_assist"] == 0


def test_assist_linkage_detects_mismatch():
    events = [
        {
            "id": "S1",
            "type": {"name": "Shot"},
            "team": {"name": "Blue FC"},
            "shot": {"outcome": {"name": "Goal"}, "key_pass_id": "P9", "type": {"name": "Open Play"}},
        }
    ]
    report = vp.assist_linkage(events)
    assert report["goal_assist_passes"] == 0
    assert report["goals_with_key_pass"] == 1
    assert report["goal_key_passes_not_flagged_goal_assist"] == 1


def test_duplicate_event_ids():
    assert vp.duplicate_event_ids([{"id": "a"}, {"id": "a"}, {"id": "b"}]) == ["a"]


def test_reconcile_match_ok_and_mismatch():
    events = [_shot("Blue FC", "Goal"), _shot("Blue FC", "Goal")]
    record = {"match_id": 1, "home_team_name": "Blue FC", "away_team_name": "Red FC", "home_score": 2, "away_score": 0}
    ok = vp.reconcile_match(events, record)
    assert ok["home_ok"] and ok["away_ok"]

    mismatch = vp.reconcile_match(events, {**record, "home_score": 1})
    assert not mismatch["home_ok"]


def test_check_season_totals_zero_deltas():
    player_match = pd.DataFrame(
        {
            "minutes": [90, 30],
            "goals": [2, 0],
            "own_goals": [0, 0],
            "assists": [1, 0],
            "shots": [2, 0],
            "shots_on_target": [2, 0],
            "xg": [1.0, 0.0],
            "npxg": [1.0, 0.0],
            "xa": [0.5, 0.0],
            "penalty_goals": [0, 0],
            "penalty_attempts": [0, 0],
            "started": [True, False],
        }
    )
    player_season = pd.DataFrame(
        {
            "minutes": [120],
            "goals": [2],
            "own_goals": [0],
            "assists": [1],
            "shots": [2],
            "shots_on_target": [2],
            "xg": [1.0],
            "npxg": [1.0],
            "xa": [0.5],
            "penalty_goals": [0],
            "penalty_attempts": [0],
            "appearances": [2],
            "starts": [1],
        }
    )
    deltas = vp.check_season_totals(player_match, player_season)
    assert all(value == 0 for value in deltas.values())


def test_render_audit_markdown():
    report = {
        "matches_checked": 2,
        "score_mismatches": [],
        "fixture_goals": 5,
        "event_goals": 5,
        "duplicate_event_ids": 0,
        "assist_linkage": {"goals": 5, "goal_assist_passes": 3},
        "season_total_deltas": {"goals": 0},
    }
    text = vp.render_audit_markdown(report)
    assert "Phase 2 aggregation audit" in text
    assert "Matches checked: 2" in text
