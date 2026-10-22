from src.data import validate


def test_check_competitions_valid(competitions_fixture):
    assert validate.check_competitions(competitions_fixture) == []


def test_check_competitions_flags_missing_keys():
    payload = [{"competition_id": 2, "season_id": 99}]
    issues = validate.check_competitions(payload)
    assert any("has no competition_name" in issue for issue in issues)


def test_check_competitions_flags_duplicates():
    row = {
        "competition_id": 2,
        "season_id": 99,
        "competition_name": "PL",
        "season_name": "x",
        "match_available": True,
    }
    issues = validate.check_competitions([row, dict(row)])
    assert any("duplicate competition 2/99" in issue for issue in issues)


def test_check_competitions_non_list():
    assert validate.check_competitions({"not": "a list"}) == ["competitions is not a list"]


def test_check_matches_valid(matches_fixture):
    assert validate.check_matches(matches_fixture) == []


def test_check_matches_flags_duplicate_id(matches_fixture):
    payload = [matches_fixture[0], dict(matches_fixture[0])]
    assert any("duplicate match 1001" in issue for issue in validate.check_matches(payload))


def test_check_matches_flags_bad_status(matches_fixture):
    payload = [dict(matches_fixture[0], match_status="postponed")]
    assert any("match_status" in issue for issue in validate.check_matches(payload))


def test_check_matches_flags_bad_team(matches_fixture):
    payload = [dict(matches_fixture[0], home_team="not-a-dict")]
    assert any("invalid home team" in issue for issue in validate.check_matches(payload))


def test_check_matches_non_list():
    assert validate.check_matches("nope") == ["matches is not a list"]
