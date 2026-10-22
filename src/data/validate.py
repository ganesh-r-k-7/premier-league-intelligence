def check_competitions(payload):
    issues = []
    if not isinstance(payload, list):
        return ["competitions is not a list"]
    seen = set()
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            issues.append("competition row %d is not an object" % index)
            continue
        for key in (
            "competition_id",
            "season_id",
            "competition_name",
            "season_name",
            "match_available",
        ):
            if key not in row:
                issues.append("competition row %d has no %s" % (index, key))
        competition_id = row.get("competition_id")
        season_id = row.get("season_id")
        if isinstance(competition_id, int) and isinstance(season_id, int):
            key = (competition_id, season_id)
            if key in seen:
                issues.append("duplicate competition %d/%d" % (competition_id, season_id))
            seen.add(key)
    return issues


def check_matches(payload):
    issues = []
    if not isinstance(payload, list):
        return ["matches is not a list"]
    seen = set()
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            issues.append("match row %d is not an object" % index)
            continue
        for key in (
            "match_id",
            "match_date",
            "competition",
            "season",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "match_status",
        ):
            if key not in row:
                issues.append("match row %d has no %s" % (index, key))
        match_id = row.get("match_id")
        if isinstance(match_id, int):
            if match_id in seen:
                issues.append("duplicate match %d" % match_id)
            seen.add(match_id)
        competition = row.get("competition")
        if not isinstance(competition, dict):
            issues.append("match row %d has an invalid competition" % index)
        elif not isinstance(competition.get("competition_id"), int):
            issues.append("match row %d has an invalid competition" % index)
        season = row.get("season")
        if not isinstance(season, dict):
            issues.append("match row %d has an invalid season" % index)
        elif not isinstance(season.get("season_id"), int):
            issues.append("match row %d has an invalid season" % index)
        home = row.get("home_team")
        if not isinstance(home, dict):
            issues.append("match row %d has an invalid home team" % index)
        elif not isinstance(home.get("home_team_name"), str):
            issues.append("match row %d has an invalid home team" % index)
        away = row.get("away_team")
        if not isinstance(away, dict):
            issues.append("match row %d has an invalid away team" % index)
        elif not isinstance(away.get("away_team_name"), str):
            issues.append("match row %d has an invalid away team" % index)
        for key in ("home_score", "away_score"):
            value = row.get(key)
            if value is not None and not isinstance(value, int):
                issues.append("match row %d has an invalid %s" % (index, key))
        if row.get("match_status") != "available":
            issues.append("match row %d has an unexpected match_status" % index)
    return issues
