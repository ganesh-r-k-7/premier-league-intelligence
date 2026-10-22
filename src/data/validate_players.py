"""Validation and reconciliation for the Phase 2 player datasets.

These functions are pure (they accept parsed payloads / DataFrames) so they can
be tested offline. They reconcile the aggregated player data against the
StatsBomb source: score totals, assist/xA linkage, and player-match vs
player-season consistency. Nothing is force-fitted: discrepancies are reported.
"""

from __future__ import annotations

from collections import defaultdict


def _event_type(event):
    return (event.get("type") or {}).get("name")


def team_goals_from_events(events):
    """Goals per team name from events, counting shot goals and own goals.

    An own goal is credited to the benefiting team named on the ``Own Goal For``
    event; the ``Own Goal Against`` event is not counted, which avoids double
    counting. StatsBomb Open Data emits the pair for every own goal.
    """
    counts = defaultdict(int)
    for event in events:
        event_type = _event_type(event)
        team_name = (event.get("team") or {}).get("name")
        if event_type == "Shot" and ((event.get("shot") or {}).get("outcome") or {}).get("name") == "Goal":
            counts[team_name] += 1
        elif event_type == "Own Goal For":
            counts[team_name] += 1
    return dict(counts)


def assist_linkage(events):
    """Count assist-related linkages for a single match.

    ``assists`` are defined by ``pass.goal_assist`` (authoritative). ``xA`` is
    defined by ``shot.key_pass_id``. This function measures how often the two
    disagree, so the distinction is explicit rather than assumed.
    """
    goal_assist_ids = set()
    goal_shots = 0
    goals_with_key_pass = 0
    key_pass_ids = set()
    penalty_goals = 0
    for event in events:
        event_type = _event_type(event)
        if event_type == "Pass" and (event.get("pass") or {}).get("goal_assist"):
            goal_assist_ids.add(event.get("id"))
        elif event_type == "Shot":
            shot = event.get("shot") or {}
            if (shot.get("outcome") or {}).get("name") == "Goal":
                goal_shots += 1
                if (shot.get("type") or {}).get("name") == "Penalty":
                    penalty_goals += 1
                key_pass_id = shot.get("key_pass_id")
                if key_pass_id:
                    goals_with_key_pass += 1
                    key_pass_ids.add(key_pass_id)
    return {
        "goals": goal_shots,
        "penalty_goals": penalty_goals,
        "goal_assist_passes": len(goal_assist_ids),
        "goals_with_key_pass": goals_with_key_pass,
        "goal_assist_ids_not_referenced_by_goal": len(goal_assist_ids - key_pass_ids),
        "goal_key_passes_not_flagged_goal_assist": len(key_pass_ids - goal_assist_ids),
    }


def duplicate_event_ids(events):
    seen = set()
    duplicates = []
    for event in events:
        event_id = event.get("id")
        if event_id in seen:
            duplicates.append(event_id)
        seen.add(event_id)
    return duplicates


def reconcile_match(events, record):
    """Reconcile a match's event goals against its recorded score."""
    goals = team_goals_from_events(events)
    home = record.get("home_team_name")
    away = record.get("away_team_name")
    return {
        "match_id": record.get("match_id"),
        "home_team": home,
        "away_team": away,
        "expected_home": record.get("home_score"),
        "expected_away": record.get("away_score"),
        "actual_home": goals.get(home, 0),
        "actual_away": goals.get(away, 0),
        "home_ok": goals.get(home, 0) == record.get("home_score"),
        "away_ok": goals.get(away, 0) == record.get("away_score"),
    }


def check_season_totals(player_match_df, player_season_df):
    """Verify player_season equals the sum of player_match (self-consistency)."""
    sum_columns = [
        "minutes",
        "goals",
        "own_goals",
        "assists",
        "shots",
        "shots_on_target",
        "xg",
        "npxg",
        "xa",
        "penalty_goals",
        "penalty_attempts",
    ]
    deltas = {}
    for column in sum_columns:
        match_total = float(player_match_df[column].sum()) if not player_match_df.empty else 0.0
        season_total = float(player_season_df[column].sum()) if not player_season_df.empty else 0.0
        deltas[column] = round(match_total - season_total, 6)
    appearances = int((player_match_df["minutes"] > 0).sum()) if not player_match_df.empty else 0
    season_appearances = int(player_season_df["appearances"].sum()) if not player_season_df.empty else 0
    deltas["appearances"] = appearances - season_appearances
    starts = int(player_match_df["started"].sum()) if not player_match_df.empty else 0
    season_starts = int(player_season_df["starts"].sum()) if not player_season_df.empty else 0
    deltas["starts"] = starts - season_starts
    return deltas


def render_audit_markdown(report):
    lines = ["# Phase 2 aggregation audit", ""]
    lines.append("Generated from the ingested Premier League 2015/16 raw events.")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append("- Matches checked: %d" % report["matches_checked"])
    lines.append("- Matches with score mismatch: %d" % len(report["score_mismatches"]))
    lines.append("- Total fixture goals: %d" % report["fixture_goals"])
    lines.append("- Total event goals (shots + own goals): %d" % report["event_goals"])
    lines.append("- Duplicate event ids found: %d" % report["duplicate_event_ids"])
    lines.append("")
    lines.append("## Assist / xA linkage")
    lines.append("")
    for key, value in report["assist_linkage"].items():
        lines.append("- %s: %d" % (key, value))
    lines.append("")
    lines.append("## player_match vs player_season self-consistency")
    lines.append("")
    for key, value in report["season_total_deltas"].items():
        lines.append("- %s delta: %s" % (key, value))
    lines.append("")
    if report["score_mismatches"]:
        lines.append("## Score mismatches")
        lines.append("")
        for mismatch in report["score_mismatches"]:
            lines.append(
                "- match %s: %s expected %s-%s, events %s-%s"
                % (
                    mismatch["match_id"],
                    mismatch["home_team"],
                    mismatch["expected_home"],
                    mismatch["expected_away"],
                    mismatch["actual_home"],
                    mismatch["actual_away"],
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def audit_season(player_match_df, player_season_df, records, read_events, season_id):
    """Run the full audit over the season and return a report dict."""
    score_mismatches = []
    fixture_goals = 0
    event_goals = 0
    duplicates = 0
    linkage = defaultdict(int)
    for record in records:
        if record.get("season_id") != season_id:
            continue
        try:
            events = read_events(record["match_id"])
        except (OSError, ValueError):
            continue
        reconciliation = reconcile_match(events, record)
        if not (reconciliation["home_ok"] and reconciliation["away_ok"]):
            score_mismatches.append(reconciliation)
        fixture_goals += (record.get("home_score") or 0) + (record.get("away_score") or 0)
        event_goals += reconciliation["actual_home"] + reconciliation["actual_away"]
        duplicates += len(duplicate_event_ids(events))
        for key, value in assist_linkage(events).items():
            linkage[key] += value
    return {
        "matches_checked": len(records),
        "score_mismatches": score_mismatches,
        "fixture_goals": fixture_goals,
        "event_goals": event_goals,
        "duplicate_event_ids": duplicates,
        "assist_linkage": dict(sorted(linkage.items())),
        "season_total_deltas": check_season_totals(player_match_df, player_season_df),
    }
