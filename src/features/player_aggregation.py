"""Event-to-player aggregation for Phase 2 (goals and assists forecasting).

This module turns StatsBomb Open Data events and lineups into two tabular
datasets:

- ``player_match``: one row per player per match they appeared in.
- ``player_season``: one row per player per season (aggregated from player_match).

The functions are pure and offline-testable: they accept already-parsed payloads
and never touch the network. Ingestion of the raw events/lineups is handled in
``src/data/ingest.py``.

Derivation rules (grounded in StatsBomb Open Data spec v4.0.0 and verified
against match 3754217):

- Goals: Shot events whose ``shot.outcome.name == "Goal"``. Penalty goals are
  included in ``goals`` and also counted in ``penalty_goals``. Own goals are
  NOT shots and are never added to a player's goals; they are tracked in
  ``own_goals`` using the ``Own Goal Against`` event's player.
- Assists: Pass events with ``pass.goal_assist == True``.
- xG: sum of ``shot.statsbomb_xg`` over a player's shots (penalties included).
  ``npxg`` excludes penalty shots.
- xA: for every shot with a ``shot.key_pass_id``, the referenced pass's player
  is credited with that shot's ``statsbomb_xg``.
- Minutes/appearances/starts: starters come from the ``Starting XI`` event's
  ``tactics.lineup``; substitutions come from ``Substitution`` events (the
  event's ``player`` goes off, ``substitution.replacement`` comes on); red
  cards end a player's match early. A player appears if they have minutes > 0.
- Position: the player's most frequent event ``position.name`` in the match.

Missing information is never inferred silently: unknown players and malformed
rows are reported as warnings and preserved where possible.
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections import Counter, defaultdict

import pandas as pd

from src.data import config

logger = logging.getLogger("plintel.features.player_aggregation")

SOURCE_NAME = config.STATSBOMB_SOURCE_NAME
SOURCE_URL = config.STATSBOMB_SOURCE_URL

SHOT_ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved to Post"}
RED_CARD_NAMES = {"Red Card", "Second Yellow"}
DEFAULT_MATCH_END_MINUTE = 90

PLAYER_MATCH_COLUMNS = [
    "match_id",
    "match_date",
    "season_id",
    "season_name",
    "competition_id",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "position",
    "started",
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
    "source",
    "source_url",
    "ingested_at",
]

PLAYER_SEASON_COLUMNS = [
    "player_id",
    "player_name",
    "season_id",
    "season_name",
    "competition_id",
    "team_id",
    "team_name",
    "position",
    "appearances",
    "starts",
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
    "source",
    "source_url",
    "ingested_at",
]

_SUM_COLUMNS = [
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


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def dedupe_events(events):
    """Drop duplicate events by event id, keeping the first occurrence."""
    seen = set()
    unique = []
    for event in events:
        event_id = event.get("id") if isinstance(event, dict) else None
        if event_id is not None:
            if event_id in seen:
                logger.warning("duplicate event id ignored: %s", event_id)
                continue
            seen.add(event_id)
        unique.append(event)
    return unique


def _event_type(event):
    return (event.get("type") or {}).get("name")


def _player(event):
    return event.get("player") or {}


def _team(event):
    return event.get("team") or {}


def _collect_card(event):
    for key in ("foul_committed", "bad_behaviour"):
        card = (event.get(key) or {}).get("card") or {}
        if card.get("name"):
            return card.get("name")
    return None


def _is_red_card(event):
    return _collect_card(event) in RED_CARD_NAMES


def match_end_minute(events):
    """The match's final minute, from the latest Half End event."""
    minutes = [
        event.get("minute")
        for event in events
        if _event_type(event) == "Half End" and isinstance(event.get("minute"), int)
    ]
    return max(minutes) if minutes else DEFAULT_MATCH_END_MINUTE


def starting_players(events):
    """Player ids listed in any Starting XI event."""
    starters = set()
    for event in events:
        if _event_type(event) != "Starting XI":
            continue
        lineup = (event.get("tactics") or {}).get("lineup") or []
        for entry in lineup:
            player_id = (entry.get("player") or {}).get("id")
            if isinstance(player_id, int):
                starters.add(player_id)
    return starters


def compute_minutes(events, end_minute):
    """Return ``(minutes_by_player, starters)`` for a match.

    Minutes are computed from Starting XI and Substitution events, shortened by
    red cards. Player Off/On events (temporary absences such as injury
    treatment) are intentionally ignored; permanent substitutions are the
    authoritative source.
    """
    starters = starting_players(events)
    sub_off = {}
    sub_on = {}
    send_off = {}
    for event in events:
        if _event_type(event) == "Substitution":
            off = _player(event)
            on = (event.get("substitution") or {}).get("replacement") or {}
            minute = event.get("minute", 0)
            if isinstance(off.get("id"), int):
                sub_off[off["id"]] = minute
            if isinstance(on.get("id"), int):
                sub_on[on["id"]] = minute
        elif _is_red_card(event):
            player_id = _player(event).get("id")
            if isinstance(player_id, int):
                send_off[player_id] = min(send_off.get(player_id, end_minute), event.get("minute", end_minute))

    minutes = {}
    for player_id in set(starters) | set(sub_on):
        start = 0 if player_id in starters else sub_on.get(player_id, 0)
        end = end_minute
        if player_id in sub_off:
            end = min(end, sub_off[player_id])
        if player_id in send_off:
            end = min(end, send_off[player_id])
        minutes[player_id] = max(0, end - start)
    return minutes, starters


def lineup_lookup(lineups):
    """Map player id -> {player_name, team_id, team_name} from lineup payloads."""
    lookup = {}
    for team in lineups:
        team_id = team.get("team_id")
        team_name = team.get("team_name")
        for player in team.get("lineup", []):
            player_id = player.get("player_id")
            if isinstance(player_id, int):
                lookup[player_id] = {
                    "player_name": player.get("player_name"),
                    "team_id": team_id,
                    "team_name": team_name,
                }
    return lookup


def _blank_stats():
    return {
        "goals": 0,
        "own_goals": 0,
        "assists": 0,
        "shots": 0,
        "shots_on_target": 0,
        "xg": 0.0,
        "npxg": 0.0,
        "xa": 0.0,
        "penalty_goals": 0,
        "penalty_attempts": 0,
    }


def _position_counts(events):
    counts = defaultdict(Counter)
    for event in events:
        player_id = _player(event).get("id")
        position = (event.get("position") or {}).get("name")
        if isinstance(player_id, int) and position:
            counts[player_id][position] += 1
    return counts


def _attacking_stats(events):
    stats = defaultdict(_blank_stats)
    event_names = defaultdict(dict)

    shot_xg_by_pass = {}
    for event in events:
        if _event_type(event) != "Shot":
            continue
        shot = event.get("shot") or {}
        key_pass_id = shot.get("key_pass_id")
        if key_pass_id:
            shot_xg_by_pass[key_pass_id] = shot.get("statsbomb_xg") or 0.0

    for event in events:
        event_type = _event_type(event)
        player_id = _player(event).get("id")
        if isinstance(player_id, int) and _player(event).get("name"):
            event_names[player_id] = _player(event).get("name")
        if not isinstance(player_id, int):
            continue

        if event_type == "Shot":
            shot = event.get("shot") or {}
            outcome = (shot.get("outcome") or {}).get("name")
            is_penalty = (shot.get("type") or {}).get("name") == "Penalty"
            xg = shot.get("statsbomb_xg") or 0.0
            row = stats[player_id]
            row["shots"] += 1
            row["xg"] += xg
            if is_penalty:
                row["penalty_attempts"] += 1
            else:
                row["npxg"] += xg
            if outcome == "Goal":
                row["goals"] += 1
                if is_penalty:
                    row["penalty_goals"] += 1
            if outcome in SHOT_ON_TARGET_OUTCOMES:
                row["shots_on_target"] += 1
        elif event_type == "Pass":
            pass_obj = event.get("pass") or {}
            if pass_obj.get("goal_assist"):
                stats[player_id]["assists"] += 1
            event_id = event.get("id")
            if event_id in shot_xg_by_pass:
                stats[player_id]["xa"] += shot_xg_by_pass[event_id]
        elif event_type == "Own Goal Against":
            stats[player_id]["own_goals"] += 1
    return stats, event_names


def aggregate_match_players(events, lineups, match, ingested_at=None):
    """Aggregate one match into player-match rows.

    Returns ``(rows, warnings)``. ``match`` must provide match_id, match_date,
    season_id, season_name and competition_id.
    """
    ingested_at = ingested_at or _utcnow()
    events = dedupe_events(events)
    lookup = lineup_lookup(lineups)
    end_minute = match_end_minute(events)
    minutes, starters = compute_minutes(events, end_minute)
    stats, event_names = _attacking_stats(events)
    positions = _position_counts(events)

    warnings = []
    player_ids = set(minutes) | set(stats)
    for player_id in sorted(player_ids):
        if player_id not in lookup:
            name = event_names.get(player_id) or stats.get(player_id, {}).get("_name")
            warnings.append("player %s not present in lineups; using event data" % player_id)

    rows = []
    for player_id in sorted(player_ids):
        info = lookup.get(player_id, {})
        row = {
            "match_id": match.get("match_id"),
            "match_date": match.get("match_date"),
            "season_id": match.get("season_id"),
            "season_name": match.get("season_name"),
            "competition_id": match.get("competition_id"),
            "player_id": player_id,
            "player_name": info.get("player_name") or event_names.get(player_id),
            "team_id": info.get("team_id"),
            "team_name": info.get("team_name"),
            "position": (positions[player_id].most_common(1)[0][0] if positions[player_id] else None),
            "started": player_id in starters,
            "minutes": minutes.get(player_id, 0),
            "source": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "ingested_at": ingested_at,
        }
        row.update(stats.get(player_id, _blank_stats()))
        rows.append({column: row.get(column) for column in PLAYER_MATCH_COLUMNS})
    return rows, warnings


def build_player_match_frame(rows):
    if not rows:
        return pd.DataFrame(columns=PLAYER_MATCH_COLUMNS)
    return pd.DataFrame(rows, columns=PLAYER_MATCH_COLUMNS)


def build_player_season_frame(player_match_df):
    """Aggregate player-match rows into one row per player per season."""
    if player_match_df is None or player_match_df.empty:
        return pd.DataFrame(columns=PLAYER_SEASON_COLUMNS)

    groups = player_match_df.groupby(["player_id", "season_id", "competition_id"], dropna=False)
    records = []
    for (player_id, season_id, competition_id), group in groups:
        by_minutes = group.sort_values("minutes", ascending=False)
        record = {
            "player_id": player_id,
            "player_name": group["player_name"].dropna().iloc[0] if group["player_name"].notna().any() else None,
            "season_id": season_id,
            "season_name": group["season_name"].dropna().iloc[0] if group["season_name"].notna().any() else None,
            "competition_id": competition_id,
            "team_id": by_minutes["team_id"].iloc[0],
            "team_name": by_minutes["team_name"].iloc[0],
            "position": by_minutes["position"].iloc[0],
            "appearances": int((group["minutes"] > 0).sum()),
            "starts": int(group["started"].sum()),
            "source": group["source"].iloc[0],
            "source_url": group["source_url"].iloc[0],
            "ingested_at": group["ingested_at"].max(),
        }
        for column in _SUM_COLUMNS:
            record[column] = group[column].sum()
        records.append({column: record.get(column) for column in PLAYER_SEASON_COLUMNS})
    return pd.DataFrame(records, columns=PLAYER_SEASON_COLUMNS)
