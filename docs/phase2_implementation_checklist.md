# Phase 2 implementation checklist — goals and assists

Source of truth: `PROJECT_PLAN.md` ("Phase 2 data strategy") and the master
specification. Phase 2 uses **StatsBomb Open Data** only (attribution required).
Understat and FBref are explicitly out of scope.

This document is the Stage 1 design deliverable. Stage 2 has been completed:
all 380 Premier League 2015/16 matches' events and lineups have been downloaded,
validated, and aggregated into `player_match.parquet` and `player_season.parquet`.
See `reports/phase2_stage2_report.md` for the full Stage 2 report.

## 1. Source files needed

Premier League 2015/2016, competition id `2`, season id `27` (380 matches,
confirmed from `data/raw/statsbomb/matches/2_27.json`).

| File | URL pattern | Purpose |
| --- | --- | --- |
| Matches | `data/matches/2/27.json` | already ingested (Phase 1) |
| Lineups | `data/lineups/{match_id}.json` | appearances, squad, position |
| Events | `data/events/{match_id}.json` | goals, assists, xG, xA, minutes, starts |

The incomplete 2003/2004 season (season id `44`, only 38 of 380 matches) is
**not** ingested in Phase 2. It is insufficient for evaluation and mixes eras.

## 2. Storage estimate

Measured on 2026-10-17 from the live source (HTTP `Content-Length` samples):

| Item | Per file | Count | Total |
| --- | --- | --- | --- |
| Events | ~283 KB mean (252–307 KB, n=12) | 380 | **~1.1 GB** |
| Lineups | ~1.85 KB mean (n=15) | 380 | **~7.2 MB** |
| Processed `player_match.parquet` | — | 10,469 rows | **~258 KB** |
| Processed `player_season.parquet` | — | 550 rows | **~48 KB** |

Total new download for Phase 2: **~1.1 GB**. This corrects the "~108 MB"
estimate in the original design — per-file event sizes are much larger than
the 12-match sample suggested. This corrects the "2–4 MB each, gigabytes total"
figure still present in `src/eda.py` (to be fixed when Phase 2 code touches
that file).

## 3. Schemas

Both tables carry provenance: `source`, `source_url`, `ingested_at`.

### `player_match` (one row per player per match appeared)

`match_id, match_date, season_id, season_name, competition_id, player_id,
player_name, team_id, team_name, position, started, minutes, goals, own_goals,
assists, shots, shots_on_target, xg, npxg, xa, penalty_goals,
penalty_attempts, source, source_url, ingested_at`

Actual: 10,469 rows (one per player per match with minutes >= 0, including
0-minute substitutes who entered at match end).

### `player_season` (one row per player per season)

`player_id, player_name, season_id, season_name, competition_id, team_id,
team_name, position, appearances, starts, minutes, goals, own_goals, assists,
shots, shots_on_target, xg, npxg, xa, penalty_goals, penalty_attempts, source,
source_url, ingested_at`

Identifiers are StatsBomb ids (`player_id`, `team_id`, `match_id`). Team and
position for a season are taken from the match with the most minutes. No
cross-source player mapping is needed in Phase 2.

## 4. Derivation rules

Verified against StatsBomb Open Data Events v4.0.0 and lineups v2.0.0, and
checked against real match `3754217`.

| Field | Rule |
| --- | --- |
| `goals` | `Shot` events with `shot.outcome.name == "Goal"` (penalties included) |
| `penalty_goals` | goal shots with `shot.type.name == "Penalty"` |
| `penalty_attempts` | all shots with `shot.type.name == "Penalty"` |
| `own_goals` | `Own Goal Against` event's `player` (never added to `goals`) |
| `assists` | `Pass` events with `pass.goal_assist == True` |
| `shots` | count of `Shot` events |
| `shots_on_target` | shot outcome in `{Goal, Saved, Saved to Post}` |
| `xg` | sum of `shot.statsbomb_xg` (penalties included) |
| `npxg` | sum of `shot.statsbomb_xg` excluding `Penalty` shots |
| `xa` | for each shot with `shot.key_pass_id`, credit the referenced pass's player with that shot's `statsbomb_xg` |
| `started` | player id present in a `Starting XI` event's `tactics.lineup` |
| `appearances` | player has `minutes > 0` |
| `minutes` | starter: `sub_off`/red-card/match-end minus 0; substitute: match end minus sub-on minute; `minute` is absolute across periods |
| `position` | most frequent event `position.name` for the player in the match |

Minutes sources: `Starting XI` (starters), `Substitution` (`player` off,
`substitution.replacement` on), red cards (`foul_committed.card` /
`bad_behaviour.card` in `{Red Card, Second Yellow}`), and the last `Half End`
minute as match end. Temporary `Player Off`/`Player On` events (e.g. injury
treatment) are deliberately ignored. Match end uses the actual final minute
(e.g. 90 + stoppage), which slightly inflates per-90 denominators; documented,
not hidden.

Missing information is never inferred: unknown players and malformed rows are
returned as warnings and kept where possible.

## 5. Edge cases and test coverage

Implemented in `src/features/player_aggregation.py`; tested in
`tests/test_player_aggregation.py` against synthetic fixtures
(`tests/fixtures/events_synthetic.json`, `lineups_synthetic.json`).

| Edge case | Test |
| --- | --- |
| Duplicate event ids | `test_dedupe_events_drops_duplicate_ids` |
| Open-play + penalty goals | `test_goals_penalties_and_xg` |
| Own goal not credited as a goal | `test_own_goal_not_credited_as_goal` |
| Assist and xA linkage | `test_assists_and_xa` |
| Substitutions and minutes | `test_minutes_from_substitutions` |
| Red card shortens minutes | `test_minutes_shortened_by_red_card` |
| Unused substitute absent | `test_unused_substitute_has_no_row` |
| Shots on target vs off target | `test_shots_on_target` |
| Unknown player warned, not dropped | `test_unknown_player_reported_not_dropped` |
| Events without a player ignored | `test_events_without_player_are_ignored` |
| Provenance columns | `test_provenance_columns` |
| Player-season aggregation | `test_player_season_aggregation` |
| Empty inputs | `test_empty_frames_have_expected_columns` |

## 6. Pipeline design

```
data/raw/statsbomb/matches/2_27.json ─┐
data/raw/statsbomb/lineups/{id}.json ─┼─► aggregate_match_players ─► player_match ─► player_season
data/raw/statsbomb/events/{id}.json ──┘        (pure functions)        parquet        parquet
```

- Ingestion (`src/data/ingest.py`, Stage 2) downloads and caches raw JSON only,
  reusing the Phase 1 download/provenance helpers.
- Aggregation (`src/features/player_aggregation.py`) is pure and offline; it is
  tested before any download.
- Aggregates are validated against source totals (Stage 2, step 9).

## 7. Evaluation design (Stage 3)

- Rolling-origin, chronological, **within** 2015/2016. At each cut-off
  matchweek, features use only matches before the cut-off; targets are
  goals/assists over a later horizon. No random splits.
- Separate targets: goals and assists.
- Baselines: global mean; per-position mean; last-observed; minutes-
  proportional expectation (team goals × player minute share).
- Metrics: MAE, RMSE, Poisson deviance, Spearman rank / top-K precision for
  leaderboards, reported against every baseline.
- A single complete season cannot support cross-season validation; this
  limitation is stated in the results and documentation.

## 8. Stage checklist

### Stage 1 — prepare
- [x] Implementation checklist and design documented (this file)
- [x] Aggregation rules documented and grounded in the real schema
- [x] `player_match` / `player_season` schemas with provenance
- [x] Synthetic offline fixtures and edge-case tests
- [x] Storage estimate and source-file list
- [x] Run tests — 44 passed (29 Phase 1 + 15 Phase 2)

### Stage 2 — implement and ingest (completed)
- [x] Extend `src/data/ingest.py` to fetch lineups + events for season 27 (`src/data/ingest_players.py`)
- [x] Offline tests for the new ingestion (monkeypatched network) — `tests/test_ingest_players.py`
- [x] Download ~109 MB, raw snapshots preserved with attribution — 380 events + 380 lineups (1.1 GB total)
- [x] Build `player_match` and `player_season` parquet — 10,469 rows / 550 rows
- [x] Validate aggregates against source totals; investigate discrepancies — `reports/phase2_audit.md` (0 mismatches, 0 duplicates, all deltas 0)

### Stage 3 — forecast and evaluate (not started)
- [ ] Goals and assists models with baselines
- [ ] Rolling-origin evaluation within 2015/16
- [ ] MAE/RMSE/count metrics vs baselines, saved reproducibly
- [ ] Limitations documented
## 9. Known limitations

- One complete PL season only; no cross-season validation is possible.
- 2003/2004 excluded (partial, 38/380).
- Match-end minutes include stoppage time, slightly inflating per-90 rates.
  Actual match end minutes range from 91 to 101 across the season.
- Position is a modal event position, not the lineup's declared role.
- StatsBomb ids do not join to other providers without a future mapping table.
- One player (Jordi Gómez, Sunderland) appears with 0 minutes: entered as a
  substitute at the match-end minute. Correctly counted as not an appearance
  (minutes > 0 required), but present in `player_match` for source fidelity.
- Raw event files total ~1.1 GB, much larger than the original ~108 MB estimate
  (per-file sizes are 3–10x larger than the 12-match sample suggested).
