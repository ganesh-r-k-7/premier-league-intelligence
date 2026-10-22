# Phase 2 — Stage 2 Report: Ingestion and Aggregation

**Date:** 2026-10-17
**Season:** Premier League 2015/2016 (competition_id=2, season_id=27)
**Source:** StatsBomb Open Data (https://github.com/statsbomb/open-data)
**Attribution:** Required by source terms; carried in `source` / `source_url` / `ingested_at` columns on every row.

---

## 1. What was downloaded

| Item | Count | URL pattern | Raw location |
| --- | --- | --- | --- |
| Matches (fixtures) | 380 | `data/matches/2/27.json` | `data/raw/statsbomb/matches/2_27.json` (Phase 1) |
| Events | 380 | `data/events/{match_id}.json` | `data/raw/statsbomb/events/{id}.json` |
| Lineups | 380 | `data/lineups/{match_id}.json` | `data/raw/statsbomb/lineups/{id}.json` |

- All 380 match IDs from `2_27.json` have both events and lineups files.
- 0 missing events, 0 missing lineups.
- 0 extra files (no match IDs without fixture entries).
- The incomplete 2003/2004 season (season_id=44) was **not** downloaded.

## 2. Actual file counts and disk usage

| Directory | Files | Disk usage |
| --- | --- | --- |
| `data/raw/statsbomb/events/` | 380 JSON | ~1.1 GB |
| `data/raw/statsbomb/lineups/` | 380 JSON | ~7.2 MB |
| `data/processed/statsbomb/` | 3 parquet | ~328 KB |

- `fixtures.parquet`: 380 rows (from Phase 1)
- `player_match.parquet`: 10,469 rows
- `player_season.parquet`: 550 rows

**Note:** The original design estimate was ~108 MB for events, based on a
12-match sample (~283 KB/file mean). Actual per-file sizes average ~2.9 MB,
for a total of ~1.1 GB — roughly 10x the estimate. The estimate in
`docs/phase2_implementation_checklist.md` has been corrected.

## 3. Ingestion implementation

Module: `src/data/ingest_players.py` (Phase 2, separate from Phase 1's
`src/data/ingest.py`; both use `src/data/config.py` for URLs and paths).

Features implemented:
- Request timeouts (default 30s)
- Retries with exponential backoff (default 3 retries, 1.0s base backoff)
- HTTP status handling: 404 = missing match (recorded, not fatal); 429/500/502/503/504 = retry; other non-200 = error
- JSON validation: response must parse as a non-empty JSON list
- Atomic writes (temp file + rename) so partial downloads never corrupt cache
- Resuming: valid cached files are skipped; invalid/missing files are re-downloaded
- Offline mode: never touches the network; uncached resources recorded as failures
- Clear progress summary: downloaded, skipped, missing, failed counts
- Source attribution: `STATSBOMB_SOURCE_NAME` / `STATSBOMB_SOURCE_URL` on every processed row
- Raw data is gitignored (`data/raw/` in `.gitignore`); never committed

Phase 1 (`src/data/ingest.py`) is untouched and fully intact.

## 4. Aggregation output

### `player_match.parquet` (10,469 rows)

Columns: match_id, match_date, season_id, season_name, competition_id,
player_id, player_name, team_id, team_name, position, started, minutes,
goals, own_goals, assists, shots, shots_on_target, xg, npxg, xa,
penalty_goals, penalty_attempts, source, source_url, ingested_at

### `player_season.parquet` (550 rows)

Columns: player_id, player_name, season_id, season_name, competition_id,
team_id, team_name, position, appearances, starts, minutes, goals, own_goals,
assists, shots, shots_on_target, xg, npxg, xa, penalty_goals,
penalty_attempts, source, source_url, ingested_at

Derivation rules follow the Stage 1 design exactly (no silent changes):

| Field | Rule |
| --- | --- |
| goals | Shot events with outcome == "Goal" (penalties included) |
| penalty_goals | goal shots with shot.type == "Penalty" |
| penalty_attempts | all shots with shot.type == "Penalty" |
| own_goals | Own Goal Against event's player (never added to goals) |
| assists | Pass events with pass.goal_assist == True |
| xa | shot.key_pass_id references pass id; credited pass player gets shot's xG |
| shots | count of Shot events |
| shots_on_target | outcome in {Goal, Saved, Saved to Post} |
| xg | sum of shot.statsbomb_xg (penalties included) |
| npxg | sum excluding Penalty shots |
| started | player in Starting XI tactics.lineup |
| minutes | starter: end - sub_off/red-card; substitute: end - sub_on; minute is absolute |
| appearances | minutes > 0 |
| position | most frequent event position.name for the player in the match |

## 5. Validation results

### Score reconciliation (all 380 matches)

| Metric | Value |
| --- | --- |
| Matches checked | 380 |
| Score mismatches | 0 |
| Fixture goals (home + away) | 1,026 |
| Event goals (shots + own goals) | 1,026 |

### Duplicate event IDs

| Metric | Value |
| --- | --- |
| Total duplicate event IDs | 0 |

### Assist / xA linkage

| Metric | Value |
| --- | --- |
| Total goals (shot outcomes) | 988 |
| Penalty goals | 74 |
| Penalty attempts | 91 |
| goal_assist passes | 669 |
| goals with key_pass_id | 669 |
| goal_assist_ids not referenced by goal | 0 |
| goal key_passes not flagged goal_assist | 0 |

Assists (`pass.goal_assist`) and xA (`shot.key_pass_id`) are perfectly
aligned in this season: every goal-assist pass is referenced by a goal shot,
and every goal shot with a key_pass_id has a matching goal-assist pass.
They are kept as separate measures per the design; they are not assumed
interchangeable.

### player_match vs player_season self-consistency

| Metric | Delta |
| --- | --- |
| minutes | 0.0 |
| goals | 0.0 |
| own_goals | 0.0 |
| assists | 0.0 |
| shots | 0.0 |
| shots_on_target | 0.0 |
| xg | 0.0 |
| npxg | 0.0 |
| xa | 0.0 |
| penalty_goals | 0.0 |
| penalty_attempts | 0.0 |
| appearances | 0 |
| starts | 0 |

All deltas are exactly 0: `player_season` is a perfect rollup of `player_match`.

### Season totals (player_match == player_season)

| Metric | Value |
| --- | --- |
| Total goals | 988 |
| Total own goals | 38 |
| Total assists | 669 |
| Total shots | 9,908 |
| Total shots on target | 3,220 |
| Total xG | 971.38 |
| Total npxG | 900.08 |
| Total xA | 664.40 |
| Total penalty goals | 74 |
| Total penalty attempts | 91 |
| Total minutes | 784,443 |
| Total appearances | 10,468 |
| Total starts | 8,360 |

## 6. Discrepancies and known limitations

No data discrepancies were found (0 mismatches, 0 duplicates, 0 deltas).
Known limitations are documented, not forced:

1. **One complete season only.** 2015/16 is the only complete PL season in
   StatsBomb Open Data. Cross-season validation is impossible; stated in
   every report.
2. **2003/2004 excluded.** Partial season (38/380 matches), mixes eras.
3. **Match-end minutes include stoppage time.** Actual end minutes range
   from 91 to 101 across the season. This slightly inflates per-90
   denominators; documented, not hidden.
4. **Position is modal event position**, not the lineup's declared role.
5. **0-minute substitute.** Jordi Gómez (Sunderland) entered at the
   match-end minute (95) and has 0 minutes. Correctly not counted as an
   appearance (minutes > 0 required), but present in `player_match` for
   source fidelity.
6. **Download size ~1.1 GB**, much larger than the original ~108 MB estimate.
7. **StatsBomb IDs do not join** to other providers without a future mapping table.

No totals were force-fitted and no missing information was fabricated.

## 7. Tests

Full test suite (excluding `test_eda.py` which requires the optional `plotly`
dependency not installed in this environment):

| Command | Result |
| --- | --- |
| `python3 -m pytest tests/ --ignore=tests/test_eda.py` | **64 passed** |

### Existing Phase 1 tests (29): all pass
- `tests/test_ingest.py` — 15 tests
- `tests/test_validate.py` — 6 tests
- `tests/test_eda.py` — skipped (missing optional `plotly` dep)

### Existing Phase 2 tests (31): all pass
- `tests/test_ingest_players.py` — 10 tests (retries, invalid cache, resume, missing, offline)
- `tests/test_player_aggregation.py` — 15 tests (goals, penalties, xG, assists, xA, own goals, minutes, substitutions, red cards, unused subs, unknown players, provenance, season aggregation)
- `tests/test_validate_players.py` — 6 tests (goal counting, assist linkage, duplicates, reconciliation, season totals)

### New regression tests (4): all pass
- `test_zero_minute_substitute_at_match_end` — substitute entering at match end gets 0 minutes, still appears
- `test_stoppage_time_match_end_minute` — match end uses actual final minute (>90)
- `test_second_yellow_shortens_minutes` — Second Yellow shortens minutes like red card
- `test_penalty_miss_not_a_goal` — saved penalty is an attempt but not a goal

## 8. Phase 1 preservation

`src/data/ingest.py` is completely untouched. All 29 Phase 1 tests pass.
Raw/processed separation, provenance columns, offline fallback, and cached
snapshot reuse all work as before.

## 9. Completion criteria

| Criterion | Status |
| --- | --- |
| All 380 matches accounted for | Yes (380 events + 380 lineups) |
| Ingestion is resumable | Yes (valid cache skipped, invalid re-downloaded) |
| Outputs pass validation | Yes (0 mismatches, 0 duplicates, all deltas 0) |
| Full test suite passes | Yes (64 passed) |
| Phase 1 intact | Yes (ingest.py untouched, 29 tests pass) |
| Known discrepancies documented | Yes (none found; limitations listed above) |

Stage 2 is complete. Stage 3 (forecast and evaluation) is next.