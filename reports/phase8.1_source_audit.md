# Phase 8.1: Multi-Season Data Source Audit

**Date:** 2026-10-28  
**Status:** Research Complete — Ready for Review  
**Git Commit:** fb2e017 (Phase 7 baseline)

---

## 1. Current Project Coverage (Verified)

| Dataset | Seasons | Matches | Teams | Players | Source |
|---------|---------|---------|-------|---------|--------|
| Fixtures | 2015/16 only | 380 | 20 | — | StatsBomb Open Data |
| Player-Match | 2015/16 only | 10,469 rows | 20 | 550 | StatsBomb events + lineups |
| Player-Season | 2015/16 only | 550 | 20 | 550 | StatsBomb aggregated |
| Partial 2003/04 | Fixtures only | 38 | 20 | — | StatsBomb (19/38 match weeks) |

**Key constraint:** Player-level data (goals, assists, xG, xA, minutes, shots) exists **only for 2015/16**. The 2003/04 season has fixtures but no event/lineup data, so no player statistics can be derived.

All evaluation pipelines (Phases 2–4) correctly filter to 2015/16. Dashboard `load_fixtures()` now filters by `TARGET_SEASON_ID=27` by default.

---

## 2. Candidate Data Sources — Detailed Assessment

### A. StatsBomb Open Data (Current Primary)

| Aspect | Details |
|--------|---------|
| **URL** | https://github.com/statsbomb/open-data |
| **License** | Open for research, attribution required |
| **PL Coverage** | 2015/16 (complete, 380 matches), 2003/04 (partial, 38 matches, 19 match weeks) |
| **Other Comps** | 24 competitions total (WSL, Champions League, World Cups, etc.) |
| **Granularity** | Match events (every touch), lineups, 360-frame data for selected matches |
| **Player IDs** | StatsBomb internal IDs (stable within source) |
| **Team IDs** | StatsBomb internal IDs |
| **Match IDs** | StatsBomb internal IDs |
| **Available Fields** | Goals, assists, xG, xA, shots, shots on target, passes, carries, pressures, etc. |
| **Access** | Direct HTTP from GitHub raw; no API key; rate limits not documented |
| **Redistribution** | Permitted with attribution; raw JSON ~1.1 GB for 2015/16 PL events |
| **Limitations** | Only 1 complete PL season; no player metadata (age, DOB, height, foot, nationality) beyond name/position |

**Verdict:** Keep as primary for 2015/16. Cannot expand PL coverage.

---

### B. openfootball/football.json (Match Results)

| Aspect | Details |
|--------|---------|
| **URL** | https://github.com/openfootball/football.json |
| **License** | Public Domain (CC0) — no restrictions |
| **PL Coverage** | 2010/11 through 2026/27 (17 seasons, 380 matches each = ~6,460 matches) |
| **Granularity** | Match-level: date, time, round, team1, team2, score (HT/FT) |
| **Player Data** | **None** — match results only |
| **Team Names** | Full club names (e.g., "Manchester City FC") — need mapping to StatsBomb IDs |
| **Match IDs** | None — would need composite key (date + teams) |
| **Access** | Direct HTTP from GitHub raw; JSON format; small files (~50 KB/season) |
| **Redistribution** | Unrestricted (Public Domain) |
| **Limitations** | No player statistics, no xG, no lineups, no minutes |

**Verdict:** **Recommended for match-level historical results (2010/11–2026/27).** Can extend match prediction and league simulation to 17 seasons. Requires team name mapping to StatsBomb IDs for 2015/16 overlap.

---

### C. openfootball/england (Match Results + Squad Lists)

| Aspect | Details |
|--------|---------|
| **URL** | https://github.com/openfootball/england |
| **License** | Public Domain (CC0) |
| **PL Coverage** | 2000/01 through 2026/27 (27 seasons) + lower divisions |
| **Granularity** | Match results (Football.TXT format) + squad lists (player names, positions, nationalities, DOB where available) |
| **Player Data** | Squad rosters only — **no match-level player statistics** |
| **Player Metadata** | Names, positions, nationalities, date of birth (where available) — useful for age-based scouting |
| **Access** | GitHub raw; text format; parsable |
| **Redistribution** | Unrestricted (Public Domain) |
| **Limitations** | No xG, no minutes, no goals/assists per match; squad lists may be incomplete for early seasons |

**Verdict:** **Recommended for player registry enrichment (age, nationality, position history) and extended match results back to 2000/01.** Requires player name matching to StatsBomb/FPL IDs.

---

### D. vaastav/Fantasy-Premier-League (Player-Match Stats)

| Aspect | Details |
|--------|---------|
| **URL** | https://github.com/vaastav/Fantasy-Premier-League |
| **License** | MIT (per repository) — permissive, allows redistribution |
| **PL Coverage** | 2016/17 through 2025/26 (10 complete seasons) + 2026/27 (partial) |
| **Granularity** | Gameweek-level per player (merged_gw.csv per season) |
| **Player IDs** | FPL element IDs (stable within FPL, 2016/17 onward) |
| **Team IDs** | FPL team IDs (stable) |
| **Available Fields (all seasons):** | minutes, goals, assists, clean sheets, saves, bonus, BPS, influence, creativity, threat, ICT index, total_points, value, transfers, was_home, opponent_team, team_a_score, team_h_score, kickoff_time, round (GW) |
| **Available Fields (2022/23+):** | **expected_goals (xG), expected_assists (xA), expected_goal_involvements, expected_goals_conceded, starts** |
| **Player Metadata (cleaned_merged_seasons.csv):** | name, position (GK/DEF/MID/FWD), team, season |
| **Master Team List:** | FPL team ID ↔ team name mapping per season |
| **Access** | GitHub raw CSV/JSON; ~500 MB total for all seasons |
| **Redistribution** | MIT license — permitted with attribution |
| **Limitations** | xG/xA only from 2022/23; no shot-level data; FPL IDs ≠ StatsBomb IDs; player name matching required; some columns (xP) have lookahead bias per repo documentation |

**Verdict:** **Recommended as primary multi-season player-stat source (2016/17–2025/26).** Provides 10 seasons of player-match data with xG/xA for the last 4 seasons. Requires player identity reconciliation.

---

### E. Understat (Player xG/xA + Match xG)

| Aspect | Details |
|--------|---------|
| **URL** | https://understat.com/league/EPL |
| **License** | **No open license** — robots.txt: `User-agent: * Disallow: /` |
| **PL Coverage** | 2014/15 through 2026/27 (13 seasons) |
| **Granularity** | Match xG, player xG/xA, shots, position, team |
| **Access** | Web UI with CSV/JSON/XLSX export buttons; no public API |
| **Redistribution** | Not permitted (ToS unclear, scraping explicitly disallowed) |
| **Limitations** | Legal/access risk; no stable player IDs across seasons; scraping blocked |

**Verdict:** **Do not use.** Licensing and scraping restrictions make it unsuitable for a reproducible pipeline.

---

### F. FBref / Sports-Reference

| Aspect | Details |
|--------|---------|
| **URL** | https://fbref.com/en/comps/9/history/Premier-League-Seasons |
| **License** | ToS prohibits scraping and AI use; Cloudflare protected |
| **PL Coverage** | 1992/93–present (full history) |
| **Granularity** | Match results, team stats, player stats (basic + advanced with xG from ~2017) |
| **Access** | JavaScript-heavy; Cloudflare challenge; no API |
| **Redistribution** | Prohibited |
| **Limitations** | Access blocked in this environment; legal restrictions |

**Verdict:** **Do not use.** Access and licensing barriers.

---

### G. Football-Data.co.uk (Match Results + Betting Odds)

| Aspect | Details |
|--------|---------|
| **URL** | https://www.football-data.co.uk/englandm.php |
| **License** | Not clearly stated; historical use tolerated for research |
| **PL Coverage** | 1993/94–present (CSV per season) |
| **Granularity** | Match results, scores, half-time scores, betting odds (1X2, O/U, Asian handicap) |
| **Player Data** | None |
| **Access** | HTTP 403 from this environment; may work elsewhere |
| **Redistribution** | Unclear |
| **Limitations** | Blocked in this environment; no player stats; no xG |

**Verdict:** **Fallback only** for match results if openfootball sources fail. Not accessible here.

---

### H. jokecamp/FootballData (Mirror of football-data.co.uk + More)

| Aspect | Details |
|--------|---------|
| **URL** | https://github.com/jokecamp/FootballData |
| **License** | MIT |
| **PL Coverage** | 1992/93–2015/16 (EPL 1992-2015 folder), 2011-2019, 2015-16, 2016-17 |
| **Granularity** | Match results (CSV) from football-data.co.uk |
| **Player Data** | None |
| **Access** | GitHub raw |
| **Redistribution** | MIT — permitted |
| **Limitations** | Appears incomplete/unmaintained; last commit years ago; only match results |

**Verdict:** Superseded by openfootball/football.json which has cleaner, more recent data.

---

## 3. Compatibility Comparison Matrix

| Requirement | football.json | openfootball/england | FPL (vaastav) | StatsBomb (current) |
|-------------|---------------|---------------------|---------------|---------------------|
| Match results (2010+) | ✅ 17 seasons | ✅ 27 seasons | ✅ 10 seasons (via fixture) | ✅ 1 season |
| Player-match stats | ❌ | ❌ (rosters only) | ✅ 10 seasons | ✅ 1 season |
| xG / xA (player) | ❌ | ❌ | ✅ 2022/23+ | ✅ 2015/16 |
| Minutes / appearances | ❌ | ❌ | ✅ All | ✅ 2015/16 |
| Player metadata (age, DOB) | ❌ | ✅ Partial | ❌ (position only) | ❌ |
| Stable player IDs | ❌ | ❌ | ✅ FPL element ID | ✅ StatsBomb ID |
| Stable team IDs | ❌ (names only) | ❌ (names only) | ✅ FPL team ID | ✅ StatsBomb ID |
| Match IDs | ❌ | ❌ | ❌ (composite key) | ✅ StatsBomb match_id |
| License | CC0 (Public Domain) | CC0 (Public Domain) | MIT | Attribution required |
| Access | GitHub raw (easy) | GitHub raw (parsable) | GitHub raw (CSV) | GitHub raw (JSON, large) |

**Key Integration Challenge:** No common player identifier across sources. Reconciliation must use **name + team + season + position** fuzzy matching, with manual verification for high-value players.

---

## 4. Recommended Data Architecture

### 4.1 Provider-Specific Adapters (New `src/data/ingest_<provider>.py` modules)

```
src/data/
├── config.py                 # Shared paths, constants
├── ingest.py                 # StatsBomb fixtures (existing)
├── ingest_players.py         # StatsBomb events/lineups (existing)
├── ingest_fpl.py             # NEW: FPL player-match + metadata
├── ingest_openfootball_matches.py  # NEW: football.json match results
├── ingest_openfootball_squads.py   # NEW: england squad lists
├── player_mapping.py         # NEW: cross-provider player identity resolution
├── team_mapping.py           # NEW: cross-provider team identity resolution
├── validate.py               # Existing validation
└── validate_players.py       # Existing validation
```

### 4.2 Raw Snapshot Preservation

```
data/raw/
├── statsbomb/                # Existing (events/, lineups/, matches/)
├── fpl/                      # NEW: merged_gw.csv, cleaned_merged_seasons.csv, master_team_list.csv per season
├── openfootball/
│   ├── football.json/        # NEW: en.1.json per season
│   └── england/              # NEW: 1-premierleague.txt, squad lists per season
```

### 4.3 Standardized Processed Schemas

| Table | Key Columns | Provenance |
|-------|-------------|------------|
| `matches` | match_id, season_id, match_date, match_week, home_team_id, away_team_id, home_score, away_score, home_xg, away_xg, source | football.json (primary), StatsBomb (2015/16) |
| `teams` | team_id, team_name, season_id, source_team_id, source | FPL master_team_list + openfootball mapping |
| `players` | player_id, player_name, birth_date, nationality, height, preferred_foot, source_player_id, source | openfootball squads + FPL position + StatsBomb |
| `player_season` | player_id, season_id, team_id, position, appearances, starts, minutes, goals, assists, xg, xa, shots, shots_on_target, source | FPL (primary), StatsBomb (2015/16 overlay) |
| `player_match` | player_id, match_id, team_id, started, minutes, goals, assists, xg, xa, shots, shots_on_target, source | FPL (primary), StatsBomb (2015/16 overlay) |
| `player_mapping` | player_id, statsbomb_id, fpl_id, openfootball_name, confidence, notes | NEW: identity resolution |

### 4.4 Identity Resolution Strategy

1. **Team Mapping:** Exact name match first (2015/16 overlap), then fuzzy match with manual review for discrepancies.
2. **Player Mapping:** 
   - Step 1: Exact (name + team + season + position) across sources
   - Step 2: Fuzzy name (Levenshtein) + team + season
   - Step 3: Manual curation for high-value players (top 100 by minutes)
   - Store mapping with confidence score (0.0–1.0) and reviewer notes
3. **Season Alignment:** All sources use "YYYY-YY" or "YYYY/YYYY" — normalize to `season_id` (e.g., 2015 for 2015/16).

### 4.5 Incremental Updates & Reproducibility

- Each adapter is a standalone script with `--offline` mode using cached raw files
- Raw files are immutable; processed tables are deterministic rebuilds
- `requirements.txt` pins all dependencies
- Git tracks only source code and configs; `data/raw/` and `data/processed/` are gitignored
- Ingestion logs (`ingested_at`, source URLs, raw file hashes) in every processed row

---

## 5. Multi-Season Evaluation Design

### 5.1 Season Boundaries for Validation

| Phase | Task | Training Seasons | Validation Season | Notes |
|-------|------|------------------|-------------------|-------|
| 2 | Goals/Assists | 2016/17–2021/22 | 2022/23 | Requires xG (FPL from 2022/23) |
| 2 | Goals/Assists (no xG) | 2016/17–2022/23 | 2023/24 | 8 seasons train, 1 test |
| 3 | Market Value | — | — | No market value source identified |
| 4 | Match Outcome | 2010/11–2021/22 | 2022/23 | 12 seasons train via football.json |
| 4 | League Sim | 2010/11–2022/23 | 2023/24 | Full-season simulation |
| 5 | Scouting | 2016/17–2022/23 | 2023/24 | Retrospective validation |

**Chronological split is mandatory** — no random splits. Features at prediction time `t` must use only data from seasons `< t`.

### 5.2 Feature Availability Rules

| Feature | Available From | Notes |
|---------|---------------|-------|
| Goals, assists, minutes, appearances | 2016/17 | FPL baseline |
| xG, xA (player) | 2022/23 | FPL added these columns |
| Starts | 2022/23 | FPL added `starts` column |
| Match xG (team) | 2014/15 | Understat (not usable) / footyStats? / estimate from results |
| Player age | 2000/01 | openfootball squads (DOB where available) |
| Market value | Never | No public source with history |

### 5.3 Cross-Provider Reconciliation Checks

1. **Match Results Consistency:** For 2015/16 and 2016/17–2025/26 overlap, verify home/away scores match between football.json, FPL fixtures, and StatsBomb (where available).
2. **Team Name Mapping:** 1:1 mapping between football.json names, FPL team names, and StatsBomb team names for all overlapping seasons.
3. **Player Aggregation:** Sum of FPL `player_match` minutes/goals/assists per player-season ≈ `player_season` totals (within 1% tolerance).
4. **Missingness Audit:** Report % of players in FPL with no StatsBomb/openfootball match for each season.

### 5.4 Metrics vs. 2015/16 Benchmark

- **Player Goals/Assists MAE:** Compare multi-season model MAE on 2023/24 holdout vs. 2015/16 single-season baseline (Recent Form: 0.153 MAE goals)
- **Match Outcome Log Loss:** Compare multi-season model on 2022/23 vs. 2015/16 Global Frequencies (1.095)
- **League Simulation:** Title probability calibration — do simulated title frequencies match observed over 10+ seasons?

---

## 6. Proposed Ingestion & Validation Stages

| Stage | Scope | Deliverable | Dependencies |
|-------|-------|-------------|--------------|
| 8.1 | **Source Audit** (this report) | `reports/phase8.1_source_audit.md` | — |
| 8.2 | Team & Season Registry | `data/processed/teams.parquet`, `seasons.parquet` | football.json, FPL master_team_list |
| 8.3 | Match Results Ingestion | `data/processed/matches.parquet` (2010/11–2025/26) | football.json adapter |
| 8.4 | FPL Player-Match Ingestion | `data/processed/fpl_player_match.parquet`, `fpl_player_season.parquet` | FPL adapter |
| 8.5 | Player Identity Resolution | `data/processed/player_mapping.parquet` | StatsBomb 2015/16, FPL 2016/17+, openfootball squads |
| 8.6 | Unified Player-Season Table | `data/processed/player_season_unified.parquet` | Merge StatsBomb (2015/16) + FPL (2016/17+) |
| 8.7 | Extended Evaluation | Updated Phase 2–5 reports on multi-season data | All above |
| 8.8 | Dashboard Integration | New "Season Selector", multi-season charts | Dashboard data loaders |

**Decision Required Before Stage 8.3:** Confirm football.json as primary match-results source and FPL as primary player-stat source. Accept MIT/CC0 licenses.

---

## 7. Explicit Decisions Requiring Review

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| 1 | Primary match-results source for 2010/11–2025/26 | football.json (CC0) vs. Football-Data.co.uk (unclear) vs. openfootball/england (CC0) | **football.json** — structured JSON, 17 PL seasons, Public Domain |
| 2 | Primary player-stat source for 2016/17–2025/26 | FPL (MIT, 10 seasons, xG from 2022/23) vs. StatsBomb (only 2015/16) | **FPL** — only source with 10 seasons of player-match data |
| 3 | Player identity resolution approach | Automated fuzzy + manual curation vs. exact-only | **Automated fuzzy + manual review for top 100 players** — balance coverage and accuracy |
| 4 | Market value data | Transfermarkt (subscription) vs. none | **None for now** — no viable public source; document as limitation |
| 5 | 2003/04 StatsBomb partial data | Include in fixtures table vs. exclude | **Exclude from player modelling** (no events/lineups); keep in fixtures for completeness with `data_quality="partial"` flag |
| 6 | xG availability cut-off | Only model xG for 2022/23+ vs. impute for earlier | **Only 2022/23+** — no reliable imputation; document clearly |
| 7 | License compliance for combined dataset | CC0 + MIT + Attribution → combined license | **Permissive** — all sources allow redistribution with attribution; maintain per-source provenance columns |

---

## 8. Report Location & Git Status

- **Report:** `reports/phase8.1_source_audit.md` (this file)
- **Project Plan:** Updated in `PROJECT_PLAN.md` with Phase 8.1 status
- **Git Status:** Clean — no code changes, only documentation

```bash
git status
# On branch master
# Untracked files:
#   reports/phase8.1_source_audit.md
#   (generated parquet files in reports/ — correctly gitignored)
```

---

## 9. Uncertainties & Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FPL player IDs not stable across seasons | Low | High | FPL docs state element IDs are stable; verify on 2016/17→2017/18 overlap |
| football.json team names don't match FPL/StatsBomb | Medium | Medium | Build mapping table; 2015/16 overlap enables validation |
| FPL xG lookahead bias (xP column) | Confirmed | Medium | Exclude `xP`; use only `expected_goals`, `expected_assists` |
| openfootball squad lists incomplete for early seasons | Medium | Low | Flag missing DOB/nationality; don't impute |
| StatsBomb 2015/16 player IDs don't match FPL 2016/17 for same players | High | High | Expect zero overlap — different ID systems; rely on name/team matching |
| FPL data stops updating after 2024/25 (repo notice) | High | Medium | 2025/26 and 2026/27 may be incomplete; verify match counts |

---

## 10. Next Steps (Awaiting Review)

1. **Review this report** — confirm source selections and architecture
2. **Approve Stage 8.2** — team/season registry ingestion
3. **Decide on player identity resolution budget** — manual curation effort for top players
4. **Confirm no market-value pursuit** without Transfermarkt access

**Do not proceed to implementation until review sign-off.**