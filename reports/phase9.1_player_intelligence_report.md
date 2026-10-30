# Phase 9.1: Multi-Season Player Intelligence Report

**Date:** 2026-10-29  
**Status:** Complete  
**Git Commit:** TBD

---

## Summary

Phase 9.1 implements a comprehensive multi-season player intelligence system using historical Premier League data from FPL (2016/17–2025/26) and StatsBomb (2015/16). The system provides reusable metrics for player performance analysis across multiple seasons, with identity safeguards and position-specific benchmarking.

---

## 1. Modules Created

### `src/features/player_intelligence.py`
Main module implementing the player intelligence analytics:

**Core Functions:**
- `compute_performance_metrics()` — Per-90 rates, efficiency metrics, playing time metrics, xG/xA availability flags
- `compute_position_benchmarks()` — Position-specific season benchmarks (median, mean, std, percentiles)
- `merge_with_benchmarks()` — Percentile ranks and z-scores relative to position-season peers
- `compute_season_consistency()` — Season-to-season consistency (CV, trend, R²)
- `compute_match_consistency()` — Match-to-match variability (rolling CV, streaks)
- `compute_season_over_season_changes()` — Season-over-season changes with pct change
- `compute_development_trends()` — Linear trend (slope, R², classification)
- `compute_breakout_score()` — Breakout detection (production, minutes, per-90, consistency)
- `compute_decline_score()` — Decline detection (inverse of breakout)
- `get_fpl_player_seasons()` — Load FPL player seasons with filtering
- `get_position_band_fpl/statsbomb/auto()` — Position band mapping for FPL/StatsBomb
- `get_same_identity_records()` — Identity-aware career retrieval
- `get_career_trajectory()` — Full career trajectory with metrics
- `find_breakout_players()` / `find_declining_players()` — Dashboard queries
- `get_top_performers_by_metric()` / `get_consistency_leaders()` — Dashboard helpers
- `get_development_summary()` — Player development summary

**Configuration:** `PlayerIntelligenceConfig` dataclass with all thresholds and weights.

---

## 2. Data Sources & Coverage

| Source | Seasons | Players (min 450 min) | Key Metrics |
|--------|---------|----------------------|-------------|
| FPL (remediated) | 2016/17–2025/26 (10) | 3,971 player-seasons | Goals, assists, xG, xA, points, minutes, starts, appearances |
| StatsBomb | 2015/16 | 550 | Goals, assists, xG, npxG, xA, shots, shots on target |

**Position Coverage:**
- FPL: 2016–2019 lack position data (2,620 UNKNOWN); 2020–2025 fully covered
- StatsBomb: Full position data (GK, DEF, MID, WING, FWD)

---

## 3. Key Analytical Capabilities

### A. Performance Production
- Per-90 rates: goals, assists, xG, xA, shots, shots on target, clean sheets, saves
- Efficiency: goals/shot, goals/shot on target, goals over xG, assists over xA
- Playing time: starts rate, minutes per appearance, points per 90/appearance
- xG/xA/starts availability flags

### B. Consistency Analysis
- **Season consistency:** CV, trend (slope, R²), range across 2+ seasons
- **Match consistency:** Rolling mean/std/CV (5-match window), streak metrics
- **Consistency leaders:** Lowest CV for any metric

### C. Player Development
- **Season-over-season changes:** Absolute and % change for 25+ metrics
- **Development trends:** Linear regression slope, R², trend classification (improving/declining/stable/unclear)
- **Trend classification:** improving (>0.1, R²>0.3), declining (<-0.1, R²>0.3), stable, unclear

### D. Breakout Detection
**Score components:**
- Production improvement (goals/assists/points) — 30%
- Minutes increase — 20%
- Per-90 efficiency improvement — 30%
- Consistency improvement — 20%

**Top 2024 breakout candidates:** Antonee Robinson (39.0), Rodrigo Bentancur (18.9), Erling Haaland (12.8)

### E. Decline Detection
**Score components:**
- Production decline — 40%
- Minutes decline — 20%
- Per-90 decline — 20%
- Consistency decline — 20%

**Top 2023 decline:** John Egan (0.76), Mason Holgate (0.72), Aaron Hickey (0.72)

---

## 3. Position Benchmarks (2020–2025)

| Season | Band | Players | Goals/90 Median | Assists/90 Median | xG/90 Median |
|--------|------|---------|-----------------|-------------------|--------------|
| 2023 | DEF | 155 | 0.039 | 0.050 | 0.047 |
| 2023 | MID | 172 | 0.18 | 0.09 | 0.14 |
| 2023 | FWD | 44 | 0.40 | 0.16 | 0.40 |
| 2023 | GK | 32 | 0.00 | 0.00 | 0.00 |

*Benchmarks include: median, mean, std, q25/q75, p10/p90 for 17 metrics*

---

## 4. Identity Safeguards

### FPL Element ID Reuse
- **Audit:** 866 element IDs scanned across 10 seasons
- **Critical:** ID 748 used by 5 different players in 2024/25
- **High:** 372 IDs reused across adjacent seasons (expected FPL behavior)
- **Medium:** 468 IDs reused across non-adjacent seasons
- **Review file:** `fpl_id_reuse_severity` column added to review file

### Identity-Aware Analysis
- **Canonical mappings:** Only 1 confirmed (test entry from Phase 8.5C)
- **Career trajectories:** Only built for confirmed identities
- **Unverified identities:** Explicitly flagged, excluded from longitudinal analysis
- **FPL ID reuse:** Season-scoped canonical IDs prevent collisions

---

## 5. Dashboard Integration

### New Page: "Player Intelligence"
- **Filters:** Season, position band, metric, min minutes
- **Tabs:**
  - Top Performers (goals/90, assists/90, points, xG/90, etc.)
  - Consistency Leaders (lowest CV)
  - Breakout Candidates (high breakout_score_pctl)
  - Declining Players (high decline_score_pctl)
  - Development Profiles (career trajectory, trends)
  - Position Benchmarks (median/q25/q75/p10/p90)

### Helper Functions for Dashboard
- `get_top_performers_by_metric()` — Top N by any metric
- `get_consistency_leaders()` — Lowest CV for any metric
- `find_breakout_players()` / `find_declining_players()` — Season filters
- `get_development_summary()` — Player profile with trends
- `get_career_trajectory()` — Identity-aware career (confirmed only)

---

## 5. Testing

### New Tests: `tests/test_player_intelligence.py` (22 tests)
- Utility functions: per-90, safe_divide, position bands, trend classification
- Data loading: filtering by season/minutes
- Performance metrics: computation, zero minutes handling
- Position benchmarks: multi-season, multi-band
- Consistency: season CV/trend, match-level rolling
- Development: season-over-season, linear trends
- Breakout/Decline: scoring, percentile ranking
- Dashboard helpers: top performers, consistency leaders
- Identity safeguards: position band mapping
- Determinism: reproducible breakout scores

### Test Results
```
22 passed (test_player_intelligence.py)
31 passed (test_player_identity_curation.py)
24 passed (test_data_contracts.py)
38 passed (test_dashboard.py)
Total: 115 passed
```

### Full Suite (Core)
```
Data Contracts: 79/79 checks passed (100%)
Player Identity: 31/31 tests passed
Data Contracts: 24/24 tests passed
Dashboard: 38/38 tests passed
```

---

## 6. Files Created/Modified

### New Files
```
src/features/player_intelligence.py          # Main module (~1,500 lines)
tests/test_player_intelligence.py            # 22 tests
reports/phase9.1_player_intelligence_report.md
```

### Modified Files
```
src/features/player_intelligence.py          # Updated position band handling
src/features/player_scouting.py              # (unchanged, referenced)
tests/test_player_intelligence.py            # 22 new tests
PROJECT_PLAN.md                              # Phase 9.1 completion
README.md                                    # Status update
```

---

## 6. Known Limitations

1. **No confirmed identities:** 1,363 review candidates still pending; career trajectories only work for confirmed IDs
2. **Early seasons lack position:** 2016–2019 FPL seasons have no position data (2,620 UNKNOWN)
3. **FPL ID reuse pervasive:** 866 element IDs scanned; only 25 stable across all seasons
4. **No StatsBomb multi-season:** Only 2015/16 available; no longitudinal StatsBomb analysis
5. **Breakout/Decline weights:** Heuristic, not validated against ground truth
6. **No market values:** Cannot assess financial efficiency or transfer value

---

## 7. Reproducible Commands

```bash
# Load data and compute metrics
python -c "
from src.features.player_intelligence import *
df = load_fpl_player_seasons(min_minutes=450)
df = compute_performance_metrics(df)
benchmarks = compute_position_benchmarks(df)
df = merge_with_benchmarks(df, benchmarks)

# Find breakout players in 2024
breakout = find_breakout_players(df, season=2024)
print(breakout.head(10))

# Find declining players in 2023
decline = find_declining_players(df, season=2023)
print(decline.head(10))

# Top performers
top = get_top_performers_by_metric(df, 'goals_per_90', season=2023, top_n=10)
print(top)

# Consistency leaders
consistency = get_consistency_leaders(df, metric='total_points', top_n=10)
print(consistency)
"

# Run tests
pytest tests/test_player_intelligence.py -v

# Full validation
python -m src.data.data_contracts

# Launch dashboard
streamlit run src/dashboard/app.py
# Navigate to "🔍 Player Intelligence"
```

---

## 8. Phase 9.1 Exit Gate — MET

✅ Player intelligence module created with 15+ analytical functions  
✅ Position benchmarks for 6 seasons × 4 bands (24 benchmarks)  
✅ Consistency analysis (season CV, trends, match-level rolling)  
✅ Breakout/decline detection with percentile ranking  
✅ Position benchmarks with 17 metrics across 4 bands × 6 seasons  
✅ Identity safeguards: season-scoped IDs, FPL reuse audit, pending review  
✅ 22 new tests pass; 115 total tests pass; 0 regressions  
✅ Data contracts: 79/79 checks pass  
✅ Dashboard integration: Player Intelligence page with filters, rankings, profiles  
✅ Documentation: report, PROJECT_PLAN.md, README.md updated  

---

## 9. Remaining Work (Post-9.1)

1. **Human review** of 1,363 identity candidates (296 cross-provider, 777 same-provider, 290 unresolved)
2. **External lookup** for 296 cross-provider candidates (Transfermarkt/Wikipedia)
3. **Apply confirmed mappings** → `canonical_identity_mapping.parquet`
4. **FPL element ID 748** resolution (requires external data)
5. **Phase 9.2:** Market value integration, transfer prediction, contract analysis

---

## 9. Commands

```bash
# Run new tests
pytest tests/test_player_intelligence.py -v

# Run full test suite (core)
pytest tests/ -k "not integration" -q

# Validate data contracts
python -m src.data.data_contracts

# Launch dashboard with Player Intelligence tab
streamlit run src/dashboard/app.py
```