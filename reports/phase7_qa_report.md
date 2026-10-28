# Phase 7 Quality Assurance and Reproducibility Report

**Date:** 2026-10-28  
**Status:** Complete  
**Git Commit:** 0941c78 (Phase 6) → TBD (Phase 7)

---

## Summary

Phase 7 hardens the Premier League Intelligence application by reconciling dataset inconsistencies, verifying end-to-end dashboard behavior, confirming reproducibility, expanding tests, and updating documentation.

---

## 1. Dataset Consistency Audit

### Issue Discovered

The dashboard reported **418 matches** and **29 teams**, while Phase 4 match-outcome evaluation reports used **380 matches** and **20 teams**.

### Root Cause

The `data/processed/statsbomb/fixtures.parquet` file contained two seasons:
- **2015/2016** (season_id=27): 380 matches, 20 teams, complete Premier League season
- **2003/2004** (season_id=2): 38 matches, 20 teams, **partial** (only 19 of 38 match weeks)

Player data (`player_season.parquet`, `player_match.parquet`) only contains 2015/16 data.

The dashboard's `load_fixtures()` was loading all 418 matches, but evaluation pipelines correctly filtered to 2015/16 only.

### Fix Applied

Updated `src/dashboard/data_loader.py`:
- `load_fixtures(season_id=None)` now filters by `TARGET_SEASON_ID=27` (2015/16) by default
- `get_team_names(season_id=None)` accepts optional season filter
- All dashboard pages automatically use the correct 380-match dataset

### Verification

| Dataset | Before Fix | After Fix | Evaluation Reports |
|---------|------------|-----------|-------------------|
| Matches | 418 | **380** | 380 |
| Teams | 29 | **20** | 20 |
| Players | 550 | 550 | 550 |

All dashboard counts now match evaluation reports.

---

## 2. Dashboard End-to-End Verification

### Launch Procedure Verified

```bash
source .venv/bin/activate
streamlit run src/dashboard/app.py
```

Server starts successfully on `http://localhost:8501`.

### Section-by-Section Verification

| Section | Status | Key Interactions Tested |
|---------|--------|------------------------|
| **Overview** | ✅ Works | Season stats display, top scorers table, report viewer dropdown |
| **Player Explorer** | ✅ Works | Name search, team filter, position filter, min-minutes slider, percentile bars, teammate comparison |
| **Scouting & Comparisons** | ✅ Works | 4 tabs: player comparison (side-by-side + radar), position percentiles (top-20 + histogram), priority weights (sliders + coverage), similar players (cosine/euclidean + feature contributions) |
| **Match Analysis** | ✅ Works | 4 tabs: baseline metrics table + charts, feature model evaluation, confusion matrix placeholder, report viewer |
| **League Simulator** | ✅ Works | Provider selection, simulation count, random seed, mode selection, progress bar, results tables + charts |

### Error Handling Verified

- Missing data files: Shows clear error with setup instructions
- Empty search results: Shows friendly empty state
- Invalid player selections: Handled gracefully
- Simulation bounds: Capped at 10,000 simulations

---

## 3. Reproducibility Verification

### Fixed Random Seeds

| Component | Seed | Verified |
|-----------|------|----------|
| League Simulator | `SimulatorConfig(random_seed=42)` | ✅ Identical results across runs |
| GLM Models | `random_state=42` in PoissonRegressor/NegativeBinomial | ✅ Deterministic |
| Match Model | `random_state=42` in LogisticRegression | ✅ Deterministic |
| Player Similarity | Fixed seed in cosine/euclidean computation | ✅ Deterministic |

### Data Ingestion

- `python -m src.data.ingest_players --offline` uses cached raw JSON files
- Raw files are immutable snapshots under `data/raw/statsbomb/`
- Processed Parquet files are deterministic given same raw input
- `--force` flag available for re-download

### Model Evaluation

- Chronological rolling-origin: training uses `match_week < cutoff` only
- Leakage tests in `tests/test_leakage.py` verify no future data contamination
- All 334 core tests pass including 4 leakage tests for baselines, 6 for match features, 4 for GLM, 4 for NB GLM, 6 for league simulator

### Environment

- Python 3.14.4 (system)
- Virtual environment `.venv/` with pinned `requirements.txt`
- All dependencies pinned to specific versions
- No unpinned or floating dependencies

### Git Hygiene

- `.gitignore` excludes: `data/raw/`, `data/processed/*.parquet`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `*.log`
- No raw data, generated Parquet, secrets, or virtual environments committed
- Only source code, tests, markdown reports, and configuration files tracked

---

## 4. Tests and Code Quality

### Test Suite Results

```
====================== 356 passed, 28 warnings in ~14s =======================
```

Breakdown:
- Core tests: 334 (data ingestion, features, models, evaluation, leakage prevention)
- Dashboard tests: 22 (`tests/test_dashboard.py` - data loaders, utilities, edge cases)

### New/Updated Tests

`tests/test_dashboard.py` covers:
- Data loader functions (existence, schema, filtering, search)
- Utility functions (formatting, display names, percentiles)
- Edge cases (empty results, missing teams/positions)
- Error handling (`DataLoadError`)

### Code Quality

- No new linting dependencies introduced
- Existing code style preserved (no formatter changes)
- Type hints maintained in new code
- No duplicate tests added

---

## 5. Documentation Updates

### README.md

- Corrected dataset coverage: **380 matches, 20 teams, 550 players** (2015/16 PL only)
- Updated Phase 7 status to ✅ Complete
- Fixed limitations list (removed incorrect "cup matches included")
- Added clear installation, data ingestion, and launch instructions

### PROJECT_PLAN.md

- Added Phase 7 completion summary with audit details
- Documented dataset fix, verification steps, reproducibility checks
- Phase 7 Exit Gate marked as MET

### Reports

- `reports/phase7_qa_report.md` (this document)

---

## 6. GitHub Checkpoint

### Files Modified

```
M  PROJECT_PLAN.md
M  README.md
M  src/dashboard/data_loader.py
M  tests/test_dashboard.py
```

### Files Added

```
A  reports/phase7_qa_report.md
```

### Untracked (Correctly Ignored)

- `data/raw/statsbomb/` — immutable raw JSON (~1.1 GB)
- `data/processed/statsbomb/*.parquet` — generated datasets
- `reports/phase*_*.parquet` — generated report artifacts
- `.venv/` — virtual environment
- `Premier_League_Intelligence_Specification.md:Zone.Identifier` — Windows artifact

### Commit

```
git add PROJECT_PLAN.md README.md src/dashboard/data_loader.py tests/test_dashboard.py reports/phase7_qa_report.md
git commit -m "chore: harden reproducibility and dashboard validation"
git push origin master
```

---

## 7. Known Limitations (Unchanged)

1. **Single season only** — 2015/16 PL only; 2003/04 partial data not used in evaluation
2. **No market values** — Scouting is performance-only
3. **No injury/rotation signals** — Lineup data shows who played, not why others didn't
4. **Independent Poisson assumption** — Scoreline simulation ignores goal dependence
5. **Models don't beat baselines** — Feature models underperform simple heuristics on this data

All limitations are explicitly displayed in the dashboard UI.

---

## 8. Phase 7 Exit Gate — MET

✅ Dataset consistency reconciled (380 matches, 20 teams everywhere)  
✅ Dashboard launches and all sections function  
✅ Fixed random seeds produce repeatable outputs  
✅ README commands work in clean environment  
✅ All 356 tests pass  
✅ No raw data, generated Parquet, secrets, or venv committed  
✅ Documentation updated with actual coverage and limitations  

---

## Next Steps (Phase 8 — Optional)

- Multi-season data upgrade (requires additional StatsBomb seasons)
- Fantasy points / captain optimization
- Golden Boot prediction
- Scenario analysis
- AI analyst narrative generation