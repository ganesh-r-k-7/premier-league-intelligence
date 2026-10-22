# Phase 3 Stage 3.1 — Player Feature Engineering Report

**Date:** 2026-10-21
**Status:** Complete — all tests pass, leakage prevention verified

---

## 1. Summary

This stage implements a pure, offline-testable feature engineering module for player performance prediction. The module computes 14 features per player per target matchweek using only historical data available before the cutoff (`match_week < target_matchweek`).

**Files created:**
- `src/features/player_features.py` — Feature extraction module
- `tests/test_player_features.py` — 20 unit tests (all passing)
- `reports/phase3_stage31_report.md` — This report

**Test results:** 134 total tests pass (114 existing + 20 new)

---

## 2. Feature Definitions

### 2.1 Player History Features

| Feature | Description | Computation | Fallback Chain |
|---------|-------------|-------------|----------------|
| `player_goals_per_90` | Historical goals per 90 minutes | `sum(goals) / sum(minutes) * 90` over player history | Position band league median → Global league median |
| `player_assists_per_90` | Historical assists per 90 minutes | `sum(assists) / sum(minutes) * 90` over player history | Position band league median → Global league median |
| `player_recent_goals_mean_3` | Mean goals in last 3 matches | Mean of goals in most recent 3 matches (by match_week) | Player history mean → Global mean |
| `player_recent_assists_mean_3` | Mean assists in last 3 matches | Mean of assists in most recent 3 matches | Player history mean → Global mean |
| `player_minutes_per_match` | Mean minutes played per match | Mean of minutes over player history | Position/team median → Team median → Global median |
| `player_starts_rate` | Fraction of matches started | Mean of `started` boolean over player history | Position/team rate → Team rate → Global rate |

**Fallback behavior:**
- If `len(player_history) >= min_history` (default 3) AND `sum(minutes) > 0`: use player's own per-90 rates
- If `len(player_history) >= min_history` but `sum(minutes) == 0`: use player's match-level means (not per-90)
- If `len(player_history) < min_history`: fall back to position band league median per-90
- If position unavailable: fall back to global league median per-90

### 2.2 Team Context Features

| Feature | Description | Source | Fallback |
|---------|-------------|--------|----------|
| `team_goals_per_match` | Team's average goals scored per match | Fixtures history (home_score/away_score for team) | Global average |
| `team_xg_per_match` | Team's average xG per match | Aggregated from player_match team xG per match | 0.0 (unavailable) |
| `opponent_goals_conceded_per_match` | Opponent's average goals conceded per match | Fixtures history (goals conceded by opponent) | Global average |

### 2.3 Position Context Features

| Feature | Description | Computation | Fallback |
|---------|-------------|-------------|----------|
| `position_goals_per_90_league` | League-wide goals per 90 for player's position band | `sum(goals) / sum(minutes) * 90` for all players in same position band | Global league median |

**Position Banding (25 raw positions → 5 bands):**
| Band | Raw Positions |
|------|---------------|
| GK | Goalkeeper |
| DEF | Left/Right/Center Back, Left/Right Back, Left/Right Wing Back |
| MID | Left/Right/Center Defensive Midfield, Left/Right/Center Midfield, Left/Right Attacking Midfield, Center Attacking Midfield |
| WING | Left/Right Wing |
| FWD | Left/Right/Center Forward, Left/Right Center Forward |
| UNKNOWN | None, unrecognized positions |

### 2.4 Fixture Context Features

| Feature | Description | Computation |
|---------|-------------|-------------|
| `is_home` | 1.0 if player's team is home, 0.0 if away | From target fixture: `home_team_id == player_team_id` |
| `days_rest` | Days since player's last appearance | `(target_match_date - last_history_match_date).days`, min 0 |

---

## 3. Configuration

```python
@dataclass(frozen=True)
class FeatureConfig:
    min_history: int = 3          # Minimum matches for player-specific rates
    recent_window: int = 3        # Recent form window (matches)
    position_grouping: Literal["raw", "banded"] = "banded"
```

---

## 4. Leakage Prevention

All features are computed **exclusively from data with `match_week < target_matchweek`**:

| Feature | Leakage Safeguard |
|---------|-------------------|
| Player rates | History filtered to `match_week < cutoff` |
| Recent form | Only matches before cutoff, sorted descending by match_week |
| Team stats | Fixtures filtered to `match_week < cutoff` |
| Opponent stats | Fixtures filtered to `match_week < cutoff` |
| Position stats | Only historical player observations before cutoff |
| Days rest | Last appearance from history before cutoff |
| Home/away | From target fixture (known pre-match) — no outcome data used |

**Verified by leakage tests:**
- `test_future_data_injection_no_leakage`: Appending fake high-scoring matchweek 3 data does not change cutoff 2 features
- `test_future_fixture_injection_no_leakage`: Adding future fixtures does not change team/opponent stats at earlier cutoff

---

## 5. Test Results

### 5.1 New Tests (20 tests in `tests/test_player_features.py`)

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestPositionBands` | 2 | Position → band mapping correctness |
| `TestPrepareHistory` | 3 | History filtering, empty cutoff, merge correctness |
| `TestPlayerFeatures` | 10 | Core feature computation, fallbacks, sparse history, zero minutes, missing position, team/opponent/position temporal isolation, rest days, no target-match leakage |
| `TestLeakage` | 2 | Future data injection, future fixture injection |
| `TestComputeFeaturesForCutoff` | 2 | Batch computation, empty cutoff |

**All 20 tests pass.**

### 5.2 Full Suite

```
134 passed, 3 warnings in 3.14s
```

Warnings are pre-existing (empty array handling in MAE test, Poisson deviance negative prediction warning).

---

## 6. Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Single season (2015/16)** | No cross-season validation possible; overfitting risk | Explicitly documented; no performance claims |
| **Position from modal event** | May not match lineup role; affects position band assignment | Documented; position banding reduces granularity issues |
| **No injury/suspension data** | Cannot model availability changes | Days rest proxy only; documented |
| **Team xG from player_match** | Approximate (sum of player xG ≠ team xG exactly) | Marked as "unavailable" fallback when empty; documented |
| **Position bands group sparse positions** | Loss of granularity (e.g., CF vs WING) | Banding reduces sparse estimates; 5 bands from 25 positions |
| **No opponent strength ratings** | Only goals conceded used | Simple proxy; could add xGA or Elo if data available |
| **No player attributes (age, height, etc.)** | Limited player profiling | StatsBomb Open Data doesn't provide these |
| **Stoppage time in minutes** | Per-90 denominators slightly inflated (91-101 max) | Documented; consistent across all features |

---

## 7. Usage Example

```python
from src.features.player_features import FeatureConfig, compute_features_for_cutoff
import pandas as pd

player_match = pd.read_parquet("data/processed/statsbomb/player_match.parquet")
fixtures = pd.read_parquet("data/processed/statsbomb/fixtures.parquet")

config = FeatureConfig(min_history=3, recent_window=3)

# Compute features for matchweek 20
features_df = compute_features_for_cutoff(player_match, fixtures, 20, config)

# features_df has one row per player in matchweek 20 with columns:
# player_goals_per_90, player_assists_per_90, player_recent_goals_mean_3,
# player_recent_assists_mean_3, player_minutes_per_match, player_starts_rate,
# team_goals_per_match, team_xg_per_match, opponent_goals_conceded_per_match,
# position_goals_per_90_league, is_home, days_rest, fallback_info, etc.
```

---

## 8. Next Steps (Stage 3.2)

With features available, Stage 3.2 will implement:
1. Poisson/Negative Binomial GLM with L2 regularization for goals/assists
2. Model evaluation integrated into existing rolling-origin framework
3. Comparison against Phase 2 baselines (global mean, player mean, recent form, minutes-adjusted)
4. Extended reporting with calibration plots and feature coefficients

**Do not proceed to Stage 3.2 until Stage 3.1 is reviewed and approved.**