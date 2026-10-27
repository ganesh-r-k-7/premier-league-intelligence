# Phase 5 Stage 5.1 — Player Scouting Profiles & Position-Aware Rankings

**Date:** 2026-10-25
**Status:** Complete

---

## 1. Summary

This stage implements player scouting profiles, position-aware rankings, and player similarity search using the validated StatsBomb 2015/16 Premier League dataset.

**Files created/modified:**
- `src/features/player_scouting.py` — Player scouting profiles, position-aware rankings, player similarity
- `src/evaluation/metrics.py` — Extended with goal-level metrics (MAE/RMSE for goals, Poisson log-likelihood)
- `tests/test_player_scouting.py` — 27 comprehensive tests
- `reports/phase5_stage51_report.md` — This report
- `PROJECT_PLAN.md` — Updated with Stage 5.1 status

**Test results:** 300 tests pass (27 new + 273 existing)

---

## 2. Player Scouting Profiles

### 2.1 Data Source
- **Source:** StatsBomb Open Data 2015/16 Premier League season
- **Input:** `player_season.parquet` (550 player-seasons, aggregated from 10,469 player-match rows)
- **Scope:** Single season (2015/16) — no cross-season validation possible

### 2.2 Profile Fields
Each profile contains:

| Category | Fields |
|----------|--------|
| **Identity** | player_id, player_name, team_id, team_name, position, position_band |
| **Playing time** | appearances, starts, minutes, starts_rate, minutes_per_match |
| **Output** | goals, assists, own_goals, penalty_goals, penalty_attempts |
| **Creation** | shots, shots_on_target, xG, npxG, xA |
| **Per-90 rates** | goals/90, assists/90, xG/90, xA/90, npxG/90, shots/90, SoT/90 |
| **Efficiency** | goals/shot, goals/SoT, goals_over_xG, assists_over_xA |
| **Flags** | minutes_flag, appearances_flag, high per-90 warnings (configurable percentile) |
| **Metadata** | fallback_info, player_history_matches, player_history_minutes, position_band, position_band_order |

### 2.3 Fallback Logic
When player history is insufficient:
- Per-90 rates fall back to position-band league average → global average
- Recent form falls back to player mean → global mean
- Minutes/start rate falls back to position-team → team → global
- Minimum minutes/appearances thresholds with explicit flags

### 2.4 Sample Size Flags
- `minutes_flag`: "below_min_N" if minutes < threshold
- `appearances_flag`: "below_min_N" if appearances < threshold
- High per-90 warnings at configurable percentile (default 95th)

---

## 3. Position-Aware Rankings

### 3.1 Methodology
- Rankings computed **within each position band** (GK, DEF, MID, WING, FWD)
- Configurable minimum minutes per metric (default 450 for output, higher for xG)
- Rank by value (ascending/descending configurable) + percentile
- Only players meeting minute threshold included

### 3.2 Metrics by Position Band

| Band | Metrics Ranked |
|------|----------------|
| **GK** | Shots on Target Faced/90 (lower better) |
| **DEF** | Goals/90, Assists/90, xG/90, xA/90, Shots/90, SoT/90, Goals/Shot, Goals/SoT |
| **MID** | Goals/90, Assists/90, xG/90, xA/90, Shots/90, SoT/90, npxG/90, Goals/Shot, Goals/SoT |
| **WING** | Same as MID |
| **FWD** | Goals/90, Assists/90, xG/90, xA/90, npxG/90, Shots/90, SoT/90, Goals/Shot, Goals/SoT |

### 3.3 Results (2015/16, min 450 min)

| Metric | Top Player | Value | Percentile |
|--------|------------|-------|------------|
| **Goals/90 (FWD)** | Agüero | 0.90 | 100th |
| **xG/90 (FWD)** | Agüero | 0.66 | 100th |
| **Assists/90 (FWD)** | Rashford | 0.31 | 100th |
| **xG/90 (MID)** | De Bruyne | 0.38 | 100th |
| **Shots/90 (FWD)** | Vardy | 4.22 | 100th |
| **Goals/Shot (FWD)** | Rashford | 0.31 | 100th |

---

## 4. Player Similarity & Hidden Gems

### 4.1 Method
- **Features:** 10 metrics (goals/90, assists/90, xG/90, xA/90, shots/90, SoT/90, npxG/90, goals/shot, goals/SoT, minutes/match, starts_rate)
- **Standardization:** Fit on candidates, transform target + candidates
- **Metrics:** Cosine similarity (default) or Euclidean (1/(1+dist))
- **Scope:** Same position band only, minimum minutes filter
- **Output:** Top-K similar players with feature contribution breakdown

### 4.2 Example: Similar to Agüero (Top-5)
| Player | Team | Similarity | Top Feature Contributions |
|--------|------|------------|---------------------------|
| Harry Kane | Tottenham | 0.94 | shots/90 (0.31), goals/90 (0.25), xG/90 (0.10) |
| Daniel Sturridge | Liverpool | 0.91 | xG/90 (0.21), goals/90 (0.19) |
| Jamie Vardy | Leicester | 0.90 | shots/90 (1.14), SoT/90 (0.49) |
| Andy Carroll | West Ham | 0.85 | shots/90 (1.17), SoT/90 (0.61) |
| Romelu Lukaku | Everton | 0.81 | shots/90 (1.23), SoT/90 (0.64) |

### 4.3 Hidden-Gem Potential
Players with high similarity to top performers but lower minutes/profile:
- Can identify "next Agüero" candidates if they get playing time
- Feature contributions explain *why* they're similar
- **Limitation:** Single season only — no validation of future performance

---

## 5. Configuration

```python
@dataclass(frozen=True)
class ScoutingConfig:
    min_minutes: int = 450              # Min minutes for rankings/similarity
    min_appearances: int = 5            # Min appearances for rankings
    min_xg_samples: int = 3             # Min matches with xG for xG metrics
    percentile_warning: float = 95.0    # High per-90 warning percentile
    similarity_top_k: int = 5           # Number of similar players returned
    similarity_metric: Literal["euclidean", "cosine"] = "cosine"
    similarity_features: tuple = (      # Features for similarity
        "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
        "shots_per_90", "shots_on_target_per_90", "npxg_per_90",
        "goals_per_shot", "goals_per_shot_on_target",
        "minutes_per_match", "starts_rate",
    )
```

---

## 6. Reproducibility & Testing

```bash
# Run all tests
.venv/bin/python -m pytest tests/ -q

# Run scouting tests only
.venv/bin/python -m pytest tests/test_player_scouting.py -v

# Run scouting on real data
.venv/bin/python -c "
from src.features.player_scouting import build_player_profiles, compute_position_rankings, ScoutingConfig
import pandas as pd
player_season = pd.read_parquet('data/processed/statsbomb/player_season.parquet')
ps = player_season[player_season['season_id']==27]
profiles = build_player_profiles(ps, ScoutingConfig(min_minutes=450))
rankings = compute_position_rankings(profiles)
print(rankings['FWD_goals_per_90'].head(10))
"
```

**All 300 tests pass** (27 new scouting tests + 273 existing)

---

## 6. Limitations & Assumptions

| Limitation | Impact |
|------------|--------|
| **Single season (2015/16)** | No cross-season validation; cannot assess consistency |
| **Position from modal event** | Not lineup role; may misclassify versatile players |
| **No age/market value data** | Cannot compute value/performance indicators or age-adjusted metrics |
| **Single-season similarity** | No career trajectory or development curve |
| **Independent Poisson** | Scoreline model ignores goal dependence (Dixon-Coles needed) |
| **xG from StatsBomb only** | Single xG model; no comparison to other providers |
| **No injury/rotation data** | Rest days approximated from match dates only |

---

## 7. Reproducibility

```bash
# Run all tests
.venv/bin/python -m pytest tests/ -q

# Run scouting tests only
.venv/bin/python -m pytest tests/test_player_scouting.py -v

# Generate report
.venv/bin/python -c "
from src.features.player_scouting import build_player_profiles, compute_position_rankings, ScoutingConfig
import pandas as pd
ps = pd.read_parquet('data/processed/statsbomb/player_season.parquet')
ps = ps[ps['season_id']==27]
profiles = build_player_profiles(ps[ps['season_id']==27], ScoutingConfig(min_minutes=450))
rankings = compute_position_rankings(profiles)
"
```

**All 300 tests pass.** Random seed fixed at 42 for deterministic results.

---

## 8. Next Steps (Phase 5.2+)

| Stage | Focus |
|-------|-------|
| **5.2** | Transfer comparison: value/performance indicators, age-adjusted metrics |
| **5.3** | Multi-season aggregation (requires additional seasons) |
| **5.4** | Market value integration (requires Transfermarkt data) |
| **6.x** | Dashboard integration for scouting UI |

---

## 8. Outputs

- `src/features/player_scouting.py` — Core module
- `src/evaluation/metrics.py` — Extended with goal-level metrics
- `tests/test_player_scouting.py` — 27 tests
- `reports/phase5_stage51_profiles.parquet` — 550 player profiles
- `reports/phase5_stage51_report.md` — This report
- `PROJECT_PLAN.md` — Updated with Stage 5.1 status