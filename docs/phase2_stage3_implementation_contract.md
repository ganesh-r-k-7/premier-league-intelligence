# Phase 2 Stage 3 — Final Implementation Contract: Goals & Assists Forecasting Baselines

**Status:** Read-only design for approval. No implementation yet.

---

## 1. Prediction Task

**Primary task:** For each player listed in a target match's lineup (starters + used substitutes + unused substitutes), predict the player's **goals** and **assists** in that target match using only completed matches played before the target match's matchweek.

**Unit of prediction:** Expected match-level counts (not per-90 rates). Rates may be used as intermediate calculations but must be converted to expected counts using expected minutes (see §2.4).

**Evaluation method:** Chronological rolling-origin evaluation within the 2015/16 Premier League season (38 matchweeks).

- At each prediction point (cutoff matchweek `cw`), features are computed from matches with `match_week < cw`.
- Targets are the actual goals/assists recorded in matches with `match_week == cw`.
- Models/baselines are refit/refreshed before every target matchweek.

**Separate experiment (optional, out of scope for primary baseline evaluation):**
Remaining-season cumulative forecasts (e.g., "predict total goals from matchweek `cw+1` to end of season"). If implemented later, must use separate metrics, separate code paths, and clearly distinct interpretation. Not part of this contract.

---

## 2. Baseline Contracts

All baselines produce **expected goals** and **expected assists** for a target match as non-negative floats. Primary outputs are **match-level counts**.

### 2.1 Global Historical Mean

| Element | Specification |
|---------|---------------|
| **Inputs** | `player_match` rows with `match_week < cw` (all players, all positions). |
| **Computation** | `global_goals_mean = sum(goals) / N`, `global_assists_mean = sum(assists) / N` where `N` = number of player-match rows in history. |
| **Output per player** | `(global_goals_mean, global_assists_mean)` — same for every player. |
| **Fallback** | If history is empty (e.g., `cw=1`), return `(0.0, 0.0)`. |
| **Units** | Expected goals/assists per match. |

### 2.2 Player Historical Mean (with Minimum History)

| Element | Specification |
|---------|---------------|
| **Inputs** | `player_match` rows for the specific `player_id` with `match_week < cw`. |
| **Parameter** | `min_history` (default: 3). Minimum number of historical player-match rows required. |
| **Computation** | If `len(history) >= min_history`: `player_goals_mean = sum(goals) / len(history)`, `player_assists_mean = sum(assists) / len(history)`. Else: **fallback to Global Historical Mean** (§2.1). |
| **Output per player** | `(player_goals_mean, player_assists_mean)` or fallback. |
| **Units** | Expected goals/assists per match. |

### 2.3 Recent-Form Baseline

| Element | Specification |
|---------|---------------|
| **Inputs** | `player_match` rows for the specific `player_id` with `match_week < cw`, sorted by `match_week` descending. |
| **Parameter** | `window` (default: 5). Number of most recent matches to consider. |
| **Computation** | Take up to `window` most recent matches. If `len(recent) >= 1`: `recent_goals_mean = sum(goals) / len(recent)`, `recent_assists_mean = sum(assists) / len(recent)`. Else: **fallback to Player Historical Mean** (§2.2) → if that also falls back, **Global Historical Mean** (§2.1). |
| **Output per player** | `(recent_goals_mean, recent_assists_mean)` or chained fallback. |
| **Units** | Expected goals/assists per match. |

### 2.4 Minutes-Adjusted Baseline

| Element | Specification |
|---------|---------------|
| **Concept** | Expected counts = per-90 rate × expected_minutes / 90. Never use actual target-match minutes (leakage). |
| **Inputs** | `player_match` rows for the specific `player_id` with `match_week < cw`. |
| **Parameters** | `min_history` (default: 3) for rate estimation; `expected_minutes_method` (default: `"position_team_median"`). |
| **Rate estimation** | If `len(history) >= min_history` and `sum(minutes) > 0`: `goals_per_90 = sum(goals) / sum(minutes) * 90`, `assists_per_90 = sum(assists) / sum(minutes) * 90`. Else: **fallback to Player Historical Mean** (§2.2) → **Global Historical Mean** (§2.1). |
| **Expected minutes estimation** (pre-match, no leakage): |
| | **Method `"position_team_median"` (default):** Median minutes of all players at the same `position` on the same `team_id` in historical matches (`match_week < cw`). If no such players, fall back to team median across all positions. If still unavailable, fall back to global median minutes for that position. If unavailable, fall back to 90. |
| | **Method `"player_median"`:** Player's own historical median minutes (requires `min_history` matches with `minutes > 0`). Fallback chain: Player Historical Mean → Global Historical Mean. |
| | **Method `"fixed_90"`:** Always 90 minutes (simple, no leakage risk). |
| **Output per player** | `expected_goals = goals_per_90 * expected_minutes / 90`, `expected_assists = assists_per_90 * expected_minutes / 90`. |
| **Fallback** | If no defensible expected-minutes estimate (all fallbacks exhausted), use `fixed_90` and log a warning. Document the fallback path per player per matchweek. |
| **Units** | Expected goals/assists per match. |

---

## 3. Evaluation Populations

Two distinct populations are defined and reported **separately**. Every baseline is evaluated on identical rows within each population.

### Population A: All Lineup Players (Primary for "Selection" Use Case)
- **Definition:** Every player appearing in the `lineups` payload for a target match (starters, used substitutes, unused substitutes).
- **In `player_match` terms:** All rows with `match_id` in target matchweek, regardless of `minutes`.
- **Includes:** The 0-minute substitute (e.g., Jordi Gómez, Sunderland, matchweek 38).
- **Size per matchweek:** ~220–230 rows (11 starters + ~3 used subs + ~7 unused subs per team × 2 teams × 10 matches).
- **Rationale:** Matches real-world "who should I start/select?" decisions where unused subs are part of the squad.

### Population B: Players Who Played (Primary for "Performance" Use Case)
- **Definition:** Players with `minutes > 0` in the target match.
- **In `player_match` terms:** Subset of Population A with `minutes > 0`.
- **Excludes:** Unused substitutes (0 minutes).
- **Size per matchweek:** ~220 rows (11 starters + ~3 used subs per team × 2 teams × 10 matches).
- **Rationale:** Evaluates prediction quality for players who actually took the pitch.

**Reporting rule:** Both populations reported in all tables. The "primary" population for the portfolio is **Population B** (players who played), as it aligns with standard football analytics evaluation. Population A is reported for transparency and squad-selection use cases.

**Consistency rule:** Within each population and matchweek, every baseline receives exactly the same evaluation rows. No baseline-specific filtering.

---

## 4. Cutoff and Leakage Policy

### 4.1 Chronology Source
- **Primary:** `match_week` from `fixtures.parquet` (integer 1–38).
- **Secondary (tie-break within week):** `match_date` then `kick_off` — but **all matches in the same matchweek are treated as simultaneous** for cutoff purposes. No within-week ordering is used for feature computation.

### 4.2 Cutoff Definition
- For target matchweek `cw` (1 ≤ `cw` ≤ 38):
  - **Training/feature history:** All `player_match` rows where the match's `match_week < cw`.
  - **Test/target:** All `player_match` rows where the match's `match_week == cw`.
- **Matchweek 1 (`cw=1`):** No historical data. All baselines fall back to `(0.0, 0.0)` or global means computed from empty history (which yield 0).

### 4.3 Leakage Prevention Rules
1. **No target-match data in features:** Features for `cw` never use `match_week >= cw`.
2. **No actual minutes in minutes-adjusted baseline:** Expected minutes estimated from history only (§2.4).
3. **No future team/position changes:** Player's `team_id` and `position` for a target match are taken from the target match's lineup (known pre-match). Historical rates are computed using the player's historical team/position at the time of each historical match. No "future team" leakage.
4. **Completed matches only:** The dataset contains only completed fixtures (`match_status == "available"`). No partial/unplayed matches exist. This is documented as a dataset property.

### 4.4 Leakage Test (Mandatory)
A test must be implemented that:
1. Takes the full `player_match` dataset.
2. Artificially appends/modifies rows for a future matchweek (e.g., adds fake high-scoring matches for a player in matchweek 20).
3. Runs the prediction pipeline for cutoff `cw=15`.
4. Asserts that predictions for `cw=15` are **identical** (bitwise or within floating-point tolerance) with and without the injected future data.
5. This test lives in `tests/test_leakage.py` and is part of the CI suite.

---

## 5. Evaluation and Metrics

### 5.1 Cutoff Points and Horizons
| Cutoff matchweek (`cw`) | Target horizon | Notes |
|-------------------------|----------------|-------|
| 1 | Matchweek 1 | No history; all baselines = 0 or global fallback |
| 2 | Matchweek 2 | 10 matches of history |
| ... | ... | ... |
| 37 | Matchweek 37 | 36 matchweeks of history |
| 38 | Matchweek 38 | 37 matchweeks of history |

**Total evaluation points:** 38 matchweeks (rolling origin, refit every week).

**Refresh policy:** Baselines are recomputed from scratch at each cutoff using only `match_week < cw`. No incremental state carried forward (stateless recomputation).

### 5.2 Metrics (Computed per Population, per Matchweek, and in Aggregate)

| Metric | Formula | Notes |
|--------|---------|-------|
| **MAE** | `mean(|y - ŷ|)` | Goals and assists separately. |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Goals and assists separately. |
| **Poisson Deviance** | `2 * mean(ŷ - y + y * log(y / ŷ))` with convention `0 * log(0) = 0` | **Only computed when:** `y` are non-negative integers (true) AND `ŷ >= 0` for all rows. If any `ŷ < 0` (should not happen with these baselines), Poisson deviance is **not reported** for that baseline/population/matchweek; a warning is logged. Zero predictions (`ŷ = 0`) handled via `y * log(y/ŷ) = 0` when `y = 0`. |
| **Sample size** | `N` | Reported per matchweek and in aggregate. |

### 5.3 Reporting Requirements
- Identical evaluation rows across all baselines within each population.
- Per-matchweek tables (38 rows × 4 baselines × 2 populations × 2 targets = 608 metric rows).
- Aggregate tables (weighted by sample size).
- No performance claims made in advance (e.g., "baseline X should beat Y"). Observed differences reported neutrally.
- All metrics saved to `reports/phase2_stage3_metrics.parquet` and summarized in `reports/phase2_stage3_report.md`.

---

## 6. Deliverable: Function Signatures, Algorithms, and Tests

### 6.1 Core Function Signatures

```python
# src/features/baselines.py

from dataclasses import dataclass
from typing import Literal
import pandas as pd

@dataclass(frozen=True)
class BaselineConfig:
    min_history: int = 3
    recent_window: int = 5
    expected_minutes_method: Literal["position_team_median", "player_median", "fixed_90"] = "position_team_median"

def compute_global_mean(history: pd.DataFrame) -> tuple[float, float]:
    """Return (goals_mean, assists_mean) from history DataFrame."""
    ...

def compute_player_mean(history: pd.DataFrame, player_id: int, min_history: int) -> tuple[float, float]:
    """Return (goals_mean, assists_mean) for player; fallback to global if insufficient history."""
    ...

def compute_recent_form(history: pd.DataFrame, player_id: int, window: int, global_fallback: tuple[float, float]) -> tuple[float, float]:
    """Return (goals_mean, assists_mean) from recent window; chained fallback."""
    ...

def estimate_expected_minutes(
    history: pd.DataFrame,
    fixtures_history: pd.DataFrame,
    target_match: pd.Series,  # row from fixtures with match_id, team_id, etc.
    player_id: int,
    method: Literal["position_team_median", "player_median", "fixed_90"]
) -> float:
    """Estimate expected minutes for player in target_match using only history.
    Never uses target_match minutes. Returns float >= 0.
    """
    ...

def compute_minutes_adjusted(
    history: pd.DataFrame,
    fixtures_history: pd.DataFrame,
    target_match: pd.Series,
    player_id: int,
    config: BaselineConfig
) -> tuple[float, float]:
    """Return (expected_goals, expected_assists) using per-90 rate * expected_minutes / 90.
    Chained fallback: player rate -> global rate; expected_minutes -> fixed_90.
    """
    ...

def evaluate_baselines_at_cutoff(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    cutoff_week: int,
    config: BaselineConfig,
    population: Literal["all_lineup", "played_only"]
) -> pd.DataFrame:
    """Run all 4 baselines for a single cutoff week.
    Returns DataFrame with columns:
    [match_id, player_id, player_name, team_id, position, minutes_actual,
     goals_actual, assists_actual,
     global_goals_pred, global_assists_pred,
     player_mean_goals_pred, player_mean_assists_pred,
     recent_form_goals_pred, recent_form_assists_pred,
     minutes_adj_goals_pred, minutes_adj_assists_pred,
     population, cutoff_week]
    """
    ...

def run_rolling_evaluation(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    config: BaselineConfig,
    cutoffs: list[int] = list(range(1, 39))
) -> dict[str, pd.DataFrame]:
    """Run full rolling-origin evaluation for both populations.
    Returns dict with keys:
    - 'predictions_all_lineup': concatenated predictions for Population A
    - 'predictions_played_only': concatenated predictions for Population B
    - 'metrics_all_lineup': per-cutoff + aggregate metrics for Population A
    - 'metrics_played_only': per-cutoff + aggregate metrics for Population B
    """
    ...
```

### 6.2 Chronological Evaluation Algorithm

```python
def run_rolling_evaluation(player_match, fixtures, config, cutoffs):
    # 1. Merge match_week into player_match from fixtures
    pm = player_match.merge(
        fixtures[["match_id", "match_week"]],
        on="match_id", how="left", validate="many:one"
    )
    assert pm["match_week"].notna().all(), "All player_match rows must have match_week"

    # 2. Prepare fixtures history lookup (team/position medians per matchweek)
    #    Pre-compute for efficiency: for each matchweek cw, compute medians from matches < cw

    all_predictions_a = []
    all_predictions_b = []

    for cw in cutoffs:
        history = pm[pm["match_week"] < cw].copy()
        target = pm[pm["match_week"] == cw].copy()
        fixtures_history = fixtures[fixtures["match_week"] < cw].copy()
        target_fixtures = fixtures[fixtures["match_week"] == cw].copy()

        if target.empty:
            continue

        # Population A: all lineup players (target rows as-is)
        pred_a = evaluate_baselines_at_cutoff(
            history, fixtures_history, cw, config, "all_lineup"
        )
        all_predictions_a.append(pred_a)

        # Population B: filter to minutes > 0
        pred_b = pred_a[pred_a["minutes_actual"] > 0].copy()
        pred_b["population"] = "played_only"
        all_predictions_b.append(pred_b)

    # 3. Concatenate and compute metrics
    ...
```

### 6.3 Fallback Rules Summary (Decision Tree)

```
Global Mean:
  history empty -> (0.0, 0.0)

Player Mean:
  len(player_history) >= min_history -> player mean
  else -> Global Mean

Recent Form:
  len(recent_matches) >= 1 -> recent mean
  else -> Player Mean (which may fallback to Global)

Minutes-Adjusted:
  Rate estimation:
    len(history) >= min_history AND sum(minutes) > 0 -> player per-90 rate
    else -> Player Mean rate (fallback chain)

  Expected minutes:
    method="position_team_median":
      position+team median (history) -> team median -> position median -> 90
    method="player_median":
      player median (minutes>0, min_history) -> 90
    method="fixed_90": 90

  Final:
    expected_goals = rate_goals_per_90 * expected_minutes / 90
    (same for assists)
```

### 6.4 Required Tests (Read-Only Design — Test Specifications)

| Test ID | Description | Location |
|---------|-------------|----------|
| `test_global_mean_empty_history` | `cw=1` returns `(0.0, 0.0)` | `tests/test_baselines.py` |
| `test_player_mean_fallback` | Player with 2 matches (< min_history=3) falls back to global mean | `tests/test_baselines.py` |
| `test_recent_form_chained_fallback` | Player with 0 recent matches falls back through player mean to global | `tests/test_baselines.py` |
| `test_minutes_adjusted_rate_fallback` | Player with 0 historical minutes falls back to player mean rate | `tests/test_baselines.py` |
| `test_minutes_adjusted_expected_minutes_fallback` | `position_team_median` falls back through team -> position -> 90 | `tests/test_baselines.py` |
| `test_no_leakage_future_data` | Injecting future matchweek data does not change predictions at earlier cutoff | `tests/test_leakage.py` |
| `test_identical_rows_across_baselines` | Within each population/cutoff, all baselines evaluated on same rows | `tests/test_evaluation.py` |
| `test_poisson_deviance_zero_predictions` | Poisson deviance handles `ŷ=0` correctly (0*log(0)=0 convention) | `tests/test_metrics.py` |
| `test_poisson_deviance_negative_prediction_skip` | If any `ŷ < 0`, Poisson deviance not computed, warning logged | `tests/test_metrics.py` |
| `test_population_a_includes_unused_subs` | Population A includes 0-minute players; Population B excludes them | `tests/test_evaluation.py` |
| `test_cutoff_matchweek_boundary` | `match_week < cw` strictly; same-week matches never in history | `tests/test_evaluation.py` |
| `test_matchweek_1_no_history` | All baselines produce valid outputs for `cw=1` | `tests/test_evaluation.py` |

### 6.5 Output Artifacts

| File | Description |
|------|-------------|
| `src/features/baselines.py` | Baseline implementations (pure functions) |
| `src/evaluation/rolling.py` | Rolling-origin evaluation orchestration |
| `src/evaluation/metrics.py` | MAE, RMSE, Poisson deviance with zero/negative handling |
| `tests/test_baselines.py` | Unit tests for each baseline |
| `tests/test_leakage.py` | Leakage test (mandatory) |
| `tests/test_metrics.py` | Metric computation tests |
| `tests/test_evaluation.py` | End-to-end evaluation tests |
| `reports/phase2_stage3_metrics.parquet` | Raw per-row predictions + metrics |
| `reports/phase2_stage3_report.md` | Human-readable summary with tables |

---

## 7. Remaining Limitations (Documented, Not Fixed)

1. **Single season only (2015/16):** No cross-season validation possible. All results are in-sample for this season only. Generalizability to other seasons is unknown.

2. **Matchweek = simultaneous assumption:** Matches within the same matchweek are treated as concurrent. In reality, some are played Friday/Monday. This means Saturday matches in week `cw` could theoretically use Friday results as "history," but we conservatively exclude all same-week matches. This is a deliberate bias toward safety.

3. **No opponent strength / team strength features:** Baselines are purely historical averages. Context (opponent quality, home/away, team form) is not modeled. This is by design for baselines.

4. **Position from modal event, not lineup role:** Player position in `player_match` is the most frequent event position, which may differ from the lineup's declared formation role. This affects `position_team_median` expected-minutes estimation.

5. **Stoppage time in minutes:** Match-end minutes include stoppage time (91–101), slightly inflating per-90 denominators. Documented, not corrected.

6. **Unused substitutes in Population A:** 0-minute players receive predictions but have actuals of 0. This inflates MAE/RMSE for Population A vs B. Both reported.

7. **No injury/suspension/rotation signal:** Expected minutes use historical medians only. Real pre-match information (press conferences, injury reports) is not available in StatsBomb Open Data.

8. **Penalty goals included in goals:** `goals` includes penalties. Baselines predict total goals. If non-penalty goals are desired, a separate target (`goals - penalty_goals`) would be needed.

9. **One 0-minute substitute in data:** Jordi Gómez (Sunderland) appears in Population A with 0 minutes. Handled correctly.

---

## 8. Approval Checklist

Before implementation begins, confirm:

- [ ] Prediction task definition (match-level counts, rolling origin)
- [ ] Four baseline contracts (inputs, outputs, fallbacks, units)
- [ ] Two evaluation populations (all lineup vs played-only)
- [ ] Cutoff policy (matchweek, simultaneous within week, leakage test)
- [ ] Metrics (MAE, RMSE, Poisson deviance with zero-handling)
- [ ] Function signatures and algorithm structure
- [ ] Test specifications
- [ ] Limitations acknowledged

**Next step upon approval:** Implement `src/features/baselines.py`, `src/evaluation/rolling.py`, `src/evaluation/metrics.py`, and tests. Then run full evaluation and produce reports.

---

*End of Implementation Contract. Awaiting approval.*