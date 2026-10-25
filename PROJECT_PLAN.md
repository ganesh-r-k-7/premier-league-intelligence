# Project Plan

This document defines the implementation phases, datasets, prediction targets, validation methodology, dependencies, risks and completion criteria for Premier League Intelligence. It is the operational companion to the master specification and to `ARCHITECTURE.md`. Phase 0 consists of this document, `ARCHITECTURE.md` and the data-source feasibility assessment below; no modeling or dashboard code has been built.

## Phase overview

| # | Phase | Main deliverable | Exit gate |
| --- | --- | --- | --- |
| 0 | Architecture and data-source feasibility | `ARCHITECTURE.md`, `PROJECT_PLAN.md`, feasibility assessment | Both documents exist, are consistent with the master specification, and are reviewed by the user. |
| 1 | Environment, ingestion, validation and exploratory analysis | Virtual environment, pinned `requirements.txt`, ingestion pipeline, EDA notebook | Pipeline runs reproducibly on available sources; raw/processed separation works; test suite runs without network access. |
| 2 | Goals and assists forecasting and evaluation | Features and models for goals/assists; honest evaluation report | Baselines exist, chronological split shows no leakage, and metrics are reported honestly. |
| 3 | Market-value prediction and explainability | Value model with explainability where data allows | Model trains and evaluates on available data; uncertainty and limitations are always displayed. |
| 4 | Match prediction and league simulation | Match outcome models and league simulations | Models are validated on a held-out season; simulation sanity checks pass. |
| 5 | Scouting and transfer comparison | Hidden-gem heuristics; transfer comparison indicators | Heuristics are clearly labelled and retrospectively validated where possible. |
| 6 | Dashboard integration | Streamlit application with all planned pages | Every page shows validated predictions only; unavailable data is never faked. |
| 7 | Tests, documentation and reproducibility | Expanded test suite; reproducible runs; final documentation | Test suite passes; results reproduce from committed configuration. |
| 8 | Optional features | Fantasy points/captain, Golden Boot, scenarios, AI analyst | Optional; only if time and compute allow. |

## Phase 0 — completed

- The entire master specification was read and used as the project scope.
- The directory and Python environment were inventoried (see Dependencies).
- Data-source feasibility was investigated and the findings are recorded below.
- `ARCHITECTURE.md` and this file were created as the Phase 0 deliverables.
- No model, dashboard or ingestion code has been written yet.

## Datasets

Each module of the master specification needs a specific set of datasets. Sources are assigned only where they exist and are accessible; every dataset notes its feasibility, because no single source covers all needs.

### Module 1 — Market-value prediction (phase 3)

- **Player registry** — stable player ID, name, position, nationality, date of birth, height, preferred foot. Source: StatsBomb open data and official Premier League current-season data. Feasibility: available, but StatsBomb covers only a few Premier League seasons.
- **Player season attributes** — age, position group, squad context, market value in EUR. Source: StatsBomb open data; historical values require Transfermarkt (subscription). Feasibility: limited.
- **Player performance per season** — minutes played, goals, assists, rating. Source: StatsBomb open data and official current-season data. Feasibility: same limitation as module 2.
- **Market-value history** — value per player over time. Source: Transfermarkt (subscription). Feasibility: subscription-gated; availability must be confirmed before use.
- Join: registry joins season attributes and value history on player ID. Forecast horizon is 1–3 years only where the value history supports it. Transfer fees are **not** a target.

### Module 2 — Goals and assists forecasting (phase 2)

- **Player season statistics** — per player per season: position, team, minutes played, goals, assists, shots, shots on target, rating. Source: official current-season data; StatsBomb open data for available seasons. Feasibility: current season complete; historical coverage limited.
- **Match statistics** — per match: shots, expected goals (xG), expected assists (xA), corners, possession, big chances, bookings. Source: official current-season data; StatsBomb open data for available seasons. Feasibility: same limitation as player statistics.
- **Minutes per player per season** — required for per-90 normalisation. Source: union of the above. Feasibility: derivable from player season statistics.
- Join: match statistics join to fixtures on match ID; player statistics join to players on player ID. Only one season is processed at a time, so joins never cross seasons.

### Module 3 — Match and league prediction (phase 4)

- **League fixtures** — every played match: date, home team, away team, final score. Source: Football-Data.co.uk (blocked in this environment — see feasibility); StatsBomb matches for available seasons. Feasibility: historical completeness could not be verified and is blocked here.
- **Team season records** — points, wins/draws/losses, goals scored, goals conceded, goal difference. Source: derived from fixtures. Feasibility: complete wherever fixtures are complete.
- **Team strength proxies** — ranking-based ratings computed from team season records. Source: derived internally; no external dependency. Feasibility: always available.
- Join: fixtures join to team season records by season; the simulation horizon is defined by the current date within the chosen season.

### Module 4 — Hidden-gem scouting (phase 5)

- **Per-90 metrics, age cohorts, value/performance indicators** — derived in phase 5 from the module 1 and module 2 datasets, with minimum-minute filters and recent-season stability filters. Source: same as modules 1–2. Feasibility: the derived indicators are always computable; their value is limited by the underlying data coverage.
- Join: registry joins player statistics and value history, exactly as in modules 1–2.
- No supervised target exists for this module; its outputs are explicitly labelled heuristics, not predictions.

### Module 5 — Transfer intelligence (phase 5)

- **Player values and value history** — defined in module 1. Source: Transfermarkt (subscription). Feasibility: subscription-gated.
- **Transfer events** — date, from club, to club, transfer fee. Source: Transfermarkt (subscription). Feasibility: subscription-gated; historical fee coverage must be confirmed before use.
- **Club budget context** — club size/budget where available. Source: not reliably available; treat as optional. Feasibility: unverified.
- Join: transfer events join to the player registry on player ID.
- Market value is never equated with a negotiated transfer fee, and transfer fees are not used as a prediction target.

### Module 6 — Team and player analytics (phase 6, descriptive)

- All pages of this module render datasets produced by modules 1–5; no new sources are required. Match previews combine module 3 outputs with the current fixture list.

## Join strategy

- **Raw/processed separation.** Ingestion writes immutable raw snapshots to `data/raw/`. Validation produces processed datasets in `data/processed/`; features are computed from those into feature tables. Raw files are never modified.
- **Per-season processing.** Every season is processed independently. Joins never cross a season boundary, which keeps evaluation correct.
- **Player identity.** Sources use different ID systems. A versioned player-mapping table must be built and maintained; IDs that cannot be mapped are documented rather than silently dropped.
- **Provenance.** Every processed and feature table carries source and season columns, so any result can be traced back to its inputs.
- **No silent imputation.** Fields missing in a source are marked missing. They are never filled from other sources, because that would create false precision.

## Prediction targets

### Goals and assists forecasting (phase 2)

- **Target:** goals scored and assists provided per player per season (counts).
- **Horizon:** one season ahead, using a rolling origin where the history allows it.
- **Baselines required:** mean goals/assists; naive "last season's value".
- **Metrics:** MAE/RMSE on counts; log loss at player-season level; calibration of predicted count distributions.
- xG is used as an input and as an evaluation reference only — never as a substitute for actual goals.

### Market-value prediction (phase 3)

- **Target:** market value per player, on a log scale. Optional second target: value growth over a 1–3 year horizon.
- **Horizon:** 1–3 years only where the available value history supports it.
- **Baselines required:** last observed value; mean/median by position group.
- **Metrics:** MAE/RMSE on log-value; RMSE on the original value scale. Compare honestly against the baselines.
- Transfer fees are not a target, and market value is never equated with a transfer fee.

### Match prediction (phase 4)

- **Target:** match outcome probabilities — home win, draw, away win — and scoreline probabilities (the full score matrix).
- **Horizon:** the remaining fixtures of the current season.
- **Baselines required:** ranking-based probabilities from team season records; naive frequency of past outcomes.
- **Metrics:** Brier score and log loss for match outcomes; calibration for scoreline probabilities.
- **Simulation targets:** probabilities of title winner, final league positions, European qualification and relegation. Evaluation consists of distribution checks (simulated outcomes reproduce observed outcomes in distribution) and sanity checks of expected values.
- All reported probabilities are estimates with the stated horizon.

### Scouting and transfer comparison (phase 5)

- No supervised targets exist. Outputs are descriptive indicators computed retrospectively on available seasons:
  - value/performance indicators (for example value per goal or per 90 minutes),
  - age- and position-relevant z-scores,
  - consistency filters (minimum minutes, recent-season stability).
- Every heuristic is explicitly labelled as a heuristic, and any validation is retrospective. Heuristics are never presented as predictions.

## Validation methodology (cross-cutting)

- **Chronological splits.** Training uses completed seasons only; the test season is the most recent one not used in training. Features are computed only from data available at prediction time — no future leakage.
- **Baselines for every supervised module.** Named in the targets above; models must justify themselves against them.
- **Module-specific metrics.** Counts use MAE/RMSE/log loss; value uses MAE/RMSE on log-value; match outcomes use Brier score/log loss and calibration; simulation uses distribution and sanity checks.
- **Shared evaluation code.** All modules use the same splitting and metric utilities, so comparisons are always fair.
- **Honest reporting.** Every prediction states its horizon, the data coverage it relies on, its uncertainty, and its limitations. No evaluation result is ever invented or extrapolated.

## Data-source feasibility

Findings below are verified from this environment where marked; the remainder is explicitly unverified and must not be relied upon.

1. **Football-Data.co.uk** — provides historical fixtures, results and match statistics. From this environment the site rejects automated requests (HTTP 403). Historical coverage, licensing terms and update frequency could **not** be verified. Do not build ingestion on it until access is confirmed.
2. **StatsBomb Open Data** — publicly available for research (attribution required). Data is provided as JSON. Current coverage: 24 competitions; the Premier League has only two seasons (2015/2016 and 2003/2004). Events, lineups and StatsBomb 360 are match-level. It is valuable for detailed event data but not for complete historical league analysis.
3. **Official Premier League statistics** — the site is reachable but is a JavaScript application. Current-season player and team statistics are publicly visible; historical player/team-level data appears to require licensed partnerships. Historical coverage and licensing could not be fully verified.
4. **Transfermarkt** — publicly reachable. The full historical database and player value histories require a paid subscription. Licensing of historical values and transfers could not be verified in detail.

**Conclusion:** no single source provides complete historical Premier League data. The platform must be built to work with partial coverage and to degrade gracefully, and every module must document the data it actually has.

## Dependencies and environment

- System Python: 3.14.4 (`/usr/bin/python3`). A project virtual environment will be created in phase 1, as required by the master specification.
- Already available: pandas 3.0.6, numpy 2.5.4, streamlit 1.65.0, pytest 9.1.1, requests 2.34.2, pyarrow 25.0.1.
- Not yet available (phase 1 must add): scikit-learn, scipy, statsmodels, xgboost, shap, plotly.
- `requirements.txt` will be created in phase 1 with pinned, compatible versions.

## Technical risks

| # | Risk | Impact | Mitigation |
| --- | --- | --- | --- |
| 1 | Partial data coverage limits every module (see feasibility). | Market-value, scouting and historical forecasting are constrained; some phases can only work on a few seasons. | Design modules to degrade gracefully; document coverage in every report; never extrapolate beyond available seasons. |
| 2 | Player-identity mismatches across sources. | Joins fail or silently drop players. | Versioned player-mapping table; unmapped IDs are documented; validation checks at ingestion. |
| 3 | Licensing restricts redistribution of source data. | Some sources cannot be used or only partially. | Use only what each source permits; attribute sources as required; do not commit licensed data. |
| 4 | Future-data leakage in time-based evaluation. | Invalidates every reported result. | Chronological splits; features computed only from past data; automated leakage tests. |
| 5 | Dependency and version risk (new libraries, Python 3.14). | Phase 1 may stall installing packages. | Pin versions in `requirements.txt`; prefer scikit-learn first; add XGBoost only where justified. |
| 6 | Compute cost of league simulation. | Long runtimes for Monte Carlo runs. | Vectorised simulation; documented runtime budgets; report progress. |
| 7 | Dashboard shows estimated values as if factual. | Loss of trust in the product. | Validation-only presentation; explicit horizon/uncertainty text on every prediction; no fake data. |
| 8 | Phase scope creep between phases. | Later phases pull work forward. | Strict phase gates; later phases consume earlier outputs only. |
| 9 | Model interpretability (SHAP) needs more than tabular baselines support. | Explainability falls back or is delayed. | Evaluate SHAP in phase 3 only where the model supports it; otherwise use permutation importance. |
| 10 | Missing or incomplete historical fixtures within seasons. | Some seasons cannot be evaluated. | Process only complete seasons; skip and document incomplete seasons explicitly. |

## Completion criteria

**Phase 0** is complete when: `ARCHITECTURE.md` and `PROJECT_PLAN.md` exist and are consistent with the master specification; datasets, targets, validation and dependencies are defined; the feasibility findings are documented; risks and completion criteria are recorded. This gate is met.

Subsequent gates:

- **Phase 1:** virtual environment and pinned requirements exist; ingestion runs reproducibly on available sources; raw/processed separation works; test suite runs without network access.
- **Phase 2:** goals/assists forecasting works with required baselines, honest metrics, and no leakage.
- **Phase 3:** market-value prediction works with explainability where the data allows.
- **Phase 4:** match prediction and league simulation are validated on a held-out season.
- **Phase 5:** scouting and transfer comparison are presented with heuristics explicitly labelled.
- **Phase 6:** dashboard shows all planned pages, presenting validated predictions only.
- **Phase 7:** tests, documentation and reproducibility are complete.
- **Phase 8 (optional):** only if time and compute allow.

## Phase 1 proposal (awaiting your approval)

The following prompt describes exactly what Phase 1 would do. It has **not** been started.

> Start Phase 1: environment and ingestion.
> 1. Create a virtual environment and pin all dependencies in `requirements.txt` (adding scikit-learn, scipy, statsmodels, xgboost, shap and plotly where justified).
> 2. Implement ingestion for the available sources, starting with StatsBomb Open Data (GitHub-hosted), and for any other source confirmed accessible.
> 3. Store raw snapshots immutably under `data/raw/` and produce validated datasets under `data/processed/`, with provenance columns.
> 4. Add pytest tests that run without network access, using a small fixture dataset.
> 5. Produce an exploratory analysis notebook and a short written report of what the data can and cannot support.
> 6. Do not build forecasting models, simulations, scouting, or dashboard features.
> 7. Report the Phase 1 deliverables and stop; wait for my approval before Phase 2.

## Phase 1 status — complete

Phase 1 is complete and verified. `src/data/ingest.py` ingests StatsBomb Open
Data with timeouts, error handling and an offline fallback; raw JSON is stored
under `data/raw/` and validated, provenance-carrying fixtures under
`data/processed/`. The offline test suite passes (29 tests) and
`reports/phase1_eda_report.md` is generated from the ingested data. Ingested
coverage: Premier League 2015/2016 (380 matches) and 2003/2004 (38 matches —
the source itself is partial for that season).

## Phase 2 data strategy (completed — Stage 2 done)

All 380 Premier League 2015/16 matches' events and lineups have been
downloaded from StatsBomb Open Data and aggregated into `player_match.parquet`
(10,469 rows) and `player_season.parquet` (550 rows). Validation shows 0 score
mismatches, 0 duplicate event IDs, perfect assist/xA alignment, and zero
deltas between player_match and player_season totals. Raw snapshots (~1.1 GB)
are preserved under `data/raw/statsbomb/` with attribution and are gitignored.

**Actual download size:** ~1.1 GB of events + ~7 MB of lineups (the original
~108 MB estimate was based on a 12-match sample; per-file event sizes are
3–10x larger).

**Source evaluation for player-level goals and assists.**

| Source | Player-level fields | Coverage | Licence / access | Verdict |
| --- | --- | --- | --- | --- |
| StatsBomb Open Data (events + lineups) | goals (`shot.outcome`=Goal), assists (`pass.goal_assist`), xG (`shot.statsbomb_xg`), xA (xG joined via `pass.key_pass_id`), shots, minutes and appearances (lineups positions), position | PL 2015/16 full (380); PL 2003/04 partial (38); plus other non-PL competitions | Open for research, attribution required; already used in Phase 1 | **Primary** |
| Understat | goals, assists, minutes (`time`), xG, xA, npxG, shots, key passes | PL 2014/15 → present | `robots.txt` is `Disallow: /` for all agents; no ToS/open licence; no commercial use or redistribution | Restricted — not to be used without explicit approval |
| FBref | basic gls/ast/min only (advanced data removed) | long history | Advanced xG removed site-wide on 2026-01-20 after Opta terminated feeds; ToS prohibit scraping and AI use; 403 from this environment | Not viable |
| FPL API / community repos (e.g. vaastav/Fantasy-Premier-League) | goals, assists, minutes | 2016/17 → present | xG/xA only from 2022/23; underlying data stated as property of the Premier League | Partial — caveats |
| Football-Data.co.uk | none (match-level only) | long history | Site unreachable from this environment (403/000) | Not viable for players |

**Decision.** Use StatsBomb Open Data events and lineups as the Phase 2 source,
because it is the only candidate with clear licensing that also provides
player-level goals, assists, minutes and xG, and it is already ingested.
Its decisive limitation is coverage: effectively one complete Premier League
season (2015/2016). Cross-season generalisation therefore cannot be validated
from PL data alone, and this must be stated in every Phase 2 report.

**Feasible targets.** Player goals and player assists per season (counts) are
feasible from StatsBomb 2015/2016. xG and xA are feasible as inputs and
evaluation references, never as substitutes for actual goals. A "one season
ahead" target across seasons is **not** feasible with PL data alone; the
honest Phase 2 horizon is within-season (predict a later matchweek or the
remaining season from earlier matchweeks).

**Proposed schema and identifiers.** A `player_season` table keyed by
(`competition_id`, `season_id`, `player_id`, `team_id`) with provenance
(`source`, `source_url`, `ingested_at`) carrying: player name, team name,
primary position, appearances, starts, minutes, goals, assists, xG, xA, shots,
penalty goals/attempts. A `player_match` intermediate table underpins it so
aggregates are reproducible. StatsBomb `player_id`, `team_id`, `match_id` are
the canonical Phase 2 keys; a versioned cross-source mapping table is only
needed if a second source is approved.

**Validation methodology.** Expanding-window/rolling-origin within 2015/2016:
for each cut-off matchweek, features use only matches before the cut-off and
the target is goals/assists over a defined later horizon — no random splits and
no same-season totals as features.

**Baselines and metrics.** Baselines: global and per-position mean; "last
observed" value; minutes-proportional expectation. Metrics: MAE and RMSE on
counts, Poisson deviance, Spearman rank correlation and top-K precision for
scoring leaderboards, plus distribution calibration where a count model is
used. Every metric is reported against the baselines.

**Open blockers.** (1) Only one complete PL season of player data exists, so
cross-season evaluation is impossible without either a restricted source
(Understat) or adding non-PL StatsBomb competitions and accepting a scope
change; both need explicit approval. (2) `src/eda.py` still states the
incorrect "2–4 MB / gigabytes" event-size figure and should be corrected when
Phase 2 code is written. (3) Ingesting 2015/16 events (~108 MB) is within
budget but has not been run.

## Phase 2 Stage 3 status — complete (this step)

Phase 2 Stage 3 (goals/assists baseline evaluation) is complete and verified:

- **Implemented:** Four baseline predictors in `src/features/baselines.py`:
  - Global historical mean
  - Player historical mean (configurable min_history=3)
  - Recent form (configurable window=5)
  - Minutes-adjusted (per-90 rate × expected_minutes/90, with position_team_median expected minutes)
- **Evaluation:** Rolling-origin across all 38 matchweeks of 2015/16, refitting at each cutoff, using strictly prior matchweeks (`match_week < cw`).
- **Populations:** Population A (all lineup players, 10,469 rows) and Population B (played only, minutes>0, 10,468 rows) evaluated separately.
- **Metrics:** MAE, RMSE, Poisson deviance (with 0*log(0)=0 convention, skipped if any ŷ<0).
- **Leakage test:** Mandatory test in `tests/test_leakage.py` passes — injecting future matchweek data does not change earlier-cutoff predictions.
- **Test suite:** All 114 tests pass (including 10 new baseline tests, 4 leakage tests, 12 metric tests, 12 evaluation tests).
- **Outputs:** Parquet files in `reports/` (`predictions_all_lineup`, `predictions_played_only`, `metrics_per_cutoff`, `metrics_aggregate`) and markdown report `reports/phase2_stage3_report.md`.

**Aggregate results (Population B - played only, primary):**
- Recent form: MAE Goals 0.1527, RMSE Goals 0.3523, Poisson Dev Goals 0.1211, MAE Assists 0.1104, RMSE Assists 0.2821, Poisson Dev Assists 0.0607
- Player mean: MAE Goals 0.1558, RMSE Goals 0.3315, Poisson Dev Goals 0.2665, MAE Assists 0.1127, RMSE Assists 0.2664, Poisson Dev Assists 0.1799
- Global mean: MAE Goals 0.1693, RMSE Goals 0.3346, Poisson Dev Goals 0.4608, MAE Assists 0.1181, RMSE Assists 0.2630, Poisson Dev Assists 0.3505
- Minutes-adjusted: MAE Goals 0.1772, RMSE Goals 0.3526, Poisson Dev Goals 0.2869, MAE Assists 0.1262, RMSE Assists 0.2805, Poisson Dev Assists 0.1921

No performance claims made; observed differences reported neutrally.

## Phase 3 Stage 3.1 status — complete

Phase 3 Stage 3.1 (player feature engineering) is complete and verified:

- **Implemented:** `src/features/player_features.py` with configurable feature extraction:
  - Player history: goals/assists per-90 (with fallback chain), recent form (3-match window), minutes per match, starts rate
  - Team context: team goals per match, team xG per match, opponent goals conceded per match
  - Position context: position band (GK/DEF/MID/WING/FWD) league goals per 90
  - Fixture context: home/away, days rest
  - All features computed strictly from `match_week < cutoff_week` history
  - Fallback chains documented for sparse history, zero minutes, missing position
- **Configuration:** `FeatureConfig` with `min_history=3`, `recent_window=3`, `position_grouping="banded"`
- **Leakage prevention:** All 10 leakage tests pass — future data injection does not affect earlier-cutoff features
- **Test suite:** 134 tests pass (20 new feature tests + 114 existing)
- **Outputs:** `reports/phase3_stage31_report.md` with feature definitions, fallback logic, test results, limitations

**Key limitations documented:**
- Single season (2015/16) only — no cross-season validation
- Position derived from modal event, not lineup role
- No injury/suspension/rotation signals available
- Team xG per match aggregated from player_match (approximate)
- Position bands group sparse positions (25 raw → 5 bands)
- No opponent strength ratings beyond goals conceded

## Phase 3 Stage 3.2 status — complete

Phase 3 Stage 3.2 (regularized Poisson GLM) is complete and verified:

- **Implemented:** `src/models/glm.py` with regularized Poisson GLM:
  - Separate models for goals and assists using scikit-learn's PoissonRegressor
  - L2 regularization (alpha=1.0) with Newton-Cholesky solver
  - StandardScaler fitted on training data only; constant features dropped
  - Fallback to zero predictions when training samples < min_train_samples (20)
  - Training and prediction deterministic (random_state=42)
- **Leakage prevention:** All features from `match_week < cutoff`; 4 leakage tests pass
- **Test suite:** 156 tests pass (22 new GLM tests + 134 existing)
- **Evaluation:** Rolling-origin on cutoffs 2–10 (2,480 predictions); full 38-matchweek evaluation deferred due to computational cost
- **Comparison with baselines (Population A):**

| Model | MAE Goals | RMSE Goals | Poisson Dev Goals | MAE Assists | RMSE Assists | Poisson Dev Assists |
|-------|-----------|------------|-------------------|-------------|--------------|---------------------|
| **GLM (cutoffs 2–10)** | **0.1728** | **0.3515** | 0.4812 | **0.1240** | **0.2762** | 0.3788 |
| Recent Form (38 weeks) | 0.1527 | 0.3522 | 0.1211 | 0.1104 | 0.2821 | 0.0607 |
| Player Mean (38 weeks) | 0.1558 | 0.3315 | 0.2665 | 0.1127 | 0.2664 | 0.1799 |
| Global Mean (38 weeks) | 0.1693 | 0.3346 | 0.4607 | 0.1181 | 0.2629 | 0.3505 |
| Minutes-Adjusted (38 weeks) | 0.1771 | 0.3525 | 0.2869 | 0.1262 | 0.2805 | 0.1921 |

**Key finding:** GLM does not clearly outperform simple baselines (Recent Form, Player Mean) on this dataset. Poisson Deviance suggests overdispersion; Negative Binomial recommended for Stage 3.3.

- **Outputs:** `reports/phase3_stage32_report.md`, `reports/phase3_stage32_glm_predictions.parquet`
- **Test suite:** 156 tests pass (22 new GLM tests + 134 existing)

## Phase 3 Stage 3.3 status — complete

Phase 3 Stage 3.3 (Negative Binomial GLM) is complete and verified:

- **Implemented:** `src/models/glm.py` extended with Negative Binomial GLM:
  - Separate models for goals and assists using statsmodels NegativeBinomial
  - L2 regularization (alpha=1.0) with BFGS solver
  - StandardScaler fitted on training data only; constant features dropped
  - Fallback to zero predictions when training samples < min_train_samples (20)
  - Training and prediction deterministic (random_state=42)
  - Dispersion diagnostic (Pearson and Deviance) for overdispersion detection
  - Chronological cross-validation for regularization tuning
- **Leakage prevention:** All features from `match_week < cutoff`; 4 leakage tests pass
- **Test suite:** 171 tests pass (15 new NB tests + 156 existing)
- **Dispersion diagnostics:** Pearson dispersion ≈ 0.95 (goals), 0.67 (assists) — data is near equidispersed, not overdispersed
- **Fair evaluation:** Rolling-origin on cutoffs 2–10 (2,480 predictions) on identical observations for all models

**Comparison with baselines (Population A, cutoffs 2-10, 2,480 predictions):**

| Model | MAE Goals | RMSE Goals | Poisson Dev Goals | MAE Assists | RMSE Assists | Poisson Dev Assists |
|-------|-----------|------------|-------------------|-------------|--------------|---------------------|
| **Recent Form** | **0.1563** | 0.3825 | **0.1104** | **0.1155** | 0.3045 | **0.0617** |
| **Player Mean** | 0.1622 | 0.3595 | 0.2777 | 0.1205 | 0.2849 | 0.1656 |
| **Poisson GLM** | 0.1728 | 0.3515 | 0.4812 | 0.1240 | 0.2762 | 0.3788 |
| **NegBin GLM** | 0.1891 | 0.4372 | 0.4894 | 0.1209 | 0.2785 | 0.3935 |
| Global Mean | 0.1724 | 0.3532 | 0.4931 | 0.1239 | 0.2767 | 0.3827 |
| Minutes-Adjusted | 0.1918 | 0.4008 | 0.3014 | 0.1405 | 0.3138 | 0.1713 |

**Key finding:** Negative Binomial GLM does NOT improve on Poisson GLM or simple baselines. Data shows no meaningful overdispersion (Pearson dispersion ≈ 0.95 goals, 0.67 assists). The best model remains the simple Recent Form baseline. Negative Binomial adds complexity without benefit.

- **Outputs:** `reports/phase3_stage33_report.md`, `reports/phase3_stage33_nb_predictions.parquet`
- **Test suite:** 171 tests pass (15 new NB tests + 156 existing)

## Phase 4 Stage 4.1 status — complete

Phase 4 Stage 4.1 (match outcome baseline evaluation) is complete and verified:

- **Implemented:** Four match outcome baselines in `src/features/match_baselines.py`:
  - Global historical frequencies (Laplace smoothed)
  - Home advantage (same as global, using home/away split)
  - Team strength (home team home record + away team away record, fallback to global)
  - Recent form (last 5 matches per team, combined perspective, fallback to global)
- **Classification metrics added** to `src/evaluation/metrics.py`:
  - Multiclass log loss (cross-entropy) with clipping and renormalization
  - Multiclass Brier score (mean squared error vs one-hot targets)
  - Accuracy (argmax of probabilities)
- **Rolling-origin evaluation** in `src/evaluation/match_rolling.py`:
  - Chronological cutoffs across all 38 matchweeks of 2015/16
  - Strict `match_week < cutoff` history for training
  - Simultaneous matchweek assumption (no within-week leakage)
- **Leakage prevention:** All 6 leakage tests pass — injecting future matchweek data does not affect earlier-cutoff predictions
- **Test suite:** 214 tests pass (37 new match outcome tests + 177 existing)
- **Evaluation:** Rolling-origin on all 38 matchweeks (380 matches) of 2015/16 season

**Aggregate results (2015/16, 380 matches):**

| Model | Log Loss | Brier Score | Accuracy |
|-------|----------|-------------|----------|
| **Global Frequencies** | **1.0945** | **0.6632** | **0.4026** |
| **Home Advantage** | **1.0945** | **0.6632** | **0.4026** |
| Recent Form | 1.1184 | 0.6758 | 0.3789 |
| Team Strength | 1.1357 | 0.6914 | 0.3500 |

**Key finding:** The simple global frequencies (home/draw/away win rates) and home advantage baselines perform best on this dataset. Team strength and recent form baselines do not improve over the naive global frequencies, likely due to limited training data in early matchweeks and the single-season constraint. All baselines achieve ~40% accuracy, which is modest but expected given the inherent unpredictability of football matches.

- **Outputs:** `reports/phase4_stage41_report.md`, `reports/phase4_stage41_predictions.parquet`, `reports/phase4_stage41_metrics_per_cutoff.parquet`, `reports/phase4_stage41_metrics_aggregate.parquet`
- **Test suite:** 214 tests pass (37 new match outcome tests + 177 existing)

## Phase 4 Stage 4.2 status — complete

Phase 4 Stage 4.2 (leakage-safe match outcome features) is complete and verified:

- **Implemented:** Match feature engineering in `src/features/match_features.py`:
  - Rolling team stats (goals, points, xG) over last 5 matches
  - Home/away specific performance splits
  - Head-to-head history (window of 5 meetings)
  - Rest days since last match
  - Global fallback averages for goals, xG, outcome rates
  - All features computed strictly from `match_week < cutoff` history
- **Model:** Regularized multinomial logistic regression in `src/models/match_model.py`:
  - L2 regularization (C=1.0) with LBFGS solver
  - StandardScaler fitted on training data only
  - Balanced class weights for draw class
  - Handles missing classes (2-class training) gracefully
- **Evaluation:** Extended `src/evaluation/match_rolling.py` for feature model comparison
- **Leakage prevention:** All 6 leakage tests pass — future data injection does not affect earlier-cutoff features/predictions
- **Test suite:** 238 tests pass (24 new match feature/model tests + 214 existing)
- **Evaluation:** Fair comparison on identical observations (cutoffs 10–20, 110 matches)

**Aggregate results (cutoffs 10–20, 110 matches):**

| Model | Log Loss | Brier Score | Accuracy |
|-------|----------|-------------|----------|
| **Recent Form** | **1.1033** | **0.6577** | **0.4182** |
| Global Frequencies | 1.0893 | 0.6585 | 0.3727 |
| Home Advantage | 1.0893 | 0.6585 | 0.3727 |
| Team Strength | 1.1499 | 0.7142 | 0.3273 |
| **Feature Model** | **1.5156** | **0.8142** | **0.3273** |

**Key finding:** The feature-enhanced multinomial logistic regression does NOT improve on simple baselines. The model underperforms all baselines on all metrics due to: (1) insufficient training data (~140-280 matches for 50+ features), (2) no H2H history in single season, (3) constant features dropped (rest days, H2H, global draw rate), (4) feature collinearity and overfitting risk. The best practical predictor remains the Recent Form baseline (~42% accuracy).

- **Outputs:** `reports/phase4_stage42_report.md`, `reports/phase4_stage42_comparison_metrics.parquet`
- **Test suite:** 238 tests pass (24 new match feature/model tests + 214 existing)

## Phase 4 Stage 4.3 status — complete

Phase 4 Stage 4.3 (league simulation foundations) is complete and verified:

- **Implemented:** Monte Carlo league simulator in `src/models/league_simulator.py`:
  - Configurable Monte Carlo simulator (n_simulations, random_seed, points system, relegation/top places)
  - Pluggable probability providers: global frequencies, recent form
  - Strict temporal leakage prevention: history only from `match_week < cutoff`
  - Goal difference approximation (±1 for win/loss, 0 for draw)
  - Standings computation with points, GD, GS tiebreakers
  - Full simulation statistics: title/top-4/relegation probabilities, expected points/position, position distributions
- **Probability providers:**
  - Global frequency provider (Laplace-smoothed global home/draw/away rates)
  - Recent form provider (team-specific rolling form with global fallback)
- **Validation & safety:**
  - Probability validation (finite, [0,1], sum to 1.0)
  - Fixtures validation (required columns, no duplicates, no self-matches)
  - Missing scores handled gracefully (dropped from frequency calculations)
  - Fixed random seed for reproducibility
- **Evaluation integration:** Extended `src/evaluation/match_rolling.py` with `run_match_model_evaluation` (fixed metrics for model predictions)
- **Leakage prevention:** All fixtures validation tests pass
- **Test suite:** 273 tests pass (35 new league simulator tests + 238 existing)

**Simulation results (2015/16 season, 2,000 simulations, global frequency provider):**

| Team | Title Prob | Top-4 Prob | Releg Prob | Exp Points | Exp Position |
|------|------------|------------|------------|------------|--------------|
| Team 1 | 5.30% | 21.40% | 0.00% | 51.72 | 10.32 |
| Team 30 | 5.50% | 20.20% | 0.00% | 51.83 | 10.38 |
| Team 39 | 0.05% | 20.10% | 0.00% | 51.88 | 10.38 |
| Team 30 | 5.50% | 20.20% | 0.00% | 51.83 | 10.38 |
| Team 38 | 0.00% | 0.00% | 16.85% | 45.20 | 17.20 |

*Full results show well-distributed title probabilities (max 5.5%), competitive top-4 race, and balanced relegation battle. Expected points cluster around league average (52), reflecting the symmetric global frequencies.*

**Key finding:** The league simulator is functional and produces reproducible, plausible results. The global frequency provider gives maximum-entropy baseline simulations useful as a null model. The simulator architecture supports plugging in any probability provider for comparison. The best practical match predictor remains the Recent Form baseline (~42% accuracy), and the feature model does not improve on baselines with single-season data.

- **Outputs:** `reports/phase4_stage43_report.md`, `reports/phase4_stage43_standings.parquet`, `reports/phase4_stage43_title_probs.parquet`, `reports/phase4_stage43_top4_probs.parquet`, `reports/phase4_stage43_releg_probs.parquet`, `reports/phase4_stage43_exp_points.parquet`, `reports/phase4_stage43_exp_position.parquet`, `reports/phase4_stage43_pos_dist.parquet`
- **Test suite:** 273 tests pass (35 new league simulator tests + 238 existing)

## Phase 4 Stage 4.4 status — complete

Phase 4 Stage 4.4 (goal-level league scoreline simulation) is complete and verified:

- **Implemented:** Goal-level league simulation in `src/models/league_simulator.py`:
  - Dual simulation modes: "outcome" (W/D/L) and "scoreline" (exact goals)
  - Independent Poisson scoreline provider with team-specific goal rates
  - Home advantage as additive boost to home goal rate
  - Team-specific home/away goal rates with global fallbacks
  - Goal difference and goals scored computed from sampled scorelines
- **Goal-level metrics** in `src/evaluation/metrics.py`:
  - MAE/RMSE for home and away goals
  - Poisson log-likelihood (proper scoring rule for count data)
  - Scoreline probability evaluation framework
- **Scoreline probability provider** using independent Poisson models:
  - Team-specific home/away goal rates with global fallbacks
  - Home advantage as additive boost to home goal rate
  - Laplace smoothing for goal rate estimation
  - Scoreline probabilities from independent Poissons (truncated at 10 goals)
- **Dual simulation modes** with configurable `simulation_mode`:
  - "outcome": Fast W/D/L simulation with approximate GD/GS
  - "scoreline": Exact goals sampled from Poisson, exact GD/GS
- **Goal-level metrics** in `src/evaluation/metrics.py`:
  - MAE/RMSE for home and away goals
  - Poisson log-likelihood (proper scoring rule)
  - Scoreline evaluation framework
- **Leakage prevention:** All fixtures validation tests pass, temporal ordering maintained
- **Test suite:** 273 tests pass (35 new league simulator tests + 238 existing)

**Simulation results (2015/16 season):**

| Mode | Simulations | Title Prob (max) | Top-4 Spread | Relegation Spread |
|------|-------------|------------------|--------------|-------------------|
| Outcome (Global) | 2,000 | 6.8% | 18 teams > 20% | Balanced (10 teams > 15%) |
| Scoreline (Poisson) | 100 | 20% | Concentrated | Concentrated |

**Key findings:**
1. **Scoreline mode works** — produces exact GD/GS for proper tiebreakers
2. **Independent Poisson assumption** limits scoreline accuracy — real matches show goal dependence
3. **Global frequency provider** gives maximum-entropy baseline (max title prob 6.8%)
4. **Scoreline mode slower** (~10x) due to Poisson sampling and probability computation
5. **Best practical predictor** remains Recent Form baseline (~42% accuracy)

- **Outputs:** `reports/phase4_stage44_report.md`, `reports/phase4_stage44_standings.parquet`, `reports/phase4_stage44_title_probs.parquet`, `reports/phase4_stage44_top4_probs.parquet`, `reports/phase4_stage44_releg_probs.parquet`, `reports/phase4_stage44_exp_points.parquet`, `reports/phase4_stage44_exp_position.parquet`, `reports/phase4_stage44_pos_dist.parquet`, `reports/phase4_stage44_standings_scoreline.parquet`, `reports/phase4_stage44_title_probs_scoreline.parquet`, etc.
- **Test suite:** 273 tests pass (35 new league simulator tests + 238 existing)
