# Premier League Intelligence — Master Specification

## Vision
Build a portfolio-quality Premier League football analytics platform using real football data, machine learning, statistical forecasting, simulations, explainability, and an interactive dashboard. Predictions must be clearly labelled as estimates and validated honestly.

## Core modules
1. **Market-value prediction:** Estimate player value using age, position, minutes, performance and reliable contextual data. Forecast 1–3 years only if the data supports it. Distinguish market value from transfer fees.
2. **Goals and assists forecasting:** Predict goals and assists for a defined period using historical output, playing time, position, team strength and opponent context where available. Distinguish xG from actual goals.
3. **Match and league prediction:** Estimate home/draw/away and scoreline probabilities. Simulate remaining fixtures for title, final positions, European qualification and relegation probabilities.
4. **Hidden-gem scouting:** Compare position- and age-relevant players using per-90 metrics, minimum-minute filters and value/performance indicators. Separate heuristics from validated predictions.
5. **Transfer intelligence:** Compare players by value, performance, age, position and budget. Do not equate market value with a negotiated transfer fee.
6. **Team and player analytics:** Player profiles, club summaries, trends, comparisons and match previews.

## Optional features
Fantasy Premier League expected-points/captain recommendations; Golden Boot probabilities; what-if scenarios; career trajectory visualisations; AI football analyst grounded in verified app data.

## Technology
Python, pandas, NumPy, scikit-learn, XGBoost where justified, SciPy/statsmodels where appropriate, SHAP where suitable, Streamlit, Plotly, pytest, Jupyter, Git/GitHub, and SQLite or Parquet where suitable. Use a virtual environment and compatible dependency versions. Avoid unnecessary paid services.

## Candidate data sources to investigate
- Football-Data.co.uk: historical fixtures, results and match statistics (coverage varies by season).
- Official Premier League statistics: player/team data subject to availability and permitted access.
- StatsBomb Open Data: detailed event data for selected competitions/seasons, not necessarily every Premier League season.
- Transfermarkt or another documented source: possible historical values/transfers, subject to licensing and terms.

Verify data fields, historical coverage, licensing, update frequency, join keys, gaps and reproducibility. Do not bypass access controls or fabricate missing data. Do not assume these sources form a complete unified dataset.

## Validation requirements
Use chronological/season-based train/test splits; prevent future-data leakage; establish baselines; report suitable metrics (regression MAE/RMSE, goal-count metrics, match-probability Brier score/log loss/calibration); perform league simulation sanity checks. Display prediction horizon, data freshness, uncertainty and limitations. Never invent evaluation results or claim accuracy without held-out testing.

## Planned dashboard pages
Overview; Player Explorer; Market Value Predictor; Goals & Assists Forecast; Match Predictor; Title-Race Simulator; Hidden Gems & Scouting; Transfer Comparison; Model Performance & Methodology.

Design direction: professional dark-navy interface with restrained green accents, clear typography, useful filters and Plotly charts. Never show fake live data when ingestion is not configured.

## Suggested project structure
```text
premier-league-intelligence/
  app/ (Home.py, pages/, components/)
  src/ (data/, features/, models/, evaluation/, simulation/, scouting/, visualization/, utils/)
  data/ (raw/, processed/)
  models/
  notebooks/
  reports/
  tests/
  scripts/
  docs/
  .gitignore
  README.md
  ARCHITECTURE.md
  PROJECT_PLAN.md
  requirements.txt
```
Keep raw and processed data separate. Avoid committing secrets, private data, large datasets or unnecessary model artifacts.

## Phases
0. Architecture and data-source feasibility.
1. Environment, ingestion, validation and exploratory analysis.
2. Goals/assists features, forecasting and evaluation.
3. Market-value prediction and explainability, subject to data availability.
4. Match prediction and league simulation.
5. Scouting and transfer comparison.
6. Dashboard integration.
7. Tests, documentation and reproducibility.
8. Optional fantasy, scenarios and AI analyst.

## Instructions to OpenCode
Act as a senior data scientist, ML engineer, football analytics researcher and software architect.
1. Inspect the directory before changing files.
2. Read this entire specification and use it as the project scope.
3. Start with Phase 0 only; do not build the entire app at once.
4. Create `ARCHITECTURE.md` and `PROJECT_PLAN.md`.
5. Document data feasibility/gaps, module datasets, join strategy, targets, validation, dependencies, risks and completion criteria.
6. Verify source claims using available documentation; do not fabricate findings.
7. Keep code modular, readable and tested; explain decisions clearly.
8. Do not overwrite unrelated files, bypass access restrictions or hardcode secrets.
9. After Phase 0, report files created, proposed architecture, data feasibility, risks and the exact next prompt to start Phase 1.
10. Wait for my explicit approval before implementing Phase 1.

End goal: a reproducible, portfolio-quality football analytics product with explainable and honestly evaluated predictions.
