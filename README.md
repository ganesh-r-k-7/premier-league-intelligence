# Premier League Intelligence

A football analytics platform built on StatsBomb Open Data, implementing player performance forecasting, match prediction, league simulation, and scouting tools — all with honest validation and explicit limitations.

## Current Status

**Phase 7 Complete** — Dashboard hardened, reproducibility verified, documentation updated.

- **Data**: Premier League 2015/16 season (StatsBomb Open Data)
- **Coverage**: 380 matches, 20 teams, 550 players (2015/16 PL only; 2003/04 partial data available but not used in evaluation)
- **Test Suite**: 356 tests passing

## Quick Start

```bash
# 1. Clone and enter directory
git clone https://github.com/ganesh-r-k-7/premier-league-intelligence
cd premier-league-intelligence

# 2. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Ingest data (uses cached raw files offline)
python -m src.data.ingest_players --offline

# 4. Launch dashboard
streamlit run src/dashboard/app.py
```

Open `http://localhost:8501` in your browser.

## Dashboard Sections

| Section | Description |
|---------|-------------|
| **Overview** | Dataset coverage, season statistics, top scorers, report viewer |
| **Player Explorer** | Search/filter players, view per-90 metrics, percentile ranks, teammate comparisons |
| **Scouting & Comparisons** | Side-by-side player comparison, position percentiles, configurable scouting priority scores, similar players analysis |
| **Match Analysis** | Evaluated baseline performance (4 baselines), feature model evaluation, confusion matrices, reports |
| **League Simulator** | Monte Carlo simulation with 3 probability providers, title/top-4/relegation probabilities, expected final table |

## Project Phases

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Architecture & feasibility | ✅ Complete |
| 1 | Environment, ingestion, EDA | ✅ Complete |
| 2 | Goals/assists forecasting & evaluation | ✅ Complete |
| 3 | Market-value prediction & explainability | ✅ Complete (GLM + NegBin) |
| 4 | Match prediction & league simulation | ✅ Complete |
| 5 | Scouting & transfer comparison | ✅ Complete |
| 6 | Dashboard integration | ✅ Complete |
| 7 | Tests, docs, reproducibility | ✅ **Complete** |
| 8 | Optional features | ⏳ Future |

## Key Results Summary

### Phase 2: Goals/Assists Baselines (Rolling-origin, 38 matchweeks)
| Model | MAE Goals | RMSE Goals | Poisson Dev Goals |
|-------|-----------|------------|-------------------|
| Recent Form | 0.1527 | 0.3523 | 0.1211 |
| Player Mean | 0.1558 | 0.3315 | 0.2665 |
| Global Mean | 0.1693 | 0.3346 | 0.4608 |
| Minutes-Adjusted | 0.1772 | 0.3526 | 0.2869 |

### Phase 3: GLM & Negative Binomial
- Poisson GLM and NegBin GLM do **not** clearly outperform simple baselines
- Data shows no meaningful overdispersion (Pearson dispersion ≈ 0.95 goals, 0.67 assists)
- Best predictor remains Recent Form baseline

### Phase 4: Match Outcome & League Simulation
| Model | Log Loss | Brier Score | Accuracy |
|-------|----------|-------------|----------|
| Global Frequencies | 1.0945 | 0.6632 | 40.3% |
| Recent Form | 1.1184 | 0.6758 | 37.9% |
| Team Strength | 1.1357 | 0.6914 | 35.0% |
| Feature Model | 1.5156 | 0.8142 | 32.7% |

- League simulator functional with 2,000+ simulations
- Global frequency provider: max title probability ~6.8%
- Scoreline mode (independent Poisson) works but ~10x slower

### Phase 5: Scouting & Comparisons
- Position-aware rankings (GK/DEF/MID/WING/FWD) for goals/90, xG/90, assists/90, etc.
- Similarity analysis: Harry Kane most similar to Agüero (cosine=0.94)
- Scouting priority: multi-component scoring with configurable weights
- **Explicitly labelled as heuristics, not predictions or valuations**

## Data Sources

| Source | Coverage | License | Status |
|--------|----------|---------|--------|
| StatsBomb Open Data | PL 2015/16 (full), 2003/04 (partial) | Open for research, attribution required | ✅ Primary |
| Understat | PL 2014/15+ | robots.txt disallows, no open license | ❌ Not used |
| FBref | Basic only (xG removed 2026) | ToS prohibits scraping | ❌ Not used |
| Football-Data.co.uk | Match-level only | 403 from this environment | ❌ Not viable |
| Transfermarkt | Subscription required | Subscription-gated | ❌ Not used |

## Limitations (Always Displayed)

1. **Single season only** — No cross-season validation possible with PL data (2015/16 only; 2003/04 partial data available but not used in evaluation)
2. **No market values** — Scouting is performance-only, no financial data
3. **No injury/rotation signals** — Lineup data shows who played, not why others didn't
4. **Independent Poisson assumption** — Scoreline simulation ignores goal dependence
5. **Models don't beat baselines** — Feature models underperform simple heuristics on this data

## Project Structure

```
premier-league-intelligence/
├── src/
│   ├── data/           # Ingestion, validation, config
│   ├── features/       # Player/match features, baselines, scouting
│   ├── models/         # GLM, match model, league simulator
│   ├── evaluation/     # Metrics, rolling evaluation
│   └── dashboard/      # Streamlit app (Phase 6)
├── data/
│   ├── raw/            # Immutable StatsBomb JSON (gitignored, ~1.1 GB)
│   └── processed/      # Parquet datasets (gitignored)
├── reports/            # Markdown & parquet reports
├── tests/              # 356 tests
├── requirements.txt    # Pinned dependencies
├── PROJECT_PLAN.md     # Phase tracking & decisions
└── ARCHITECTURE.md     # System design
```

## Running Tests

```bash
source .venv/bin/activate
python -m pytest           # All 356 tests
python -m pytest tests/test_dashboard.py -v  # Dashboard tests only
```

## Reports

All phase reports in `reports/`:
- `phase1_eda_report.md` — Exploratory data analysis
- `phase2_stage3_report.md` — Baseline evaluation
- `phase3_stage31_report.md` — Player feature engineering
- `phase3_stage32_report.md` — Poisson GLM
- `phase3_stage33_report.md` — Negative Binomial GLM
- `phase4_stage41_report.md` — Match outcome baselines
- `phase4_stage42_report.md` — Match feature model
- `phase4_stage43_report.md` — League simulation (outcome mode)
- `phase4_stage44_report.md` — League simulation (scoreline mode)
- `phase5_stage51_report.md` — Player scouting profiles
- `phase5_stage52_report.md` — Player comparison & priority
- `phase6_dashboard_report.md` — Dashboard integration summary

## Attribution

**Data Source**: StatsBomb Open Data — https://github.com/statsbomb/open-data  
Used under open data license for research purposes. Attribution required.

## License

This project is for research and educational purposes. StatsBomb data subject to their terms.