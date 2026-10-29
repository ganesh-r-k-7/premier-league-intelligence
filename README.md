# Premier League Intelligence

A football analytics platform built on StatsBomb Open Data, implementing player performance forecasting, match prediction, league simulation, and scouting tools — all with honest validation and explicit limitations.

## Current Status

**Phase 8.3C Complete** — Player identity curation workflow for 1,067 unmatched FPL player-season rows and cross-provider matching.

- **Data**: Three sources covering 2010/11–2025/26
  - **StatsBomb**: 2015/16 (380 matches, 20 teams, 550 players, event data)
  - **OpenFootball**: 2010/11–2025/26 (16 seasons, 6,080 matches, match results only)
  - **FPL**: 2016/17–2025/26 (10 seasons, 253,900 player-match rows, 7,364 player-seasons, xG/xA from 2022/23)
- **Test Suite**: 419 tests passing (388 core + 31 identity curation)

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
python -m src.data.ingest_multi_season --offline

# 4. Launch dashboard
streamlit run src/dashboard/app.py
```

Open `http://localhost:8501` in your browser.

## Dashboard Sections

| Section | Description |
|---------|-------------|
| **Overview** | Multi-source, multi-season coverage summary, season-level trends, source comparison |
| **Player Explorer** | Search/filter players across StatsBomb (2015/16) and FPL (2016/17–2025/26); per-90 metrics, percentile ranks, teammate comparisons |
| **Scouting & Comparisons** | Side-by-side player comparison, position percentiles, configurable scouting priority scores, similar players analysis (per source/season) |
| **Match Analysis** | Season summary, match results table, model evaluation (StatsBomb 2015/16 only) |
| **League Simulator** | Monte Carlo simulation (StatsBomb 2015/16 only — requires event data) |

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
| 7 | Tests, docs, reproducibility | ✅ Complete |
| 8.1 | Multi-season data source audit | ✅ Complete |
| 8.2 | Multi-season data ingestion | ✅ Complete |
| 8.3A | Multi-season dashboard integration | ✅ **Complete** |
| 8.3B | Multi-season model evaluation & retraining | ✅ **Complete** |
| 8.3C | Player identity resolution | ✅ **Complete** |
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

### Phase 8.1/8.2: Multi-Season Data
| Source | License | Seasons | Matches | Player Stats | xG/xA |
|--------|---------|---------|---------|--------------|-------|
| **StatsBomb** | Attribution | 2015/16 | 380 | ✅ | ✅ |
| **OpenFootball** | CC0 | 2010/11–2025/26 (15 complete) | 6,080 | ❌ | ❌ |
| **FPL** | MIT | 2016/17–2025/26 (10) | 253,900 gw | ✅ | 2022/23+ |

## Data Sources

| Source | Coverage | License | Status |
|--------|----------|---------|--------|
| StatsBomb Open Data | PL 2015/16 (full), 2003/04 (partial) | Open for research, attribution required | ✅ Primary |
| openfootball/football.json | PL 2010/11–2025/26 (match results) | CC0 (Public Domain) | ✅ Complete |
| vaastav/Fantasy-Premier-League | PL 2016/17–2025/26 (player stats) | MIT | ✅ Complete |
| Understat | PL 2014/15+ | robots.txt disallows, no open license | ❌ Not used |
| FBref | Basic only (xG removed 2026) | ToS prohibits scraping | ❌ Not used |
| Football-Data.co.uk | Match-level only | 403 from this environment | ❌ Not viable |
| Transfermarkt | Subscription required | Subscription-gated | ❌ Not used |

### Phase 8.3B: Multi-Season Model Evaluation Results

| Model | MAE Goals | RMSE Goals | Poisson Dev | MAE Assists | RMSE Assists | Poisson Dev |
|-------|-----------|------------|-------------|-------------|--------------|-------------|
| **Recent Form (baseline)** | **0.274** | **0.471** | **0.201** | **0.182** | **0.332** | **0.131** |
| Poisson GLM | 0.291 | 0.498 | 0.228 | 0.194 | 0.352 | 0.151 |
| NegBin GLM | 0.289 | 0.492 | 0.225 | 0.191 | 0.348 | 0.148 |
| Global Mean | 0.287 | 0.482 | 0.214 | 0.189 | 0.341 | 0.142 |
| Player Mean | 0.271 | 0.471 | 0.201 | 0.182 | 0.332 | 0.131 |
| Minutes-Adjusted | 0.281 | 0.482 | 0.214 | 0.182 | 0.332 | 0.145 |

**Key Finding:** No model outperforms the simple Recent Form baseline. The FPL data's limited history (only 6 seasons with xG/xA) constrains model complexity.

### Limitations (Always Displayed)

1. **StatsBomb single season only** — No cross-season validation possible with PL data
2. **OpenFootball partial 2025/26** — Some malformed scores
3. **FPL xG/xA only 2022/23+** — Legacy seasons lack advanced metrics
4. **FPL missing team/position** for legacy seasons (~35% rows)
5. **No cross-provider player matching** — Different ID systems, no season overlap
6. **No market values** — Scouting is performance-only
7. **League Simulator only for StatsBomb 2015/16** — Requires event data
7. **Models don't beat baselines** — Feature models underperform simple heuristics on this data

## Project Structure

```
premier-league-intelligence/
├── src/
│   ├── data/           # Ingestion, validation, config
│   ├── features/       # Player/match features, baselines, scouting
│   ├── models/         # GLM, match model, league simulator
│   ├── evaluation/     # Metrics, rolling evaluation
│   └── dashboard/      # Streamlit app (Phases 6, 8.3A)
├── data/
│   ├── raw/            # Immutable StatsBomb JSON, OpenFootball JSON, FPL CSV (gitignored)
│   └── processed/      # Parquet datasets (gitignored)
├── reports/            # Markdown & parquet reports
├── tests/              # 372 tests
├── requirements.txt    # Pinned dependencies
├── PROJECT_PLAN.md     # Phase tracking & decisions
└── ARCHITECTURE.md     # System design
```

## Running Tests

```bash
source .venv/bin/activate
python -m pytest           # All 372 tests
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
- `phase8.1_source_audit.md` — Multi-season source audit
- `phase8.2_ingestion_report.md` — Multi-season ingestion report
- `phase8.3A_dashboard_report.md` — Multi-season dashboard integration report
- `phase8.3B_multiseason_model_evaluation.md` — Multi-season model evaluation report
- `phase8.3C_player_identity_curation.md` — Player identity curation report

## Attribution

**Data Sources**: 
- StatsBomb Open Data — https://github.com/statsbomb/open-data  
- openfootball/football.json — https://github.com/openfootball/football.json
- vaastav/Fantasy-Premier-League — https://github.com/vaastav/Fantasy-Premier-League

Used under open data licenses for research purposes. Attribution required.

## License

This project is for research and educational purposes. Source data subject to their respective terms.