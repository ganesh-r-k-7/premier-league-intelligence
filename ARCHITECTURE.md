# Architecture

This document describes the target architecture of Premier League Intelligence and the responsibilities of each component. It is the technical companion to the master specification and to `PROJECT_PLAN.md`, which defines phases, datasets, targets and validation. Currently only the master specification and the Phase 0 documents exist; the remaining directories will be populated phase by phase.

## System overview

Raw football data from external sources flows through ingestion, validation and feature engineering, into models, and finally into evaluation, simulation, scouting and presentation. The dashboard is a pure consumer of the processed results — it never reads raw sources directly.

```text
external sources ──► ingestion & validation ──► processed datasets
                                                    │
                                features  ┼──► evaluation ──► simulation / scouting
                                                    │
                                                    ▼
                            visualization ──► dashboard
```

Key properties:

- Raw data is immutable and separated from derived data.
- The modeling core is developed in phases 1–5; the dashboard consumes it in phase 6.
- All evaluation code is shared across model modules, so every prediction is validated with the same methodology.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `app/` | Streamlit multipage application. `Home.py` is the entry point; `pages/` contains one page per dashboard module; `components/` contains shared UI building blocks. |
| `src/data/` | Data ingestion, validation and storage. Raw files are stored under `data/raw/`; validated datasets under `data/processed/`. |
| `src/features/` | Feature engineering, per prediction module. Produces feature tables that models consume. |
| `src/models/` | Model implementations and their serialized artifacts. Each module owns its models. |
| `src/evaluation/` | Metrics, train/test splitting, baseline comparisons and report generation. Shared by all modules. |
| `src/simulation/` | Monte Carlo simulation of remaining fixtures and league outcomes (phase 4). |
| `src/scouting/` | Hidden-gem heuristics and transfer-comparison indicators (phase 5). |
| `src/visualization/` | Plotly chart builders, reused by the dashboard. |
| `src/utils/` | Shared helpers: configuration, logging, file handling. |
| `notebooks/` | Exploratory analysis. Reading code lives in `src/`; notebooks never redefine pipeline logic. |
| `reports/` | Generated analysis and evaluation reports. |
| `tests/` | pytest test suite for features, evaluation, simulation and key ingestion functions. |
| `scripts/` | Command-line entry points for pipeline stages (ingest, features, evaluate). |
| `docs/` | This file and supporting documentation. |
| `data/raw/` | Immutable raw snapshots from external sources. |
| `data/processed/` | Validated, feature-ready datasets. |
| `models/` | Serialized model artifacts. Not committed; regenerated deterministically. |
| Root | `requirements.txt`, `.gitignore`, `README.md`. |

## Key design decisions

- **Raw/processed separation.** Raw files are written once and never modified. Derived datasets are validated before use, and every derived table carries provenance columns (source, season). This keeps reproducibility and allows reprocessing without touching raw data.
- **Reproducible inference.** A model is fully described by (a) the processed dataset snapshot used, (b) the trained pipeline, and (c) the serialization. Inference reproduces predictions from those inputs alone; no state from a training run is reused implicitly.
- **Configuration.** All paths, source URLs and hyperparameters come from a single configuration module. No credentials or API keys are stored in code or committed.
- **Honest dashboard.** The dashboard displays only validated predictions, clearly states the prediction horizon, data coverage and uncertainty, and explicitly marks unavailable or estimated values. It never shows data that has not been ingested and validated.
- **Shared evaluation.** Every supervised module uses the same chronological splitting and the same baseline requirements. Metrics, not model accuracy, are the primary reporting artefact.
- **Testing strategy.** Unit tests cover feature engineering and evaluation utilities. Integration tests use a small fixture dataset instead of live sources, so the suite runs without network access.
- **Modularity across phases.** Phases 1–3 build the modeling core; phases 4–5 add simulation and scouting on top; phase 6 presents results. Later phases consume the outputs of earlier ones and never re-implement them.
- **Scope discipline.** Only the features listed in the master specification are implemented. Optional features are phase 8 and are excluded until explicitly approved.

## Cross-phase integration

- Phase 1 produces validated raw and processed datasets. Phases 2–3 consume them and add model code.
- Phase 2 evaluation utilities are reused by phases 3–5.
- Phase 4 (match prediction and league simulation) consumes team and player states derived in phases 1–3, and relies on the evaluation utilities to prove its models.
- Phase 5 (scouting and transfer comparison) consumes the same player states; its outputs are descriptive indicators plus retrospective validations.
- Phase 6 (dashboard) reads only processed datasets and serialized models. Raw sources are never exposed to the dashboard.
- Phase 7 (tests, documentation, reproducibility) wraps up by locking the interfaces between all of the above.
