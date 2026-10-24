# Phase 3 Stage 3.3 — Negative Binomial GLM Evaluation Report

**Date:** 2026-10-24
**Status:** Complete

---

## 1. Summary

This stage implements a Negative Binomial GLM for player goal and assist prediction to address potential overdispersion in count data, and evaluates it against the four Phase 2 baselines and the Poisson GLM from Stage 3.2 on **identical observations**.

**Files created/modified:**
- `src/models/glm.py` — Extended with Negative Binomial GLM implementation, dispersion diagnostics, and chronological regularization tuning (fixed `tol` → `gtol` for BFGS; skip matchweek 1 in training)
- `src/features/player_features.py` — Fixed duplicate merge when `match_week` already present
- `tests/test_glm.py` — Extended with 15 new tests for NB GLM, dispersion diagnostics, and regularization tuning (total 37 tests)
- `reports/phase3_stage33_report.md` — This report

**Test results:** 177 tests pass (all tests including new NB tests)

---

## 2. Model Specification

### 2.1 Negative Binomial GLM Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Distribution** | Negative Binomial (NB2) | Handles overdispersion (variance > mean) |
| **Link function** | Log (canonical) | Ensures non-negative predictions |
| **Regularization** | L2 (ridge) on coefficients | Prevents overfitting |
| **Regularization strength (α)** | 1.0 | Default; moderate shrinkage |
| **Solver** | BFGS (statsmodels) | Handles L2 penalty |
| **Max iterations** | 1000 | Sufficient for convergence |
| **Convergence tolerance (gtol)** | 1e-4 | Gradient norm tolerance for BFGS |
| **Minimum training samples** | 50 | Fallback to zero predictions if fewer |

### 2.2 Poisson GLM Configuration (from Stage 3.2)

| Parameter | Value |
|-----------|-------|
| **Distribution** | Poisson |
| **Link function** | Log (canonical) |
| **Regularization** | L2 (ridge) |
| **Regularization strength (α)** | 1.0 |
| **Solver** | Newton-Cholesky (scikit-learn) |
| **Max iterations** | 1000 |
| **Convergence tolerance** | 1e-4 |
| **Minimum training samples** | 20 |

### 2.3 Dispersion Diagnostic

Implemented Pearson and Deviance-based dispersion diagnostics on training data:

- **Pearson dispersion** = Σ(y - μ)²/μ / (n - p)
- **Deviance dispersion** = 2×Σ(μ - y + y×log(y/μ)) / (n - p)

Values ≈ 1.0 indicate equidispersion (Poisson appropriate); > 1.0 indicates overdispersion.

### 2.4 Regularization Tuning

Chronological cross-validation with 3 folds:
- Training data split chronologically (older folds train, next fold validates)
- Alpha grid: [0.01, 0.1, 1.0, 10.0]
- Metric: Pearson dispersion on validation fold (lower is better)
- Falls back to default α if insufficient data for CV

---

## 3. Evaluation Protocol (Fair Comparison)

### 3.1 Chronological Rolling-Origin

All models evaluated on **identical observations** using the same protocol:

- **Population**: All lineup players (Population A)
- **Cutoffs for fair comparison**: Matchweeks 3–6 (where all three models produce non-fallback predictions)
  - Cutoff 2: GLM/NB have no training data (matchweek 1 features are constant) → fallback to zeros
  - Cutoffs 3–6: All models have training data from matchweeks 2+
- **Training data**: Matchweeks `< cutoff` AND `≥ 2` (skip matchweek 1 due to constant features)
- **Test data**: Matchweek `== cutoff`
- **Refit**: Models retrained from scratch at each cutoff
- **Rows per cutoff**: ~276 (all lineup players)
- **Total evaluation rows (cutoffs 3–6)**: 1,105 (4 cutoffs × ~276)
- **Total evaluation rows (cutoffs 2–10, Poisson GLM + baselines)**: 2,480

### 3.2 Excluded Observations

| Cutoff | Reason for Exclusion |
|--------|---------------------|
| 1 | No prior matchweeks for any model |
| 2 | Training data only from matchweek 1 → all features constant → fallback to zeros for GLM/NB |
| 7–10 | NB not evaluated (computational cost); Poisson GLM evaluated on 2–10 |

### 3.3 Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MAE** | `mean(|y - ŷ|)` | Goals & assists separately |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Goals & assists separately |
| **Poisson Deviance** | `2 × mean(ŷ - y + y × log(y/ŷ))` | `0 × log(0) = 0` convention; skipped if any `ŷ < 0` |

---

## 4. Dispersion Diagnostics (Training Data at Cutoff 3)

| Target | Mean | Variance | Var/Mean | Pearson Dispersion | Deviance Dispersion |
|--------|------|----------|----------|-------------------|---------------------|
| Goals | 0.083 | 0.102 | 1.22 | 0.95 | 0.46 |
| Assists | 0.065 | 0.048 | 0.73 | 0.67 | 0.32 |

**Interpretation:**
- Goals: Slight raw overdispersion (Var/Mean = 1.22), but Pearson dispersion ≈ 0.95 suggests near-equidispersion after accounting for covariates
- Assists: Slight underdispersion (Var/Mean = 0.73), Pearson dispersion = 0.67
- **Conclusion**: Data is close to equidispersed; strong overdispersion is NOT present. The Poisson assumption is adequate.

---

## 5. Results

### 5.1 Fair Comparison: All Models on Identical Observations (Cutoffs 3–6, 1,105 rows)

| Model | MAE Goals | RMSE Goals | Poisson Dev Goals | MAE Assists | RMSE Assists | Poisson Dev Assists |
|-------|-----------|------------|-------------------|-------------|--------------|---------------------|
| **Recent Form** | **0.1437** | 0.3488 | **0.0905** | **0.1062** | 0.2806 | **0.0621** |
| **Player Mean** | 0.1495 | 0.3286 | 0.2753 | 0.1089 | 0.2563 | 0.1482 |
| **Poisson GLM** | 0.1497 | **0.3108** | 0.4238 | 0.1069 | **0.2380** | 0.3219 |
| Global Mean | 0.1561 | 0.3125 | 0.4361 | 0.1103 | 0.2385 | 0.3255 |
| Minutes-Adjusted | 0.1791 | 0.3614 | 0.2938 | 0.1318 | 0.2908 | 0.1574 |
| **NegBin GLM** | **0.2039** | **0.6209** | **0.5063** | **0.1483** | **1.1566** | **0.3838** |

### 5.2 Poisson GLM + Baselines (Cutoffs 2–10, 2,480 rows)

| Model | MAE Goals | RMSE Goals | Poisson Dev Goals | MAE Assists | RMSE Assists | Poisson Dev Assists |
|-------|-----------|------------|-------------------|-------------|--------------|---------------------|
| **Recent Form** | **0.1563** | 0.3825 | **0.1104** | **0.1155** | 0.3045 | **0.0617** |
| **Player Mean** | 0.1622 | 0.3595 | 0.2777 | 0.1205 | 0.2849 | 0.1656 |
| **Poisson GLM** | 0.1590 | 0.3522 | 0.4117 | 0.1153 | 0.2770 | 0.3230 |
| Global Mean | 0.1724 | 0.3532 | 0.4931 | 0.1239 | 0.2767 | 0.3827 |
| Minutes-Adjusted | 0.1918 | 0.4008 | 0.3014 | 0.1405 | 0.3138 | 0.1713 |

### 5.3 Key Observations

1. **Recent Form** is the best overall model on both evaluation sets (lowest MAE, lowest Poisson Deviance for both goals and assists)
2. **Player Mean** is second best
3. **Poisson GLM** performs similarly to Global Mean on cutoffs 2–10, slightly better on MAE Goals (0.1590 vs 0.1724) but worse on Poisson Deviance; on cutoffs 3–6 it matches Player Mean on MAE
4. **Negative Binomial GLM performs WORSE than Poisson GLM and all baselines** on all metrics
5. Both GLMs are outperformed by simple baselines (Recent Form, Player Mean)

### 5.4 Per-Cutoff Patterns (Cutoffs 3–6)

| Cutoff | Recent Form | Player Mean | Poisson GLM | NegBin GLM | Global Mean | Min-Adj |
|--------|-------------|-------------|-------------|------------|-------------|---------|
| **MAE Goals** | | | | | | |
| 3 | **0.1511** | 0.1601 | 0.1503 | 0.1377 | 0.1601 | 0.2201 |
| 4 | **0.1317** | 0.1402 | 0.1443 | 0.2235 | 0.1524 | 0.1586 |
| 5 | **0.1578** | 0.1599 | 0.1563 | 0.3129 | 0.1588 | 0.1821 |
| 6 | **0.1342** | 0.1378 | 0.1481 | 0.1404 | 0.1530 | 0.1563 |

| Cutoff | Recent Form | Player Mean | Poisson GLM | NegBin GLM | Global Mean | Min-Adj |
|--------|-------------|-------------|-------------|------------|-------------|---------|
| **MAE Assists** | | | | | | |
| 3 | **0.1114** | 0.1146 | 0.1118 | **0.0549** | 0.1146 | 0.1632 |
| 4 | **0.1011** | 0.1051 | 0.1012 | 0.3529 | 0.1047 | 0.1208 |
| 5 | 0.1168 | 0.1167 | **0.1149** | 0.0849 | 0.1202 | 0.1340 |
| 6 | **0.0957** | 0.0993 | 0.0997 | 0.1002 | 0.1020 | 0.1099 |

**Notes:**
- NegBin GLM has highly variable performance across cutoffs (e.g., CW4 MAE Goals 0.22 vs 0.14 at CW3)
- NegBin GLM RMSE spikes at CW4 (Goals: 0.94, Assists: 2.27) due to extreme predictions (max 13.6 goals, 27.5 assists)
- Poisson GLM is more stable across cutoffs

---

## 6. Analysis

### Why Negative Binomial Fails to Improve

1. **No meaningful overdispersion**: Dispersion diagnostics show data is nearly equidispersed (Pearson ~0.95 for goals, 0.67 for assists). The Negative Binomial's extra dispersion parameter adds complexity without benefit.

2. **Estimation instability**: The NB model has an additional parameter (dispersion) that is hard to estimate reliably with sparse, zero-inflated data (92% zero goals, 95% zero assists).

3. **Overfitting / extreme predictions**: The NB model produces extreme predictions at some cutoffs (max 13.6 goals, 27.5 assists in a single match), leading to very high RMSE.

4. **Regularization not tuned per cutoff**: The fixed α=1.0 may not be optimal for NB, but chronological CV showed similar results.

5. **Cutoff 2 fallback**: Both GLMs fall back to zeros at cutoff 2 (no usable training data), which artificially improves their MAE on sparse targets but is not a meaningful model.

### Comparison with Baselines (Ranked by MAE Goals on cutoffs 3–6)

| Rank | Model | MAE Goals | MAE Assists | RMSE Goals | RMSE Assists | Poisson Dev Goals | Poisson Dev Assists |
|------|-------|-----------|-------------|------------|--------------|-------------------|---------------------|
| 1 | **Recent Form** | **0.1437** | **0.1062** | 0.3488 | 0.2806 | **0.0905** | **0.0621** |
| 2 | **Player Mean** | 0.1495 | 0.1089 | 0.3286 | 0.2563 | 0.2753 | 0.1482 |
| 3 | **Poisson GLM** | 0.1497 | 0.1069 | **0.3108** | **0.2380** | 0.4238 | 0.3219 |
| 4 | Global Mean | 0.1561 | 0.1103 | 0.3125 | 0.2385 | 0.4361 | 0.3255 |
| 5 | Minutes-Adjusted | 0.1791 | 0.1318 | 0.3614 | 0.2908 | 0.2938 | 0.1574 |
| 6 | **NegBin GLM** | **0.2039** | **0.1483** | **0.6209** | **1.1566** | **0.5063** | **0.3838** |

---

## 7. Convergence and Fallback Behavior

| Cutoff | Poisson GLM Fallback | NB Fallback | Notes |
|--------|---------------------|-------------|-------|
| 2 | Yes (constant features) | Yes (constant features) | Both predict zeros |
| 3 | No | No | 276 training samples (MW2) |
| 4 | No | No | 552 training samples (MW2-3) |
| 5 | No | No | 825 training samples (MW2-4) |
| 6 | No | No | 1101 training samples (MW2-5) |
| 7–10 | No | Not evaluated | — |

**Fallback counts (cutoffs 2–10):**
- Poisson GLM: 1/9 cutoffs (cutoff 2)
- Negative Binomial: 1/4 evaluated cutoffs (cutoff 2), 0/4 for cutoffs 3–6

---

## 8. Limitations

| Limitation | Impact |
|------------|--------|
| **Single season (2015/16)** | No cross-season validation; cannot assess generalization |
| **Cutoffs 3–6 for full comparison** | Only 1,105 observations for 3-model comparison; 2,480 for Poisson+baselines |
| **No meaningful overdispersion** | NB not expected to help; data near equidispersed |
| **NB convergence issues** | High RMSE at some cutoffs suggests instability; extreme predictions |
| **Fixed α=1.0** | Not tuned per cutoff; chronological CV showed similar results |
| **No opponent strength** | Only goals conceded used; no xGA available |
| **No injury/rotation signals** | Days rest is a weak proxy |
| **Sparse targets** | 92% zero goals, 95% zero assists; hard to model |
| **Matchweek 1 features constant** | First matchweek cannot be used for training (no history) |

---

## 9. Conclusion

**The Negative Binomial GLM does not improve predictions** on this dataset. The data does not exhibit strong overdispersion (Pearson dispersion ≈ 0.95 goals, 0.67 assists), so the Poisson assumption is adequate. The additional complexity of the Negative Binomial model adds estimation variance without reducing bias, resulting in worse performance on all metrics.

**The best model remains the simple Recent Form baseline** (last 5 matches), followed by Player Mean. The Poisson GLM provides no meaningful improvement over these simple baselines on this single-season, sparse dataset.

**Recommendation**: Do not pursue more complex count models (Zero-Inflated, Hurdle) without evidence of overdispersion. Instead, invest in better features (opponent xGA, team form, player age, rolling xG/xA) or consider that for this sparse, zero-inflated single-season data, simple historical averages are hard to beat.

---

## 10. Reproducibility

All code in `src/models/glm.py` and `tests/test_glm.py`. Key commands:

```bash
# Run all tests
.venv/bin/python3 -m pytest tests/ -v

# Run GLM tests only
.venv/bin/python3 -m pytest tests/test_glm.py -v
```

All 177 tests pass. Random seed fixed at 42 for deterministic behavior.