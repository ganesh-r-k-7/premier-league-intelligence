# Phase 8.5C: Identity Review and Mapping Validation Report

**Date:** 2026-10-29  
**Status:** Complete  
**Git Commit:** TBD

---

## Summary

Phase 8.5C executes the human-review workflow for player identity candidates and validates the resulting canonical mapping integration. The work verifies the review tool, applies confirmed mappings, and validates downstream integration safety.

---

## 1. Review Tool Verification

### 1.1 Streamlit Tool Functionality
- **Launch:** `streamlit run src/dashboard/app.py` → Navigate to "🔍 Identity Review"
- **Filters verified:** candidate_type, evidence_score range, status, FPL ID reuse severity, season, team
- **Detail view:** Expands to show both source records, evidence scores, normalized names, FPL ID reuse severity, quality flags, external reference links
- **Bulk actions:** Multi-select → Confirm/Reject/Ambiguous with shared notes (explicit selection required)
- **External links:** Transfermarkt/Wikipedia search buttons generate search URLs (no live checks, no scraping)
- **Auto-save:** Status changes persist immediately to CSV; session resumable across restarts
- **Export:** Download reviewed CSV; "Generate Canonical Mapping" button creates `canonical_identity_mapping.parquet`

### 1.2 Persistence Test
- **Test:** Applied 2 confirmed + 1 rejected + 1 ambiguous decisions → saved CSV → reloaded → decisions persisted correctly
- **Result:** All 4 decisions (2 confirmed, 1 rejected, 1 ambiguous) retained with reviewer notes, timestamps, external URLs

---

## 2. Review File State (v2)

**File:** `data/processed/review/player_identity_review_v2.csv`  
**Total candidates:** 1,363 (all `status: "pending"`)

| Candidate Type | Count | Evidence Score Range | Notes |
|----------------|-------|---------------------|-------|
| Cross-provider (SB 2015/16 ↔ FPL 2016/17) | 296 | 0.6 only | Name exact + season adjacent; FPL 2016 lacks team/position |
| Same-provider continuity (FPL 2020–2025 ↔ prior) | 777 | 0.8–1.0 | 698 at 1.0, 75 at 0.9, 4 at 0.8 |
| Unresolved | 290 | 0.0 | No prior season with same canonical name + team |

**All 1,363 candidates remain `status: "pending"`** — no human review performed yet.

---

## 3. Validation & Conflict Detection

### 3.1 Validation Results (v2 review file)
```
Total rows: 1,363
Status counts: {'pending': 1363}
Candidate types: {'same_provider_continuity': 777, 'cross_provider': 296, 'unresolved': 290}
Missing required: 0
Duplicate mappings: 0
Invalid status: 0
Invalid candidate_type: 0
Invalid external URLs: 0
FPL ID reuse warnings: 192 (both sources High/Critical severity)
Transitive conflicts: 0
One-to-one violations (confirmed): 0
```

### 3.2 Conflict Detection Tests

| Test | Result |
|------|--------|
| Same src_a (StatsBomb) → multiple src_b (FPL) in same season | ✅ Detected: `src_a_multiple: 1` |
| Invalid external URL format | ✅ Detected: `invalid_external_urls: 1` |
| Transitive conflict (A↔B confirmed, B↔C confirmed, A↔C not) | ✅ Logic implemented (0 found in test data) |
| Invalid status values | ✅ Detected |
| Invalid candidate_type values | ✅ Detected |

---

## 4. Mapping Application

### 4.1 Apply Confirmed Mappings
```bash
python -m src.data.player_identity_curation apply data/processed/review/player_identity_review_v2.csv
```

**Behavior (verified):**
- Only processes `status == "confirmed"` rows
- Runs validation first; aborts on errors
- Produces `data/processed/canonical_identity_mapping.parquet`
- Does NOT modify `player_registry.parquet`
- Returns empty DataFrame if no confirmed mappings

### 4.2 Canonical Identity Schema (Season-Scoped)
```python
{
    "canonical_id": int,          # MD5 hash of source_a|id|season|source_b|id|season
    "source_a": str,
    "source_a_id": str,
    "source_a_season": int,
    "source_a_name": str,
    "source_a_team": str,
    "source_b": str,
    "source_b_id": str,
    "source_b_season": int,
    "source_b_name": str,
    "source_b_team": str,
    "evidence_score": float,
    "candidate_type": str,
    "reviewer": str,
    "reviewed_at": str,
    "notes": str,
    "external_reference_url": str,
}
```

**Key Principle:** Season-scoped keys protect against FPL ID reuse. Each confirmed link is between two specific source records (provider + id + season), not between global provider IDs.

### 4.3 Test Application (2 confirmed mappings)
| Canonical ID | Source A | Source B | Evidence | Type |
|--------------|----------|----------|----------|------|
| 243520837824670 | StatsBomb 2988 (Memphis Depay, 2015/16) | FPL 258 (Memphis_Depay, 2016/17) | 0.6 | cross_provider |
| 122702808638845 | FPL 54 (Adam Lallana, 2020/21) | FPL 53 (Adam Lallana, 2021/22) | 1.0 | same_provider_continuity |

---

## 5. Downstream Integration Safety

### 5.1 Data Contract Validation
```
OVERALL: 79/79 checks passed (100.0%)
```
All 11 processed datasets pass schema, key uniqueness, null keys, numeric types, source separation, season coverage.

### 5.2 FPL ID Reuse Handling
- **Audit:** 866 element IDs scanned → 1 Critical (ID 748: 5 players in 2024/25), 372 High, 468 Medium
- **Review file:** `fpl_id_reuse_severity` column added (192 candidates with High/Critical on both sides)
- **Canonical mapping:** Season-scoped keys prevent ID reuse collisions
- **Unresolved flag:** ID 748 (2024/25) flagged with `quality_flag: "fpl_id_reuse_different_players"` — unresolvable without external data

### 5.3 Source Boundaries Preserved
- Canonical mapping links source records only — does NOT equate FPL metrics with StatsBomb metrics
- No unapproved cross-provider joins in downstream code
- StatsBomb 2015/16 behavior preserved (no regressions)

---

## 6. Test Results

| Test Suite | Tests | Status |
|------------|-------|--------|
| `test_player_identity_curation.py` | 31 | ✅ Pass |
| `test_data_contracts.py` | 24 | ✅ Pass |
| `test_dashboard.py` | 38 | ✅ Pass |
| **Total** | **93** | ✅ **Pass** |

**No regressions** in StatsBomb 2015/16, FPL models, or dashboard behavior.

---

## 7. Remaining Work

| Item | Status | Notes |
|------|--------|-------|
| Human review of 1,363 candidates | ⏳ Pending | 296 cross-provider, 777 same-provider, 290 unresolved |
| External lookup for 296 cross-provider | ⏳ Pending | Transfermarkt/Wikipedia for transfer history |
| Apply confirmed mappings | ⏳ Pending | Run after review complete |
| FPL element ID 748 resolution | ⏳ Pending | Requires external data (5 players in 2024/25) |

---

## 8. Reproducible Commands

```bash
# Generate review file v2 (with remediated data + enhanced columns)
python -m src.data.player_identity_curation generate --version 2

# Run FPL ID reuse audit
python -m src.data.fpl_id_audit

# Add severity to existing review file
python -m src.data.fpl_id_audit add-severity

# Validate review file
python -m src.data.player_identity_curation validate data/processed/review/player_identity_review_v2.csv

# Launch review tool (local-only)
streamlit run src/dashboard/app.py
# Navigate to "🔍 Identity Review" in sidebar

# Apply confirmed mappings
python -m src.data.player_identity_curation apply data/processed/review/player_identity_review_v2.csv

# Check review completeness
python -m src.data.player_identity_curation completeness data/processed/review/player_identity_review_v2.csv

# Run tests
python -m pytest tests/test_player_identity_curation.py tests/test_data_contracts.py -v

# Full test suite (core)
python -m pytest tests/ -k "not integration" -q
```

---

## 8. Phase 8.5C Exit Gate — MET

✅ Streamlit review tool verified (filtering, persistence, bulk actions, external links)  
✅ Review file v2 generated with all enhanced columns (1,363 candidates, all pending)  
✅ Validation suite detects conflicts, invalid URLs, FPL ID reuse, transitive conflicts  
✅ Mapping application produces season-scoped canonical IDs, preserves external refs  
✅ 93 tests pass (31 identity + 24 data contracts + 38 dashboard)  
✅ No regressions in StatsBomb 2015/16, FPL models, or dashboard  
✅ No fabricated decisions — all 1,363 candidates remain pending  
✅ Source boundaries preserved — no unapproved cross-provider joins  
✅ Documentation complete: report, updated PROJECT_PLAN.md, README.md  

---

## 9. Next Steps (Post-8.5C)

1. **Human review** of 1,363 candidates using Streamlit tool
2. **Apply confirmed mappings** → `canonical_identity_mapping.parquet`
3. **Optional:** OpenFootball xG estimation for match outcome models
4. **Optional:** Dashboard polish (season comparison charts, export)
5. **Phase 8.6** (if pursued): Market value data integration, Golden Boot prediction, scenario analysis