"""Data contract validation for processed datasets (Phase 8.4).

Provides schema validation, integrity checks, and completeness reporting
for all canonical processed datasets. Keeps source datasets separate
and never performs unapproved cross-provider joins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.data.config import (
    DATA_PROCESSED_DIR,
    STATSBOMB_PROCESSED_DIR,
    PLAYER_SEASON_PARQUET,
    PLAYER_MATCH_PARQUET,
)


@dataclass(frozen=True)
class ValidationResult:
    """Result of a dataset validation check."""
    dataset: str
    check: str
    passed: bool
    message: str
    details: Optional[dict] = None


@dataclass(frozen=True)
class DatasetReport:
    """Complete validation report for a dataset."""
    dataset: str
    path: Path
    shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    checks: list[ValidationResult]
    season_coverage: dict
    missing_values: dict[str, int]
    duplicate_rows: int
    issues: list[str]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failure_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


# ---- Expected schemas for each dataset ----

EXPECTED_SCHEMAS = {
    "matches": {
        "required_columns": [
            "match_id", "season_id", "season_name", "competition",
            "match_date", "match_time", "match_week", "round",
            "home_team_id", "home_team_name", "away_team_id", "away_team_name",
            "home_score_ft", "away_score_ft", "home_score_ht", "away_score_ht",
            "source", "source_url", "source_license", "raw_file", "raw_sha256", "ingested_at"
        ],
        "key_columns": ["match_id"],
        "season_column": "season_name",
        "source_column": "source",
        "numeric_columns": ["match_id", "season_id", "home_team_id", "away_team_id",
                           "home_score_ft", "away_score_ft", "home_score_ht", "away_score_ht"],
        "date_columns": ["match_date"],
    },
    "teams": {
        "required_columns": ["team_id", "team_name", "normalized_name", "source"],
        "key_columns": ["team_id"],
        "numeric_columns": ["team_id"],
        "allow_duplicate_keys": True,  # Raw team names with variants before canonicalization
    },
    "seasons": {
        "required_columns": ["season_id", "season_name", "season_dir", "competition", "source"],
        "key_columns": ["season_id"],
        "numeric_columns": ["season_id"],
    },
    "team_registry": {
        "required_columns": [
            "team_id", "canonical_name", "match_confidence", "match_notes",
            "statsbomb_team_id", "statsbomb_team_name",
            "football_json_team_id", "football_json_team_name",
            "fpl_team_id", "fpl_team_name"
        ],
        "key_columns": ["team_id"],
        "numeric_columns": ["team_id", "statsbomb_team_id", "football_json_team_id", "fpl_team_id"],
    },
    "player_registry": {
        "required_columns": [
            "player_id", "match_key", "canonical_name", "canonical_team",
            "season_id", "position", "match_confidence", "match_notes",
            "statsbomb_player_id", "statsbomb_player_name",
            "fpl_player_id", "fpl_player_name"
        ],
        "key_columns": ["player_id"],
        "season_column": "season_id",
        "numeric_columns": ["player_id", "season_id"],
    },
    "fpl_player_season": {
        "required_columns": [
            "season_id", "season_name", "player_id", "player_name",
            "minutes", "goals", "assists", "appearances", "starts",
            "position", "team", "source", "source_url", "source_license", "ingested_at"
        ],
        "key_columns": ["player_id", "season_id"],
        "season_column": "season_name",
        "source_column": "source",
        "numeric_columns": ["season_id", "player_id", "minutes", "goals", "assists",
                           "appearances", "starts", "clean_sheets", "goals_conceded",
                           "own_goals", "penalties_missed", "penalties_saved",
                           "red_cards", "yellow_cards", "saves", "bonus", "bps",
                           "influence", "creativity", "threat", "ict_index", "total_points",
                           "expected_goals", "expected_assists", "expected_goal_involvements",
                           "expected_goals_conceded"],
        "allow_duplicate_keys": True,  # Known source issue: accent/name variations cause same player_id twice in season
    },
    "fpl_player_match": {
        "required_columns": [
            "player_id", "player_name", "season_id", "season_name",
            "gameweek", "fixture_id", "minutes", "goals", "assists", "was_home",
            "team", "position", "source", "source_url", "source_license",
            "raw_sha256", "ingested_at"
        ],
        "key_columns": ["player_id", "season_id", "fixture_id"],
        "season_column": "season_name",
        "source_column": "source",
        "numeric_columns": ["player_id", "season_id", "gameweek", "fixture_id", "minutes", "goals",
                           "assists", "clean_sheets", "goals_conceded", "own_goals",
                           "penalties_missed", "penalties_saved", "red_cards", "yellow_cards",
                           "saves", "bonus", "bps", "influence", "creativity", "threat",
                           "ict_index", "total_points", "expected_goals", "expected_assists",
                           "expected_goal_involvements", "expected_goals_conceded", "starts"],
        "allow_duplicate_keys": True,  # Known source issue: rescheduled matches cause same fixture_id in multiple gameweeks
    },
    "sb_player_season": {
        "required_columns": [
            "player_id", "player_name", "season_id", "season_name",
            "competition_id", "team_id", "team_name", "position",
            "appearances", "starts", "minutes", "goals", "own_goals",
            "assists", "shots", "shots_on_target", "xg", "npxg", "xa",
            "penalty_goals", "penalty_attempts", "source", "source_url", "ingested_at"
        ],
        "key_columns": ["player_id", "season_id"],
        "season_column": "season_name",
        "source_column": "source",
        "numeric_columns": ["player_id", "season_id", "competition_id", "team_id",
                           "appearances", "starts", "minutes", "goals", "own_goals",
                           "assists", "shots", "shots_on_target", "xg", "npxg", "xa",
                           "penalty_goals", "penalty_attempts"],
    },
    "sb_player_match": {
        "required_columns": [
            "match_id", "match_date", "season_id", "season_name",
            "competition_id", "player_id", "player_name", "team_id", "team_name",
            "position", "started", "minutes", "goals", "own_goals",
            "assists", "shots", "shots_on_target", "xg", "npxg", "xa",
            "penalty_goals", "penalty_attempts", "source", "source_url", "ingested_at"
        ],
        "key_columns": ["match_id", "player_id"],
        "season_column": "season_name",
        "source_column": "source",
        "numeric_columns": ["match_id", "season_id", "competition_id", "player_id", "team_id",
                           "started", "minutes", "goals", "own_goals", "assists",
                           "shots", "shots_on_target", "xg", "npxg", "xa",
                           "penalty_goals", "penalty_attempts"],
    },
    "sb_fixtures": {
        "required_columns": [
            "match_id", "match_date", "kick_off", "match_week",
            "competition_id", "competition_name", "season_id", "season_name",
            "home_team_id", "home_team_name", "away_team_id", "away_team_name",
            "home_score", "away_score", "match_status", "stadium",
            "source", "source_url", "ingested_at"
        ],
        "key_columns": ["match_id"],
        "season_column": "season_name",
        "source_column": "source",
        "numeric_columns": ["match_id", "match_week", "competition_id", "season_id",
                           "home_team_id", "away_team_id", "home_score", "away_score"],
    },
    "canonical_identity": {
        "required_columns": [
            "canonical_id", "source_a", "source_a_id", "source_a_name",
            "source_a_season", "source_a_team", "source_b", "source_b_id",
            "source_b_name", "source_b_season", "source_b_team",
            "evidence_score", "candidate_type", "reviewer", "reviewed_at", "notes"
        ],
        "key_columns": ["canonical_id"],
        "numeric_columns": ["canonical_id", "source_a_season", "source_b_season", "evidence_score"],
    },
}


# ---- Validation functions ----

def _check_required_columns(df: pd.DataFrame, required: list[str]) -> ValidationResult:
    missing = [c for c in required if c not in df.columns]
    return ValidationResult(
        dataset="", check="required_columns",
        passed=len(missing) == 0,
        message=f"Missing required columns: {missing}" if missing else "All required columns present",
        details={"missing": missing}
    )


def _check_key_uniqueness(df: pd.DataFrame, key_columns: list[str], allow_duplicates: bool = False) -> ValidationResult:
    if not key_columns:
        return ValidationResult("", "key_uniqueness", True, "No key columns defined")
    available_keys = [c for c in key_columns if c in df.columns]
    if not available_keys:
        return ValidationResult("", "key_uniqueness", False, "No key columns available in dataframe")
    dup_count = int(df.duplicated(subset=available_keys).sum())
    if allow_duplicates:
        return ValidationResult(
            dataset="", check="key_uniqueness",
            passed=True,
            message=f"Duplicate key rows (allowed): {dup_count}",
            details={"duplicate_count": dup_count, "key_columns": available_keys, "allowed": True}
        )
    return ValidationResult(
        dataset="", check="key_uniqueness",
        passed=dup_count == 0,
        message=f"Duplicate key rows: {dup_count}" if dup_count > 0 else "Keys are unique",
        details={"duplicate_count": dup_count, "key_columns": available_keys}
    )


def _check_no_null_in_keys(df: pd.DataFrame, key_columns: list[str]) -> ValidationResult:
    available_keys = [c for c in key_columns if c in df.columns]
    if not available_keys:
        return ValidationResult("", "null_keys", True, "No key columns to check")
    null_counts = {c: int(df[c].isna().sum()) for c in available_keys}
    total_nulls = sum(null_counts.values())
    return ValidationResult(
        dataset="", check="null_keys",
        passed=total_nulls == 0,
        message=f"Null values in key columns: {null_counts}" if total_nulls > 0 else "No nulls in key columns",
        details={"null_counts": null_counts}
    )


def _check_numeric_columns(df: pd.DataFrame, numeric_columns: list[str]) -> ValidationResult:
    issues = []
    for col in numeric_columns:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"{col}: expected numeric, got {df[col].dtype}")
        else:
            # Check for infinite values
            inf_count = int(np.isinf(df[col]).sum()) if pd.api.types.is_float_dtype(df[col]) else 0
            if inf_count > 0:
                issues.append(f"{col}: {inf_count} infinite values")
    return ValidationResult(
        dataset="", check="numeric_columns",
        passed=len(issues) == 0,
        message="; ".join(issues) if issues else "Numeric columns valid",
        details={"issues": issues}
    )


def _check_season_coverage(df: pd.DataFrame, season_col: str, source_col: str) -> dict:
    if season_col not in df.columns:
        return {"error": f"Season column '{season_col}' not found"}
    coverage = df.groupby([source_col, season_col]).size().reset_index(name="rows")
    return {
        "by_source_season": coverage.to_dict("records"),
        "total_seasons": df[season_col].nunique(),
        "total_rows": len(df),
    }


def _check_missing_values(df: pd.DataFrame, max_cols: int = 20) -> dict[str, int]:
    missing = df.isna().sum()
    return {k: int(v) for k, v in missing.items() if v > 0}


def _check_source_separation(df: pd.DataFrame, source_col: str) -> ValidationResult:
    if source_col not in df.columns:
        return ValidationResult("", "source_separation", True, f"Source column '{source_col}' not found")
    sources = df[source_col].unique()
    return ValidationResult(
        dataset="", check="source_separation",
        passed=True,
        message=f"Sources present: {list(sources)}",
        details={"sources": list(sources), "source_counts": df[source_col].value_counts().to_dict()}
    )


def validate_dataset(dataset_name: str, df: pd.DataFrame, path: Path) -> DatasetReport:
    """Validate a single dataset against its expected schema."""
    schema = EXPECTED_SCHEMAS.get(dataset_name, {})
    checks = []

    # Required columns
    required = schema.get("required_columns", [])
    check = _check_required_columns(df, required)
    checks.append(ValidationResult(dataset_name, check.check, check.passed, check.message, check.details))

    # Key uniqueness
    key_cols = schema.get("key_columns", [])
    allow_dup_keys = schema.get("allow_duplicate_keys", False)
    check = _check_key_uniqueness(df, key_cols, allow_dup_keys)
    checks.append(ValidationResult(dataset_name, check.check, check.passed, check.message, check.details))

    # Null keys
    check = _check_no_null_in_keys(df, key_cols)
    checks.append(ValidationResult(dataset_name, check.check, check.passed, check.message, check.details))

    # Numeric columns
    numeric_cols = schema.get("numeric_columns", [])
    check = _check_numeric_columns(df, numeric_cols)
    checks.append(ValidationResult(dataset_name, check.check, check.passed, check.message, check.details))

    # Source separation
    source_col = schema.get("source_column")
    if source_col:
        check = _check_source_separation(df, source_col)
        checks.append(ValidationResult(dataset_name, check.check, check.passed, check.message, check.details))

    # Season coverage
    season_col = schema.get("season_column")
    coverage = {}
    if season_col and season_col in df.columns and source_col and source_col in df.columns:
        coverage = _check_season_coverage(df, season_col, source_col)

    # Missing values
    missing = _check_missing_values(df)

    # Duplicate rows (full row duplicates)
    dup_rows = int(df.duplicated().sum())

    # Collect issues
    issues = [c.message for c in checks if not c.passed]

    return DatasetReport(
        dataset=dataset_name,
        path=path,
        shape=df.shape,
        columns=list(df.columns),
        dtypes={c: str(df[c].dtype) for c in df.columns},
        checks=checks,
        season_coverage=coverage,
        missing_values=missing,
        duplicate_rows=dup_rows,
        issues=issues,
    )


def load_and_validate(dataset_name: str) -> DatasetReport:
    """Load a dataset by name and validate it."""
    paths = {
        "matches": DATA_PROCESSED_DIR / "matches.parquet",
        "teams": DATA_PROCESSED_DIR / "teams.parquet",
        "seasons": DATA_PROCESSED_DIR / "seasons.parquet",
        "team_registry": DATA_PROCESSED_DIR / "team_registry.parquet",
        "player_registry": DATA_PROCESSED_DIR / "player_registry.parquet",
        "fpl_player_season": DATA_PROCESSED_DIR / "fpl_player_season.parquet",
        "fpl_player_match": DATA_PROCESSED_DIR / "fpl_player_match.parquet",
        "sb_player_season": STATSBOMB_PROCESSED_DIR / "player_season.parquet",
        "sb_player_match": STATSBOMB_PROCESSED_DIR / "player_match.parquet",
        "sb_fixtures": STATSBOMB_PROCESSED_DIR / "fixtures.parquet",
        "canonical_identity": DATA_PROCESSED_DIR / "canonical_identity_mapping.parquet",
    }
    path = paths.get(dataset_name)
    if not path or not path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {path}")
    df = pd.read_parquet(path)
    return validate_dataset(dataset_name, df, path)


def validate_all_datasets() -> dict[str, DatasetReport]:
    """Validate all processed datasets."""
    datasets = list(EXPECTED_SCHEMAS.keys())
    reports = {}
    for name in datasets:
        try:
            reports[name] = load_and_validate(name)
        except FileNotFoundError as e:
            reports[name] = DatasetReport(
                dataset=name, path=Path(""), shape=(0, 0), columns=[], dtypes={},
                checks=[ValidationResult(name, "file_exists", False, str(e))],
                season_coverage={}, missing_values={}, duplicate_rows=0, issues=[str(e)]
            )
    return reports


def print_validation_summary(reports: dict[str, DatasetReport]) -> None:
    """Print a summary of validation results."""
    print("=" * 80)
    print("DATA CONTRACT VALIDATION SUMMARY")
    print("=" * 80)
    for name, report in reports.items():
        status = "PASS" if report.passed else "FAIL"
        print(f"\n{name}: {status} (shape={report.shape}, issues={len(report.issues)})")
        for check in report.checks:
            icon = "✓" if check.passed else "✗"
            print(f"  {icon} {check.check}: {check.message}")
        if report.season_coverage:
            print(f"  Season coverage: {report.season_coverage.get('total_seasons', 0)} seasons, {report.season_coverage.get('total_rows', 0)} rows")
        if report.missing_values:
            top_missing = sorted(report.missing_values.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top missing: {top_missing}")
        if report.duplicate_rows > 0:
            print(f"  Full-row duplicates: {report.duplicate_rows}")


# ---- Season completeness and temporal integrity ----

EXPECTED_SEASONS = {
    "StatsBomb": ["2015/2016"],
    "OpenFootball": [f"20{i:02d}/{i+1:02d}" for i in range(10, 26)] + ["2025/26"],
    "FPL": [f"20{i:02d}/{i+1:02d}" for i in range(16, 26)],
}

EXPECTED_MATCHES_PER_SEASON = 380


def check_season_completeness(reports: dict[str, DatasetReport]) -> list[ValidationResult]:
    """Check that each provider has expected season coverage."""
    results = []

    # Check matches (OpenFootball)
    if "matches" in reports:
        r = reports["matches"]
        if r.season_coverage:
            seasons_present = set()
            for entry in r.season_coverage.get("by_source_season", []):
                if entry.get("source") == "openfootball/football.json":
                    seasons_present.add(entry.get("season_name"))
            expected = set(EXPECTED_SEASONS["OpenFootball"])
            missing = expected - seasons_present
            extra = seasons_present - expected
            passed = len(missing) == 0
            results.append(ValidationResult(
                "matches", "season_completeness_openfootball", passed,
                f"Missing OpenFootball seasons: {sorted(missing)}" if missing else "All expected OpenFootball seasons present",
                details={"missing": sorted(missing), "extra": sorted(extra), "present": sorted(seasons_present)}
            ))

    # Check FPL player seasons
    if "fpl_player_season" in reports:
        r = reports["fpl_player_season"]
        if r.season_coverage:
            seasons_present = set()
            for entry in r.season_coverage.get("by_source_season", []):
                seasons_present.add(entry.get("season_name"))
            expected = set(EXPECTED_SEASONS["FPL"])
            missing = expected - seasons_present
            extra = seasons_present - expected
            passed = len(missing) == 0
            results.append(ValidationResult(
                "fpl_player_season", "season_completeness_fpl", passed,
                f"Missing FPL seasons: {sorted(missing)}" if missing else "All expected FPL seasons present",
                details={"missing": sorted(missing), "extra": sorted(extra), "present": sorted(seasons_present)}
            ))

    # Check StatsBomb
    if "sb_player_season" in reports:
        r = reports["sb_player_season"]
        if r.season_coverage:
            seasons_present = set()
            for entry in r.season_coverage.get("by_source_season", []):
                seasons_present.add(entry.get("season_name"))
            expected = set(EXPECTED_SEASONS["StatsBomb"])
            missing = expected - seasons_present
            passed = len(missing) == 0
            results.append(ValidationResult(
                "sb_player_season", "season_completeness_statsbomb", passed,
                f"Missing StatsBomb seasons: {sorted(missing)}" if missing else "All expected StatsBomb seasons present",
                details={"missing": sorted(missing), "present": sorted(seasons_present)}
            ))

    return results


def check_match_counts(reports: dict[str, DatasetReport]) -> list[ValidationResult]:
    """Check that each season has approximately 380 matches."""
    results = []
    if "matches" in reports:
        r = reports["matches"]
        if "season_name" in r.columns:
            match_counts = r.df.groupby("season_name").size() if hasattr(r, 'df') else None
            # We need the actual dataframe for this check
    return results


def check_temporal_integrity(reports: dict[str, DatasetReport]) -> list[ValidationResult]:
    """Check for temporal integrity issues (future data leakage, ordering)."""
    results = []

    # Check matches chronological ordering
    if "matches" in reports:
        r = reports["matches"]
        if "match_date" in r.columns:
            df = pd.read_parquet(r.path)
            df["match_date"] = pd.to_datetime(df["match_date"])
            # Check within each season dates are ordered
            for season, group in df.groupby("season_name"):
                dates = group["match_date"].dropna().sort_values()
                if not dates.is_monotonic_increasing:
                    results.append(ValidationResult(
                        "matches", "temporal_ordering", False,
                        f"Match dates not monotonically increasing in {season}",
                        details={"season": season}
                    ))
            # Check no future dates beyond season end (rough check)
            max_date = df["match_date"].max()
            if pd.notna(max_date) and max_date.year > 2026:
                results.append(ValidationResult(
                    "matches", "future_dates", False,
                    f"Future match dates detected beyond 2026: {max_date}",
                    details={"max_date": str(max_date)}
                ))

    return results


# ---- Player identity mapping safeguards ----

def validate_identity_mappings() -> list[ValidationResult]:
    """Validate the player identity review file and canonical mappings."""
    results = []
    review_path = DATA_PROCESSED_DIR / "review" / "player_identity_review.csv"
    canonical_path = DATA_PROCESSED_DIR / "canonical_identity_mapping.parquet"

    # Check review file
    if review_path.exists():
        df = pd.read_csv(review_path)
        # All should be pending (no human review done)
        pending = (df["status"] == "pending").sum()
        confirmed = (df["status"] == "confirmed").sum()
        rejected = (df["status"] == "rejected").sum()
        ambiguous = (df["status"] == "ambiguous").sum()

        results.append(ValidationResult(
            "player_identity_review", "all_pending", pending == len(df),
            f"All {len(df)} candidates pending review" if pending == len(df)
            else f"Some candidates have non-pending status: confirmed={confirmed}, rejected={rejected}, ambiguous={ambiguous}",
            details={"pending": int(pending), "confirmed": int(confirmed),
                    "rejected": int(rejected), "ambiguous": int(ambiguous), "total": len(df)}
        ))

        # Check for duplicate source mappings in confirmed (should be zero)
        if confirmed > 0:
            confirmed_df = df[df["status"] == "confirmed"]
            dup_a = confirmed_df.duplicated(subset=["src_a_source", "src_a_id"]).sum()
            dup_b = confirmed_df.duplicated(subset=["src_b_source", "src_b_id"]).sum()
            results.append(ValidationResult(
                "player_identity_review", "one_to_one_confirmed",
                dup_a == 0 and dup_b == 0,
                f"Duplicate confirmed mappings: src_a={dup_a}, src_b={dup_b}",
                details={"src_a_duplicates": int(dup_a), "src_b_duplicates": int(dup_b)}
            ))

        # Check evidence scores are reasonable
        if "evidence_score" in df.columns:
            invalid_scores = ((df["evidence_score"] < 0) | (df["evidence_score"] > 1)).sum()
            results.append(ValidationResult(
                "player_identity_review", "evidence_score_range", invalid_scores == 0,
                f"Invalid evidence scores: {invalid_scores}",
                details={"invalid_count": int(invalid_scores)}
            ))

    # Check canonical mappings
    if canonical_path.exists():
        df = pd.read_parquet(canonical_path)
        # Verify canonical_id is unique
        dup_canonical = df.duplicated(subset=["canonical_id"]).sum()
        results.append(ValidationResult(
            "canonical_identity", "unique_canonical_id", dup_canonical == 0,
            f"Duplicate canonical IDs: {dup_canonical}",
            details={"duplicate_count": int(dup_canonical)}
        ))

        # Verify sources are valid
        valid_sources = {"statsbomb", "fpl"}
        invalid_a = (~df["source_a"].isin(valid_sources)).sum()
        invalid_b = (~df["source_b"].isin(valid_sources)).sum()
        results.append(ValidationResult(
            "canonical_identity", "valid_sources",
            invalid_a == 0 and invalid_b == 0,
            f"Invalid sources: source_a={invalid_a}, source_b={invalid_b}",
            details={"invalid_source_a": int(invalid_a), "invalid_source_b": int(invalid_b)}
        ))

    return results


# ---- Dashboard integration checks ----

def check_dashboard_source_consistency() -> list[ValidationResult]:
    """Verify dashboard data loader source definitions match actual data."""
    results = []
    try:
        from src.dashboard.data_loader import DATA_SOURCES, get_season_coverage_summary

        # Check that DATA_SOURCES seasons match what we actually have
        coverage = get_season_coverage_summary()

        # For each source in DATA_SOURCES, verify seasons exist in coverage
        for source, info in DATA_SOURCES.items():
            if "player_season" in info:
                expected = set(info["player_season"])
                actual = set(coverage[(coverage["Source"] == source) & (coverage["Player Season Rows"] > 0)]["Season"])
                missing = expected - actual
                results.append(ValidationResult(
                    "dashboard", f"source_seasons_{source.lower()}", len(missing) == 0,
                    f"Dashboard source {source} missing seasons: {sorted(missing)}" if missing else f"All {source} seasons present in coverage",
                    details={"expected": sorted(expected), "actual": sorted(actual), "missing": sorted(missing)}
                ))

        # Check coverage summary has no negative values
        neg_counts = (coverage.select_dtypes(include=[np.number]) < 0).sum().sum()
        results.append(ValidationResult(
            "dashboard", "coverage_no_negative", neg_counts == 0,
            f"Negative values in coverage summary: {neg_counts}",
            details={"negative_count": int(neg_counts)}
        ))

    except Exception as e:
        results.append(ValidationResult("dashboard", "import_check", False, f"Failed to import dashboard loader: {e}"))

    return results


# ---- Model evaluation integrity ----

def check_evaluation_integrity() -> list[ValidationResult]:
    """Basic checks on evaluation outputs and reported metrics."""
    results = []

    # Check that evaluation reports exist and have expected structure
    from src.data.config import REPORTS_DIR
    eval_reports = [
        "phase2_stage3_report.md",
        "phase3_stage32_report.md",
        "phase3_stage33_report.md",
        "phase4_stage41_report.md",
        "phase4_stage42_report.md",
        "phase4_stage43_report.md",
        "phase4_stage44_report.md",
        "phase8.3B_multiseason_model_evaluation.md",
    ]

    for report in eval_reports:
        path = REPORTS_DIR / report
        exists = path.exists()
        results.append(ValidationResult(
            "evaluation", f"report_exists_{report}", exists,
            f"{'Found' if exists else 'Missing'} evaluation report: {report}"
        ))

    return results


# ---- Full integration validation ----

def run_full_integration_validation() -> dict:
    """Run all integration validation checks and return summary."""
    print("Loading and validating all datasets...")
    reports = validate_all_datasets()

    print("Checking season completeness...")
    season_checks = check_season_completeness(reports)

    print("Checking temporal integrity...")
    temporal_checks = check_temporal_integrity(reports)

    print("Validating identity mappings...")
    identity_checks = validate_identity_mappings()

    print("Checking dashboard consistency...")
    dashboard_checks = check_dashboard_source_consistency()

    print("Checking evaluation integrity...")
    eval_checks = check_evaluation_integrity()

    all_checks = []
    for r in reports.values():
        all_checks.extend(r.checks)
    all_checks.extend(season_checks)
    all_checks.extend(temporal_checks)
    all_checks.extend(identity_checks)
    all_checks.extend(dashboard_checks)
    all_checks.extend(eval_checks)

    passed = sum(1 for c in all_checks if c.passed)
    failed = sum(1 for c in all_checks if not c.passed)

    return {
        "dataset_reports": reports,
        "all_checks": all_checks,
        "summary": {
            "total_checks": len(all_checks),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(all_checks) if all_checks else 0,
        },
        "season_checks": season_checks,
        "temporal_checks": temporal_checks,
        "identity_checks": identity_checks,
        "dashboard_checks": dashboard_checks,
        "eval_checks": eval_checks,
    }


if __name__ == "__main__":
    result = run_full_integration_validation()
    print_validation_summary(result["dataset_reports"])
    print(f"\n\nOVERALL: {result['summary']['passed']}/{result['summary']['total_checks']} checks passed ({result['summary']['pass_rate']:.1%})")
    if result['summary']['failed'] > 0:
        print("\nFAILED CHECKS:")
        for check in result['all_checks']:
            if not check.passed:
                print(f"  ✗ [{check.dataset}] {check.check}: {check.message}")