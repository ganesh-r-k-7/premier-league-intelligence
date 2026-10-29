"""Tests for Phase 8.4 data contracts and integration validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.data_contracts import (
    ValidationResult,
    DatasetReport,
    EXPECTED_SCHEMAS,
    _check_required_columns,
    _check_key_uniqueness,
    _check_no_null_in_keys,
    _check_numeric_columns,
    _check_source_separation,
    _check_season_coverage,
    _check_missing_values,
    validate_dataset,
    load_and_validate,
    validate_all_datasets,
    check_season_completeness,
    check_temporal_integrity,
    validate_identity_mappings,
    check_dashboard_source_consistency,
    check_evaluation_integrity,
    run_full_integration_validation,
    EXPECTED_SEASONS,
    EXPECTED_MATCHES_PER_SEASON,
)


class TestValidationHelpers:
    """Tests for individual validation helper functions."""

    def test_check_required_columns_pass(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _check_required_columns(df, ["a", "b"])
        assert result.passed is True
        assert result.details["missing"] == []

    def test_check_required_columns_fail(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = _check_required_columns(df, ["a", "b"])
        assert result.passed is False
        assert "b" in result.details["missing"]

    def test_check_key_uniqueness_pass(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        result = _check_key_uniqueness(df, ["id"])
        assert result.passed is True
        assert result.details["duplicate_count"] == 0

    def test_check_key_uniqueness_fail(self):
        df = pd.DataFrame({"id": [1, 1, 2], "val": ["a", "b", "c"]})
        result = _check_key_uniqueness(df, ["id"])
        assert result.passed is False
        assert result.details["duplicate_count"] == 1

    def test_check_key_uniqueness_allowed_duplicates(self):
        df = pd.DataFrame({"id": [1, 1, 2], "val": ["a", "b", "c"]})
        result = _check_key_uniqueness(df, ["id"], allow_duplicates=True)
        assert result.passed is True
        assert result.details["allowed"] is True
        assert result.details["duplicate_count"] == 1

    def test_check_no_null_in_keys_pass(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        result = _check_no_null_in_keys(df, ["id"])
        assert result.passed is True

    def test_check_no_null_in_keys_fail(self):
        df = pd.DataFrame({"id": [1, None, 3], "val": ["a", "b", "c"]})
        result = _check_no_null_in_keys(df, ["id"])
        assert result.passed is False
        assert result.details["null_counts"]["id"] == 1

    def test_check_numeric_columns_pass(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        result = _check_numeric_columns(df, ["a", "b"])
        assert result.passed is True

    def test_check_numeric_columns_fail_dtype(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        result = _check_numeric_columns(df, ["a"])
        assert result.passed is False
        assert "expected numeric" in result.details["issues"][0]

    def test_check_numeric_columns_fail_inf(self):
        df = pd.DataFrame({"a": [1.0, float("inf"), 3.0]})
        result = _check_numeric_columns(df, ["a"])
        assert result.passed is False
        assert "infinite values" in result.details["issues"][0]

    def test_check_source_separation(self):
        df = pd.DataFrame({"source": ["A", "A", "B"], "val": [1, 2, 3]})
        result = _check_source_separation(df, "source")
        assert result.passed is True
        assert "A" in result.details["sources"]
        assert "B" in result.details["sources"]
        assert result.details["source_counts"]["A"] == 2

    def test_check_season_coverage(self):
        df = pd.DataFrame({
            "source": ["A", "A", "B"],
            "season": ["2020/21", "2021/22", "2020/21"],
            "val": [1, 2, 3]
        })
        result = _check_season_coverage(df, "season", "source")
        assert result["total_seasons"] == 2
        assert result["total_rows"] == 3
        assert len(result["by_source_season"]) == 3

    def test_check_missing_values(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, 3], "c": [1, 2, 3]})
        result = _check_missing_values(df)
        assert result["a"] == 1
        assert result["b"] == 1
        assert "c" not in result


class TestExpectedSchemas:
    """Tests that expected schemas are well-formed."""

    def test_all_schemas_have_required_fields(self):
        for name, schema in EXPECTED_SCHEMAS.items():
            assert "required_columns" in schema
            assert "key_columns" in schema
            assert "numeric_columns" in schema
            assert isinstance(schema["required_columns"], list)
            assert isinstance(schema["key_columns"], list)
            assert isinstance(schema["numeric_columns"], list)

    def test_expected_seasons_format(self):
        # StatsBomb uses 2015/2016 format
        assert "2015/2016" in EXPECTED_SEASONS["StatsBomb"]
        # OpenFootball and FPL use 2015/16 format
        assert "2015/16" in EXPECTED_SEASONS["OpenFootball"]
        assert "2016/17" in EXPECTED_SEASONS["FPL"]


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_validate_mock_dataset(self):
        """Test validation with a mock dataset matching schema."""
        # Create a mock matches-like dataset
        df = pd.DataFrame({
            "match_id": [1, 2, 3],
            "season_id": [2020, 2020, 2021],
            "season_name": ["2020/21", "2020/21", "2021/22"],
            "competition": ["PL", "PL", "PL"],
            "match_date": pd.to_datetime(["2020-08-01", "2020-08-02", "2021-08-01"]),
            "match_time": ["15:00", "17:30", "15:00"],
            "match_week": [None, None, None],
            "round": ["1", "1", "1"],
            "home_team_id": [10, 20, 10],
            "home_team_name": ["Team A", "Team B", "Team A"],
            "away_team_id": [20, 30, 30],
            "away_team_name": ["Team B", "Team C", "Team C"],
            "home_score_ft": [2, 1, 3],
            "away_score_ft": [1, 1, 0],
            "home_score_ht": [1, 0, 2],
            "away_score_ht": [0, 1, 0],
            "source": ["test", "test", "test"],
            "source_url": ["url", "url", "url"],
            "source_license": ["MIT", "MIT", "MIT"],
            "raw_file": ["f1", "f2", "f3"],
            "raw_sha256": ["abc", "def", "ghi"],
            "ingested_at": ["2024-01-01", "2024-01-01", "2024-01-01"],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df.to_parquet(path)
            report = validate_dataset("matches", df, path)

            assert report.dataset == "matches"
            assert report.shape == (3, 22)
            assert report.passed is True
            assert len(report.issues) == 0


class TestIntegrationValidation:
    """Integration tests for the full validation pipeline."""

    def test_validate_all_datasets_returns_dict(self):
        """Test that validate_all_datasets returns a dict of reports."""
        reports = validate_all_datasets()
        assert isinstance(reports, dict)
        assert len(reports) > 0
        # All expected datasets should be present
        for name in EXPECTED_SCHEMAS.keys():
            assert name in reports

    def test_run_full_integration_validation(self):
        """Test the full integration validation runs without error."""
        result = run_full_integration_validation()
        assert "dataset_reports" in result
        assert "all_checks" in result
        assert "summary" in result
        assert result["summary"]["total_checks"] > 0
        assert 0 <= result["summary"]["pass_rate"] <= 1


class TestSeasonCompleteness:
    """Tests for season completeness checks."""

    def test_check_season_completeness_structure(self):
        """Test that season completeness check returns expected structure."""
        reports = validate_all_datasets()
        results = check_season_completeness(reports)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, ValidationResult)
            assert r.dataset in ["matches", "fpl_player_season", "sb_player_season"]


class TestTemporalIntegrity:
    """Tests for temporal integrity checks."""

    def test_check_temporal_integrity_structure(self):
        """Test that temporal integrity check returns expected structure."""
        reports = validate_all_datasets()
        results = check_temporal_integrity(reports)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, ValidationResult)
            assert r.dataset == "matches"


class TestIdentityMappings:
    """Tests for player identity mapping safeguards."""

    def test_validate_identity_mappings(self):
        """Test identity mapping validation runs."""
        results = validate_identity_mappings()
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, ValidationResult)
            assert r.dataset in ["player_identity_review", "canonical_identity"]


class TestDashboardConsistency:
    """Tests for dashboard source consistency checks."""

    def test_check_dashboard_source_consistency(self):
        """Test dashboard consistency check runs."""
        results = check_dashboard_source_consistency()
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, ValidationResult)
            assert r.dataset == "dashboard"


class TestEvaluationIntegrity:
    """Tests for evaluation integrity checks."""

    def test_check_evaluation_integrity(self):
        """Test evaluation integrity check runs."""
        results = check_evaluation_integrity()
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, ValidationResult)
            assert r.dataset == "evaluation"


class TestDataContractDeterminism:
    """Tests that validation is deterministic."""

    def test_validate_dataset_deterministic(self):
        """Repeated validation of same data gives same results."""
        df = pd.DataFrame({
            "match_id": [1, 2],
            "season_id": [2020, 2020],
            "season_name": ["2020/21", "2020/21"],
            "competition": ["PL", "PL"],
            "match_date": pd.to_datetime(["2020-08-01", "2020-08-02"]),
            "match_time": ["15:00", "17:30"],
            "match_week": [None, None],
            "round": ["1", "1"],
            "home_team_id": [10, 20],
            "home_team_name": ["A", "B"],
            "away_team_id": [20, 30],
            "away_team_name": ["B", "C"],
            "home_score_ft": [2, 1],
            "away_score_ft": [1, 1],
            "home_score_ht": [1, 0],
            "away_score_ht": [0, 1],
            "source": ["test", "test"],
            "source_url": ["url", "url"],
            "source_license": ["MIT", "MIT"],
            "raw_file": ["f1", "f2"],
            "raw_sha256": ["abc", "def"],
            "ingested_at": ["2024-01-01", "2024-01-01"],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df.to_parquet(path)

            r1 = validate_dataset("matches", df, path)
            r2 = validate_dataset("matches", df, path)

            assert r1.passed == r2.passed
            assert len(r1.checks) == len(r2.checks)
            for c1, c2 in zip(r1.checks, r2.checks):
                assert c1.passed == c2.passed
                assert c1.check == c2.check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])