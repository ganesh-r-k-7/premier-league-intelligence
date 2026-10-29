"""Tests for Phase 8.3C player identity curation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.player_identity_curation import (
    CandidateMatch,
    _normalize_name_for_matching,
    _jaro_winkler_similarity,
    _find_cross_provider_candidates,
    _find_same_provider_continuity,
    generate_review_file,
    validate_mappings,
    apply_confirmed_mappings,
    _season_year,
)


class TestNameNormalization:
    """Tests for name normalization."""

    def test_fpl_underscores_converted(self):
        assert _normalize_name_for_matching("David_Ospina") == "david ospina"
        assert _normalize_name_for_matching("Mesut_Özil") == "mesut ozil"

    def test_accents_removed(self):
        assert _normalize_name_for_matching("Gnégnéri Yaya Touré") == "gnegneri yaya toure"
        assert _normalize_name_for_matching("Jürgen Klopp") == "jurgen klopp"
        assert _normalize_name_for_matching("André Silva") == "andre silva"

    def test_suffixes_removed(self):
        assert _normalize_name_for_matching("John Smith Jr.") == "john smith"
        assert _normalize_name_for_matching("John Smith Jr") == "john smith"
        assert _normalize_name_for_matching("John Smith III") == "john smith"

    def test_special_chars_removed(self):
        assert _normalize_name_for_matching("Jean-Pierre") == "jean pierre"
        assert _normalize_name_for_matching("O'Neill") == "oneill"
        assert _normalize_name_for_matching("St. John") == "st john"

    def test_whitespace_collapsed(self):
        assert _normalize_name_for_matching("  John   Smith  ") == "john smith"

    def test_case_insensitive(self):
        assert _normalize_name_for_matching("JOHN SMITH") == "john smith"

    def test_empty_input(self):
        assert _normalize_name_for_matching("") == ""
        assert _normalize_name_for_matching(None) == ""


class TestJaroWinklerSimilarity:
    """Tests for Jaro-Winkler similarity."""

    def test_identical_strings(self):
        assert _jaro_winkler_similarity("john smith", "john smith") == 1.0

    def test_completely_different(self):
        sim = _jaro_winkler_similarity("john smith", "jane doe")
        assert 0 <= sim < 0.5

    def test_similar_names(self):
        sim = _jaro_winkler_similarity("john smith", "jon smith")
        assert 0.8 < sim < 1.0


class TestSeasonYearMapping:
    """Tests for season_id to year mapping."""

    def test_statsbomb_season(self):
        assert _season_year(27) == 2015

    def test_fpl_seasons(self):
        assert _season_year(2016) == 2016
        assert _season_year(2023) == 2023
        assert _season_year(2025) == 2025

    def test_unknown_season(self):
        assert _season_year(999) is None


class TestCrossProviderCandidates:
    """Tests for cross-provider candidate generation."""

    @pytest.fixture
    def sb_data(self):
        return pd.DataFrame({
            "player_id": [1001, 1002],
            "player_name": ["Kevin De Bruyne", "Harry Kane"],
            "team_name": ["Manchester City", "Tottenham Hotspur"],
            "season_id": [27, 27],
            "position": ["Midfielder", "Forward"],
        })

    @pytest.fixture
    def fpl_data(self):
        return pd.DataFrame({
            "player_id": [1, 2, 3],
            "player_name": ["Kevin_De Bruyne", "Harry_Kane", "Mohamed_Salah"],
            "team": ["Man City", "Spurs", "Liverpool"],
            "season_id": [2016, 2016, 2016],
            "position": [None, None, None],
        })

    def test_exact_normalized_name_match(self, sb_data, fpl_data):
        candidates = _find_cross_provider_candidates(sb_data, fpl_data)
        # Should find 2 matches: Kevin De Bruyne and Harry Kane
        assert len(candidates) == 2
        names = {c.src_a_name for c in candidates}
        assert "Kevin De Bruyne" in names
        assert "Harry Kane" in names

    def test_season_adjacent_true(self, sb_data, fpl_data):
        candidates = _find_cross_provider_candidates(sb_data, fpl_data)
        for c in candidates:
            assert c.season_adjacent is True

    def test_evidence_score_with_name_only(self, sb_data, fpl_data):
        candidates = _find_cross_provider_candidates(sb_data, fpl_data)
        for c in candidates:
            # name exact (0.5) + team_match (0.3) + season adjacent (0.1) = 0.9
            # because test fixture teams canonicalize to same
            assert c.evidence_score == 0.9

    def test_no_match_for_different_name(self, sb_data, fpl_data):
        candidates = _find_cross_provider_candidates(sb_data, fpl_data)
        names_a = {c.src_a_name for c in candidates}
        assert "Mohamed Salah" not in names_a  # not in SB data


class TestSameProviderContinuity:
    """Tests for same-provider continuity candidate generation."""

    @pytest.fixture
    def fpl_data(self):
        return pd.DataFrame({
            "player_id": [1, 1, 2, 2, 3],
            "player_name": ["Kevin_De Bruyne", "Kevin_De Bruyne", "Harry_Kane", "Harry_Kane", "New_Player"],
            "team": ["Man City", "Man City", "Spurs", "Spurs", "Arsenal"],
            "season_id": [2020, 2021, 2020, 2021, 2020],
            "position": ["MID", "MID", "FWD", "FWD", "MID"],
        })

    @pytest.fixture
    def unmatched_data(self):
        return pd.DataFrame({
            "player_id": [1, 2, 3],
            "player_name": ["Kevin_De Bruyne", "Harry_Kane", "New_Player"],
            "team": ["Man City", "Spurs", "Arsenal"],
            "season_id": [2021, 2021, 2020],
            "position": ["MID", "FWD", "MID"],
            "source": ["fpl", "fpl", "fpl"],
        })

    def test_continuity_found(self, fpl_data, unmatched_data):
        candidates = _find_same_provider_continuity(fpl_data, unmatched_data)
        # Should find 2 continuity candidates (Kevin, Harry) + 1 unresolved (New_Player)
        cont = [c for c in candidates if c.candidate_type == "same_provider_continuity"]
        unres = [c for c in candidates if c.candidate_type == "unresolved"]
        assert len(cont) == 2
        assert len(unres) == 1

    def test_evidence_score_full_match(self, fpl_data, unmatched_data):
        candidates = _find_same_provider_continuity(fpl_data, unmatched_data)
        cont = [c for c in candidates if c.candidate_type == "same_provider_continuity"]
        for c in cont:
            # name + team + season_adjacent + position = 1.0
            assert c.evidence_score == 1.0
            assert c.team_match is True
            assert c.season_adjacent is True
            assert c.position_match is True

    def test_unresolved_no_prior_season(self, fpl_data, unmatched_data):
        candidates = _find_same_provider_continuity(fpl_data, unmatched_data)
        unres = [c for c in candidates if c.candidate_type == "unresolved"]
        assert len(unres) == 1
        assert unres[0].src_a_name == "New_Player"
        assert unres[0].evidence_score == 0.0


class TestReviewFileGeneration:
    """Tests for review file generation."""

    def test_generate_review_file_structure(self):
        """Test that review file has all required columns."""
        # Use minimal test data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create minimal test parquet files
            sb = pd.DataFrame({
                "player_id": [1],
                "player_name": ["Test Player"],
                "team_name": ["Team A"],
                "season_id": [27],
                "position": ["MID"],
            })
            fpl = pd.DataFrame({
                "player_id": [1],
                "player_name": ["Test_Player"],
                "team": ["Team A"],
                "season_id": [2016],
                "position": ["MID"],
            })
            unmatched = pd.DataFrame({
                "player_id": [1],
                "player_name": ["Test_Player"],
                "team": ["Team A"],
                "season_id": [2020],
                "position": ["MID"],
                "source": ["fpl"],
            })

            sb_path = tmp / "sb.parquet"
            fpl_path = tmp / "fpl.parquet"
            unmatched_path = tmp / "unmatched.csv"
            out_path = tmp / "review.csv"

            sb.to_parquet(sb_path)
            fpl.to_parquet(fpl_path)
            unmatched.to_csv(unmatched_path, index=False)

            result = generate_review_file(sb_path, fpl_path, unmatched_path, out_path)

            assert result.exists()
            df = pd.read_csv(result)

            # Check required columns
            required = [
                "src_a_name", "src_a_id", "src_a_source", "src_a_team", "src_a_season",
                "src_a_position", "src_b_name", "src_b_id", "src_b_source",
                "src_b_team", "src_b_season", "src_b_position",
                "name_similarity", "team_match", "season_adjacent",
                "position_match", "evidence_score", "candidate_type",
                "status", "reviewer_notes", "reviewed_by", "reviewed_at"
            ]
            for col in required:
                assert col in df.columns

            # All statuses should be pending
            assert (df["status"] == "pending").all()


class TestValidation:
    """Tests for mapping validation."""

    def test_validate_missing_required_columns(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame({"src_a_name": ["A"]}).to_csv(f.name, index=False)
            results = validate_mappings(Path(f.name))
            assert results["missing_required"]
            assert any(r["column"] == "src_a_id" for r in results["missing_required"])

    def test_validate_duplicate_src_a(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame({
                "src_a_name": ["A", "A"],
                "src_a_id": ["1", "1"],
                "src_a_source": ["fpl", "fpl"],
                "src_b_id": ["2", "3"],
                "src_b_source": ["fpl", "fpl"],
                "status": ["confirmed", "confirmed"],
                "candidate_type": ["same_provider_continuity", "same_provider_continuity"],
                "evidence_score": [1.0, 1.0],
                "name_similarity": [1.0, 1.0],
                "team_match": [True, True],
                "season_adjacent": [True, True],
                "position_match": [True, True],
            })
            df.to_csv(f.name, index=False)
            results = validate_mappings(Path(f.name))
            assert results["duplicate_mappings"]["src_a_duplicates"] == 2

    def test_validate_invalid_status(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame({
                "src_a_name": ["A"], "src_a_id": ["1"], "src_a_source": ["fpl"],
                "src_b_id": ["2"], "src_b_source": ["fpl"],
                "status": ["invalid_status"],
                "candidate_type": ["same_provider_continuity"],
                "evidence_score": [1.0], "name_similarity": [1.0],
                "team_match": [True], "season_adjacent": [True], "position_match": [True],
            })
            df.to_csv(f.name, index=False)
            results = validate_mappings(Path(f.name))
            assert results["invalid_status"] == 1

    def test_validate_invalid_candidate_type(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame({
                "src_a_name": ["A"], "src_a_id": ["1"], "src_a_source": ["fpl"],
                "src_b_id": ["2"], "src_b_source": ["fpl"],
                "status": ["confirmed"],
                "candidate_type": ["invalid_type"],
                "evidence_score": [1.0], "name_similarity": [1.0],
                "team_match": [True], "season_adjacent": [True], "position_match": [True],
            })
            df.to_csv(f.name, index=False)
            results = validate_mappings(Path(f.name))
            assert results["invalid_candidate_type"] == 1

    def test_validate_one_to_one_violations(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame({
                "src_a_name": ["A", "B"], "src_a_id": ["1", "2"], "src_a_source": ["fpl", "fpl"],
                "src_b_id": ["3", "3"], "src_b_source": ["fpl", "fpl"],
                "status": ["confirmed", "confirmed"],
                "candidate_type": ["same_provider_continuity", "same_provider_continuity"],
                "evidence_score": [1.0, 1.0], "name_similarity": [1.0, 1.0],
                "team_match": [True, True], "season_adjacent": [True, True], "position_match": [True, True],
            })
            df.to_csv(f.name, index=False)
            results = validate_mappings(Path(f.name))
            assert results["one_to_one_violations"]["src_b_multiple"] == 1


class TestApplyConfirmedMappings:
    """Tests for applying confirmed mappings."""

    def test_apply_no_confirmed(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame({
                "src_a_name": ["A"], "src_a_id": ["1"], "src_a_source": ["fpl"],
                "src_b_id": ["2"], "src_b_source": ["fpl"],
                "status": ["pending"],
                "candidate_type": ["same_provider_continuity"],
                "evidence_score": [1.0], "name_similarity": [1.0],
                "team_match": [True], "season_adjacent": [True], "position_match": [True],
            })
            df.to_csv(f.name, index=False)
            result = apply_confirmed_mappings(Path(f.name))
            assert result.empty

    def test_apply_confirmed_creates_canonical_mapping(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame({
                "src_a_name": ["A"], "src_a_id": ["1"], "src_a_source": ["statsbomb"],
                "src_a_team": ["Team A"], "src_a_season": [27], "src_a_position": ["MID"],
                "src_b_name": ["A"], "src_b_id": ["2"], "src_b_source": ["fpl"],
                "src_b_team": ["Team A"], "src_b_season": [2016], "src_b_position": ["MID"],
                "status": ["confirmed"],
                "candidate_type": ["cross_provider"],
                "evidence_score": [0.9], "name_similarity": [1.0],
                "team_match": [True], "season_adjacent": [True], "position_match": [True],
                "reviewer_notes": ["test"], "reviewed_by": ["tester"], "reviewed_at": ["2024-01-01"],
            })
            df.to_csv(f.name, index=False)
            result = apply_confirmed_mappings(Path(f.name))
            assert len(result) == 1
            assert "canonical_id" in result.columns
            assert result.iloc[0]["source_a"] == "statsbomb"
            assert result.iloc[0]["source_b"] == "fpl"


class TestDeterministicOutput:
    """Tests that candidate generation is deterministic."""

    def test_cross_provider_deterministic(self):
        sb = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["Player A", "Player B"],
            "team_name": ["Team X", "Team Y"],
            "season_id": [27, 27],
            "position": ["MID", "FWD"],
        })
        fpl = pd.DataFrame({
            "player_id": [10, 20],
            "player_name": ["Player_A", "Player_B"],
            "team": ["Team X", "Team Y"],
            "season_id": [2016, 2016],
            "position": [None, None],
        })

        c1 = _find_cross_provider_candidates(sb, fpl)
        c2 = _find_cross_provider_candidates(sb, fpl)

        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a.to_dict() == b.to_dict()

    def test_same_provider_deterministic(self):
        fpl = pd.DataFrame({
            "player_id": [1, 1],
            "player_name": ["Player_A", "Player_A"],
            "team": ["Team X", "Team X"],
            "season_id": [2020, 2021],
            "position": ["MID", "MID"],
        })
        unmatched = pd.DataFrame({
            "player_id": [1],
            "player_name": ["Player_A"],
            "team": ["Team X"],
            "season_id": [2021],
            "position": ["MID"],
            "source": ["fpl"],
        })

        c1 = _find_same_provider_continuity(fpl, unmatched)
        c2 = _find_same_provider_continuity(fpl, unmatched)

        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a.to_dict() == b.to_dict()


class TestCandidateMatchDataclass:
    """Tests for CandidateMatch dataclass."""

    def test_to_dict(self):
        c = CandidateMatch(
            src_a_name="A", src_a_id="1", src_a_source="statsbomb",
            src_a_team="T1", src_a_season=2015, src_a_position="MID",
            src_b_name="B", src_b_id="2", src_b_source="fpl",
            src_b_team="T1", src_b_season=2016, src_b_position="MID",
            name_similarity=1.0, team_match=True, season_adjacent=True,
            position_match=True, evidence_score=1.0, candidate_type="cross_provider",
        )
        d = c.to_dict()
        assert d["src_a_name"] == "A"
        assert d["evidence_score"] == 1.0
        assert d["status"] == "pending"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])