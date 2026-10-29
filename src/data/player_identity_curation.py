"""Player identity curation workflow for Phase 8.3C.

Provides tooling for human review of player identity matches across
StatsBomb, FPL, and OpenFootball providers.

Key concepts:
- A "source record" is a provider-specific player-season row (e.g., FPL element 123 in 2023/24).
- A "canonical identity" is a stable player entity across providers and seasons.
- Candidate matches link source records to canonical identities with evidence.
- Human review is required; no silent auto-approval.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data import config as data_config
from src.data.player_mapping import (
    build_player_registry,
    export_unmatched_for_review,
    get_canonical_player_name,
    load_player_registry,
    save_player_registry,
)
from src.data.team_mapping import get_canonical_name, normalize_team_name

# Map provider season_id to starting year for season adjacency checks
SEASON_ID_TO_YEAR = {
    # StatsBomb
    27: 2015,
    # FPL
    2016: 2016,
    2017: 2017,
    2018: 2018,
    2019: 2019,
    2020: 2020,
    2021: 2021,
    2022: 2022,
    2023: 2023,
    2024: 2024,
    2025: 2025,
}

def _season_year(season_id: int) -> Optional[int]:
    """Get starting year for a season_id."""
    return SEASON_ID_TO_YEAR.get(int(season_id))


@dataclass(frozen=True)
class CandidateMatch:
    """A candidate player match for human review."""
    # Source A record (e.g., StatsBomb)
    src_a_name: str
    src_a_id: str
    src_a_source: str  # "statsbomb" | "fpl"
    src_a_team: str
    src_a_season: int
    src_a_position: Optional[str]

    # Source B record (e.g., FPL) - or None for same-provider continuity
    src_b_name: Optional[str]
    src_b_id: Optional[str]
    src_b_source: Optional[str]
    src_b_team: Optional[str]
    src_b_season: Optional[int]
    src_b_position: Optional[str]

    # Matching evidence
    name_similarity: float
    team_match: bool
    season_adjacent: bool
    position_match: bool
    evidence_score: float
    candidate_type: str  # "cross_provider" | "same_provider_continuity" | "unresolved"

    # Human review fields
    status: str = "pending"  # pending, confirmed, rejected, ambiguous
    reviewer_notes: str = ""
    reviewed_by: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "src_a_name": self.src_a_name,
            "src_a_id": self.src_a_id,
            "src_a_source": self.src_a_source,
            "src_a_team": self.src_a_team,
            "src_a_season": self.src_a_season,
            "src_a_position": self.src_a_position,
            "src_b_name": self.src_b_name,
            "src_b_id": self.src_b_id,
            "src_b_source": self.src_b_source,
            "src_b_team": self.src_b_team,
            "src_b_season": self.src_b_season,
            "src_b_position": self.src_b_position,
            "name_similarity": self.name_similarity,
            "team_match": self.team_match,
            "season_adjacent": self.season_adjacent,
            "position_match": self.position_match,
            "evidence_score": self.evidence_score,
            "candidate_type": self.candidate_type,
            "status": self.status,
            "reviewer_notes": self.reviewer_notes,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
        }


# ---- Name normalization ----

def _normalize_name_for_matching(name: str) -> str:
    """Normalize player name for matching: underscores, accents, special chars."""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    # Underscores (FPL format) to spaces
    s = s.replace("_", " ")
    # Unicode normalization and accent removal
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    # Remove common suffixes
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    # Remove special chars
    s = s.replace("-", " ").replace("'", "").replace(".", "").replace(",", "")
    # Collapse whitespace
    s = " ".join(s.split())
    return s.strip()


def _jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler similarity via difflib.SequenceMatcher."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, s1, s2).ratio()


# ---- Candidate generation ----

def _find_cross_provider_candidates(
    sb_df: pd.DataFrame,
    fpl_df: pd.DataFrame,
    name_threshold: float = 0.95,
) -> list[CandidateMatch]:
    """Find cross-provider candidates between StatsBomb (2015/16) and FPL (2016/17+).

    StatsBomb only has 2015/16 (season_id=27). FPL has 2016/17-2025/26.
    We match on exact normalized name + adjacent season (gap <= 1 year) + team continuity.
    """
    candidates = []

    # Prepare StatsBomb (only 2015/16)
    sb = sb_df.copy()
    sb["_norm_name"] = sb["player_name"].apply(_normalize_name_for_matching)
    sb["_canon_team"] = sb["team_name"].apply(lambda t: get_canonical_name(t, "statsbomb"))
    sb = sb[sb["season_id"] == 27]

    # Prepare FPL
    fpl = fpl_df.copy()
    fpl["_norm_name"] = fpl["player_name"].apply(_normalize_name_for_matching)
    fpl["_canon_team"] = fpl["team"].apply(lambda t: get_canonical_name(t, "fpl"))

    # For each StatsBomb player, find FPL candidates in adjacent seasons
    for _, sb_row in sb.iterrows():
        sb_name = _normalize_name_for_matching(sb_row["player_name"])
        sb_team = sb_row["_canon_team"]
        sb_season_year = _season_year(sb_row["season_id"])
        sb_pos = sb_row.get("position")

        # FPL adjacent season: 2016/17 (season_id=2016)
        fpl_adjacent = fpl[fpl["season_id"] == 2016]

        for _, fpl_row in fpl_adjacent.iterrows():
            fpl_name = fpl_row["_norm_name"]

            if sb_name != fpl_name:
                continue  # require exact normalized name for cross-provider

            fpl_team = fpl_row["_canon_team"]
            fpl_season_year = _season_year(fpl_row["season_id"])
            fpl_pos = fpl_row.get("position")

            # Team match (FPL 2016 has NaN team, so this will be False)
            team_match = (isinstance(sb_team, str) and isinstance(fpl_team, str)
                          and sb_team == fpl_team)

            # Season adjacent: 2015 vs 2016
            season_adjacent = (sb_season_year is not None and fpl_season_year is not None
                               and abs(sb_season_year - fpl_season_year) <= 1)

            # Position match (FPL 2016 has NaN)
            pos_match = (isinstance(sb_pos, str) and isinstance(fpl_pos, str)
                         and sb_pos.lower() == fpl_pos.lower())

            # Evidence score: name exact (0.5) + team (0.3) + season (0.1) + position (0.1)
            evidence_score = 0.5
            if team_match:
                evidence_score += 0.3
            if season_adjacent:
                evidence_score += 0.1
            if pos_match:
                evidence_score += 0.1

            candidate_type = "cross_provider"

            candidates.append(CandidateMatch(
                src_a_name=sb_row["player_name"],
                src_a_id=str(sb_row["player_id"]),
                src_a_source="statsbomb",
                src_a_team=sb_row["team_name"],
                src_a_season=int(sb_row["season_id"]),
                src_a_position=sb_pos if isinstance(sb_pos, str) else None,
                src_b_name=fpl_row["player_name"],
                src_b_id=str(fpl_row["player_id"]),
                src_b_source="fpl",
                src_b_team=fpl_row["team"] if isinstance(fpl_row["team"], str) else None,
                src_b_season=int(fpl_row["season_id"]),
                src_b_position=fpl_pos if isinstance(fpl_pos, str) else None,
                name_similarity=1.0,
                team_match=team_match,
                season_adjacent=season_adjacent,
                position_match=pos_match,
                evidence_score=round(evidence_score, 3),
                candidate_type=candidate_type,
            ))

    return candidates


def _find_same_provider_continuity(
    fpl_df: pd.DataFrame,
    unmatched_df: pd.DataFrame,
) -> list[CandidateMatch]:
    """Find same-provider continuity candidates for unmatched FPL player-season rows.

    FPL element IDs are NOT stable across seasons. We link unmatched rows
    to other FPL seasons of the same canonical identity using name + team continuity.
    """
    candidates = []

    # Build a reference of all FPL player-season rows with canonical names/teams
    fpl = fpl_df.copy()
    fpl["_norm_name"] = fpl["player_name"].apply(_normalize_name_for_matching)
    fpl["_canon_team"] = fpl["team"].apply(lambda t: get_canonical_name(t, "fpl"))

    # For each unmatched row, find other seasons with same canonical name + team
    for _, u_row in unmatched_df.iterrows():
        u_name = _normalize_name_for_matching(u_row["player_name"])
        u_team = get_canonical_name(u_row["team"] if isinstance(u_row["team"], str) else "", "fpl")
        u_season = int(u_row["season_id"])
        u_pos = u_row.get("position")

        # Find other FPL seasons with same canonical name + team
        matches = fpl[
            (fpl["_norm_name"] == u_name) &
            (fpl["_canon_team"] == u_team) &
            (fpl["season_id"] != u_season)
        ]

        if matches.empty:
            # No continuity candidate - unresolved
            candidates.append(CandidateMatch(
                src_a_name=u_row["player_name"],
                src_a_id=str(u_row["player_id"]),
                src_a_source="fpl",
                src_a_team=u_row["team"] if isinstance(u_row["team"], str) else None,
                src_a_season=u_season,
                src_a_position=u_pos if isinstance(u_pos, str) else None,
                src_b_name=None,
                src_b_id=None,
                src_b_source=None,
                src_b_team=None,
                src_b_season=None,
                src_b_position=None,
                name_similarity=1.0,
                team_match=False,
                season_adjacent=False,
                position_match=False,
                evidence_score=0.0,
                candidate_type="unresolved",
            ))
            continue

        # Take the most recent prior season as canonical reference
        prior = matches[matches["season_id"] < u_season]
        if not prior.empty:
            ref = prior.sort_values("season_id").iloc[-1]
        else:
            ref = matches.sort_values("season_id").iloc[0]

        ref_name = ref["player_name"]
        ref_team = ref["team"] if isinstance(ref["team"], str) else None
        ref_season = int(ref["season_id"])
        ref_pos = ref.get("position")

        team_match = (isinstance(u_team, str) and isinstance(ref_team, str)
                      and u_team == get_canonical_name(ref_team, "fpl"))
        season_adjacent = (abs(_season_year(u_season) - _season_year(ref_season)) <= 1
                           if _season_year(u_season) and _season_year(ref_season) else False)
        pos_match = (isinstance(u_pos, str) and isinstance(ref_pos, str)
                     and u_pos.lower() == ref_pos.lower())

        evidence_score = 0.5  # exact normalized name
        if team_match:
            evidence_score += 0.3
        if season_adjacent:
            evidence_score += 0.1
        if pos_match:
            evidence_score += 0.1

        if evidence_score >= 0.8:
            confidence_cat = "high"
        elif evidence_score >= 0.6:
            confidence_cat = "ambiguous"
        else:
            confidence_cat = "rejected"

        candidates.append(CandidateMatch(
            src_a_name=u_row["player_name"],
            src_a_id=str(u_row["player_id"]),
            src_a_source="fpl",
            src_a_team=u_row["team"] if isinstance(u_row["team"], str) else None,
            src_a_season=u_season,
            src_a_position=u_pos if isinstance(u_pos, str) else None,
            src_b_name=ref_name,
            src_b_id=str(ref["player_id"]),
            src_b_source="fpl",
            src_b_team=ref_team,
            src_b_season=ref_season,
            src_b_position=ref_pos if isinstance(ref_pos, str) else None,
            name_similarity=1.0,
            team_match=team_match,
            season_adjacent=season_adjacent,
            position_match=pos_match,
            evidence_score=round(evidence_score, 3),
            candidate_type="same_provider_continuity",
        ))

    return candidates


def generate_review_file(
    statsbomb_path: Optional[Path] = None,
    fpl_path: Optional[Path] = None,
    unmatched_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a review file for human curation of player identity matches.

    Args:
        statsbomb_path: Path to StatsBomb player-season parquet
        fpl_path: Path to FPL player-season parquet
        unmatched_path: Path to unmatched players CSV (from Phase 8.2)
        output_path: Output path for review CSV

    Returns:
        Path to the generated review file
    """
    if statsbomb_path is None:
        statsbomb_path = data_config.STATSBOMB_PROCESSED_DIR / "player_season.parquet"
    if fpl_path is None:
        fpl_path = data_config.DATA_PROCESSED_DIR / "fpl_player_season.parquet"
    if unmatched_path is None:
        unmatched_path = data_config.REPORTS_DIR / "player_mapping_unmatched.csv"
    if output_path is None:
        output_path = data_config.DATA_PROCESSED_DIR / "review" / "player_identity_review.csv"

    # Load data
    sb = pd.read_parquet(statsbomb_path)
    fpl = pd.read_parquet(fpl_path)

    # Load unmatched (FPL player-season rows that failed registry lookup)
    if unmatched_path.exists():
        unmatched = pd.read_csv(unmatched_path)
        # Ensure it's FPL only (Phase 8.2 exported FPL unmatched due to team alias bug)
        unmatched = unmatched[unmatched["source"] == "fpl"].copy()
    else:
        unmatched = pd.DataFrame()

    # Generate cross-provider candidates (SB 2015/16 ↔ FPL 2016/17)
    cross_candidates = _find_cross_provider_candidates(sb, fpl)

    # Generate same-provider continuity candidates for unmatched FPL rows
    same_provider_candidates = _find_same_provider_continuity(fpl, unmatched)

    # Combine all candidates
    all_candidates = cross_candidates + same_provider_candidates

    # Convert to DataFrame
    review_rows = [c.to_dict() for c in all_candidates]
    review_df = pd.DataFrame(review_rows)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(output_path, index=False)
    print(f"Generated review file: {output_path} ({len(review_df)} candidates)")
    print(f"  cross_provider: {len(cross_candidates)}")
    print(f"  same_provider_continuity: {len(same_provider_candidates)}")
    return output_path


# ---- Validation ----

def validate_mappings(mappings_path: Path) -> dict:
    """Validate a mappings CSV file for consistency.

    Returns:
        Dict with validation results
    """
    mappings = pd.read_csv(mappings_path)

    results = {
        "total_rows": len(mappings),
        "status_counts": mappings["status"].value_counts().to_dict() if "status" in mappings.columns else {},
        "candidate_type_counts": mappings["candidate_type"].value_counts().to_dict() if "candidate_type" in mappings.columns else {},
        "missing_required": [],
        "duplicate_mappings": [],
        "invalid_status": 0,
        "invalid_candidate_type": 0,
    }

    # Check required columns
    required_cols = [
        "src_a_name", "src_a_id", "src_a_source", "src_a_team", "src_a_season",
        "src_b_name", "src_b_id", "src_b_source", "src_b_team", "src_b_season",
        "name_similarity", "team_match", "season_adjacent", "position_match",
        "evidence_score", "candidate_type", "status"
    ]
    for col in required_cols:
        if col not in mappings.columns:
            mappings[col] = ""
        missing = mappings[mappings[col].isna() | (mappings[col] == "")]
        if not missing.empty:
            results["missing_required"].append({"column": col, "count": len(missing)})

    # Check for duplicate source mappings (same src_a_id mapped multiple times)
    if "src_a_id" in mappings.columns and "src_a_source" in mappings.columns:
        dup = mappings[mappings.duplicated(subset=["src_a_source", "src_a_id"], keep=False)]
        results["duplicate_mappings"] = {
            "src_a_duplicates": len(dup),
            "duplicate_details": dup[["src_a_source", "src_a_id", "src_b_id", "status"]].to_dict("records") if not dup.empty else []
        }

    # Validate status values
    valid_statuses = ["pending", "confirmed", "rejected", "ambiguous"]
    if "status" in mappings.columns:
        invalid = mappings[~mappings["status"].isin(valid_statuses)]
        results["invalid_status"] = len(invalid)

    # Validate candidate_type values
    valid_types = ["cross_provider", "same_provider_continuity", "unresolved"]
    if "candidate_type" in mappings.columns:
        invalid = mappings[~mappings["candidate_type"].isin(valid_types)]
        results["invalid_candidate_type"] = len(invalid)

    # For confirmed mappings, check one-to-one
    confirmed = mappings[mappings["status"] == "confirmed"]
    if len(confirmed) > 0:
        # src_a should map to at most one src_b and vice versa
        a_counts = confirmed.groupby(["src_a_source", "src_a_id"]).size()
        b_counts = confirmed.groupby(["src_b_source", "src_b_id"]).size()
        results["one_to_one_violations"] = {
            "src_a_multiple": int((a_counts > 1).sum()),
            "src_b_multiple": int((b_counts > 1).sum()),
        }

    return results


def apply_confirmed_mappings(
    mappings_path: Path,
    registry_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Apply human-reviewed confirmed mappings to build a canonical identity mapping.

    This does NOT modify the existing player_registry.parquet.
    Instead, it produces a separate canonical_identity_mapping.parquet.

    Args:
        mappings_path: Path to CSV with human-reviewed mappings
        registry_path: Path to existing player registry (for reference)

    Returns:
        DataFrame of canonical identity mappings
    """
    mappings = pd.read_csv(mappings_path)
    confirmed = mappings[mappings["status"] == "confirmed"].copy()

    if confirmed.empty:
        print("No confirmed mappings to apply")
        return pd.DataFrame()

    # Validate one-to-one
    validation = validate_mappings(mappings_path)
    if validation.get("one_to_one_violations", {}).get("src_a_multiple", 0) > 0:
        print("WARNING: Some src_a map to multiple src_b in confirmed mappings")
    if validation.get("one_to_one_violations", {}).get("src_b_multiple", 0) > 0:
        print("WARNING: Some src_b map to multiple src_a in confirmed mappings")

    # Build canonical identity mapping
    # Each confirmed pair defines a link between two source records
    # We assign a canonical_id as hash of the pair
    canonical_rows = []
    for _, row in confirmed.iterrows():
        # Stable ID from both source records
        key = f"{row['src_a_source']}|{row['src_a_id']}|{row['src_b_source']}|{row['src_b_id']}"
        import hashlib
        canonical_id = int(hashlib.md5(key.encode()).hexdigest()[:12], 16)

        canonical_rows.append({
            "canonical_id": canonical_id,
            "source_a": row["src_a_source"],
            "source_a_id": row["src_a_id"],
            "source_a_name": row["src_a_name"],
            "source_a_season": row["src_a_season"],
            "source_a_team": row["src_a_team"],
            "source_b": row["src_b_source"],
            "source_b_id": row["src_b_id"],
            "source_b_name": row["src_b_name"],
            "source_b_season": row["src_b_season"],
            "source_b_team": row["src_b_team"],
            "evidence_score": row["evidence_score"],
            "candidate_type": row["candidate_type"],
            "reviewer": row.get("reviewed_by", ""),
            "reviewed_at": row.get("reviewed_at", ""),
            "notes": row.get("reviewer_notes", ""),
        })

    canonical_df = pd.DataFrame(canonical_rows)

    # Save
    out_path = data_config.DATA_PROCESSED_DIR / "canonical_identity_mapping.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_df.to_parquet(out_path, index=False)
    print(f"Applied {len(canonical_df)} confirmed mappings -> {out_path}")
    return canonical_df


# ---- CLI ----

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        out = generate_review_file()
        print(f"Review file generated: {out}")
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        if len(sys.argv) < 3:
            print("Usage: python -m src.data.player_identity_curation validate <mappings.csv>")
            sys.exit(1)
        results = validate_mappings(Path(sys.argv[2]))
        print("Validation results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
    elif len(sys.argv) > 1 and sys.argv[1] == "apply":
        if len(sys.argv) < 3:
            print("Usage: python -m src.data.player_identity_curation apply <mappings.csv>")
            sys.exit(1)
        apply_confirmed_mappings(Path(sys.argv[2]))
    else:
        print("Usage: python -m src.data.player_identity_curation [generate|validate|apply]")