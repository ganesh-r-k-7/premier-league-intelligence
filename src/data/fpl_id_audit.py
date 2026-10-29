"""FPL element ID reuse audit for Phase 8.5B.

Systematically identifies all FPL element IDs reused for different players
across seasons. FPL element IDs are NOT stable across seasons — the same
element_id can be assigned to different players in different seasons.

This audit produces a report classifying the severity of each reuse case.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data import config as data_config


def _normalize_name_for_matching(name: str) -> str:
    """Normalize player name for matching: underscores, accents, special chars."""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = s.replace("_", " ")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    s = s.replace("-", " ").replace("'", "").replace(".", "").replace(",", "")
    s = " ".join(s.split())
    return s.strip()


def _classify_reuse_severity(group: pd.DataFrame) -> tuple[str, str]:
    """
    Classify the severity of FPL element ID reuse for a group of rows
    sharing the same element_id.

    Returns:
        (severity, details) where severity is one of:
        - "Critical": Same season, different normalized names/teams
        - "High": Adjacent seasons, different normalized names, same team
        - "Medium": Non-adjacent seasons, different normalized names
        - "Low": Same normalized name, team/position changes (legitimate)
        - "None": No reuse detected
    """
    if len(group) <= 1:
        return "None", "Single record for this element_id"

    # Check for same-season conflicts
    season_groups = group.groupby("season_id")
    for season_id, season_group in season_groups:
        if len(season_group) > 1:
            # Multiple rows for same element_id in same season
            unique_names = season_group["player_name"].nunique()
            unique_teams = season_group["team"].nunique() if "team" in season_group.columns else 0
            if unique_names > 1 or unique_teams > 1:
                return "Critical", (
                    f"Same season {season_id}: {unique_names} different names, "
                    f"{unique_teams} different teams"
                )

    # Sort by season for adjacent season check
    group = group.sort_values("season_id")
    seasons = group["season_id"].tolist()
    names = group["player_name"].tolist()
    teams = group["team"].tolist() if "team" in group.columns else [None] * len(group)

    # Check for adjacent season name changes with same team
    for i in range(len(seasons) - 1):
        season_diff = seasons[i + 1] - seasons[i]
        if season_diff <= 1:  # Adjacent or same season
            name1 = _normalize_name_for_matching(names[i])
            name2 = _normalize_name_for_matching(names[i + 1])
            if name1 != name2:
                team1 = teams[i] if i < len(teams) else None
                team2 = teams[i + 1] if (i + 1) < len(teams) else None
                if team1 and team2 and team1 == team2:
                    return "High", (
                        f"Adjacent seasons {seasons[i]}-{seasons[i+1]}: "
                        f"name changed ({names[i]} -> {names[i+1]}) "
                        f"but same team ({team1})"
                    )

    # Check for non-adjacent season name changes
    unique_normalized_names = set(_normalize_name_for_matching(n) for n in names)
    if len(unique_normalized_names) > 1:
        return "Medium", (
            f"Multiple names across seasons: {len(unique_normalized_names)} "
            f"distinct normalized names ({', '.join(sorted(unique_normalized_names))})"
        )

    # Same normalized name but team/position changes
    unique_teams = set(group["team"].dropna().tolist()) if "team" in group.columns else set()
    unique_positions = set(group["position"].dropna().tolist()) if "position" in group.columns else set()
    if len(unique_teams) > 1 or len(unique_positions) > 1:
        return "Low", (
            f"Same name, team changes: {len(unique_teams)}, "
            f"position changes: {len(unique_positions)}"
        )

    return "None", "No significant reuse pattern detected"


def run_fpl_id_reuse_audit(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the FPL element ID reuse audit on the remediated player-season data.

    Args:
        input_path: Path to fpl_player_season_remediated.parquet
        output_path: Path to write audit CSV

    Returns:
        DataFrame with audit results
    """
    if input_path is None:
        input_path = data_config.DATA_PROCESSED_DIR / "fpl_player_season_remediated.parquet"
    if output_path is None:
        output_path = data_config.DATA_PROCESSED_DIR / "review" / "fpl_element_id_reuse_audit.csv"

    df = pd.read_parquet(input_path)

    # Group by element_id (player_id)
    results = []
    for element_id, group in df.groupby("player_id"):
        group = group.copy()
        seasons_used = group["season_id"].nunique()
        unique_names = group["player_name"].nunique()

        # Normalized names for better comparison
        group["_norm_name"] = group["player_name"].apply(_normalize_name_for_matching)
        unique_normalized_names = group["_norm_name"].nunique()

        # Team and position variety
        unique_teams = group["team"].nunique() if "team" in group.columns else 0
        unique_positions = group["position"].nunique() if "position" in group.columns else 0

        # Quality flags
        quality_flags = group["quality_flag"].tolist() if "quality_flag" in group.columns else []
        has_quality_flag = any(flag for flag in quality_flags if flag)

        # Determine severity
        severity, details = _classify_reuse_severity(group)

        # Season range
        first_season = int(group["season_id"].min())
        last_season = int(group["season_id"].max())

        # Collect distinct names
        distinct_names = sorted(group["player_name"].unique().tolist())
        distinct_teams = sorted([t for t in group["team"].unique().tolist() if pd.notna(t)]) if "team" in group.columns else []
        distinct_positions = sorted([p for p in group["position"].unique().tolist() if pd.notna(p)]) if "position" in group.columns else []

        results.append({
            "element_id": int(element_id),
            "seasons_used": int(seasons_used),
            "unique_names": int(unique_names),
            "unique_normalized_names": int(unique_normalized_names),
            "unique_teams": int(unique_teams),
            "unique_positions": int(unique_positions),
            "severity": severity,
            "details": details,
            "affected_record_count": len(group),
            "first_season": first_season,
            "last_season": last_season,
            "distinct_names": "; ".join(distinct_names),
            "distinct_teams": "; ".join(distinct_teams),
            "distinct_positions": "; ".join(distinct_positions),
            "has_quality_flag": has_quality_flag,
            "quality_flags": "; ".join([f for f in quality_flags if f]),
        })

    audit_df = pd.DataFrame(results)
    audit_df = audit_df.sort_values(["severity", "element_id"], ascending=[True, True]).reset_index(drop=True)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_path, index=False)

    print(f"FPL ID Reuse Audit complete: {len(audit_df)} element IDs analyzed")
    print(f"  Critical: {len(audit_df[audit_df.severity == 'Critical'])}")
    print(f"  High: {len(audit_df[audit_df.severity == 'High'])}")
    print(f"  Medium: {len(audit_df[audit_df.severity == 'Medium'])}")
    print(f"  Low: {len(audit_df[audit_df.severity == 'Low'])}")
    print(f"  None: {len(audit_df[audit_df.severity == 'None'])}")
    print(f"Output written to: {output_path}")

    return audit_df


def add_fpl_id_reuse_severity_to_review(
    review_path: Path,
    audit_path: Path,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Add fpl_id_reuse_severity column to the review file based on audit results.

    Args:
        review_path: Path to player_identity_review.csv
        audit_path: Path to fpl_element_id_reuse_audit.csv
        output_path: Optional output path (defaults to overwriting review_path)

    Returns:
        Updated review DataFrame
    """
    review_df = pd.read_csv(review_path)
    audit_df = pd.read_csv(audit_path)

    # Create severity lookup for src_a and src_b
    severity_lookup = dict(zip(audit_df["element_id"], audit_df["severity"]))

    def get_severity(row, src_col: str) -> Optional[str]:
        """Get severity for a source record."""
        if row[f"{src_col}_source"] != "fpl":
            return None
        element_id = row[f"{src_col}_id"]
        try:
            return severity_lookup.get(int(element_id))
        except (ValueError, TypeError):
            return None

    # Add severity for both sources
    review_df["src_a_fpl_id_reuse_severity"] = review_df.apply(lambda r: get_severity(r, "src_a"), axis=1)
    review_df["src_b_fpl_id_reuse_severity"] = review_df.apply(lambda r: get_severity(r, "src_b"), axis=1)

    # Combined severity: max of both sources
    severity_order = {"None": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    def max_severity(row):
        sevs = [row["src_a_fpl_id_reuse_severity"], row["src_b_fpl_id_reuse_severity"]]
        valid = [s for s in sevs if s is not None]
        if not valid:
            return None
        return max(valid, key=lambda s: severity_order.get(s, 0))

    review_df["fpl_id_reuse_severity"] = review_df.apply(max_severity, axis=1)

    output_path = output_path or review_path
    review_df.to_csv(output_path, index=False)
    print(f"Added fpl_id_reuse_severity to review file: {output_path}")
    print(f"  Severity distribution: {review_df['fpl_id_reuse_severity'].value_counts().to_dict()}")

    return review_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "add-severity":
        review_path = data_config.DATA_PROCESSED_DIR / "review" / "player_identity_review.csv"
        audit_path = data_config.DATA_PROCESSED_DIR / "review" / "fpl_element_id_reuse_audit.csv"
        add_fpl_id_reuse_severity_to_review(review_path, audit_path)
    else:
        run_fpl_id_reuse_audit()