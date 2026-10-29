"""Historical data quality remediation for Phase 8.5A.

Investigates and resolves known data-quality issues in FPL and OpenFootball datasets.
All corrections are reproducible and auditable with provenance preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.data.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    STATSBOMB_PROCESSED_DIR,
)


# ---- FPL Player-Season Remediation ----

FPL_PLAYER_SEASON_REMEDIATIONS = [
    {
        "player_id": 120,
        "season_id": 2023,
        "issue": "name_encoding",
        "description": "Yegor Yarmoliuk vs Yegor Yarmolyuk - transliteration difference, same player",
        "keep_idx": 4968,  # row with more minutes (630 vs 40)
        "drop_idx": 4969,
    },
    {
        "player_id": 515,
        "season_id": 2018,
        "issue": "name_encoding",
        "description": "Çaglar Söyüncü - accented vs non-accented encoding, same player",
        "keep_idx": 1844,  # row with more minutes (281 vs 90)
        "drop_idx": 1845,
    },
    {
        "player_id": 620,
        "season_id": 2023,
        "issue": "name_typo",
        "description": "Michael Olakigbe vs Michale Olakigbe - typo, same player",
        "keep_idx": 5468,  # row with more minutes (43 vs 8) and correct spelling
        "drop_idx": 5469,
    },
    {
        "player_id": 687,
        "season_id": 2023,
        "issue": "name_encoding",
        "description": "Djordje Petrovic vs Đorđe Petrović - transliteration difference, same player",
        "keep_idx": 5537,  # row with more minutes (1620 vs 366) and correct transliteration
        "drop_idx": 5536,
    },
    {
        "player_id": 802,
        "season_id": 2023,
        "issue": "name_variation",
        "description": "Max Kinsey vs Max Kinsey-Wellings - hyphenated surname variation, same player",
        "keep_idx": 5653,  # row with more appearances (14 vs 4)
        "drop_idx": 5652,
    },
    {
        "player_id": 748,
        "season_id": 2024,
        "issue": "fpl_id_reuse",
        "description": "Ivan Juric vs Simon Rusk - FPL element 748 reused for two different people in 2024/25, both 0 minutes",
        "action": "flag_only",  # Cannot determine correct player; flag both
        "keep_idx": None,
        "drop_idx": None,
    },
]


def remediate_fpl_player_season(
    input_path: Path,
    output_path: Path,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Remediate FPL player-season duplicates.

    Args:
        input_path: Path to source parquet file
        output_path: Path to write remediated parquet file
        dry_run: If True, only return the remediated DataFrame without writing

    Returns:
        Remediated DataFrame
    """
    df = pd.read_parquet(input_path)
    original_len = len(df)

    # Track remediation actions
    actions = []

    # Apply each remediation
    for rem in FPL_PLAYER_SEASON_REMEDIATIONS:
        pid = rem["player_id"]
        sid = rem["season_id"]
        issue = rem["issue"]

        if issue == "fpl_id_reuse":
            # Cannot resolve - flag both rows by adding a quality flag column
            if "quality_flag" not in df.columns:
                df["quality_flag"] = ""
            mask = (df["player_id"] == pid) & (df["season_id"] == sid)
            df.loc[mask, "quality_flag"] = "fpl_id_reuse_different_players"
            actions.append({
                "player_id": pid,
                "season_id": sid,
                "action": "flagged",
                "reason": rem["description"],
            })
        else:
            # Drop the less complete row
            drop_idx = rem["drop_idx"]
            keep_idx = rem["keep_idx"]
            if drop_idx in df.index:
                dropped = df.loc[drop_idx].copy()
                df = df.drop(index=drop_idx)
                actions.append({
                    "player_id": pid,
                    "season_id": sid,
                    "action": "dropped_duplicate",
                    "kept_index": int(keep_idx),
                    "dropped_index": int(drop_idx),
                    "reason": rem["description"],
                    "kept_minutes": int(dropped.get("minutes", 0)) if keep_idx == drop_idx else int(df.loc[keep_idx, "minutes"]) if keep_idx in df.index else None,
                    "dropped_minutes": int(dropped.get("minutes", 0)),
                })

    # Reset index
    df = df.reset_index(drop=True)

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Remediated FPL player-season: {original_len} -> {len(df)} rows")
        print(f"Actions: {len(actions)}")
        for a in actions:
            print(f"  {a}")

    return df, actions


# ---- FPL Player-Match Remediation ----

def remediate_fpl_player_match(
    input_path: Path,
    output_path: Path,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Remediate FPL player-match rescheduled fixture duplicates.

    The 69 duplicate groups are all from fixture 275 in 2019/20 (Arsenal vs Man City),
    which was postponed from GW29 (2020-03-11) to GW39 (2020-06-17) due to COVID-19.
    Both gameweek entries exist in the raw data with the same fixture_id.

    Strategy: Keep both rows but add a `rescheduled_match` flag column.
    The GW29 row represents the postponed fixture (0 minutes), GW39 the played match.
    """
    df = pd.read_parquet(input_path)
    original_len = len(df)

    # Identify the rescheduled fixture duplicates
    # All are fixture_id=275, season_id=2019, appearing in GW29 and GW39
    rescheduled_mask = (df["fixture_id"] == 275) & (df["season_id"] == 2019)

    if "rescheduled_match" not in df.columns:
        df["rescheduled_match"] = False

    df.loc[rescheduled_mask, "rescheduled_match"] = True

    # Also add a column indicating which gameweek is the "played" one
    if "match_status" not in df.columns:
        df["match_status"] = "scheduled"

    # GW39 rows have actual minutes > 0 and scores; GW29 have 0 minutes
    gw29_mask = rescheduled_mask & (df["gameweek"] == 29)
    gw39_mask = rescheduled_mask & (df["gameweek"] == 39)

    df.loc[gw29_mask, "match_status"] = "postponed"
    df.loc[gw39_mask, "match_status"] = "played"

    actions = [{
        "fixture_id": 275,
        "season_id": 2019,
        "action": "flagged_rescheduled",
        "description": "Arsenal vs Man City postponed from GW29 (2020-03-11) to GW39 (2020-06-17) due to COVID-19",
        "gw29_rows": int(gw29_mask.sum()),
        "gw39_rows": int(gw39_mask.sum()),
    }]

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Remediated FPL player-match: {original_len} -> {len(df)} rows")
        print(f"Flagged {rescheduled_mask.sum()} rows as rescheduled match")

    return df, actions


# ---- OpenFootball 2025/26 Match Status ----

def remediate_openfootball_matches(
    input_path: Path,
    output_path: Path,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Add match_status to OpenFootball matches to distinguish unplayed future matches.

    The 27 matches with NaN scores in 2025/26 are unplayed future matches.
    Raw data has score: [0, 0] which is football.json's format for unplayed matches.
    """
    df = pd.read_parquet(input_path)
    original_len = len(df)

    if "match_status" not in df.columns:
        df["match_status"] = "scheduled"

    # Identify 2025/26 OpenFootball matches with missing FT scores
    of_2526_mask = (
        (df["season_name"] == "2025/26") &
        (df["source"] == "openfootball/football.json") &
        (df["home_score_ft"].isna()) &
        (df["away_score_ft"].isna())
    )

    df.loc[of_2526_mask, "match_status"] = "unplayed"

    # For other seasons, played matches have scores
    played_mask = (
        (df["source"] == "openfootball/football.json") &
        (df["home_score_ft"].notna()) &
        (df["away_score_ft"].notna())
    )
    df.loc[played_mask, "match_status"] = "played"

    actions = [{
        "season": "2025/26",
        "source": "openfootball/football.json",
        "action": "added_match_status",
        "unplayed_count": int(of_2526_mask.sum()),
        "description": "Unplayed future matches with score [0,0] in raw data converted to NaN",
    }]

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Remediated OpenFootball matches: {original_len} rows, {of_2526_mask.sum()} marked unplayed")

    return df, actions


# ---- Main Orchestration ----

def run_all_remediations(dry_run: bool = False) -> dict:
    """Run all data quality remediations.

    Args:
        dry_run: If True, don't write output files

    Returns:
        Dict with remediation results for each dataset
    """
    results = {}

    # FPL Player-Season
    print("=" * 60)
    print("Remediating FPL player-season duplicates...")
    print("=" * 60)
    fpl_ps_in = DATA_PROCESSED_DIR / "fpl_player_season.parquet"
    fpl_ps_out = DATA_PROCESSED_DIR / "fpl_player_season_remediated.parquet"
    df_ps, actions_ps = remediate_fpl_player_season(fpl_ps_in, fpl_ps_out, dry_run=dry_run)
    results["fpl_player_season"] = {"original_rows": len(pd.read_parquet(fpl_ps_in)), "remediated_rows": len(df_ps), "actions": actions_ps}

    # FPL Player-Match
    print("\n" + "=" * 60)
    print("Remediating FPL player-match rescheduled fixtures...")
    print("=" * 60)
    fpl_pm_in = DATA_PROCESSED_DIR / "fpl_player_match.parquet"
    fpl_pm_out = DATA_PROCESSED_DIR / "fpl_player_match_remediated.parquet"
    df_pm, actions_pm = remediate_fpl_player_match(fpl_pm_in, fpl_pm_out, dry_run=dry_run)
    results["fpl_player_match"] = {"original_rows": len(pd.read_parquet(fpl_pm_in)), "remediated_rows": len(df_pm), "actions": actions_pm}

    # OpenFootball Matches
    print("\n" + "=" * 60)
    print("Remediating OpenFootball 2025/26 match status...")
    print("=" * 60)
    of_in = DATA_PROCESSED_DIR / "matches.parquet"
    of_out = DATA_PROCESSED_DIR / "matches_remediated.parquet"
    df_of, actions_of = remediate_openfootball_matches(of_in, of_out, dry_run=dry_run)
    results["matches"] = {"original_rows": len(pd.read_parquet(of_in)), "remediated_rows": len(df_of), "actions": actions_of}

    return results


if __name__ == "__main__":
    import sys

    dry_run = "--dry-run" in sys.argv
    results = run_all_remediations(dry_run=dry_run)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset, info in results.items():
        print(f"\n{dataset}:")
        print(f"  Original rows: {info['original_rows']}")
        print(f"  Remediated rows: {info['remediated_rows']}")
        for action in info["actions"]:
            print(f"  - {action}")