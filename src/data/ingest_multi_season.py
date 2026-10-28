"""Unified ingestion runner for Phase 8.2 multi-season data.

Coordinates provider-specific adapters and builds cross-provider registries.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data import config
from src.data.ingest_openfootball_matches import run as run_openfootball_matches
from src.data.ingest_fpl import run as run_fpl
from src.data.team_mapping import build_team_registry, save_team_registry, load_team_registry, get_canonical_name
from src.data.player_mapping import build_player_registry, save_player_registry, load_player_registry, export_unmatched_for_review

logger = logging.getLogger("plintel.ingest.multi_season")


def run_full_ingestion(
    offline: bool = False,
    timeout: int = 60,
    skip_fpl: bool = False,
    skip_openfootball: bool = False,
    fpl_seasons: Optional[list[str]] = None,
    openfootball_seasons: Optional[list[str]] = None,
) -> dict:
    """Run the complete Phase 8.2 multi-season ingestion pipeline.

    Args:
        offline: Use cached raw files only.
        timeout: HTTP timeout in seconds.
        skip_fpl: Skip FPL player data ingestion.
        skip_openfootball: Skip OpenFootball match data ingestion.
        fpl_seasons: Specific FPL seasons to process.
        openfootball_seasons: Specific OpenFootball seasons to process.

    Returns:
        Summary dict with all results.
    """
    summary = {
        "openfootball": {},
        "fpl": {},
        "team_registry": {},
        "player_registry": {},
        "overall_success": True,
    }

    # 1. Ingest OpenFootball match results
    if not skip_openfootball:
        logger.info("=" * 60)
        logger.info("STEP 1: Ingesting OpenFootball match results")
        logger.info("=" * 60)
        try:
            of_summary = run_openfootball_matches(
                offline=offline,
                timeout=timeout,
                seasons=openfootball_seasons,
            )
            summary["openfootball"] = of_summary
        except Exception as exc:
            logger.error("OpenFootball ingestion failed: %s", exc)
            summary["openfootball"] = {"error": str(exc)}
            summary["overall_success"] = False
    else:
        logger.info("Skipping OpenFootball ingestion (--skip-openfootball)")

    # 2. Ingest FPL player data
    if not skip_fpl:
        logger.info("=" * 60)
        logger.info("STEP 2: Ingesting FPL player data")
        logger.info("=" * 60)
        try:
            fpl_summary = run_fpl(
                offline=offline,
                timeout=timeout,
                seasons=fpl_seasons,
            )
            summary["fpl"] = fpl_summary
        except Exception as exc:
            logger.error("FPL ingestion failed: %s", exc)
            summary["fpl"] = {"error": str(exc)}
            summary["overall_success"] = False
    else:
        logger.info("Skipping FPL ingestion (--skip-fpl)")

    # 3. Build team registry
    logger.info("=" * 60)
    logger.info("STEP 3: Building team registry")
    logger.info("=" * 60)

    # Load source team data
    of_teams = None
    of_matches_path = config.DATA_PROCESSED_DIR / "matches.parquet"
    if of_matches_path.exists():
        of_matches = pd.read_parquet(of_matches_path)
        # Build teams df from matches - use canonical names for IDs
        from src.data.team_mapping import normalize_team_name, get_canonical_name
        all_names = pd.concat([of_matches["home_team_name"], of_matches["away_team_name"]]).unique()
        of_teams = pd.DataFrame({
            "team_id": [int(hashlib.md5(get_canonical_name(n, "football.json").lower().encode()).hexdigest()[:8], 16) for n in all_names],
            "team_name": all_names,
        })
        # Deduplicate by team_id (same canonical name -> same ID)
        of_teams = of_teams.drop_duplicates(subset=["team_id"]).reset_index(drop=True)

    fpl_teams = None
    fpl_master_path = config.DATA_RAW_DIR / "fpl" / "master_team_list.csv"
    if fpl_master_path.exists():
        fpl_master = pd.read_csv(fpl_master_path)
        # Deduplicate by team_id (keep first occurrence)
        fpl_teams = fpl_master.drop_duplicates(subset=["team"]).reset_index(drop=True)

    # StatsBomb teams (from existing 2015/16 data)
    sb_teams = None
    sb_fixtures_path = config.STATSBOMB_PROCESSED_DIR / "fixtures.parquet"
    if sb_fixtures_path.exists():
        sb_fixtures = pd.read_parquet(sb_fixtures_path)
        sb_teams = pd.DataFrame({
            "team_id": pd.concat([sb_fixtures["home_team_id"], sb_fixtures["away_team_id"]]).unique(),
            "team_name": pd.concat([sb_fixtures["home_team_name"], sb_fixtures["away_team_name"]]).unique(),
        })

    team_registry = build_team_registry(
        statsbomb_teams=sb_teams,
        football_json_teams=of_teams,
        fpl_teams=fpl_teams,
    )
    registry_path = save_team_registry(team_registry)
    logger.info("Built team registry with %d teams -> %s", len(team_registry), registry_path)
    summary["team_registry"] = {
        "teams": len(team_registry),
        "path": str(registry_path),
    }

    # 4. Build player registry
    logger.info("=" * 60)
    logger.info("STEP 4: Building player registry")
    logger.info("=" * 60)

    sb_players = None
    sb_players_path = config.STATSBOMB_PROCESSED_DIR / "player_season.parquet"
    if sb_players_path.exists():
        sb_players = pd.read_parquet(sb_players_path)

    fpl_players = None
    fpl_player_season_path = config.DATA_PROCESSED_DIR / "fpl_player_season.parquet"
    if fpl_player_season_path.exists():
        fpl_players = pd.read_parquet(fpl_player_season_path)

    player_registry = build_player_registry(
        statsbomb_players=sb_players,
        fpl_players=fpl_players,
    )
    player_registry_path = save_player_registry(player_registry)
    logger.info("Built player registry with %d players -> %s", len(player_registry), player_registry_path)
    summary["player_registry"] = {
        "players": len(player_registry),
        "path": str(player_registry_path),
        "confidence_distribution": player_registry["match_confidence"].value_counts().to_dict(),
    }

    # 5. Export unmatched players for review
    if sb_players is not None and fpl_players is not None:
        review_path = config.REPORTS_DIR / "player_mapping_unmatched.csv"
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        n_unmatched = export_unmatched_for_review(
            sb_players, fpl_players, player_registry, review_path
        )
        logger.info("Exported %d unmatched players for review -> %s", n_unmatched, review_path)
        summary["player_registry"]["unmatched_for_review"] = n_unmatched
        summary["player_registry"]["review_path"] = str(review_path)

    return summary


def run_team_registry_only(
    offline: bool = False,
    timeout: int = 60,
) -> dict:
    """Build team registry from existing processed data (no new ingestion)."""
    logger.info("Building team registry from existing processed data...")

    # Load source team data
    of_teams = None
    of_matches_path = config.DATA_PROCESSED_DIR / "matches.parquet"
    if of_matches_path.exists():
        of_matches = pd.read_parquet(of_matches_path)
        # Build teams df from matches - use canonical names for IDs
        all_names = pd.concat([of_matches["home_team_name"], of_matches["away_team_name"]]).unique()
        of_teams = pd.DataFrame({
            "team_id": [int(hashlib.md5(get_canonical_name(n, "football.json").lower().encode()).hexdigest()[:8], 16) for n in all_names],
            "team_name": all_names,
        })
        # Deduplicate by team_id (same canonical name -> same ID)
        of_teams = of_teams.drop_duplicates(subset=["team_id"]).reset_index(drop=True)

    fpl_teams = None
    fpl_master_path = config.DATA_RAW_DIR / "fpl" / "master_team_list.csv"
    if fpl_master_path.exists():
        fpl_master = pd.read_csv(fpl_master_path)
        # Deduplicate by team_id (keep first occurrence)
        fpl_teams = fpl_master.drop_duplicates(subset=["team"]).reset_index(drop=True)

    sb_teams = None
    sb_fixtures_path = config.STATSBOMB_PROCESSED_DIR / "fixtures.parquet"
    if sb_fixtures_path.exists():
        sb_fixtures = pd.read_parquet(sb_fixtures_path)
        sb_teams = pd.DataFrame({
            "team_id": pd.concat([sb_fixtures["home_team_id"], sb_fixtures["away_team_id"]]).unique(),
            "team_name": pd.concat([sb_fixtures["home_team_name"], sb_fixtures["away_team_name"]]).unique(),
        })

    team_registry = build_team_registry(
        statsbomb_teams=sb_teams,
        football_json_teams=of_teams,
        fpl_teams=fpl_teams,
    )
    registry_path = save_team_registry(team_registry)

    return {
        "teams": len(team_registry),
        "path": str(registry_path),
    }


def run_player_registry_only() -> dict:
    """Build player registry from existing processed data."""
    logger.info("Building player registry from existing processed data...")

    sb_players = None
    sb_players_path = config.STATSBOMB_PROCESSED_DIR / "player_season.parquet"
    if sb_players_path.exists():
        sb_players = pd.read_parquet(sb_players_path)

    fpl_players = None
    fpl_player_season_path = config.DATA_PROCESSED_DIR / "fpl_player_season.parquet"
    if fpl_player_season_path.exists():
        fpl_players = pd.read_parquet(fpl_player_season_path)

    player_registry = build_player_registry(
        statsbomb_players=sb_players,
        fpl_players=fpl_players,
    )
    registry_path = save_player_registry(player_registry)

    return {
        "players": len(player_registry),
        "path": str(registry_path),
        "confidence_distribution": player_registry["match_confidence"].value_counts().to_dict(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 8.2 multi-season ingestion pipeline.")
    parser.add_argument("--offline", action="store_true", help="use cached raw files only")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--skip-fpl", action="store_true", help="skip FPL player data ingestion")
    parser.add_argument("--skip-openfootball", action="store_true", help="skip OpenFootball match data ingestion")
    parser.add_argument("--fpl-season", action="append", dest="fpl_seasons", help="specific FPL season(s)")
    parser.add_argument("--openfootball-season", action="append", dest="openfootball_seasons", help="specific OpenFootball season(s)")
    parser.add_argument("--team-registry-only", action="store_true", help="only build team registry from existing data")
    parser.add_argument("--player-registry-only", action="store_true", help="only build player registry from existing data")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        if args.team_registry_only:
            summary = run_team_registry_only(offline=args.offline, timeout=args.timeout)
        elif args.player_registry_only:
            summary = run_player_registry_only()
        else:
            summary = run_full_ingestion(
                offline=args.offline,
                timeout=args.timeout,
                skip_fpl=args.skip_fpl,
                skip_openfootball=args.skip_openfootball,
                fpl_seasons=args.fpl_seasons,
                openfootball_seasons=args.openfootball_seasons,
            )
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    logger.info("Pipeline complete: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())