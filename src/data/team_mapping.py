"""Team identity resolution across data providers (Phase 8.2).

Builds a canonical team registry with stable team_ids and mappings between
provider-specific team identifiers (StatsBomb, openfootball/football.json, FPL).

Team names vary across sources:
- StatsBomb: "Manchester United", "Man Utd", etc.
- football.json: "Manchester United FC", "Manchester City FC"
- FPL: "Man Utd", "Man City", "Spurs", "Wolves", etc.

This module provides deterministic, auditable team matching.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data import config

# Manual canonical team name overrides for known discrepancies
# Key: normalized source name -> canonical name
CANONICAL_TEAM_NAMES = {
    # StatsBomb / football.json style
    "manchester united": "Manchester United",
    "manchester city": "Manchester City",
    "tottenham hotspur": "Tottenham Hotspur",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    "west ham united": "West Ham United",
    "west bromwich albion": "West Bromwich Albion",
    "newcastle united": "Newcastle United",
    "nottingham forest": "Nottingham Forest",
    "sheffield united": "Sheffield United",
    "afc bournemouth": "AFC Bournemouth",
    "brighton hove albion": "Brighton & Hove Albion",
    "leeds united": "Leeds United",
    "leicester city": "Leicester City",
    "crystal palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "liverpool": "Liverpool",
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "aston villa": "Aston Villa",
    "brentford": "Brentford",
    "bournemouth": "AFC Bournemouth",
    "burnley": "Burnley",
    "luton town": "Luton Town",
    "sheffield": "Sheffield United",
    "wolves": "Wolverhampton Wanderers",
    "spurs": "Tottenham Hotspur",
    "man utd": "Manchester United",
    "man city": "Manchester City",
    "nottm forest": "Nottingham Forest",
    "nott'm forest": "Nottingham Forest",
    "wba": "West Bromwich Albion",
    "w brom": "West Bromwich Albion",
    "west brom": "West Bromwich Albion",
    "soton": "Southampton",
    "southampton": "Southampton",
    "stoke": "Stoke City",
    "stoke city": "Stoke City",
    "sunderland": "Sunderland",
    "swansea": "Swansea City",
    "swansea city": "Swansea City",
    "watford": "Watford",
    "west ham": "West Ham United",
    "westham": "West Ham United",
}

# Source-specific known aliases (source -> {alias: canonical})
SOURCE_ALIASES = {
    "fpl": {
        "Man Utd": "Manchester United",
        "Man City": "Manchester City",
        "Spurs": "Tottenham Hotspur",
        "Wolves": "Wolverhampton Wanderers",
        "West Brom": "West Bromwich Albion",
        "Nott'm Forest": "Nottingham Forest",
        "Sheff Utd": "Sheffield United",
        "AFC Bournemouth": "AFC Bournemouth",
        "Brighton": "Brighton & Hove Albion",
        "Leeds": "Leeds United",
        "Leicester": "Leicester City",
        "Palace": "Crystal Palace",
        "Everton": "Everton",
        "Fulham": "Fulham",
        "Liverpool": "Liverpool",
        "Arsenal": "Arsenal",
        "Chelsea": "Chelsea",
        "Villa": "Aston Villa",
        "Brentford": "Brentford",
        "Bournemouth": "AFC Bournemouth",
        "Burnley": "Burnley",
        "Luton": "Luton Town",
        "Sheff Utd": "Sheffield United",
        "West Ham": "West Ham United",
    },
    "football.json": {
        "Manchester United FC": "Manchester United",
        "Manchester City FC": "Manchester City",
        "Tottenham Hotspur FC": "Tottenham Hotspur",
        "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
        "West Ham United FC": "West Ham United",
        "West Bromwich Albion FC": "West Bromwich Albion",
        "Newcastle United FC": "Newcastle United",
        "Nottingham Forest FC": "Nottingham Forest",
        "Sheffield United FC": "Sheffield United",
        "AFC Bournemouth": "AFC Bournemouth",
        "Brighton & Hove Albion FC": "Brighton & Hove Albion",
        "Leeds United FC": "Leeds United",
        "Leicester City FC": "Leicester City",
        "Crystal Palace FC": "Crystal Palace",
        "Everton FC": "Everton",
        "Fulham FC": "Fulham",
        "Liverpool FC": "Liverpool",
        "Arsenal FC": "Arsenal",
        "Chelsea FC": "Chelsea",
        "Aston Villa FC": "Aston Villa",
        "Brentford FC": "Brentford",
        "Burnley FC": "Burnley",
        "Luton Town FC": "Luton Town",
        "Southampton FC": "Southampton",
        "Stoke City FC": "Stoke City",
        "Sunderland FC": "Sunderland",
        "Swansea City FC": "Swansea City",
        "Watford FC": "Watford",
        "West Ham United FC": "West Ham United",
        "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
        "Norwich City FC": "Norwich City",
        "Watford FC": "Watford",
        "West Bromwich Albion FC": "West Bromwich Albion",
        "Leeds United FC": "Leeds United",
        "Leicester City FC": "Leicester City",
    },
    "statsbomb": {
        "Manchester United": "Manchester United",
        "Manchester City": "Manchester City",
        "Tottenham Hotspur": "Tottenham Hotspur",
        "Wolverhampton Wanderers": "Wolverhampton Wanderers",
        "West Ham United": "West Ham United",
        "West Bromwich Albion": "West Bromwich Albion",
        "Newcastle United": "Newcastle United",
        "Nottingham Forest": "Nottingham Forest",
        "Sheffield United": "Sheffield United",
        "AFC Bournemouth": "AFC Bournemouth",
        "Brighton & Hove Albion": "Brighton & Hove Albion",
        "Leeds United": "Leeds United",
        "Leicester City": "Leicester City",
        "Crystal Palace": "Crystal Palace",
        "Everton": "Everton",
        "Fulham": "Fulham",
        "Liverpool": "Liverpool",
        "Arsenal": "Arsenal",
        "Chelsea": "Chelsea",
        "Aston Villa": "Aston Villa",
        "Brentford": "Brentford",
        "Bournemouth": "AFC Bournemouth",
        "Burnley": "Burnley",
        "Luton Town": "Luton Town",
        "Southampton": "Southampton",
        "Stoke City": "Stoke City",
        "Sunderland": "Sunderland",
        "Swansea City": "Swansea City",
        "Watford": "Watford",
        "West Bromwich Albion": "West Bromwich Albion",
        "Norwich City": "Norwich City",
        "Middlesbrough": "Middlesbrough",
        "Hull City": "Hull City",
        "Queens Park Rangers": "Queens Park Rangers",
        "Reading": "Reading",
        "Blackburn Rovers": "Blackburn Rovers",
        "Bolton Wanderers": "Bolton Wanderers",
        "Charlton Athletic": "Charlton Athletic",
        "Portsmouth": "Portsmouth",
        "Birmingham City": "Birmingham City",
        "Blackpool": "Blackpool",
        "Derby County": "Derby County",
        "Ipswich Town": "Ipswich Town",
        "Cardiff City": "Cardiff City",
        "Huddersfield Town": "Huddersfield Town",
        "Millwall": "Millwall",
        "Barnsley": "Barnsley",
        "Peterborough United": "Peterborough United",
        "Bristol City": "Bristol City",
        "Preston North End": "Preston North End",
        "Queens Park Rangers": "Queens Park Rangers",
        "Sheffield Wednesday": "Sheffield Wednesday",
        "Nottingham Forest": "Nottingham Forest",
        "Coventry City": "Coventry City",
        "Swansea City": "Swansea City",
        "Middlesbrough": "Middlesbrough",
    },
}


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching: lowercase, remove suffixes, strip."""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    # Remove common suffixes
    for suffix in [" fc", " afc", " utd", " united", " city", " town", " rovers", " wanderers", " albion", " athletic", " county"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
    # Remove special chars
    s = s.replace("&", "and").replace(".", "").replace("'", "")
    return s.strip()


def get_canonical_name(name: str, source: Optional[str] = None) -> str:
    """Map a source team name to canonical name.

    Args:
        name: Raw team name from a source.
        source: Optional source hint ('fpl', 'football.json', 'statsbomb').

    Returns:
        Canonical team name, or empty string if no mapping found.
    """
    if not isinstance(name, str):
        return ""

    # Try source-specific aliases first
    if source and source in SOURCE_ALIASES:
        if name in SOURCE_ALIASES[source]:
            return SOURCE_ALIASES[source][name]

    # Try normalized matching
    norm = normalize_team_name(name)
    if norm in CANONICAL_TEAM_NAMES:
        return CANONICAL_TEAM_NAMES[norm]

    # Return original if no mapping
    return name


def build_team_registry(
    statsbomb_teams: Optional[pd.DataFrame] = None,
    football_json_teams: Optional[pd.DataFrame] = None,
    fpl_teams: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a unified team registry with stable team_ids and cross-provider mappings.

    Args:
        statsbomb_teams: DataFrame with 'team_id', 'team_name' from StatsBomb.
        football_json_teams: DataFrame with 'team_id', 'team_name' from football.json.
        fpl_teams: DataFrame with 'team' (FPL team_id), 'team_name' from FPL master_team_list.

    Returns:
        DataFrame with columns:
        - team_id: stable canonical team ID (hash-based)
        - canonical_name: canonical team name
        - statsbomb_team_id: StatsBomb team_id if matched
        - statsbomb_team_name: StatsBomb team_name if matched
        - football_json_team_id: football.json team_id (hash) if matched
        - football_json_team_name: football.json team_name if matched
        - fpl_team_id: FPL team ID if matched
        - fpl_team_name: FPL team_name if matched
        - match_confidence: 'exact' | 'normalized' | 'manual' | 'unmatched'
        - match_notes: explanation of matching
    """
    all_teams = []

    # Collect all team names from all sources with provenance
    source_teams = []

    if statsbomb_teams is not None and not statsbomb_teams.empty:
        for _, row in statsbomb_teams.iterrows():
            source_teams.append({
                "source": "statsbomb",
                "source_id": row.get("team_id"),
                "source_name": row.get("team_name"),
                "canonical": get_canonical_name(row.get("team_name", ""), "statsbomb"),
            })

    if football_json_teams is not None and not football_json_teams.empty:
        for _, row in football_json_teams.iterrows():
            source_teams.append({
                "source": "football.json",
                "source_id": row.get("team_id"),
                "source_name": row.get("team_name"),
                "canonical": get_canonical_name(row.get("team_name", ""), "football.json"),
            })

    if fpl_teams is not None and not fpl_teams.empty:
        for _, row in fpl_teams.iterrows():
            source_teams.append({
                "source": "fpl",
                "source_id": row.get("team"),
                "source_name": row.get("team_name"),
                "canonical": get_canonical_name(row.get("team_name", ""), "fpl"),
            })

    if not source_teams:
        return pd.DataFrame()

    # Group by canonical name
    from collections import defaultdict
    canonical_groups = defaultdict(list)
    for st in source_teams:
        canonical_groups[st["canonical"]].append(st)

    # Build registry
    registry_rows = []
    for canonical_name, entries in canonical_groups.items():
        # Determine match confidence
        sources_present = {e["source"] for e in entries}
        if len(entries) > 1:
            confidence = "exact" if all(e["source_name"].lower() == canonical_name.lower() for e in entries) else "normalized"
        else:
            confidence = "single_source"

        # Generate stable team_id from canonical name
        team_id = int(hashlib.md5(canonical_name.lower().encode()).hexdigest()[:8], 16)

        row = {
            "team_id": team_id,
            "canonical_name": canonical_name,
            "match_confidence": confidence,
            "match_notes": f"Matched from {len(entries)} source(s): {', '.join(sources_present)}",
            "statsbomb_team_id": None,
            "statsbomb_team_name": None,
            "football_json_team_id": None,
            "football_json_team_name": None,
            "fpl_team_id": None,
            "fpl_team_name": None,
        }

        for e in entries:
            if e["source"] == "statsbomb":
                row["statsbomb_team_id"] = e["source_id"]
                row["statsbomb_team_name"] = e["source_name"]
            elif e["source"] == "football.json":
                row["football_json_team_id"] = e["source_id"]
                row["football_json_team_name"] = e["source_name"]
            elif e["source"] == "fpl":
                row["fpl_team_id"] = e["source_id"]
                row["fpl_team_name"] = e["source_name"]

        registry_rows.append(row)

    registry_df = pd.DataFrame(registry_rows)
    registry_df = registry_df.sort_values("team_id").reset_index(drop=True)

    return registry_df


def load_team_registry(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load team registry from parquet file."""
    path = path or (config.DATA_PROCESSED_DIR / "team_registry.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_team_registry(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Save team registry to parquet file."""
    path = path or (config.DATA_PROCESSED_DIR / "team_registry.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def map_team_id(
    source_name: str,
    source: str,
    registry: pd.DataFrame,
) -> Optional[int]:
    """Look up canonical team_id for a source team name.

    Args:
        source_name: Team name from the source.
        source: Source identifier ('statsbomb', 'football.json', 'fpl').
        registry: Team registry DataFrame from build_team_registry.

    Returns:
        Canonical team_id, or None if not found.
    """
    if registry is None or registry.empty:
        return None

    canonical = get_canonical_name(source_name, source)
    match = registry[registry["canonical_name"] == canonical]
    if not match.empty:
        return int(match.iloc[0]["team_id"])

    # Try fuzzy match on source-specific columns
    col_map = {
        "statsbomb": "statsbomb_team_name",
        "football.json": "football_json_team_name",
        "fpl": "fpl_team_name",
    }
    col = col_map.get(source)
    if col and col in registry.columns:
        match = registry[registry[col] == source_name]
        if not match.empty:
            return int(match.iloc[0]["team_id"])

    return None


def get_team_name(team_id: int, registry: pd.DataFrame) -> Optional[str]:
    """Get canonical team name for a team_id."""
    if registry is None or registry.empty:
        return None
    match = registry[registry["team_id"] == team_id]
    if not match.empty:
        return match.iloc[0]["canonical_name"]
    return None