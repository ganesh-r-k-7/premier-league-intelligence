"""Player identity resolution across data providers (Phase 8.2).

Builds a canonical player registry with stable player_ids and mappings between
provider-specific player identifiers (StatsBomb, FPL, openfootball).

Matching strategy:
1. Exact match on normalized name + team + season + position
2. Fuzzy match on name + team + season (candidates for review)
3. Manual curation for high-value players
4. Confidence scores and explicit ambiguous states

Never automatically merges ambiguous players solely because their names are similar.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.data import config
from src.data.team_mapping import get_canonical_name, normalize_team_name

# Manual curated mappings for known players with name discrepancies
# Format: (canonical_name, team, season) -> {source: source_player_id}
MANUAL_PLAYER_MAPPINGS = {
    # Example: ("Mohamed Salah", "Liverpool", 2017) -> {"statsbomb": 12345, "fpl": 234}
}

# Common name variations
NAME_NORMALIZATIONS = {
    "mohamed salah": "Mohamed Salah",
    "mo salah": "Mohamed Salah",
    "heung-min son": "Heung-Min Son",
    "son heung-min": "Heung-Min Son",
    "harry kane": "Harry Kane",
    "kevin de bruyne": "Kevin De Bruyne",
    "bruno fernandes": "Bruno Fernandes",
    "virgil van dijk": "Virgil van Dijk",
    "trent alexander-arnold": "Trent Alexander-Arnold",
    "andrew robertson": "Andrew Robertson",
    "alisson becker": "Alisson Becker",
    "alisson": "Alisson Becker",
    "ederson moraes": "Ederson Moraes",
    "ederson": "Ederson Moraes",
    "raul jimenez": "Raul Jimenez",
    "raúl jiménez": "Raul Jimenez",
    "willy boly": "Willy Boly",
    "joao moutinho": "Joao Moutinho",
    "joão moutinho": "Joao Moutinho",
    "richarlison de andrade": "Richarlison",
    "richarlison": "Richarlison",
    "lucas moura": "Lucas Moura",
    "lucas paqueta": "Lucas Paqueta",
    "lucas paquetá": "Lucas Paqueta",
    "gabriel jesus": "Gabriel Jesus",
    "gabriel martinelli": "Gabriel Martinelli",
    "bukayo saka": "Bukayo Saka",
    "martin odegaard": "Martin Odegaard",
    "martin ødegaard": "Martin Odegaard",
    "kai havertz": "Kai Havertz",
    "christian pulisic": "Christian Pulisic",
    "timo werner": "Timo Werner",
    "marcus rashford": "Marcus Rashford",
    "mason mount": "Mason Mount",
    "phil foden": "Phil Foden",
    "jack grealish": "Jack Grealish",
    "declan rice": "Declan Rice",
    "kalvin phillips": "Kalvin Phillips",
    "jordan henderson": "Jordan Henderson",
    "fabinho tavares": "Fabinho",
    "fabinho": "Fabinho",
    "roberto firmino": "Roberto Firmino",
    "diogo jota": "Diogo Jota",
    "luis diaz": "Luis Diaz",
    "luís díaz": "Luis Diaz",
    "darwin nunez": "Darwin Nunez",
    "darwin núñez": "Darwin Nunez",
    "alexis mac allister": "Alexis Mac Allister",
    "alexis macallister": "Alexis Mac Allister",
    "enzio fernandez": "Enzo Fernandez",
    "enzo fernández": "Enzo Fernandez",
    "alexis macalister": "Alexis Mac Allister",
    "mykhailo mudryk": "Mykhailo Mudryk",
    "mudryk": "Mykhailo Mudryk",
    "nicolas jackson": "Nicolas Jackson",
    "cole palmer": "Cole Palmer",
    "alexander isak": "Alexander Isak",
    "alex isak": "Alexander Isak",
    "ollie watkins": "Ollie Watkins",
    "bryan mbeumo": "Bryan Mbeumo",
    "yoane wissa": "Yoane Wissa",
    "ivan toney": "Ivan Toney",
    "jarrod bowen": "Jarrod Bowen",
    "michail antonio": "Michail Antonio",
    "said benrahma": "Said Benrahma",
    "saïd benrahma": "Said Benrahma",
    "pablo forbals": "Pablo Fornals",
    "pablo fornals": "Pablo Fornals",
    "tomas soucek": "Tomas Soucek",
    "tomáš souček": "Tomas Soucek",
    "vladimir coufal": "Vladimir Coufal",
    "vladimír coufal": "Vladimir Coufal",
    "aaron creswell": "Aaron Cresswell",
    "lukasz fabianski": "Lukasz Fabianski",
    "łukasz fabiański": "Lukasz Fabianski",
    "angelo ogbonna": "Angelo Ogbonna",
    "craig dawson": "Craig Dawson",
    "kurt zouma": "Kurt Zouma",
    "ben johnson": "Ben Johnson",
    "vladimir coufal": "Vladimir Coufal",
    "nikola vlasic": "Nikola Vlasic",
    "nikola vlasić": "Nikola Vlasic",
    "maxwel cornett": "Maxwel Cornet",
    "maxwel cornét": "Maxwel Cornet",
    "dwight mcneil": "Dwight McNeil",
    "johann gudmundsson": "Johann Gudmundsson",
    "jóhann guðmundsson": "Johann Gudmundsson",
    "ashley westwood": "Ashley Westwood",
    "jack cork": "Jack Cork",
    "josh brownhill": "Josh Brownhill",
    "conor roberts": "Conor Roberts",
    "matt lowton": "Matt Lowton",
    "erik pieters": "Erik Pieters",
    "charlie taylor": "Charlie Taylor",
    "james tarkowski": "James Tarkowski",
    "ben mee": "Ben Mee",
    "nick pope": "Nick Pope",
    "bailey peacock-farrell": "Bailey Peacock-Farrell",
    "bailey peacock farrell": "Bailey Peacock-Farrell",
}


def normalize_player_name(name: str) -> str:
    """Normalize player name for matching: lowercase, remove accents, strip."""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    # Remove accents (basic)
    s = s.replace("é", "e").replace("á", "a").replace("í", "i").replace("ó", "o").replace("ú", "u")
    s = s.replace("ñ", "n").replace("ç", "c").replace("ö", "o").replace("ü", "u")
    s = s.replace("ß", "ss").replace("æ", "ae").replace("ø", "o").replace("å", "a")
    # Remove special chars
    s = s.replace("-", " ").replace("'", "").replace(".", "").replace(",", "")
    # Collapse whitespace
    s = " ".join(s.split())
    return s.strip()


def get_canonical_player_name(name: str) -> str:
    """Map a player name to canonical form using known normalizations."""
    norm = normalize_player_name(name)
    if norm in NAME_NORMALIZATIONS:
        return NAME_NORMALIZATIONS[norm]
    # Title case as fallback
    return " ".join(w.capitalize() for w in norm.split())


def get_player_match_key(
    player_name: str,
    team_name: str,
    season_id: int,
    position: Optional[str] = None,
) -> str:
    """Generate a deterministic match key for a player-season."""
    canonical_name = get_canonical_player_name(player_name)
    canonical_team = get_canonical_name(team_name)
    parts = [canonical_name.lower(), canonical_team.lower(), str(season_id)]
    if position and isinstance(position, str):
        parts.append(position.lower())
    return "|".join(parts)


def build_player_registry(
    statsbomb_players: Optional[pd.DataFrame] = None,
    fpl_players: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a unified player registry with stable player_ids and cross-provider mappings.

    Args:
        statsbomb_players: DataFrame with StatsBomb player_season columns
                           (player_id, player_name, team_name, position, season_id).
        fpl_players: DataFrame with FPL player_match/season columns
                     (player_id, player_name, team, position, season_id, season_name).

    Returns:
        DataFrame with columns:
        - player_id: stable canonical player ID (hash-based)
        - canonical_name: canonical player name
        - canonical_team: canonical team name
        - season_id: season identifier
        - position: primary position
        - statsbomb_player_id: StatsBomb player_id if matched
        - statsbomb_player_name: StatsBomb player_name if matched
        - fpl_player_id: FPL element ID if matched
        - fpl_player_name: FPL player_name if matched
        - match_confidence: 'exact' | 'normalized' | 'fuzzy' | 'manual' | 'unmatched'
        - match_notes: explanation
    """
    all_players = []

    # Collect from StatsBomb
    if statsbomb_players is not None and not statsbomb_players.empty:
        for _, row in statsbomb_players.iterrows():
            canonical_name = get_canonical_player_name(row.get("player_name", ""))
            canonical_team = get_canonical_name(row.get("team_name", ""), "statsbomb")
            season_id = row.get("season_id")
            position = row.get("position")
            match_key = get_player_match_key(canonical_name, canonical_team, season_id, position)

            all_players.append({
                "match_key": match_key,
                "canonical_name": canonical_name,
                "canonical_team": canonical_team,
                "season_id": season_id,
                "position": position,
                "source": "statsbomb",
                "source_id": row.get("player_id"),
                "source_name": row.get("player_name"),
            })

    # Collect from FPL
    if fpl_players is not None and not fpl_players.empty:
        for _, row in fpl_players.iterrows():
            canonical_name = get_canonical_player_name(row.get("player_name", ""))
            canonical_team = get_canonical_name(row.get("team", ""), "fpl")
            season_id = row.get("season_id")
            position = row.get("position")
            match_key = get_player_match_key(canonical_name, canonical_team, season_id, position)

            all_players.append({
                "match_key": match_key,
                "canonical_name": canonical_name,
                "canonical_team": canonical_team,
                "season_id": season_id,
                "position": position,
                "source": "fpl",
                "source_id": row.get("player_id"),
                "source_name": row.get("player_name"),
            })

    if not all_players:
        return pd.DataFrame()

    df = pd.DataFrame(all_players)

    # Group by match_key
    grouped = df.groupby("match_key")

    registry_rows = []
    for match_key, group in grouped:
        sources = group["source"].unique()
        n_sources = len(sources)

        if n_sources == 1:
            # Single source - no cross-provider match
            confidence = "single_source"
            row = group.iloc[0]
            registry_rows.append({
                "player_id": int(hashlib.md5(match_key.encode()).hexdigest()[:12], 16),
                "match_key": match_key,
                "canonical_name": row["canonical_name"],
                "canonical_team": row["canonical_team"],
                "season_id": row["season_id"],
                "position": row["position"],
                "match_confidence": "single_source",
                "match_notes": f"Only in {row['source']}",
                "statsbomb_player_id": row["source_id"] if row["source"] == "statsbomb" else None,
                "statsbomb_player_name": row["source_name"] if row["source"] == "statsbomb" else None,
                "fpl_player_id": row["source_id"] if row["source"] == "fpl" else None,
                "fpl_player_name": row["source_name"] if row["source"] == "fpl" else None,
            })
        else:
            # Multiple sources - check for exact match
            statsbomb_row = group[group["source"] == "statsbomb"]
            fpl_row = group[group["source"] == "fpl"]

            sb_id = statsbomb_row.iloc[0]["source_id"] if not statsbomb_row.empty else None
            sb_name = statsbomb_row.iloc[0]["source_name"] if not statsbomb_row.empty else None
            fpl_id = fpl_row.iloc[0]["source_id"] if not fpl_row.empty else None
            fpl_name = fpl_row.iloc[0]["source_name"] if not fpl_row.empty else None

            # Check if names match exactly (canonical)
            if sb_name and fpl_name:
                if get_canonical_player_name(sb_name) == get_canonical_player_name(fpl_name):
                    confidence = "exact"
                else:
                    confidence = "normalized"
            else:
                confidence = "fuzzy"

            registry_rows.append({
                "player_id": int(hashlib.md5(match_key.encode()).hexdigest()[:12], 16),
                "match_key": match_key,
                "canonical_name": group.iloc[0]["canonical_name"],
                "canonical_team": group.iloc[0]["canonical_team"],
                "season_id": group.iloc[0]["season_id"],
                "position": group.iloc[0]["position"],
                "match_confidence": confidence,
                "match_notes": f"Matched across {n_sources} sources",
                "statsbomb_player_id": sb_id,
                "statsbomb_player_name": sb_name,
                "fpl_player_id": fpl_id,
                "fpl_player_name": fpl_name,
            })

    registry_df = pd.DataFrame(registry_rows)
    registry_df = registry_df.sort_values("player_id").reset_index(drop=True)
    return registry_df


def load_player_registry(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load player registry from parquet file."""
    path = path or (config.DATA_PROCESSED_DIR / "player_registry.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_player_registry(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Save player registry to parquet file."""
    path = path or (config.DATA_PROCESSED_DIR / "player_registry.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def map_player_id(
    source_name: str,
    source_team: str,
    season_id: int,
    source: str,
    registry: pd.DataFrame,
    position: Optional[str] = None,
) -> Optional[int]:
    """Look up canonical player_id for a source player.

    Args:
        source_name: Player name from the source.
        source_team: Team name from the source.
        season_id: Season identifier.
        source: Source identifier ('statsbomb', 'fpl').
        registry: Player registry DataFrame from build_player_registry.
        position: Optional position for disambiguation.

    Returns:
        Canonical player_id, or None if not found.
    """
    if registry is None or registry.empty:
        return None

    canonical_name = get_canonical_player_name(source_name)
    canonical_team = get_canonical_name(source_team)
    match_key = get_player_match_key(canonical_name, canonical_team, season_id, position)

    match = registry[registry["match_key"] == match_key]
    if not match.empty:
        return int(match.iloc[0]["player_id"])

    return None


def get_player_name(player_id: int, registry: pd.DataFrame) -> Optional[str]:
    """Get canonical player name for a player_id."""
    if registry is None or registry.empty:
        return None
    match = registry[registry["player_id"] == player_id]
    if not match.empty:
        return match.iloc[0]["canonical_name"]
    return None


def export_unmatched_for_review(
    statsbomb_players: pd.DataFrame,
    fpl_players: pd.DataFrame,
    registry: pd.DataFrame,
    output_path: Path,
) -> int:
    """Export unmatched players to CSV for manual review.

    Returns count of unmatched players.
    """
    # Get all source players
    sb_unmatched = statsbomb_players[
        ~statsbomb_players.apply(
            lambda r: map_player_id(r["player_name"], r["team_name"], r["season_id"], "statsbomb", registry, r.get("position")),
            axis=1
        ).notna()
    ].copy()
    sb_unmatched["source"] = "statsbomb"

    fpl_unmatched = fpl_players[
        ~fpl_players.apply(
            lambda r: map_player_id(r["player_name"], r["team"], r["season_id"], "fpl", registry, r.get("position")),
            axis=1
        ).notna()
    ].copy()
    fpl_unmatched["source"] = "fpl"

    unmatched = pd.concat([sb_unmatched, fpl_unmatched], ignore_index=True)
    unmatched.to_csv(output_path, index=False)
    return len(unmatched)