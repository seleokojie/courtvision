#!/usr/bin/env python3
"""
Enrich shot data from NBA CDN live play-by-play API.

This script fetches additional shot context from the NBA CDN that isn't
available in our base CSV:
- descriptor: pullup, driving, step back, fadeaway, bank, running, floating
- qualifiers: fastbreak, 2ndchance, fromturnover, pointsinthepaint

The CDN has data for games from 2019-20 season onwards (game_ids 219xxxxx+).
"""

import json
import os
import time
from pathlib import Path

import polars as pl
import requests

# Configuration
CDN_BASE_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{}.json"
CACHE_DIR = Path("data/cdn_cache")
OUTPUT_FILE = Path("data/enriched_shots.parquet")
RATE_LIMIT_DELAY = 0.5  # 2 requests per second to avoid rate limiting
MAX_RETRIES = 3

# Games from 2019-20 onwards are available on CDN
# Our game_id format: 21900001 -> CDN format: 0021900001 (add "00" prefix)
MIN_CDN_GAME_ID = 21900000


def get_eligible_game_ids(csv_path: str = "data/nba_plays.csv") -> list[int]:
    """Get game IDs that are available on the CDN (2019+ seasons)."""
    df = pl.scan_csv(csv_path).select("game_id").collect()
    game_ids = df["game_id"].unique().to_list()
    
    # Filter to games available on CDN
    eligible = [g for g in game_ids if g >= MIN_CDN_GAME_ID and g < 22500000]
    return sorted(eligible)


def fetch_game_from_cdn(game_id: int, use_cache: bool = True) -> dict | None:
    """Fetch play-by-play data for a game from NBA CDN."""
    cdn_game_id = f"00{game_id}"
    cache_file = CACHE_DIR / f"{cdn_game_id}.json"
    
    # Check cache first
    if use_cache and cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    
    # Fetch from CDN
    url = CDN_BASE_URL.format(cdn_game_id)
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                
                # Cache the response
                if use_cache:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, "w") as f:
                        json.dump(data, f)
                
                return data
            elif resp.status_code == 403:
                # Game not available on CDN
                return None
            elif resp.status_code == 429:
                # Rate limited - wait and retry
                time.sleep(5)
                continue
            else:
                print(f"  Warning: Game {game_id} returned status {resp.status_code}")
                return None
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                print(f"  Error fetching game {game_id}: {e}")
                return None
    
    return None


def extract_shot_features(game_data: dict) -> list[dict]:
    """Extract shot features from CDN game data."""
    shots = []
    
    game_id = game_data.get("game", {}).get("gameId", "")
    actions = game_data.get("game", {}).get("actions", [])
    
    for action in actions:
        action_type = action.get("actionType", "")
        
        # Only process shots (2pt and 3pt)
        if action_type not in ["2pt", "3pt"]:
            continue
        
        # Extract qualifiers as individual boolean features
        qualifiers = action.get("qualifiers", [])
        
        # Handle shot_distance - sometimes it's a string like "Mid-Range"
        raw_distance = action.get("shotDistance")
        if isinstance(raw_distance, (int, float)):
            shot_distance = int(raw_distance)
        else:
            shot_distance = None  # Will need to compute from x,y if needed
        
        shot = {
            "cdn_game_id": game_id,
            "game_id": int(game_id[2:]) if len(game_id) == 10 else None,  # Remove "00" prefix
            "action_number": action.get("actionNumber"),
            "period": action.get("period"),
            "clock": action.get("clock"),
            "player_id": action.get("personId"),
            "player_name": action.get("playerName"),
            "team_id": action.get("teamId"),
            "team_tricode": action.get("teamTricode"),
            
            # Shot result
            "shot_result": action.get("shotResult"),
            "shot_made": 1 if action.get("shotResult") == "Made" else 0,
            
            # Location
            "shot_distance": shot_distance,
            "x": action.get("x"),
            "y": action.get("y"),
            "area": action.get("area"),
            "area_detail": action.get("areaDetail"),
            
            # Shot type info
            "action_type": action_type,
            "sub_type": action.get("subType"),
            "descriptor": action.get("descriptor"),
            
            # Qualifiers as boolean features
            "is_fastbreak": "fastbreak" in qualifiers,
            "is_second_chance": "2ndchance" in qualifiers,
            "is_from_turnover": "fromturnover" in qualifiers,
            "is_points_in_paint": "pointsinthepaint" in qualifiers,
            
            # Raw qualifiers for reference
            "qualifiers_raw": ",".join(qualifiers) if qualifiers else "",
        }
        
        shots.append(shot)
    
    return shots


def enrich_shots(game_ids: list[int], progress_every: int = 50) -> pl.DataFrame:
    """Fetch and enrich shot data for multiple games."""
    all_shots = []
    successful = 0
    failed = 0
    from_cache = 0
    
    # Check which games are already cached
    cached_ids = set()
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            try:
                gid = int(f.stem[2:])  # Remove "00" prefix
                cached_ids.add(gid)
            except ValueError:
                pass
    
    print(f"Enriching {len(game_ids)} games from NBA CDN...")
    print(f"  Already cached: {len(cached_ids)} games")
    
    for i, game_id in enumerate(game_ids):
        if i > 0 and i % progress_every == 0:
            print(f"  Progress: {i}/{len(game_ids)} games ({successful} ok, {failed} failed, {from_cache} from cache)")
        
        is_cached = game_id in cached_ids
        game_data = fetch_game_from_cdn(game_id, use_cache=True)
        
        if game_data:
            shots = extract_shot_features(game_data)
            all_shots.extend(shots)
            successful += 1
            if is_cached:
                from_cache += 1
        else:
            failed += 1
        
        # Only rate limit network requests (not cache hits)
        if not is_cached:
            time.sleep(RATE_LIMIT_DELAY)
    
    print(f"  Complete: {successful} games enriched, {failed} failed, {from_cache} from cache")
    print(f"  Total shots: {len(all_shots)}")
    
    # Use infer_schema_length=None to scan all records for consistent schema
    return pl.DataFrame(all_shots, infer_schema_length=None)


def main():
    """Main entry point."""
    print("=" * 60)
    print("NBA CDN Data Enrichment Pipeline")
    print("=" * 60)
    
    # Get eligible games
    print("\n1. Finding eligible games...")
    game_ids = get_eligible_game_ids()
    print(f"   Found {len(game_ids)} games available on CDN (2019-2024 seasons)")
    
    # Check cache status
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = len(list(CACHE_DIR.glob("*.json")))
    print(f"   Already cached: {cached} games")
    
    # Enrich shots
    print("\n2. Fetching shot data from CDN...")
    enriched_df = enrich_shots(game_ids)
    
    # Show stats
    print("\n3. Enrichment Statistics:")
    print(f"   Total shots: {len(enriched_df)}")
    print(f"   Unique games: {enriched_df['game_id'].n_unique()}")
    print(f"   Unique players: {enriched_df['player_id'].n_unique()}")
    
    # Descriptor distribution
    print("\n   Descriptor distribution:")
    desc_counts = enriched_df.group_by("descriptor").agg(pl.len().alias("count")).sort("count", descending=True)
    for row in desc_counts.head(10).iter_rows(named=True):
        desc = row["descriptor"] or "(none)"
        print(f"     {desc}: {row['count']:,}")
    
    # Qualifier feature distribution
    print("\n   Qualifier features:")
    for col in ["is_fastbreak", "is_second_chance", "is_from_turnover", "is_points_in_paint"]:
        count = enriched_df.filter(pl.col(col)).height
        pct = 100 * count / len(enriched_df)
        print(f"     {col}: {count:,} ({pct:.1f}%)")
    
    # Save to parquet
    print(f"\n4. Saving to {OUTPUT_FILE}...")
    enriched_df.write_parquet(OUTPUT_FILE)
    print(f"   Saved {len(enriched_df):,} shots to {OUTPUT_FILE}")
    
    print("\n" + "=" * 60)
    print("Enrichment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
