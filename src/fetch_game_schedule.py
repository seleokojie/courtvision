#!/usr/bin/env python3
"""
Fetch game schedule (game_id -> date mapping) from NBA CDN.

This extracts game dates from the CDN play-by-play data, building a lookup
table that allows us to join shot data with injury reports by date.

The CDN play-by-play JSON includes a 'gameTimeUTC' field we can parse.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import requests

# Configuration
CDN_BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{}.json"
SCHEDULE_OUTPUT = Path("data/game_schedule.parquet")
RATE_LIMIT_DELAY = 0.3
MAX_RETRIES = 3

# Games from 2019-20 onwards are available on CDN
MIN_CDN_GAME_ID = 21900000


def get_game_ids_from_plays(csv_path: str = "data/nba_plays.csv") -> list[int]:
    """Get unique game IDs from the plays data."""
    df = pl.scan_csv(csv_path).select("game_id").collect()
    game_ids = df["game_id"].unique().to_list()
    
    # Filter to games in injury data date range (2021-2024) and available on CDN
    # Season 2021-22 starts with 22100001, 2022-23 with 22200001, 2023-24 with 22300001
    eligible = [g for g in game_ids if 22100000 <= g < 22400000]
    return sorted(eligible)


def get_game_ids_from_enriched() -> list[int]:
    """Get game IDs from already-enriched shots data."""
    enriched_path = Path("data/enriched_shots.parquet")
    if not enriched_path.exists():
        return []
    
    df = pl.read_parquet(enriched_path)
    return df["game_id"].unique().to_list()


def fetch_game_date_from_cdn(game_id: int) -> dict | None:
    """Fetch game date from CDN boxscore data."""
    cdn_game_id = f"00{game_id}"
    url = CDN_BOXSCORE_URL.format(cdn_game_id)
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                game_info = data.get("game", {})
                
                # Extract date from gameTimeUTC or gameTimeLocal
                game_time = game_info.get("gameTimeUTC") or game_info.get("gameTimeLocal")
                if game_time:
                    # Parse ISO format: "2021-10-19T23:30:00Z" or "2021-10-19T18:30:00-05:00"
                    try:
                        # Handle both Z suffix and timezone offset
                        game_time_clean = game_time.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(game_time_clean)
                        return {
                            "game_id": game_id,
                            "game_date": dt.date(),
                            "home_team": game_info.get("homeTeam", {}).get("teamTricode"),
                            "away_team": game_info.get("awayTeam", {}).get("teamTricode"),
                        }
                    except ValueError:
                        pass
                
                return None
            elif resp.status_code == 403:
                return None
            elif resp.status_code == 429:
                time.sleep(5)
                continue
        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
    return None


def load_existing_schedule() -> dict:
    """Load existing schedule data if available."""
    if SCHEDULE_OUTPUT.exists():
        df = pl.read_parquet(SCHEDULE_OUTPUT)
        return {row["game_id"]: row for row in df.iter_rows(named=True)}
    return {}


def build_schedule_from_enriched():
    """
    Alternative: Extract game dates from already-cached CDN data.
    This is faster if we already have the CDN cache.
    """
    cache_dir = Path("data/cdn_cache")
    if not cache_dir.exists():
        return {}
    
    schedule = {}
    for cache_file in cache_dir.glob("*.json"):
        try:
            with open(cache_file) as f:
                data = json.load(f)
            
            game_info = data.get("game", {})
            game_id = int(cache_file.stem.lstrip("0"))
            game_time = game_info.get("gameTimeUTC") or game_info.get("gameTimeLocal")
            
            if game_time:
                dt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
                schedule[game_id] = {
                    "game_id": game_id,
                    "game_date": dt.date(),
                    "home_team": game_info.get("homeTeam", {}).get("teamTricode"),
                    "away_team": game_info.get("awayTeam", {}).get("teamTricode"),
                }
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
    
    return schedule


def main():
    print("=" * 60)
    print("Game Schedule Fetcher")
    print("=" * 60)
    
    # First try to build from existing CDN cache
    print("\nChecking CDN cache for game dates...")
    schedule = build_schedule_from_enriched()
    print(f"  Found {len(schedule)} games in cache")
    
    # Get target game IDs
    target_games = get_game_ids_from_plays()
    print(f"\nTarget games (2021-2024 seasons): {len(target_games)}")
    
    # Filter to games we don't have yet
    games_to_fetch = [g for g in target_games if g not in schedule]
    print(f"Games needing fetch: {len(games_to_fetch)}")
    
    if games_to_fetch:
        print(f"\nFetching dates for {len(games_to_fetch)} games...")
        print("(This may take a while due to rate limiting)")
        
        for i, game_id in enumerate(games_to_fetch):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(games_to_fetch)} ({100*i/len(games_to_fetch):.1f}%)")
            
            result = fetch_game_date_from_cdn(game_id)
            if result:
                schedule[game_id] = result
            
            time.sleep(RATE_LIMIT_DELAY)
        
        print(f"  Fetched {len([g for g in games_to_fetch if g in schedule])} new game dates")
    
    # Convert to DataFrame and save
    if schedule:
        records = list(schedule.values())
        df = pl.DataFrame(records)
        df.write_parquet(SCHEDULE_OUTPUT)
        print(f"\nSaved schedule for {len(df)} games to {SCHEDULE_OUTPUT}")
        
        # Show date range
        dates = df["game_date"].sort()
        print(f"Date range: {dates.min()} to {dates.max()}")
    else:
        print("\nNo schedule data to save")


if __name__ == "__main__":
    main()
