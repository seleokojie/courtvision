"""
Fetch player metadata (height, weight, position) from nba_api.

This script fetches player physical attributes for all players in our dataset
and caches them to a parquet file for fast lookup during training.

Features:
- Incremental caching: saves progress to JSON cache file
- Resume capability: restarts from where it left off if interrupted
- Rate limit handling: configurable delay between requests
"""

import json
import time
from pathlib import Path

import polars as pl
from nba_api.stats.endpoints import CommonPlayerInfo

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
PLAYS_CSV = DATA_DIR / "nba_plays.csv"
PLAYER_METADATA_PATH = DATA_DIR / "player_metadata.parquet"
CACHE_PATH = DATA_DIR / "player_cache.json"  # Intermediate cache for resume
FAILED_PATH = DATA_DIR / "player_failed.json"  # Track failed IDs to skip on resume

# Rate limiting (increased to avoid timeouts)
REQUEST_DELAY = 1.5  # seconds between requests (was 0.6)


def parse_height_to_inches(height_str: str) -> int | None:
    """Convert height string like '6-11' to inches (83)."""
    if not height_str or height_str == "":
        return None
    try:
        parts = height_str.split("-")
        if len(parts) == 2:
            feet, inches = int(parts[0]), int(parts[1])
            return feet * 12 + inches
    except (ValueError, AttributeError):
        pass
    return None


def parse_position(position_str: str) -> tuple[bool, bool, bool]:
    """
    Parse position string to boolean flags.
    
    Returns: (is_guard, is_forward, is_center)
    
    Examples:
        "Guard" -> (True, False, False)
        "Forward-Center" -> (False, True, True)
        "Guard-Forward" -> (True, True, False)
    """
    if not position_str:
        return (False, False, False)
    
    pos = position_str.lower()
    is_guard = "guard" in pos
    is_forward = "forward" in pos
    is_center = "center" in pos
    
    return (is_guard, is_forward, is_center)


def get_unique_player_ids() -> list[int]:
    """Get all unique valid player IDs from our dataset."""
    df = pl.scan_csv(PLAYS_CSV).select("player1_id").collect()
    # Real NBA player IDs are 6+ digits
    player_ids = (
        df.filter(pl.col("player1_id") > 100000)["player1_id"]
        .unique()
        .sort()
        .to_list()
    )
    return player_ids


def load_cache() -> dict[int, dict]:
    """Load cached player data from JSON file."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)
            # Convert string keys back to int
            return {int(k): v for k, v in data.items()}
    return {}


def save_cache(cache: dict[int, dict]) -> None:
    """Save cache to JSON file."""
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def delete_cache() -> None:
    """Delete the cache files after successful completion."""
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
        print(f"Deleted cache file: {CACHE_PATH}")
    if FAILED_PATH.exists():
        FAILED_PATH.unlink()
        print(f"Deleted failed IDs file: {FAILED_PATH}")


def load_failed_ids() -> set[int]:
    """Load set of player IDs that previously failed."""
    if FAILED_PATH.exists():
        with open(FAILED_PATH, "r") as f:
            return set(json.load(f))
    return set()


def save_failed_ids(failed_ids: set[int]) -> None:
    """Save failed IDs to JSON file."""
    with open(FAILED_PATH, "w") as f:
        json.dump(list(failed_ids), f)


def fetch_player_info(player_id: int) -> dict | None:
    """Fetch player info from nba_api."""
    try:
        info = CommonPlayerInfo(player_id=player_id)
        df = info.get_data_frames()[0]
        
        if len(df) == 0:
            return None
            
        row = df.iloc[0]
        height_inches = parse_height_to_inches(row.get("HEIGHT", ""))
        is_guard, is_forward, is_center = parse_position(row.get("POSITION", ""))
        
        return {
            "player_id": int(player_id),
            "player_name": row.get("DISPLAY_FIRST_LAST", ""),
            "height_inches": height_inches,
            "weight": int(row["WEIGHT"]) if row.get("WEIGHT") else None,
            "position": row.get("POSITION", ""),
            "is_guard": is_guard,
            "is_forward": is_forward,
            "is_center": is_center,
        }
    except Exception as e:
        print(f"  Error fetching player {player_id}: {e}")
        return None


def fetch_all_players(player_ids: list[int], cache: dict[int, dict]) -> list[dict]:
    """Fetch metadata for all players with progress reporting and caching."""
    total = len(player_ids)
    
    # Load previously failed IDs to skip them
    failed_ids = load_failed_ids()
    if failed_ids:
        print(f"Skipping {len(failed_ids)} previously failed player IDs")
    
    # Filter out already cached players AND failed players
    remaining_ids = [pid for pid in player_ids if pid not in cache and pid not in failed_ids]
    cached_count = len(cache)
    skipped_count = len(failed_ids)
    
    if cached_count > 0:
        print(f"Found {cached_count} players in cache, {len(remaining_ids)} remaining to fetch")
    
    print(f"Fetching metadata for {len(remaining_ids)} players...")
    print(f"Estimated time: {len(remaining_ids) * REQUEST_DELAY / 60:.1f} minutes")
    print()
    
    failed_this_run = 0
    
    for i, player_id in enumerate(remaining_ids):
        progress_num = cached_count + skipped_count + i + 1
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Progress: {progress_num}/{total} ({100 * progress_num / total:.1f}%) - Failed: {failed_this_run}")
        
        info = fetch_player_info(player_id)
        if info:
            cache[player_id] = info
            # Save cache after each successful fetch
            if (i + 1) % 10 == 0:
                save_cache(cache)
        else:
            failed_this_run += 1
            failed_ids.add(player_id)
            # Save failed IDs periodically
            if failed_this_run % 5 == 0:
                save_failed_ids(failed_ids)
            
        time.sleep(REQUEST_DELAY)
    
    # Final saves
    save_cache(cache)
    save_failed_ids(failed_ids)
    
    print(f"\nCompleted: {len(cache)} successful, {len(failed_ids)} total failed")
    return list(cache.values())


def save_to_parquet(records: list[dict]) -> None:
    """Save player metadata to parquet file."""
    df = pl.DataFrame(records)
    df.write_parquet(PLAYER_METADATA_PATH)
    print(f"Saved {len(records)} player records to {PLAYER_METADATA_PATH}")
    print(f"File size: {PLAYER_METADATA_PATH.stat().st_size / 1024:.1f} KB")


def load_player_metadata() -> pl.DataFrame:
    """Load player metadata from parquet file."""
    if not PLAYER_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Player metadata not found at {PLAYER_METADATA_PATH}. "
            "Run this script first to fetch the data."
        )
    return pl.read_parquet(PLAYER_METADATA_PATH)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Player Metadata Fetcher")
    print("=" * 60)
    
    # Check if we already have complete data
    if PLAYER_METADATA_PATH.exists():
        existing = pl.read_parquet(PLAYER_METADATA_PATH)
        print(f"Found existing metadata for {len(existing)} players")
        response = input("Re-fetch all players? (y/N): ").strip().lower()
        if response != "y":
            print("Keeping existing data.")
            return
        # If re-fetching, also clear cache
        delete_cache()
    
    # Load cache (for resume capability)
    cache = load_cache()
    if cache:
        print(f"Resuming from cache with {len(cache)} players already fetched")
    
    # Get unique player IDs
    player_ids = get_unique_player_ids()
    print(f"Found {len(player_ids)} unique players in dataset")
    
    # Fetch all player metadata (with caching)
    records = fetch_all_players(player_ids, cache)
    
    # Save to parquet
    if records:
        save_to_parquet(records)
        
        # Delete cache after successful completion
        delete_cache()
        
        # Show summary
        df = pl.DataFrame(records)
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total players: {len(df)}")
        print(f"  Players with height: {df.filter(pl.col('height_inches').is_not_null()).height}")
        print(f"  Guards: {df.filter(pl.col('is_guard')).height}")
        print(f"  Forwards: {df.filter(pl.col('is_forward')).height}")
        print(f"  Centers: {df.filter(pl.col('is_center')).height}")
        heights = df.filter(pl.col('height_inches').is_not_null())['height_inches']
        if len(heights) > 0:
            print(f"  Height range: {heights.min()}-{heights.max()} inches")


if __name__ == "__main__":
    main()
