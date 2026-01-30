#!/usr/bin/env python3
"""
Enrich shot data with injury context from historical injury reports.

This script parses the injury database and creates features that capture:
- Whether a player was recently injured
- Days since returning from injury
- Type of injury (leg injuries affect shooting more)
- Whether playing through a minor injury (Probable/Questionable status)

The injury data covers Oct 2021 - June 2024.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

# Configuration
INJURY_CSV_PATH = Path("data/Injury Database - Oct 2021 - June 2024.csv")
GAME_SCHEDULE_PATH = Path("data/game_schedule.parquet")
OUTPUT_FILE = Path("data/injury_features.parquet")
SHOT_INJURY_OUTPUT = Path("data/shots_with_injury.parquet")

# How many days after returning from "Out" status to flag as "returning"
RETURN_WINDOW_DAYS = 14

# Injury types that affect shooting (leg/lower body injuries)
LEG_INJURY_PATTERNS = [
    r'ankle', r'knee', r'foot', r'hamstring', r'calf', r'quad', 
    r'groin', r'hip', r'achilles', r'leg', r'thigh', r'shin',
    r'toe', r'heel', r'navicular', r'patellar', r'acl', r'mcl',
    r'meniscus', r'plantar'
]

# Shooting-related injuries (arm/hand)
SHOOTING_INJURY_PATTERNS = [
    r'shoulder', r'wrist', r'hand', r'finger', r'thumb', r'elbow', r'arm'
]


def parse_date(date_str: str) -> datetime:
    """Parse date string like '10/19/2021' to datetime."""
    return datetime.strptime(date_str, "%m/%d/%Y")


def extract_injury_type(reason: str) -> dict:
    """Classify injury by body part affected."""
    reason_lower = reason.lower() if reason else ""
    
    is_leg_injury = any(re.search(p, reason_lower) for p in LEG_INJURY_PATTERNS)
    is_shooting_injury = any(re.search(p, reason_lower) for p in SHOOTING_INJURY_PATTERNS)
    is_illness = 'illness' in reason_lower or 'covid' in reason_lower or 'protocol' in reason_lower
    is_rest = 'rest' in reason_lower
    is_personal = 'personal' in reason_lower or 'not with team' in reason_lower
    
    return {
        'is_leg_injury': is_leg_injury,
        'is_shooting_injury': is_shooting_injury,
        'is_illness': is_illness,
        'is_rest': is_rest,
        'is_personal': is_personal,
    }


def load_injury_data() -> pl.DataFrame:
    """Load and parse the injury database."""
    print(f"Loading injury data from {INJURY_CSV_PATH}...")
    
    df = pl.read_csv(INJURY_CSV_PATH)
    print(f"  Loaded {len(df):,} injury records")
    
    # Parse dates
    df = df.with_columns(
        pl.col("DATE").str.strptime(pl.Date, "%m/%d/%Y").alias("game_date")
    )
    
    # Normalize player names: "Irving, Kyrie" -> "Kyrie Irving"
    df = df.with_columns(
        pl.col("PLAYER").map_elements(
            lambda x: " ".join(reversed(x.split(", "))) if ", " in x else x,
            return_dtype=pl.Utf8
        ).str.to_uppercase().alias("player_name_normalized")
    )
    
    # Extract injury type features
    df = df.with_columns([
        pl.col("REASON").map_elements(
            lambda x: extract_injury_type(x)['is_leg_injury'],
            return_dtype=pl.Boolean
        ).alias("is_leg_injury"),
        pl.col("REASON").map_elements(
            lambda x: extract_injury_type(x)['is_shooting_injury'],
            return_dtype=pl.Boolean
        ).alias("is_shooting_injury"),
        pl.col("REASON").map_elements(
            lambda x: extract_injury_type(x)['is_illness'],
            return_dtype=pl.Boolean
        ).alias("is_illness"),
    ])
    
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"  Unique players: {df['player_name_normalized'].n_unique():,}")
    
    return df


def extract_last_name(full_name: str) -> str:
    """Extract last name from full name, handling common edge cases."""
    if not full_name:
        return ""
    parts = full_name.strip().split()
    if not parts:
        return ""
    # Handle suffixes like Jr., III, etc.
    last = parts[-1]
    if last in ['JR.', 'SR.', 'II', 'III', 'IV', 'JR', 'SR']:
        if len(parts) > 1:
            return parts[-2].upper()
    return last.upper()


def build_player_injury_timeline(injury_df: pl.DataFrame) -> tuple[dict, dict]:
    """
    Build a timeline of injury events per player.
    
    Returns:
        - timeline: full_name -> list of events
        - last_name_to_full: last_name -> list of full_names (for fuzzy matching)
    """
    timeline = {}
    last_name_to_full = {}
    
    for row in injury_df.iter_rows(named=True):
        player = row['player_name_normalized']
        if player not in timeline:
            timeline[player] = []
            # Also index by last name
            last_name = extract_last_name(player)
            if last_name:
                if last_name not in last_name_to_full:
                    last_name_to_full[last_name] = []
                if player not in last_name_to_full[last_name]:
                    last_name_to_full[last_name].append(player)
        
        timeline[player].append({
            'date': row['game_date'],
            'status': row['STATUS'],
            'is_leg': row['is_leg_injury'],
            'is_shooting': row['is_shooting_injury'],
            'is_illness': row['is_illness'],
            'reason': row['REASON'],
        })
    
    # Sort each player's timeline by date
    for player in timeline:
        timeline[player].sort(key=lambda x: x['date'])
    
    return timeline, last_name_to_full


def get_injury_features_for_game(
    player_name: str, 
    game_date, 
    timeline: dict,
    last_name_lookup: dict = None,
    return_window: int = RETURN_WINDOW_DAYS
) -> dict:
    """
    Get injury-related features for a player on a specific game date.
    
    Features returned:
    - is_returning_from_injury: Was "Out" within last N days, now playing
    - days_since_injury: Days since last "Out" status (capped at 365)
    - had_leg_injury: Last injury was leg-related
    - had_shooting_injury: Last injury was arm/hand related
    - is_playing_hurt: Listed as Probable/Questionable on game day
    """
    features = {
        'is_returning_from_injury': 0,
        'days_since_injury': 365,  # Default to long time (no recent injury)
        'had_leg_injury': 0,
        'had_shooting_injury': 0,
        'is_playing_hurt': 0,
    }
    
    # Try exact match first
    lookup_name = player_name
    if player_name not in timeline and last_name_lookup:
        # Try last name match
        last_name = extract_last_name(player_name)
        if last_name and last_name in last_name_lookup:
            candidates = last_name_lookup[last_name]
            if len(candidates) == 1:
                # Unambiguous match
                lookup_name = candidates[0]
            # If multiple candidates, skip (ambiguous)
    
    if lookup_name not in timeline:
        return features
    
    player_events = timeline[lookup_name]
    
    # Convert game_date to date object if needed
    if hasattr(game_date, 'date'):
        game_date = game_date.date()
    
    # Find most recent injury before this game
    last_out_event = None
    same_day_status = None
    
    for event in player_events:
        event_date = event['date']
        if hasattr(event_date, 'date'):
            event_date = event_date
        
        if event_date <= game_date:
            if event['status'] == 'Out':
                last_out_event = event
            if event_date == game_date:
                same_day_status = event
    
    # Check if playing hurt (listed but not Out)
    if same_day_status and same_day_status['status'] in ['Probable', 'Questionable', 'Doubtful']:
        features['is_playing_hurt'] = 1
    
    # Check recent injury history
    if last_out_event:
        event_date = last_out_event['date']
        if hasattr(event_date, 'date'):
            event_date = event_date
        
        days_since = (game_date - event_date).days
        features['days_since_injury'] = min(days_since, 365)
        
        if days_since <= return_window:
            features['is_returning_from_injury'] = 1
        
        if last_out_event['is_leg']:
            features['had_leg_injury'] = 1
        if last_out_event['is_shooting']:
            features['had_shooting_injury'] = 1
    
    return features


def create_injury_lookup(injury_df: pl.DataFrame) -> tuple[dict, dict]:
    """
    Create a lookup structure for fast feature extraction.
    
    Returns: 
        - timeline: player_name -> list of events
        - last_name_lookup: last_name -> list of full names
    """
    timeline, last_name_lookup = build_player_injury_timeline(injury_df)
    
    # Get all unique player-date combinations from injury reports
    # plus a reasonable range for feature computation
    all_dates = injury_df['game_date'].unique().sort()
    min_date = all_dates.min()
    max_date = all_dates.max()
    
    print(f"  Building injury timeline from {min_date} to {max_date}...")
    print(f"  Players with injury records: {len(timeline)}")
    print(f"  Unique last names indexed: {len(last_name_lookup)}")
    
    return timeline, last_name_lookup


def enrich_shots_with_injury_data(
    shots_df: pl.DataFrame,
    injury_timeline: dict,
) -> pl.DataFrame:
    """
    Add injury features to shot data.
    
    Expects shots_df to have:
    - player_name (or we'll need to join on player_id)
    - game_date (or derive from game_id)
    """
    print("Enriching shots with injury features...")
    
    # We need to compute features per row based on player + date
    # For efficiency, we'll batch by unique player-date combinations
    
    # Extract unique player-date pairs
    if 'game_date' not in shots_df.columns:
        # Derive date from game_id if needed
        # Game ID format: 21900001 -> season 2019-20
        # This is a simplification - in production, join with game schedule
        print("  Warning: No game_date column, injury features will be limited")
        return shots_df
    
    # Normalize player names in shots data
    if 'player_name' in shots_df.columns:
        shots_df = shots_df.with_columns(
            pl.col("player_name").str.to_uppercase().alias("player_name_normalized")
        )
    
    # Compute features for each row
    injury_features = []
    
    for row in shots_df.iter_rows(named=True):
        player = row.get('player_name_normalized', row.get('player_name', ''))
        game_date = row.get('game_date')
        
        if player and game_date:
            features = get_injury_features_for_game(player, game_date, injury_timeline)
        else:
            features = {
                'is_returning_from_injury': 0,
                'days_since_injury': 365,
                'had_leg_injury': 0,
                'had_shooting_injury': 0,
                'is_playing_hurt': 0,
            }
        injury_features.append(features)
    
    # Add features to dataframe
    features_df = pl.DataFrame(injury_features)
    shots_df = pl.concat([shots_df, features_df], how="horizontal")
    
    return shots_df


def generate_injury_summary():
    """Generate summary statistics about the injury data."""
    injury_df = load_injury_data()
    
    print("\n=== Injury Data Summary ===")
    
    # Status distribution
    status_counts = injury_df.group_by("STATUS").agg(pl.len().alias("count")).sort("count", descending=True)
    print("\nStatus Distribution:")
    for row in status_counts.iter_rows(named=True):
        print(f"  {row['STATUS']}: {row['count']:,}")
    
    # Injury type distribution
    leg_count = injury_df.filter(pl.col("is_leg_injury")).height
    shooting_count = injury_df.filter(pl.col("is_shooting_injury")).height
    illness_count = injury_df.filter(pl.col("is_illness")).height
    
    print(f"\nInjury Types:")
    print(f"  Leg/Lower Body: {leg_count:,} ({100*leg_count/len(injury_df):.1f}%)")
    print(f"  Arm/Shooting: {shooting_count:,} ({100*shooting_count/len(injury_df):.1f}%)")
    print(f"  Illness/Protocol: {illness_count:,} ({100*illness_count/len(injury_df):.1f}%)")
    
    # Most injured players
    player_counts = (
        injury_df
        .filter(pl.col("STATUS") == "Out")
        .group_by("player_name_normalized")
        .agg(pl.len().alias("out_games"))
        .sort("out_games", descending=True)
        .head(10)
    )
    
    print("\nMost Games Missed (Top 10):")
    for row in player_counts.iter_rows(named=True):
        print(f"  {row['player_name_normalized']}: {row['out_games']} games")
    
    return injury_df


def load_game_schedule() -> dict | None:
    """Load game schedule (game_id -> date mapping)."""
    if not GAME_SCHEDULE_PATH.exists():
        print(f"  Warning: No game schedule at {GAME_SCHEDULE_PATH}")
        print("  Run fetch_game_schedule.py first for game-level injury features")
        return None
    
    df = pl.read_parquet(GAME_SCHEDULE_PATH)
    schedule = {}
    for row in df.iter_rows(named=True):
        schedule[row["game_id"]] = row["game_date"]
    
    print(f"  Loaded schedule for {len(schedule)} games")
    return schedule


def enrich_shots_with_injury_features(
    injury_timeline: dict,
    last_name_lookup: dict,
    game_schedule: dict,
    enriched_shots_path: Path = Path("data/enriched_shots.parquet"),
) -> pl.DataFrame | None:
    """
    Add injury features to enriched shots data using game-level date matching.
    
    This is Option 2: accurate game-level injury context.
    """
    if not enriched_shots_path.exists():
        print(f"  No enriched shots data at {enriched_shots_path}")
        return None
    
    print(f"\nEnriching shots with injury features...")
    shots_df = pl.read_parquet(enriched_shots_path)
    print(f"  Loaded {len(shots_df):,} shots")
    
    # Build schedule lookup as a DataFrame for efficient join
    schedule_records = [
        {"game_id": gid, "game_date": gdate} 
        for gid, gdate in game_schedule.items()
    ]
    schedule_df = pl.DataFrame(schedule_records)
    
    # Join shots with schedule
    shots_with_schedule = shots_df.join(schedule_df, on="game_id", how="inner")
    print(f"  Matched {len(shots_with_schedule):,} shots to game dates ({100*len(shots_with_schedule)/len(shots_df):.1f}%)")
    print(f"  Skipped {len(shots_df) - len(shots_with_schedule):,} shots without schedule data")
    
    # Now compute injury features for each row
    # Normalize player names
    shots_with_schedule = shots_with_schedule.with_columns(
        pl.col("player_name").str.to_uppercase().alias("player_name_upper")
    )
    
    # Compute injury features row by row
    injury_features_list = []
    for row in shots_with_schedule.iter_rows(named=True):
        player_name = row.get("player_name_upper", "")
        game_date = row.get("game_date")
        features = get_injury_features_for_game(player_name, game_date, injury_timeline, last_name_lookup)
        injury_features_list.append(features)
    
    # Add injury features as new columns
    injury_df = pl.DataFrame(injury_features_list)
    result_df = pl.concat([shots_with_schedule, injury_df], how="horizontal")
    
    # Drop temporary column
    result_df = result_df.drop("player_name_upper")
    
    # Save the enriched data
    result_df.write_parquet(SHOT_INJURY_OUTPUT)
    print(f"  Saved to {SHOT_INJURY_OUTPUT}")
    
    # Show some stats
    returning = result_df.filter(pl.col("is_returning_from_injury") == 1).height
    playing_hurt = result_df.filter(pl.col("is_playing_hurt") == 1).height
    leg_injury = result_df.filter(pl.col("had_leg_injury") == 1).height
    
    print(f"\n  Injury feature coverage:")
    print(f"    Returning from injury: {returning:,} shots ({100*returning/len(result_df):.2f}%)")
    print(f"    Playing hurt: {playing_hurt:,} shots ({100*playing_hurt/len(result_df):.2f}%)")
    print(f"    Had leg injury: {leg_injury:,} shots ({100*leg_injury/len(result_df):.2f}%)")
    
    return result_df


def main():
    """Main entry point - generate injury timeline lookup and save."""
    print("=" * 60)
    print("Injury Data Enrichment Pipeline")
    print("=" * 60)
    
    # Load and analyze injury data
    injury_df = generate_injury_summary()
    
    # Build timeline lookup
    print("\nBuilding injury timeline...")
    timeline, last_name_lookup = create_injury_lookup(injury_df)
    
    # Save the processed data
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save injury records with extracted features
    injury_df.write_parquet(OUTPUT_FILE)
    print(f"\nSaved processed injury data to {OUTPUT_FILE}")
    
    # Load game schedule for game-level matching
    print("\nLoading game schedule...")
    game_schedule = load_game_schedule()
    
    # If we have schedule data, enrich the shots
    if game_schedule:
        enrich_shots_with_injury_features(timeline, last_name_lookup, game_schedule)
    
    # Example: test feature extraction (using last name matching)
    print("\n=== Testing Feature Extraction ===")
    test_cases = [
        ("IRVING", datetime(2021, 10, 27).date()),  # Last name only like shot data
        ("WILLIAMSON", datetime(2022, 1, 15).date()),
        ("LEONARD", datetime(2022, 3, 1).date()),
    ]
    
    for player, date in test_cases:
        features = get_injury_features_for_game(player, date, timeline, last_name_lookup)
        print(f"\n{player} on {date}:")
        for k, v in features.items():
            print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
    print("Injury enrichment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
