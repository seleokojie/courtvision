import polars as pl
import json
import time
import os
import re
from kafka import KafkaProducer

# Chunk size for streaming through large files
CHUNK_SIZE = 10000

# Get Kafka bootstrap servers from environment variable (default to localhost for local dev)
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')


def parse_time_remaining(time_str):
    """Parse 'MM:SS' format to seconds remaining in period."""
    if time_str is None:
        return 360  # Default to 6 minutes (mid-period)
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except:
        pass
    return 360


def extract_shot_type(description):
    """Extract shot type features from description text."""
    desc_lower = description.lower() if description else ""
    return {
        'is_dunk': 1 if 'dunk' in desc_lower else 0,
        'is_layup': 1 if 'layup' in desc_lower else 0,
        'is_hook': 1 if 'hook' in desc_lower else 0,
        'is_tip': 1 if 'tip' in desc_lower else 0,
        'is_fadeaway': 1 if 'fadeaway' in desc_lower or 'fade away' in desc_lower else 0,
        'is_bank': 1 if 'bank' in desc_lower else 0,
        'is_alley_oop': 1 if 'alley oop' in desc_lower else 0,
        'is_pullup': 1 if 'pull-up' in desc_lower or 'pullup' in desc_lower or 'pull up' in desc_lower else 0,
    }


def extract_cdn_features(description):
    """Extract CDN-style descriptor features from description text.
    
    These match the 12 CDN features used in the 26-feature model:
    - 4 context flags (derived from description when possible)
    - 8 descriptor one-hot features
    """
    desc_lower = description.lower() if description else ""
    return {
        # Context flags (limited detection from description)
        'is_fastbreak': 1 if 'fast break' in desc_lower or 'fastbreak' in desc_lower else 0,
        'is_second_chance': 1 if 'putback' in desc_lower or 'tip' in desc_lower else 0,
        'is_from_turnover': 0,  # Cannot detect from description alone
        'is_points_in_paint': 1 if 'paint' in desc_lower or 'lane' in desc_lower else 0,
        # Descriptor one-hot encoded
        'desc_pullup': 1 if 'pull-up' in desc_lower or 'pullup' in desc_lower or 'pull up' in desc_lower else 0,
        'desc_driving': 1 if 'driving' in desc_lower else 0,
        'desc_step_back': 1 if 'step back' in desc_lower or 'stepback' in desc_lower else 0,
        'desc_fadeaway': 1 if 'fadeaway' in desc_lower or 'fade away' in desc_lower else 0,
        'desc_running': 1 if 'running' in desc_lower else 0,
        'desc_floating': 1 if 'floating' in desc_lower or 'floater' in desc_lower else 0,
        'desc_turnaround': 1 if 'turnaround' in desc_lower or 'turn around' in desc_lower else 0,
        'desc_cutting': 1 if 'cutting' in desc_lower or 'cut' in desc_lower else 0,
    }


def stream_data(file_path):
    # Wait for Kafka to wake up
    time.sleep(10)
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    
    print("Loading and streaming data...")
    total_sent = 0
    
    try:
        # Use Polars lazy scanning with batched collection for memory efficiency
        batches = pl.scan_csv(file_path).collect_batches(chunk_size=CHUNK_SIZE)
        
        for df in batches:
            
            # Filter to shot events only (eventmsgtype: 1=Made, 2=Missed)
            df = df.filter(pl.col('eventmsgtype').is_in([1, 2]))
            
            # Combine home and visitor descriptions to get shot info
            df = df.with_columns([
                (pl.col('homedescription').fill_null('') + pl.col('visitordescription').fill_null('')).alias('description')
            ])
            
            # Extract shot distance using regex
            df = df.with_columns([
                pl.col('description').str.extract(r"(\d+)'", 1).cast(pl.Int32).alias('shot_distance')
            ])
            
            # Add event type column
            df = df.with_columns([
                pl.when(pl.col('eventmsgtype') == 1)
                .then(pl.lit('Made Shot'))
                .otherwise(pl.lit('Missed Shot'))
                .alias('event_type')
            ])
            
            # Drop rows without distance info
            df = df.filter(pl.col('shot_distance').is_not_null())
            
            # Stream events to Kafka
            for row in df.iter_rows(named=True):
                description = row['description']
                distance = int(row['shot_distance'])
                shot_types = extract_shot_type(description)
                cdn_features = extract_cdn_features(description)
                
                # Determine if home team shot
                is_home = 1 if (row['homedescription'] and row['homedescription'] != '') else 0
                
                event = {
                    "game_id": str(row['game_id']),
                    "player_id": int(row['player1_id']) if row['player1_id'] else 0,
                    "player": str(row['player1_name']) if row['player1_name'] else "Unknown",
                    "distance": distance,
                    "period": int(row['period']) if row['period'] else 1,
                    "seconds_remaining": parse_time_remaining(row['pctimestring']),
                    "is_three": 1 if distance > 23 else 0,
                    **shot_types,  # Unpack shot type features (includes is_pullup)
                    "is_home": is_home,
                    **cdn_features,  # Unpack CDN features for 26-feature model
                    "result": row['event_type'],
                    "timestamp": time.time()
                }
                producer.send('raw-shot-events', value=event)
                total_sent += 1
                
                # 0.05s sleep = ~20 events/sec. Remove sleep for stress testing.
                time.sleep(0.05)
            
            print(f"Processed batch, total events sent: {total_sent}")
                
    except FileNotFoundError:
        print("Error: 'nba_plays.csv' not found in data/ folder.")
        return
    
    producer.flush()
    print(f"Simulation complete. Total events sent: {total_sent}")

if __name__ == "__main__":
    stream_data('data/nba_plays.csv')
