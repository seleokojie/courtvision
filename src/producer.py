import polars as pl
import json
import time
from kafka import KafkaProducer

# Chunk size for streaming through large files
CHUNK_SIZE = 10000

def stream_data(file_path):
    # Wait for Kafka to wake up
    time.sleep(10)
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
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
                event = {
                    "game_id": str(row['game_id']),
                    "player": str(row['player1_name']) if row['player1_name'] else "Unknown",
                    "distance": int(row['shot_distance']),
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
