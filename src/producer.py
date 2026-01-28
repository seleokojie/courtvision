import pandas as pd
import json
import time
from kafka import KafkaProducer

def stream_data(file_path):
    # Wait for Kafka to wake up
    time.sleep(10)
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    
    # Load Kaggle Data (simulating a live feed)
    try:
        df = pd.read_csv(file_path)
        df = df[df['EVENT_TYPE'].isin(['Made Shot', 'Missed Shot'])]
    except FileNotFoundError:
        print("Error: 'nba_plays.csv' not found in data/ folder.")
        return

    print(f"Starting simulation with {len(df)} rows...")
    
    for _, row in df.iterrows():
        event = {
            "game_id": row.get('GAME_ID'),
            "player": row.get('PLAYER_NAME'),
            "distance": row.get('SHOT_DISTANCE', 0),
            "result": row.get('EVENT_TYPE'),
            "timestamp": time.time()
        }
        producer.send('raw-shot-events', value=event)
        # 0.05s sleep = ~20 events/sec. Remove sleep for stress testing.
        time.sleep(0.05)

if __name__ == "__main__":
    stream_data('data/nba_plays.csv')
