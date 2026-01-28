import json
import joblib
import psycopg2
import time
import os
from kafka import KafkaConsumer

# Get config from environment variables (defaults for local development)
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
DB_HOST = os.environ.get('DB_HOST', 'localhost')

# DB Connection with retry logic
def get_db_connection():
    while True:
        try:
            return psycopg2.connect(f"dbname=courtvision user=admin password=password host={DB_HOST}")
        except:
            time.sleep(2)

conn = get_db_connection()
cur = conn.cursor()

# Load Model - check host mount first (for retraining updates), then local
def load_model():
    model_paths = ['host/model.pkl', 'model.pkl']
    for path in model_paths:
        try:
            model = joblib.load(path)
            print(f"Model loaded successfully from {path}.")
            return model
        except:
            continue
    print("WARNING: No model found. Using heuristic fallback.")
    return None

model = load_model()
last_model_check = time.time()
MODEL_RELOAD_INTERVAL = 60  # Check for new model every 60 seconds

consumer = KafkaConsumer(
    'raw-shot-events',
    bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Worker listening for events...")
for message in consumer:
    # Periodically reload model to pick up retraining updates
    if time.time() - last_model_check > MODEL_RELOAD_INTERVAL:
        new_model = load_model()
        if new_model is not None:
            model = new_model
        last_model_check = time.time()
    
    data = message.value
    dist = data['distance']
    
    # Inference
    if model:
        # Predict probability of 'Made' (Index 1)
        prob = model.predict_proba([[dist]])[0][1]
        xp = float(prob * (3 if dist > 23 else 2))
    else:
        # Fallback logic (Heuristic)
        xp = 1.0 if dist < 10 else 0.8
        
    grade = 'A' if xp > 1.1 else ('B' if xp > 0.9 else 'C')
    
    # Write to DB
    cur.execute(
        "INSERT INTO shot_telemetry (game_id, player_name, shot_distance, expected_points, shot_grade) VALUES (%s, %s, %s, %s, %s)",
        (str(data['game_id']), str(data['player']), int(dist), float(xp), grade)
    )
    conn.commit()
