import json
import joblib
import psycopg2
import time
from kafka import KafkaConsumer

# DB Connection with retry logic
def get_db_connection():
    while True:
        try:
            return psycopg2.connect("dbname=courtvision user=admin password=password host=localhost")
        except:
            time.sleep(2)

conn = get_db_connection()
cur = conn.cursor()

# Load Model
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully.")
except:
    print("WARNING: No model found. Using heuristic fallback.")
    model = None

consumer = KafkaConsumer(
    'raw-shot-events',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Worker listening for events...")
for message in consumer:
    data = message.value
    dist = data['distance']
    
    # Inference
    if model:
        # Predict probability of 'Made' (Index 1)
        prob = model.predict_proba([[dist]])[0][1]
        xp = prob * (3 if dist > 23 else 2)
    else:
        # Fallback logic (Heuristic)
        xp = 1.0 if dist < 10 else 0.8
        
    grade = 'A' if xp > 1.1 else ('B' if xp > 0.9 else 'C')
    
    # Write to DB
    cur.execute(
        "INSERT INTO shot_telemetry (game_id, player_name, shot_distance, expected_points, shot_grade) VALUES (%s, %s, %s, %s, %s)",
        (data['game_id'], data['player'], dist, xp, grade)
    )
    conn.commit()
