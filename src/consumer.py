import json
import joblib
import psycopg2
import time
import os
import numpy as np
from kafka import KafkaConsumer

# Get config from environment variables (defaults for local development)
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
DB_HOST = os.environ.get('DB_HOST', 'localhost')

# Feature order must match training (retrain.py FEATURE_NAMES)
FEATURE_ORDER = [
    'shot_distance',
    'period',
    'seconds_remaining',
    'is_three',
    'is_dunk',
    'is_layup',
    'is_hook',
    'is_tip',
    'is_fadeaway',
    'is_bank',
    'is_alley_oop',
    'player_skill_rating'
]

# Default league average for unknown players
LEAGUE_AVG_FG = 0.45

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

# Load player skill ratings artifact
def load_player_skills():
    skill_paths = ['host/player_skills.pkl', 'player_skills.pkl']
    for path in skill_paths:
        try:
            skills = joblib.load(path)
            print(f"Player skills loaded from {path} ({len(skills)} players).")
            return skills
        except:
            continue
    print("WARNING: No player_skills.pkl found. Using league average for all players.")
    return {}

model = load_model()
player_skills = load_player_skills()
last_model_check = time.time()
MODEL_RELOAD_INTERVAL = 60  # Check for new model every 60 seconds

consumer = KafkaConsumer(
    'raw-shot-events',
    bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

def build_feature_vector(data, player_skills):
    """Build feature vector from event data in correct order."""
    player = data.get('player', 'Unknown')
    skill_rating = player_skills.get(player, LEAGUE_AVG_FG)
    
    features = [
        data.get('distance', 15),           # shot_distance
        data.get('period', 1),              # period
        data.get('seconds_remaining', 360), # seconds_remaining
        data.get('is_three', 0),            # is_three
        data.get('is_dunk', 0),             # is_dunk
        data.get('is_layup', 0),            # is_layup
        data.get('is_hook', 0),             # is_hook
        data.get('is_tip', 0),              # is_tip
        data.get('is_fadeaway', 0),         # is_fadeaway
        data.get('is_bank', 0),             # is_bank
        data.get('is_alley_oop', 0),        # is_alley_oop
        skill_rating,                        # player_skill_rating
    ]
    return np.array(features).reshape(1, -1)


def get_shot_type_label(data):
    """Get human-readable shot type for logging."""
    if data.get('is_dunk'):
        return 'Dunk'
    elif data.get('is_layup'):
        return 'Layup'
    elif data.get('is_hook'):
        return 'Hook'
    elif data.get('is_tip'):
        return 'Tip'
    elif data.get('is_fadeaway'):
        return 'Fadeaway'
    elif data.get('is_bank'):
        return 'Bank'
    elif data.get('is_alley_oop'):
        return 'Alley Oop'
    elif data.get('is_three'):
        return '3PT Jump Shot'
    else:
        return 'Jump Shot'


print("Worker listening for events (XGBoost model with enhanced features)...")
for message in consumer:
    # Periodically reload model to pick up retraining updates
    if time.time() - last_model_check > MODEL_RELOAD_INTERVAL:
        new_model = load_model()
        if new_model is not None:
            model = new_model
        new_skills = load_player_skills()
        if new_skills:
            player_skills = new_skills
        last_model_check = time.time()
    
    data = message.value
    dist = data.get('distance', 15)
    
    # Inference
    if model:
        # Build feature vector and predict
        X = build_feature_vector(data, player_skills)
        prob = model.predict_proba(X)[0][1]  # Probability of 'Made'
        xp = float(prob * (3 if dist > 23 else 2))
    else:
        # Fallback logic (Heuristic)
        xp = 1.0 if dist < 10 else 0.8
        
    grade = 'A' if xp > 1.1 else ('B' if xp > 0.9 else 'C')
    
    # Write to DB
    cur.execute(
        "INSERT INTO shot_telemetry (game_id, player_name, shot_distance, expected_points, shot_grade) VALUES (%s, %s, %s, %s, %s)",
        (str(data['game_id']), str(data.get('player', 'Unknown')), int(dist), float(xp), grade)
    )
    conn.commit()
