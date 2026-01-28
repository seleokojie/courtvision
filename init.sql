CREATE TABLE IF NOT EXISTS shot_telemetry (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50),
    player_name VARCHAR(100),
    shot_distance INT,
    expected_points FLOAT,
    shot_grade CHAR(2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
