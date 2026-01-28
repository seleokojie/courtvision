# CourtVision - Real-Time NBA Shot Quality Telemetry Pipeline

A high-throughput real-time NBA telemetry pipeline demonstrating distributed systems, event-driven architecture, and MLOps principles.

## Overview

CourtVision solves the "batch latency" problem by evaluating NBA shot quality (Expected Points) in sub-50ms latency using an event-driven microservices architecture.

**Key Features:**
- **Ingestion**: Simulate high-velocity telemetry (10k+ events/sec) using historical data
- **Decoupling**: Apache Kafka handles backpressure and enables "time-travel" replayability
- **Inference**: Real-time ML predictions (Logistic Regression) for shot quality
- **Persistence**: Enriched time-series data stored in PostgreSQL
- **MLOps**: Automated retraining loop for handling concept drift

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Producer   │────▶│    Kafka    │────▶│  Consumer   │────▶│ PostgreSQL  │
│  (Streamer) │     │   Broker    │     │ (Inference) │     │    (DB)     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                                        ┌───────────┐
                                        │  Retrain  │
                                        │  Service  │
                                        └───────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Message Broker | Apache Kafka | Event streaming with log-based persistence |
| Database | PostgreSQL | Time-series storage for enriched telemetry |
| ML Framework | Scikit-Learn | Shot probability prediction |
| Containerization | Docker + DevContainers | Reproducible development environment |
| Language | Python 3.11 | Application code |

## Prerequisites

- Docker Desktop
- VS Code with Dev Containers extension (recommended)

## Dataset Setup

**Important**: You must download the NBA dataset from Kaggle before running the pipeline.

1. Go to [NBA Database on Kaggle](https://www.kaggle.com/datasets/wyattowalsh/basketball)
2. Download the dataset
3. Extract and place the plays data as `data/nba_plays.csv`

The CSV should contain columns: `GAME_ID`, `PLAYER_NAME`, `SHOT_DISTANCE`, `EVENT_TYPE`

## Quick Start

### 1. Clone and Open in DevContainer

```bash
git clone https://github.com/seleokojie/courtvision.git
cd courtvision
```

Open in VS Code and use "Reopen in Container" when prompted.

### 2. Start Infrastructure

```bash
docker-compose up -d
```

Verify services are running:
```bash
docker-compose ps
```

### 3. Train the Model

```bash
python src/retrain.py
```

### 4. Start the Consumer (Terminal 1)

```bash
python src/consumer.py
```

### 5. Start the Producer (Terminal 2)

```bash
python src/producer.py
```

## Services

| Service | Port | URL |
|---------|------|-----|
| Kafka UI | 8080 | http://localhost:8080 |
| PostgreSQL | 5432 | localhost:5432 |
| Kafka Broker | 9092 | localhost:9092 |

## Project Structure

```
courtvision/
├── .devcontainer/
│   └── devcontainer.json    # VS Code DevContainer config
├── data/
│   └── nba_plays.csv        # (User must download from Kaggle)
├── src/
│   ├── producer.py          # Event streamer (simulates live feed)
│   ├── consumer.py          # Inference engine + DB persistence
│   └── retrain.py           # MLOps retraining loop
├── docker-compose.yml       # Infrastructure services
├── init.sql                 # Database schema
├── requirements.txt         # Python dependencies
├── LICENSE
└── README.md
```

## Data Schema

**Kafka Topic**: `raw-shot-events`

```json
{
  "game_id": "002230001",
  "player": "LeBron James",
  "distance": 24,
  "result": "Made Shot",
  "timestamp": 1706457600
}
```

**Database Table**: `shot_telemetry`

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary Key |
| game_id | VARCHAR | Game Identifier |
| player_name | VARCHAR | Player Name |
| shot_distance | INT | Distance in feet |
| expected_points | FLOAT | Model Output (e.g., 1.12) |
| shot_grade | CHAR | Quality (A/B/C) |
| created_at | TIMESTAMP | Record timestamp |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.