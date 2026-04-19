# Bitcoin Data Pipeline

Containerized Bitcoin OHLC data pipeline that ingests historical and realtime candles from Binance, stores them in PostgreSQL, and orchestrates jobs with Apache Airflow.

---

## Features

- Historical backfill DAG for bootstrap loading
- Realtime DAG scheduled every minute
- PostgreSQL persistence with idempotent inserts
- Docker Compose stack for local reproducible setup
- Airflow UI for scheduling and monitoring

---

## Architecture

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ Binance API  │ ───▶  │  Airflow DAG │ ───▶  │ PostgreSQL DB│
└──────────────┘       └──────────────┘       └──────────────┘
                           │
                           ▼
                  ┌────────────────────┐
                  │    Airflow UI      │
                  └────────────────────┘
```

---

## Tech Stack

- Docker & Docker Compose  
- Apache Airflow  
- PostgreSQL  
- pgAdmin  
- Python  
- Binance API  

---

## Project Structure

```
Bitcoin-Pipeline/
├── dags/
│   ├── bitcoin_historical_loader.py   # Backfill historical candles
│   └── bitcoin_realtime_pipeline.py   # Realtime minute ingestion
├── docker-compose.yml                 # Airflow + Postgres + pgAdmin stack
├── .env.example                       # Example environment config
├── data_fetcher/
│   └── fetch_binance_ohlc_complete.py
└── README.md
```

---

## Quick Setup

### 1️⃣ Prerequisites

- Docker & Docker Compose installed  
- At least 4GB free RAM  

### 2) Clone the repository

```bash
git clone <your-repo-url>
cd Projects/Bitcoin-Pipeline
```

### 3) Configure environment variables

Create `.env` from the example file:

```bash
cp .env.example .env
```

Then edit `.env` values if needed:

```env
# PostgreSQL Configuration
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=bitcoin

# pgAdmin Configuration
PGADMIN_DEFAULT_EMAIL=admin@admin.com
PGADMIN_DEFAULT_PASSWORD=admin

# Airflow Configuration
AIRFLOW_UID=50000
```

### 4) Build and start containers

```bash
docker-compose up -d --build
```

---

### 5) Access services

| Service           | URL                     | Credentials             |
|-------------------|-------------------------|--------------------------|
| Airflow UI        | [http://localhost:8080](http://localhost:8080) | `admin` / `admin`       |
| pgAdmin           | [http://localhost:5050](http://localhost:5050) | `admin@admin.com` / `admin` |

---

## Airflow Configuration

1. Open Airflow UI → Admin → Connections  
2. Create a Postgres connection:

| Field         | Value                      |
|---------------|----------------------------|
| Conn Id       | `postgres_conn`            |
| Conn Type     | `Postgres`                 |
| Host          | `postgres`                 |
| Schema        | `bitcoin`                  |
| Login         | `airflow`                  |
| Password      | `airflow`                  |
| Port          | `5432`                     |

---

## Pipeline Workflow

Historical DAG: `bitcoin_historical_loader`
- Trigger manually to backfill candles from configured start date.

Realtime DAG: `bitcoin_realtime_pipeline`
- Runs every minute and appends latest candle.

---

## Database Schema

Table: `bitcoin_ohlc`

| Column     | Type      | Description |
|------------|-----------|-------------|
| timestamp  | TIMESTAMP | Candle timestamp (PK) |
| open       | FLOAT     | Open price |
| high       | FLOAT     | High price |
| low        | FLOAT     | Low price |
| close      | FLOAT     | Close price |

---

## Common Commands

```bash
# View running containers
docker-compose ps

# Follow Airflow scheduler logs
docker-compose logs -f airflow-scheduler

# Query PostgreSQL for data
docker-compose exec postgres psql -U airflow -d airflow -c "SELECT * FROM bitcoin_ohlc ORDER BY timestamp DESC LIMIT 5;"
```

---

## Future Enhancements

- Add data quality checks and alerts in DAG tasks
- Add partitioned storage strategy for long retention
- Add cloud deployment profile (managed Airflow + Postgres)
- Add downstream feature engineering and model training jobs

---

## Contributing

Contributions welcome! Fork this repo and submit your improvements.

---

Improvements and PRs are welcome.

