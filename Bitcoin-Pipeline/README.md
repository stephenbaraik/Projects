# Bitcoin Data Pipeline

A robust, containerized data pipeline for collecting, processing, and storing cryptocurrency data from Binance using Apache Airflow, PostgreSQL, and Docker.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Binance API   │───▶│  Data Fetcher   │───▶│   PostgreSQL    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                      ┌─────────────────┐    ┌─────────────────┐
                      │ Apache Airflow  │───▶│   Monitoring    │
                      └─────────────────┘    └─────────────────┘
```

## 🚀 Features

- **Real-time Data Collection**: Fetches OHLC data from Binance API
- **Multiple Timeframes**: Supports 1m, 5m, 1h, 4h, 1d intervals
- **Data Validation**: Quality checks for gaps, anomalies, and freshness
- **Automated Scheduling**: Runs every hour with Airflow orchestration
- **Error Handling**: Comprehensive retry logic and error logging
- **Scalable**: Easily add new symbols and timeframes
- **Monitoring**: Pipeline logs and execution tracking
- **Data Cleanup**: Automatic cleanup of old data

## 📋 Prerequisites

- Docker and Docker Compose
- At least 4GB RAM available for containers
- 10GB+ free disk space for data storage

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd Bitcoin-Pipeline
```

2. **Set up environment variables** (optional):
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

3. **Build and start the services**:
```bash
# Build all containers
docker-compose build

# Start all services
docker-compose up -d

# Check if all services are running
docker-compose ps
```

4. **Initialize the database** (first time only):
```bash
# The database will be automatically initialized on first run
# Check logs to ensure successful initialization
docker-compose logs postgres
```

5. **Access Airflow UI**:
- URL: http://localhost:8080
- Username: `admin`
- Password: `admin`

## ⚙️ Configuration

### Airflow Variables

Configure the pipeline through Airflow Variables:

1. Go to **Admin → Variables** in Airflow UI
2. Add the following variables:

| Variable Name | Default Value | Description |
|---------------|---------------|-------------|
| `crypto_symbols` | `BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT,LINKUSDT` | Comma-separated list of trading pairs |
| `crypto_timeframes` | `1h,4h,1d` | Comma-separated list of timeframes |
| `postgres_conn_id` | `postgres_default` | PostgreSQL connection ID |

### Environment Variables

Set these in your environment or `.env` file:

```bash
# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=bitcoin_data
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow

# Data Fetcher Configuration
SYMBOLS=BTCUSDT,ETHUSDT,ADAUSDT
TIMEFRAMES=1h,4h,1d
LIMIT=1000
BACKFILL_HOURS=24

# Airflow Configuration
AIRFLOW__CORE__EXECUTOR=CeleryExecutor
AIRFLOW__CORE__LOAD_EXAMPLES=false
```

## 📊 Database Schema

### Tables

1. **`bitcoin.ohlc_data`**: Stores OHLC candlestick data
2. **`bitcoin.pipeline_logs`**: Execution logs and monitoring
3. **`bitcoin.api_rate_limits`**: API rate limiting tracking

### Key Columns

**ohlc_data**:
- `symbol`: Trading pair (e.g., 'BTCUSDT')
- `timeframe`: Time interval (e.g., '1h', '1d')
- `open_time`, `close_time`: Timestamp range
- `open_price`, `high_price`, `low_price`, `close_price`: OHLC prices
- `volume`: Trading volume

## 🔧 Usage

### Starting the Pipeline

1. **Enable the DAG**:
   - Go to Airflow UI
   - Find `btc_data_pipeline` DAG
   - Toggle the switch to enable it

2. **Manual Trigger**:
   - Click on the DAG name
   - Click "Trigger DAG" button

3. **Monitor Execution**:
   - View task logs in Airflow UI
   - Check database for stored data

### Querying Data

Connect to PostgreSQL and query the data:

```sql
-- Get latest OHLC data
SELECT * FROM bitcoin.latest_ohlc ORDER BY symbol, timeframe;

-- Daily summary
SELECT * FROM bitcoin.daily_summary ORDER BY symbol, date DESC;

-- Check pipeline status
SELECT * FROM bitcoin.pipeline_logs ORDER BY created_at DESC LIMIT 10;
```

### Adding New Symbols

1. Update the `crypto_symbols` Airflow Variable
2. The pipeline will automatically start collecting data for new symbols

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `postgres` | 5432 | PostgreSQL database |
| `redis` | 6379 | Redis for Celery backend |
| `airflow-webserver` | 8080 | Airflow web interface |
| `airflow-scheduler` | - | Airflow scheduler |
| `airflow-worker` | - | Celery worker |

## 📈 Monitoring

### Airflow UI
- **DAG Status**: Monitor pipeline execution
- **Task Logs**: Detailed execution logs
- **XCom**: Inter-task communication data

### Database Monitoring
```sql
-- Pipeline execution summary
SELECT 
    dag_id,
    task_id,
    status,
    COUNT(*) as count,
    AVG(duration_seconds) as avg_duration
FROM bitcoin.pipeline_logs 
GROUP BY dag_id, task_id, status
ORDER BY dag_id, task_id;

-- Data freshness check
SELECT 
    symbol,
    timeframe,
    MAX(close_time) as latest_data,
    COUNT(*) as record_count
FROM bitcoin.ohlc_data 
GROUP BY symbol, timeframe
ORDER BY symbol, timeframe;
```

### Health Checks
```bash
# Check service health
docker-compose ps

# View logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f postgres

# Check database connection
docker-compose exec postgres psql -U airflow -d bitcoin_data -c "SELECT COUNT(*) FROM bitcoin.ohlc_data;"
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Failed**:
   ```bash
   # Check postgres container
   docker-compose logs postgres
   
   # Restart postgres
   docker-compose restart postgres
   ```

2. **Airflow Services Not Starting**:
   ```bash
   # Check all services
   docker-compose ps
   
   # Restart airflow services
   docker-compose restart airflow-webserver airflow-scheduler
   ```

3. **API Rate Limiting**:
   - The pipeline includes automatic rate limiting
   - Check `bitcoin.api_rate_limits` table for usage
   - Reduce fetch frequency if needed

4. **Disk Space Issues**:
   ```bash
   # Check disk usage
   docker system df
   
   # Clean up old data
   docker-compose exec postgres psql -U airflow -d bitcoin_data -c "DELETE FROM bitcoin.ohlc_data WHERE close_time < NOW() - INTERVAL '90