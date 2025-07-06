# 🚀 Bitcoin Data Pipeline

A robust, containerized **Bitcoin OHLC Data Pipeline** fetching real-time and historical cryptocurrency price data from Binance, storing it in PostgreSQL, orchestrated with Apache Airflow—all inside Docker.

---

## 🌟 Features

✅ Real-time & historical OHLC price data from Binance  
✅ Data stored in structured PostgreSQL tables  
✅ Workflow orchestration with Apache Airflow  
✅ Full monitoring via Airflow UI  
✅ Easy setup with Docker & Docker Compose  
✅ Next step: Azure Cloud deployment, Machine Learning pipelines, and Power BI dashboards  

---

## 🏗️ Architecture

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

## 📦 Tech Stack

- Docker & Docker Compose  
- Apache Airflow  
- PostgreSQL  
- pgAdmin  
- Python  
- Binance API  

---

## 🗃️ Project Structure

```
.
├── dags/
│   └── bitcoin_dag.py           # Airflow DAGs for OHLC pipeline
├── docker-compose.yml           # Multi-container Docker setup
├── .env                         # Environment variables
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ⚡ Quick Setup Guide

### 1️⃣ Prerequisites

- Docker & Docker Compose installed  
- At least 4GB free RAM  

### 2️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd bitcoin-data-pipeline
```

### 3️⃣ Configure Environment Variables

Example `.env` file:

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

**Create the `.env` file:**

```bash
cp .env.example .env
# Edit values as needed
```

---

### 4️⃣ Build and Start Containers

```bash
docker-compose up -d --build
```

---

### 5️⃣ Access Services

| Service           | URL                     | Credentials             |
|-------------------|-------------------------|--------------------------|
| Airflow UI        | [http://localhost:8080](http://localhost:8080) | `admin` / `admin`       |
| pgAdmin           | [http://localhost:5050](http://localhost:5050) | `admin@admin.com` / `admin` |

---

## 🎛️ Airflow Configuration

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

## 🗓️ Pipeline Workflow

DAG: `bitcoin_ohlc_full_pipeline`

✅ Fetch historical OHLC data from Binance  
✅ Continuously fetch live OHLC data every minute  
✅ Store data in PostgreSQL  
✅ Full orchestration and monitoring via Airflow  

---

## 🗄️ Database Schema

**Table:** `ohlc_data`

| Column     | Type      | Description          |
|------------|-----------|----------------------|
| symbol     | TEXT      | Trading pair (e.g., BTCUSDT) |
| interval   | TEXT      | Timeframe (e.g., 1m, 1h) |
| timestamp  | TIMESTAMP | OHLC data timestamp  |
| open       | DECIMAL   | Opening price        |
| high       | DECIMAL   | Highest price        |
| low        | DECIMAL   | Lowest price         |
| close      | DECIMAL   | Closing price        |
| volume     | DECIMAL   | Trade volume         |

---

## 🔧 Common Commands

```bash
# View running containers
docker-compose ps

# Follow Airflow scheduler logs
docker-compose logs -f airflow-scheduler

# Query PostgreSQL for data
docker-compose exec postgres psql -U airflow -d bitcoin -c "SELECT * FROM ohlc_data LIMIT 5;"
```

---

## 📊 Future Enhancements

✅ Real-time Bitcoin OHLC data pipeline (Complete)  
🚀 Next: Azure Cloud Deployment (ACI + ACR)  
🧠 Add Machine Learning pipelines for price prediction  
📊 Visualize insights & predictions using Power BI  

---

## 🤝 Contributing

Contributions welcome! Fork this repo and submit your improvements.

---

## 📢 Stay Connected

Follow along as this project evolves from local containers to full cloud deployment with ML-powered analytics!

