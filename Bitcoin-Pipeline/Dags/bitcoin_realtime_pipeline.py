from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime
import requests
import logging

BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"

default_args = {
    'start_date': datetime(2024, 1, 1),
}


def create_table():
    hook = PostgresHook(postgres_conn_id="postgres_conn")
    with hook.get_conn() as conn, conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bitcoin_ohlc (
                timestamp TIMESTAMP PRIMARY KEY,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT
            )
        """)
        conn.commit()


def fetch_latest_ohlc():
    create_table()

    url = f"https://api.binance.com/api/v3/klines?symbol={BINANCE_SYMBOL}&interval={BINANCE_INTERVAL}&limit=1"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list) or not data:
            raise ValueError(f"Unexpected response: {data}")

        latest = data[0]
        timestamp = datetime.utcfromtimestamp(latest[0] / 1000)

        hook = PostgresHook(postgres_conn_id="postgres_conn")
        with hook.get_conn() as conn, conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO bitcoin_ohlc (timestamp, open, high, low, close)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (timestamp) DO NOTHING
            """, (
                timestamp, float(latest[1]), float(latest[2]),
                float(latest[3]), float(latest[4])
            ))
            conn.commit()

        logging.info(f"Inserted latest OHLC: {timestamp}")

    except Exception as e:
        logging.error(f"Failed to fetch latest OHLC: {e}")
        raise


with DAG(
    dag_id="bitcoin_realtime_pipeline",
    schedule_interval="* * * * *",  # Every minute
    default_args=default_args,
    catchup=False
) as dag:

    realtime_task = PythonOperator(
        task_id="fetch_latest_ohlc",
        python_callable=fetch_latest_ohlc
    )
