from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime, timedelta
import requests
import logging
import time

BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"
BINANCE_LIMIT = 1000  # Max candles per call

default_args = {
    'start_date': datetime(2018, 1, 1),
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


def fetch_historical_ohlc():
    """
    Loads historical data in batches of 1000 candles starting from a specified date.
    Modify start_time to go as far back as needed.
    """
    create_table()

    start_time = datetime(2024, 1, 1)  # <-- Change to desired historical start
    end_time = datetime.utcnow()

    hook = PostgresHook(postgres_conn_id="postgres_conn")

    while start_time < end_time:
        start_ms = int(start_time.timestamp() * 1000)
        url = f"https://api.binance.com/api/v3/klines?symbol={BINANCE_SYMBOL}&interval={BINANCE_INTERVAL}&limit={BINANCE_LIMIT}&startTime={start_ms}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                logging.info(f"No more data after {start_time}")
                break

            with hook.get_conn() as conn, conn.cursor() as cursor:
                for entry in data:
                    timestamp = datetime.utcfromtimestamp(entry[0] / 1000)
                    cursor.execute("""
                        INSERT INTO bitcoin_ohlc (timestamp, open, high, low, close)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (timestamp) DO NOTHING
                    """, (
                        timestamp, float(entry[1]), float(entry[2]),
                        float(entry[3]), float(entry[4])
                    ))
                conn.commit()

            logging.info(f"Inserted batch starting {start_time}")

            # Move to next batch (next timestamp)
            last_ts = int(data[-1][0]) / 1000
            start_time = datetime.utcfromtimestamp(last_ts) + timedelta(minutes=1)

            time.sleep(0.5)  # Avoid API rate limits

        except Exception as e:
            logging.error(f"Error fetching batch starting {start_time}: {e}")
            break


with DAG(
    dag_id="bitcoin_historical_loader",
    schedule_interval=None,  # Manual run only
    default_args=default_args,
    catchup=False
) as dag:

    historical_task = PythonOperator(
        task_id="fetch_full_historical_ohlc",
        python_callable=fetch_historical_ohlc
    )
