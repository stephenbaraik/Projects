from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime
import requests

default_args = {
    'start_date': datetime(2024, 1, 1),
}

BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"  # 1-minute candles
BINANCE_LIMIT = 1000  # Max per API call


def fetch_historical_ohlc():
    url = f"https://api.binance.com/api/v3/klines?symbol={BINANCE_SYMBOL}&interval={BINANCE_INTERVAL}&limit={BINANCE_LIMIT}"
    response = requests.get(url)
    data = response.json()

    if not isinstance(data, list):
        raise ValueError(f"Unexpected response from Binance: {data}")

    hook = PostgresHook(postgres_conn_id="postgres_conn")
    conn = hook.get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bitcoin_ohlc (
            timestamp TIMESTAMP PRIMARY KEY,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT
        )
    """)

    for entry in data:
        timestamp = datetime.utcfromtimestamp(float(entry[0]) / 1000)
        open_price = float(entry[1])
        high_price = float(entry[2])
        low_price = float(entry[3])
        close_price = float(entry[4])

        cursor.execute("""
            INSERT INTO bitcoin_ohlc (timestamp, open, high, low, close)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (timestamp) DO NOTHING
        """, (timestamp, open_price, high_price, low_price, close_price))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted {len(data)} OHLC records from Binance.")


def fetch_latest_ohlc():
    url = f"https://api.binance.com/api/v3/klines?symbol={BINANCE_SYMBOL}&interval={BINANCE_INTERVAL}&limit=1"
    response = requests.get(url)
    data = response.json()

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Unexpected response from Binance: {data}")

    latest_entry = data[0]
    timestamp = datetime.utcfromtimestamp(float(latest_entry[0]) / 1000)
    open_price = float(latest_entry[1])
    high_price = float(latest_entry[2])
    low_price = float(latest_entry[3])
    close_price = float(latest_entry[4])

    hook = PostgresHook(postgres_conn_id="postgres_conn")
    conn = hook.get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO bitcoin_ohlc (timestamp, open, high, low, close)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (timestamp) DO NOTHING
    """, (timestamp, open_price, high_price, low_price, close_price))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Latest OHLC inserted from Binance: {timestamp} | O:{open_price}, H:{high_price}, L:{low_price}, C:{close_price}")


with DAG(
    dag_id="bitcoin_ohlc_binance_pipeline",
    schedule_interval="* * * * *",  # Every minute
    default_args=default_args,
    catchup=False
) as dag:
    
    historical_task = PythonOperator(
        task_id="fetch_historical_ohlc",
        python_callable=fetch_historical_ohlc
    )

    realtime_task = PythonOperator(
        task_id="fetch_latest_ohlc",
        python_callable=fetch_latest_ohlc
    )

    historical_task >> realtime_task
