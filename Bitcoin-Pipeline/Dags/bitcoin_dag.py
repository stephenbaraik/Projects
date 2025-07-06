from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime
import requests

default_args = {
    'start_date': datetime(2024, 1, 1),
}

def fetch_historical_ohlc():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=max"
    response = requests.get(url)
    data = response.json()

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
        open_price = entry[1]
        high_price = entry[2]
        low_price = entry[3]
        close_price = entry[4]

        cursor.execute("""
            INSERT INTO bitcoin_ohlc (timestamp, open, high, low, close)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (timestamp) DO NOTHING
        """, (timestamp, open_price, high_price, low_price, close_price))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted {len(data)} historical OHLC records.")


def fetch_latest_ohlc():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1"
    response = requests.get(url)
    data = response.json()

    latest_entry = data[-1]  # Latest OHLC data
    timestamp = datetime.utcfromtimestamp(float(latest_entry[0]) / 1000)
    open_price = latest_entry[1]
    high_price = latest_entry[2]
    low_price = latest_entry[3]
    close_price = latest_entry[4]

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
    print(f"Latest OHLC inserted: {timestamp} | O:{open_price}, H:{high_price}, L:{low_price}, C:{close_price}")


with DAG(
    dag_id="bitcoin_ohlc_full_pipeline",
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
