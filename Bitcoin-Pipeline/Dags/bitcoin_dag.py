from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import psycopg2
import os
import time

default_args = {
    'start_date': datetime(2024, 1, 1),
}

def fetch_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    response = requests.get(url)
    price = response.json()["bitcoin"]["usd"]

    conn = psycopg2.connect(
        host="postgres",
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bitcoin_price (
            timestamp TIMESTAMP,
            price_usd FLOAT
        )
    """)
    cursor.execute(
        "INSERT INTO bitcoin_price (timestamp, price_usd) VALUES (%s, %s)",
        (datetime.now(), price)
    )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted price: {price} USD")

with DAG("bitcoin_price_dag", schedule_interval="@hourly", default_args=default_args, catchup=False) as dag:
    fetch_price = PythonOperator(
        task_id="fetch_bitcoin_price",
        python_callable=fetch_bitcoin_price
    )
