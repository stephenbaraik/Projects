import os
import requests
import psycopg2
from datetime import datetime

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "btc_db")
DB_USER = os.getenv("DB_USER", "btc_user")
DB_PASS = os.getenv("DB_PASS", "btc_pass")

def fetch_ohlc():
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": 100}
    response = requests.get(BINANCE_URL, params=params)
    response.raise_for_status()
    return response.json()

def insert_data(data):
    conn = psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()
    for row in data:
        ts = datetime.fromtimestamp(row[0] / 1000.0)
        cur.execute("""
            INSERT INTO btc_ohlc (timestamp, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (timestamp) DO NOTHING
        """, (ts, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])))
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    ohlc = fetch_ohlc()
    insert_data(ohlc)
    print("Data inserted successfully.")
