import os
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import psycopg2

DB_HOST = os.environ['DB_HOST']
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASS = os.environ['DB_PASS']

url = "https://api.binance.com/api/v3/klines"
symbol = "BTCUSDT"
interval = "1m"
limit = 1000

def date_to_ms(dt):
    return int(dt.timestamp() * 1000)

with psycopg2.connect(
    host=DB_HOST,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS
) as conn:
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bitcoin_ohlc (
                date TIMESTAMP PRIMARY KEY,
                open NUMERIC,
                high NUMERIC,
                low NUMERIC,
                close NUMERIC,
                volume NUMERIC
            )
        """)
        conn.commit()

        cursor.execute("SELECT MAX(date) FROM bitcoin_ohlc")
        result = cursor.fetchone()
        start_time = result[0] or datetime(2017, 8, 17)
        print(f"Starting from: {start_time}")

        end_time = datetime.utcnow()

        while start_time < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": date_to_ms(start_time),
                "limit": limit
            }

            response = requests.get(url, params=params)
            if response.status_code != 200:
                print("Error fetching data:", response.text)
                break

            data = response.json()
            if not data:
                print("No new data.")
                break

            rows = []
            for entry in data:
                dt = datetime.utcfromtimestamp(entry[0] / 1000)
                rows.append((
                    dt, float(entry[1]), float(entry[2]),
                    float(entry[3]), float(entry[4]), float(entry[5])
                ))

            for row in rows:
                try:
                    cursor.execute("""
                        INSERT INTO bitcoin_ohlc (date, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (date) DO NOTHING
                    """, row)
                except Exception as e:
                    print("Insert error:", e)

            conn.commit()
            last_time = rows[-1][0]
            start_time = last_time + timedelta(milliseconds=1)
            print(f"Fetched up to: {last_time}")

            time.sleep(0.5)

print("Data fetching complete.")
