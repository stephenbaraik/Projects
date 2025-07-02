import os
import requests
import psycopg2
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "btc_db")
DB_USER = os.getenv("DB_USER", "btc_user")
DB_PASS = os.getenv("DB_PASS", "btc_pass")

def fetch_ohlc():
    logging.info("Fetching OHLC data from Binance...")
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": 100}
    response = requests.get(BINANCE_URL, params=params)
    response.raise_for_status()
    logging.info("Data fetched successfully.")
    return response.json()

def insert_data(data):
    try:
        logging.info("Connecting to database...")
        conn = psycopg2.connect(
            host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS
        )
        cur = conn.cursor()
        logging.info("Inserting data...")
        for row in data:
            ts = datetime.fromtimestamp(row[0] / 1000.0)
            try:
                cur.execute("""
                    INSERT INTO btc_ohlc (timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp) DO NOTHING
                """, (ts, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])))
            except Exception as row_err:
                logging.error(f"Failed to insert row at {ts}: {row_err}", exc_info=True)

        conn.commit()
        cur.close()
        conn.close()
        logging.info("Data inserted successfully.")
    except Exception as db_err:
        logging.error(f"Database connection or insertion failed: {db_err}", exc_info=True)

if __name__ == "__main__":
    try:
        ohlc = fetch_ohlc()
        insert_data(ohlc)
        logging.info("Script completed successfully.")
    except Exception as e:
        logging.error(f"Script failed: {e}", exc_info=True)
