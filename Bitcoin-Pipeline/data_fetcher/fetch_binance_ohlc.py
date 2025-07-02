import requests
import os
import psycopg2
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1"

logging.info("Fetching OHLC data from Binance...")
response = requests.get(url)
data = response.json()

ohlc = data[0]
timestamp = int(ohlc[0])
open_price = float(ohlc[1])
high = float(ohlc[2])
low = float(ohlc[3])
close = float(ohlc[4])
volume = float(ohlc[5])

logging.info("Connecting to database...")
conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
cur = conn.cursor()

logging.info("Inserting data into btc_price table...")
cur.execute("""
    INSERT INTO btc_price (timestamp, open, high, low, close, volume)
    VALUES (%s, %s, %s, %s, %s, %s)
""", (timestamp, open_price, high, low, close, volume))

conn.commit()
cur.close()
conn.close()
logging.info("Data inserted successfully. Script completed.")