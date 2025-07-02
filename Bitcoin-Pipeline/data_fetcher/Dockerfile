FROM python:3.11

WORKDIR /app

COPY requirements.txt .
COPY data_fetcher/ .

RUN pip install -r requirements.txt

CMD ["python", "fetch_binance_ohlc.py"]
