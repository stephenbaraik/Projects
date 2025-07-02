FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY fetch_binance_ohlc.py .
CMD ["python", "fetch_binance_ohlc.py"]