CREATE TABLE IF NOT EXISTS btc_ohlc (
    timestamp TIMESTAMP PRIMARY KEY,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT
);
