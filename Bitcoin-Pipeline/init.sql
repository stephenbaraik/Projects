CREATE TABLE IF NOT EXISTS bitcoin_ohlc (
    date TIMESTAMP PRIMARY KEY,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC
);
