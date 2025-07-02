CREATE TABLE IF NOT EXISTS btc_price (
    id SERIAL PRIMARY KEY,
    timestamp BIGINT,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC
);