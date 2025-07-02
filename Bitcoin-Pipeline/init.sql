-- Initialize Bitcoin data pipeline database

-- Create database if not exists (handled by docker postgres env vars)

-- Create schema for bitcoin data
CREATE SCHEMA IF NOT EXISTS bitcoin;

-- Create table for OHLC data
CREATE TABLE IF NOT EXISTS bitcoin.ohlc_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    open_price DECIMAL(20, 8) NOT NULL,
    high_price DECIMAL(20, 8) NOT NULL,
    low_price DECIMAL(20, 8) NOT NULL,
    close_price DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    quote_asset_volume DECIMAL(20, 8),
    number_of_trades INTEGER,
    taker_buy_base_asset_volume DECIMAL(20, 8),
    taker_buy_quote_asset_volume DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, open_time)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_timeframe ON bitcoin.ohlc_data(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_ohlc_open_time ON bitcoin.ohlc_data(open_time);
CREATE INDEX IF NOT EXISTS idx_ohlc_close_time ON bitcoin.ohlc_data(close_time);
CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_timeframe_time ON bitcoin.ohlc_data(symbol, timeframe, open_time);

-- Create table for data pipeline logs
CREATE TABLE IF NOT EXISTS bitcoin.pipeline_logs (
    id SERIAL PRIMARY KEY,
    dag_id VARCHAR(100) NOT NULL,
    task_id VARCHAR(100) NOT NULL,
    execution_date TIMESTAMP NOT NULL,
    symbol VARCHAR(20),
    timeframe VARCHAR(10),
    records_processed INTEGER DEFAULT 0,
    status VARCHAR(20) NOT NULL, -- SUCCESS, FAILED, RUNNING
    error_message TEXT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for pipeline logs
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_dag_task ON bitcoin.pipeline_logs(dag_id, task_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_execution_date ON bitcoin.pipeline_logs(execution_date);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_status ON bitcoin.pipeline_logs(status);

-- Create table for API rate limiting tracking
CREATE TABLE IF NOT EXISTS bitcoin.api_rate_limits (
    id SERIAL PRIMARY KEY,
    api_source VARCHAR(50) NOT NULL,
    endpoint VARCHAR(200) NOT NULL,
    requests_made INTEGER DEFAULT 0,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    limit_per_window INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(api_source, endpoint, window_start)
);

-- Create index for rate limiting
CREATE INDEX IF NOT EXISTS idx_rate_limits_api_endpoint ON bitcoin.api_rate_limits(api_source, endpoint);
CREATE INDEX IF NOT EXISTS idx_rate_limits_window ON bitcoin.api_rate_limits(window_start, window_end);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_ohlc_data_updated_at 
    BEFORE UPDATE ON bitcoin.ohlc_data 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_rate_limits_updated_at 
    BEFORE UPDATE ON bitcoin.api_rate_limits 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create view for latest OHLC data
CREATE OR REPLACE VIEW bitcoin.latest_ohlc AS
SELECT DISTINCT ON (symbol, timeframe) 
    symbol,
    timeframe,
    open_time,
    close_time,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    created_at
FROM bitcoin.ohlc_data
ORDER BY symbol, timeframe, open_time DESC;

-- Create view for daily summary
CREATE OR REPLACE VIEW bitcoin.daily_summary AS
SELECT 
    symbol,
    DATE(open_time) as date,
    MIN(open_time) as first_timestamp,
    MAX(close_time) as last_timestamp,
    COUNT(*) as record_count,
    MIN(low_price) as daily_low,
    MAX(high_price) as daily_high,
    SUM(volume) as total_volume
FROM bitcoin.ohlc_data
WHERE timeframe = '1h'
GROUP BY symbol, DATE(open_time)
ORDER BY symbol, date DESC;

-- Insert initial configuration data
INSERT INTO bitcoin.pipeline_logs (dag_id, task_id, execution_date, status, records_processed)
VALUES ('btc_data_pipeline', 'initialization', CURRENT_TIMESTAMP, 'SUCCESS', 0)
ON CONFLICT DO NOTHING;

-- Grant permissions to airflow user (already the owner, but explicit is better)
GRANT ALL PRIVILEGES ON SCHEMA bitcoin TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA bitcoin TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA bitcoin TO airflow;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA bitcoin TO airflow;

-- Create backup user (optional - for production)
-- CREATE USER bitcoin_reader WITH PASSWORD 'readonly_password';
-- GRANT CONNECT ON DATABASE bitcoin_data TO bitcoin_reader;
-- GRANT USAGE ON SCHEMA bitcoin TO bitcoin_reader;
-- GRANT SELECT ON ALL TABLES IN SCHEMA bitcoin TO bitcoin_reader;