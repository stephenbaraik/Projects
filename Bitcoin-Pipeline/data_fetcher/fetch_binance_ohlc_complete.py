#!/usr/bin/env python3
"""
Complete Binance OHLC Data Fetcher
Fetches cryptocurrency OHLC data from Binance API and stores in PostgreSQL
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values
from retry import retry
from pydantic import BaseModel, ValidationError
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/binance_fetcher.log')
    ]
)
logger = logging.getLogger(__name__)

class OHLCRecord(BaseModel):
    """Pydantic model for OHLC data validation"""
    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_asset_volume: Optional[float] = None
    number_of_trades: Optional[int] = None
    taker_buy_base_asset_volume: Optional[float] = None
    taker_buy_quote_asset_volume: Optional[float] = None

class BinanceDataFetcher:
    """
    Fetches OHLC data from Binance API and stores in PostgreSQL
    """
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'bitcoin_data'),
            'user': os.getenv('POSTGRES_USER', 'airflow'),
            'password': os.getenv('POSTGRES_PASSWORD', 'airflow')
        }
        self.connection = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    def connect_db(self) -> bool:
        """Establish database connection with retry logic"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.connection = psycopg2.connect(**self.db_config)
                self.connection.autocommit = False
                logger.info("Database connection established")
                return True
            except psycopg2.Error as e:
                logger.error(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("Failed to connect to database after all retries")
                    return False
        return False
    
    def close_db(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    @retry(tries=3, delay=1, backoff=2)
    def fetch_klines(self, symbol: str, interval: str, limit: int = 1000, 
                     start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[List]:
        """
        Fetch kline/candlestick data from Binance API
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval ('1m', '5m', '1h', '1d', etc.)
            limit: Number of records to fetch (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            List of kline data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        try:
            response = requests.get(f"{self.base_url}/klines", params=params, timeout=30)
            response.raise_for_status()
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            data = response.json()
            logger.info(f"Fetched {len(data)} records for {symbol} {interval}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise
    
    def transform_klines(self, klines_data: List[List], symbol: str, timeframe: str) -> List[OHLCRecord]:
        """
        Transform raw kline data to OHLCRecord objects
        
        Args:
            klines_data: Raw kline data from API
            symbol: Trading pair symbol
            timeframe: Time interval
            
        Returns:
            List of validated OHLC records
        """
        records = []
        
        for kline in klines_data:
            try:
                record = OHLCRecord(
                    symbol=symbol,
                    timeframe=timeframe,
                    open_time=datetime.fromtimestamp(kline[0] / 1000),
                    close_time=datetime.fromtimestamp(kline[6] / 1000),
                    open_price=float(kline[1]),
                    high_price=float(kline[2]),
                    low_price=float(kline[3]),
                    close_price=float(kline[4]),
                    volume=float(kline[5]),
                    quote_asset_volume=float(kline[7]) if kline[7] else None,
                    number_of_trades=int(kline[8]) if kline[8] else None,
                    taker_buy_base_asset_volume=float(kline[9]) if kline[9] else None,
                    taker_buy_quote_asset_volume=float(kline[10]) if kline[10] else None
                )
                records.append(record)
            except (ValidationError, ValueError, IndexError) as e:
                logger.error(f"Failed to transform kline data: {e}")
                continue
                
        logger.info(f"Transformed {len(records)} valid records")
        return records
    
    def get_last_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        Get the last timestamp for a symbol-timeframe combination
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time interval
            
        Returns:
            Last timestamp or None if no data exists
        """
        if not self.connection:
            return None
            
        try:
            with self.connection.cursor() as cursor:
                query = """
                    SELECT MAX(close_time) 
                    FROM bitcoin.ohlc_data 
                    WHERE symbol = %s AND timeframe = %s
                """
                cursor.execute(query, (symbol, timeframe))
                result = cursor.fetchone()
                return result[0] if result and result[0] else None
        except psycopg2.Error as e:
            logger.error(f"Failed to get last timestamp: {e}")
            return None
    
    def insert_ohlc_data(self, records: List[OHLCRecord]) -> Tuple[int, int]:
        """
        Insert OHLC records into database with upsert logic
        
        Args:
            records: List of OHLC records to insert
            
        Returns:
            Tuple of (inserted_count, updated_count)
        """
        if not records or not self.connection:
            return 0, 0
            
        inserted_count = 0
        updated_count = 0
        
        try:
            with self.connection.cursor() as cursor:
                # Prepare data for bulk insert
                data_tuples = [
                    (
                        record.symbol,
                        record.timeframe,
                        record.open_time,
                        record.close_time,
                        record.open_price,
                        record.high_price,
                        record.low_price,
                        record.close_price,
                        record.volume,
                        record.quote_asset_volume,
                        record.number_of_trades,
                        record.taker_buy_base_asset_volume,
                        record.taker_buy_quote_asset_volume
                    )
                    for record in records
                ]
                
                # Upsert query
                insert_query = """
                    INSERT INTO bitcoin.ohlc_data (
                        symbol, timeframe, open_time, close_time, 
                        open_price, high_price, low_price, close_price, volume,
                        quote_asset_volume, number_of_trades, 
                        taker_buy_base_asset_volume, taker_buy_quote_asset_volume
                    ) VALUES %s
                    ON CONFLICT (symbol, timeframe, open_time) 
                    DO UPDATE SET
                        close_time = EXCLUDED.close_time,
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        quote_asset_volume = EXCLUDED.quote_asset_volume,
                        number_of_trades = EXCLUDED.number_of_trades,
                        taker_buy_base_asset_volume = EXCLUDED.taker_buy_base_asset_volume,
                        taker_buy_quote_asset_volume = EXCLUDED.taker_buy_quote_asset_volume,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING (xmax = 0) AS inserted
                """
                
                # Execute bulk insert
                cursor.execute("SELECT version()")
                
                execute_values(
                    cursor, insert_query, data_tuples, 
                    template=None, page_size=100
                )
                
                # Count results
                results = cursor.fetchall()
                inserted_count = sum(1 for result in results if result[0])
                updated_count = len(results) - inserted_count
                
                self.connection.commit()
                logger.info(f"Database operation completed: {inserted_count} inserted, {updated_count} updated")
                
        except psycopg2.Error as e:
            logger.error(f"Database insert failed: {e}")
            self.connection.rollback()
            raise
            
        return inserted_count, updated_count
    
    def log_pipeline_execution(self, dag_id: str, task_id: str, symbol: str, 
                             timeframe: str, records_processed: int, 
                             status: str, error_message: Optional[str] = None,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None):
        """Log pipeline execution to database"""
        if not self.connection:
            return
            
        try:
            with self.connection.cursor() as cursor:
                duration = None
                if start_time and end_time:
                    duration = int((end_time - start_time).total_seconds())
                
                insert_query = """
                    INSERT INTO bitcoin.pipeline_logs (
                        dag_id, task_id, execution_date, symbol, timeframe,
                        records_processed, status, error_message, 
                        start_time, end_time, duration_seconds
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """
                
                cursor.execute(insert_query, (
                    dag_id, task_id, datetime.now(), symbol, timeframe,
                    records_processed, status, error_message,
                    start_time, end_time, duration
                ))
                
                self.connection.commit()
                
        except psycopg2.Error as e:
            logger.error(f"Failed to log pipeline execution: {e}")
    
    def fetch_and_store_data(self, symbol: str, timeframe: str, 
                           limit: int = 1000, backfill_hours: int = 24) -> Dict:
        """
        Main method to fetch and store OHLC data
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time interval
            limit: Number of records to fetch
            backfill_hours: Hours to backfill if no existing data
            
        Returns:
            Dictionary with execution results
        """
        start_time = datetime.now()
        dag_id = "btc_data_pipeline"
        task_id = f"fetch_{symbol}_{timeframe}"
        
        try:
            logger.info(f"Starting data fetch for {symbol} {timeframe}")
            
            # Get last timestamp from database
            last_timestamp = self.get_last_timestamp(symbol, timeframe)
            
            # Determine start time for API call
            api_start_time = None
            if last_timestamp:
                # Start from last timestamp + 1 interval
                api_start_time = int((last_timestamp + timedelta(hours=1)).timestamp() * 1000)
                logger.info(f"Continuing from last timestamp: {last_timestamp}")
            else:
                # Backfill from specified hours ago
                backfill_start = datetime.now() - timedelta(hours=backfill_hours)
                api_start_time = int(backfill_start.timestamp() * 1000)
                logger.info(f"Starting backfill from: {backfill_start}")
            
            # Fetch data from API
            klines_data = self.fetch_klines(
                symbol=symbol, 
                interval=timeframe, 
                limit=limit,
                start_time=api_start_time
            )
            
            if not klines_data:
                logger.warning(f"No data returned from API for {symbol} {timeframe}")
                self.log_pipeline_execution(
                    dag_id, task_id, symbol, timeframe, 0, "SUCCESS",
                    start_time=start_time, end_time=datetime.now()
                )
                return {"status": "success", "records_processed": 0}
            
            # Transform data
            records = self.transform_klines(klines_data, symbol, timeframe)
            
            if not records:
                logger.warning(f"No valid records after transformation for {symbol} {timeframe}")
                self.log_pipeline_execution(
                    dag_id, task_id, symbol, timeframe, 0, "SUCCESS",
                    start_time=start_time, end_time=datetime.now()
                )
                return {"status": "success", "records_processed": 0}
            
            # Store data in database
            inserted, updated = self.insert_ohlc_data(records)
            total_processed = inserted + updated
            
            # Log successful execution
            end_time = datetime.now()
            self.log_pipeline_execution(
                dag_id, task_id, symbol, timeframe, total_processed, "SUCCESS",
                start_time=start_time, end_time=end_time
            )
            
            logger.info(f"Data fetch completed successfully: {total_processed} records processed")
            return {
                "status": "success",
                "records_processed": total_processed,
                "inserted": inserted,
                "updated": updated,
                "duration_seconds": int((end_time - start_time).total_seconds())
            }
            
        except Exception as e:
            end_time = datetime.now()
            error_message = str(e)
            logger.error(f"Data fetch failed: {error_message}")
            
            # Log failed execution
            self.log_pipeline_execution(
                dag_id, task_id, symbol, timeframe, 0, "FAILED",
                error_message=error_message, start_time=start_time, end_time=end_time
            )
            
            return {
                "status": "failed",
                "error": error_message,
                "duration_seconds": int((end_time - start_time).total_seconds())
            }


def main():
    """Main execution function"""
    # Configuration
    SYMBOLS = os.getenv('BTC_SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')
    TIMEFRAMES = os.getenv('BTC_TIMEFRAMES', '1h').split(',')
    LIMIT = int(os.getenv('BTC_LIMIT', '1000'))
    BACKFILL_HOURS = int(os.getenv('BTC_BACKFILL_HOURS', '24'))
    
    logger.info(f"Starting Bitcoin data fetcher with symbols: {SYMBOLS}, timeframes: {TIMEFRAMES}")
    
    fetcher = BinanceDataFetcher()
    
    try:
        # Connect to database
        if not fetcher.connect_db():
            logger.error("Failed to connect to database")
            sys.exit(1)
        
        # Fetch data for each symbol and timeframe combination
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                logger.info(f"Processing {symbol} {timeframe}")
                result = fetcher.fetch_and_store_data(
                    symbol=symbol.strip(),
                    timeframe=timeframe.strip(),
                    limit=LIMIT,
                    backfill_hours=BACKFILL_HOURS
                )
                
                if result["status"] == "failed":
                    logger.error(f"Failed to process {symbol} {timeframe}: {result.get('error')}")
                else:
                    logger.info(f"Successfully processed {symbol} {timeframe}: {result['records_processed']} records")
    
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        sys.exit(1)
    
    finally:
        fetcher.close_db()
    
    logger.info("Bitcoin data fetcher completed")


if __name__ == "__main__":
    main()
